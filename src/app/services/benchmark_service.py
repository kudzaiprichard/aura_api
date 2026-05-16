"""Benchmark orchestration (§6.8–6.11, §7.7, §8.7, Phase 10).

Two responsibilities:

  * Curated datasets — admin-uploaded CSVs of (sender, subject, body, label)
    rows. Ingestion mirrors the training buffer import surface (size cap,
    MIME check, SHA-256 over the source file stored on the header row).

  * Benchmark runs — per-version evaluation of one dataset, executed in a
    background task. Each version is loaded fresh via
    `PhishingDetector.load(version)` so unregistered / non-active versions
    can participate; `app.state.detector` is never touched (§8.7).

Events fan out on the SSE broker under topic `benchmark:{id}` — one
`status` event on start and finish, one `version_done` event per completed
version with the full per-version metrics payload, and an `error` event on
failure. Winners are not stored — they're computed at response-build time
from the persisted `model_benchmark_version_results` rows so adding a late
version later cannot rewrite prior snapshots.
"""
from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import logging
import time
from datetime import datetime, timezone
from typing import Any, Sequence
from uuid import UUID

import numpy as np
from sklearn.metrics import roc_auc_score
from sqlalchemy import event, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.configs import (
    benchmark as benchmark_config,
    inference as inference_config,
    sse as sse_config,
)
from src.app.dtos.requests import (
    BenchmarkDatasetCreateRequest,
    BenchmarkRunRequest,
)
from src.app.dtos.responses import BenchmarkEvent, BenchmarkVersionResult
from src.app.models.benchmark_dataset import BenchmarkDataset
from src.app.models.benchmark_dataset_row import BenchmarkDatasetRow
from src.app.models.enums import BenchmarkStatus
from src.app.models.model_benchmark import ModelBenchmark
from src.app.models.model_benchmark_version_result import (
    ModelBenchmarkVersionResult,
)
from src.app.models.user import User
from src.app.repositories.benchmark_dataset_repository import (
    BenchmarkDatasetRepository,
)
from src.app.repositories.benchmark_dataset_row_repository import (
    BenchmarkDatasetRowRepository,
)
from src.app.repositories.model_benchmark_repository import (
    ModelBenchmarkRepository,
)
from src.app.repositories.model_benchmark_version_result_repository import (
    ModelBenchmarkVersionResultRepository,
)
from src.core.background_jobs import BackgroundJobRunner
from src.core.sse import SSEBroker, SSEEvent, monotonic_event_id
from src.shared.database import async_session
from src.shared.exceptions import (
    BadRequestException,
    ConflictException,
    NotFoundException,
)
from src.shared.inference import PhishingDetector
from src.shared.inference.registry import ModelRegistry
from src.shared.responses import ErrorDetail


log = logging.getLogger("aura.benchmark")


_CSV_REQUIRED_COLUMNS = ("sender", "subject", "body", "label")
_CSV_ALLOWED_MIMES = {
    "text/csv",
    "application/csv",
    "application/vnd.ms-excel",
    "text/plain",
    "application/octet-stream",
}
_CSV_ALLOWED_EXTENSIONS = (".csv",)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _topic(benchmark_id: UUID) -> str:
    return f"benchmark:{benchmark_id}"


def _event(kind: str, payload: dict[str, Any]) -> SSEEvent:
    return SSEEvent(data=payload, event=kind, id=monotonic_event_id())


# Commit-visibility retry budget used by the background workers. We've
# observed the `after_commit` hook fire before a fresh `async_session()`
# can SELECT the just-inserted row (~2/11 runs in logs). A handful of 50ms
# retries absorbs the gap without re-introducing the "schedule before
# commit" race the event hook was meant to prevent.
_ROW_DURABLE_RETRIES = 5
_ROW_DURABLE_BACKOFF_SECONDS = 0.05


class BenchmarkService:
    """Owns `POST /benchmarks/datasets`, the benchmark run lifecycle, and
    the SSE stream.

    The service receives a `session` for synchronous CRUD from the
    controller thread (list/get/delete/create_dataset) and
    `background_jobs` + `sse_broker` for the asynchronous run. Every DB
    write performed from the background worker runs against a fresh
    `async_session()` — the controller's session is bound to the request
    and has already committed by the time the worker starts."""

    def __init__(
        self,
        *,
        session: AsyncSession,
        dataset_repository: BenchmarkDatasetRepository,
        dataset_row_repository: BenchmarkDatasetRowRepository,
        benchmark_repository: ModelBenchmarkRepository,
        result_repository: ModelBenchmarkVersionResultRepository,
        sse_broker: SSEBroker,
        background_jobs: BackgroundJobRunner,
    ):
        self.session = session
        self.dataset_repo = dataset_repository
        self.dataset_row_repo = dataset_row_repository
        self.benchmark_repo = benchmark_repository
        self.result_repo = result_repository
        self.sse_broker = sse_broker
        self.background_jobs = background_jobs

    # ── datasets: reads ──

    async def list_datasets(
        self, *, page: int, page_size: int
    ) -> tuple[Sequence[BenchmarkDataset], int]:
        return await self.dataset_repo.paginate_filtered(
            page=page, page_size=page_size
        )

    async def get_dataset(self, dataset_id: UUID) -> BenchmarkDataset:
        dataset = await self.dataset_repo.get_by_id(dataset_id)
        if dataset is None:
            raise NotFoundException(
                message="Benchmark dataset not found",
                error_detail=ErrorDetail(
                    title="Not Found",
                    code="BENCHMARK_DATASET_NOT_FOUND",
                    status=404,
                    details=[f"No benchmark dataset found with id {dataset_id}"],
                ),
            )
        return dataset

    # ── datasets: writes ──

    async def create_dataset(
        self,
        *,
        payload: bytes,
        filename: str | None,
        content_type: str | None,
        request: BenchmarkDatasetCreateRequest,
        actor: User,
    ) -> BenchmarkDataset:
        """Validate + dedupe-guard + bulk-insert a CSV upload. Rejects the
        upload on name collision (409) before touching the bytes so the user
        sees the failure fast; oversize / unsupported-type rejections fire
        before parsing for the same reason."""
        existing = await self.dataset_repo.get_by_name(request.name)
        if existing is not None:
            raise ConflictException(
                message="A benchmark dataset with that name already exists",
                error_detail=ErrorDetail(
                    title="Conflict",
                    code="BENCHMARK_DATASET_NAME_TAKEN",
                    status=409,
                    details=[
                        f"dataset name {request.name!r} is already in use"
                    ],
                ),
            )

        size_cap = benchmark_config.csv_max_bytes
        row_cap = benchmark_config.csv_max_rows

        size = len(payload)
        if size > size_cap:
            raise BadRequestException(
                message="CSV upload exceeds the configured size cap",
                error_detail=ErrorDetail(
                    title="Payload Too Large",
                    code="BENCHMARK_CSV_TOO_LARGE",
                    status=413,
                    details=[
                        f"Received {size} bytes; "
                        f"benchmark.csv_max_bytes = {size_cap}"
                    ],
                ),
            )

        self._validate_csv_mime(filename=filename, content_type=content_type)

        file_sha = hashlib.sha256(payload).hexdigest()

        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise BadRequestException(
                message="CSV must be UTF-8 encoded",
                error_detail=ErrorDetail(
                    title="Bad Request",
                    code="BENCHMARK_CSV_ENCODING",
                    status=400,
                    details=[f"UTF-8 decode failed at byte {exc.start}"],
                ),
            ) from exc

        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames is None:
            raise BadRequestException(
                message="CSV is empty or missing a header row",
                error_detail=ErrorDetail(
                    title="Bad Request",
                    code="BENCHMARK_CSV_EMPTY",
                    status=400,
                    details=["No header row found in the uploaded CSV"],
                ),
            )

        normalised_headers = {
            (h or "").strip().lower(): (h or "") for h in reader.fieldnames
        }
        missing = [
            col for col in _CSV_REQUIRED_COLUMNS if col not in normalised_headers
        ]
        if missing:
            raise BadRequestException(
                message="CSV is missing required columns",
                error_detail=ErrorDetail(
                    title="Bad Request",
                    code="BENCHMARK_CSV_MISSING_COLUMNS",
                    status=400,
                    details=[
                        f"Missing column(s): {', '.join(missing)}. "
                        f"Required: {', '.join(_CSV_REQUIRED_COLUMNS)}"
                    ],
                ),
            )

        col_sender = normalised_headers["sender"]
        col_subject = normalised_headers["subject"]
        col_body = normalised_headers["body"]
        col_label = normalised_headers["label"]

        rows: list[BenchmarkDatasetRow] = []
        label_distribution = {0: 0, 1: 0}
        row_errors: list[str] = []

        total_rows = 0
        # DictReader's header is row 1, so data rows start at 2.
        for row_index, raw in enumerate(reader, start=2):
            total_rows += 1
            if total_rows > row_cap:
                raise BadRequestException(
                    message="CSV exceeds the configured row cap",
                    error_detail=ErrorDetail(
                        title="Payload Too Large",
                        code="BENCHMARK_CSV_TOO_MANY_ROWS",
                        status=413,
                        details=[
                            f"More than {row_cap} data rows; "
                            f"benchmark.csv_max_rows = {row_cap}"
                        ],
                    ),
                )

            sender = (raw.get(col_sender) or "").strip()
            subject = (raw.get(col_subject) or "").strip()
            body = (raw.get(col_body) or "").strip()
            label_raw = (raw.get(col_label) or "").strip()

            per_row: list[str] = []
            if not sender:
                per_row.append("sender is required")
            if not body:
                per_row.append("body is required")
            try:
                label_int = int(label_raw)
            except ValueError:
                per_row.append(
                    f"label must be 0 or 1 (got {label_raw!r})"
                )
                label_int = None
            else:
                if label_int not in (0, 1):
                    per_row.append(
                        f"label must be 0 or 1 (got {label_int})"
                    )
                    label_int = None

            if per_row:
                row_errors.append(
                    f"row {row_index}: " + "; ".join(per_row)
                )
                continue

            assert label_int is not None  # refined by the branches above
            label_distribution[label_int] += 1
            rows.append(
                BenchmarkDatasetRow(
                    benchmark_dataset_id=None,  # back-filled after header insert
                    sender=sender,
                    subject=subject,
                    body=body,
                    label=label_int,
                )
            )

        if row_errors:
            # Benchmark datasets are treated as labelled ground truth — every
            # row counts toward the score. Partial ingestion would silently
            # change the denominator, so refuse the upload wholesale.
            raise BadRequestException(
                message="CSV contains invalid rows; no rows were ingested",
                error_detail=ErrorDetail(
                    title="Bad Request",
                    code="BENCHMARK_CSV_ROW_ERRORS",
                    status=400,
                    details=row_errors[:50],
                ),
            )
        if not rows:
            raise BadRequestException(
                message="CSV contained no data rows",
                error_detail=ErrorDetail(
                    title="Bad Request",
                    code="BENCHMARK_CSV_NO_ROWS",
                    status=400,
                    details=["Header present but no data rows were found"],
                ),
            )

        dataset = BenchmarkDataset(
            name=request.name,
            description=request.description,
            row_count=len(rows),
            label_distribution={
                str(k): v for k, v in label_distribution.items()
            },
            uploaded_by=actor.id,
            source_csv_sha256=file_sha,
        )
        dataset = await self.dataset_repo.create(dataset)

        for row in rows:
            row.benchmark_dataset_id = dataset.id
        await self.dataset_row_repo.bulk_insert(rows)

        # Never log row bodies — per §4.3 / Phase 12 hardening, only the
        # file hash, actor, and counts make it into the audit line.
        log.info(
            "benchmark.dataset_created actor_id=%s dataset_id=%s "
            "name=%s file_sha256=%s filename=%s rows=%d",
            actor.id,
            dataset.id,
            dataset.name,
            file_sha,
            filename or "<unnamed>",
            len(rows),
        )
        return dataset

    async def delete_dataset(self, dataset_id: UUID) -> None:
        """Remove a dataset only if no benchmark has ever referenced it.

        Enforcing this in code (in addition to the RESTRICT FK on
        `model_benchmarks.benchmark_dataset_id`) gives the caller a clean
        409 with an actionable error instead of a raw IntegrityError."""
        dataset = await self.get_dataset(dataset_id)
        reference_count = await self.benchmark_repo.count_by_dataset(
            dataset_id
        )
        if reference_count > 0:
            raise ConflictException(
                message="Dataset is referenced by one or more benchmarks",
                error_detail=ErrorDetail(
                    title="Conflict",
                    code="BENCHMARK_DATASET_IN_USE",
                    status=409,
                    details=[
                        f"{reference_count} benchmark(s) reference this dataset; "
                        "delete them first"
                    ],
                ),
            )
        await self.dataset_repo.delete(dataset)
        log.info("benchmark.dataset_deleted dataset_id=%s", dataset_id)

    # ── benchmarks: reads ──

    async def list_benchmarks(
        self,
        *,
        page: int,
        page_size: int,
        status: BenchmarkStatus | None = None,
        dataset_id: UUID | None = None,
    ) -> tuple[Sequence[ModelBenchmark], int]:
        return await self.benchmark_repo.paginate_filtered(
            page=page,
            page_size=page_size,
            status=status,
            dataset_id=dataset_id,
        )

    async def get_benchmark(
        self, benchmark_id: UUID
    ) -> tuple[ModelBenchmark, Sequence[ModelBenchmarkVersionResult]]:
        benchmark = await self.benchmark_repo.get_by_id(benchmark_id)
        if benchmark is None:
            raise NotFoundException(
                message="Benchmark not found",
                error_detail=ErrorDetail(
                    title="Not Found",
                    code="BENCHMARK_NOT_FOUND",
                    status=404,
                    details=[f"No benchmark found with id {benchmark_id}"],
                ),
            )
        results = await self.result_repo.list_by_benchmark(benchmark_id)
        return benchmark, results

    # ── benchmarks: writes ──

    async def start_run(
        self, request: BenchmarkRunRequest, *, actor: User
    ) -> ModelBenchmark:
        """Validate, create a PENDING row, submit the background worker,
        and return the row. Every caller-visible refusal lands before any
        state is mutated."""
        if not request.versions:
            # pydantic's min_length=1 catches this too; keep the explicit
            # check so we raise a 400 rather than surfacing Pydantic's
            # validation envelope for what is really a service-level rule.
            raise BadRequestException(
                message="At least one version is required",
                error_detail=ErrorDetail(
                    title="Bad Request",
                    code="BENCHMARK_VERSIONS_EMPTY",
                    status=400,
                    details=["versions must contain at least one entry"],
                ),
            )
        max_versions = benchmark_config.max_versions_per_run
        if len(request.versions) > max_versions:
            raise BadRequestException(
                message="Too many versions in one run",
                error_detail=ErrorDetail(
                    title="Bad Request",
                    code="BENCHMARK_TOO_MANY_VERSIONS",
                    status=400,
                    details=[
                        f"Received {len(request.versions)} versions; "
                        f"benchmark.max_versions_per_run = {max_versions}"
                    ],
                ),
            )
        # Dedupe while preserving order — running the same version twice
        # would overwrite the earlier row at the unique (benchmark_id,
        # version) index.
        seen: set[str] = set()
        versions: list[str] = []
        for v in request.versions:
            if v in seen:
                continue
            seen.add(v)
            versions.append(v)

        registry = self._registry()
        known = set(registry.list_versions())
        unknown = [v for v in versions if v not in known]
        if unknown:
            raise BadRequestException(
                message="One or more versions are not registered",
                error_detail=ErrorDetail(
                    title="Bad Request",
                    code="BENCHMARK_UNKNOWN_VERSIONS",
                    status=400,
                    details=[
                        f"unknown versions: {unknown}; "
                        f"known versions: {sorted(known)}"
                    ],
                ),
            )

        dataset = await self.get_dataset(request.dataset_id)

        benchmark = ModelBenchmark(
            benchmark_dataset_id=dataset.id,
            triggered_by=actor.id,
            versions=versions,
            status=BenchmarkStatus.PENDING,
            started_at=None,
            finished_at=None,
            error_message=None,
        )
        benchmark = await self.benchmark_repo.create(benchmark)

        # Defer the worker until the request's transaction actually commits.
        # `get_db` owns the commit (CLAUDE.md) and fires it on clean return of
        # the endpoint, which happens AFTER this service call. Submitting the
        # task eagerly races that commit: the worker opens a fresh session,
        # reads the benchmark row via `session.get`, and with READ COMMITTED
        # isolation finds nothing because the insert is still uncommitted in
        # another transaction — the run then logs "vanished before execution;
        # aborting" and stays PENDING forever. Hooking `after_commit` means
        # the task is only scheduled once the row is durable, and a rollback
        # never orphans a worker.
        benchmark_id = benchmark.id

        def _submit_worker(_sync_session: Any) -> None:
            self.background_jobs.submit(
                self._execute_run(benchmark_id=benchmark_id),
                name=f"benchmark-{benchmark_id}",
            )

        event.listen(
            self.session.sync_session,
            "after_commit",
            _submit_worker,
            once=True,
        )
        log.info(
            "benchmark submitted id=%s dataset_id=%s versions=%s actor_id=%s",
            benchmark.id, dataset.id, versions, actor.id,
        )
        return benchmark

    # ── background worker ──

    async def _execute_run(self, *, benchmark_id: UUID) -> None:
        """Runs as an `asyncio.Task` under `BackgroundJobRunner`. Owns the
        transaction for every write it issues; lands the benchmark as
        FAILED on any exception so callers that subscribe late still see
        the final status."""
        try:
            await self._execute_locked(benchmark_id=benchmark_id)
        except Exception as exc:  # noqa: BLE001
            log.exception("benchmark failed id=%s", benchmark_id)
            await self._fail(benchmark_id, message=str(exc))

    async def _execute_locked(self, *, benchmark_id: UUID) -> None:
        topic = _topic(benchmark_id)

        # Load the benchmark header + dataset rows once, up-front. The
        # detector loads happen per-version inside the loop so a later
        # version can't read stale state captured here.
        versions: list[str] = []
        dataset_id: UUID | None = None
        rows: Sequence[BenchmarkDatasetRow] = []
        for attempt in range(_ROW_DURABLE_RETRIES):
            async with async_session() as session:
                async with session.begin():
                    benchmark = await session.get(ModelBenchmark, benchmark_id)
                    if benchmark is not None:
                        versions = list(benchmark.versions)
                        dataset_id = benchmark.benchmark_dataset_id
                        row_repo = BenchmarkDatasetRowRepository(session)
                        rows = await row_repo.list_by_dataset(dataset_id)
                        break
            # First-read miss = the outer request's commit hasn't landed
            # yet; wait a tick and retry before declaring the row missing.
            await asyncio.sleep(_ROW_DURABLE_BACKOFF_SECONDS)
        else:
            log.warning(
                "benchmark id=%s not found after %d retries (%.0fms); "
                "aborting",
                benchmark_id,
                _ROW_DURABLE_RETRIES,
                _ROW_DURABLE_RETRIES * _ROW_DURABLE_BACKOFF_SECONDS * 1000,
            )
            return

        if not rows:
            raise BadRequestException(
                message="Benchmark dataset is empty",
                error_detail=ErrorDetail(
                    title="Bad Request",
                    code="BENCHMARK_DATASET_EMPTY",
                    status=400,
                    details=[
                        f"dataset {dataset_id} has no rows; nothing to score"
                    ],
                ),
            )

        labels = np.asarray([int(r.label) for r in rows], dtype=np.int64)
        emails = [
            {
                "sender": r.sender,
                "subject": r.subject,
                "body": r.body,
            }
            for r in rows
        ]

        # Flip to RUNNING + stamp started_at so concurrent readers see a
        # consistent lifecycle.
        async with async_session() as session:
            async with session.begin():
                await session.execute(
                    update(ModelBenchmark)
                    .where(ModelBenchmark.id == benchmark_id)
                    .values(
                        status=BenchmarkStatus.RUNNING,
                        started_at=_now(),
                    )
                )

        self.sse_broker.publish(
            topic,
            _event(
                "status",
                BenchmarkEvent(
                    benchmarkId=benchmark_id,
                    kind="status",
                    status=BenchmarkStatus.RUNNING.value,
                    message=f"evaluating {len(versions)} version(s) on "
                            f"{len(rows)} row(s)",
                    emittedAt=_now(),
                ).model_dump(by_alias=True, mode="json"),
            ),
        )

        models_dir = inference_config.models_dir

        for version in versions:
            # Per §8.7, each version loads fresh — we never read
            # app.state.detector here, so a non-active version can be
            # benchmarked without touching the production singleton.
            detector_v = PhishingDetector.load(
                version=version, models_root=models_dir
            )

            t_start = time.monotonic_ns()
            preds = detector_v.predict_batch(emails)
            t_end = time.monotonic_ns()

            metrics = self._score(
                predictions=preds,
                labels=labels,
                total_elapsed_ns=t_end - t_start,
            )

            result_row = ModelBenchmarkVersionResult(
                model_benchmark_id=benchmark_id,
                version=version,
                accuracy=metrics["accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1=metrics["f1"],
                roc_auc=metrics["roc_auc"],
                ece=metrics["ece"],
                confusion_matrix=metrics["confusion_matrix"],
                per_zone_counts=metrics["per_zone_counts"],
                prediction_ms_p50=metrics["prediction_ms_p50"],
                prediction_ms_p95=metrics["prediction_ms_p95"],
            )
            async with async_session() as session:
                async with session.begin():
                    session.add(result_row)
                    await session.flush()
                    await session.refresh(result_row)

            self.sse_broker.publish(
                topic,
                _event(
                    "version_done",
                    BenchmarkEvent(
                        benchmarkId=benchmark_id,
                        kind="version_done",
                        version=version,
                        result=BenchmarkVersionResult.from_row(result_row),
                        emittedAt=_now(),
                    ).model_dump(by_alias=True, mode="json"),
                ),
            )

        # Flip to SUCCEEDED + stamp finished_at.
        async with async_session() as session:
            async with session.begin():
                await session.execute(
                    update(ModelBenchmark)
                    .where(ModelBenchmark.id == benchmark_id)
                    .values(
                        status=BenchmarkStatus.SUCCEEDED,
                        finished_at=_now(),
                    )
                )

        self.sse_broker.publish(
            topic,
            _event(
                "status",
                BenchmarkEvent(
                    benchmarkId=benchmark_id,
                    kind="status",
                    status=BenchmarkStatus.SUCCEEDED.value,
                    message=f"completed {len(versions)} version(s)",
                    emittedAt=_now(),
                ).model_dump(by_alias=True, mode="json"),
            ),
        )
        log.info(
            "benchmark succeeded id=%s versions=%s rows=%d",
            benchmark_id, versions, len(rows),
        )

    async def _fail(self, benchmark_id: UUID, *, message: str) -> None:
        try:
            async with async_session() as session:
                async with session.begin():
                    await session.execute(
                        update(ModelBenchmark)
                        .where(ModelBenchmark.id == benchmark_id)
                        .values(
                            status=BenchmarkStatus.FAILED,
                            error_message=message,
                            finished_at=_now(),
                        )
                    )
        except Exception:  # noqa: BLE001
            log.exception(
                "failed to record failure for benchmark id=%s", benchmark_id
            )

        try:
            self.sse_broker.publish(
                _topic(benchmark_id),
                _event(
                    "error",
                    BenchmarkEvent(
                        benchmarkId=benchmark_id,
                        kind="error",
                        status=BenchmarkStatus.FAILED.value,
                        message=message,
                        emittedAt=_now(),
                    ).model_dump(by_alias=True, mode="json"),
                ),
            )
        except Exception:  # noqa: BLE001
            log.exception(
                "failed to publish failure event for benchmark id=%s",
                benchmark_id,
            )

    # ── scoring ──

    def _score(
        self,
        *,
        predictions: list,
        labels: np.ndarray,
        total_elapsed_ns: int,
    ) -> dict[str, Any]:
        """Compute every metric stored on `model_benchmark_version_results`.

        Uses each prediction's own `threshold` to derive the hard label (so
        the detector's zone policy is honoured end-to-end); the probability
        is always the calibrated `phishing_probability` per the spec. ECE
        is the standard equal-width binning estimate — confidence is
        `max(phish, 1-phish)` and accuracy within each bin is measured
        against the ground-truth label.
        """
        probs = np.asarray(
            [float(p.phishing_probability) for p in predictions],
            dtype=np.float64,
        )
        preds = np.asarray(
            [int(p.predicted_label) for p in predictions],
            dtype=np.int64,
        )

        tp = int(((preds == 1) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        total = tp + tn + fp + fn

        accuracy = (tp + tn) / total if total else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) else 0.0
        )

        # roc_auc_score raises if only one class is present in `labels`.
        # A single-class benchmark dataset is a valid edge case (e.g. a
        # phishing-only holdout); report 0.0 so the side-by-side table
        # still renders rather than failing the whole run.
        if len(set(int(x) for x in labels.tolist())) < 2:
            roc_auc = 0.0
        else:
            roc_auc = float(roc_auc_score(labels, probs))

        ece = self._expected_calibration_error(
            probs=probs,
            labels=labels,
            num_bins=benchmark_config.ece_num_bins,
        )

        per_zone = {"NOT_SPAM": 0, "REVIEW": 0, "SPAM": 0}
        for p in predictions:
            zone = p.confidence_zone
            key = zone.value if zone is not None else "NOT_SPAM"
            per_zone[key] = per_zone.get(key, 0) + 1

        # Per-prediction latency from the batch call isn't available (the
        # detector hands back one timing for the whole batch), so evenly
        # distribute the wall-clock budget across rows. This matches how
        # downstream dashboards would render an "average per-row" cost and
        # keeps p50 == p95 on uniform batches, which is accurate.
        n = max(1, len(predictions))
        per_pred_ms = (total_elapsed_ns / 1_000_000.0) / n
        p50 = per_pred_ms
        p95 = per_pred_ms

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "ece": float(ece),
            "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "per_zone_counts": per_zone,
            "prediction_ms_p50": float(p50),
            "prediction_ms_p95": float(p95),
        }

    @staticmethod
    def _expected_calibration_error(
        *,
        probs: np.ndarray,
        labels: np.ndarray,
        num_bins: int,
    ) -> float:
        """Equal-width binning ECE. Confidence is `max(p, 1-p)`; accuracy
        is 1 if the predicted class matches the label, else 0. The return
        is a weighted mean of |accuracy - confidence| across bins."""
        if len(probs) == 0:
            return 0.0
        confidences = np.maximum(probs, 1.0 - probs)
        # Predicted class is the one with the higher probability (ties
        # break toward the positive class — rare and doesn't bias ECE).
        preds = (probs >= 0.5).astype(np.int64)
        accuracies = (preds == labels).astype(np.float64)

        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
        ece = 0.0
        total = float(len(probs))
        for i in range(num_bins):
            low = bin_edges[i]
            high = bin_edges[i + 1]
            if i == num_bins - 1:
                mask = (confidences >= low) & (confidences <= high)
            else:
                mask = (confidences >= low) & (confidences < high)
            if not mask.any():
                continue
            bin_conf = float(confidences[mask].mean())
            bin_acc = float(accuracies[mask].mean())
            weight = float(mask.sum()) / total
            ece += weight * abs(bin_acc - bin_conf)
        return ece

    # ── helpers ──

    def _registry(self) -> ModelRegistry:
        return ModelRegistry(inference_config.models_dir)

    @staticmethod
    def _validate_csv_mime(
        *, filename: str | None, content_type: str | None
    ) -> None:
        ext_ok = bool(filename) and filename.lower().endswith(
            _CSV_ALLOWED_EXTENSIONS
        )
        mime = (content_type or "").lower().split(";", 1)[0].strip()
        mime_ok = mime in _CSV_ALLOWED_MIMES if mime else True
        if not ext_ok and not mime_ok:
            raise BadRequestException(
                message="CSV upload rejected: unsupported file type",
                error_detail=ErrorDetail(
                    title="Unsupported Media Type",
                    code="BENCHMARK_CSV_UNSUPPORTED_TYPE",
                    status=415,
                    details=[
                        f"filename={filename!r}, content_type={content_type!r}; "
                        f"expected a .csv file with one of: "
                        f"{sorted(_CSV_ALLOWED_MIMES)}"
                    ],
                ),
            )

    # ── SSE helpers ──

    def sse_topic(self, benchmark_id: UUID) -> str:
        return _topic(benchmark_id)

    @property
    def sse_heartbeat_seconds(self) -> float:
        return sse_config.heartbeat_seconds
