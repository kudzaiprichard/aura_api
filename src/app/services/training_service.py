"""Training-run orchestration (§3.5, §8.6, Phase 8).

Owns the online-learning lifecycle:

  1. Resolve a slice of `training_buffer_items` from filters + selection.
  2. Apply the balance strategy (UNDERSAMPLE / OVERSAMPLE / NONE).
  3. Load the labelled holdout so `OnlineLearner` populates
     `performance_before/after` — closing the §2.7 silent-skip gap.
  4. Run `OnlineLearner.partial_fit_batch` off the event loop (the call is
     blocking and takes the portalocker on `{models_root}/.lock`).
  5. Persist a `training_runs` row with metadata + metrics, update every
     consumed buffer item's `consumed_in_run_ids`, and fan-out progress
     events on the SSE broker.

One run at a time — the service holds `app.state.training_lock` for the
duration and also records active runs in SQL so a persisted PENDING/RUNNING
row from a crashed worker still blocks a fresh submission (§3.5).
"""
from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import UUID

import numpy as np
import pandas as pd
from sqlalchemy import event, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.configs import inference as inference_config
from src.configs import sse as sse_config
from src.configs import training as training_config
from src.app.models.enums import (
    BalanceStrategy,
    TrainingBufferSource,
    TrainingRunStatus,
)
from src.app.models.training_buffer_item import TrainingBufferItem
from src.app.models.training_run import TrainingRun
from src.app.models.user import User
from src.app.dtos.requests import TrainingRunRequest
from src.app.dtos.responses import TrainingRunEvent
from src.app.repositories.training_buffer_repository import (
    TrainingBufferRepository,
)
from src.app.repositories.training_run_repository import TrainingRunRepository
from src.core.background_jobs import BackgroundJobRunner
from src.core.sse import SSEBroker, SSEEvent, monotonic_event_id
from src.shared.database import async_session
from src.shared.exceptions import (
    BadRequestException,
    ConflictException,
    NotFoundException,
    ServiceUnavailableException,
)
from src.shared.inference import OnlineLearner
from src.shared.inference.registry import ModelRegistry
from src.shared.responses import ErrorDetail


log = logging.getLogger("aura.training.runs")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _topic(run_id: UUID) -> str:
    return f"training-run:{run_id}"


def _event(kind: str, payload: dict[str, Any]) -> SSEEvent:
    """SSE frame carrying a `TrainingRunEvent` payload. `id` is monotonic so
    reconnecting clients can resume via Last-Event-ID, and `event` carries
    the kind so EventSource handlers can fan out without parsing the body."""
    return SSEEvent(
        data=payload, event=kind, id=monotonic_event_id()
    )


# Commit-visibility retry budget used by the background worker. The
# `after_commit` hook in `start_run` can fire before a fresh
# `async_session()` sees the just-inserted row (the benchmark service logs
# caught this at ~3ms). Retrying the row load for a few hundred ms absorbs
# the gap without re-introducing a schedule-before-commit race.
_ROW_DURABLE_RETRIES = 5
_ROW_DURABLE_BACKOFF_SECONDS = 0.05


class TrainingService:
    """Owns `POST /training/runs` and the SSE progress stream.

    The service receives a `session` for synchronous queries from the
    controller thread (status checks, detail reads) and `background_jobs` +
    `training_lock` for the asynchronous fit. Every DB write performed from
    the background worker runs against a fresh `async_session()` — the
    controller's session is bound to the request and has already committed
    by the time the worker starts."""

    def __init__(
        self,
        *,
        session: AsyncSession,
        training_run_repository: TrainingRunRepository,
        training_buffer_repository: TrainingBufferRepository,
        sse_broker: SSEBroker,
        background_jobs: BackgroundJobRunner,
        training_lock: asyncio.Lock,
        app_state: Any,
    ):
        self.session = session
        self.run_repo = training_run_repository
        self.buffer_repo = training_buffer_repository
        self.sse_broker = sse_broker
        self.background_jobs = background_jobs
        self.training_lock = training_lock
        # Needed so the background worker can publish `cancel` flags on the
        # same dict the controller reads. `app.state.training_cancel_flags`
        # is lazily initialised on first access.
        self.app_state = app_state

    # ── cancel flag registry ──

    def _cancel_flags(self) -> dict[UUID, bool]:
        flags = getattr(self.app_state, "training_cancel_flags", None)
        if flags is None:
            flags = {}
            self.app_state.training_cancel_flags = flags
        return flags

    # ── public API ──

    async def start_run(
        self, request: TrainingRunRequest, *, actor: User
    ) -> TrainingRun:
        """Validate the request, create a PENDING row, submit the background
        worker, and return the row. Every caller-visible refusal lands as a
        409 before any state is mutated."""
        if request.auto_promote and not self._holdout_dir_has_data():
            raise ConflictException(
                message="Auto-promote requires a holdout set",
                error_detail=ErrorDetail(
                    title="Conflict",
                    code="PROMOTION_GUARD_UNSAFE",
                    status=409,
                    details=[
                        "auto_promote=true but no holdout is configured at "
                        f"training.holdout_path ({training_config.holdout_path}); "
                        "refusing to promote without pre/post metrics (§2.7).",
                    ],
                ),
            )

        registry = self._registry()
        source_version = request.source_version or registry.active_version()
        if source_version is None:
            raise ServiceUnavailableException(
                message="No source model version is available",
                error_detail=ErrorDetail(
                    title="Service Unavailable",
                    code="TRAINING_NO_SOURCE_VERSION",
                    status=503,
                    details=[
                        "No active version in the registry and no "
                        "source_version was supplied",
                    ],
                ),
            )
        if source_version not in registry.list_versions():
            raise BadRequestException(
                message="Unknown source_version",
                error_detail=ErrorDetail(
                    title="Bad Request",
                    code="TRAINING_UNKNOWN_SOURCE_VERSION",
                    status=400,
                    details=[
                        f"source_version={source_version!r} is not registered; "
                        f"known versions: {registry.list_versions()}"
                    ],
                ),
            )

        await self._ensure_no_run_in_progress()
        await self._ensure_buffer_unlocked()
        await self._ensure_slice_non_empty(request)

        run = TrainingRun(
            triggered_by=actor.id,
            status=TrainingRunStatus.PENDING,
            source_version=source_version,
            new_version=None,
            batch_size=0,
            iterations=0,
            balance_strategy=request.selection.balance_strategy,
            shuffle=request.selection.shuffle,
            seed=request.selection.seed,
            min_delta_f1=request.min_delta_f1,
            max_iter_per_call=request.max_iter_per_call,
            performance_before=None,
            performance_after=None,
            oov_rate_subject=None,
            oov_rate_body=None,
            buffer_snapshot_ids=[],
            class_counts_before_balance={"0": 0, "1": 0},
            class_counts_after_balance={"0": 0, "1": 0},
            promoted=False,
            error_message=None,
        )
        run = await self.run_repo.create(run)

        # Defer the worker until the request's transaction actually commits
        # (mirrors the fix in BenchmarkService). The worker's first UPDATE
        # against the run row would silently match zero rows if the insert is
        # still uncommitted, leaving the run stuck in PENDING until the later
        # writes (also races) happen to land after the commit. Hooking
        # `after_commit` makes the scheduling deterministic.
        run_id = run.id
        frozen_request = request.model_copy(
            update={"source_version": source_version}
        )

        def _submit_worker(_sync_session: Any) -> None:
            self.background_jobs.submit(
                self._execute_run(run_id=run_id, request=frozen_request),
                name=f"training-run-{run_id}",
            )

        event.listen(
            self.session.sync_session,
            "after_commit",
            _submit_worker,
            once=True,
        )
        log.info(
            "training_run submitted id=%s source_version=%s actor_id=%s",
            run.id, source_version, actor.id,
        )
        return run

    async def list_runs(
        self,
        *,
        page: int,
        page_size: int,
        status: TrainingRunStatus | None = None,
        triggered_by: UUID | None = None,
    ) -> tuple[Sequence[TrainingRun], int]:
        return await self.run_repo.paginate_filtered(
            page=page,
            page_size=page_size,
            status=status,
            triggered_by=triggered_by,
        )

    async def get_run(self, run_id: UUID) -> TrainingRun:
        run = await self.run_repo.get_by_id(run_id)
        if run is None:
            raise NotFoundException(
                message="Training run not found",
                error_detail=ErrorDetail(
                    title="Not Found",
                    code="TRAINING_RUN_NOT_FOUND",
                    status=404,
                    details=[f"No training run found with id {run_id}"],
                ),
            )
        return run

    async def preview_run(
        self, request: TrainingRunRequest
    ) -> dict[str, Any]:
        """Dry-run variant of `start_run` — counts the rows the filter +
        selection would pull, without touching `training_runs` or locks.

        Returns the same feasibility verdict `_ensure_slice_non_empty` uses,
        so a feasible preview is a reliable predictor that the real POST
        will clear the gate (modulo buffer changes between calls)."""
        counts = await self._count_selectable_by_label(
            self.session, request
        )
        n0 = counts.get(0, 0)
        n1 = counts.get(1, 0)
        strategy = request.selection.balance_strategy
        empty = self._slice_would_be_empty(counts, strategy)

        reason_code: str | None = None
        message: str | None = None
        if empty:
            if (n0 + n1) == 0:
                reason_code = "FILTER_EMPTY"
                message = "No buffer rows match the current filter."
            else:
                reason_code = "BALANCE_INFEASIBLE"
                message = (
                    f"{strategy.value} requires both classes; matched "
                    f"class 0={n0}, class 1={n1}."
                )

        return {
            "matched_total": n0 + n1,
            "matched_class_0": n0,
            "matched_class_1": n1,
            "balance_strategy": strategy.value,
            "feasible": not empty,
            "reason_code": reason_code,
            "message": message,
        }

    async def cancel(self, run_id: UUID) -> TrainingRun:
        """Set the cooperative cancel flag. The background worker checks it
        before invoking the blocking fit; if the fit is already in flight
        the flag is consulted again on the next event-loop turn. Either way
        the run lands as FAILED with `error_message="cancelled"` (§3.5)."""
        run = await self.get_run(run_id)
        if run.status not in (
            TrainingRunStatus.PENDING, TrainingRunStatus.RUNNING,
        ):
            raise ConflictException(
                message="Run is not cancellable",
                error_detail=ErrorDetail(
                    title="Conflict",
                    code="TRAINING_RUN_NOT_CANCELLABLE",
                    status=409,
                    details=[
                        f"Run is {run.status.value}; only PENDING or RUNNING "
                        f"runs can be cancelled"
                    ],
                ),
            )
        self._cancel_flags()[run.id] = True
        log.info("training_run cancel requested id=%s", run.id)
        return run

    # ── validation helpers ──

    async def _ensure_no_run_in_progress(self) -> None:
        active = await self.run_repo.count_in_progress()
        if active > 0:
            raise ConflictException(
                message="A training run is already in progress",
                error_detail=ErrorDetail(
                    title="Conflict",
                    code="TRAINING_IN_PROGRESS",
                    status=409,
                    details=[
                        f"{active} training run(s) currently PENDING or RUNNING; "
                        "wait for completion before starting another",
                    ],
                ),
            )

    async def _ensure_buffer_unlocked(self) -> None:
        """Mirrors `TrainingBufferService.status()`'s gate rules. Duplicating
        a small amount of logic here keeps the training path self-contained
        — we must not reach across to TrainingBufferService from a sibling
        service (both own separate state)."""
        observed = await self.buffer_repo.count_by_label()
        class_counts = {0: observed.get(0, 0), 1: observed.get(1, 0)}
        size = sum(class_counts.values())

        min_batch = training_config.min_batch_size
        min_per_class = training_config.min_per_class
        balance_delta = training_config.require_balance_delta

        blockers: list[str] = []
        if size < min_batch:
            blockers.append(
                f"Buffer size {size} below the required minimum of {min_batch}"
            )
        for label, count in class_counts.items():
            if count < min_per_class:
                blockers.append(
                    f"Class {label} count {count} below the per-class minimum "
                    f"of {min_per_class}"
                )
        if size > 0:
            minority_share = min(class_counts.values()) / size
            min_share_required = max(0.0, 0.5 - balance_delta)
            if minority_share < min_share_required:
                blockers.append(
                    f"Buffer is too imbalanced: minority share "
                    f"{minority_share:.3f} below the required "
                    f"{min_share_required:.3f} "
                    f"(require_balance_delta={balance_delta:.3f})"
                )
        if blockers:
            raise ConflictException(
                message="Training buffer is locked",
                error_detail=ErrorDetail(
                    title="Conflict",
                    code="TRAINING_BUFFER_LOCKED",
                    status=409,
                    details=blockers,
                ),
            )

    # ── inference helpers ──

    def _registry(self) -> ModelRegistry:
        return ModelRegistry(inference_config.models_dir)

    def _holdout_dir_has_data(self) -> bool:
        path = Path(training_config.holdout_path)
        if not path.exists():
            return False
        if path.is_file():
            return path.stat().st_size > 0
        return any(path.glob("*.csv"))

    def _load_holdout(
        self,
    ) -> tuple[pd.DataFrame, np.ndarray] | None:
        """Read the holdout CSV(s) under `training.holdout_path` and return
        a `(df, y)` pair shaped for `OnlineLearner.holdout_set`.

        Each CSV must carry `sender, subject, body, label`. Missing or empty
        paths return None so callers can still proceed without metrics when
        `auto_promote=false` — the run completes with `performance_*=None`.
        """
        path = Path(training_config.holdout_path)
        if not path.exists():
            return None

        if path.is_file():
            files = [path]
        else:
            files = sorted(path.glob("*.csv"))
        if not files:
            return None

        frames: list[pd.DataFrame] = []
        for f in files:
            try:
                frames.append(pd.read_csv(f))
            except Exception:  # noqa: BLE001
                log.exception("Failed to read holdout CSV %s", f)
        if not frames:
            return None
        df = pd.concat(frames, ignore_index=True)
        required = {"sender", "subject", "body", "label"}
        if not required.issubset(df.columns):
            log.warning(
                "Holdout set at %s missing required columns %s; skipping",
                path, required - set(df.columns),
            )
            return None
        df = df.dropna(subset=["label"]).copy()
        y = df["label"].astype(int).to_numpy()
        df = df[["sender", "subject", "body"]].fillna("").astype(str)
        return df, y

    # ── buffer slicing ──

    def _apply_filters(self, stmt: Any, filters: Any) -> Any:
        """Attach the request's filter predicates to a `select()` over
        `training_buffer_items`. Shared by `_select_buffer` (which loads the
        slice in the worker) and `_count_selectable_by_label` (the pre-submit
        gate) so both paths agree on which rows are selectable."""
        if filters.sources:
            stmt = stmt.where(
                TrainingBufferItem.source.in_(list(filters.sources))
            )
        if filters.date_from is not None:
            stmt = stmt.where(
                TrainingBufferItem.created_at >= filters.date_from
            )
        if filters.date_to is not None:
            stmt = stmt.where(
                TrainingBufferItem.created_at <= filters.date_to
            )
        if filters.categories:
            stmt = stmt.where(
                TrainingBufferItem.category.in_(list(filters.categories))
            )
        return stmt

    async def _select_buffer(
        self, session: AsyncSession, request: TrainingRunRequest
    ) -> list[TrainingBufferItem]:
        """Resolve the raw slice of `training_buffer_items` per the filter
        and `max_size` caps before balancing. Ordered by created_at ASC so
        the slice is deterministic across retries, with `random.Random(seed)`
        used for any subsequent shuffle."""
        stmt = self._apply_filters(
            select(TrainingBufferItem), request.filters
        )
        stmt = stmt.order_by(TrainingBufferItem.created_at.asc())
        rows = (await session.execute(stmt)).scalars().all()
        return list(rows)

    async def _count_selectable_by_label(
        self, session: AsyncSession, request: TrainingRunRequest
    ) -> dict[int, int]:
        """Count buffer rows matching the request's filters grouped by label,
        using the same predicates as `_select_buffer`. Drives the pre-submit
        emptiness gate without loading the rows."""
        stmt = self._apply_filters(
            select(
                TrainingBufferItem.label,
                func.count(TrainingBufferItem.id),
            ),
            request.filters,
        ).group_by(TrainingBufferItem.label)
        result = await session.execute(stmt)
        return {int(label): int(count) for label, count in result.all()}

    @staticmethod
    def _slice_would_be_empty(
        counts: dict[int, int], strategy: BalanceStrategy
    ) -> bool:
        """Return True iff `_apply_balance` would yield zero rows given the
        per-label counts. UNDERSAMPLE trims both classes to the smaller one,
        so a zero-count class collapses the slice entirely; NONE/OVERSAMPLE
        only fail when the whole filtered set is empty (OVERSAMPLE skips
        empty classes, it does not fail)."""
        n0 = counts.get(0, 0)
        n1 = counts.get(1, 0)
        if strategy == BalanceStrategy.UNDERSAMPLE:
            return min(n0, n1) == 0
        return (n0 + n1) == 0

    @staticmethod
    def _empty_slice_error(
        counts: dict[int, int], strategy: BalanceStrategy
    ) -> BadRequestException:
        """Build a `TRAINING_EMPTY_SLICE` 400 whose `details` explain *why*
        the slice is empty.

        Two distinct failure modes share this code (frontend compat):
          - filter-empty: no rows matched the filter at all.
          - balance-infeasible: rows exist but UNDERSAMPLE needs both labels
            and one label's count is zero.
        The details array is ordered so the first entry is the actionable
        diagnosis and subsequent entries carry the numeric context the UI
        can parse for hints.
        """
        n0 = counts.get(0, 0)
        n1 = counts.get(1, 0)
        total = n0 + n1
        if total == 0:
            details = [
                "No buffer rows matched the filter.",
                "Relax the filter: widen the date window, remove the sources "
                "list, or pick different categories.",
                f"matchedTotal=0 matchedClass0=0 matchedClass1=0 "
                f"strategy={strategy.value}",
            ]
        else:
            details = [
                f"Filter matched {total} row(s), but balance strategy "
                f"{strategy.value} requires both classes to be present.",
                f"Class 0 has {n0} row(s); Class 1 has {n1} row(s). "
                "Switch strategy to NONE or OVERSAMPLE, or widen filters to "
                "include the missing class.",
                f"matchedTotal={total} matchedClass0={n0} matchedClass1={n1} "
                f"strategy={strategy.value}",
            ]
        return BadRequestException(
            message="Selected buffer slice is empty",
            error_detail=ErrorDetail(
                title="Bad Request",
                code="TRAINING_EMPTY_SLICE",
                status=400,
                details=details,
            ),
        )

    async def _ensure_slice_non_empty(
        self, request: TrainingRunRequest
    ) -> None:
        """Pre-submit gate: refuse a run whose filters + balance strategy
        would produce an empty training slice. Mirrors the worker-side check
        in `_execute_locked` so the caller sees a 400 on POST rather than a
        201 followed by an asynchronously FAILED run."""
        counts = await self._count_selectable_by_label(
            self.session, request
        )
        strategy = request.selection.balance_strategy
        if self._slice_would_be_empty(counts, strategy):
            # TEMP DEBUG — remove once the root-cause filter is identified.
            log.warning(
                "training_run gate rejected: empty slice "
                "filters=%s strategy=%s max_size=%s counts=%s",
                request.filters.model_dump(mode="json"),
                strategy.value,
                request.selection.max_size,
                counts,
            )
            raise self._empty_slice_error(counts, strategy)

    def _apply_balance(
        self,
        rows: list[TrainingBufferItem],
        request: TrainingRunRequest,
    ) -> list[TrainingBufferItem]:
        """Apply the requested balance strategy + max_size cap + shuffle.

        Ordering rule — independently of strategy:
          1. Balance first (the strategy picks which rows survive/duplicate).
          2. Shuffle the result when `selection.shuffle=True`.
          3. Apply `selection.max_size` last (so the cap never biases class
             ratios when the balance produced a post-balance excess).
        """
        selection = request.selection
        rng = random.Random(selection.seed)

        by_label: dict[int, list[TrainingBufferItem]] = {0: [], 1: []}
        for r in rows:
            by_label.setdefault(int(r.label), []).append(r)

        balanced: list[TrainingBufferItem] = []
        strategy = selection.balance_strategy

        if strategy == BalanceStrategy.UNDERSAMPLE:
            min_count = min(len(by_label[0]), len(by_label[1]))
            for label in (0, 1):
                pool = list(by_label[label])
                rng.shuffle(pool)
                balanced.extend(pool[:min_count])
        elif strategy == BalanceStrategy.OVERSAMPLE:
            max_count = max(len(by_label[0]), len(by_label[1]))
            for label in (0, 1):
                pool = list(by_label[label])
                if not pool:
                    continue
                while len(pool) < max_count:
                    pool.append(rng.choice(by_label[label]))
                balanced.extend(pool)
        else:
            balanced = list(rows)

        if selection.shuffle:
            rng.shuffle(balanced)
        if selection.max_size is not None and len(balanced) > selection.max_size:
            balanced = balanced[: selection.max_size]
        return balanced

    # ── background worker ──

    async def _execute_run(
        self, *, run_id: UUID, request: TrainingRunRequest
    ) -> None:
        """Runs as an `asyncio.Task` under `BackgroundJobRunner`. Owns the
        transaction for every write it issues; swallows cancellation to land
        the run as FAILED with `error_message="cancelled"` so callers that
        resubscribe after the fact still see the final status."""
        try:
            async with self.training_lock:
                await self._execute_locked(run_id=run_id, request=request)
        except asyncio.CancelledError:
            await self._fail_run(
                run_id,
                message="cancelled",
                status=TrainingRunStatus.FAILED,
            )
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("training_run failed id=%s", run_id)
            await self._fail_run(
                run_id,
                message=str(exc),
                status=TrainingRunStatus.FAILED,
            )
        finally:
            self._cancel_flags().pop(run_id, None)

    async def _execute_locked(
        self, *, run_id: UUID, request: TrainingRunRequest
    ) -> None:
        topic = _topic(run_id)

        # Wait for the outer request's commit to be visible before we
        # UPDATE. A raw UPDATE against a not-yet-visible row matches zero
        # rows silently, leaving the run wedged in PENDING with no error.
        # Benchmarks hit this race intermittently (~2/11 in logs); training
        # has the same pattern, just a worse failure mode without this guard.
        for attempt in range(_ROW_DURABLE_RETRIES):
            async with async_session() as session:
                async with session.begin():
                    run = await session.get(TrainingRun, run_id)
                    if run is not None:
                        break
            await asyncio.sleep(_ROW_DURABLE_BACKOFF_SECONDS)
        else:
            log.warning(
                "training_run id=%s not found after %d retries (%.0fms); "
                "aborting",
                run_id,
                _ROW_DURABLE_RETRIES,
                _ROW_DURABLE_RETRIES * _ROW_DURABLE_BACKOFF_SECONDS * 1000,
            )
            return

        self.sse_broker.publish(
            topic,
            _event(
                "status",
                TrainingRunEvent(
                    runId=run_id,
                    kind="status",
                    status=TrainingRunStatus.RUNNING.value,
                    message="run started",
                    emittedAt=_now(),
                ).model_dump(by_alias=True, mode="json"),
            ),
        )

        # Flip the row to RUNNING + stamp started_at so concurrent readers
        # see a consistent lifecycle.
        async with async_session() as session:
            async with session.begin():
                await session.execute(
                    update(TrainingRun)
                    .where(TrainingRun.id == run_id)
                    .values(
                        status=TrainingRunStatus.RUNNING,
                        started_at=_now(),
                    )
                )

        # Cooperative cancellation point — before we load anything heavy.
        if self._cancel_flags().get(run_id):
            raise asyncio.CancelledError()

        # ── select + balance buffer ──
        async with async_session() as session:
            async with session.begin():
                rows = await self._select_buffer(session, request)

        class_counts_before = {
            "0": sum(1 for r in rows if int(r.label) == 0),
            "1": sum(1 for r in rows if int(r.label) == 1),
        }
        # Race guard: the buffer can change between `_ensure_slice_non_empty`
        # (pre-submit) and this worker pickup. Re-check using the same balance
        # rules so we fail fast with an actionable message rather than letting
        # a silent-empty slice through to the learner.
        strategy = request.selection.balance_strategy
        raw_counts = {0: class_counts_before["0"], 1: class_counts_before["1"]}
        if self._slice_would_be_empty(raw_counts, strategy):
            raise self._empty_slice_error(raw_counts, strategy)
        balanced = self._apply_balance(rows, request)
        class_counts_after = {
            "0": sum(1 for r in balanced if int(r.label) == 0),
            "1": sum(1 for r in balanced if int(r.label) == 1),
        }
        snapshot_ids = [r.id for r in balanced]

        self.sse_broker.publish(
            topic,
            _event(
                "iteration",
                TrainingRunEvent(
                    runId=run_id,
                    kind="iteration",
                    iteration=0,
                    metrics={
                        "batchSize": len(balanced),
                        "classCountsBefore": class_counts_before,
                        "classCountsAfter": class_counts_after,
                    },
                    message="slice resolved",
                    emittedAt=_now(),
                ).model_dump(by_alias=True, mode="json"),
            ),
        )

        if self._cancel_flags().get(run_id):
            raise asyncio.CancelledError()

        # ── fit ──
        holdout = self._load_holdout()
        learner = OnlineLearner(
            registry=self._registry(),
            holdout_set=holdout,
        )
        emails = [
            {
                "sender": r.sender,
                "subject": r.subject,
                "body": r.body,
                "label": int(r.label),
            }
            for r in balanced
        ]

        loop = asyncio.get_running_loop()
        source_version = request.source_version
        result = await loop.run_in_executor(
            None,
            lambda: learner.partial_fit_batch(
                emails,
                source_version=source_version,
                max_iter_per_call=request.max_iter_per_call,
            ),
        )

        self.sse_broker.publish(
            topic,
            _event(
                "version_registered",
                TrainingRunEvent(
                    runId=run_id,
                    kind="version_registered",
                    new_version=result.new_version,
                    metrics={
                        "performanceBefore": dict(result.performance_before or {}),
                        "performanceAfter": dict(result.performance_after or {}),
                    },
                    oovRateSubject=result.oov_rate_subject,
                    oovRateBody=result.oov_rate_body,
                    emittedAt=_now(),
                ).model_dump(by_alias=True, mode="json"),
            ),
        )

        # ── persist success + stamp buffer rows ──
        async with async_session() as session:
            async with session.begin():
                await session.execute(
                    update(TrainingRun)
                    .where(TrainingRun.id == run_id)
                    .values(
                        status=TrainingRunStatus.SUCCEEDED,
                        new_version=result.new_version,
                        batch_size=len(balanced),
                        iterations=result.iterations,
                        performance_before=(
                            dict(result.performance_before or {}) or None
                        ),
                        performance_after=(
                            dict(result.performance_after or {}) or None
                        ),
                        oov_rate_subject=result.oov_rate_subject,
                        oov_rate_body=result.oov_rate_body,
                        buffer_snapshot_ids=snapshot_ids,
                        class_counts_before_balance=class_counts_before,
                        class_counts_after_balance=class_counts_after,
                        finished_at=_now(),
                    )
                )
                # Append run_id to every consumed buffer row's provenance
                # array. Stamps duplicates in OVERSAMPLE only once per row.
                unique_ids = list({r.id for r in balanced})
                if unique_ids:
                    stmt = select(TrainingBufferItem).where(
                        TrainingBufferItem.id.in_(unique_ids)
                    )
                    items = (await session.execute(stmt)).scalars().all()
                    for item in items:
                        current = list(item.consumed_in_run_ids or [])
                        if run_id not in current:
                            current.append(run_id)
                            item.consumed_in_run_ids = current

        # ── auto-promote (guarded) ──
        auto_promoted = False
        if request.auto_promote:
            if not (result.performance_after or {}):
                # Per §2.7: refuse silently-unsafe promotions. The run stays
                # SUCCEEDED; the refusal is surfaced on the SSE stream and
                # leaves the row unpromoted for the admin to action manually.
                self.sse_broker.publish(
                    topic,
                    _event(
                        "error",
                        TrainingRunEvent(
                            runId=run_id,
                            kind="error",
                            message=(
                                "PROMOTION_GUARD_UNSAFE: performance_after "
                                "empty; promotion refused"
                            ),
                            emittedAt=_now(),
                        ).model_dump(by_alias=True, mode="json"),
                    ),
                )
            else:
                await loop.run_in_executor(
                    None,
                    lambda: learner.promote(
                        result.new_version,
                        min_delta_f1=request.min_delta_f1,
                    ),
                )
                auto_promoted = True
                async with async_session() as session:
                    async with session.begin():
                        await session.execute(
                            update(TrainingRun)
                            .where(TrainingRun.id == run_id)
                            .values(
                                promoted=True,
                                promoted_at=_now(),
                                promoted_by=None,
                            )
                        )

        self.sse_broker.publish(
            topic,
            _event(
                "status",
                TrainingRunEvent(
                    runId=run_id,
                    kind="status",
                    status=TrainingRunStatus.SUCCEEDED.value,
                    new_version=result.new_version,
                    message=(
                        "promoted" if auto_promoted else "completed"
                    ),
                    emittedAt=_now(),
                ).model_dump(by_alias=True, mode="json"),
            ),
        )
        log.info(
            "training_run succeeded id=%s new_version=%s promoted=%s",
            run_id, result.new_version, auto_promoted,
        )

    async def _fail_run(
        self,
        run_id: UUID,
        *,
        message: str,
        status: TrainingRunStatus,
    ) -> None:
        try:
            async with async_session() as session:
                async with session.begin():
                    await session.execute(
                        update(TrainingRun)
                        .where(TrainingRun.id == run_id)
                        .values(
                            status=status,
                            error_message=message,
                            finished_at=_now(),
                        )
                    )
        except Exception:  # noqa: BLE001
            log.exception("failed to record failure for run id=%s", run_id)

        try:
            self.sse_broker.publish(
                _topic(run_id),
                _event(
                    "status",
                    TrainingRunEvent(
                        runId=run_id,
                        kind="status",
                        status=status.value,
                        message=message,
                        emittedAt=_now(),
                    ).model_dump(by_alias=True, mode="json"),
                ),
            )
        except Exception:  # noqa: BLE001
            log.exception("failed to publish failure event for run id=%s", run_id)

    # ── SSE helper ──

    def sse_topic(self, run_id: UUID) -> str:
        return _topic(run_id)

    @property
    def sse_heartbeat_seconds(self) -> float:
        return sse_config.heartbeat_seconds
