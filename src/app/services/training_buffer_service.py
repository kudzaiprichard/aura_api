import csv
import hashlib
import io
import logging
from datetime import datetime
from typing import Sequence, Tuple
from uuid import UUID

from src.configs import training as training_config
from src.app.helpers.quality_gate import (
    quality_gate_max_oov,
    violates_quality_gate,
)
from src.app.models.enums import TrainingBufferSource
from src.app.models.training_buffer_item import TrainingBufferItem
from src.app.models.user import User
from src.app.repositories.training_buffer_repository import (
    TrainingBufferRepository,
)
from src.shared.exceptions import (
    BadRequestException,
    NotFoundException,
)
from src.shared.inference import PhishingDetector
from src.shared.responses import ErrorDetail


log = logging.getLogger("aura.training.buffer")


_CSV_REQUIRED_COLUMNS = ("sender", "subject", "body", "label")
_CSV_OPTIONAL_COLUMNS = ("category",)
_CSV_ALLOWED_MIMES = {
    "text/csv",
    "application/csv",
    "application/vnd.ms-excel",
    "text/plain",
    "application/octet-stream",
}
_CSV_ALLOWED_EXTENSIONS = (".csv",)


def _content_sha256(sender: str, subject: str, body: str) -> str:
    payload = f"{sender}\n{subject}\n{body}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class BufferImportRowError:
    """Per-row error captured during CSV parsing.

    Carried as a plain object (rather than a dict) so the service caller can
    inspect both fields without string parsing; the controller projects it to
    the response DTO."""

    __slots__ = ("row_number", "message")

    def __init__(self, row_number: int, message: str):
        self.row_number = row_number
        self.message = message


class BufferImportSummary:
    __slots__ = (
        "total_rows",
        "inserted",
        "duplicates",
        "errors",
        "file_sha256",
    )

    def __init__(
        self,
        *,
        total_rows: int,
        inserted: int,
        duplicates: int,
        errors: list[BufferImportRowError],
        file_sha256: str,
    ):
        self.total_rows = total_rows
        self.inserted = inserted
        self.duplicates = duplicates
        self.errors = errors
        self.file_sha256 = file_sha256


class BufferStatus:
    __slots__ = (
        "size",
        "class_counts",
        "unlocked",
        "blockers",
        "min_batch_size",
        "min_per_class",
        "require_balance_delta",
    )

    def __init__(
        self,
        *,
        size: int,
        class_counts: dict[int, int],
        unlocked: bool,
        blockers: list[str],
        min_batch_size: int,
        min_per_class: int,
        require_balance_delta: float,
    ):
        self.size = size
        self.class_counts = class_counts
        self.unlocked = unlocked
        self.blockers = blockers
        self.min_batch_size = min_batch_size
        self.min_per_class = min_per_class
        self.require_balance_delta = require_balance_delta


class TrainingBufferService:
    """Owns the buffer status / CSV-import / CRUD surface (§3.4, §3.6, §7.4).

    The buffer table itself is fed in two places:
      * `ReviewService._create_training_buffer_entry` for confirmed reviews
      * `import_csv` here for admin uploads

    The transaction is owned by the FastAPI dependency (`get_db`); writes only
    flush, the request boundary commits."""

    def __init__(
        self,
        buffer_repository: TrainingBufferRepository,
        detector: PhishingDetector | None = None,
    ):
        self.buffer_repo = buffer_repository
        # Phase 12 — CSV import enforces the OOV quality gate row-by-row
        # when enabled and a detector is loaded. No detector → gate is
        # skipped; rejections are reported as row-level errors rather than
        # a request-level failure so the caller still gets per-row detail.
        self.detector = detector

    # ── status ──

    async def status(self) -> BufferStatus:
        """Compute the unlock state against the configured thresholds.

        `class_counts` is always reported as `{0: n, 1: m}` so the UI can render
        zero-class bars; `blockers` lists every reason the buffer is locked so
        admins see the full picture rather than being told one issue at a time.
        """
        observed = await self.buffer_repo.count_by_label()
        # Spec is strictly binary (label ∈ {0, 1}); always surface both classes.
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
                    f"{minority_share:.3f} below the required {min_share_required:.3f} "
                    f"(require_balance_delta={balance_delta:.3f})"
                )

        return BufferStatus(
            size=size,
            class_counts=class_counts,
            unlocked=not blockers,
            blockers=blockers,
            min_batch_size=min_batch,
            min_per_class=min_per_class,
            require_balance_delta=balance_delta,
        )

    # ── reads ──

    async def list(
        self,
        *,
        page: int,
        page_size: int,
        label: int | None = None,
        source: TrainingBufferSource | None = None,
        category: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> Tuple[Sequence[TrainingBufferItem], int]:
        return await self.buffer_repo.paginate_filtered(
            page=page,
            page_size=page_size,
            label=label,
            source=source,
            category=category,
            date_from=date_from,
            date_to=date_to,
        )

    async def get(self, item_id: UUID) -> TrainingBufferItem:
        item = await self.buffer_repo.get_by_id(item_id)
        if item is None:
            raise NotFoundException(
                message="Buffer item not found",
                error_detail=ErrorDetail(
                    title="Not Found",
                    code="BUFFER_ITEM_NOT_FOUND",
                    status=404,
                    details=[f"No buffer item found with id {item_id}"],
                ),
            )
        return item

    # ── writes ──

    async def delete(self, item_id: UUID) -> None:
        item = await self.get(item_id)
        await self.buffer_repo.delete(item)
        log.info("training_buffer item deleted id=%s", item_id)

    async def import_csv(
        self,
        *,
        payload: bytes,
        filename: str | None,
        content_type: str | None,
        actor: User,
    ) -> BufferImportSummary:
        """Validate + dedupe + bulk-insert a CSV upload.

        The size and MIME checks fire *before* parsing, so an oversize upload
        gets a 413 without us walking the file. Parsing happens against the
        bytes we've already loaded — the controller is responsible for
        streaming the upload up to the size cap.
        """
        size_cap = training_config.csv_max_bytes
        row_cap = training_config.csv_max_rows

        size = len(payload)
        if size > size_cap:
            raise BadRequestException(
                message="CSV upload exceeds the configured size cap",
                error_detail=ErrorDetail(
                    title="Payload Too Large",
                    code="BUFFER_CSV_TOO_LARGE",
                    status=413,
                    details=[
                        f"Received {size} bytes; "
                        f"training.csv_max_bytes = {size_cap}"
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
                    code="BUFFER_CSV_ENCODING",
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
                    code="BUFFER_CSV_EMPTY",
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
                    code="BUFFER_CSV_MISSING_COLUMNS",
                    status=400,
                    details=[
                        f"Missing column(s): {', '.join(missing)}. "
                        f"Required: {', '.join(_CSV_REQUIRED_COLUMNS)}; "
                        f"optional: {', '.join(_CSV_OPTIONAL_COLUMNS)}"
                    ],
                ),
            )

        col_sender = normalised_headers["sender"]
        col_subject = normalised_headers["subject"]
        col_body = normalised_headers["body"]
        col_label = normalised_headers["label"]
        col_category = normalised_headers.get("category")

        errors: list[BufferImportRowError] = []
        candidates: list[dict] = []
        seen_in_file: set[str] = set()
        duplicates_in_file = 0

        total_rows = 0
        # csv.DictReader is 1-indexed at the *header*, so data rows start at 2.
        for row_index, raw in enumerate(reader, start=2):
            total_rows += 1
            if total_rows > row_cap:
                raise BadRequestException(
                    message="CSV exceeds the configured row cap",
                    error_detail=ErrorDetail(
                        title="Payload Too Large",
                        code="BUFFER_CSV_TOO_MANY_ROWS",
                        status=413,
                        details=[
                            f"More than {row_cap} data rows; "
                            f"training.csv_max_rows = {row_cap}"
                        ],
                    ),
                )

            sender = (raw.get(col_sender) or "").strip()
            subject = (raw.get(col_subject) or "").strip()
            body = (raw.get(col_body) or "").strip()
            label_raw = (raw.get(col_label) or "").strip()
            category = None
            if col_category is not None:
                cat_value = (raw.get(col_category) or "").strip()
                category = cat_value or None

            row_errors: list[str] = []
            if not sender:
                row_errors.append("sender is required")
            if not body:
                row_errors.append("body is required")
            label_int: int | None = None
            try:
                label_int = int(label_raw)
            except ValueError:
                row_errors.append(
                    f"label must be 0 or 1 (got {label_raw!r})"
                )
            else:
                if label_int not in (0, 1):
                    row_errors.append(
                        f"label must be 0 or 1 (got {label_int})"
                    )
            if category is not None and len(category) > 64:
                row_errors.append(
                    f"category exceeds 64 chars (got {len(category)})"
                )

            if row_errors:
                errors.append(
                    BufferImportRowError(
                        row_number=row_index,
                        message="; ".join(row_errors),
                    )
                )
                continue

            rejected, rate = violates_quality_gate(
                detector=self.detector,
                subject=subject,
                body=body,
            )
            if rejected:
                errors.append(
                    BufferImportRowError(
                        row_number=row_index,
                        message=(
                            f"quality gate rejected: OOV rate "
                            f"{rate:.3f} exceeds the configured maximum "
                            f"{quality_gate_max_oov():.3f}"
                        ),
                    )
                )
                continue

            sha = _content_sha256(sender, subject, body)
            if sha in seen_in_file:
                duplicates_in_file += 1
                continue
            seen_in_file.add(sha)
            candidates.append(
                {
                    "sender": sender,
                    "subject": subject,
                    "body": body,
                    "label": label_int,
                    "category": category,
                    "content_sha256": sha,
                }
            )

        existing = await self.buffer_repo.existing_content_sha256s(
            c["content_sha256"] for c in candidates
        )
        to_insert = [c for c in candidates if c["content_sha256"] not in existing]

        entities = [
            TrainingBufferItem(
                sender=c["sender"],
                subject=c["subject"],
                body=c["body"],
                label=c["label"],
                source=TrainingBufferSource.CSV_IMPORT,
                source_prediction_event_id=None,
                source_review_item_id=None,
                category=c["category"],
                content_sha256=c["content_sha256"],
                contributed_by=actor.id,
                consumed_in_run_ids=[],
            )
            for c in to_insert
        ]
        await self.buffer_repo.bulk_insert(entities)

        duplicates = duplicates_in_file + len(candidates) - len(to_insert)

        # Audit log per §4.3 / Phase 12 hardening — never the body, only the
        # file hash, actor, and counts.
        log.info(
            "training_buffer.csv_import actor_id=%s file_sha256=%s "
            "filename=%s total_rows=%d inserted=%d duplicates=%d errors=%d",
            actor.id,
            file_sha,
            filename or "<unnamed>",
            total_rows,
            len(entities),
            duplicates,
            len(errors),
        )

        return BufferImportSummary(
            total_rows=total_rows,
            inserted=len(entities),
            duplicates=duplicates,
            errors=errors,
            file_sha256=file_sha,
        )

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
                    code="BUFFER_CSV_UNSUPPORTED_TYPE",
                    status=415,
                    details=[
                        f"filename={filename!r}, content_type={content_type!r}; "
                        f"expected a .csv file with one of: "
                        f"{sorted(_CSV_ALLOWED_MIMES)}"
                    ],
                ),
            )
