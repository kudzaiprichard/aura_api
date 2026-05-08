import logging
from datetime import datetime
from uuid import UUID

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.repositories.drift_event_repository import DriftEventRepository
from src.shared.exceptions import BadRequestException
from src.shared.inference import DriftMonitor, DriftSignal
from src.shared.responses import ErrorDetail


log = logging.getLogger("aura.drift")


class DriftService:
    """Read-side adapter over `DriftMonitor` (live cumulative state from
    JSONL replay) plus the `drift_events` SQL mirror (for bucketed queries
    over time).

    The two stores share the same source of truth: the SQL mirror is written
    inside review-confirm / prediction transactions; the JSONL log is written
    by the `DriftMonitor` itself either through the detector at predict time
    or through the post-commit hook at confirm time. Reconciliation on boot
    rebuilds SQL from JSONL when they diverge — the JSONL log wins by design.
    """

    def __init__(
        self,
        session: AsyncSession,
        drift_event_repository: DriftEventRepository,
        drift_monitor: DriftMonitor,
    ):
        self.session = session
        self.drift_event_repo = drift_event_repository
        self.drift_monitor = drift_monitor

    # ── reads ──

    def signal(self) -> DriftSignal:
        """Live `DriftSignal` straight from the JSONL-backed monitor."""
        return self.drift_monitor.drift_signal()

    def confusion_matrix(self) -> dict[str, int]:
        return self.drift_monitor.confusion_matrix()

    async def history(
        self,
        *,
        bucket: str,
        date_from: datetime | None,
        date_to: datetime | None,
        timezone_name: str,
    ) -> list[dict]:
        """Bucketed FPR over time, computed from the SQL mirror.

        `bucket` is `'hour'` or `'day'`. `timezone_name` is an IANA tz id —
        date_trunc honours it so day boundaries align to the caller's wall
        clock per the §3.9 / Phase 5 acceptance criteria.
        """
        return await self.drift_event_repo.bucketed_fpr(
            bucket=bucket,
            date_from=date_from,
            date_to=date_to,
            timezone_name=timezone_name,
        )

    # ── writes ──

    def update_threshold(self, fpr_threshold: float) -> DriftSignal:
        """Mutate the live monitor's FPR threshold without restart.

        Takes effect on the next call to `signal()` (which is what every
        request reads). The same validation rules `DriftMonitor` enforces in
        its constructor apply here: must be a real number in `[0, 1]`.
        """
        if isinstance(fpr_threshold, bool) or not isinstance(
            fpr_threshold, (int, float)
        ):
            raise BadRequestException(
                message="fprThreshold must be a number",
                error_detail=ErrorDetail(
                    title="Bad Request",
                    code="DRIFT_THRESHOLD_INVALID",
                    status=400,
                    details=["fprThreshold must be a numeric value"],
                ),
            )
        if fpr_threshold < 0.0 or fpr_threshold > 1.0:
            raise BadRequestException(
                message="fprThreshold must be in [0, 1]",
                error_detail=ErrorDetail(
                    title="Bad Request",
                    code="DRIFT_THRESHOLD_OUT_OF_RANGE",
                    status=400,
                    details=[
                        f"Received {fpr_threshold!r}; valid range is [0, 1]"
                    ],
                ),
            )
        previous = self.drift_monitor.fpr_threshold
        self.drift_monitor.fpr_threshold = float(fpr_threshold)
        log.info(
            "drift fpr_threshold updated %.6f -> %.6f",
            previous, self.drift_monitor.fpr_threshold,
        )
        return self.drift_monitor.drift_signal()

    async def record_manual_confirmation(
        self,
        *,
        prediction_id: UUID,
        confirmed_label: int,
        occurred_at: datetime,
    ) -> DriftSignal:
        """Admin-driven manual confirmation (§7.8 POST /drift/confirm).

        Same feedback path as the review-confirm flow minus the review row
        and training-buffer entry: the SQL mirror is written in-transaction
        and the JSONL update is scheduled post-commit so the two stay in
        lockstep.
        """
        if confirmed_label not in (0, 1):
            raise BadRequestException(
                message="confirmedLabel must be 0 or 1",
                error_detail=ErrorDetail(
                    title="Bad Request",
                    code="DRIFT_CONFIRM_LABEL_INVALID",
                    status=400,
                    details=[
                        f"Received {confirmed_label!r}; valid values are 0, 1"
                    ],
                ),
            )

        await self.drift_event_repo.record_confirmation(
            prediction_id=prediction_id,
            confirmed_label=int(confirmed_label),
            occurred_at=occurred_at,
        )

        pid_str = str(prediction_id)
        label = int(confirmed_label)
        monitor = self.drift_monitor

        def _on_commit(_session) -> None:
            try:
                monitor.record_confirmation(pid_str, label)
            except Exception:
                # Mirror is committed; the JSONL replay on the next boot
                # will reconcile from the SQL side if this hop drops.
                log.exception(
                    "drift_monitor.record_confirmation failed (manual) "
                    "prediction_id=%s",
                    pid_str,
                )

        event.listen(
            self.session.sync_session,
            "after_commit",
            _on_commit,
            once=True,
        )

        return monitor.drift_signal()
