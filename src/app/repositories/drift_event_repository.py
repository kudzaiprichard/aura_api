from datetime import datetime
from typing import Iterable, Sequence
from uuid import UUID

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.models.drift_event import (
    DriftEvent,
    EVENT_TYPE_CONFIRMATION,
    EVENT_TYPE_PREDICTION,
)
from src.shared.database import BaseRepository


# Buckets accepted by `bucketed_fpr` — kept narrow so the public API is the
# single source of truth (no guessing what postgres date_trunc accepts).
_VALID_BUCKETS = {"hour", "day"}


class DriftEventRepository(BaseRepository[DriftEvent]):
    """SQL mirror of the JSONL drift log (§3.9, §6.12).

    Writes happen alongside the review-confirm / prediction transactions so
    the SQL state and the row-state changes commit together. The JSONL log
    remains the source of truth; reconciliation on boot rebuilds this table
    when they diverge.
    """

    def __init__(self, session: AsyncSession):
        super().__init__(DriftEvent, session)

    # ── reads ──

    async def count_total(self) -> int:
        stmt = select(func.count()).select_from(DriftEvent)
        return (await self.session.execute(stmt)).scalar_one()

    async def count_by_event_type(self) -> dict[str, int]:
        stmt = select(DriftEvent.event_type, func.count()).group_by(
            DriftEvent.event_type
        )
        rows = (await self.session.execute(stmt)).all()
        return {event_type: int(c) for event_type, c in rows}

    async def bucketed_fpr(
        self,
        *,
        bucket: str,
        date_from: datetime | None,
        date_to: datetime | None,
        timezone_name: str = "UTC",
    ) -> list[dict]:
        """Hour- or day-bucketed FPR computed from the drift_events mirror.

        FPR per bucket = fp / (fp + tn) over confirmations whose paired
        prediction landed in the same bucket. We bucket on the confirmation's
        `occurred_at` so a delayed analyst decision is attributed to when the
        feedback signal materialised — that matches how the JSONL-backed
        cumulative `DriftMonitor` accumulates the same numbers.

        `timezone_name` is an IANA tz id; date_trunc honours it so day
        boundaries align to the caller's wall clock.
        """
        if bucket not in _VALID_BUCKETS:
            raise ValueError(
                f"bucket must be one of {sorted(_VALID_BUCKETS)}, got {bucket!r}"
            )

        confirmations = (
            select(
                DriftEvent.prediction_id.label("prediction_id"),
                DriftEvent.confirmed_label.label("confirmed_label"),
                DriftEvent.occurred_at.label("occurred_at"),
            )
            .where(DriftEvent.event_type == EVENT_TYPE_CONFIRMATION)
            .where(DriftEvent.confirmed_label.isnot(None))
        )
        if date_from is not None:
            confirmations = confirmations.where(
                DriftEvent.occurred_at >= date_from
            )
        if date_to is not None:
            confirmations = confirmations.where(
                DriftEvent.occurred_at <= date_to
            )
        confirmations_sub = confirmations.subquery()

        # Latest predicted_label per prediction_id (the JSONL only ever writes
        # one prediction record per id, but this guards against any post-hoc
        # rebuild that imported duplicates from a corrupt log).
        predictions_sub = (
            select(
                DriftEvent.prediction_id.label("prediction_id"),
                DriftEvent.predicted_label.label("predicted_label"),
            )
            .where(DriftEvent.event_type == EVENT_TYPE_PREDICTION)
            .where(DriftEvent.predicted_label.isnot(None))
            .subquery()
        )

        bucket_col = func.date_trunc(
            bucket, confirmations_sub.c.occurred_at, timezone_name
        ).label("bucket")
        fp_expr = func.sum(
            case(
                (
                    (predictions_sub.c.predicted_label == 1)
                    & (confirmations_sub.c.confirmed_label == 0),
                    1,
                ),
                else_=0,
            )
        ).label("fp")
        tn_expr = func.sum(
            case(
                (
                    (predictions_sub.c.predicted_label == 0)
                    & (confirmations_sub.c.confirmed_label == 0),
                    1,
                ),
                else_=0,
            )
        ).label("tn")
        tp_expr = func.sum(
            case(
                (
                    (predictions_sub.c.predicted_label == 1)
                    & (confirmations_sub.c.confirmed_label == 1),
                    1,
                ),
                else_=0,
            )
        ).label("tp")
        fn_expr = func.sum(
            case(
                (
                    (predictions_sub.c.predicted_label == 0)
                    & (confirmations_sub.c.confirmed_label == 1),
                    1,
                ),
                else_=0,
            )
        ).label("fn")
        confirmed_count = func.count(confirmations_sub.c.prediction_id).label(
            "confirmed"
        )

        stmt = (
            select(
                bucket_col,
                fp_expr,
                tn_expr,
                tp_expr,
                fn_expr,
                confirmed_count,
            )
            .select_from(
                confirmations_sub.join(
                    predictions_sub,
                    predictions_sub.c.prediction_id
                    == confirmations_sub.c.prediction_id,
                )
            )
            .group_by(bucket_col)
            .order_by(bucket_col.asc())
        )

        rows = (await self.session.execute(stmt)).all()
        out: list[dict] = []
        for row in rows:
            fp = int(row.fp or 0)
            tn = int(row.tn or 0)
            tp = int(row.tp or 0)
            fn = int(row.fn or 0)
            denom = fp + tn
            fpr = (fp / denom) if denom > 0 else 0.0
            out.append(
                {
                    "bucket": row.bucket,
                    "false_positive_rate": fpr,
                    "fp": fp,
                    "tn": tn,
                    "tp": tp,
                    "fn": fn,
                    "confirmed": int(row.confirmed or 0),
                }
            )
        return out

    async def list_recent(
        self,
        *,
        limit: int = 50,
        event_type: str | None = None,
    ) -> Sequence[DriftEvent]:
        stmt = select(DriftEvent)
        if event_type is not None:
            stmt = stmt.where(DriftEvent.event_type == event_type)
        stmt = stmt.order_by(DriftEvent.occurred_at.desc()).limit(limit)
        return (await self.session.execute(stmt)).scalars().all()

    # ── writes ──

    async def record_prediction(
        self,
        *,
        prediction_id: UUID,
        predicted_label: int,
        predicted_probability: float,
        model_version: str | None,
        occurred_at: datetime,
    ) -> DriftEvent:
        return await self.create(
            DriftEvent(
                event_type=EVENT_TYPE_PREDICTION,
                prediction_id=prediction_id,
                predicted_label=int(predicted_label),
                predicted_probability=float(predicted_probability),
                confirmed_label=None,
                model_version=model_version,
                occurred_at=occurred_at,
            )
        )

    async def record_confirmation(
        self,
        *,
        prediction_id: UUID,
        confirmed_label: int,
        occurred_at: datetime,
    ) -> DriftEvent:
        return await self.create(
            DriftEvent(
                event_type=EVENT_TYPE_CONFIRMATION,
                prediction_id=prediction_id,
                predicted_label=None,
                predicted_probability=None,
                confirmed_label=int(confirmed_label),
                model_version=None,
                occurred_at=occurred_at,
            )
        )

    # ── reconciliation ──

    async def truncate(self) -> None:
        """Used by lifespan reconciliation: wipe the mirror and rebuild from
        JSONL when the two diverge. Issued as TRUNCATE so the index pages drop
        too — bulk DELETE on a divergent mirror would be much slower.
        """
        await self.session.execute(
            DriftEvent.__table__.delete()
        )
        await self.session.flush()

    async def bulk_insert(self, events: Iterable[DriftEvent]) -> int:
        """Bulk insert pre-built DriftEvent rows during reconciliation. Returns
        the count actually added (callers pass an iterable so a generator can
        stream from the JSONL replay without materialising the whole log).
        """
        added = 0
        for ev in events:
            self.session.add(ev)
            added += 1
        if added > 0:
            await self.session.flush()
        return added
