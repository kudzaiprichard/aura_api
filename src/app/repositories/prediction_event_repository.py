from datetime import datetime
from typing import Sequence, Tuple
from uuid import UUID

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.models.drift_event import DriftEvent, EVENT_TYPE_CONFIRMATION
from src.app.models.enums import ConfidenceZone
from src.app.models.prediction_event import PredictionEvent
from src.shared.database import BaseRepository


_VALID_METRIC_BUCKETS = {"hour", "day"}
_VALID_VOLUME_BUCKETS = {"hour", "day"}


class PredictionEventRepository(BaseRepository[PredictionEvent]):
    def __init__(self, session: AsyncSession):
        super().__init__(PredictionEvent, session)

    async def paginate_filtered(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        requester_id: UUID | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        model_version: str | None = None,
        predicted_label: int | None = None,
        confidence_zone: ConfidenceZone | None = None,
    ) -> Tuple[Sequence[PredictionEvent], int]:
        """Paginate prediction_events with the filter surface exposed by
        GET /api/v1/analysis/predictions.

        Results are always ordered by predicted_at DESC so the newest
        predictions appear first.
        """
        stmt = select(PredictionEvent)

        if requester_id is not None:
            stmt = stmt.where(PredictionEvent.requester_id == requester_id)
        if date_from is not None:
            stmt = stmt.where(PredictionEvent.predicted_at >= date_from)
        if date_to is not None:
            stmt = stmt.where(PredictionEvent.predicted_at <= date_to)
        if model_version is not None:
            stmt = stmt.where(PredictionEvent.model_version == model_version)
        if predicted_label is not None:
            stmt = stmt.where(PredictionEvent.predicted_label == predicted_label)
        if confidence_zone is not None:
            stmt = stmt.where(PredictionEvent.confidence_zone == confidence_zone)

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await self.session.execute(count_stmt)).scalar_one()

        offset = (page - 1) * page_size
        stmt = (
            stmt.order_by(PredictionEvent.predicted_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all(), total

    async def paginate_for_install(
        self,
        *,
        install_id: UUID,
        page: int = 1,
        page_size: int = 20,
    ) -> Tuple[Sequence[PredictionEvent], int]:
        """Newest-first slice of prediction_events attributable to a single
        extension install, for the admin activity feed.

        Extension-sourced rows are tagged at write time with
        `request_id = f"ext:{install_id.hex}:{rid}"` (see
        `extension_email_service.tagged_request_id`). There is no
        install_id column on prediction_events, so the install scope is
        carried entirely in the request_id prefix. The pattern is
        produced by the writer-side helper so format changes stay in
        lockstep.
        """
        # Imported lazily to avoid a service → repository import edge.
        from src.app.services.extension_email_service import (
            install_request_id_pattern,
        )

        pattern = install_request_id_pattern(install_id)
        base = select(PredictionEvent).where(
            PredictionEvent.request_id.like(pattern)
        )

        count_stmt = select(func.count()).select_from(base.subquery())
        total = (await self.session.execute(count_stmt)).scalar_one()

        offset = (page - 1) * page_size
        stmt = (
            base.order_by(PredictionEvent.predicted_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all(), int(total)

    async def bucketed_metrics_for_versions(
        self,
        *,
        versions: Sequence[str],
        bucket: str,
        date_from: datetime | None,
        date_to: datetime | None,
        timezone_name: str = "UTC",
    ) -> list[dict]:
        """Per-version bucketed confusion-matrix counts for Phase 9 §7.6.

        Joins prediction_events (filtered by model_version) to drift_events
        confirmations on the shared `prediction_id`. Buckets on the
        confirmation's `occurred_at` so the metric is attributed to the
        moment the feedback landed — mirroring how `DriftEventRepository.
        bucketed_fpr` attributes FPR.

        Returns one row per (version, bucket) with raw tp/tn/fp/fn + the
        derived accuracy / precision / recall / f1 / fpr — leaving the final
        DTO shaping to the service layer.
        """
        if bucket not in _VALID_METRIC_BUCKETS:
            raise ValueError(
                f"bucket must be one of {sorted(_VALID_METRIC_BUCKETS)}, "
                f"got {bucket!r}"
            )
        if not versions:
            return []

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

        predictions_sub = (
            select(
                PredictionEvent.prediction_id.label("prediction_id"),
                PredictionEvent.predicted_label.label("predicted_label"),
                PredictionEvent.model_version.label("model_version"),
            )
            .where(PredictionEvent.prediction_id.isnot(None))
            .where(PredictionEvent.model_version.in_(list(versions)))
            .subquery()
        )

        bucket_col = func.date_trunc(
            bucket, confirmations_sub.c.occurred_at, timezone_name
        ).label("bucket")
        version_col = predictions_sub.c.model_version.label("model_version")

        def _count_case(predicted: int, confirmed: int, label: str):
            return func.sum(
                case(
                    (
                        (predictions_sub.c.predicted_label == predicted)
                        & (confirmations_sub.c.confirmed_label == confirmed),
                        1,
                    ),
                    else_=0,
                )
            ).label(label)

        tp_expr = _count_case(1, 1, "tp")
        tn_expr = _count_case(0, 0, "tn")
        fp_expr = _count_case(1, 0, "fp")
        fn_expr = _count_case(0, 1, "fn")
        confirmed_count = func.count(confirmations_sub.c.prediction_id).label(
            "confirmed"
        )

        stmt = (
            select(
                version_col,
                bucket_col,
                tp_expr,
                tn_expr,
                fp_expr,
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
            .group_by(version_col, bucket_col)
            .order_by(version_col.asc(), bucket_col.asc())
        )

        rows = (await self.session.execute(stmt)).all()
        out: list[dict] = []
        for row in rows:
            tp = int(row.tp or 0)
            tn = int(row.tn or 0)
            fp = int(row.fp or 0)
            fn = int(row.fn or 0)
            total = tp + tn + fp + fn
            accuracy = (tp + tn) / total if total > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                (2 * precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            fpr_denom = fp + tn
            fpr = fp / fpr_denom if fpr_denom > 0 else 0.0
            out.append(
                {
                    "version": row.model_version,
                    "bucket": row.bucket,
                    "tp": tp,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "confirmed": int(row.confirmed or 0),
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "false_positive_rate": fpr,
                }
            )
        return out

    # ── dashboard queries (§9.1) ──

    async def recent_events(self, *, limit: int) -> Sequence[PredictionEvent]:
        """Newest `limit` prediction events across all requesters, newest first.

        Used by `AdminDashboardResponse.recentPredictions`; the requester
        relationship is eager-loaded via the model's `lazy="joined"` setting,
        so callers may read `event.requester` without an extra round-trip.
        """
        stmt = (
            select(PredictionEvent)
            .order_by(PredictionEvent.predicted_at.desc())
            .limit(max(0, limit))
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def classification_breakdown(
        self,
        *,
        date_from: datetime,
        date_to: datetime,
        timezone_name: str = "UTC",
    ) -> list[dict]:
        """Daily spam / legitimate / review counts across all prediction events
        in the window.

        `review` is bucketed independently of the 0/1 predicted label because
        the §9.1 contract exposes it as a third lane alongside the two
        classifier labels; the count is therefore not a subset of either lane.
        """
        bucket_col = func.date_trunc(
            "day", PredictionEvent.predicted_at, timezone_name
        ).label("bucket")
        phishing_expr = func.sum(
            case((PredictionEvent.predicted_label == 1, 1), else_=0)
        ).label("phishing")
        legitimate_expr = func.sum(
            case((PredictionEvent.predicted_label == 0, 1), else_=0)
        ).label("legitimate")
        review_expr = func.sum(
            case(
                (
                    PredictionEvent.confidence_zone == ConfidenceZone.REVIEW,
                    1,
                ),
                else_=0,
            )
        ).label("review")

        stmt = (
            select(bucket_col, phishing_expr, legitimate_expr, review_expr)
            .where(PredictionEvent.predicted_at >= date_from)
            .where(PredictionEvent.predicted_at < date_to)
            .group_by(bucket_col)
            .order_by(bucket_col.asc())
        )
        rows = (await self.session.execute(stmt)).all()
        return [
            {
                "bucket": row.bucket,
                "phishing": int(row.phishing or 0),
                "legitimate": int(row.legitimate or 0),
                "review": int(row.review or 0),
            }
            for row in rows
        ]

    async def volume_buckets(
        self,
        *,
        bucket: str,
        date_from: datetime,
        date_to: datetime,
        timezone_name: str = "UTC",
    ) -> list[dict]:
        """Per-bucket count of prediction events within the window.

        `bucket` accepts `hour` or `day`; anything else raises `ValueError` so
        the controller's validation surface stays the single source of truth.
        """
        if bucket not in _VALID_VOLUME_BUCKETS:
            raise ValueError(
                f"bucket must be one of {sorted(_VALID_VOLUME_BUCKETS)}, "
                f"got {bucket!r}"
            )
        bucket_col = func.date_trunc(
            bucket, PredictionEvent.predicted_at, timezone_name
        ).label("bucket")
        stmt = (
            select(bucket_col, func.count().label("count"))
            .where(PredictionEvent.predicted_at >= date_from)
            .where(PredictionEvent.predicted_at < date_to)
            .group_by(bucket_col)
            .order_by(bucket_col.asc())
        )
        rows = (await self.session.execute(stmt)).all()
        return [
            {"t": row.bucket, "count": int(row.count or 0)} for row in rows
        ]
