from datetime import datetime
from typing import Sequence, Tuple
from uuid import UUID

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import lazyload

from src.app.models.enums import (
    AutoReviewAgreement as _AutoReviewAgreement,
    ReviewItemStatus,
    ReviewVerdict,
)
from src.app.models.review_item import ReviewItem
from src.shared.database import BaseRepository


_PENDING_STATUSES = (
    ReviewItemStatus.UNASSIGNED,
    ReviewItemStatus.ASSIGNED,
    ReviewItemStatus.IN_PROGRESS,
)

_MINE_OPEN_STATUSES = (
    ReviewItemStatus.ASSIGNED,
    ReviewItemStatus.IN_PROGRESS,
)


class ReviewItemRepository(BaseRepository[ReviewItem]):
    def __init__(self, session: AsyncSession):
        super().__init__(ReviewItem, session)

    # ── queries ──

    async def paginate_filtered(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        assigned_to: UUID | None = None,
        status: ReviewItemStatus | None = None,
        statuses: Sequence[ReviewItemStatus] | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> Tuple[Sequence[ReviewItem], int]:
        stmt = select(ReviewItem)
        if assigned_to is not None:
            stmt = stmt.where(ReviewItem.assigned_to == assigned_to)
        if status is not None:
            stmt = stmt.where(ReviewItem.status == status)
        if statuses:
            stmt = stmt.where(ReviewItem.status.in_(list(statuses)))
        if date_from is not None:
            stmt = stmt.where(ReviewItem.created_at >= date_from)
        if date_to is not None:
            stmt = stmt.where(ReviewItem.created_at <= date_to)

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await self.session.execute(count_stmt)).scalar_one()

        offset = (page - 1) * page_size
        stmt = (
            stmt.order_by(ReviewItem.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all(), total

    async def get_by_prediction_event_id(
        self, prediction_event_id: UUID
    ) -> ReviewItem | None:
        stmt = select(ReviewItem).where(
            ReviewItem.prediction_event_id == prediction_event_id
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()

    # ── claim (FOR UPDATE SKIP LOCKED) ──

    async def claim_for_update(self, review_item_id: UUID) -> ReviewItem | None:
        """Load a row for claim under `FOR UPDATE SKIP LOCKED` semantics.

        The lock is held on the current transaction only — callers must update
        the row inside the same session. Returns `None` when the row is already
        locked by a concurrent transaction, so two analysts racing on the same
        item produce a clear "someone else won" signal rather than a wait.

        The lock statement disables the model's default `lazy="joined"` loads
        because Postgres refuses `FOR UPDATE` on the nullable side of an outer
        join. After acquiring the lock we re-fetch the same row with the normal
        eager loads so callers can access `item.prediction_event`, `assignee`,
        and `decider` without triggering a lazy load under async.
        """
        lock_stmt = (
            select(ReviewItem.id)
            .where(ReviewItem.id == review_item_id)
            .with_for_update(skip_locked=True)
            .execution_options(populate_existing=True)
        )
        locked_id = (await self.session.execute(lock_stmt)).scalar_one_or_none()
        if locked_id is None:
            return None

        # Same transaction, lock still held. Re-fetch with the eager joins
        # the model declares so the service can access relationships safely.
        hydrate_stmt = select(ReviewItem).where(ReviewItem.id == locked_id)
        result = await self.session.execute(hydrate_stmt)
        return result.unique().scalars().first()

    # ── round-robin helper ──

    async def assigned_counts_for_users(
        self, user_ids: Sequence[UUID]
    ) -> dict[UUID, int]:
        """Return `{user_id: count}` for open review items assigned to each
        user. Used by the round-robin / least-loaded assignment strategy.
        """
        if not user_ids:
            return {}
        stmt = (
            select(ReviewItem.assigned_to, func.count(ReviewItem.id))
            .where(
                ReviewItem.assigned_to.in_(list(user_ids)),
                ReviewItem.status.in_(
                    [
                        ReviewItemStatus.ASSIGNED,
                        ReviewItemStatus.IN_PROGRESS,
                    ]
                ),
            )
            .group_by(ReviewItem.assigned_to)
        )
        result = await self.session.execute(stmt)
        counts = {uid: 0 for uid in user_ids}
        for assigned_to, count in result.all():
            if assigned_to is not None:
                counts[assigned_to] = int(count)
        return counts

    # ── dashboard queries (§9.1 / §9.2) ──

    async def pending_queue_stats(
        self, *, sla_seconds: int
    ) -> dict:
        """Aggregate age + status counts across the pending review queue.

        Pending = {UNASSIGNED, ASSIGNED, IN_PROGRESS}; age is measured from
        `created_at` so a freshly-claimed ticket still counts against the SLA.
        Empty-queue defaults (`average_age_seconds=0`, `oldest_age_seconds=0`)
        keep downstream DTO fields non-nullable.
        """
        age_expr = func.extract(
            "epoch", func.now() - ReviewItem.created_at
        )
        stmt = select(
            func.count().label("pending"),
            func.sum(
                case((ReviewItem.status == ReviewItemStatus.UNASSIGNED, 1), else_=0)
            ).label("unassigned"),
            func.sum(
                case((ReviewItem.status == ReviewItemStatus.IN_PROGRESS, 1), else_=0)
            ).label("in_progress"),
            func.coalesce(func.avg(age_expr), 0.0).label("avg_age"),
            func.coalesce(func.max(age_expr), 0.0).label("max_age"),
            func.sum(
                case((age_expr > sla_seconds, 1), else_=0)
            ).label("breached"),
        ).where(ReviewItem.status.in_(list(_PENDING_STATUSES)))
        row = (await self.session.execute(stmt)).one()
        return {
            "pending_count": int(row.pending or 0),
            "unassigned_count": int(row.unassigned or 0),
            "in_progress_count": int(row.in_progress or 0),
            "average_age_seconds": float(row.avg_age or 0.0),
            "oldest_age_seconds": float(row.max_age or 0.0),
            "sla_breach_count": int(row.breached or 0),
        }

    async def my_queue_stats(
        self, *, user_id: UUID, sla_seconds: int
    ) -> dict:
        """Age + SLA-breach aggregate for the caller's open assignments."""
        age_expr = func.extract(
            "epoch", func.now() - ReviewItem.created_at
        )
        stmt = select(
            func.count().label("assigned"),
            func.coalesce(func.avg(age_expr), 0.0).label("avg_age"),
            func.coalesce(func.max(age_expr), 0.0).label("max_age"),
            func.sum(
                case((age_expr > sla_seconds, 1), else_=0)
            ).label("breached"),
        ).where(
            ReviewItem.assigned_to == user_id,
            ReviewItem.status.in_(list(_MINE_OPEN_STATUSES)),
        )
        row = (await self.session.execute(stmt)).one()
        return {
            "assigned_count": int(row.assigned or 0),
            "average_age_seconds": float(row.avg_age or 0.0),
            "oldest_age_seconds": float(row.max_age or 0.0),
            "sla_breach_count": int(row.breached or 0),
        }

    async def analyst_decided_counts(
        self,
        *,
        user_ids: Sequence[UUID],
        since: datetime,
        status: ReviewItemStatus = ReviewItemStatus.CONFIRMED,
    ) -> dict[UUID, int]:
        """`{user_id: count}` of items a given analyst decided since `since`.

        Filters on `decided_at` (not `updated_at`) so the count tracks the
        moment the verdict landed; a later reassignment cannot retroactively
        add rows to the window.
        """
        if not user_ids:
            return {}
        stmt = (
            select(ReviewItem.decided_by, func.count(ReviewItem.id))
            .where(
                ReviewItem.decided_by.in_(list(user_ids)),
                ReviewItem.status == status,
                ReviewItem.decided_at.isnot(None),
                ReviewItem.decided_at >= since,
            )
            .group_by(ReviewItem.decided_by)
        )
        result = await self.session.execute(stmt)
        counts = {uid: 0 for uid in user_ids}
        for decided_by, count in result.all():
            if decided_by is not None:
                counts[decided_by] = int(count)
        return counts

    async def analyst_avg_decision_seconds(
        self, *, user_ids: Sequence[UUID], since: datetime
    ) -> dict[UUID, float]:
        """Average `decided_at - claimed_at` per analyst over the window.

        Only rows where both timestamps are populated count, so queue items
        that were assigned but never claimed before being reassigned away do
        not distort the mean."""
        if not user_ids:
            return {}
        duration_expr = func.extract(
            "epoch", ReviewItem.decided_at - ReviewItem.claimed_at
        )
        stmt = (
            select(
                ReviewItem.decided_by,
                func.coalesce(func.avg(duration_expr), 0.0),
            )
            .where(
                ReviewItem.decided_by.in_(list(user_ids)),
                ReviewItem.status == ReviewItemStatus.CONFIRMED,
                ReviewItem.decided_at.isnot(None),
                ReviewItem.claimed_at.isnot(None),
                ReviewItem.decided_at >= since,
            )
            .group_by(ReviewItem.decided_by)
        )
        result = await self.session.execute(stmt)
        out = {uid: 0.0 for uid in user_ids}
        for decided_by, avg_seconds in result.all():
            if decided_by is not None:
                out[decided_by] = float(avg_seconds or 0.0)
        return out

    async def personal_decision_stats(
        self, *, user_id: UUID, since: datetime
    ) -> dict:
        """Per-caller aggregate of decisions since `since` — counts by final
        status plus a mean decision duration across all CONFIRMED rows."""
        duration_expr = func.extract(
            "epoch", ReviewItem.decided_at - ReviewItem.claimed_at
        )
        stmt = select(
            func.sum(
                case(
                    (ReviewItem.status == ReviewItemStatus.CONFIRMED, 1),
                    else_=0,
                )
            ).label("completed"),
            func.sum(
                case(
                    (ReviewItem.status == ReviewItemStatus.DEFERRED, 1),
                    else_=0,
                )
            ).label("deferred"),
            func.sum(
                case(
                    (ReviewItem.status == ReviewItemStatus.ESCALATED, 1),
                    else_=0,
                )
            ).label("escalated"),
            func.coalesce(
                func.avg(
                    case(
                        (
                            (
                                ReviewItem.status
                                == ReviewItemStatus.CONFIRMED
                            )
                            & ReviewItem.claimed_at.isnot(None),
                            duration_expr,
                        ),
                        else_=None,
                    )
                ),
                0.0,
            ).label("avg_decision_seconds"),
        ).where(
            ReviewItem.decided_by == user_id,
            ReviewItem.decided_at.isnot(None),
            ReviewItem.decided_at >= since,
        )
        row = (await self.session.execute(stmt)).one()
        return {
            "completed": int(row.completed or 0),
            "deferred": int(row.deferred or 0),
            "escalated": int(row.escalated or 0),
            "average_decision_seconds": float(row.avg_decision_seconds or 0.0),
        }

    async def recent_decisions_for_user(
        self, *, user_id: UUID, limit: int
    ) -> Sequence[ReviewItem]:
        """Reverse-chronological list of the caller's most recent
        CONFIRMED/DEFERRED/ESCALATED items — backs `recentSubmissions`."""
        stmt = (
            select(ReviewItem)
            .where(
                ReviewItem.decided_by == user_id,
                ReviewItem.status.in_(
                    [
                        ReviewItemStatus.CONFIRMED,
                        ReviewItemStatus.DEFERRED,
                        ReviewItemStatus.ESCALATED,
                    ]
                ),
                ReviewItem.decided_at.isnot(None),
            )
            .order_by(ReviewItem.decided_at.desc())
            .limit(max(0, limit))
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def label_mix_for_user(
        self, *, user_id: UUID, since: datetime
    ) -> dict[str, int]:
        """Phishing / legitimate verdict counts across the caller's CONFIRMED
        items in the window. Missing verdicts are excluded — CONFIRMED
        implies both a verdict and a `decided_at`."""
        stmt = (
            select(ReviewItem.verdict, func.count(ReviewItem.id))
            .where(
                ReviewItem.decided_by == user_id,
                ReviewItem.status == ReviewItemStatus.CONFIRMED,
                ReviewItem.decided_at.isnot(None),
                ReviewItem.decided_at >= since,
                ReviewItem.verdict.isnot(None),
            )
            .group_by(ReviewItem.verdict)
        )
        result = await self.session.execute(stmt)
        counts = {"phishing": 0, "legitimate": 0}
        for verdict, count in result.all():
            if verdict is ReviewVerdict.PHISHING:
                counts["phishing"] = int(count)
            elif verdict is ReviewVerdict.LEGITIMATE:
                counts["legitimate"] = int(count)
        return counts

    async def auto_review_agreement_counts_for_user(
        self, *, user_id: UUID, since: datetime
    ) -> dict[str, int]:
        """Per-caller CONFIRMED counts bucketed by auto-review outcome — used
        by `autoReviewUsage` to compute the agreement rate and with/without
        splits."""
        stmt = select(
            func.sum(
                case((ReviewItem.auto_review_used.is_(True), 1), else_=0)
            ).label("with_auto"),
            func.sum(
                case((ReviewItem.auto_review_used.is_(False), 1), else_=0)
            ).label("without_auto"),
            func.sum(
                case(
                    (
                        ReviewItem.auto_review_agreement
                        == _AutoReviewAgreement.AGREED,
                        1,
                    ),
                    else_=0,
                )
            ).label("agreed"),
            func.sum(
                case(
                    (
                        ReviewItem.auto_review_agreement
                        == _AutoReviewAgreement.OVERRIDDEN,
                        1,
                    ),
                    else_=0,
                )
            ).label("overridden"),
        ).where(
            ReviewItem.decided_by == user_id,
            ReviewItem.status == ReviewItemStatus.CONFIRMED,
            ReviewItem.decided_at.isnot(None),
            ReviewItem.decided_at >= since,
        )
        row = (await self.session.execute(stmt)).one()
        return {
            "with_auto_review": int(row.with_auto or 0),
            "without_auto_review": int(row.without_auto or 0),
            "agreed_count": int(row.agreed or 0),
            "overridden_count": int(row.overridden or 0),
        }
