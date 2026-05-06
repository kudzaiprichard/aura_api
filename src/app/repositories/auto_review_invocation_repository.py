from datetime import datetime
from typing import Sequence, Tuple
from uuid import UUID

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.models.auto_review_invocation import AutoReviewInvocation
from src.app.models.enums import AutoReviewOutcome
from src.shared.database import BaseRepository


class AutoReviewInvocationRepository(BaseRepository[AutoReviewInvocation]):
    def __init__(self, session: AsyncSession):
        super().__init__(AutoReviewInvocation, session)

    async def paginate_for_item(
        self,
        review_item_id: UUID,
        *,
        page: int = 1,
        page_size: int = 20,
    ) -> Tuple[Sequence[AutoReviewInvocation], int]:
        """History view for `GET /review/queue/{id}/auto-reviews`."""
        stmt = select(AutoReviewInvocation).where(
            AutoReviewInvocation.review_item_id == review_item_id
        )
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await self.session.execute(count_stmt)).scalar_one()

        offset = (page - 1) * page_size
        stmt = (
            stmt.order_by(AutoReviewInvocation.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all(), total

    async def paginate_for_actor(
        self,
        triggered_by: UUID,
        *,
        page: int = 1,
        page_size: int = 20,
    ) -> Tuple[Sequence[AutoReviewInvocation], int]:
        """Reverse-chronological feed of invocations driven by a given user."""
        stmt = select(AutoReviewInvocation).where(
            AutoReviewInvocation.triggered_by == triggered_by
        )
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await self.session.execute(count_stmt)).scalar_one()

        offset = (page - 1) * page_size
        stmt = (
            stmt.order_by(AutoReviewInvocation.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all(), total

    async def list_by_batch_group(
        self, batch_group_id: UUID
    ) -> Sequence[AutoReviewInvocation]:
        stmt = (
            select(AutoReviewInvocation)
            .where(AutoReviewInvocation.batch_group_id == batch_group_id)
            .order_by(AutoReviewInvocation.created_at.asc())
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    # ── dashboard queries (§9.2) ──

    async def stats_for_actor_since(
        self, *, triggered_by: UUID, since: datetime
    ) -> dict[str, int]:
        """Total + failure counts of the caller's invocations since `since`.

        Returns zero-initialised counts on an empty window so the dashboard
        can compute agreement ratios without a missing-row branch."""
        stmt = select(
            func.count().label("total"),
            func.sum(
                case(
                    (
                        AutoReviewInvocation.outcome == AutoReviewOutcome.FAILURE,
                        1,
                    ),
                    else_=0,
                )
            ).label("failures"),
        ).where(
            AutoReviewInvocation.triggered_by == triggered_by,
            AutoReviewInvocation.created_at >= since,
        )
        row = (await self.session.execute(stmt)).one()
        return {
            "invocation_count": int(row.total or 0),
            "invocation_failure_count": int(row.failures or 0),
        }

    async def recent_for_actor_since(
        self,
        *,
        triggered_by: UUID,
        since: datetime,
        limit: int,
    ) -> Sequence[AutoReviewInvocation]:
        """Newest-first invocations inside the window — feeds the
        `recentInvocations` slice with the joined `review_item` already
        eager-loaded by the model's relationship configuration."""
        stmt = (
            select(AutoReviewInvocation)
            .where(
                AutoReviewInvocation.triggered_by == triggered_by,
                AutoReviewInvocation.created_at >= since,
            )
            .order_by(AutoReviewInvocation.created_at.desc())
            .limit(max(0, limit))
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()
