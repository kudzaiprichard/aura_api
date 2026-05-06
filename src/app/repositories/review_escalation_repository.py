from typing import Sequence, Tuple
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.models.review_escalation import ReviewEscalation
from src.shared.database import BaseRepository


class ReviewEscalationRepository(BaseRepository[ReviewEscalation]):
    def __init__(self, session: AsyncSession):
        super().__init__(ReviewEscalation, session)

    async def paginate_filtered(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        resolved: bool | None = None,
        escalated_to: UUID | None = None,
    ) -> Tuple[Sequence[ReviewEscalation], int]:
        stmt = select(ReviewEscalation)
        if resolved is True:
            stmt = stmt.where(ReviewEscalation.resolved_at.is_not(None))
        elif resolved is False:
            stmt = stmt.where(ReviewEscalation.resolved_at.is_(None))
        if escalated_to is not None:
            stmt = stmt.where(ReviewEscalation.escalated_to == escalated_to)

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await self.session.execute(count_stmt)).scalar_one()

        offset = (page - 1) * page_size
        stmt = (
            stmt.order_by(ReviewEscalation.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all(), total

    async def get_open_for_item(
        self, review_item_id: UUID
    ) -> ReviewEscalation | None:
        stmt = (
            select(ReviewEscalation)
            .where(
                ReviewEscalation.review_item_id == review_item_id,
                ReviewEscalation.resolved_at.is_(None),
            )
            .order_by(ReviewEscalation.created_at.desc())
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()
