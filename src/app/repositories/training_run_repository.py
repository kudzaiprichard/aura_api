from datetime import datetime
from typing import Sequence, Tuple
from uuid import UUID

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from src.app.models.enums import TrainingRunStatus
from src.app.models.training_run import TrainingRun
from src.shared.database import BaseRepository


class TrainingRunRepository(BaseRepository[TrainingRun]):
    def __init__(self, session: AsyncSession):
        super().__init__(TrainingRun, session)

    async def paginate_filtered(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        status: TrainingRunStatus | None = None,
        triggered_by: UUID | None = None,
    ) -> Tuple[Sequence[TrainingRun], int]:
        stmt = select(TrainingRun)
        if status is not None:
            stmt = stmt.where(TrainingRun.status == status)
        if triggered_by is not None:
            stmt = stmt.where(TrainingRun.triggered_by == triggered_by)

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await self.session.execute(count_stmt)).scalar_one()

        offset = (page - 1) * page_size
        stmt = (
            stmt.order_by(TrainingRun.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all(), total

    async def count_in_progress(self) -> int:
        """Active runs — PENDING or RUNNING. Used by `POST /runs` to enforce
        the single-run invariant (§3.5): we hold an in-process lock, but a
        persisted check also catches a run that is still alive after a crashed
        worker released the lock prematurely.
        """
        stmt = (
            select(func.count())
            .select_from(TrainingRun)
            .where(
                TrainingRun.status.in_(
                    [TrainingRunStatus.PENDING, TrainingRunStatus.RUNNING]
                )
            )
        )
        return (await self.session.execute(stmt)).scalar_one()

    async def latest(self) -> TrainingRun | None:
        stmt = select(TrainingRun).order_by(
            TrainingRun.created_at.desc()
        ).limit(1)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def get_by_new_version(self, version: str) -> TrainingRun | None:
        stmt = select(TrainingRun).where(TrainingRun.new_version == version)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    # ── dashboard queries (§9.1) ──

    async def recent_runs_touched_since(
        self, *, since: datetime, limit: int
    ) -> Sequence[TrainingRun]:
        """Runs whose `started_at`, `finished_at`, or `created_at` lands in
        `[since, now]`, eager-loading the triggering user.

        Used to project into unified `recentActivity` events
        (STARTED / SUCCEEDED / FAILED). The join makes actor projection cheap
        when the service loops over the rows."""
        stmt = (
            select(TrainingRun)
            .where(
                or_(
                    TrainingRun.created_at >= since,
                    TrainingRun.started_at >= since,
                    TrainingRun.finished_at >= since,
                )
            )
            .options(joinedload(TrainingRun.triggered_by_user))
            .order_by(TrainingRun.created_at.desc())
            .limit(max(0, limit))
        )
        result = await self.session.execute(stmt)
        return result.scalars().unique().all()
