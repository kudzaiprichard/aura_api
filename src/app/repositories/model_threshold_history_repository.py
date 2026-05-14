from datetime import datetime
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.models.model_threshold_history import ModelThresholdHistory
from src.shared.database import BaseRepository


class ModelThresholdHistoryRepository(BaseRepository[ModelThresholdHistory]):
    def __init__(self, session: AsyncSession):
        super().__init__(ModelThresholdHistory, session)

    async def current_for_version(
        self, version: str
    ) -> ModelThresholdHistory | None:
        """The one open row per version (effective_to IS NULL). The partial
        unique index `ux_model_threshold_history_current_per_version` makes
        this a single-row lookup.
        """
        stmt = select(ModelThresholdHistory).where(
            ModelThresholdHistory.version == version,
            ModelThresholdHistory.effective_to.is_(None),
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def close_current(
        self, *, version: str, closed_at: datetime
    ) -> None:
        """Close the current row for `version` (if any) by stamping
        `effective_to`. The partial unique index enforces that there is at
        most one open row per version, so a blanket UPDATE is safe.
        """
        stmt = (
            update(ModelThresholdHistory)
            .where(
                ModelThresholdHistory.version == version,
                ModelThresholdHistory.effective_to.is_(None),
            )
            .values(effective_to=closed_at)
        )
        await self.session.execute(stmt)

    async def open(
        self,
        *,
        version: str,
        decision_threshold: float,
        review_low_threshold: float | None,
        review_high_threshold: float | None,
        set_by: UUID | None,
        effective_from: datetime,
    ) -> ModelThresholdHistory:
        row = ModelThresholdHistory(
            version=version,
            decision_threshold=decision_threshold,
            review_low_threshold=review_low_threshold,
            review_high_threshold=review_high_threshold,
            set_by=set_by,
            effective_from=effective_from,
            effective_to=None,
        )
        return await self.create(row)
