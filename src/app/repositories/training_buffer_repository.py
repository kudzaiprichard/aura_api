from datetime import datetime
from typing import Iterable, Sequence, Tuple
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.models.enums import TrainingBufferSource
from src.app.models.training_buffer_item import TrainingBufferItem
from src.shared.database import BaseRepository


class TrainingBufferRepository(BaseRepository[TrainingBufferItem]):
    def __init__(self, session: AsyncSession):
        super().__init__(TrainingBufferItem, session)

    async def paginate_filtered(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        label: int | None = None,
        source: TrainingBufferSource | None = None,
        category: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> Tuple[Sequence[TrainingBufferItem], int]:
        stmt = select(TrainingBufferItem)
        if label is not None:
            stmt = stmt.where(TrainingBufferItem.label == label)
        if source is not None:
            stmt = stmt.where(TrainingBufferItem.source == source)
        if category is not None:
            stmt = stmt.where(TrainingBufferItem.category == category)
        if date_from is not None:
            stmt = stmt.where(TrainingBufferItem.created_at >= date_from)
        if date_to is not None:
            stmt = stmt.where(TrainingBufferItem.created_at <= date_to)

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await self.session.execute(count_stmt)).scalar_one()

        offset = (page - 1) * page_size
        stmt = (
            stmt.order_by(TrainingBufferItem.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all(), total

    async def exists_by_content_sha256(self, content_sha256: str) -> bool:
        stmt = (
            select(func.count())
            .select_from(TrainingBufferItem)
            .where(TrainingBufferItem.content_sha256 == content_sha256)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one() > 0

    async def exists_by_source_prediction_event_id(
        self, prediction_event_id: UUID
    ) -> bool:
        stmt = (
            select(func.count())
            .select_from(TrainingBufferItem)
            .where(
                TrainingBufferItem.source_prediction_event_id
                == prediction_event_id
            )
        )
        result = await self.session.execute(stmt)
        return result.scalar_one() > 0

    async def count_by_label(self) -> dict[int, int]:
        """Return a `{label: count}` map covering every label seen in the
        buffer. Labels with zero rows are absent — callers fill in zeros for
        any class they expect (0 / 1 for the binary phishing task)."""
        stmt = (
            select(
                TrainingBufferItem.label,
                func.count(TrainingBufferItem.id),
            )
            .group_by(TrainingBufferItem.label)
        )
        result = await self.session.execute(stmt)
        return {int(label): int(count) for label, count in result.all()}

    async def existing_content_sha256s(
        self, sha256s: Iterable[str]
    ) -> set[str]:
        """Return the subset of `sha256s` already present in the buffer.

        Used by the CSV import path to filter duplicates in one round-trip
        instead of N exists() probes."""
        sha_list = list({s for s in sha256s if s})
        if not sha_list:
            return set()
        stmt = select(TrainingBufferItem.content_sha256).where(
            TrainingBufferItem.content_sha256.in_(sha_list)
        )
        result = await self.session.execute(stmt)
        return {row[0] for row in result.all()}

    async def bulk_insert(
        self, entities: Sequence[TrainingBufferItem]
    ) -> Sequence[TrainingBufferItem]:
        """Add and flush a batch of buffer rows in one statement."""
        if not entities:
            return entities
        self.session.add_all(entities)
        await self.session.flush()
        return entities

    # ── dashboard queries (§9.1) ──

    async def count_since(self, since: datetime) -> int:
        """Count buffer rows with `created_at >= since` — drives the
        `recentAdditions24h` dashboard field."""
        stmt = (
            select(func.count())
            .select_from(TrainingBufferItem)
            .where(TrainingBufferItem.created_at >= since)
        )
        return int((await self.session.execute(stmt)).scalar_one())
