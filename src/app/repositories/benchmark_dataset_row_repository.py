from typing import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.models.benchmark_dataset_row import BenchmarkDatasetRow
from src.shared.database import BaseRepository


class BenchmarkDatasetRowRepository(BaseRepository[BenchmarkDatasetRow]):
    def __init__(self, session: AsyncSession):
        super().__init__(BenchmarkDatasetRow, session)

    async def bulk_insert(
        self, entities: Sequence[BenchmarkDatasetRow]
    ) -> Sequence[BenchmarkDatasetRow]:
        """Add and flush a batch of dataset rows in one statement."""
        if not entities:
            return entities
        self.session.add_all(entities)
        await self.session.flush()
        return entities

    async def list_by_dataset(
        self, dataset_id: UUID
    ) -> Sequence[BenchmarkDatasetRow]:
        """Return every row for a dataset ordered by `created_at ASC` so the
        benchmark run feeds the detector in insertion order (keeps labels and
        inputs aligned deterministically)."""
        stmt = (
            select(BenchmarkDatasetRow)
            .where(BenchmarkDatasetRow.benchmark_dataset_id == dataset_id)
            .order_by(BenchmarkDatasetRow.created_at.asc())
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()
