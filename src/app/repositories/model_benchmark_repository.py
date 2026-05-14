from typing import Sequence, Tuple
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.models.enums import BenchmarkStatus
from src.app.models.model_benchmark import ModelBenchmark
from src.shared.database import BaseRepository


class ModelBenchmarkRepository(BaseRepository[ModelBenchmark]):
    def __init__(self, session: AsyncSession):
        super().__init__(ModelBenchmark, session)

    async def paginate_filtered(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        status: BenchmarkStatus | None = None,
        dataset_id: UUID | None = None,
    ) -> Tuple[Sequence[ModelBenchmark], int]:
        stmt = select(ModelBenchmark)
        if status is not None:
            stmt = stmt.where(ModelBenchmark.status == status)
        if dataset_id is not None:
            stmt = stmt.where(ModelBenchmark.benchmark_dataset_id == dataset_id)

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await self.session.execute(count_stmt)).scalar_one()

        offset = (page - 1) * page_size
        stmt = (
            stmt.order_by(ModelBenchmark.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all(), total

    async def count_by_dataset(self, dataset_id: UUID) -> int:
        stmt = (
            select(func.count())
            .select_from(ModelBenchmark)
            .where(ModelBenchmark.benchmark_dataset_id == dataset_id)
        )
        return (await self.session.execute(stmt)).scalar_one()
