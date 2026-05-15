from typing import Sequence, Tuple

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.models.benchmark_dataset import BenchmarkDataset
from src.shared.database import BaseRepository


class BenchmarkDatasetRepository(BaseRepository[BenchmarkDataset]):
    def __init__(self, session: AsyncSession):
        super().__init__(BenchmarkDataset, session)

    async def paginate_filtered(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
    ) -> Tuple[Sequence[BenchmarkDataset], int]:
        stmt = select(BenchmarkDataset)
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await self.session.execute(count_stmt)).scalar_one()

        offset = (page - 1) * page_size
        stmt = (
            stmt.order_by(BenchmarkDataset.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all(), total

    async def get_by_name(self, name: str) -> BenchmarkDataset | None:
        stmt = select(BenchmarkDataset).where(BenchmarkDataset.name == name)
        return (await self.session.execute(stmt)).scalar_one_or_none()
