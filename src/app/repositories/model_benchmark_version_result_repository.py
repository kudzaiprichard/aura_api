from typing import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.models.model_benchmark_version_result import (
    ModelBenchmarkVersionResult,
)
from src.shared.database import BaseRepository


class ModelBenchmarkVersionResultRepository(
    BaseRepository[ModelBenchmarkVersionResult]
):
    def __init__(self, session: AsyncSession):
        super().__init__(ModelBenchmarkVersionResult, session)

    async def list_by_benchmark(
        self, benchmark_id: UUID
    ) -> Sequence[ModelBenchmarkVersionResult]:
        """Per-version rows in insertion order — matches the sequence the
        service wrote them to the SSE stream, so the side-by-side payload
        and the event log align one-to-one."""
        stmt = (
            select(ModelBenchmarkVersionResult)
            .where(ModelBenchmarkVersionResult.model_benchmark_id == benchmark_id)
            .order_by(ModelBenchmarkVersionResult.created_at.asc())
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()
