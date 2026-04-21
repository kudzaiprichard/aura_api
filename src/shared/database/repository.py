from typing import TypeVar, Generic, Type, Sequence, Dict, Any, Optional, Tuple
from uuid import UUID
from sqlalchemy import select, func, asc, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select
from src.shared.database.base_model import BaseModel
from src.shared.exceptions.exceptions import InternalServerException
from src.shared.responses.api_response import ErrorDetail

T = TypeVar("T", bound=BaseModel)


class BaseRepository(Generic[T]):
    def __init__(self, model: Type[T], session: AsyncSession):
        self.model = model
        self.session = session

    # ──────────────────────────────────────────
    # Single record
    # ──────────────────────────────────────────

    async def get_by_id(self, id: UUID) -> T | None:
        return await self.session.get(self.model, id)

    async def get_one(self, **filters: Any) -> T | None:
        """Get a single record matching the given filters."""
        stmt = select(self.model).filter_by(**filters)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def exists(self, **filters: Any) -> bool:
        """Check if a record matching the given filters exists."""
        stmt = select(func.count()).select_from(self.model).filter_by(**filters)
        result = await self.session.execute(stmt)
        return result.scalar_one() > 0

    # ──────────────────────────────────────────
    # Multiple records
    # ──────────────────────────────────────────

    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        order_by: Optional[str] = None,
        descending: bool = False,
        **filters: Any,
    ) -> Sequence[T]:
        """Get records with optional filtering, ordering, and offset/limit."""
        stmt = self._apply_filters(select(self.model), **filters)
        stmt = self._apply_ordering(stmt, order_by, descending)
        stmt = stmt.offset(skip).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def paginate(
        self,
        page: int = 1,
        page_size: int = 20,
        order_by: Optional[str] = None,
        descending: bool = False,
        **filters: Any,
    ) -> Tuple[Sequence[T], int]:
        """
        Get a page of records with total count.

        Returns:
            (records, total_count)
        """
        base = self._apply_filters(select(self.model), **filters)

        # Total count
        count_stmt = select(func.count()).select_from(base.subquery())
        total = (await self.session.execute(count_stmt)).scalar_one()

        # Page of records
        stmt = self._apply_ordering(base, order_by, descending)
        offset = (page - 1) * page_size
        stmt = stmt.offset(offset).limit(page_size)
        result = await self.session.execute(stmt)
        records = result.scalars().all()

        return records, total

    async def count(self, **filters: Any) -> int:
        """Count records with optional filtering."""
        stmt = select(func.count()).select_from(self.model)
        if filters:
            stmt = stmt.filter_by(**filters)
        result = await self.session.execute(stmt)
        return result.scalar_one()

    # ──────────────────────────────────────────
    # Write operations
    # ──────────────────────────────────────────

    async def create(self, entity: T) -> T:
        self.session.add(entity)
        await self.session.flush()
        await self.session.refresh(entity)
        return entity

    async def create_many(
        self, entities: Sequence[T], refresh: bool = False
    ) -> Sequence[T]:
        """Bulk insert multiple entities. Set refresh=True to hydrate server-side defaults."""
        self.session.add_all(entities)
        await self.session.flush()
        if refresh:
            for entity in entities:
                await self.session.refresh(entity)
        return entities

    async def update(self, entity: T, data: Dict[str, Any]) -> T:
        for key, value in data.items():
            if not hasattr(entity, key):
                raise InternalServerException(
                    message=f"{self.model.__name__} has no attribute '{key}'",
                    error_detail=ErrorDetail(
                        title="Internal Server Error",
                        code="INVALID_UPDATE_FIELD",
                        status=500,
                        details=[f"{self.model.__name__} has no attribute '{key}'"],
                    ),
                )
            setattr(entity, key, value)
        await self.session.flush()
        await self.session.refresh(entity)
        return entity

    async def delete(self, entity: T) -> None:
        await self.session.delete(entity)
        await self.session.flush()

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    def _apply_filters(self, stmt: Select, **filters: Any) -> Select:
        if filters:
            stmt = stmt.filter_by(**filters)
        return stmt

    def _apply_ordering(
        self, stmt: Select, order_by: Optional[str], descending: bool
    ) -> Select:
        if order_by and hasattr(self.model, order_by):
            col = getattr(self.model, order_by)
            stmt = stmt.order_by(desc(col) if descending else asc(col))
        return stmt
