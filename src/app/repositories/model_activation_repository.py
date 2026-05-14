from datetime import datetime
from typing import Any, Sequence, Tuple
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from src.app.models.enums import ModelActivationKind
from src.app.models.model_activation import ModelActivation
from src.shared.database import BaseRepository


class ModelActivationRepository(BaseRepository[ModelActivation]):
    def __init__(self, session: AsyncSession):
        super().__init__(ModelActivation, session)

    async def record(
        self,
        *,
        kind: ModelActivationKind,
        version: str,
        previous_version: str | None,
        actor_id: UUID | None,
        reason: str | None = None,
        metrics_snapshot: dict[str, Any] | None = None,
    ) -> ModelActivation:
        row = ModelActivation(
            kind=kind,
            version=version,
            previous_version=previous_version,
            actor_id=actor_id,
            reason=reason,
            metrics_snapshot=metrics_snapshot,
        )
        return await self.create(row)

    async def latest_activate_or_promote(
        self, *, exclude_version: str | None = None
    ) -> ModelActivation | None:
        """Most recent ACTIVATE/PROMOTE — used by rollback to find the prior
        active version when the registry does not carry a direct pointer.
        """
        stmt = select(ModelActivation).where(
            ModelActivation.kind.in_(
                [ModelActivationKind.ACTIVATE, ModelActivationKind.PROMOTE]
            )
        )
        if exclude_version is not None:
            stmt = stmt.where(ModelActivation.version != exclude_version)
        stmt = stmt.order_by(ModelActivation.created_at.desc()).limit(1)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def paginate_for_version(
        self, *, version: str, page: int = 1, page_size: int = 20
    ) -> Tuple[Sequence[ModelActivation], int]:
        stmt = select(ModelActivation).where(ModelActivation.version == version)
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await self.session.execute(count_stmt)).scalar_one()
        offset = (page - 1) * page_size
        stmt = (
            stmt.order_by(ModelActivation.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        rows = (await self.session.execute(stmt)).scalars().all()
        return rows, total

    # ── dashboard queries (§9.1) ──

    async def recent_since(
        self, *, since: datetime, limit: int
    ) -> Sequence[ModelActivation]:
        """Activation log entries from `[since, now]`, newest first, with the
        actor joined for cheap attribution."""
        stmt = (
            select(ModelActivation)
            .where(ModelActivation.created_at >= since)
            .options(joinedload(ModelActivation.actor))
            .order_by(ModelActivation.created_at.desc())
            .limit(max(0, limit))
        )
        result = await self.session.execute(stmt)
        return result.scalars().unique().all()

    async def latest_for_version(
        self, *, version: str
    ) -> ModelActivation | None:
        """Most recent activation entry for `version` — the source of truth
        for `activeModel.activatedAt`/`promoted`."""
        stmt = (
            select(ModelActivation)
            .where(ModelActivation.version == version)
            .order_by(ModelActivation.created_at.desc())
            .limit(1)
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()
