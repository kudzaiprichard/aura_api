from typing import Sequence
from uuid import UUID
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.shared.database import BaseRepository
from src.app.models.user import User
from src.app.models.enums import Role


class UserRepository(BaseRepository[User]):
    def __init__(self, session: AsyncSession):
        super().__init__(User, session)

    async def get_by_email(self, email: str) -> User | None:
        stmt = select(User).where(User.email == email)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def get_by_username(self, username: str) -> User | None:
        stmt = select(User).where(User.username == username)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def email_exists(self, email: str) -> bool:
        return await self.exists(email=email)

    async def username_exists(self, username: str) -> bool:
        return await self.exists(username=username)

    async def count_active_admins(self, exclude_user_id: UUID | None = None) -> int:
        stmt = select(func.count()).select_from(User).where(
            User.role == Role.ADMIN, User.is_active.is_(True)
        )
        if exclude_user_id is not None:
            stmt = stmt.where(User.id != exclude_user_id)
        result = await self.session.execute(stmt)
        return result.scalar_one()

    # ── dashboard queries (§9.1) ──

    async def list_active_analysts(self) -> Sequence[User]:
        """Every active IT_ANALYST user, sorted by username for stable
        per-analyst row ordering in the admin workload view."""
        stmt = (
            select(User)
            .where(User.role == Role.IT_ANALYST, User.is_active.is_(True))
            .order_by(User.username.asc())
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()
