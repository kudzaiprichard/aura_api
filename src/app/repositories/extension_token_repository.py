from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from src.shared.database import BaseRepository
from src.app.models.extension_token import ExtensionToken


class ExtensionTokenRepository(BaseRepository[ExtensionToken]):
    def __init__(self, session: AsyncSession):
        super().__init__(ExtensionToken, session)

    async def get_by_hash(self, token_hash: str) -> ExtensionToken | None:
        stmt = select(ExtensionToken).where(
            ExtensionToken.token_hash == token_hash
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def get_by_hash_with_install(
        self, token_hash: str
    ) -> ExtensionToken | None:
        """Same as `get_by_hash` but eager-loads the owning install — used by
        `require_install` so a single round-trip both validates the token and
        returns the install to the route handler."""
        stmt = (
            select(ExtensionToken)
            .where(ExtensionToken.token_hash == token_hash)
            .options(joinedload(ExtensionToken.install))
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def revoke_by_hash(
        self, token_hash: str, *, reason: str | None = None
    ) -> int:
        stmt = (
            update(ExtensionToken)
            .where(
                ExtensionToken.token_hash == token_hash,
                ExtensionToken.is_revoked.is_(False),
            )
            .values(is_revoked=True, revoked_reason=reason)
        )
        result = await self.session.execute(stmt)
        return result.rowcount or 0

    async def revoke_all_for_install(
        self, install_id: UUID, *, reason: str | None = None
    ) -> int:
        stmt = (
            update(ExtensionToken)
            .where(
                ExtensionToken.install_id == install_id,
                ExtensionToken.is_revoked.is_(False),
            )
            .values(is_revoked=True, revoked_reason=reason)
        )
        result = await self.session.execute(stmt)
        return result.rowcount or 0

    async def count_active_for_install(self, install_id: UUID) -> int:
        now = datetime.now(timezone.utc)
        stmt = (
            select(func.count())
            .select_from(ExtensionToken)
            .where(
                ExtensionToken.install_id == install_id,
                ExtensionToken.is_revoked.is_(False),
                ExtensionToken.expires_at > now,
            )
        )
        result = await self.session.execute(stmt)
        return int(result.scalar_one() or 0)

    async def cleanup_expired(self) -> int:
        now = datetime.now(timezone.utc)
        stmt = delete(ExtensionToken).where(ExtensionToken.expires_at < now)
        result = await self.session.execute(stmt)
        await self.session.flush()
        return result.rowcount or 0
