from uuid import UUID
from datetime import datetime, timezone
from typing import Sequence

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from src.shared.database import BaseRepository
from src.app.models.token import Token
from src.app.models.enums import TokenType


class TokenRepository(BaseRepository[Token]):
    def __init__(self, session: AsyncSession):
        super().__init__(Token, session)

    async def get_by_hash(self, token_hash: str) -> Token | None:
        stmt = select(Token).where(Token.token_hash == token_hash)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def is_hash_valid(self, token_hash: str) -> bool:
        record = await self.get_by_hash(token_hash)
        return record is not None and record.is_valid

    async def revoke_by_hash(self, token_hash: str) -> None:
        stmt = (
            update(Token)
            .where(Token.token_hash == token_hash)
            .values(is_revoked=True)
        )
        await self.session.execute(stmt)

    async def revoke_all_user_tokens(self, user_id: UUID) -> None:
        stmt = (
            update(Token)
            .where(Token.user_id == user_id, Token.is_revoked.is_(False))
            .values(is_revoked=True)
        )
        await self.session.execute(stmt)

    async def revoke_user_tokens_by_type(self, user_id: UUID, token_type: TokenType) -> None:
        stmt = (
            update(Token)
            .where(
                Token.user_id == user_id,
                Token.token_type == token_type,
                Token.is_revoked.is_(False),
            )
            .values(is_revoked=True)
        )
        await self.session.execute(stmt)

    async def cleanup_expired(self) -> int:
        now = datetime.now(timezone.utc)
        del_stmt = delete(Token).where(Token.expires_at < now)
        result = await self.session.execute(del_stmt)
        await self.session.flush()
        return result.rowcount or 0

    # ── dashboard queries (§9.1) ──

    async def recent_logins_since(
        self, *, since: datetime, limit: int
    ) -> Sequence[Token]:
        """Access-token rows issued since `since`, newest first, eager-loading
        the owning user.

        The auth service issues one ACCESS + one REFRESH token per login /
        refresh flow; restricting to ACCESS collapses the pair into a single
        LOGIN event in the dashboard feed."""
        stmt = (
            select(Token)
            .where(
                Token.token_type == TokenType.ACCESS,
                Token.created_at >= since,
            )
            .options(joinedload(Token.user))
            .order_by(Token.created_at.desc())
            .limit(max(0, limit))
        )
        result = await self.session.execute(stmt)
        return result.scalars().unique().all()
