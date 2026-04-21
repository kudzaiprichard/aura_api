from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from src.shared.database.engine import async_session


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Transactional session — auto-commits on success, rolls back on error."""
    async with async_session() as session:
        async with session.begin():
            yield session


async def get_db_no_transaction() -> AsyncGenerator[AsyncSession, None]:
    """Session without an outer transaction — caller manages commit/rollback.

    Note: this does NOT enforce read-only at the connection level. A repository
    write inside this scope will still flush to the DB; only the implicit
    commit-on-clean-exit is missing. Name reflects what it actually does.
    """
    async with async_session() as session:
        yield session
