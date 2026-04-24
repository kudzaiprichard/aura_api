import asyncio
import logging

from src.shared.database import async_session
from src.app.repositories.token_repository import TokenRepository
from src.configs import security

logger = logging.getLogger(__name__)


async def _run_once() -> None:
    async with async_session() as session:
        async with session.begin():
            repo = TokenRepository(session)
            count = await repo.cleanup_expired()
            if count:
                logger.info("Token cleanup: removed %d expired token(s)", count)


async def start_token_cleanup() -> None:
    """Background loop that purges expired tokens on a fixed interval.

    Runs once immediately on startup so a backlog from a previous downtime
    is cleared without waiting a full interval.
    """
    interval = security.token_cleanup_interval_seconds
    while True:
        try:
            await _run_once()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Token cleanup failed: %s", e)
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
