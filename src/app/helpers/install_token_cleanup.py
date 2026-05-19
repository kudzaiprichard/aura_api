import asyncio
import logging

from src.shared.database import async_session
from src.app.repositories.extension_token_repository import (
    ExtensionTokenRepository,
)
from src.configs import security

logger = logging.getLogger(__name__)


async def _run_once() -> None:
    async with async_session() as session:
        async with session.begin():
            repo = ExtensionTokenRepository(session)
            count = await repo.cleanup_expired()
            if count:
                logger.info(
                    "Extension token cleanup: removed %d expired token(s)",
                    count,
                )


async def start_install_token_cleanup() -> None:
    """Background loop that purges expired extension tokens on a fixed interval.

    Mirrors `start_token_cleanup` (dashboard JWTs) — runs once immediately so
    a backlog from prior downtime is cleared without waiting a full interval,
    then sleeps for `security.extension_token_cleanup_interval_seconds`.
    """
    interval = security.extension_token_cleanup_interval_seconds
    while True:
        try:
            await _run_once()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Extension token cleanup failed: %s", e)
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
