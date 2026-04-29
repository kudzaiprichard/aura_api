"""In-process asyncio job runner used by long-lived background work
(training runs, benchmark executions, etc.).

Modelled on the `start_token_cleanup` pattern already in the codebase:
submitted coroutines run as `asyncio.Task`s, the runner keeps strong
references to prevent GC, exceptions are logged rather than silently
swallowed, and shutdown cancels + awaits every outstanding task.
"""
import asyncio
import logging
from typing import Awaitable, Coroutine

logger = logging.getLogger(__name__)


class BackgroundJobRunner:
    def __init__(self) -> None:
        self._tasks: set[asyncio.Task] = set()
        self._stopped = False

    def submit(self, coro: Coroutine | Awaitable, *, name: str | None = None) -> asyncio.Task:
        if self._stopped:
            raise RuntimeError("BackgroundJobRunner is stopped; cannot submit new jobs")
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._on_done)
        return task

    def _on_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                "Background job %r failed: %s",
                task.get_name(),
                exc,
                exc_info=exc,
            )

    async def stop(self) -> None:
        self._stopped = True
        if not self._tasks:
            return
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
