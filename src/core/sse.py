"""In-process Server-Sent Events plumbing.

Two pieces, both raw-ASGI by design (BaseHTTPMiddleware buffers response
bodies and breaks SSE — see `RequestLoggingMiddleware` for the same rule
applied to the existing logging middleware):

  * `SSEBroker` — topic-keyed pub/sub. Each subscribe call returns a fresh
    `asyncio.Queue` plus a cleanup handle. Publishes fan out to every live
    subscriber on the topic. A bounded per-topic ring buffer holds recent
    events so a reconnecting client can replay from `Last-Event-ID`.

  * `sse_response` — a coroutine that drives one HTTP response from a topic
    subscription. Drives the ASGI `send` directly — never goes through any
    Starlette response object that would buffer the stream.

Phase 7 ships this for the reference echo endpoint; Phases 8 and 10 will
build on it for training-run and benchmark progress.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable

from starlette.types import Receive, Scope, Send


log = logging.getLogger("aura.sse")


# Sent to clients whenever no real event has fired in `heartbeat_seconds`.
# The colon-prefixed line is an SSE "comment" per the spec — it keeps the
# connection alive without firing an event handler in the browser.
_HEARTBEAT_FRAME = b": keepalive\n\n"


@dataclass(frozen=True)
class SSEEvent:
    """A single fan-out payload.

    `id` is opaque to the broker and is what `Last-Event-ID` replay matches
    against. Callers typically pass a monotonic counter or a UUID — anything
    that is unique within the topic's replay window.
    """

    data: Any
    event: str | None = None
    id: str | None = None
    retry_ms: int | None = None

    def encode(self) -> bytes:
        """Render the event as wire bytes per the EventSource spec.

        Multi-line data is split across `data:` lines. Non-string payloads
        are JSON-encoded so the field stays valid SSE text.
        """
        lines: list[str] = []
        if self.event is not None:
            lines.append(f"event: {self.event}")
        if self.id is not None:
            lines.append(f"id: {self.id}")
        if self.retry_ms is not None:
            lines.append(f"retry: {int(self.retry_ms)}")

        if isinstance(self.data, (str, bytes)):
            text = self.data.decode("utf-8") if isinstance(self.data, bytes) else self.data
        else:
            text = json.dumps(self.data, default=str, separators=(",", ":"))

        for line in text.splitlines() or [""]:
            lines.append(f"data: {line}")
        lines.append("")
        lines.append("")
        return "\n".join(lines).encode("utf-8")


class _Subscription:
    """One live subscriber on one topic. Holds the queue plus the metadata
    the broker needs to address it during fan-out and cleanup."""

    __slots__ = ("topic", "queue", "_dropped")

    def __init__(self, topic: str, queue_max: int):
        self.topic = topic
        self.queue: asyncio.Queue[SSEEvent] = asyncio.Queue(maxsize=queue_max)
        self._dropped = 0

    def push(self, event: SSEEvent) -> None:
        """Best-effort enqueue. If the subscriber is wedged (full queue),
        drop the oldest event so the broker is never blocked by one slow
        consumer. The dropped count is logged when the connection ends."""
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._dropped += 1
            try:
                self.queue.put_nowait(event)
            except asyncio.QueueFull:
                # Another producer beat us to it; the event is gone but the
                # subscriber is still alive — better than blocking fan-out.
                self._dropped += 1

    @property
    def dropped(self) -> int:
        return self._dropped


class SSEBroker:
    """Topic-keyed pub/sub with bounded subscriber queues and replay.

    Lifetime is tied to the application: constructed in the lifespan,
    drained on shutdown. Methods are safe to call from request handlers
    without external locking — `set` / `dict` mutations under the GIL plus
    `asyncio.Queue` for hand-off do the necessary synchronisation.
    """

    def __init__(
        self,
        *,
        subscriber_queue_max: int = 256,
        replay_window_size: int = 128,
    ):
        if subscriber_queue_max < 1:
            raise ValueError("subscriber_queue_max must be >= 1")
        if replay_window_size < 0:
            raise ValueError("replay_window_size must be >= 0")
        self._subs: dict[str, set[_Subscription]] = {}
        self._replay: dict[str, deque[SSEEvent]] = {}
        self._subscriber_queue_max = subscriber_queue_max
        self._replay_window_size = replay_window_size
        self._stopped = False

    # ── publish ──

    def publish(self, topic: str, event: SSEEvent) -> int:
        """Fan-out + ring-buffer record in one call. Returns the number of
        live subscribers the event was pushed to (zero is a normal,
        non-error state — events publish even when nobody is listening)."""
        if self._stopped:
            return 0
        self._record_replay(topic, event)
        subs = self._subs.get(topic)
        if not subs:
            return 0
        for sub in tuple(subs):
            sub.push(event)
        return len(subs)

    def _record_replay(self, topic: str, event: SSEEvent) -> None:
        if self._replay_window_size == 0 or event.id is None:
            return
        ring = self._replay.get(topic)
        if ring is None:
            ring = deque(maxlen=self._replay_window_size)
            self._replay[topic] = ring
        ring.append(event)

    # ── subscribe ──

    def subscribe(
        self, topic: str, *, last_event_id: str | None = None
    ) -> tuple[_Subscription, list[SSEEvent], Callable[[], None]]:
        """Register a subscriber and return `(subscription, replay, cleanup)`.

        `replay` is the ordered list of events still in the ring after
        `last_event_id` (empty when the id is unknown or replay is off).
        `cleanup` is idempotent; the consumer must call it exactly once on
        disconnect / cancel — `sse_response` does this in a `finally`."""
        if self._stopped:
            raise RuntimeError("SSEBroker is stopped; cannot subscribe")
        sub = _Subscription(topic, self._subscriber_queue_max)
        self._subs.setdefault(topic, set()).add(sub)

        replay: list[SSEEvent] = []
        if last_event_id is not None and self._replay_window_size > 0:
            ring = self._replay.get(topic) or ()
            seen = False
            for ev in ring:
                if seen:
                    replay.append(ev)
                elif ev.id == last_event_id:
                    seen = True

        cleaned = False

        def cleanup() -> None:
            nonlocal cleaned
            if cleaned:
                return
            cleaned = True
            bucket = self._subs.get(topic)
            if bucket is None:
                return
            bucket.discard(sub)
            if not bucket:
                self._subs.pop(topic, None)

        return sub, replay, cleanup

    # ── lifecycle ──

    def subscriber_count(self, topic: str | None = None) -> int:
        if topic is None:
            return sum(len(b) for b in self._subs.values())
        return len(self._subs.get(topic) or ())

    async def stop(self) -> None:
        """Drop every live subscriber. Each in-flight `sse_response` notices
        on its next queue read or heartbeat tick and returns cleanly."""
        self._stopped = True
        # Snapshot first — `cleanup()` mutates `_subs` and we are iterating it.
        all_subs = [s for bucket in self._subs.values() for s in bucket]
        self._subs.clear()
        self._replay.clear()
        # Wake every waiting reader so the response coroutine can exit. We
        # send a sentinel `None` via cancellation: closing the queue isn't
        # supported in stdlib, so we cancel any pending getters by enqueuing
        # a synthetic terminator the response loop treats as end-of-stream.
        for sub in all_subs:
            try:
                sub.queue.put_nowait(_STOP)
            except asyncio.QueueFull:
                # Subscriber is wedged; the response coroutine will fall out
                # on the next heartbeat tick when it probes for disconnect.
                pass


# Sentinel pushed into subscriber queues at shutdown so blocked readers wake.
_STOP = SSEEvent(data="__sse_stop__", event="__sse_stop__")


# ── ASGI response ──

async def sse_response(
    scope: Scope,
    receive: Receive,
    send: Send,
    *,
    broker: SSEBroker,
    topic: str,
    heartbeat_seconds: float,
    initial: AsyncIterator[SSEEvent] | None = None,
    on_disconnect: Callable[[], Awaitable[None]] | None = None,
) -> None:
    """Drive one SSE response from a broker subscription.

    Strict raw-ASGI: every byte goes through `send` directly, no Starlette
    `StreamingResponse` (it composes with `BaseHTTPMiddleware` which would
    re-buffer the body). Heartbeats double as the disconnect probe — the
    `receive` channel is polled with a tight timeout each tick; a
    `http.disconnect` message terminates the loop within one heartbeat.

    `initial` lets the caller push a snapshot or a Last-Event-ID replay
    before the live tail begins; it is fully drained before the broker
    queue is read.
    """
    if scope["type"] != "http":
        raise RuntimeError("sse_response only supports http scope")

    last_event_id = _last_event_id_from_scope(scope)
    sub, replay, cleanup = broker.subscribe(topic, last_event_id=last_event_id)

    await send({
        "type": "http.response.start",
        "status": 200,
        "headers": [
            (b"content-type", b"text/event-stream; charset=utf-8"),
            (b"cache-control", b"no-cache, no-transform"),
            (b"connection", b"keep-alive"),
            # Disable proxy buffering — same reason BaseHTTPMiddleware is banned.
            (b"x-accel-buffering", b"no"),
        ],
    })

    disconnected = False

    try:
        # 1. Replay anything past last_event_id, then any initial snapshot.
        for ev in replay:
            await send({"type": "http.response.body", "body": ev.encode(), "more_body": True})
        if initial is not None:
            async for ev in initial:
                await send({"type": "http.response.body", "body": ev.encode(), "more_body": True})

        # 2. Live tail. Each loop iteration either delivers an event, sends
        #    a heartbeat, or notices the client has gone away.
        while True:
            try:
                event = await asyncio.wait_for(sub.queue.get(), timeout=heartbeat_seconds)
            except asyncio.TimeoutError:
                event = None

            if await _client_disconnected(receive):
                disconnected = True
                break

            if event is _STOP:
                break

            if event is None:
                await send({"type": "http.response.body", "body": _HEARTBEAT_FRAME, "more_body": True})
                continue

            await send({"type": "http.response.body", "body": event.encode(), "more_body": True})

    except (asyncio.CancelledError, ConnectionError, OSError):
        disconnected = True
        raise
    finally:
        cleanup()
        if sub.dropped:
            log.warning(
                "sse subscriber dropped %d event(s) topic=%s (slow consumer)",
                sub.dropped, topic,
            )
        if on_disconnect is not None:
            try:
                await on_disconnect()
            except Exception:  # noqa: BLE001
                log.exception("sse on_disconnect callback failed topic=%s", topic)
        if not disconnected:
            try:
                await send({"type": "http.response.body", "body": b"", "more_body": False})
            except Exception:  # noqa: BLE001
                pass


def _last_event_id_from_scope(scope: Scope) -> str | None:
    """EventSource clients ship the resume token as the `Last-Event-ID`
    header (case-insensitive). Browsers also accept it as the `lastEventId`
    query string for fallbacks; we only honour the header per spec."""
    for k, v in scope.get("headers", ()):
        if k == b"last-event-id":
            try:
                return v.decode("latin-1")
            except UnicodeDecodeError:
                return None
    return None


async def _client_disconnected(receive: Receive) -> bool:
    """Non-blocking probe: pull any pending ASGI messages off the receive
    queue and report whether the client has hung up. Returns False quickly
    when nothing is waiting — we're called once per heartbeat tick."""
    try:
        message = await asyncio.wait_for(receive(), timeout=0)
    except asyncio.TimeoutError:
        return False
    if message["type"] == "http.disconnect":
        return True
    return False


# Convenience for callers that want a monotonic event id without booking
# a UUID per event. Wraps `time.monotonic_ns()` so ids stay sortable across
# a process lifetime even when wall-clock jumps.
def monotonic_event_id() -> str:
    return f"{time.monotonic_ns():x}"


class SSEResponse:
    """Starlette/FastAPI-compatible response that drives a raw-ASGI SSE
    stream from a broker subscription.

    Looks enough like a `Response` (no `__call__` indirection through any
    BaseHTTPMiddleware-style buffering) for FastAPI to return it directly
    from an endpoint. The work happens in `__call__(scope, receive, send)`,
    which is exactly the point where Starlette would otherwise instantiate
    a `StreamingResponse` that we cannot use here (it composes with
    BaseHTTPMiddleware and breaks SSE).
    """

    media_type = "text/event-stream"

    def __init__(
        self,
        *,
        broker: SSEBroker,
        topic: str,
        heartbeat_seconds: float,
        initial: AsyncIterator[SSEEvent] | None = None,
        on_disconnect: Callable[[], Awaitable[None]] | None = None,
    ):
        self.broker = broker
        self.topic = topic
        self.heartbeat_seconds = heartbeat_seconds
        self.initial = initial
        self.on_disconnect = on_disconnect

    async def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        await sse_response(
            scope,
            receive,
            send,
            broker=self.broker,
            topic=self.topic,
            heartbeat_seconds=self.heartbeat_seconds,
            initial=self.initial,
            on_disconnect=self.on_disconnect,
        )
