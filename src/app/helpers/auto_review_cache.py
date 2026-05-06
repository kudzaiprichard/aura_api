"""Phase 12 — short-TTL LRU cache for auto-reviewer calls.

Caches only successful verdicts keyed on the SHA-256 of the normalised
``(sender, subject, body, model_name)`` tuple. Failures are never memoised so
a transient upstream error always re-calls the provider. Threadsafe because
``ReviewService._invoke_and_persist`` reads/writes the cache across different
event-loop tasks.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from src.shared.inference import AutoReviewSuccess


def _normalise(value: str) -> str:
    # Strip surrounding whitespace and collapse case so trivial variations
    # (trailing newline, shifted capitalisation) still hit the same row.
    return value.strip().lower()


def compute_cache_key(
    *, sender: str, subject: str, body: str, model_name: str
) -> str:
    payload = (
        f"{_normalise(sender)}\n"
        f"{_normalise(subject)}\n"
        f"{_normalise(body)}\n"
        f"{_normalise(model_name)}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class _Entry:
    success: AutoReviewSuccess
    expires_at: float


class AutoReviewCache:
    """Fixed-size LRU with per-entry TTL.

    The cache is intentionally tiny and short-lived — its job is to absorb the
    exact-duplicate traffic that happens when an analyst batch-triggers the
    same email body twice or when `/predict` is replayed in quick succession.
    It is NOT a correctness boundary: a miss always falls back to the
    provider, a hit returns the prior verdict verbatim.
    """

    def __init__(self, *, max_size: int, ttl_seconds: int) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._max_size = int(max_size)
        self._ttl_seconds = int(ttl_seconds)
        self._store: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @property
    def ttl_seconds(self) -> int:
        return self._ttl_seconds

    @property
    def max_size(self) -> int:
        return self._max_size

    def get(self, key: str) -> AutoReviewSuccess | None:
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.expires_at <= now:
                # Expired — drop the row so repeated lookups don't churn.
                self._store.pop(key, None)
                self._misses += 1
                return None
            self._store.move_to_end(key)
            self._hits += 1
            return entry.success

    def set(self, key: str, success: AutoReviewSuccess) -> None:
        now = time.monotonic()
        entry = _Entry(success=success, expires_at=now + self._ttl_seconds)
        with self._lock:
            self._store[key] = entry
            self._store.move_to_end(key)
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._store),
                "max_size": self._max_size,
                "ttl_seconds": self._ttl_seconds,
            }

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0
