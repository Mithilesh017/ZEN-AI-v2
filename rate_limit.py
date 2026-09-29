"""
rate_limit.py — Per-user request throttling
===========================================

A chat request costs real money (Groq inference, HuggingFace embeddings,
Pinecone queries), so a single account hammering /api/chat — a runaway script,
a stuck retry loop, or plain abuse — can drain the quota for everyone.

The limiter is in-memory and per process: with several workers the effective
allowance is multiplied by the worker count. That is fine for its purpose,
which is to stop runaway usage rather than to meter it exactly. Swap in Redis
if the limit ever needs to be shared across processes.
"""

import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    """
    Thread-safe sliding-window limiter.

    `rules` is a list of (max_calls, window_seconds); a call is allowed only if
    every rule still has room, which lets a short burst rule sit alongside a
    longer sustained one.
    """

    def __init__(self, rules: list[tuple[int, int]]):
        if not rules:
            raise ValueError("at least one rule is required")
        if any(calls < 1 or window <= 0 for calls, window in rules):
            raise ValueError("rules must have a positive call count and window")

        self._rules = sorted(rules, key=lambda rule: rule[1])
        self._longest_window = max(window for _, window in rules)
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._next_sweep = 0.0

    def check(self, key: str, now: float | None = None) -> int | None:
        """
        Record a call for `key`.

        Returns None when the call is allowed, or the number of seconds to wait
        before retrying when a rule is exhausted (suitable for `Retry-After`).
        A rejected call is not recorded, so being throttled cannot extend the
        penalty indefinitely.
        """
        now = time.monotonic() if now is None else now

        with self._lock:
            if now >= self._next_sweep:
                self._sweep(now)

            hits = self._hits[key]
            cutoff = now - self._longest_window
            while hits and hits[0] <= cutoff:
                hits.popleft()

            for max_calls, window in self._rules:
                window_start = now - window
                in_window = [hit for hit in hits if hit > window_start]
                if len(in_window) >= max_calls:
                    # Room frees up once the oldest call in this window ages out.
                    return max(1, int(in_window[0] + window - now) + 1)

            hits.append(now)
            return None

    def reset(self, key: str) -> None:
        """Forget a key's history (used by tests)."""
        with self._lock:
            self._hits.pop(key, None)

    def _sweep(self, now: float) -> None:
        """Drop keys with no recent calls so idle users don't accumulate."""
        cutoff = now - self._longest_window
        for key in [k for k, hits in self._hits.items() if not hits or hits[-1] <= cutoff]:
            del self._hits[key]
        self._next_sweep = now + self._longest_window
