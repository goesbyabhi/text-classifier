"""
Per-IP rate limiting in a Durable Object.

Why a DO and not KV or the platform `ratelimit` binding: KV is edge-cached and
eventually consistent, so a counter written this second is invisible to the
next request; Cloudflare's native ratelimit binding registered but never
decremented in testing. A DO is single-threaded and strongly consistent,
which is exactly what a counter needs.

The counting logic is plain code kept importable without the Workers runtime
so pytest can exercise it; the RateLimiter DurableObject is a thin wrapper and
is only defined when the runtime SDK is importable (i.e. inside the Worker).
One DO instance per (tier, ip): idFromName("fast:1.2.3.4").
"""

from __future__ import annotations

import json
import time


class RateLimiterLogic:
    """Windowed counters over a storage backend that has .get/.put.

    Two windows per instance: a minute and a day. The stored record is the
    epoch it started on and the count since; a record from a previous window
    is discarded and counting restarts.
    """

    def __init__(self, storage, minute_ms=60_000, day_ms=86_400_000):
        self.storage = storage
        self.minute_ms = minute_ms
        self.day_ms = day_ms

    async def check(self, limit, daily, cost=1):
        now_ms = int(time.time() * 1000)

        minute = await self._window("m", limit, cost, self.minute_ms, now_ms)
        if minute["limited"]:
            return {**minute, "daily_remaining": None}

        day = await self._window("d", daily, cost, self.day_ms, now_ms)
        if day["limited"]:
            return {**day, "daily_remaining": 0}

        return {
            "limited": False,
            "remaining": minute["remaining"],
            "daily_remaining": day["remaining"],
        }

    async def _window(self, key, limit, cost, window_ms, now_ms):
        window = now_ms // window_ms
        raw = await self.storage.get(key)
        rec = json.loads(raw) if raw else {"w": window, "n": 0}
        if rec["w"] != window:
            rec = {"w": window, "n": 0}
        if rec["n"] >= limit:
            return {
                "limited": True,
                "scope": "day" if window_ms == self.day_ms else "minute",
                "remaining": 0,
                "reset_in": (window_ms - (now_ms % window_ms)) / 1000,
            }
        rec["n"] += cost
        await self.storage.put(key, json.dumps(rec))
        return {"limited": False, "remaining": max(0, limit - rec["n"])}


try:
    from workers import DurableObject
except ImportError:  # pragma: no cover - not present outside the runtime
    DurableObject = None


if DurableObject is not None:  # pragma: no cover - only inside the Worker

    class RateLimiter(DurableObject):
        """RPC surface for the Worker: check(limit, daily, cost=1)."""

        def __init__(self, ctx, env):
            super().__init__(ctx, env)

        async def check(self, limit, daily, cost=1):
            logic = RateLimiterLogic(self.ctx.storage)
            return await logic.check(limit, daily, cost)