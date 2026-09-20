import asyncio

from limiter import RateLimiterLogic


class FakeStorage:
    def __init__(self):
        self.data = {}

    async def get(self, key):
        return self.data.get(key)

    async def put(self, key, value):
        self.data[key] = value


def make_logic(minute_ms=60000, day_ms=86400000):
    storage = FakeStorage()
    logic = RateLimiterLogic(storage, minute_ms=minute_ms, day_ms=day_ms)
    return logic, storage


def run(coro):
    return asyncio.run(coro)


def test_permits_under_limit():
    logic, _ = make_logic()
    res = run(logic.check(3, 10))
    assert res["limited"] is False
    assert res["remaining"] == 2
    assert res["daily_remaining"] == 9


def test_blocks_at_minute_limit():
    logic, _ = make_logic()
    for _ in range(3):
        run(logic.check(3, 10))
    res = run(logic.check(3, 10))
    assert res["limited"] is True
    assert res["scope"] == "minute"
    assert res["remaining"] == 0


def test_counts_cost_per_input():
    logic, _ = make_logic()
    run(logic.check(3, 10, cost=3))  # one request of 3 spends the minute limit
    res = run(logic.check(3, 10, cost=1))
    assert res["limited"] is True
    assert res["daily_remaining"] is None


def test_window_rolls_over():
    logic, storage = make_logic(minute_ms=60000)
    for _ in range(3):
        run(logic.check(3, 100))
    # Move the minute counter to a previous window: counting must restart.
    storage.data["m"] = '{"w": 0, "n": 3}'
    res = run(logic.check(3, 100))
    assert res["limited"] is False
    assert res["remaining"] == 2


def test_day_limit_blocks_after_minute_passes():
    logic, _ = make_logic()
    for _ in range(10):
        run(logic.check(1000, 10))
    res = run(logic.check(1000, 10))
    assert res["limited"] is True
    assert res["scope"] == "day"