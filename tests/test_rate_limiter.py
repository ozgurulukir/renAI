import asyncio
import time

import pytest

from renai.utils.rate_limiter import AsyncRateLimiter


class TestAsyncRateLimiterInit:
    def test_delay_calculated_from_rpm(self):
        rl = AsyncRateLimiter(rpm=120)
        assert rl.delay == 0.5

    def test_delay_for_60_rpm(self):
        rl = AsyncRateLimiter(rpm=60)
        assert rl.delay == 1.0

    def test_zero_rpm_disables_limiting(self):
        rl = AsyncRateLimiter(rpm=0)
        assert rl.delay == 0
        assert rl.rpm == 0

    def test_negative_rpm_disables_limiting(self):
        rl = AsyncRateLimiter(rpm=-5)
        assert rl.delay == 0

    def test_repr(self):
        rl = AsyncRateLimiter(rpm=120)
        assert "rpm=120" in repr(rl)
        assert "delay=0.500s" in repr(rl)


class TestAsyncRateLimiterAcquire:
    @pytest.mark.asyncio
    async def test_first_acquire_does_not_wait(self):
        rl = AsyncRateLimiter(rpm=60)
        start = time.perf_counter()
        await rl.acquire()
        elapsed = time.perf_counter() - start
        assert elapsed < 0.05

    @pytest.mark.asyncio
    async def test_second_acquire_waits_for_delay(self):
        rl = AsyncRateLimiter(rpm=60)
        await rl.acquire()
        start = time.perf_counter()
        await rl.acquire()
        elapsed = time.perf_counter() - start
        assert elapsed >= 0.9
        assert elapsed < 1.2

    @pytest.mark.asyncio
    async def test_zero_rpm_acquire_returns_immediately(self):
        rl = AsyncRateLimiter(rpm=0)
        start = time.perf_counter()
        await rl.acquire()
        await rl.acquire()
        elapsed = time.perf_counter() - start
        assert elapsed < 0.05

    @pytest.mark.asyncio
    async def test_concurrent_acquires_are_serialized(self):
        rl = AsyncRateLimiter(rpm=60)
        start = time.perf_counter()
        await asyncio.gather(rl.acquire(), rl.acquire(), rl.acquire())
        elapsed = time.perf_counter() - start
        assert elapsed >= 1.8
        assert elapsed < 2.5

    @pytest.mark.asyncio
    async def test_acquire_after_delay_does_not_wait(self):
        rl = AsyncRateLimiter(rpm=600)
        await rl.acquire()
        await asyncio.sleep(rl.delay + 0.05)
        start = time.perf_counter()
        await rl.acquire()
        elapsed = time.perf_counter() - start
        assert elapsed < 0.05
