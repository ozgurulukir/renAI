import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class AsyncRateLimiter:
    """
    AsyncRateLimiter - A token-bucket rate limiter for asyncio.
    Ensures that LLM requests stay within the specified RPM (Requests Per Minute) limit.
    """

    def __init__(self, rpm: int, capacity: int = 1):
        self.rpm = rpm
        self.delay = 60.0 / rpm if rpm > 0 else 0
        self.capacity = max(1, capacity)
        self.tokens = float(self.capacity)
        self.refill_rate = rpm / 60.0 if rpm > 0 else 0.0
        self.last_refill = time.perf_counter()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """
        Wait until a token is available based on the RPM limit, then consume one.
        """
        if self.rpm <= 0:
            return

        async with self.lock:
            now = time.perf_counter()
            elapsed = now - self.last_refill
            self.tokens = min(float(self.capacity), self.tokens + elapsed * self.refill_rate)
            self.last_refill = now

            if self.tokens < 1.0:
                wait_time = (1.0 - self.tokens) / self.refill_rate
                logger.debug(f"Rate limiting: waiting {wait_time:.3f}s for next token")
                await asyncio.sleep(wait_time)
                now = time.perf_counter()
                self.tokens = min(
                    float(self.capacity),
                    self.tokens + (now - self.last_refill) * self.refill_rate,
                )
                self.last_refill = now

            self.tokens -= 1.0

    def __repr__(self):
        return f"AsyncRateLimiter(rpm={self.rpm}, delay={self.delay:.3f}s)"
