import asyncio
import logging
import time


class AsyncRateLimiter:
    """
    AsyncRateLimiter - A simple token bucket or delay-based rate limiter for asyncio.
    Ensures that LLM requests stay within the specified RPM (Requests Per Minute) limit.
    """

    def __init__(self, rpm: int):
        self.rpm = rpm
        self.delay = 60.0 / rpm if rpm > 0 else 0
        self.last_request_time = 0.0
        self.lock = asyncio.Lock()

    async def acquire(self):
        """
        Wait for the next available slot based on the RPM limit.
        """
        if self.rpm <= 0:
            return

        async with self.lock:
            current_time = time.perf_counter()
            elapsed = current_time - self.last_request_time

            if elapsed < self.delay:
                wait_time = self.delay - elapsed
                logging.debug(f"Rate limiting: waiting {wait_time:.3f}s for next slot")
                await asyncio.sleep(wait_time)
                # Update time after sleep for precision
                self.last_request_time = time.perf_counter()
            else:
                self.last_request_time = current_time

    def __repr__(self):
        return f"AsyncRateLimiter(rpm={self.rpm}, delay={self.delay:.3f}s)"
