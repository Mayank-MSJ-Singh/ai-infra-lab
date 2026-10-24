"""
Simulation runner – fires concurrent user sessions against the API.
All state is stored in a shared Stats dataclass so the dashboard can read it.
"""

import asyncio
import time
from dataclasses import dataclass, field

import httpx

from config import (
    BASE_URL,
    CONNECT_TIMEOUT,
    DELAY_BETWEEN_MSGS,
    MAX_CONCURRENCY,
    REQUEST_TIMEOUT,
    TOTAL_USERS,
)


@dataclass
class Stats:
    """Thread-safe-ish stats (single writer thread – the event loop)."""
    total_users: int = 0
    completed: int = 0
    in_flight: int = 0
    conversations_created: int = 0
    msg1_ok: int = 0
    msg2_ok: int = 0
    context_leaks: int = 0
    isolated: int = 0
    errors: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    latencies: list = field(default_factory=list)
    error_details: list = field(default_factory=list)   # Last N errors

    @property
    def elapsed(self) -> float:
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time if self.start_time else 0.0

    @property
    def avg_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    @property
    def p50_latency(self) -> float:
        if not self.latencies:
            return 0.0
        s = sorted(self.latencies)
        return s[len(s) // 2]

    @property
    def p99_latency(self) -> float:
        if not self.latencies:
            return 0.0
        s = sorted(self.latencies)
        idx = int(len(s) * 0.99)
        return s[min(idx, len(s) - 1)]

    @property
    def rps(self) -> float:
        elapsed = self.elapsed
        if elapsed <= 0:
            return 0.0
        # Each completed user does 3 requests (conv + msg1 + msg2)
        return (self.completed * 3) / elapsed


async def _simulate_one(
    client: httpx.AsyncClient,
    user_id: int,
    sem: asyncio.Semaphore,
    stats: Stats,
):
    user_name = f"User_{user_id}"

    async with sem:
        stats.in_flight += 1
        user_start = time.time()

        # 1. Create conversation
        try:
            r = await client.post(f"{BASE_URL}/conversation/")
            r.raise_for_status()
            conv_id = r.json()["conversation_id"]
            stats.conversations_created += 1
        except Exception as e:
            stats.errors += 1
            stats.error_details.append(f"[{user_name}] conv create: {e}")
            stats.error_details = stats.error_details[-20:]  # Keep last 20
            stats.in_flight -= 1
            stats.completed += 1
            return

        # 2. Send introduction message
        try:
            r = await client.post(
                f"{BASE_URL}/chat/{conv_id}",
                json={"message": f"Hello, my name is {user_name}. Remember it."},
            )
            r.raise_for_status()
            stats.msg1_ok += 1
        except Exception as e:
            stats.errors += 1
            stats.error_details.append(f"[{user_name}] msg1: {e}")
            stats.error_details = stats.error_details[-20:]
            stats.in_flight -= 1
            stats.completed += 1
            return

        await asyncio.sleep(DELAY_BETWEEN_MSGS)

        # 3. Verify context isolation
        try:
            r = await client.post(
                f"{BASE_URL}/chat/{conv_id}",
                json={"message": "What is my name?"},
            )
            r.raise_for_status()
            stats.msg2_ok += 1

            body = r.json().get("response", "")
            if user_name in body:
                stats.isolated += 1
            else:
                stats.context_leaks += 1
        except Exception as e:
            stats.errors += 1
            stats.error_details.append(f"[{user_name}] msg2: {e}")
            stats.error_details = stats.error_details[-20:]

        user_end = time.time()
        stats.latencies.append(user_end - user_start)
        stats.in_flight -= 1
        stats.completed += 1


async def run_simulation(stats: Stats):
    """Launch all user tasks and wait for them to finish."""
    stats.total_users = TOTAL_USERS
    stats.start_time = time.time()

    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    timeout = httpx.Timeout(REQUEST_TIMEOUT, connect=CONNECT_TIMEOUT)

    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [
            _simulate_one(client, i, sem, stats)
            for i in range(1, TOTAL_USERS + 1)
        ]
        await asyncio.gather(*tasks)

    stats.end_time = time.time()
