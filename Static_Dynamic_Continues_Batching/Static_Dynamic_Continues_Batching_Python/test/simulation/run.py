#!/usr/bin/env python3
"""
Entry-point for the simulation.
Launches the resource monitor, the Rich live dashboard, and the async runner
all at once.

Usage (from project root):
    .venv/bin/python test/simulation/run.py
"""

import asyncio
import sys
import os

# Ensure the simulation directory is on the path so local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console

from dashboard import build_dashboard, create_live
from monitor import ResourceMonitor
from runner import Stats, run_simulation


async def main():
    console = Console()
    stats = Stats()
    monitor = ResourceMonitor(poll_interval=1.0)
    monitor.start()

    live = create_live(console)

    async def dashboard_loop():
        """Keep refreshing the dashboard until the simulation finishes."""
        while stats.end_time == 0.0:
            live.update(build_dashboard(stats, monitor.history))
            await asyncio.sleep(0.25)
        # One final render after completion
        live.update(build_dashboard(stats, monitor.history))

    try:
        with live:
            # Run dashboard + simulation concurrently
            await asyncio.gather(
                run_simulation(stats),
                dashboard_loop(),
            )
            # Keep the final dashboard on screen for a moment
            live.update(build_dashboard(stats, monitor.history))
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()

    # Print a static final report after the live display exits
    console.print()
    console.rule("[bold cyan]FINAL REPORT[/]")
    console.print(f"  Total users        : {stats.total_users}")
    console.print(f"  Completed          : {stats.completed}")
    console.print(f"  Elapsed            : {stats.elapsed:.2f}s")
    console.print(f"  Avg latency        : {stats.avg_latency:.2f}s")
    console.print(f"  P50 latency        : {stats.p50_latency:.2f}s")
    console.print(f"  P99 latency        : {stats.p99_latency:.2f}s")
    console.print(f"  Throughput         : {stats.rps:.1f} req/s")
    console.print(f"  Context leaks      : {stats.context_leaks}")
    console.print(f"  Errors             : {stats.errors}")
    console.rule()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
