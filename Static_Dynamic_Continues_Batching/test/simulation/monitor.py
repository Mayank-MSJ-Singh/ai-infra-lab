"""
System resource monitor – collects CPU, RAM, GPU and VRAM stats.
Runs in a background thread so it never blocks the async event loop.
"""

import subprocess
import threading
import time
from dataclasses import dataclass, field

import psutil


@dataclass
class SystemSnapshot:
    cpu_percent: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    ram_percent: float = 0.0
    gpu_util_percent: float = 0.0
    gpu_mem_used_mb: float = 0.0
    gpu_mem_total_mb: float = 0.0
    gpu_name: str = "N/A"
    gpu_available: bool = False


@dataclass
class MonitorHistory:
    """Keeps the latest snapshot + peak values seen so far."""
    latest: SystemSnapshot = field(default_factory=SystemSnapshot)
    peak_cpu: float = 0.0
    peak_ram_percent: float = 0.0
    peak_gpu_util: float = 0.0
    peak_gpu_mem_mb: float = 0.0


def _read_gpu() -> dict:
    """Query nvidia-smi for GPU stats. Returns empty dict if unavailable."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return {}
        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        return {
            "name": parts[0],
            "util": float(parts[1]),
            "mem_used": float(parts[2]),
            "mem_total": float(parts[3]),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return {}


def _take_snapshot() -> SystemSnapshot:
    snap = SystemSnapshot()

    # CPU
    snap.cpu_percent = psutil.cpu_percent(interval=None)

    # RAM
    mem = psutil.virtual_memory()
    snap.ram_used_gb = mem.used / (1024 ** 3)
    snap.ram_total_gb = mem.total / (1024 ** 3)
    snap.ram_percent = mem.percent

    # GPU
    gpu = _read_gpu()
    if gpu:
        snap.gpu_available = True
        snap.gpu_name = gpu["name"]
        snap.gpu_util_percent = gpu["util"]
        snap.gpu_mem_used_mb = gpu["mem_used"]
        snap.gpu_mem_total_mb = gpu["mem_total"]

    return snap


class ResourceMonitor:
    """Polls system resources on a background thread."""

    def __init__(self, poll_interval: float = 1.0):
        self._poll_interval = poll_interval
        self._history = MonitorHistory()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Prime psutil so the first real reading isn't zero
        psutil.cpu_percent(interval=None)

    @property
    def history(self) -> MonitorHistory:
        return self._history

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self):
        while not self._stop.is_set():
            snap = _take_snapshot()
            h = self._history
            h.latest = snap
            h.peak_cpu = max(h.peak_cpu, snap.cpu_percent)
            h.peak_ram_percent = max(h.peak_ram_percent, snap.ram_percent)
            if snap.gpu_available:
                h.peak_gpu_util = max(h.peak_gpu_util, snap.gpu_util_percent)
                h.peak_gpu_mem_mb = max(h.peak_gpu_mem_mb, snap.gpu_mem_used_mb)
            self._stop.wait(self._poll_interval)
