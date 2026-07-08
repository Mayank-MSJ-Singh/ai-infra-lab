"""
Rich live-updating terminal dashboard.
Renders system resources, simulation progress, and latency stats.
"""

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from config import DASHBOARD_REFRESH_RATE, MAX_CONCURRENCY, TOTAL_USERS
from monitor import MonitorHistory
from runner import Stats


def _build_header() -> Panel:
    title = Text("⚡ LLM BATCHING SIMULATION DASHBOARD ⚡", style="bold cyan", justify="center")
    return Panel(title, style="bright_blue", height=3)


def _build_progress(stats: Stats) -> Panel:
    pct = (stats.completed / stats.total_users * 100) if stats.total_users else 0
    bar = ProgressBar(total=stats.total_users, completed=stats.completed, width=60)

    table = Table.grid(padding=(0, 2))
    table.add_row(bar)
    table.add_row(
        Text(
            f"{stats.completed}/{stats.total_users} users done  "
            f"({pct:.1f}%)  •  "
            f"In-flight: {stats.in_flight}/{MAX_CONCURRENCY}  •  "
            f"Elapsed: {stats.elapsed:.1f}s",
            style="white",
        )
    )
    return Panel(table, title="[bold green]Progress", border_style="green")


def _build_latency(stats: Stats) -> Panel:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim", width=18)
    table.add_column(style="bold white", width=12)

    table.add_row("Avg latency", f"{stats.avg_latency:.2f}s")
    table.add_row("P50 latency", f"{stats.p50_latency:.2f}s")
    table.add_row("P99 latency", f"{stats.p99_latency:.2f}s")
    table.add_row("Throughput", f"{stats.rps:.1f} req/s")

    return Panel(table, title="[bold yellow]Latency & Throughput", border_style="yellow")


def _build_results(stats: Stats) -> Panel:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim", width=22)
    table.add_column(width=12)

    table.add_row("Conversations", f"[cyan]{stats.conversations_created}[/]")
    table.add_row("Msg 1 (intro)", f"[cyan]{stats.msg1_ok}[/]")
    table.add_row("Msg 2 (verify)", f"[cyan]{stats.msg2_ok}[/]")

    leak_style = "bold red" if stats.context_leaks > 0 else "bold green"
    leak_icon = "❌" if stats.context_leaks > 0 else "✅"
    table.add_row("Context leaks", f"[{leak_style}]{stats.context_leaks} {leak_icon}[/]")
    table.add_row("Isolated", f"[bold green]{stats.isolated}[/]")
    table.add_row("Errors", f"[bold red]{stats.errors}[/]" if stats.errors else "[green]0[/]")

    return Panel(table, title="[bold magenta]Results", border_style="magenta")


def _build_resources(history: MonitorHistory) -> Panel:
    s = history.latest
    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Resource", style="dim", width=14)
    table.add_column("Current", width=14)
    table.add_column("Peak", width=14)

    # CPU
    cpu_style = "green" if s.cpu_percent < 70 else ("yellow" if s.cpu_percent < 90 else "red")
    table.add_row(
        "CPU",
        f"[{cpu_style}]{s.cpu_percent:.1f}%[/]",
        f"{history.peak_cpu:.1f}%",
    )

    # RAM
    ram_style = "green" if s.ram_percent < 70 else ("yellow" if s.ram_percent < 90 else "red")
    table.add_row(
        "RAM",
        f"[{ram_style}]{s.ram_used_gb:.1f}/{s.ram_total_gb:.1f} GB ({s.ram_percent:.1f}%)[/]",
        f"{history.peak_ram_percent:.1f}%",
    )

    # GPU
    if s.gpu_available:
        gpu_style = "green" if s.gpu_util_percent < 70 else ("yellow" if s.gpu_util_percent < 90 else "red")
        table.add_row(
            "GPU",
            f"[{gpu_style}]{s.gpu_util_percent:.0f}%[/]",
            f"{history.peak_gpu_util:.0f}%",
        )
        vram_pct = (s.gpu_mem_used_mb / s.gpu_mem_total_mb * 100) if s.gpu_mem_total_mb else 0
        vram_style = "green" if vram_pct < 70 else ("yellow" if vram_pct < 90 else "red")
        table.add_row(
            "VRAM",
            f"[{vram_style}]{s.gpu_mem_used_mb:.0f}/{s.gpu_mem_total_mb:.0f} MB[/]",
            f"{history.peak_gpu_mem_mb:.0f} MB",
        )
        gpu_title = f"[bold blue]System Resources  •  {s.gpu_name}"
    else:
        table.add_row("GPU", "[dim]not detected[/]", "—")
        gpu_title = "[bold blue]System Resources"

    return Panel(table, title=gpu_title, border_style="blue")


def _build_errors(stats: Stats) -> Panel:
    if not stats.error_details:
        content = Text("No errors 🎉", style="green italic")
    else:
        lines = "\n".join(stats.error_details[-8:])
        content = Text(lines, style="red")
    return Panel(content, title="[bold red]Recent Errors", border_style="red", height=12)


def build_dashboard(stats: Stats, history: MonitorHistory) -> Layout:
    layout = Layout()

    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="progress", size=6),
        Layout(name="middle", size=10),
        Layout(name="bottom", size=12),
    )

    layout["header"].update(_build_header())
    layout["progress"].update(_build_progress(stats))

    layout["middle"].split_row(
        Layout(name="latency"),
        Layout(name="results"),
    )
    layout["middle"]["latency"].update(_build_latency(stats))
    layout["middle"]["results"].update(_build_results(stats))

    layout["bottom"].split_row(
        Layout(name="resources"),
        Layout(name="errors"),
    )
    layout["bottom"]["resources"].update(_build_resources(history))
    layout["bottom"]["errors"].update(_build_errors(stats))

    return layout


def create_live(console: Console) -> Live:
    return Live(
        console=console,
        refresh_per_second=DASHBOARD_REFRESH_RATE,
        screen=True,
    )
