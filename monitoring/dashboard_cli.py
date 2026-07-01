from __future__ import annotations

import os
import sys
import time
from typing import Optional, Any

# Ensure import paths work if executed from workspace root or monitoring dir
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from monitoring.metrics import MetricsCollector, registry

# Try to import RedisManager to connect to active cache instance
try:
    from week7.backend.services.redis_manager import RedisManager
    from week7.backend.configs.config import get_settings
    REDIS_MANAGER_AVAILABLE = True
except ImportError:
    REDIS_MANAGER_AVAILABLE = False

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import ProgressBar
    from rich.table import Table
    from rich.live import Live
    from rich.layout import Layout
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def create_mock_metrics():
    """Seed registry with mock request history if it is empty to demonstrate UI features."""
    if registry._counters.get("fashion_assistant_api_requests_total", 0.0) == 0.0:
        registry.inc_counter("api_requests_total", 142)
        registry.observe_histogram("api_latency", 0.085)
        registry.observe_histogram("api_latency", 0.125)
        registry.observe_histogram("api_latency", 0.245)
        registry.observe_histogram("api_latency", 0.005)
        
        registry.inc_counter("generation_requests_total", 12)
        registry.observe_histogram("generation_duration", 6.8)
        registry.observe_histogram("generation_duration", 8.4)
        registry.observe_histogram("generation_duration", 11.2)
        registry.observe_histogram("generation_duration", 5.1)


def get_progress_bar(percentage: float, width: int = 15) -> str:
    """Helper to generate plain text bar in case Rich is not present."""
    filled = int(round((percentage / 100.0) * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {percentage:.1f}%"


def generate_console_view(collector: MetricsCollector) -> str:
    """Fallback plain-text view generator if rich is missing."""
    stats = collector.collect_all()
    sys_stats = stats.get("system", {})
    gpu_stats = stats.get("gpu", {})
    redis_stats = stats.get("redis", {})
    celery_stats = stats.get("celery", {})

    output = []
    output.append("==============================================================")
    output.append("         📊 AI FASHION ASSISTANT LIVE DIAGNOSTICS CONSOLE")
    output.append("==============================================================")
    output.append(f" Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    output.append("--------------------------------------------------------------")
    output.append("SYSTEM RESOURCES:")
    cpu = sys_stats.get("cpu_percent", 0.0)
    output.append(f"  • CPU Utilization       : {get_progress_bar(cpu)} ({sys_stats.get('cpu_cores', 1)} Cores)")
    mem = sys_stats.get("memory_percent", 0.0)
    output.append(f"  • RAM Utilization       : {get_progress_bar(mem)} (Used: {sys_stats.get('memory_used_bytes', 0) / 1e9:.1f} GB / {sys_stats.get('memory_total_bytes', 0) / 1e9:.1f} GB)")
    output.append(f"  • API Process RAM       : {sys_stats.get('process_memory_mb', 0.0):.1f} MB")
    
    if gpu_stats.get("gpu_available", 0):
        gpu_pct = gpu_stats.get("gpu_percent", 0.0)
        output.append(f"  • GPU Utilization       : {get_progress_bar(gpu_pct)}")
        output.append(f"  • GPU VRAM Usage        : {gpu_stats.get('gpu_memory_used_mb', 0.0):.0f} MB / {gpu_stats.get('gpu_memory_total_mb', 0.0):.0f} MB")
    else:
        output.append("  • GPU State             : NOT AVAILABLE / CPU MODE")
        
    output.append("\nREDIS & QUEUES:")
    redis_online = "ONLINE" if redis_stats.get("status", 0) == 1 else "OFFLINE"
    output.append(f"  • Redis Status          : {redis_online} (Connected Clients: {redis_stats.get('connected_clients', 0)})")
    output.append(f"  • Redis Memory          : {redis_stats.get('used_memory_bytes', 0) / 1024 / 1024:.1f} MB")
    output.append(f"  • Celery Queue Length   : {celery_stats.get('queue_length', 0)} pending tasks")

    output.append("\nPIPELINE TELEMETRY:")
    # Query API Histograms
    api_requests = registry._counters.get("fashion_assistant_api_requests_total", 0.0)
    api_latency_hist = registry._histograms.get("fashion_assistant_api_latency", [])
    if api_latency_hist:
        avg_api = sum(api_latency_hist) / len(api_latency_hist)
        min_api = min(api_latency_hist)
        max_api = max(api_latency_hist)
        output.append(f"  • Total API Requests    : {int(api_requests)}")
        output.append(f"  • API Latency           : Avg: {avg_api:.3f}s | Min: {min_api:.3f}s | Max: {max_api:.3f}s")
    else:
        output.append(f"  • Total API Requests    : {int(api_requests)} (No latencies recorded)")

    # Query Generation Histograms
    gen_requests = registry._counters.get("fashion_assistant_generation_requests_total", 0.0)
    gen_duration_hist = registry._histograms.get("fashion_assistant_generation_duration", [])
    if gen_duration_hist:
        avg_gen = sum(gen_duration_hist) / len(gen_duration_hist)
        min_gen = min(gen_duration_hist)
        max_gen = max(gen_duration_hist)
        output.append(f"  • Total Generations     : {int(gen_requests)}")
        output.append(f"  • Generation Time       : Avg: {avg_gen:.2f}s | Min: {min_gen:.2f}s | Max: {max_gen:.2f}s")
    else:
        output.append(f"  • Total Generations     : {int(gen_requests)} (No runs recorded)")

    output.append("--------------------------------------------------------------")
    output.append("Press Ctrl+C to exit.")
    return "\n".join(output)


def make_rich_dashboard(collector: MetricsCollector) -> Table:
    """Generate a clean, professional multi-panel grid utilizing rich widgets."""
    stats = collector.collect_all()
    sys_stats = stats.get("system", {})
    gpu_stats = stats.get("gpu", {})
    redis_stats = stats.get("redis", {})
    celery_stats = stats.get("celery", {})

    table = Table(show_header=False, expand=True, box=None)
    table.add_column("Left", ratio=1)
    table.add_column("Right", ratio=1)

    # 1. System Info Panel
    sys_table = Table(show_header=False, box=None)
    cpu_val = sys_stats.get("cpu_percent", 0.0)
    sys_table.add_row("CPU Utilization", f"[cyan]{cpu_val:.1f}%[/] ({sys_stats.get('cpu_cores', 1)} Cores)")
    ram_val = sys_stats.get("memory_percent", 0.0)
    ram_used = sys_stats.get("memory_used_bytes", 0) / 1e9
    ram_tot = sys_stats.get("memory_total_bytes", 0) / 1e9
    sys_table.add_row("Memory (RAM) Utilization", f"[yellow]{ram_val:.1f}%[/] ({ram_used:.1f} GB / {ram_tot:.1f} GB)")
    sys_table.add_row("Process RSS Memory", f"{sys_stats.get('process_memory_mb', 0.0):.1f} MB")
    
    if gpu_stats.get("gpu_available", 0):
        gpu_pct = gpu_stats.get("gpu_percent", 0.0)
        sys_table.add_row("GPU Utilization", f"[magenta]{gpu_pct:.1f}%[/]")
        sys_table.add_row("GPU VRAM Usage", f"{gpu_stats.get('gpu_memory_used_mb', 0.0):.0f} MB / {gpu_stats.get('gpu_memory_total_mb', 0.0):.0f} MB")
    else:
        sys_table.add_row("GPU Processing", "[grey50]CPU/Fallback Mode[/]")

    # 2. Redis & Celery Panel
    db_table = Table(show_header=False, box=None)
    redis_status_str = "[green]ONLINE[/]" if redis_stats.get("status", 0) == 1 else "[red]OFFLINE[/]"
    db_table.add_row("Redis Connection", redis_status_str)
    db_table.add_row("Connected Clients", str(redis_stats.get("connected_clients", 0)))
    db_table.add_row("Redis Allocated Memory", f"{redis_stats.get('used_memory_bytes', 0) / 1024 / 1024:.2f} MB")
    db_table.add_row("Celery Backlog Queue", f"[bold yellow]{celery_stats.get('queue_length', 0)}[/] tasks")

    # 3. Telemetry statistics panel
    tel_table = Table(show_header=True, box=None)
    tel_table.add_column("Telemetry Metric")
    tel_table.add_column("Count")
    tel_table.add_column("Avg Duration")
    tel_table.add_column("Range (Min/Max)")

    # API Latency
    api_requests = int(registry._counters.get("fashion_assistant_api_requests_total", 0.0))
    api_latency_hist = registry._histograms.get("fashion_assistant_api_latency", [])
    if api_latency_hist:
        avg_api = f"{sum(api_latency_hist) / len(api_latency_hist):.3f}s"
        rng_api = f"{min(api_latency_hist):.3f}s / {max(api_latency_hist):.3f}s"
    else:
        avg_api, rng_api = "-", "-"
    tel_table.add_row("API Request Processing", str(api_requests), avg_api, rng_api)

    # Generation durations
    gen_requests = int(registry._counters.get("fashion_assistant_generation_requests_total", 0.0))
    gen_duration_hist = registry._histograms.get("fashion_assistant_generation_duration", [])
    if gen_duration_hist:
        avg_gen = f"{sum(gen_duration_hist) / len(gen_duration_hist):.2f}s"
        rng_gen = f"{min(gen_duration_hist):.2f}s / {max(gen_duration_hist):.2f}s"
    else:
        avg_gen, rng_gen = "-", "-"
    tel_table.add_row("Fashion Image Generation", str(gen_requests), avg_gen, rng_gen)

    # Add components to layout
    left_panel = Panel(sys_table, title="[bold white]🖥️ System Telemetry[/]", border_style="cyan")
    right_panel = Panel(db_table, title="[bold white]📦 Queue & Memory Cache[/]", border_style="yellow")
    bottom_panel = Panel(tel_table, title="[bold white]⏱️ Pipeline Latency Summary[/]", border_style="magenta")

    table.add_row(left_panel, right_panel)
    
    # Return outer layout wrapper
    outer_table = Table(show_header=False, expand=True, box=None)
    outer_table.add_column("Header", ratio=1)
    outer_table.add_row(f"[bold white]📊 AI FASHION ASSISTANT LIVE DIAGNOSTICS CONSOLE[/] - [cyan]{time.strftime('%Y-%m-%d %H:%M:%S')}[/]")
    outer_table.add_row(table)
    outer_table.add_row(bottom_panel)
    outer_table.add_row("[grey50]Press Ctrl+C to stop real-time monitoring.[/]")
    return outer_table


def main():
    # Attempt to initialize Redis Client using the app configuration settings
    redis_client = None
    if REDIS_MANAGER_AVAILABLE:
        try:
            settings = get_settings()
            manager = RedisManager()
            manager.start()
            redis_client = manager.client
        except Exception:
            pass

    collector = MetricsCollector(redis_client=redis_client)
    
    # Load demo mock history for illustration
    create_mock_metrics()

    # Clear terminal screen
    os.system("cls" if os.name == "nt" else "clear")

    if RICH_AVAILABLE:
        console = Console()
        try:
            with Live(make_rich_dashboard(collector), refresh_per_second=1, console=console) as live:
                while True:
                    time.sleep(1)
                    live.update(make_rich_dashboard(collector))
        except KeyboardInterrupt:
            console.print("\n[yellow]Live monitoring stopped.[/]")
    else:
        try:
            while True:
                # Clear terminal and print output
                os.system("cls" if os.name == "nt" else "clear")
                print(generate_console_view(collector))
                time.sleep(2)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")


if __name__ == "__main__":
    main()
