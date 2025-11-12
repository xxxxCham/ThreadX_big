"""Analyze ThreadX performance bottleneck during sweep."""

import psutil
import time
import threading
from datetime import datetime


def monitor_system(duration=30, interval=2):
    """Monitor CPU, RAM, disk I/O for duration seconds."""
    print(f"🔍 Monitoring system for {duration}s (interval={interval}s)...")
    print(
        f"{'Time':<10} {'CPU%':<8} {'RAM%':<8} {'Disk Read MB/s':<16} {'Disk Write MB/s':<17} {'Threads':<10}"
    )
    print("-" * 90)

    disk_io_start = psutil.disk_io_counters()
    start_time = time.time()

    for i in range(duration // interval):
        time.sleep(interval)

        # CPU & RAM
        cpu_percent = psutil.cpu_percent(interval=0.1)
        ram_percent = psutil.virtual_memory().percent

        # Disk I/O
        disk_io_now = psutil.disk_io_counters()
        elapsed = time.time() - start_time
        read_mb_s = (
            (disk_io_now.read_bytes - disk_io_start.read_bytes) / (1024**2) / elapsed
        )
        write_mb_s = (
            (disk_io_now.write_bytes - disk_io_start.write_bytes) / (1024**2) / elapsed
        )

        # Threads count
        thread_count = threading.active_count()

        timestamp = datetime.now().strftime("%H:%M:%S")
        print(
            f"{timestamp:<10} {cpu_percent:<7.1f}% {ram_percent:<7.1f}% {read_mb_s:<15.2f} {write_mb_s:<16.2f} {thread_count:<10}"
        )

        disk_io_start = disk_io_now
        start_time = time.time()

    print("\n📊 Analysis:")
    print(
        f"   - CPU usage: {'LOW (<30%)' if cpu_percent < 30 else 'MODERATE (30-70%)' if cpu_percent < 70 else 'HIGH (>70%)'}"
    )
    print(
        f"   - Disk I/O: {'LOW (<10 MB/s)' if max(read_mb_s, write_mb_s) < 10 else 'MODERATE (10-50 MB/s)' if max(read_mb_s, write_mb_s) < 50 else 'HIGH (>50 MB/s)'}"
    )

    if cpu_percent < 30 and max(read_mb_s, write_mb_s) < 10:
        print("\n⚠️  BOTTLENECK DETECTED:")
        print(
            "   → CPU & Disk I/O both low → Likely GIL contention or inefficient parallelism"
        )
        print("   → Check: Worker synchronization, indicator caching, batch size")
    elif cpu_percent < 30 and max(read_mb_s, write_mb_s) > 50:
        print("\n⚠️  BOTTLENECK DETECTED:")
        print("   → Disk I/O high, CPU low → I/O bound workload")
        print("   → Check: Data loading strategy, cache hit rate")
    else:
        print("\n✅ System utilization looks normal")


if __name__ == "__main__":
    print("🚀 ThreadX Bottleneck Analyzer")
    print("=" * 90)
    print("\nLaunch your sweep in another terminal, then run this script.")
    print("Press ENTER when sweep is running to start monitoring...")
    input()

    monitor_system(duration=30, interval=2)
