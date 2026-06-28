"""Diagnostic wrapper: start server with tracemalloc, log memory and top allocators."""

import os
import time
import tracemalloc
import threading
from typing import Final, TextIO

# Enable tracemalloc with 25 frames of stack trace
tracemalloc.start(25)

# Monkey-patch server.py's main to add monitoring
import server as server_mod  # noqa: E402

original_main = server_mod.main

SNAPSHOT_INTERVAL: Final[int] = 30
LOG_FILE: Final[str] = "mem_diag.log"


def _monitor() -> None:
    log_fh: TextIO = open(LOG_FILE, "w", buffering=1)
    log_fh.write(f"Memory diagnosis started at {time.ctime()}\n")
    log_fh.write(f"interval={SNAPSHOT_INTERVAL}s\n")
    log_fh.write(
        f"{'elapsed_s':>10} {'rss_kb':>10} {'traced_mb':>12} {'top_alloc_mb':>12} {'top_site'}\n"
    )
    last_snapshot: tracemalloc.Snapshot | None = None
    start: float = time.time()
    while True:
        time.sleep(SNAPSHOT_INTERVAL)
        elapsed: float = time.time() - start
        rss_kb: int = 0
        try:
            with open(f"/proc/{os.getpid()}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss_kb = int(line.split()[1])
                        break
        except (FileNotFoundError, IOError, ValueError):
            pass
        snapshot: tracemalloc.Snapshot = tracemalloc.take_snapshot()
        traced_size: float = (
            sum(stat.size for stat in snapshot.statistics("lineno")) / 1024 / 1024
        )
        top_stats: list[tracemalloc.Statistic] = snapshot.statistics("lineno")[:3]
        top_site: str = ""
        top_mb: float = 0.0
        if top_stats:
            top_stat: tracemalloc.Statistic = top_stats[0]
            top_mb = top_stat.size / 1024 / 1024
            top_site = str(top_stat.traceback[0])
        log_fh.write(
            f"{elapsed:10.0f} {rss_kb:10d} {traced_size:12.1f} {top_mb:12.1f} {top_site}\n"
        )
        if last_snapshot is not None:
            diff: list[tracemalloc.StatisticDiff] = snapshot.compare_to(
                last_snapshot, "lineno"
            )
            if diff:
                top_diff: list[tracemalloc.StatisticDiff] = diff[:5]
                log_fh.write("  Top 5 growths since last check:\n")
                for d in top_diff[:5]:
                    log_fh.write(
                        f"    +{d.size_diff / 1024 / 1024:8.1f}MB {d.count_diff:6d} {d.traceback[0] if d.traceback else '?'}\n"
                    )
        last_snapshot = snapshot
        log_fh.flush()

    log_fh.close()


def main() -> None:
    t = threading.Thread(target=_monitor, daemon=True)
    t.start()
    original_main()


if __name__ == "__main__":
    main()
