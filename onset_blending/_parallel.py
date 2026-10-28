"""Run one function over many inputs across spawned processes."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from multiprocessing import get_context


def map_ordered(fn, items, context, workers: int) -> list:
    """fn(item, context) for each item, in order, across up to `workers`
    processes. Workers are spawned, so fn must be a module-level function."""
    items = list(items)
    workers = min(int(workers), len(items))
    if workers <= 1:
        return [fn(item, context) for item in items]
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))
    try:
        return list(pool.map(fn, items, repeat(context)))
    finally:
        # On a failure, drop the queued items instead of finishing them first.
        pool.shutdown(cancel_futures=True)


def workers_within(item_bytes: list[int], workers: int, budget_bytes: int | None) -> int:
    """How many items to run at once so their combined peak fits the budget,
    sized by the largest."""
    if budget_bytes is None or not item_bytes:
        return max(1, int(workers))
    return max(1, min(int(workers), budget_bytes // max(max(item_bytes), 1)))


def default_resources() -> tuple[int, int | None]:
    """(workers, memory budget): every core, sharing half the memory available
    now; other work on the machine may hold much of the rest."""
    import os

    workers = os.cpu_count() or 1
    try:
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return workers, int(line.split()[1]) * 1024 // 2
    except OSError:
        pass
    try:
        return workers, os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") // 2
    except (ValueError, OSError):
        return 1, None
