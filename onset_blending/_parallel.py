"""Run one function over many inputs across spawned processes."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from multiprocessing import get_context

import numpy as np


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


_BLAS_LIMIT = None


def limit_blas_threads(threads: int) -> None:
    """Cap this process's BLAS/OpenMP thread pools (a worker initializer)."""
    global _BLAS_LIMIT
    from threadpoolctl import threadpool_limits

    _BLAS_LIMIT = threadpool_limits(limits=int(threads))


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


class SharedArrays:
    """Arrays in shared memory for spawned workers to read without copies.

        with SharedArrays(p=probabilities, years=years) as shared:
            pool.submit(task, shared.specs)          # task calls attach(specs)
    """

    def __init__(self, **arrays):
        from multiprocessing import shared_memory

        self._blocks, self.specs = [], {}
        try:
            for name, array in arrays.items():
                array = np.ascontiguousarray(array)
                block = shared_memory.SharedMemory(create=True, size=max(array.nbytes, 1))
                self._blocks.append(block)
                np.ndarray(array.shape, array.dtype, buffer=block.buf)[...] = array
                self.specs[name] = (block.name, array.shape, array.dtype.str)
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        for block in self._blocks:
            block.close()
            block.unlink()
        self._blocks = []


_ATTACHED: dict = {}


def attach(specs: dict) -> dict:
    """Read-only views of SharedArrays, in a worker (attached once per block)."""
    from multiprocessing import shared_memory

    out = {}
    for name, (block_name, shape, dtype) in specs.items():
        if block_name not in _ATTACHED:
            _ATTACHED[block_name] = shared_memory.SharedMemory(name=block_name)
        view = np.ndarray(shape, dtype, buffer=_ATTACHED[block_name].buf)
        view.flags.writeable = False
        out[name] = view
    return out
