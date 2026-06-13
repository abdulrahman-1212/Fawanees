"""
Benchmark Utilities
===================
Quick helpers for profiling the pipeline without external dependencies.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator


@contextmanager
def timer(label: str = "", verbose: bool = True) -> Generator[dict, None, None]:
    """
    Context manager that measures wall-clock time.

    Usage
    -----
    with timer("cartoonize") as t:
        result = pipe.run(image)
    print(t["ms"])   # elapsed milliseconds
    """
    info: dict = {}
    t0 = time.perf_counter()
    yield info
    elapsed = time.perf_counter() - t0
    info["ms"] = elapsed * 1_000
    info["s"] = elapsed
    if verbose:
        tag = f"[{label}] " if label else ""
        print(f"{tag}{elapsed * 1000:.1f} ms")


def print_timings(timings: dict[str, float], title: str = "Pipeline timings") -> None:
    """Pretty-print a timings dict from PipelineResult."""
    total = sum(timings.values())
    width = max(len(k) for k in timings) if timings else 10
    print(f"\n── {title} {'─' * (40 - len(title))}")
    for stage, ms in timings.items():
        bar = "█" * int(ms / total * 30) if total > 0 else ""
        print(f"  {stage:<{width}}  {ms:6.1f} ms  {bar}")
    print(f"  {'TOTAL':<{width}}  {total:6.1f} ms")
    print("─" * 46)


__all__ = ["timer", "print_timings"]