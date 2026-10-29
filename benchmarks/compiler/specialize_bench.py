#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""A new float value on a compiled kernel against a fresh compile.

Float parameters stay function arguments of the scheduled module, so calling
a compiled kernel with a new value is a cache hit that binds the value, while
``pulse.compile(kernel)`` traces, builds, runs the passes and schedules again.
For kernels of growing size, and for a sweep of 100 amplitudes, this prints
the median time of each and exits 1 if a new value is not at least
``--min-speedup`` times cheaper than a compile.

Usage:
    python benchmarks/compiler/specialize_bench.py [--iterations N] [--min-speedup X]
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import cudaq_pulse as pulse

_QUBIT = pulse.device("transmon_17q").qudit_handles[0]
_DRIVE_COUNTS = (1, 2, 5, 10, 20)


def _drives(count: int) -> pulse.Kernel:
    """A kernel of ``count`` Gaussian drives whose amplitude is a parameter.

    The widths depend on the loop index, so the loop unrolls and the module
    grows with ``count``.
    """

    @pulse.kernel
    def drives(q, amp: float):
        d = get_drive_line(q)
        for i in range(count):
            drive(d, gaussian(40 + 4 * (i % 3), amp, 10.0))

    return drives


def _median_ms(call: Callable[[int], object], iterations: int) -> float:
    call(0)
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        call(i)
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--min-speedup", type=float, default=10.0)
    options = parser.parse_args()

    print(f"  {'drives':>6s}  {'compile (ms)':>12s}  {'new value (ms)':>14s}  {'speedup':>8s}")
    speedups = []
    for count in _DRIVE_COUNTS:
        kernel = _drives(count)
        compiled = pulse.compile(kernel)
        compile_ms = _median_ms(
            lambda i, k=kernel: pulse.compile(k).lower(_QUBIT, 0.5), options.iterations
        )
        value_ms = _median_ms(
            lambda i, c=compiled: c.lower(_QUBIT, amp=0.01 * (i + 1)), options.iterations
        )
        speedups.append((f"{count} drives", compile_ms / value_ms))
        print(f"  {count:6d}  {compile_ms:12.3f}  {value_ms:14.3f}  {speedups[-1][1]:7.1f}x")

    sweep = _drives(1)
    compiled = pulse.compile(sweep)
    amplitudes = [0.01 * (i + 1) for i in range(100)]
    start = time.perf_counter()
    for amp in amplitudes:
        compiled.lower(_QUBIT, amp=amp)
    value_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    for amp in amplitudes:
        pulse.compile(sweep).lower(_QUBIT, amp)
    compile_ms = (time.perf_counter() - start) * 1000
    speedups.append(("sweep of 100", compile_ms / value_ms))
    print(
        f"\n  sweep of 100 amplitudes: {value_ms:.2f} ms with new values, "
        f"{compile_ms:.2f} ms compiling each ({speedups[-1][1]:.1f}x)\n"
    )

    slow = [(name, s) for name, s in speedups if s < options.min_speedup]
    for name, speedup in slow:
        print(f"  FAIL {name}: a new value is only {speedup:.1f}x cheaper than a compile")
    return 1 if slow else 0


if __name__ == "__main__":
    raise SystemExit(main())
