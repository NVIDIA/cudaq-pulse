#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime qudits against a compile per placement.

A kernel whose qudit parameter is annotated ``Qudit`` compiles once and binds
the qudit at each call; without the annotation the qudit is static, so each
new qudit compiles the kernel again. For kernels of growing size this places
each on every qudit of a 17-qudit quantum system and prints the median time of the
whole placement sweep (with a controller export per call) for both, and of a
repeated call (a cache hit). The exports, the same on both sides, bound the
speedup. It exits 1 if the sweep with a runtime qudit is not at least
``--min-speedup`` times cheaper.

Usage:
    python benchmarks/compiler/relocation_bench.py [--iterations N] [--min-speedup X]
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import cudaq_pulse as pulse
from cudaq_pulse import Qudit

_HANDLES = pulse.quantum_system("transmon_17q").qudits
_DRIVE_COUNTS = (1, 5, 20)


def _drives(count: int, relocatable: bool) -> pulse.Kernel:
    """``count`` Gaussian drives and a readout on one qudit."""

    def drives(q, amp: float):
        d = get_drive_line(q)
        for i in range(count):
            drive(d, gaussian(40 + 4 * (i % 3), amp, 10.0))
        rl = get_readout_line(q)
        return readout(rl, square(100, 0.05))

    if relocatable:
        drives.__annotations__["q"] = Qudit
    return pulse.kernel(drives)


def _median_ms(call: Callable[[], object], iterations: int) -> float:
    call()
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        call()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def _sweep(count: int, relocatable: bool) -> None:
    compiled = pulse.compile(_drives(count, relocatable))
    for q in _HANDLES:
        compiled.lower(q, 0.1).export_controller()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--min-speedup", type=float, default=2.0)
    options = parser.parse_args()

    n = len(_HANDLES)
    print(
        f"  {'drives':>6s}  {f'static x{n} (ms)':>16s}  {f'runtime x{n} (ms)':>17s}"
        f"  {'speedup':>8s}  {'hit static (us)':>15s}  {'hit runtime (us)':>16s}"
    )
    slow = []
    for count in _DRIVE_COUNTS:
        static_ms = _median_ms(lambda c=count: _sweep(c, False), options.iterations)
        runtime_ms = _median_ms(lambda c=count: _sweep(c, True), options.iterations)
        hits = []
        for relocatable in (False, True):
            compiled = pulse.compile(_drives(count, relocatable))
            q = _HANDLES[3]
            hits.append(
                1000 * _median_ms(lambda c=compiled, q=q: c.lower(q, 0.1), 50 * options.iterations)
            )
        speedup = static_ms / runtime_ms
        print(
            f"  {count:6d}  {static_ms:16.2f}  {runtime_ms:17.2f}  {speedup:7.1f}x"
            f"  {hits[0]:15.2f}  {hits[1]:16.2f}"
        )
        if speedup < options.min_speedup:
            slow.append((count, speedup))
    for count, speedup in slow:
        print(f"  FAIL {count} drives: a runtime qudit sweep is only {speedup:.1f}x cheaper")
    return 1 if slow else 0


if __name__ == "__main__":
    raise SystemExit(main())
