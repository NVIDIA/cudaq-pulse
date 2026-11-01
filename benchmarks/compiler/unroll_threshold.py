# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Rolled against unrolled loops: the measurement behind the unroll threshold.

Usage:
    python benchmarks/compiler/unroll_threshold.py [--no-sim]

For loops of 1 to 512 iterations whose bodies hold 2, 3 or 6 calls, times
compiling (a fresh ``pulse.compile(k, unroll_threshold=...).lower(q)``:
trace, lowering, passes, schedule) and simulating (CPU backend) both forms,
and prints the estimated size (trip count x calls), the medians and the
unrolled / rolled ratio of the total. The default threshold is the size
where the ratio crosses 1.
"""

from __future__ import annotations

import gc
import statistics
import sys
import time

import cudaq_pulse as pulse
from cudaq_pulse.targets import Qubit, TargetDescription

TARGET = TargetDescription(
    name="one",
    qubits={0: Qubit(0, 5.0e9, -300e6, 0.0, 0.0, drive_params={"amplitude_scale_rad_per_ns": 1.0})},
)
Q = pulse.device({"target": TARGET}).qudit_handles[0]
UNROLL_ALL = 10**9


def make(calls, trips):
    """A loop of *trips* iterations whose body makes *calls* calls."""
    if calls == 2:

        @pulse.kernel
        def k(q):
            d = get_drive_line(q)
            for _ in range(trips):
                drive(d, square(4, 0.01))

    elif calls == 3:

        @pulse.kernel
        def k(q):
            d = get_drive_line(q)
            for _ in range(trips):
                drive(d, square(4, 0.01))
                wait(d, 2)

    else:

        @pulse.kernel
        def k(q):
            d = get_drive_line(q)
            for _ in range(trips):
                drive(d, square(4, 0.01))
                shift_phase(d.tone, 0.1)
                drive(d, square(4, 0.01))
                wait(d, 2)

    return k


def measure(kernel, threshold, reps, simulate):
    compile_ms, simulate_ms = [], []
    for _ in range(reps):
        gc.collect()
        t0 = time.perf_counter()
        lowered = pulse.compile(kernel, unroll_threshold=threshold).lower(Q)
        t1 = time.perf_counter()
        compile_ms.append((t1 - t0) * 1e3)
        if simulate:
            pulse.simulate(lowered, dt_ns=0.25, levels=2)
            simulate_ms.append((time.perf_counter() - t1) * 1e3)
    return statistics.median(compile_ms), statistics.median(simulate_ms) if simulate else 0.0


def main() -> int:
    simulate = "--no-sim" not in sys.argv
    print(
        f"{'calls':>5} {'trips':>5} {'size':>5} {'compile r/u ms':>16} {'simulate r/u ms':>18} {'u/r':>6}"
    )
    for calls in (2, 3, 6):
        for trips in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
            kernel = make(calls, trips)
            reps = 15 if trips <= 64 else 5
            measure(kernel, 0, 2, simulate)
            measure(kernel, UNROLL_ALL, 2, simulate)
            rc, rs = measure(kernel, 0, reps, simulate)
            uc, us = measure(kernel, UNROLL_ALL, reps, simulate)
            print(
                f"{calls:5d} {trips:5d} {calls * trips:5d} {rc:7.3f}/{uc:8.3f} "
                f"{rs:8.2f}/{us:9.2f} {(uc + us) / (rc + rs):6.2f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
