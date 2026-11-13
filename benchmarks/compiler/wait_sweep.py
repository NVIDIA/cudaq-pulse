#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A T1 sweep over its wait length.

A ``float`` wait length is a runtime parameter: the kernel compiles once and
each call binds the value. For a sweep of ``--values`` lengths this prints
the compiles, the median cost of ``lower()`` with a new length and with a
cached one, the wall time and peak traced memory of the sweep, and the median
time of simulated calls with a new length and with a repeated one. Exits 1 if
the sweep compiles more than once.

Usage:
    python benchmarks/compiler/wait_sweep.py [--values N] [--simulated N] [--json]
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
import tracemalloc
import warnings

import cudaq_pulse as pulse
from cudaq_pulse.targets import DeviceDescription, MicrowaveDrive, Qubit

_TARGET = DeviceDescription(
    name="wait-sweep",
    qubits={
        0: Qubit(
            index=0,
            frequency_hz=5.0e9,
            anharmonicity_hz=-200.0e6,
            t1_us=0.2,
            t2_star_us=0.1,
            drive=MicrowaveDrive(amplitude_scale_rad_per_ns=1.0),
        )
    },
)
_QUBIT = pulse.device(_TARGET).qudits[0]


@pulse.kernel
def t1(q, tau: float):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, square(40, 0.157))
    wait(d, tau)
    sync(d, r)
    readout(r, square(40, 0.0))


def _median_us(calls) -> float:
    times = []
    for call in calls:
        start = time.perf_counter()
        call()
        times.append((time.perf_counter() - start) * 1e6)
    return statistics.median(times)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--values", type=int, default=1000)
    parser.add_argument("--simulated", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    warnings.simplefilter("ignore", RuntimeWarning)
    taus = [float(4 * i + 1) for i in range(args.values)]

    compiled = pulse.compile(t1)
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    for tau in taus:
        compiled.lower(_QUBIT, tau)
    sweep_ms = (time.perf_counter() - start) * 1e3
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    compiles = compiled.cache_info().compiled

    fresh = iter(float(4 * i + 3) for i in range(10**6))
    new_us = _median_us([lambda: compiled.lower(_QUBIT, next(fresh))] * 400)
    cached_us = _median_us([lambda: compiled.lower(_QUBIT, taus[0])] * 400)

    def simulate(tau: float) -> None:
        pulse.simulate(compiled.lower(_QUBIT, tau), levels=2, dt_ns=0.25)

    # New lengths within a few cycles of the repeated one, so that both
    # integrate about as many steps.
    simulate(400.0)
    near = [400.0 + (k // 2 + 1) * (-1) ** k for k in range(args.simulated)]
    sim_new_ms = _median_us([lambda t=t: simulate(t) for t in near]) / 1e3
    sim_cached_ms = _median_us([lambda: simulate(400.0)] * args.simulated) / 1e3

    result = {
        "values": args.values,
        "compiles": compiles,
        "lower_new_us": round(new_us, 2),
        "lower_cached_us": round(cached_us, 2),
        "sweep_ms": round(sweep_ms, 2),
        "sweep_peak_kib": round(peak / 1024, 1),
        "simulate_new_ms": round(sim_new_ms, 3),
        "simulate_cached_ms": round(sim_cached_ms, 3),
    }
    if args.json:
        print(json.dumps(result))
    else:
        for key, value in result.items():
            print(f"{key:>20}: {value}")
    return 0 if compiles == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
