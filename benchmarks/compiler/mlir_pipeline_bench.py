#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Benchmarks for the cudaq-pulse MLIR pipeline.

Measures the median wall-clock time of each stage of compiling a kernel:
  1. trace: kernel to packed buffer
  2. build: packed buffer to in-memory MLIR module (one FFI call)
  3. passes: the default pipeline, and each pass on its own
  4. schedule: ALAP
  5. lowering: pulse -> qop -> cudm -> LLVM dialect

and reports the size of the scheduled MLIR and of the lowered LLVM dialect.

Usage:
  python benchmarks/compiler/mlir_pipeline_bench.py [--repeats N]
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from collections.abc import Callable
from typing import Any

import cudaq_pulse as pulse


def _device(frequencies: dict[int, float]) -> pulse.Device:
    """A device with these drive frequencies, indexed by qudit."""
    from cudaq_pulse.targets import Qubit, TargetDescription

    qubits = {q: Qubit(q, f, -300e6, 50.0, 30.0) for q, f in frequencies.items()}
    return pulse.device({"target": TargetDescription(name="bench", qubits=qubits)})


_STAGES = ("trace_ms", "ffi_ms", "passes_ms", "schedule_ms", "total_ms")
_PASSES = tuple(pulse.Pass)


def _median_ms(fn: Callable[[], Any], repeats: int) -> float:
    fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


# --- 1Q Rabi kernel ---
@pulse.kernel
def rabi_1q(q0):
    d0 = get_drive_line(q0)
    wf = gaussian(100, 0.1, 25.0)
    drive(d0, wf)


# --- 2Q CR-CNOT kernel ---
@pulse.kernel
def cr_cnot_2q(q0, q1):
    d0 = get_drive_line(q0)
    d1 = get_drive_line(q1)
    x90 = gaussian(40, 0.25, 10.0)
    drive(d0, x90)
    sync(d0, d1)
    cr = square(160, 0.05)
    drive(d0, cr)
    sync(d0, d1)
    drive(d1, x90)


# --- Echo DD kernel ---
@pulse.kernel
def echo_dd(q0):
    d0 = get_drive_line(q0)
    for _ in range(10):
        wf = gaussian(40, 0.3, 10.0)
        drive(d0, wf)
        wait(d0, 100)
        shift_phase(d0.tone, math.pi)
        wf_neg = gaussian(40, 0.3, 10.0)
        drive(d0, wf_neg)
        wait(d0, 100)


BENCHMARKS = [
    ("1Q Rabi", rabi_1q, 1),
    ("2Q CR-CNOT", cr_cnot_2q, 2),
    ("Echo DD (10x)", echo_dd, 1),
]


def _compile(kernel, n_qubits: int, **kwargs: Any) -> pulse.Lowered:
    device = _device({i: 5.0e9 + i * 0.1e9 for i in range(n_qubits)})
    # A fresh compiled kernel holds no specializations: this compiles.
    return pulse.compile(kernel, **kwargs).lower(*device.qudit_handles[:n_qubits])


def bench_stages(kernel, n_qubits: int, repeats: int) -> dict[str, float]:
    _compile(kernel, n_qubits)
    samples: dict[str, list[float]] = {stage: [] for stage in _STAGES}
    for _ in range(repeats):
        metrics = _compile(kernel, n_qubits).metrics
        for stage in _STAGES:
            samples[stage].append(getattr(metrics, stage))
    return {stage: statistics.median(xs) for stage, xs in samples.items()}


def bench_pass(kernel, n_qubits: int, name: str, repeats: int) -> float:
    return statistics.median(
        _compile(kernel, n_qubits, passes=(name,)).metrics.passes_ms for _ in range(repeats)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MLIR pipeline benchmark")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    print("=" * 72)
    print("cudaq-pulse MLIR Pipeline Benchmarks")
    print("=" * 72)

    print("\n--- Stage Timings (median ms) ---")
    print(f"  {'Kernel':16s}" + "".join(f"  {s.removesuffix('_ms'):>9s}" for s in _STAGES))
    for name, kernel, n in BENCHMARKS:
        stages = bench_stages(kernel, n, args.repeats)
        print(f"  {name:16s}" + "".join(f"  {stages[s]:9.3f}" for s in _STAGES))

    print("\n--- Per-Pass Timings (median ms) ---")
    print(f"  {'Pass':22s}" + "".join(f"  {name:>14s}" for name, _, _ in BENCHMARKS))
    for p in _PASSES:
        row = [bench_pass(kernel, n, p, args.repeats) for _, kernel, n in BENCHMARKS]
        print(f"  {p.value:22s}" + "".join(f"  {ms:14.3f}" for ms in row))

    print("\n--- Lowering to LLVM Dialect (median ms) ---")
    for name, kernel, n in BENCHMARKS:
        compiled = _compile(kernel, n)
        ms = _median_ms(lambda: compiled.mlir("llvm"), args.repeats)
        print(f"  {name:16s}  {ms:9.3f}")

    print("\n--- IR Size ---")
    for name, kernel, n in BENCHMARKS:
        compiled = _compile(kernel, n)
        mlir = compiled.mlir()
        llvm = compiled.mlir("llvm")
        print(
            f"  {name:16s}  pulse: {mlir.count(chr(10)):5d} lines, {len(mlir):7d} chars"
            f"   llvm: {llvm.count(chr(10)):5d} lines, {len(llvm):7d} chars"
        )

    print("\n--- Program Statistics ---")
    for name, kernel, n in BENCHMARKS:
        compiled = _compile(kernel, n)
        events = compiled.schedule
        makespan = max(e.start_cycles + e.duration_cycles for e in events)
        print(
            f"  {name:16s}  {compiled.metrics.op_count:3d} ops, {n} qubits, "
            f"{len(events):3d} timed events, makespan {makespan} cycles"
        )

    print("\n" + "=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()
