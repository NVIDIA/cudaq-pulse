# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Compile-time benchmark: cudaq-pulse on canonical workloads.

Usage:
    python benchmarks/compiler/compile_time.py [--iterations N]
        [--baseline FILE [--gate FACTOR]] [--write-baseline FILE]

Compiles each of the 6 canonical workloads with ``pulse.compile(k).lower(...)`` (the
default passes, ALAP schedule). The op count is that of the traced kernel
IR, with loops retained. With --baseline, exits 1 if any op count differs
from the baseline; adding --gate also fails any workload slower than its
baseline time x FACTOR.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cudaq_pulse as pulse


def _device(frequencies: dict[int, float]) -> pulse.Device:
    """A device with these drive frequencies, indexed by qudit."""
    from cudaq_pulse.targets import Qubit, TargetDescription

    qubits = {q: Qubit(q, f, -300e6, 50.0, 30.0) for q, f in frequencies.items()}
    return pulse.device({"target": TargetDescription(name="bench", qubits=qubits)})


@dataclass
class BenchResult:
    name: str
    ms: float
    op_count: int


# ── Workload kernels ─────────────────────────────────────────────────────────


@pulse.kernel
def bell(q0, q1):
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    shift_phase(t0, math.pi / 2)
    sx = drag(40, 0.25, 10.0, 0.5)
    drive(d0, sx, t0)
    shift_phase(t0, math.pi / 2)
    sync(d0, d1)
    drive(d1, sx, t1)
    cr = gaussian(200, 0.10, 50.0)
    drive(d0, cr, t1)
    x = square(40, 0.047 + 0j)
    drive(d0, x, t0)
    cr_neg = gaussian(200, -0.10, 50.0)
    drive(d0, cr_neg, t1)
    drive(d1, sx, t1)


@pulse.kernel
def cnot_cr(q0, q1):
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    sync(d0, d1)
    sx = drag(40, 0.025, 10.0, 0.5)
    cr = gaussian(200, 0.10, 50.0)
    cr_neg = gaussian(200, -0.10, 50.0)
    x = square(40, 0.047 + 0j)
    drive(d1, sx, t1)
    drive(d0, cr, t1)
    drive(d0, x, t0)
    drive(d0, cr_neg, t1)
    drive(d1, sx, t1)


@pulse.kernel
def qaoa4(q0, q1, q2, q3):
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    d2, t2 = get_drive_line(q2)
    d3, t3 = get_drive_line(q3)
    lines = [d0, d1, d2, d3]
    tones = [t0, t1, t2, t3]

    # Edge (0, 1)
    sync(lines[0], lines[1])
    drive(lines[1], drag(40, 0.025, 10.0, 0.5), tones[1])
    drive(lines[0], gaussian(200, 0.10, 50.0), tones[1])
    drive(lines[0], square(40, 0.047 + 0j), tones[0])
    drive(lines[0], gaussian(200, -0.10, 50.0), tones[1])
    drive(lines[1], drag(40, 0.025, 10.0, 0.5), tones[1])
    shift_phase(tones[1], 0.6)

    # Edge (1, 2)
    sync(lines[1], lines[2])
    drive(lines[2], drag(40, 0.025, 10.0, 0.5), tones[2])
    drive(lines[1], gaussian(200, 0.10, 50.0), tones[2])
    drive(lines[1], square(40, 0.047 + 0j), tones[1])
    drive(lines[1], gaussian(200, -0.10, 50.0), tones[2])
    drive(lines[2], drag(40, 0.025, 10.0, 0.5), tones[2])
    shift_phase(tones[2], 0.6)

    # Edge (2, 3)
    sync(lines[2], lines[3])
    drive(lines[3], drag(40, 0.025, 10.0, 0.5), tones[3])
    drive(lines[2], gaussian(200, 0.10, 50.0), tones[3])
    drive(lines[2], square(40, 0.047 + 0j), tones[2])
    drive(lines[2], gaussian(200, -0.10, 50.0), tones[3])
    drive(lines[3], drag(40, 0.025, 10.0, 0.5), tones[3])
    shift_phase(tones[3], 0.6)

    # Edge (3, 0)
    sync(lines[3], lines[0])
    drive(lines[0], drag(40, 0.025, 10.0, 0.5), tones[0])
    drive(lines[3], gaussian(200, 0.10, 50.0), tones[0])
    drive(lines[3], square(40, 0.047 + 0j), tones[3])
    drive(lines[3], gaussian(200, -0.10, 50.0), tones[0])
    drive(lines[0], drag(40, 0.025, 10.0, 0.5), tones[0])
    shift_phase(tones[0], 0.6)


@pulse.kernel
def syndrome(q0, q1, q2, q3, q4):
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    d2, t2 = get_drive_line(q2)
    d3, t3 = get_drive_line(q3)
    d4, t4 = get_drive_line(q4)
    lines = [d0, d1, d2, d3, d4]
    tones = [t0, t1, t2, t3, t4]

    sx = drag(40, 0.25, 10.0, 0.5)
    shift_phase(tones[0], math.pi / 2)
    drive(lines[0], sx, tones[0])
    shift_phase(tones[0], math.pi / 2)

    # data = 1
    sync(lines[0], lines[1])
    drive(lines[1], drag(40, 0.025, 10.0, 0.5), tones[1])
    drive(lines[0], gaussian(200, 0.10, 50.0), tones[1])
    drive(lines[0], square(40, 0.047 + 0j), tones[0])
    drive(lines[0], gaussian(200, -0.10, 50.0), tones[1])
    drive(lines[1], drag(40, 0.025, 10.0, 0.5), tones[1])

    # data = 2
    sync(lines[0], lines[2])
    drive(lines[2], drag(40, 0.025, 10.0, 0.5), tones[2])
    drive(lines[0], gaussian(200, 0.10, 50.0), tones[2])
    drive(lines[0], square(40, 0.047 + 0j), tones[0])
    drive(lines[0], gaussian(200, -0.10, 50.0), tones[2])
    drive(lines[2], drag(40, 0.025, 10.0, 0.5), tones[2])

    # data = 3
    sync(lines[0], lines[3])
    drive(lines[3], drag(40, 0.025, 10.0, 0.5), tones[3])
    drive(lines[0], gaussian(200, 0.10, 50.0), tones[3])
    drive(lines[0], square(40, 0.047 + 0j), tones[0])
    drive(lines[0], gaussian(200, -0.10, 50.0), tones[3])
    drive(lines[3], drag(40, 0.025, 10.0, 0.5), tones[3])

    # data = 4
    sync(lines[0], lines[4])
    drive(lines[4], drag(40, 0.025, 10.0, 0.5), tones[4])
    drive(lines[0], gaussian(200, 0.10, 50.0), tones[4])
    drive(lines[0], square(40, 0.047 + 0j), tones[0])
    drive(lines[0], gaussian(200, -0.10, 50.0), tones[4])
    drive(lines[4], drag(40, 0.025, 10.0, 0.5), tones[4])


@pulse.kernel
def dd_cpmg8(q0):
    d0, t0 = get_drive_line(q0)
    x = square(40, 0.047 + 0j)
    wait(d0, 100)
    for _ in range(8):
        drive(d0, x, t0)
        wait(d0, 200)


@pulse.kernel
def vqe_hea(q0, q1, q2, q3):
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    d2, t2 = get_drive_line(q2)
    d3, t3 = get_drive_line(q3)
    lines = [d0, d1, d2, d3]
    tones = [t0, t1, t2, t3]

    sx = drag(40, 0.25, 10.0, 0.5)
    thetas = [0.3, 0.5, 0.7, 0.9]

    for q in range(4):
        shift_phase(tones[q], -math.pi / 2)
        drive(lines[q], sx, tones[q])
        shift_phase(tones[q], thetas[q])
        drive(lines[q], sx, tones[q])
        shift_phase(tones[q], math.pi / 2)

    # Edge (0, 1)
    sync(lines[0], lines[1])
    drive(lines[1], drag(40, 0.025, 10.0, 0.5), tones[1])
    drive(lines[0], gaussian(200, 0.10, 50.0), tones[1])
    drive(lines[0], square(40, 0.047 + 0j), tones[0])
    drive(lines[0], gaussian(200, -0.10, 50.0), tones[1])
    drive(lines[1], drag(40, 0.025, 10.0, 0.5), tones[1])

    # Edge (1, 2)
    sync(lines[1], lines[2])
    drive(lines[2], drag(40, 0.025, 10.0, 0.5), tones[2])
    drive(lines[1], gaussian(200, 0.10, 50.0), tones[2])
    drive(lines[1], square(40, 0.047 + 0j), tones[1])
    drive(lines[1], gaussian(200, -0.10, 50.0), tones[2])
    drive(lines[2], drag(40, 0.025, 10.0, 0.5), tones[2])

    # Edge (2, 3)
    sync(lines[2], lines[3])
    drive(lines[3], drag(40, 0.025, 10.0, 0.5), tones[3])
    drive(lines[2], gaussian(200, 0.10, 50.0), tones[3])
    drive(lines[2], square(40, 0.047 + 0j), tones[2])
    drive(lines[2], gaussian(200, -0.10, 50.0), tones[3])
    drive(lines[3], drag(40, 0.025, 10.0, 0.5), tones[3])


# ── Workload registry: (name, kernel_fn, n_qubits, freq_dict) ───────────────

WORKLOADS = [
    ("bell", bell, 2, {0: 5.0e9, 1: 5.1e9}),
    ("cnot_cr", cnot_cr, 2, {0: 5.0e9, 1: 5.1e9}),
    ("qaoa4", qaoa4, 4, {i: 5.0e9 + 0.1e9 * i for i in range(4)}),
    ("syndrome", syndrome, 5, {i: 5.0e9 + 0.1e9 * i for i in range(5)}),
    ("dd_cpmg8", dd_cpmg8, 1, {0: 5.0e9}),
    ("vqe_hea", vqe_hea, 4, {i: 5.0e9 + 0.1e9 * i for i in range(4)}),
]


def _bench(kernel_fn, n_qubits: int, freq_dict: dict, iterations: int) -> tuple[float, int]:
    device = _device(freq_dict)
    qubits = device.qudit_handles[:n_qubits]
    op_count = pulse.compile(kernel_fn).lower(*qubits).metrics.op_count
    total = 0.0
    for _ in range(iterations):
        t0 = time.monotonic()
        # A fresh compiled kernel holds no specializations: this compiles.
        pulse.compile(kernel_fn).lower(*qubits)
        total += time.monotonic() - t0
    return (total / iterations) * 1000.0, op_count


def _check_baseline(
    results: list[BenchResult], baseline_path: Path, gate: float | None
) -> list[str]:
    baseline = json.loads(baseline_path.read_text())
    failures = []
    measured = {r.name: r for r in results}
    for name in sorted(set(baseline) | set(measured)):
        if name not in measured:
            failures.append(f"{name}: in baseline but not benchmarked")
            continue
        if name not in baseline:
            failures.append(f"{name}: benchmarked but missing from baseline")
            continue
        result, expected = measured[name], baseline[name]
        if result.op_count != expected["op_count"]:
            failures.append(
                f"{name}: op count {result.op_count} != baseline {expected['op_count']}"
            )
        if gate is not None and result.ms > expected["ms"] * gate:
            failures.append(
                f"{name}: {result.ms:.3f} ms exceeds baseline {expected['ms']:.3f} ms x {gate}"
            )
    return failures


def main():
    parser = argparse.ArgumentParser(description="Compile-time benchmark")
    parser.add_argument(
        "--baseline", type=Path, help="Fail if op counts differ from this baseline JSON"
    )
    parser.add_argument(
        "--gate",
        type=float,
        help="Also fail if a workload is slower than baseline x GATE (requires --baseline)",
    )
    parser.add_argument(
        "--write-baseline", type=Path, help="Write the measured results to this JSON file"
    )
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if args.gate is not None and args.baseline is None:
        parser.error("--gate requires --baseline")

    print(f"Running {args.iterations} iterations per workload...\n")
    print(f"  {'Workload':20s}  {'Time (ms)':>10s}  {'Ops':>6s}")
    print(f"  {'-' * 20}  {'-' * 10}  {'-' * 6}")

    results: list[BenchResult] = []
    for name, kernel_fn, n_qubits, freq_dict in WORKLOADS:
        ms, ops = _bench(kernel_fn, n_qubits, freq_dict, args.iterations)
        results.append(BenchResult(name=name, ms=ms, op_count=ops))
        print(f"  {name:20s}  {ms:10.3f}  {ops:6d}")

    print("\n--- Summary ---")
    for r in results:
        print(f"  {r.name}: {r.ms:.3f} ms ({r.op_count} ops)")

    print(f"\nAll {len(results)} workloads compiled successfully.")

    if args.write_baseline:
        args.write_baseline.write_text(
            json.dumps(
                {r.name: {"ms": round(r.ms, 3), "op_count": r.op_count} for r in results}, indent=2
            )
            + "\n"
        )
        print(f"Wrote baseline to {args.write_baseline}")

    if args.baseline:
        failures = _check_baseline(results, args.baseline, args.gate)
        if failures:
            print(f"\nRegression against {args.baseline}:")
            for failure in failures:
                print(f"  FAIL {failure}")
            sys.exit(1)
        checked = "op counts and timing" if args.gate else "op counts"
        print(f"Baseline {args.baseline}: {checked} match.")


if __name__ == "__main__":
    main()
