#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QEC scaling benchmark: compile time vs distance for surface-code-like programs.

Compiles a synthetic surface-code syndrome extraction cycle at several code
distances (trace, module build, verify, ALAP schedule; no optimization
passes) and prints each stage's time and its growth relative to the smallest
distance, to compare against the growth in qubit and op count.

Usage:
  python benchmarks/simulation/qec_scaling.py [--distances D ...] [--repeats N]
"""

from __future__ import annotations

import argparse
import statistics

import cudaq_pulse as pulse


def _device(frequencies: dict[int, float]) -> pulse.QuantumSystem:
    """A device with these drive frequencies, indexed by qudit."""
    from cudaq_pulse.targets import DeviceDescription, PhysicalQudit

    qudits = {q: PhysicalQudit(q, f, -300e6, 50.0, 30.0) for q, f in frequencies.items()}
    return pulse.quantum_system(DeviceDescription(name="bench", qudits=qudits))


_STAGES = ("trace_ms", "ffi_ms", "passes_ms", "schedule_ms", "total_ms")


def surface_code_cycle(d: int):
    """A d-distance surface-code-like syndrome extraction kernel.

    Layout:
      d^2 data qubits + (d^2 - 1) ancilla qubits = 2d^2 - 1 total
      Each ancilla interacts with up to 4 data neighbors via echoed CR, and
      Z-type ancillas are wrapped in Hadamards (phase, SX, phase).
    """
    data_count = d * d
    ancilla_count = d * d - 1
    total_count = data_count + ancilla_count
    neighbors = []
    is_z = []
    for anc_idx in range(ancilla_count):
        row, col = anc_idx // (d - 1), anc_idx % (d - 1)
        is_z.append((row + col) % 2 == 0)
        neighbors.append(
            [
                (row + dr) * d + (col + dc)
                for dr, dc in ((0, 0), (0, 1), (1, 0), (1, 1))
                if row + dr < d and col + dc < d
            ]
        )

    @pulse.kernel
    def surface(qudits):
        lines = []
        tones = []
        for q in range(total_count):
            dl = get_drive_line(qudits[q])
            lines.append(dl)
            tones.append(dl.tone)
        for anc_idx in range(ancilla_count):
            anc = data_count + anc_idx
            if is_z[anc_idx]:
                shift_phase(tones[anc], 1.5708)
                drive(lines[anc], drag(20, 0.44, 5.0, 0.5), tone=tones[anc])
                shift_phase(tones[anc], 1.5708)
            row = neighbors[anc_idx]
            for j in range(len(row)):
                data_q = row[j]
                drive(lines[data_q], gaussian(98, 0.32, 24.5), tone=tones[data_q])
                drive(lines[anc], drag(20, 0.44, 5.0, 0.5), tone=tones[anc])
            if is_z[anc_idx]:
                shift_phase(tones[anc], 1.5708)
                drive(lines[anc], drag(20, 0.44, 5.0, 0.5), tone=tones[anc])
                shift_phase(tones[anc], 1.5708)

    return surface, total_count


def bench(d: int, repeats: int) -> tuple[int, int, dict[str, float]]:
    kernel, qudit_count = surface_code_cycle(d)
    quantum_system = _device({q: 5.0e9 + 0.05e9 * (q % 8) for q in range(qudit_count)})

    def compile_once() -> pulse.Lowered:
        # A fresh compiled kernel holds no specializations: this compiles.
        return pulse.compile(kernel, passes=("verify",)).lower(quantum_system.qudits[:qudit_count])

    op_count = compile_once().metrics.op_count
    samples: dict[str, list[float]] = {stage: [] for stage in _STAGES}
    for _ in range(repeats):
        metrics = compile_once().metrics
        for stage in _STAGES:
            samples[stage].append(getattr(metrics, stage))
    return qudit_count, op_count, {stage: statistics.median(xs) for stage, xs in samples.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="QEC compile-time scaling benchmark")
    parser.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7, 9, 11, 15, 21])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if any(d < 2 for d in args.distances):
        parser.error("distances must be at least 2")

    labels = ("Trace", "Build", "Verify", "Sched", "Total")
    print(
        f"  {'d':>3s}  {'Qubits':>7s}  {'Ops':>7s}"
        + "".join(f"  {s + ' (ms)':>11s}" for s in labels)
    )
    print(f"  {'---':>3s}  {'-' * 7:>7s}  {'-' * 7:>7s}" + f"  {'-' * 11}" * len(labels))

    rows = []
    for d in sorted(args.distances):
        qudit_count, op_count, stages = bench(d, args.repeats)
        rows.append((d, qudit_count, op_count, stages["total_ms"]))
        print(
            f"  {d:3d}  {qudit_count:7d}  {op_count:7d}"
            + "".join(f"  {stages[s]:11.2f}" for s in _STAGES)
        )

    base_d, base_q, base_ops, base_t = rows[0]
    print(f"\n  Scaling relative to d={base_d} ({base_q} qubits, {base_ops} ops, {base_t:.2f} ms):")
    print(
        f"  {'d':>3s}  {'Qubits':>7s}  {'Q ratio':>8s}  {'Ops':>7s}  {'Op ratio':>9s}  {'Time ratio':>10s}"
    )
    print(
        f"  {'---':>3s}  {'-------':>7s}  {'--------':>8s}  {'-------':>7s}  {'---------':>9s}  {'----------':>10s}"
    )
    for d, nq, nops, t in rows:
        print(
            f"  {d:3d}  {nq:7d}  {nq / base_q:8.1f}x  {nops:7d}  {nops / base_ops:9.1f}x  {t / base_t:10.1f}x"
        )


if __name__ == "__main__":
    main()
