#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rabi oscillation on qubit D5 using the 17-qubit transmon target.

Demonstrates:
  - Loading the pre-defined transmon target
  - Using the quantum system's X calibration, the published single-qubit pulse
    (40 ns DRAG, sigma = 10 ns), from get_calibration
  - Compiling (default passes, ALAP schedule) a single-qubit
    pulse program
  - Lowering it to the operator (QOp) dialect the simulator consumes
"""

import shutil
import subprocess

import cudaq_pulse as pulse
from cudaq_pulse.quantum_systems import transmon_surface17

target = transmon_surface17()
qs = pulse.quantum_system("transmon_surface17")
qudit_idx = 4  # D5 -- central data qubit

qudit_d5 = target.physical_qudits[qudit_idx]
clock_ghz = qs.clock_ghz  # the program clock the schedule counts cycles of

print(f"Target: {target.name} ({len(target.physical_qudits)} qudits)")
print(f"Qubit {qudit_d5.label} (idx {qudit_idx}):")
print(f"  frequency = {qudit_d5.frequency_hz / 1e9:.3f} GHz")
print(f"  anharmonicity = {qudit_d5.anharmonicity_hz / 1e6:.1f} MHz")
print(f"  T1 = {qudit_d5.t1_us:.1f} us, T2* = {qudit_d5.t2_star_us:.1f} us")

# The quantum system's X calibration: the published pulse shape, and (with no
# published amplitude, so amplitudes are rates in rad/ns) the amplitude that
# makes it a pi rotation.
x = qs.qudits[qudit_idx].get_calibration(pulse.gates.X)
print(
    f"  DRAG pi pulse: {x.duration_ns:.0f} ns, sigma = {x.sigma_ns:.0f} ns, amp = {x.amplitude:.4f}"
)

gap = 50 * pulse.ns
rabi_point_count = 21


@pulse.kernel
def rabi_d5(qudit):
    drive_line = get_drive_line(qudit)
    pi_pulse = drive_line.get_calibration(pulse.gates.X)
    for i in range(rabi_point_count):
        scale = 2 * i / (rabi_point_count - 1)
        drive(drive_line, pi_pulse.replace(amplitude=pi_pulse.amplitude * scale))
        wait(drive_line, gap)


compiled = pulse.compile(rabi_d5, quantum_system=qs).lower(qs.qudits[qudit_idx])
total_cycles = max(e.end for e in compiled.schedule)
print(
    f"\nCompiled {compiled.metrics.op_count} ops, total length = {total_cycles / clock_ghz:.1f} ns"
)

# cudaq-pulse-opt, installed with the package, runs any pass on MLIR text.
tool = shutil.which("cudaq-pulse-opt")
if tool is None:
    raise SystemExit(
        "cudaq-pulse-opt is not on PATH; it comes with cudaq-pulse, from the "
        f"cudaq-pulse-tools wheel: pip install cudaq-pulse-tools=={pulse.__version__}"
    )
qop = subprocess.run(
    [tool, "--pulse-to-qop"], input=compiled.mlir(), capture_output=True, text=True, check=True
).stdout
print(f"Operator program: {qop.count('qop.callback_scalar')} time-dependent drive terms")
