#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Rabi oscillation on qubit D5 using the Krinner 17-qubit target.

Demonstrates:
  - Loading the pre-defined transmon target
  - Using the published single-qubit pulse (40 ns DRAG, sigma = 10 ns)
  - Compiling (verify, virtual-Z, fusion, ALAP schedule) a single-qubit
    pulse program
  - Lowering it to the operator (QOp) dialect the simulator consumes
"""

import math

import cudaq_pulse as pulse
from cudaq_pulse._native._cudaq_pulse_native import MLIRPipeline
from cudaq_pulse.targets import transmon_krinner_17q

target = transmon_krinner_17q()
qubit_idx = 4  # D5 -- central data qubit

qubit_d5 = target.qubits[qubit_idx]
drive_params = target.get_drive_params(qubit_idx)
clock_ghz = target.clock_ghz

print(f"Target: {target.name} ({target.n_qubits} qubits)")
print(f"Qubit {qubit_d5.label} (idx {qubit_idx}):")
print(f"  frequency = {qubit_d5.frequency_hz / 1e9:.3f} GHz")
print(f"  anharmonicity = {qubit_d5.anharmonicity_hz / 1e6:.1f} MHz")
print(f"  T1 = {qubit_d5.t1_us:.1f} us, T2* = {qubit_d5.t2_star_us:.1f} us")

# The target gives the pulse shape; with no published amplitude, amplitudes
# are rates in rad/ns, and a pi pulse has area pi.
duration_ns = drive_params["x_dur"]
sigma_ns = drive_params["x_sigma"]
area_ns = sigma_ns * math.sqrt(2 * math.pi) * math.erf(duration_ns / (2 * math.sqrt(2) * sigma_ns))
pi_amplitude = math.pi / area_ns
print(f"  DRAG pi pulse: {duration_ns:.0f} ns, sigma = {sigma_ns:.0f} ns, amp = {pi_amplitude:.4f}")

duration = round(duration_ns * clock_ghz)
sigma = sigma_ns * clock_ghz
n_rabi_points = 21


@pulse.kernel
def rabi_d5(qubit):
    drive_line, tone = get_drive_line(qubit)
    for i in range(n_rabi_points):
        scale = 2 * i / (n_rabi_points - 1)
        rabi_pulse = drag(duration, pi_amplitude * scale, sigma, 0.0)
        drive(drive_line, rabi_pulse, tone)
        wait(drive_line, 100)


dev = pulse.get_device(pulse.Target.TRANSMON_KRINNER_17Q)
compiled = pulse.compile(rabi_d5, device=dev).lower(dev.qudit_handles[qubit_idx])
total_vtu = max(e.start_vtu + e.duration_vtu for e in compiled.schedule)
print(f"\nCompiled {compiled.metrics.op_count} ops, total length = {total_vtu / clock_ghz:.1f} ns")

qop = MLIRPipeline().run_pulse_to_qop(compiled.mlir())
print(f"Operator program: {qop.count('qop.callback_scalar')} time-dependent drive terms")
