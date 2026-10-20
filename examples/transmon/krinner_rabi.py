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
  - Using per-qubit drive parameters from the target
  - Compiling (verify, virtual-Z, fusion, ALAP schedule) a single-qubit
    pulse program
  - Lowering it to the operator (QOp) dialect the simulator consumes
"""

import cudaq_pulse as pulse
from cudaq_pulse._native._cudaq_pulse_native import MLIRPipeline
from cudaq_pulse.targets import transmon_krinner_17q

target = transmon_krinner_17q()
qubit_idx = 4  # D5 -- central data qubit

qubit_d5 = target.qubits[qubit_idx]
drive_params = target.get_drive_params(qubit_idx)

print(f"Target: {target.name} ({target.n_qubits} qubits)")
print(f"Qubit D5 (idx {qubit_idx}):")
print(f"  frequency = {qubit_d5.frequency_hz / 1e9:.3f} GHz")
print(f"  anharmonicity = {qubit_d5.anharmonicity_hz / 1e6:.1f} MHz")
print(f"  T1 = {qubit_d5.t1_us:.1f} us, T2* = {qubit_d5.t2_star_us:.1f} us")
print(
    f"  DRAG params: amp={drive_params['x_amp']:.3f}, "
    f"sigma={drive_params['x_sigma']:.1f}, "
    f"beta={drive_params['x_beta']:.2f}"
)

amplitude = drive_params["x_amp"]
sigma = drive_params["x_sigma"]
beta = drive_params["x_beta"]
duration = drive_params["x_dur"]
n_rabi_points = 21


@pulse.kernel
def rabi_d5(qubit):
    drive_line, tone = get_drive_line(qubit)
    for i in range(n_rabi_points):
        scale = i / (n_rabi_points - 1)
        rabi_pulse = drag(duration, amplitude * scale, sigma, beta)
        drive(drive_line, rabi_pulse, tone)
        wait(drive_line, 100)


compiled = pulse.compile(
    rabi_d5, [pulse.qudit_ref(qubit_idx)], qubit_freq_hz={qubit_idx: qubit_d5.frequency_hz}
)
total_vtu = max(e.start_vtu + e.duration_vtu for e in compiled.events())
print(f"\nCompiled {compiled.metrics.op_count} ops, total length = {total_vtu / 2.0:.1f} ns")

qop = MLIRPipeline().run_pulse_to_qop(compiled.mlir)
print(f"Operator program: {qop.count('qop.callback_scalar')} time-dependent drive terms")
