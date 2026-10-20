#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Target-aware lowering demo on the Krinner target.

  kernel -> pulse.compile (verify, virtual-Z, fusion, ALAP) -> Pulse MLIR
         -> QOp (operator algebra) -> CuDensityMat -> LLVM

Prints the intermediate representations of the native lowering stack.
``pulse.evolve`` runs the same stack, adding the target's calibration
(drive scales, couplings, T1/T2) as module attributes, and executes it.
"""

import math

import cudaq_pulse as pulse
from cudaq_pulse._native._cudaq_pulse_native import MLIRPipeline
from cudaq_pulse.targets import transmon_krinner_17q

target = transmon_krinner_17q()

qubit_0_info = target.qubits[0]
qubit_1_info = target.qubits[1]
drive_params_0 = target.get_drive_params(0)
drive_params_1 = target.get_drive_params(1)


@pulse.kernel
def krinner_bell(qubit_0, qubit_1):
    drive_line_0, tone_0 = get_drive_line(qubit_0)
    drive_line_1, tone_1 = get_drive_line(qubit_1)

    # Hadamard on Q0
    shift_phase(tone_0, math.pi / 2)
    sx_pulse = drag(
        20, drive_params_0["x_amp"], drive_params_0["x_sigma"], drive_params_0["x_beta"]
    )
    drive(drive_line_0, sx_pulse, tone_0)
    shift_phase(tone_0, math.pi / 2)

    # Sync
    sync(drive_line_0, drive_line_1)

    # Echoed CR (CZ-like)
    cr = gaussian(98, 0.32, 24.0)
    drive(drive_line_0, cr, tone_1)
    x_echo = drag(20, drive_params_1["x_amp"], drive_params_1["x_sigma"], drive_params_1["x_beta"])
    drive(drive_line_1, x_echo, tone_1)
    cr_neg = gaussian(98, -0.32, 24.0)
    drive(drive_line_0, cr_neg, tone_1)
    drive(drive_line_1, x_echo, tone_1)


print("=== Stage 1: compile (verify, virtual-Z, fusion, ALAP schedule) ===")
compiled = pulse.compile(
    krinner_bell,
    [pulse.qudit_ref(), pulse.qudit_ref()],
    qubit_freq_hz={0: qubit_0_info.frequency_hz, 1: qubit_1_info.frequency_hz},
)
total_vtu = max(e.start_vtu + e.duration_vtu for e in compiled.events())
print(f"  {compiled.metrics.op_count} ops, {total_vtu / 2.0:.0f} ns total")

print("\n=== Stage 2: scheduled Pulse MLIR ===")
print(compiled.mlir)

print("=== Stage 3: operator (QOp) lowering ===")
qop = MLIRPipeline().run_pulse_to_qop(compiled.mlir)
print(f"  {qop.count('qop.callback_scalar')} time-dependent drive terms")

print("\n=== Stage 4: LLVM IR for the cuDensityMat runtime ===")
llvm = compiled.lower_to_llvm()
print(f"  {llvm.count('llvm.call')} runtime calls")
