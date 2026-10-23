#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""End-to-end GPU time evolution on the Krinner 17-qubit target.

HARDWARE REQUIREMENT: This example requires an NVIDIA GPU and
`pip install "cudaq-pulse[gpu]"`. If either is missing it raises an error
naming which one -- no silent fallback.

Pipeline:
  @pulse.kernel -> kernel.lower(handles) -> pulse.simulate(backend="gpu")
                -> native MLIR lowering -> JIT compile -> cuDensityMat GPU execution
"""

import math

import cudaq_pulse as pulse
import numpy as np
from cudaq_pulse.runtime.gpu import require_gpu_runtime

require_gpu_runtime()

dev = pulse.get_device(pulse.Target.TRANSMON_KRINNER_17Q)
target = dev.target

qubit_0_info = target.qubits[0]
qubit_1_info = target.qubits[1]
drive_params_0 = target.get_drive_params(0)
drive_params_1 = target.get_drive_params(1)


@pulse.kernel
def krinner_evolve(qubit_0, qubit_1):
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


lowered = krinner_evolve.lower(dev.qudit_handles[0], dev.qudit_handles[1])
# This sequence is shorter than 128 ns. The remaining interval evolves under
# the target's always-on coupling and T1/T2 model.
result = pulse.simulate(lowered, backend="gpu", t_end=128.0, dt_ns=0.25, integrator="rk4")

state = result.density_matrix
print(f"=== Final state shape: {state.shape} ===")
if state.ndim == 2:
    populations = np.real(np.diag(state))
    print(f"Density-matrix trace: {np.trace(state):.8f}")
else:
    populations = np.abs(state) ** 2

for basis_index, probability in enumerate(populations):
    if probability > 1.0e-6:
        print(f"  |{basis_index:02b}> : P = {probability:.6f}")
