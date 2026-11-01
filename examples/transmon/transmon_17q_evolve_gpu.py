#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""End-to-end GPU time evolution on the 17-qubit transmon target.

HARDWARE REQUIREMENT: This example requires an NVIDIA GPU and
the `[cu12]` or `[cu13]` extra. If either is missing it raises an error
naming which one -- no silent fallback.

Pipeline:
  @pulse.kernel -> kernel.lower(handles) -> pulse.simulate(backend="gpu")
                -> native MLIR lowering -> JIT compile -> cuDensityMat GPU execution

Auxiliary X1 and data qubit D2 are coupled, about 2.1 GHz apart: X1 gets a
pi/2 pulse and D2 a pi pulse, then both idle for a CZ slot (flux pulses are
not modeled yet) under the coupling and the T1/T2 of Tab. I. Only X1 and
D2 are simulated, a 4-dimensional space; the other 15 qudits are classical
spectators.
"""

import math

import cudaq_pulse as pulse
import numpy as np
from cudaq_pulse.runtime import require_gpu_runtime

require_gpu_runtime()

dev = pulse.device("transmon_17q")
target = dev.target
index = {q.label: q.index for q in target.qubits.values()}
auxiliary, data = index["X1"], index["D2"]

# The published 40 ns, sigma = 10 ns pulse; amplitudes are rates in rad/ns.
drive_params = target.get_drive_params(data)
duration_ns, sigma_ns = drive_params["x_dur"], drive_params["x_sigma"]
area_ns = sigma_ns * math.sqrt(2 * math.pi) * math.erf(duration_ns / (2 * math.sqrt(2) * sigma_ns))
pi_amplitude = math.pi / area_ns
cal = {
    "x_duration": duration_ns * pulse.ns,
    "x_sigma": sigma_ns * pulse.ns,
    "cz_slot": 98 * pulse.ns,
}


@pulse.kernel
def evolve_pair(qubit_aux, qubit_data):
    drive_line_aux = get_drive_line(qubit_aux)
    drive_line_data = get_drive_line(qubit_data)
    duration, sigma = cal["x_duration"], cal["x_sigma"]
    drive(drive_line_aux, drag(duration, pi_amplitude / 2, sigma, 0.0))
    drive(drive_line_data, drag(duration, pi_amplitude, sigma, 0.0))
    sync(drive_line_aux, drive_line_data)
    wait(drive_line_aux, cal["cz_slot"])
    wait(drive_line_data, cal["cz_slot"])


lowered = evolve_pair.lower(dev.qudit_handles[auxiliary], dev.qudit_handles[data])
# The coupling turns at the 2.1 GHz detuning of the two frames: the step
# resolves it with 24 steps per period.
result = pulse.simulate(lowered, backend="gpu", dt_ns=0.02, integrator="rk4")

state = result.density_matrix
print(f"=== Final state: qudits {result.qudits}, shape {state.shape} ===")
print(f"Density-matrix trace: {np.trace(state).real:.8f}")
populations = np.real(np.diag(state))
for label, qudit in (("X1", auxiliary), ("D2", data)):
    # result.qudits[i] is digit i of the state index.
    mode = result.qudits.index(qudit)
    excited = sum(p for i, p in enumerate(populations) if (i >> mode) & 1)
    print(f"  P({label} excited) = {excited:.4f}")
