#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import cudaq_pulse as pulse
import numpy as np
from cudaq_pulse.runtime import require_gpu_runtime

require_gpu_runtime()

dev = pulse.device("transmon_17q")
qudit_aux, qudit_data = dev.qudit("X1"), dev.qudit("D2")
auxiliary, data = qudit_aux.index, qudit_data.index
# The CZ's slot, from the coupling's calibration; the pulses are the
# qudits' X90 and X calibrations (the published 40 ns, sigma = 10 ns DRAG,
# amplitudes rates in rad/ns).
cz_slot = dev.coupling(qudit_aux, qudit_data).get_calibration(pulse.gates.CZ).duration_ns * pulse.ns


@pulse.kernel
def evolve_pair(qudit_aux, qudit_data):
    drive_line_aux = get_drive_line(qudit_aux)
    drive_line_data = get_drive_line(qudit_data)
    drive(drive_line_aux, drive_line_aux.get_calibration(pulse.gates.X90))
    drive(drive_line_data, drive_line_data.get_calibration(pulse.gates.X))
    sync(drive_line_aux, drive_line_data)
    wait(drive_line_aux, cz_slot)
    wait(drive_line_data, cz_slot)


lowered = evolve_pair.lower(qudit_aux, qudit_data)
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
