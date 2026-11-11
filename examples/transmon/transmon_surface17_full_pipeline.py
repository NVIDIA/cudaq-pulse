#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calibration-aware lowering demo on the 17-qubit transmon quantum system.

  kernel -> pulse.compile (default passes, ALAP) -> Pulse MLIR
         -> QOp (operator algebra) -> CuDensityMat -> LLVM

Prints the intermediate representations of the native lowering stack for a
coupled pair, auxiliary X1 and data qubit D2. ``pulse.simulate(..., backend="gpu")``
runs the same stack, adding the quantum system's calibration (drive scales,
couplings, T1/T2) as module attributes, and executes it.

The quantum system's CZ is a flux pulse; flux lines are not modeled yet, so the
gate's 98 ns slot is left idle here.
"""

import shutil
import subprocess

import cudaq_pulse as pulse
from cudaq_pulse.quantum_systems.transmon import CZ_LAYERS

qs = pulse.quantum_system("transmon_surface17")
clock_ghz = qs.clock_ghz  # the program clock the schedule counts cycles of
qudit_aux, qudit_data = qs.qudit("X1"), qs.qudit("D2")
coupling = qs.coupling(qudit_aux, qudit_data)
# The transmon_surface17 quantum system publishes each coupling's CZ; a generic coupling
# has none (get_calibration raises NotFoundError).
cz = coupling.get_calibration(pulse.gates.CZ)
assert cz.interaction_frequency_hz is not None and coupling.strength_hz is not None
pair = (qudit_aux.index, qudit_data.index)
layer = next(i for i, pairs in enumerate(CZ_LAYERS, 1) if pair in pairs)
print(
    f"X1-D2: J_qq/2pi = {coupling.strength_hz / 1e6:.2f} MHz, CZ layer {layer} at "
    f"{cz.interaction_frequency_hz / 1e9:.2f} GHz, {cz.duration_ns:.0f} ns"
)
cz_slot = cz.duration_ns * pulse.ns


@pulse.kernel
def cz_pair(qudit_aux, qudit_data):
    drive_line_aux = get_drive_line(qudit_aux)
    drive_line_data = get_drive_line(qudit_data)
    # The qudits' calibrated pulses: the published 40 ns, sigma = 10 ns DRAG,
    # amplitudes rates in rad/ns.
    x90 = drive_line_aux.get_calibration(pulse.gates.X90)

    # pi/2 on the auxiliary qubit, pi on the data qubit
    drive(drive_line_aux, x90)
    drive(drive_line_data, drive_line_data.get_calibration(pulse.gates.X))
    sync(drive_line_aux, drive_line_data)

    # The CZ slot
    wait(drive_line_aux, cz_slot)
    wait(drive_line_data, cz_slot)
    sync(drive_line_aux, drive_line_data)

    drive(drive_line_aux, x90)


print("=== Stage 1: compile (default passes, ALAP schedule) ===")
compiled = pulse.compile(cz_pair, quantum_system=qs).lower(qudit_aux, qudit_data)
total_cycles = max(e.end for e in compiled.schedule)
print(f"  {compiled.metrics.op_count} ops, {total_cycles / clock_ghz:.0f} ns total")

print("\n=== Stage 2: scheduled Pulse MLIR ===")
print(compiled.mlir())

print("=== Stage 3: operator (QOp) lowering, with cudaq-pulse-opt ===")
tool = shutil.which("cudaq-pulse-opt")
if tool is None:
    raise SystemExit(
        "cudaq-pulse-opt is not on PATH; it comes with cudaq-pulse, from the "
        f"cudaq-pulse-tools wheel: pip install cudaq-pulse-tools=={pulse.__version__}"
    )
qop = subprocess.run(
    [tool, "--pulse-to-qop"], input=compiled.mlir(), capture_output=True, text=True, check=True
).stdout
print(f"  {qop.count('qop.callback_scalar')} time-dependent drive terms")

print("\n=== Stage 4: LLVM IR for the cuDensityMat runtime ===")
llvm = compiled.mlir("llvm")
print(f"  {llvm.count('llvm.call')} runtime calls")
