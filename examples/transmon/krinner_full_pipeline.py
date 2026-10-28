#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Target-aware lowering demo on the Krinner target.

  kernel -> pulse.compile (default passes, ALAP) -> Pulse MLIR
         -> QOp (operator algebra) -> CuDensityMat -> LLVM

Prints the intermediate representations of the native lowering stack for a
coupled pair, auxiliary X1 and data qubit D2. ``pulse.simulate(..., backend="gpu")``
runs the same stack, adding the target's calibration (drive scales,
couplings, T1/T2) as module attributes, and executes it.

The device's CZ is a flux pulse; flux lines are not modeled yet, so the
gate's 98 ns slot is left idle here.
"""

import math
import shutil
import subprocess

import cudaq_pulse as pulse
from cudaq_pulse.targets import transmon_krinner_17q

target = transmon_krinner_17q()
dev = pulse.get_device(pulse.Target.TRANSMON_KRINNER_17Q)
index = {q.label: q.index for q in target.qubits.values()}
auxiliary, data = index["X1"], index["D2"]
coupling = next(c for c in target.couplings if {c.qubit_a, c.qubit_b} == {auxiliary, data})
# The Krinner target gives each CZ's published duration; a generic Coupling
# leaves it unspecified (None).
cz_ns = coupling.gate_duration_ns
assert cz_ns is not None
print(
    f"X1-D2: J_qq/2pi = {coupling.coupling_strength_hz / 1e6:.2f} MHz "
    f"(|11>-|20>: {coupling.gate_params['coupling_11_20_hz'] / 1e6:.0f} MHz), CZ step "
    f"{coupling.gate_params['cz_step']:.0f} at "
    f"{coupling.gate_params['interaction_frequency_hz'] / 1e9:.2f} GHz, "
    f"{cz_ns:.0f} ns"
)

# The published 40 ns, sigma = 10 ns pulse; amplitudes are rates in rad/ns.
drive_params = target.get_drive_params(data)
clock_ghz = dev.clock_ghz  # the program clock the schedule counts cycles of
duration_ns, sigma_ns = drive_params["x_dur"], drive_params["x_sigma"]
area_ns = sigma_ns * math.sqrt(2 * math.pi) * math.erf(duration_ns / (2 * math.sqrt(2) * sigma_ns))
pi_amplitude = math.pi / area_ns
cal = {
    "x_duration": duration_ns * pulse.ns,
    "x_sigma": sigma_ns * pulse.ns,
    "cz_slot": cz_ns * pulse.ns,
}


@pulse.kernel
def krinner_pair(qubit_aux, qubit_data):
    drive_line_aux, tone_aux = get_drive_line(qubit_aux)
    drive_line_data, tone_data = get_drive_line(qubit_data)
    duration, sigma = cal["x_duration"], cal["x_sigma"]

    # pi/2 on the auxiliary qubit, pi on the data qubit
    drive(drive_line_aux, drag(duration, pi_amplitude / 2, sigma, 0.0), tone_aux)
    drive(drive_line_data, drag(duration, pi_amplitude, sigma, 0.0), tone_data)
    sync(drive_line_aux, drive_line_data)

    # The CZ slot
    wait(drive_line_aux, cal["cz_slot"])
    wait(drive_line_data, cal["cz_slot"])
    sync(drive_line_aux, drive_line_data)

    drive(drive_line_aux, drag(duration, pi_amplitude / 2, sigma, 0.0), tone_aux)


print("=== Stage 1: compile (default passes, ALAP schedule) ===")
compiled = pulse.compile(krinner_pair, device=dev).lower(
    dev.qudit_handles[auxiliary], dev.qudit_handles[data]
)
total_cycles = max(e.end for e in compiled.schedule)
print(f"  {compiled.metrics.op_count} ops, {total_cycles / clock_ghz:.0f} ns total")

print("\n=== Stage 2: scheduled Pulse MLIR ===")
print(compiled.mlir())

print("=== Stage 3: operator (QOp) lowering, with cudaq-pulse-opt ===")
tool = shutil.which("cudaq-pulse-opt")
if tool is None:
    raise SystemExit("cudaq-pulse-opt is not on PATH; it is installed with cudaq-pulse")
qop = subprocess.run(
    [tool, "--pulse-to-qop"], input=compiled.mlir(), capture_output=True, text=True, check=True
).stdout
print(f"  {qop.count('qop.callback_scalar')} time-dependent drive terms")

print("\n=== Stage 4: LLVM IR for the cuDensityMat runtime ===")
llvm = compiled.mlir("llvm")
print(f"  {llvm.count('llvm.call')} runtime calls")
