#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The timeline of one error correction cycle of the Krinner surface code.

Demonstrates:
  - Building a 17-qubit pulse program from the target's lattice and CZ
    schedule (Krinner et al., Nature 605, 669 (2022), Fig. 3a and Fig. 6c)
  - The Z half: pi/2 on the Z auxiliaries, CZ steps 1-2, an echo pi pulse
    on every data qubit, steps 3-4, pi/2
  - The X half: pi/2 on the X auxiliaries and the data qubits, steps 5-6,
    echo, steps 7-8, pi/2
  - Barriers between steps, so a step lasts as long as its gates
  - Compiling it with pulse.compile (default passes, ALAP)

Each CZ is a flux pulse; flux lines are not modeled yet, so its 98 ns slot
idles both qubits. Readout, which the device overlaps with the next half
cycle, is left out.
"""

import math

import cudaq_pulse as pulse
from cudaq_pulse.targets import transmon_krinner_17q

target = transmon_krinner_17q()
dev = pulse.device("transmon_krinner_17q")
clock_ghz = dev.clock_ghz  # the program clock the schedule counts cycles of
labels = {q.index: q.label for q in target.qubits.values()}
n_qubits = target.n_qubits
data_qubits = [i for i in range(n_qubits) if labels[i].startswith("D")]
z_auxiliaries = [i for i in range(n_qubits) if labels[i].startswith("Z")]
x_auxiliaries = [i for i in range(n_qubits) if labels[i].startswith("X")]

# The CZ gates of each time step, (auxiliary, data), and their slot.
steps: dict[int, list[tuple[int, int]]] = {}
for c in target.couplings:
    steps.setdefault(int(c.gate_params["cz_step"]), []).append((c.qubit_a, c.qubit_b))
z_steps = [[steps[1], steps[2]], [steps[3], steps[4]]]
x_steps = [[steps[5], steps[6]], [steps[7], steps[8]]]
cz_ns = max(c.gate_duration_ns for c in target.couplings if c.gate_duration_ns is not None)

for s in sorted(steps):
    print(f"CZ step {s}: " + ", ".join(f"{labels[a]}-{labels[d]}" for a, d in steps[s]))

# The published 40 ns, sigma = 10 ns pulse; amplitudes are rates in rad/ns.
drive_params = target.get_drive_params(0)
duration_ns, sigma_ns = drive_params["x_dur"], drive_params["x_sigma"]
area_ns = sigma_ns * math.sqrt(2 * math.pi) * math.erf(duration_ns / (2 * math.sqrt(2) * sigma_ns))
pi_amplitude = math.pi / area_ns
half_pi_amplitude = pi_amplitude / 2
cal = {
    "x_duration": duration_ns * pulse.ns,
    "x_sigma": sigma_ns * pulse.ns,
    "cz_slot": cz_ns * pulse.ns,
}


@pulse.kernel
def surface_code_cycle(qubits):
    lines = []
    tones = []
    for i in range(n_qubits):
        line, tone = get_drive_line(qubits[i])
        lines.append(line)
        tones.append(tone)
    duration, sigma, cz_slot = cal["x_duration"], cal["x_sigma"], cal["cz_slot"]

    for auxiliaries, basis_change, halves in (
        (z_auxiliaries, [], z_steps),
        (x_auxiliaries, data_qubits, x_steps),
    ):
        for i in auxiliaries:
            drive(lines[i], drag(duration, half_pi_amplitude, sigma, 0.0), tones[i])
        for i in basis_change:
            drive(lines[i], drag(duration, half_pi_amplitude, sigma, 0.0), tones[i])
        for h in range(2):
            for gates in halves[h]:
                # A barrier over every line, then the step's CZ slots.
                for i in range(n_qubits - 1):
                    sync(lines[i], lines[i + 1])
                for i in range(n_qubits - 1):
                    sync(lines[n_qubits - 2 - i], lines[n_qubits - 1 - i])
                for a, d in gates:
                    wait(lines[a], cz_slot)
                    wait(lines[d], cz_slot)
            for i in range(n_qubits - 1):
                sync(lines[i], lines[i + 1])
            for i in range(n_qubits - 1):
                sync(lines[n_qubits - 2 - i], lines[n_qubits - 1 - i])
            if h == 0:
                # The echo between the second and third step of each half.
                for i in data_qubits:
                    drive(lines[i], drag(duration, pi_amplitude, sigma, 0.0), tones[i])
        for i in auxiliaries:
            drive(lines[i], drag(duration, half_pi_amplitude, sigma, 0.0), tones[i])
        for i in basis_change:
            drive(lines[i], drag(duration, half_pi_amplitude, sigma, 0.0), tones[i])
        for i in range(n_qubits - 1):
            sync(lines[i], lines[i + 1])
        for i in range(n_qubits - 1):
            sync(lines[n_qubits - 2 - i], lines[n_qubits - 1 - i])


compiled = pulse.compile(surface_code_cycle, device=dev).lower(dev.qudit_handles[:n_qubits])
total_ns = max(e.end for e in compiled.schedule) / clock_ghz
print(
    f"\nCompiled and verified: {compiled.metrics.op_count} ops, "
    f"total = {total_ns:.0f} ns: 8 CZ steps of 98 ns and 6 single-qubit slots of 40 ns "
    "(the device's cycle, with readout, takes 1.1 us)"
)
