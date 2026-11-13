#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The timeline of one error correction cycle of the 17-qubit surface code.

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

import cudaq_pulse as pulse
from cudaq_pulse.targets.transmon import CZ_LAYERS

dev = pulse.device("transmon_17q")
target = dev.description
clock_ghz = dev.clock_ghz  # the program clock the schedule counts cycles of
labels = {q.index: q.label for q in target.qubits.values()}
qudit_count = target.qudit_count
data_qubits = [i for i in range(qudit_count) if labels[i].startswith("D")]
z_auxiliaries = [i for i in range(qudit_count) if labels[i].startswith("Z")]
x_auxiliaries = [i for i in range(qudit_count) if labels[i].startswith("X")]

# The CZs of each time step, (auxiliary, data), and their slot.
z_steps = [list(CZ_LAYERS[0:2]), list(CZ_LAYERS[2:4])]
x_steps = [list(CZ_LAYERS[4:6]), list(CZ_LAYERS[6:8])]
cz_slot = max(c.get_calibration(pulse.gates.CZ).duration_ns for c in target.couplings) * pulse.ns

for s, layer in enumerate(CZ_LAYERS, 1):
    print(f"CZ step {s}: " + ", ".join(f"{labels[a]}-{labels[d]}" for a, d in layer))


@pulse.kernel
def surface_code_cycle(qubits):
    # Each qudit's calibrated pi and pi/2 pulses: the published 40 ns,
    # sigma = 10 ns DRAG, amplitudes rates in rad/ns.
    lines = []
    tones = []
    pi_pulses = []
    half_pi_pulses = []
    for i in range(qudit_count):
        line = get_drive_line(qubits[i])
        lines.append(line)
        tones.append(line.tone)
        pi_pulses.append(line.get_calibration(pulse.gates.X))
        half_pi_pulses.append(line.get_calibration(pulse.gates.X90))

    for auxiliaries, basis_change, halves in (
        (z_auxiliaries, [], z_steps),
        (x_auxiliaries, data_qubits, x_steps),
    ):
        for i in auxiliaries:
            drive(lines[i], half_pi_pulses[i], tone=tones[i])
        for i in basis_change:
            drive(lines[i], half_pi_pulses[i], tone=tones[i])
        for h in range(2):
            for gates in halves[h]:
                # A barrier over every line, then the step's CZ slots.
                for i in range(qudit_count - 1):
                    sync(lines[i], lines[i + 1])
                for i in range(qudit_count - 1):
                    sync(lines[qudit_count - 2 - i], lines[qudit_count - 1 - i])
                for a, d in gates:
                    wait(lines[a], cz_slot)
                    wait(lines[d], cz_slot)
            for i in range(qudit_count - 1):
                sync(lines[i], lines[i + 1])
            for i in range(qudit_count - 1):
                sync(lines[qudit_count - 2 - i], lines[qudit_count - 1 - i])
            if h == 0:
                # The echo between the second and third step of each half.
                for i in data_qubits:
                    drive(lines[i], pi_pulses[i], tone=tones[i])
        for i in auxiliaries:
            drive(lines[i], half_pi_pulses[i], tone=tones[i])
        for i in basis_change:
            drive(lines[i], half_pi_pulses[i], tone=tones[i])
        for i in range(qudit_count - 1):
            sync(lines[i], lines[i + 1])
        for i in range(qudit_count - 1):
            sync(lines[qudit_count - 2 - i], lines[qudit_count - 1 - i])


compiled = pulse.compile(surface_code_cycle, device=dev).lower(dev.qudits[:qudit_count])
total_ns = max(e.end for e in compiled.schedule) / clock_ghz
print(
    f"\nCompiled and verified: {compiled.metrics.op_count} ops, "
    f"total = {total_ns:.0f} ns: 8 CZ steps of 98 ns and 6 single-qubit slots of 40 ns "
    "(the device's cycle, with readout, takes 1.1 us)"
)
