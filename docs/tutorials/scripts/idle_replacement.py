# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Idle Replacement: Echoing a Readout's Crosstalk: the code of the tutorial, as one script.

https://nvidia.github.io/cudaq-pulse/tutorials/idle_replacement.html

Generated from docs/tutorials/idle_replacement.rst by tools/tutorial-scripts.py: edit
the page, not this file.
"""

# %% The crosstalk
import math

import cudaq_pulse as pulse
from cudaq_pulse.quantum_systems import ReadoutCrosstalk

system = pulse.quantum_system("transmon_surface17")
d5, z2 = system.qudit("D5"), system.qudit("Z2")
system = system.replace(
    readout_crosstalk=[ReadoutCrosstalk(resonator_of=z2.index, qudit=d5.index, chi_hz=-20e3)]
)
d5, z2 = system.qudit("D5"), system.qudit("Z2")
print(system.readout_crosstalk)

# Prints:
#   (ReadoutCrosstalk(resonator_of=14, qudit=4, chi_hz=-20000.0),)


# %% The conditional phase
@pulse.kernel
def data_beside_readout(data, ancilla, flip: int, quarter: int):
    data_drive = get_drive_line(data)
    ancilla_drive = get_drive_line(ancilla)
    ancilla_readout = get_readout_line(ancilla)
    if flip:
        drive(ancilla_drive, ancilla_drive.get_calibration(pulse.gates.X))
    sync(data_drive, ancilla_drive, ancilla_readout)
    drive(data_drive, data_drive.get_calibration(pulse.gates.X90))
    sync(data_drive, ancilla_readout)
    shift_frequency(ancilla_readout.tone, 2.9e6)
    readout(ancilla_readout, ancilla_readout.get_calibration(pulse.gates.READOUT))
    wait(ancilla_readout, 120)
    sync(data_drive, ancilla_readout)
    for _ in range(quarter):
        shift_phase(data_drive.tone, math.pi / 2)
    drive(data_drive, data_drive.get_calibration(pulse.gates.X90))


def conditional_phase(compiled):
    phases = []
    for flip in (0, 1):
        excited = []
        for quarter in (0, 1):
            program = compiled.lower(d5, z2, flip, quarter)
            result = pulse.simulate(program, dt_ns=0.02, levels=2)
            excited.append(result.level_populations(d5)[1])
        phases.append(math.atan2(2 * excited[1] - 1, 2 * excited[0] - 1))
    return math.degrees(math.remainder(phases[1] - phases[0], 2 * math.pi))


compiled = pulse.compile(data_beside_readout, quantum_system=system)
print(f"no fill: {conditional_phase(compiled):+.2f} degrees")

# Prints:
#   no fill: +2.15 degrees


# %% D5's idle window
listed = compiled.replace_idle(pulse.IdleReplacement(None, threshold=160 * pulse.ns, qudits=[d5]))
for window in listed.lower(d5, z2, 0, 0).idle_windows or ():
    print(f"qudit {window.qudit}: cycles {window.start} to {window.end}; {window.result}")

# Prints:
#   qudit 4: cycles 80 to 600; not filled: the rule has no fill


# %% Filling it with echoes
flip_cycles = round(
    system.physical_qudits[d5.index].get_calibration(pulse.gates.X).duration_ns
    * system.clock_ghz
)


def evenly_spaced(count):
    @pulse.kernel
    def flips(qudit, window: int):
        line = get_drive_line(qudit)
        flip = line.get_calibration(pulse.gates.X)
        gap = (window - count * flip_cycles) // count
        wait(line, gap // 2)
        for i in range(count):
            drive(line, flip)
            if i < count - 1:
                wait(line, gap)
        wait(line, window - count * flip_cycles - (count - 1) * gap - gap // 2)

    return flips


for count in (1, 2, 4):
    rule = pulse.IdleReplacement(evenly_spaced(count), threshold=160 * pulse.ns, qudits=[d5])
    phase = conditional_phase(compiled.replace_idle(rule))
    print(f"{count} flip{'s' if count > 1 else ''}: {phase:+.2f} degrees")

# Prints:
#   1 flip: +0.29 degrees
#   2 flips: -1.51 degrees
#   4 flips: -0.10 degrees
