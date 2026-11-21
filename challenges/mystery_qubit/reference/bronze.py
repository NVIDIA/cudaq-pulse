# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mystery qubit, bronze: the T1 sweep of the Coherence tutorial, taken in shots.

Two changes from docs/tutorials/coherence.rst: each point is the fraction of
10 000 shots that read 1, from lab.run, instead of an exact population from
pulse.simulate; and nothing is subtracted before the fit, because this qubit
is cold (its thermal population is 0).
"""

import cudaq_pulse as pulse
import numpy as np


@pulse.kernel
def t1(qudit, wait_ns: float):
    drive_line = get_drive_line(qudit)
    readout_line = get_readout_line(qudit)
    flip = drive_line.get_calibration(pulse.gates.X)
    measurement = readout_line.get_calibration(pulse.gates.READOUT)

    drive(drive_line, flip)
    wait(drive_line, wait_ns * pulse.ns)
    sync(drive_line, readout_line)
    return readout(readout_line, measurement)


def estimate_t1(lab):
    waits = np.linspace(0, 60_000, 7)
    fractions = []
    for wait_ns in waits:
        records = lab.run(t1, lab.qubit, wait_ns, shots_count=10_000)
        fractions.append(records.mean())

    # The decay is a straight line on a log scale, whose slope gives T1.
    slope, intercept = np.polyfit(waits, np.log(fractions), 1)
    return -1 / slope / 1000
