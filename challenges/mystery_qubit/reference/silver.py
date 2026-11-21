# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mystery qubit, silver: look first, then spend the shots where they say the most.

A shot says the most about T1 when the wait is about 1.6 T1, where a fifth of
the shots still read 1. That wait depends on the answer, so find it roughly
with a few shots, then take the rest there.
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
    # Look: double the wait until about a third of 100 shots still read 1.
    wait_ns = 10_000.0
    while True:
        fraction = lab.run(t1, lab.qubit, wait_ns, shots_count=100).mean()
        if fraction < 0.35:
            break
        wait_ns = 2 * wait_ns
    rough_t1_ns = -wait_ns / np.log(max(fraction, 0.01))

    # The fraction with no wait: the pi pulse and the decay during the readout.
    start = lab.run(t1, lab.qubit, 0.0, shots_count=1000).mean()

    # Measure: the rest of the shots, at 1.6 times the rough T1.
    wait_ns = 1.6 * rough_t1_ns
    end = lab.run(t1, lab.qubit, wait_ns, shots_count=8500).mean()
    return wait_ns / np.log(start / end) / 1000
