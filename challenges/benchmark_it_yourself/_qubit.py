# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The qubit of the challenge "Benchmark it yourself", built for a given T1 and T2*."""

import cudaq_pulse as pulse
from cudaq_pulse.quantum_systems import DragPulse, PhysicalQudit, ReadoutPulse


def benchmark_system(t1_us, t2_star_us):
    calibrations = {
        pulse.gates.X90: DragPulse(duration_ns=40.0, sigma_ns=10.0, beta_ns=0.557),
        pulse.gates.READOUT: ReadoutPulse(duration_ns=200.0, amplitude=0.05),
    }
    qudit = PhysicalQudit(
        0, 6.037e9, t1_us, t2_star_us, anharmonicity_hz=-1.7e8, calibrations=calibrations
    )
    return pulse.quantum_system(name="benchmark", qudits={0: qudit})


@pulse.kernel
def calibrated(qudit):
    drive_line = get_drive_line(qudit)
    readout_line = get_readout_line(qudit)
    drive(drive_line, drive_line.get_calibration(pulse.gates.X90))
    drive(drive_line, drive_line.get_calibration(pulse.gates.Y90))
    sync(drive_line, readout_line)
    return readout(readout_line, readout_line.get_calibration(pulse.gates.READOUT))
