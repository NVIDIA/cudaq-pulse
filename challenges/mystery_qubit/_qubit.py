# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The qubit of the challenge "Mystery qubit", built for a given T1."""

import cudaq_pulse as pulse
from cudaq_pulse.quantum_systems import DragPulse, PhysicalQudit, ReadoutPulse


def mystery_system(t1_us):
    calibrations = {
        pulse.gates.X: DragPulse(duration_ns=40.0, sigma_ns=10.0),
        pulse.gates.READOUT: ReadoutPulse(duration_ns=300.0, amplitude=0.05),
    }
    qudit = PhysicalQudit(0, 5.1e9, t1_us, 40.0, anharmonicity_hz=-2.2e8, calibrations=calibrations)
    return pulse.quantum_system(name="mystery", qudits={0: qudit})
