# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The quantum system most tests run on, and its qudit handles ``Q``."""

import dataclasses

import cudaq_pulse as pulse
from cudaq_pulse.quantum_systems import PhysicalQudit


def ladder_target(n: int = 64, **parts) -> pulse.QuantumSystem:
    """Qudit i at 5 GHz + i * 100 MHz: the frequencies most tests assume; *parts*
    are the other keywords of ``pulse.quantum_system``."""
    return pulse.quantum_system(
        name=f"ladder:{n}",
        qudits={
            i: PhysicalQudit(i, 5.0e9 + 1.0e8 * i, 50.0, 30.0, anharmonicity_hz=-300e6)
            for i in range(n)
        },
        **parts,
    )


def rebuilt(qs: pulse.QuantumSystem, **parts) -> pulse.QuantumSystem:
    """A quantum system built from *qs*'s own values (not registered, with no
    calibration); *parts* are the other keywords of ``pulse.quantum_system``."""
    return pulse.quantum_system(
        **{
            "name": qs.name,
            "qudits": qs.physical_qudits,
            "couplings": qs.couplings,
            "residual_zz": qs.residual_zz,
            "model": qs.model,
            "attribution": qs.attribution,
            "clock_ghz": qs.clock_ghz,
        }
        | parts
    )


LADDER = ladder_target(256)
Q = LADDER.qudits


def with_line_delays(controllers, delays):
    """*controllers* with their lines delayed as *delays* gives, by (role,
    qudit), in ns."""
    from cudaq_pulse.controller import DriveLine, ReadoutLine

    def lines(given, cls):
        return [
            cls(line.qudit, delay_ns=delays.get((cls._role, line.qudit), 0.0)) for line in given
        ]

    return [
        dataclasses.replace(
            c,
            drive_lines=lines(c.drive_lines, DriveLine),
            readout_lines=lines(c.readout_lines, ReadoutLine),
        )
        for c in controllers
    ]
