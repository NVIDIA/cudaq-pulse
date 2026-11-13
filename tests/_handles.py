# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The device most tests run on, and its qudit handles ``Q``."""

import dataclasses

import cudaq_pulse as pulse
from cudaq_pulse.targets import DeviceDescription, PhysicalQudit


def ladder_target(n: int = 64) -> DeviceDescription:
    """Qudit i at 5 GHz + i * 100 MHz: the frequencies most tests assume."""
    return DeviceDescription(
        name=f"ladder:{n}",
        qudits={i: PhysicalQudit(i, 5.0e9 + 1.0e8 * i, -300e6, 50.0, 30.0) for i in range(n)},
    )


LADDER = pulse.quantum_system(ladder_target(256))
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
