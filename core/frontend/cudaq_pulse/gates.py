# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The gates a quantum system is calibrated for, which ``get_calibration`` takes.

In a kernel, a line gives the calibrated pulse of its qudit's gates, ready
to play::

    d = get_drive_line(q)
    drive(d, d.get_calibration(pulse.gates.X))      # a DragPulse
    r = get_readout_line(q)
    readout(r, r.get_calibration(pulse.gates.READOUT))  # a ReadoutPulse

Outside one, a qudit handle and a coupling give the same values:
``qs.qudits[i].get_calibration(pulse.gates.X)`` and
``qs.coupling(a, b).get_calibration(pulse.gates.CZ)`` (a ``FluxCZ``).

``X``, ``X90``, ``Y`` and ``Y90`` are the gates of a drive line, ``READOUT``
of a readout line and ``CZ`` of a coupling. What a quantum system does not publish
is derived: an ``X`` without an amplitude gets the one that makes its area a
pi rotation, ``X90`` is ``X`` at half the amplitude, and ``Y`` and ``Y90``
are ``X`` and ``X90`` at +90 degrees. An unpublished ``CZ`` or ``READOUT``
is a ``NotFoundError`` naming the coupling or qudit. ``transition=(1, 2)``
asks for the calibration of another transition, which must be published.
"""

from __future__ import annotations

import enum
from typing import Final

__all__ = ["CZ", "READOUT", "X", "X90", "Y", "Y90", "Gate"]


class Gate(enum.StrEnum):
    """A gate a quantum system is calibrated for; see the module docstring."""

    X = "x"
    X90 = "x90"
    Y = "y"
    Y90 = "y90"
    CZ = "cz"
    READOUT = "readout"


X: Final = Gate.X
X90: Final = Gate.X90
Y: Final = Gate.Y
Y90: Final = Gate.Y90
CZ: Final = Gate.CZ
READOUT: Final = Gate.READOUT
