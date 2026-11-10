# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""W1: Bell-state preparation.

Two-qubit hello-world: H on q0 (via SX + virtual-Z decomposition),
CNOT via echo cross-resonance, readout of both.  Provides the
smallest non-trivial schedule in the suite.
"""

import math

import cudaq_pulse as pulse

NAME = "bell"
NUM_QUBITS = 2


@pulse.kernel
def build(q0, q1):
    d0 = get_drive_line(q0)
    d1 = get_drive_line(q1)
    r0 = get_readout_line(q0)
    r1 = get_readout_line(q1)

    # Hadamard on q0 as Rz(pi/2) · SX · Rz(pi/2)
    shift_phase(d0.tone, math.pi / 2)
    sx = drag(40, 0.25, 10.0, 0.5)
    drive(d0, sx)
    shift_phase(d0.tone, math.pi / 2)

    sync(d0, d1)

    # CNOT via echo cross-resonance
    drive(d1, sx)
    cr = gaussian(200, 0.10, 50.0)
    drive(d0, cr, tone=d1.tone)
    x_ctrl = square(40, [0.047, 0.0])
    drive(d0, x_ctrl)
    cr_neg = gaussian(200, -0.10, 50.0)
    drive(d0, cr_neg, tone=d1.tone)
    drive(d1, sx)

    sync(d0, d1, r0, r1)

    ro = square(1000, [0.05, 0.0])
    readout(r0, ro)
    readout(r1, ro)
