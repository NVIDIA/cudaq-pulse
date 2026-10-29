# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""W3: 4-qubit QAOA single layer on a ring topology.

Tests schedule density: 4 parallel Rz-diagonal rotations and 4 CR
gates on a ring, requiring explicit synchronization points.
"""

import math

import cudaq_pulse as pulse

NAME = "qaoa4"
NUM_QUBITS = 4


def _cr_cnot(d_ctrl, d_tgt):
    """Inline a CR CNOT; same op sequence as w2_cnot_cr."""
    sx = drag(40, 0.025, 10.0, 0.5)
    cr = gaussian(200, 0.10, 50.0)
    cr_neg = gaussian(200, -0.10, 50.0)
    x_c = square(40, [0.047, 0.0])

    drive(d_tgt, sx)
    drive(d_ctrl, cr, tone=d_tgt.tone)
    drive(d_ctrl, x_c)
    drive(d_ctrl, cr_neg, tone=d_tgt.tone)
    drive(d_tgt, sx)


def _cost_edge(d_ctrl, d_tgt, gamma):
    sync(d_ctrl, d_tgt)
    _cr_cnot(d_ctrl, d_tgt)
    shift_phase(d_tgt.tone, 2 * gamma)
    _cr_cnot(d_ctrl, d_tgt)


@pulse.kernel
def build(q0, q1, q2, q3):
    d0 = get_drive_line(q0)
    d1 = get_drive_line(q1)
    d2 = get_drive_line(q2)
    d3 = get_drive_line(q3)

    lines = [d0, d1, d2, d3]
    tones = [d0.tone, d1.tone, d2.tone, d3.tone]

    gamma = 0.3
    beta = 0.7

    # Cost layer: ZZ rotation via CNOT-Rz(2*gamma)-CNOT on ring edges.
    # The ring edges are spelled out one call per edge, which keeps the
    # workload's baselined op counts fixed.
    _cost_edge(d0, d1, gamma)
    _cost_edge(d1, d2, gamma)
    _cost_edge(d2, d3, gamma)
    _cost_edge(d3, d0, gamma)

    # Mixer layer: Rx(2*beta) on every qubit via SX-Rz-SX
    sx = drag(40, 0.25, 10.0, 0.5)
    for q in range(4):
        shift_phase(tones[q], math.pi / 2)
        drive(lines[q], sx, tone=tones[q])
        shift_phase(tones[q], 2 * beta)
        drive(lines[q], sx, tone=tones[q])
        shift_phase(tones[q], math.pi / 2)
