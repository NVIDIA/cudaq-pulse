# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""GPU integration tests: 8-qubit ladder system.

The ladder connectivity is a ``TargetDescription``; the tests check a
256-dimensional GPU evolution on it.

    q0 -- q1 -- q2 -- q3
    |     |     |     |
    q4 -- q5 -- q6 -- q7
"""

import math

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.targets import Coupling, Qubit, TargetDescription

from tests._handles import Q
from tests.runtime._lowering import evolve, on

_HORIZONTAL = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7)]
_VERTICAL = [(0, 4), (1, 5), (2, 6), (3, 7)]
_LADDER_EDGES = _HORIZONTAL + _VERTICAL


gpu = pytest.mark.gpu


def _ladder_8q_target(coupling_hz=1.0e6):
    qubits = {
        i: Qubit(
            index=i,
            frequency_hz=(5.0 + 0.02 * i) * 1.0e9,
            anharmonicity_hz=-200.0e6,
            t1_us=0.0,
            t2_star_us=0.0,
            drive_params={"amplitude_scale_rad_per_ns": 1.0},
        )
        for i in range(8)
    }
    couplings = [Coupling(a, b, coupling_strength_hz=coupling_hz) for a, b in _LADDER_EDGES]
    return TargetDescription(name="ladder-8q", qubits=qubits, couplings=couplings)


def test_ladder_topology_structure():
    """The ladder target encodes the expected 10-edge connectivity."""
    target = _ladder_8q_target()
    assert len(target.couplings) == 10

    edges = {tuple(sorted(pair)) for pair in target.coupling_map}
    assert (0, 1) in edges
    assert (0, 4) in edges
    assert (5, 6) in edges
    assert (3, 7) in edges
    # Non-adjacent qubits are not directly coupled.
    assert (0, 2) not in edges
    assert (0, 7) not in edges
    assert (4, 7) not in edges


def test_ladder_neighbor_degrees():
    """Corner qubits have degree 2; edge/center qubits have degree 3."""
    graph = _ladder_8q_target().connectivity_graph()
    corners = [0, 3, 4, 7]
    inner = [1, 2, 5, 6]
    for q in corners:
        assert len(graph[q]) == 2
    for q in inner:
        assert len(graph[q]) == 3


@gpu
def test_8qubit_single_qubit_excitation():
    """A pi pulse on q0 excites it within the full 256-dimensional register."""
    target = _ladder_8q_target()

    # The simulated register holds the qudits the kernel gets a line on. q0
    # alone gets a pi pulse while q1..q7 idle in |0>, and the full 8-qubit
    # (256-dim) register still evolves.
    @pulse.kernel
    def excite_q0(q0, q1, q2, q3, q4, q5, q6, q7):
        d0 = get_drive_line(q0)
        get_drive_line(q1)
        get_drive_line(q2)
        get_drive_line(q3)
        get_drive_line(q4)
        get_drive_line(q5)
        get_drive_line(q6)
        get_drive_line(q7)
        drive(d0, square(40, math.pi / 20.0))

    refs = [Q[i] for i in range(8)]
    result = evolve(
        on(target, excite_q0, *refs), t_start=0.0, t_end=20.0, num_steps=200, integrator="rk4"
    )

    state = result.final_state
    assert state.shape == (256,)
    assert np.vdot(state, state).real == pytest.approx(1.0, abs=1.0e-5)

    # A single pi pulse drives exactly one qubit, so population leaves the
    # all-ground state and lands in the single-excitation manifold (basis
    # states whose index is a power of two). This holds regardless of the
    # MSB/LSB ordering convention.
    probs = np.abs(state) ** 2
    single_excitation = float(sum(probs[1 << b] for b in range(8)))
    assert probs[0] < 0.1  # left the all-ground state
    assert single_excitation > 0.5  # one qubit excited
