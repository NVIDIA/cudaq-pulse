# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""GPU integration tests: quantum-algorithm building blocks.

Single-qubit rotations and gate-sequence identities run on the GPU. The
multi-qubit entangling schedules are checked as compile and lowering
structure tests, not as fidelity assertions.
"""

import math

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.targets import Qubit, TargetDescription

from tests._handles import Q
from tests.runtime._lowering import evolve, on, pulse_mlir

gpu = pytest.mark.gpu


def _unitary_qubit(index, frequency_hz):
    return Qubit(
        index=index,
        frequency_hz=frequency_hz,
        anharmonicity_hz=-200.0e6,
        t1_us=0.0,
        t2_star_us=0.0,
        drive_params={"amplitude_scale_rad_per_ns": 1.0},
    )


def _single_qubit_target():
    return TargetDescription(name="algo-1q", qubits={0: _unitary_qubit(0, 5.0e9)})


def _evolve_1q(kernel):
    return evolve(
        on(_single_qubit_target(), kernel, Q[0]),
        t_start=0.0,
        t_end=40.0,
        num_steps=400,
        integrator="rk4",
    ).final_state


def test_bell_schedule_mlir_structure():
    """A CR-based Bell schedule lowers to valid two-qubit MLIR."""

    @pulse.kernel
    def bell(q0, q1):
        d0 = get_drive_line(q0)
        d1 = get_drive_line(q1)
        shift_phase(d0.tone, math.pi / 2)
        drive(d0, drag(40, 0.25, 10.0, 0.5))
        shift_phase(d0.tone, math.pi / 2)
        sync(d0, d1)
        drive(d0, gaussian(160, 0.05, 40.0), tone=d1.tone)  # cross-resonance
        sync(d0, d1)

    mlir = pulse_mlir(
        (bell, Q[0], Q[1]),
        clock_ghz=2.0,
        qubit_freq_hz={0: 5.0e9, 1: 5.1e9},
    )
    assert mlir.count("pulse.qudit_alloc") == 2
    assert "pulse.sync" in mlir


def test_ghz_schedule_mlir_structure():
    """A 3-qubit GHZ-style chain schedule lowers to valid MLIR."""

    @pulse.kernel
    def ghz(q0, q1, q2):
        d0 = get_drive_line(q0)
        d1 = get_drive_line(q1)
        d2 = get_drive_line(q2)
        drive(d0, square(40, math.pi / 40.0))
        sync(d0, d1, d2)
        drive(d0, gaussian(160, 0.05, 40.0), tone=d1.tone)
        drive(d1, gaussian(160, 0.05, 40.0), tone=d2.tone)
        sync(d0, d1, d2)

    mlir = pulse_mlir(
        (ghz, Q[0], Q[1], Q[2]),
        clock_ghz=2.0,
        qubit_freq_hz={0: 5.0e9, 1: 5.1e9, 2: 5.2e9},
    )
    assert mlir.count("pulse.qudit_alloc") == 3


@gpu
def test_single_qubit_x_rotation():
    """An X(pi) rotation flips |0> -> |1>."""

    @pulse.kernel
    def x_pi(q0):
        d0 = get_drive_line(q0)
        drive(d0, square(40, math.pi / 20.0))

    assert abs(_evolve_1q(x_pi)[1]) ** 2 > 0.99


@gpu
def test_single_qubit_y_rotation():
    """A Y(pi) rotation (phase pi/2) flips |0> -> |1>."""

    @pulse.kernel
    def y_pi(q0):
        d0 = get_drive_line(q0)
        shift_phase(d0.tone, math.pi / 2)
        drive(d0, square(40, math.pi / 20.0))

    assert abs(_evolve_1q(y_pi)[1]) ** 2 > 0.99


@gpu
def test_bloch_half_rotation():
    """An X(pi/2) rotation produces an equal superposition."""

    @pulse.kernel
    def x90(q0):
        d0 = get_drive_line(q0)
        drive(d0, square(40, math.pi / 40.0))

    assert abs(_evolve_1q(x90)[1]) ** 2 == pytest.approx(0.5, abs=0.05)


@gpu
def test_gate_sequence_identity():
    """Two X(pi) rotations compose to the identity, returning to |0>."""

    @pulse.kernel
    def xx(q0):
        d0 = get_drive_line(q0)
        drive(d0, square(40, math.pi / 20.0))
        drive(d0, square(40, math.pi / 20.0))

    assert abs(_evolve_1q(xx)[0]) ** 2 > 0.95
