# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU integration tests: I/Q quadrature modulation.

The in-phase (I) quadrature drives an X rotation; the quadrature (Q)
component -- realized here as a pi/2 frame phase shift before the drive --
drives a Y rotation. The Pulse -> QOp lowering models these as
amplitude*cos(phase) (X) and amplitude*sin(phase) (Y) control terms.
"""

import math

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.targets import MicrowaveDrive, Qubit, TargetDescription

from tests._handles import Q
from tests.runtime._lowering import evolve, on, pulse_mlir

gpu = pytest.mark.gpu


def _target():
    return TargetDescription(
        name="iq-modulation",
        qubits={
            0: Qubit(
                index=0,
                frequency_hz=5.0e9,
                anharmonicity_hz=-200.0e6,
                t1_us=0.0,
                t2_star_us=0.0,
                drive=MicrowaveDrive(amplitude_scale_rad_per_ns=1.0),
            )
        },
    )


def _evolve(kernel):
    return evolve(
        on(_target(), kernel, Q[0]),
        t_start=0.0,
        t_end=20.0,
        num_steps=200,
        integrator="rk4",
    ).final_state


def test_iq_modulation_mlir_structure():
    """A quadrature (phase-shifted) drive lowers to valid MLIR."""

    @pulse.kernel
    def q_drive(q0):
        d0 = get_drive_line(q0)
        shift_phase(d0.tone, math.pi / 2)
        drive(d0, square(40, math.pi / 20.0))

    mlir = pulse_mlir((q_drive, Q[0]), clock_ghz=2.0, qubit_freq_hz={0: 5.0e9})
    assert "pulse.square" in mlir


@gpu
def test_pure_i_drives_x_rotation():
    """In-phase (phase 0) pi drive flips |0> -> |1>."""

    @pulse.kernel
    def x_pi(q0):
        d0 = get_drive_line(q0)
        drive(d0, square(40, math.pi / 20.0))

    assert abs(_evolve(x_pi)[1]) ** 2 > 0.99


@gpu
def test_pure_q_drives_y_rotation():
    """Quadrature (phase pi/2) pi drive also flips |0> -> |1>."""

    @pulse.kernel
    def y_pi(q0):
        d0 = get_drive_line(q0)
        shift_phase(d0.tone, math.pi / 2)
        drive(d0, square(40, math.pi / 20.0))

    assert abs(_evolve(y_pi)[1]) ** 2 > 0.99


@gpu
def test_iq_symmetry_equal_populations():
    """Equal-magnitude X and Y pi/2 drives yield equal |1> population."""

    @pulse.kernel
    def x90(q0):
        d0 = get_drive_line(q0)
        drive(d0, square(40, math.pi / 40.0))

    @pulse.kernel
    def y90(q0):
        d0 = get_drive_line(q0)
        shift_phase(d0.tone, math.pi / 2)
        drive(d0, square(40, math.pi / 40.0))

    p1_x = abs(_evolve(x90)[1]) ** 2
    p1_y = abs(_evolve(y90)[1]) ** 2
    assert p1_x == pytest.approx(0.5, abs=0.05)
    assert p1_y == pytest.approx(p1_x, abs=0.05)


@gpu
def test_xy_orthogonality_distinct_states():
    """X(pi/2) and Y(pi/2) produce distinguishable states (different phases)."""

    @pulse.kernel
    def x90(q0):
        d0 = get_drive_line(q0)
        drive(d0, square(40, math.pi / 40.0))

    @pulse.kernel
    def y90(q0):
        d0 = get_drive_line(q0)
        shift_phase(d0.tone, math.pi / 2)
        drive(d0, square(40, math.pi / 40.0))

    x_state = _evolve(x90)
    y_state = _evolve(y90)
    overlap = abs(np.vdot(x_state, y_state)) ** 2
    # Same populations but different relative phase => not the same state.
    assert overlap < 0.9
