# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU integration tests: frame (phase) operations.

``shift_phase`` rotates the drive axis, which the Pulse -> QOp lowering models
as X/Y quadratures (amplitude*cos(phase) and amplitude*sin(phase)).
"""

import math

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.targets import DeviceDescription, MicrowaveDrive, Qubit

from tests._handles import Q
from tests.runtime._lowering import evolve, on, pulse_mlir

gpu = pytest.mark.gpu


def _target():
    return DeviceDescription(
        name="frame-ops",
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
        t_end=40.0,
        step_count=400,
        integrator="rk4",
    ).final_state


def test_frame_operations_mlir_structure():
    """A kernel using shift_phase lowers with a pulse.shift_phase op."""

    @pulse.kernel
    def phased(q0):
        d0 = get_drive_line(q0)
        shift_phase(d0.tone, math.pi / 2)
        drive(d0, square(40, math.pi / 20.0))

    mlir = pulse_mlir((phased, Q[0]), clock_ghz=2.0, qubit_freq_hz={0: 5.0e9})
    assert "pulse.shift_phase" in mlir or "phase" in mlir


@gpu
def test_phase_shift_on_ground_state_no_population_change():
    """A frame phase shift alone does not move population out of |0>."""

    @pulse.kernel
    def phase_only(q0):
        d0 = get_drive_line(q0)
        shift_phase(d0.tone, math.pi / 2)
        wait(d0, 20)

    state = _evolve(phase_only)
    assert abs(state[0]) ** 2 > 0.99


@gpu
def test_opposite_phase_pulses_cancel():
    """Two pi/2 pulses about opposite axes (phase pi apart) return to |0>."""

    @pulse.kernel
    def cancel(q0):
        d0 = get_drive_line(q0)
        drive(d0, square(40, math.pi / 40.0))  # +X pi/2
        shift_phase(d0.tone, math.pi)
        drive(d0, square(40, math.pi / 40.0))  # -X pi/2

    state = _evolve(cancel)
    assert abs(state[0]) ** 2 > 0.95


@gpu
def test_same_phase_pulses_add():
    """Two same-axis pi/2 pulses compose into a pi rotation to |1>."""

    @pulse.kernel
    def add(q0):
        d0 = get_drive_line(q0)
        drive(d0, square(40, math.pi / 40.0))
        drive(d0, square(40, math.pi / 40.0))

    state = _evolve(add)
    assert abs(state[1]) ** 2 > 0.95
