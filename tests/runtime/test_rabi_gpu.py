# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU integration test: 1-qubit Rabi oscillation.

Verifies the MLIR emission for a Rabi experiment is structurally correct.
Full GPU execution test is marked with @pytest.mark.gpu.
"""

import math

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.targets import DeviceDescription, MicrowaveDrive, Qubit

from tests._handles import Q
from tests.runtime._lowering import evolve, on, pulse_mlir

gpu = pytest.mark.gpu


def test_rabi_mlir_structure():
    """Verify the MLIR text structure for a Rabi simulation."""

    @pulse.kernel
    def rabi(q0):
        d0 = get_drive_line(q0)
        wf = gaussian(100, 0.1, 25.0)
        drive(d0, wf)

    ir = (rabi, Q[0])
    mlir = pulse_mlir(ir, clock_ghz=2.0, qubit_freq_hz={0: 5.0e9})

    assert "module @rabi" in mlir
    assert "pulse.qudit_alloc" in mlir
    assert "pulse.get_drive_line" in mlir
    assert "pulse.gaussian" in mlir
    assert "!pulse.waveform" in mlir


@gpu
def test_rabi_gpu_execution():
    """Single-qubit Rabi oscillation on GPU."""

    target = DeviceDescription(
        name="unitary-rabi-test",
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

    @pulse.kernel
    def rabi(q0):
        d0 = get_drive_line(q0)
        wf = square(40, math.pi / 20.0)
        drive(d0, wf)

    ir = (rabi, Q[0])
    result = evolve(on(target, *ir), t_start=0.0, t_end=20.0, step_count=200, integrator="rk4")

    state = result.final_state
    assert state.shape == (2,)
    assert np.vdot(state, state).real == pytest.approx(1.0, abs=1.0e-6)
    assert abs(state[1]) ** 2 > 0.999


@gpu
@pytest.mark.parametrize("integrator", ["rk4", "magnus", "crank_nicolson"])
def test_rabi_gpu_integrator_parity(integrator):
    """All cuDensityMat integrators drive the same pi-pulse to |1>.

    Exercises the magnus (Taylor-series midpoint) and crank_nicolson
    (predictor-corrector) paths added to cudm-runtime alongside rk4, and
    confirms they agree on a closed-system Rabi flip while preserving norm.
    """

    target = DeviceDescription(
        name="unitary-rabi-parity",
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

    @pulse.kernel
    def rabi(q0):
        d0 = get_drive_line(q0)
        wf = square(40, math.pi / 20.0)
        drive(d0, wf)

    ir = (rabi, Q[0])
    result = evolve(on(target, *ir), t_start=0.0, t_end=20.0, step_count=200, integrator=integrator)

    state = result.final_state
    assert state.shape == (2,)
    assert np.vdot(state, state).real == pytest.approx(1.0, abs=1.0e-6)
    assert abs(state[1]) ** 2 > 0.999
