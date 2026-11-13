# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU integration test: T1 exponential decay."""

import math

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.quantum_systems import DeviceDescription, MicrowaveDrive, PhysicalQudit

from tests._handles import Q
from tests.runtime._lowering import evolve, on, pulse_mlir

gpu = pytest.mark.gpu


def test_t1_mlir_structure():
    """Verify T1 decay kernel produces valid MLIR."""

    @pulse.kernel
    def t1_decay(q0):
        d0 = get_drive_line(q0)
        # pi pulse to |1>
        pi_pulse = gaussian(40, 0.5, 10.0)
        drive(d0, pi_pulse)
        # wait for T1 decay
        wait(d0, 1000)

    ir = (t1_decay, Q[0])
    mlir = pulse_mlir(ir, clock_ghz=2.0, qudit_freq_hz={0: 5.0e9})

    assert "pulse.gaussian" in mlir
    assert "pulse.wait" in mlir
    assert "arith.constant 1000 : i64" in mlir


@gpu
def test_t1_decay_gpu():
    """T1 decay: after exciting to |1> and waiting, population decays."""

    target = DeviceDescription(
        name="fast-decay-test",
        qudits={
            0: PhysicalQudit(
                index=0,
                frequency_hz=5.0e9,
                anharmonicity_hz=-200.0e6,
                t1_us=0.05,
                t2_star_us=0.0,
                drive=MicrowaveDrive(amplitude_scale_rad_per_ns=1.0),
            )
        },
    )

    @pulse.kernel
    def t1_decay(q0):
        d0 = get_drive_line(q0)
        # 40 virtual units at 2 GHz is 20 ns. With H = amplitude * X / 2,
        # amplitude pi/20 applies a pi rotation.
        pi_pulse = square(40, math.pi / 20.0)
        drive(d0, pi_pulse)
        wait(d0, 1000)

    ir = (t1_decay, Q[0])
    result = evolve(on(target, *ir), t_start=0.0, t_end=520.0, step_count=520, integrator="rk4")

    state = result.final_state
    assert state.shape == (2, 2)
    assert np.trace(state) == pytest.approx(1.0, abs=1.0e-6)
    assert state[1, 1].real < 1.0e-3
