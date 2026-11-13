# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU integration test: 2-qubit Bell state from XX coupling."""

import math

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.targets import Coupling, DeviceDescription, PhysicalQudit

from tests._handles import Q
from tests.runtime._lowering import evolve, on, pulse_mlir

gpu = pytest.mark.gpu


def test_bell_mlir_structure():
    """Verify 2-qubit CR Bell MLIR has correct structure."""

    @pulse.kernel
    def bell_cr(q0, q1):
        d0 = get_drive_line(q0)
        d1 = get_drive_line(q1)
        # pi/2 X on q0
        x90 = gaussian(40, 0.25, 10.0)
        drive(d0, x90)
        sync(d0, d1)
        # CR drive on q0 at q1's frequency (simplified)
        cr = square(160, 0.05)
        drive(d0, cr)
        sync(d0, d1)
        # pi/2 X on q1
        drive(d1, x90)

    ir = (bell_cr, Q[0], Q[1])
    mlir = pulse_mlir(ir, clock_ghz=2.0, qudit_freq_hz={0: 5.0e9, 1: 5.1e9})

    assert mlir.count("pulse.qudit_alloc") == 2
    assert mlir.count("pulse.get_drive_line") == 2
    assert "pulse.sync" in mlir
    assert "pulse.square" in mlir
    assert "pulse.gaussian" in mlir


@gpu
def test_bell_gpu_execution():
    """Resonant exchange takes |10> to (|10> - i|01>) / sqrt(2) in pi / (4 g)."""

    qudits = {
        index: PhysicalQudit(
            index=index,
            frequency_hz=5.0e9,
            anharmonicity_hz=-200.0e6,
            t1_us=0.0,
            t2_star_us=0.0,
        )
        for index in range(2)
    }
    target = DeviceDescription(
        name="xx-bell-test",
        qudits=qudits,
        couplings=[Coupling((0, 1), strength_hz=5.0e6)],
    )

    @pulse.kernel
    def bell_xx(q0, q1):
        d0 = get_drive_line(q0)
        d1 = get_drive_line(q1)
        wait(d0, 50)
        wait(d1, 50)
        sync(d0, d1)

    ir = (bell_xx, Q[0], Q[1])
    # Qudit 0 is the least significant digit: index 1 is q0 excited.
    result = evolve(
        on(target, *ir),
        t_start=0.0,
        t_end=25.0,
        step_count=250,
        integrator="rk4",
        initial_state=[0.0, 1.0, 0.0, 0.0],
    )

    state = result.final_state
    expected = np.array([0.0, 1.0, -1.0j, 0.0]) / math.sqrt(2.0)
    fidelity = abs(np.vdot(expected, state)) ** 2
    assert state.shape == (4,)
    assert fidelity > 0.999
