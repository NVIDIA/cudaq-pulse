# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU integration tests: T1 amplitude damping and multi-qubit decoherence.

Uses ``TargetDescription`` T1 Lindblad terms to check T1 decay, its
monotonicity, ground-state stability, trace preservation, and independent
decay of several qubits.
"""

import math

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.targets import MicrowaveDrive, Qubit, TargetDescription

from tests._handles import Q
from tests.runtime._lowering import evolve, on, pulse_mlir

gpu = pytest.mark.gpu


def _single_qubit_target(*, t1_us, t2_star_us=0.0, frequency_hz=5.0e9):
    return TargetDescription(
        name="decoherence-test",
        qubits={
            0: Qubit(
                index=0,
                frequency_hz=frequency_hz,
                anharmonicity_hz=-200.0e6,
                t1_us=t1_us,
                t2_star_us=t2_star_us,
                drive=MicrowaveDrive(amplitude_scale_rad_per_ns=1.0),
            )
        },
    )


# A calibrated pi pulse: 40 virtual units at 2 GHz is 20 ns; with H = amp*X/2
# an amplitude of pi/20 rad/ns applies a pi rotation |0> -> |1>.
def _pi_pulse_then_wait(wait_cycles):

    @pulse.kernel
    def kernel(q0):
        d0 = get_drive_line(q0)
        drive(d0, square(40, math.pi / 20.0))
        wait(d0, wait_cycles)

    return kernel


def test_decoherence_mlir_structure():
    """A pi-pulse-plus-wait kernel lowers to structurally valid MLIR."""
    ir = (_pi_pulse_then_wait(1000), Q[0])
    mlir = pulse_mlir(ir, clock_ghz=2.0, qubit_freq_hz={0: 5.0e9})

    assert "pulse.square" in mlir
    assert "pulse.wait" in mlir


@gpu
def test_t1_excited_state_decays():
    """After a pi pulse and a wait, the excited population damps toward |0>."""
    target = _single_qubit_target(t1_us=0.2)
    ir = (_pi_pulse_then_wait(1000), Q[0])
    result = evolve(on(target, *ir), t_start=0.0, t_end=520.0, step_count=520, integrator="rk4")

    rho = result.final_state
    assert rho.shape == (2, 2)
    assert np.trace(rho).real == pytest.approx(1.0, abs=1.0e-6)
    p1 = rho[1, 1].real
    # exp(-500/200) ~ 0.082; the 20 ns drive adds a little in-pulse decay.
    assert p1 < 0.5
    assert rho[0, 0].real > p1


@gpu
def test_t1_longer_wait_more_decay():
    """Longer idle time produces strictly more T1 decay."""
    target = _single_qubit_target(t1_us=0.2)

    def _run(wait_cycles, t_end):
        ir = (_pi_pulse_then_wait(wait_cycles), Q[0])
        result = evolve(
            on(target, *ir), t_start=0.0, t_end=t_end, step_count=int(t_end), integrator="rk4"
        )
        return result.final_state[1, 1].real

    p1_short = _run(200, 120.0)  # 100 ns wait
    p1_long = _run(2000, 1020.0)  # 1000 ns wait
    assert p1_long < p1_short
    assert p1_short < 1.0


@gpu
def test_ground_state_stable_under_decoherence():
    """The ground state is the fixed point of T1 damping."""
    target = _single_qubit_target(t1_us=0.2)

    @pulse.kernel
    def idle(q0):
        d0 = get_drive_line(q0)
        wait(d0, 2000)  # 1000 ns of free evolution, no drive

    ir = (idle, Q[0])
    result = evolve(on(target, *ir), t_start=0.0, t_end=1000.0, step_count=1000, integrator="rk4")
    rho = result.final_state
    assert rho.shape == (2, 2)
    assert rho[0, 0].real > 0.98


@gpu
def test_decoherence_preserves_trace():
    """Lindblad evolution conserves the density-matrix trace."""
    target = _single_qubit_target(t1_us=0.2)
    ir = (_pi_pulse_then_wait(4000), Q[0])
    result = evolve(on(target, *ir), t_start=0.0, t_end=2020.0, step_count=2020, integrator="rk4")
    rho = result.final_state
    assert np.trace(rho).real == pytest.approx(1.0, abs=1.0e-3)


@gpu
def test_multiqubit_independent_decay():
    """Two qubits with distinct T1 both damp; the whole state stays physical."""
    target = TargetDescription(
        name="two-qubit-decoherence",
        qubits={
            0: Qubit(
                index=0,
                frequency_hz=5.0e9,
                anharmonicity_hz=-200.0e6,
                t1_us=0.2,
                t2_star_us=0.0,
                drive=MicrowaveDrive(amplitude_scale_rad_per_ns=1.0),
            ),
            1: Qubit(
                index=1,
                frequency_hz=5.2e9,
                anharmonicity_hz=-200.0e6,
                t1_us=0.1,
                t2_star_us=0.0,
                drive=MicrowaveDrive(amplitude_scale_rad_per_ns=1.0),
            ),
        },
    )

    @pulse.kernel
    def excite_both(q0, q1):
        d0 = get_drive_line(q0)
        d1 = get_drive_line(q1)
        drive(d0, square(40, math.pi / 20.0))
        drive(d1, square(40, math.pi / 20.0))
        sync(d0, d1)
        wait(d0, 1000)
        wait(d1, 1000)

    ir = (excite_both, Q[0], Q[1])
    result = evolve(on(target, *ir), t_start=0.0, t_end=520.0, step_count=520, integrator="rk4")

    rho = result.final_state
    assert rho.shape == (4, 4)
    assert np.trace(rho).real == pytest.approx(1.0, abs=1.0e-3)
    # Reduced excited populations. The first qudit is the least significant
    # digit: index 1 is q0 alone excited, 2 is q1 alone, 3 is both.
    p1_q0 = rho[1, 1].real + rho[3, 3].real
    p1_q1 = rho[2, 2].real + rho[3, 3].real
    assert p1_q0 < 0.9
    assert p1_q1 < 0.9
