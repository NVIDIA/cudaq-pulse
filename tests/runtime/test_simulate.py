# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Measurement and feedback simulation against analytic results.

Every test runs the CPU reference; the GPU-marked tests check that the
cuDensityMat backend agrees with it.
"""

from __future__ import annotations

import math

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.targets import Qubit, Target

from tests.runtime._lowering import on

# 40 virtual units at 2 GHz is 20 ns; with H = amp X / 2 these amplitudes
# rotate by pi and pi / 2.
_PI = math.pi / 20.0
_HALF_PI = math.pi / 40.0


def _target(n=1, t1_us=0.0):
    return Target(
        name="simulate-test",
        qubits={
            q: Qubit(
                index=q,
                frequency_hz=5.0e9 + 0.1e9 * q,
                anharmonicity_hz=-200.0e6,
                t1_us=t1_us,
                t2_star_us=0.0,
                drive_params={"amplitude_scale_rad_per_ns": 1.0},
            )
            for q in range(n)
        },
    )


def _active_reset(prepare):
    @pulse.kernel
    def reset(q0):
        d0, t0 = get_drive_line(q0)
        r0, rt = get_readout_line(q0)
        drive(d0, square(40, prepare), t0)
        sync(d0, r0)
        bit = readout(r0, square(200, 0.05), rt)
        if bit:
            drive(d0, square(40, _PI), t0)

    return reset


@pulse.kernel
def conditional_flip(q0, q1):
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    r0, rt = get_readout_line(q0)
    drive(d0, square(40, _HALF_PI), t0)
    sync(d0, r0)
    bit = readout(r0, square(200, 0.05), rt)
    if bit:
        drive(d1, square(40, _PI), t1)


@pulse.kernel
def measure(q0):
    r0, rt = get_readout_line(q0)
    readout(r0, square(200, 0.05), rt)


def _run(kernel, n=1, target=None, **kwargs):
    lowered = on(target or _target(n), kernel, *[pulse.qudit_ref() for _ in range(n)])
    return pulse.simulate(lowered, **kwargs)


def _rabi(amplitude):
    @pulse.kernel
    def rabi(q0):
        d0, t0 = get_drive_line(q0)
        drive(d0, square(40, amplitude), t0)

    return rabi


def test_rabi_matches_the_analytic_rotation():
    for amplitude in (_HALF_PI, _PI, 0.03):
        rho = _run(_rabi(amplitude)).density_matrix
        assert rho[1, 1].real == pytest.approx(math.sin(amplitude * 20.0 / 2) ** 2, abs=1e-8)


def test_active_reset_returns_the_excited_state_to_ground():
    result = _run(_active_reset(_PI), shots=200, seed=1)
    np.testing.assert_allclose(result.density_matrix, np.diag([1.0, 0.0]), atol=1e-7)
    assert result.records is not None and (result.records[:, 0] == 1).all()


def test_active_reset_of_a_superposition():
    result = _run(_active_reset(_HALF_PI), shots=4000, seed=2)
    # Either outcome ends in |0>: the ensemble is pure.
    np.testing.assert_allclose(result.density_matrix, np.diag([1.0, 0.0]), atol=1e-7)
    probabilities = dict(result.branches)
    assert probabilities[(0,)] == pytest.approx(0.5, abs=1e-8)
    assert probabilities[(1,)] == pytest.approx(0.5, abs=1e-8)
    ones = result.records[:, 0].mean()
    assert abs(ones - 0.5) < 4 * math.sqrt(0.25 / 4000)


def test_conditional_flip_correlates_the_qubits():
    result = _run(conditional_flip, 2, shots=1000, seed=3)
    expected = np.zeros((4, 4))
    expected[0, 0] = expected[3, 3] = 0.5  # |00> and |11>, qubit 0 least significant
    np.testing.assert_allclose(result.density_matrix, expected, atol=1e-7)
    assert sorted(result.branches) == [((0,), pytest.approx(0.5)), ((1,), pytest.approx(0.5))]


def test_decay_during_the_readout_sets_the_outcome_probability():
    t1_us = 0.5
    result = _run(measure, target=_target(1, t1_us), initial_state=[0.0, 1.0])
    # The readout is 100 ns; the measurement completes at its end.
    p1 = math.exp(-100.0 / (t1_us * 1e3))
    assert dict(result.branches)[(1,)] == pytest.approx(p1, rel=1e-6)
    np.testing.assert_allclose(np.diag(result.density_matrix).real, [1 - p1, p1], atol=1e-7)


def test_measuring_a_qutrit_dephases_all_three_levels():
    psi = np.ones(3) / math.sqrt(3)
    result = _run(measure, levels=3, initial_state=psi, shots=3000, seed=4)
    np.testing.assert_allclose(result.density_matrix, np.eye(3) / 3, atol=1e-8)
    assert {o: p for (o,), p in result.branches} == pytest.approx({0: 1 / 3, 1: 1 / 3, 2: 1 / 3})
    counts = np.bincount(result.records[:, 0], minlength=3) / 3000
    assert np.all(np.abs(counts - 1 / 3) < 4 * math.sqrt(2 / 9 / 3000))


@pulse.kernel
def repeated_reset(q0):
    d0, t0 = get_drive_line(q0)
    r0, rt = get_readout_line(q0)
    drive(d0, square(40, _HALF_PI), t0)
    for _ in range(3):
        sync(d0, r0)
        bit = readout(r0, square(200, 0.05), rt)
        if bit:
            drive(d0, square(40, _PI), t0)


def test_resets_in_a_retained_loop():
    assert "scf.for" in repeated_reset.lower(pulse.qudit_ref()).mlir()
    result = _run(repeated_reset, shots=500, seed=5)
    np.testing.assert_allclose(result.density_matrix, np.diag([1.0, 0.0]), atol=1e-7)
    # Three measurements, one per iteration; after the first reset every
    # later one reads 0.
    assert [m.iteration for m in result.measurements] == [(0,), (1,), (2,)]
    likely = sorted(o for o, p in result.branches if p > 1e-9)
    assert likely == [(0, 0, 0), (1, 0, 0)]
    assert (result.records[:, 1:] == 0).all()


def test_a_retained_loop_simulates_like_its_unrolling():
    @pulse.kernel
    def rolled(q0):
        d0, t0 = get_drive_line(q0)
        for _ in range(4):
            drive(d0, square(40, 0.02), t0)
            t0 = shift_phase(t0, 0.7)

    @pulse.kernel
    def unrolled(q0):
        d0, t0 = get_drive_line(q0)
        for i in range(4):
            drive(d0, square(40, 0.02 + 0 * i), t0)
            t0 = shift_phase(t0, 0.7)

    a = _run(rolled).density_matrix
    b = _run(unrolled).density_matrix
    assert "scf.for" in rolled.lower(pulse.qudit_ref()).mlir()
    np.testing.assert_allclose(a, b, atol=1e-10)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "kernel, n", [(_active_reset(_HALF_PI), 1), (conditional_flip, 2)], ids=["reset", "flip"]
)
def test_gpu_backend_matches_the_reference(kernel, n):
    cpu = _run(kernel, n)
    gpu = _run(kernel, n, backend="gpu")
    np.testing.assert_allclose(gpu.density_matrix, cpu.density_matrix, atol=1e-6)
    assert [o for o, _ in gpu.branches] == [o for o, _ in cpu.branches]
