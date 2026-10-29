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
from cudaq_pulse.targets import Qubit, TargetDescription

from tests._handles import Q
from tests.runtime._lowering import on

# 40 virtual units at 2 GHz is 20 ns; with H = amp X / 2 these amplitudes
# rotate by pi and pi / 2.
_PI = math.pi / 20.0
_HALF_PI = math.pi / 40.0


def _target(n=1, t1_us=0.0):
    return TargetDescription(
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
        d0 = get_drive_line(q0)
        r0 = get_readout_line(q0)
        drive(d0, square(40, prepare))
        sync(d0, r0)
        bit = readout(r0, square(200, 0.05))
        if bit:
            drive(d0, square(40, _PI))

    return reset


@pulse.kernel
def conditional_flip(q0, q1):
    d0 = get_drive_line(q0)
    d1 = get_drive_line(q1)
    r0 = get_readout_line(q0)
    drive(d0, square(40, _HALF_PI))
    sync(d0, r0)
    bit = readout(r0, square(200, 0.05))
    if bit:
        drive(d1, square(40, _PI))


@pulse.kernel
def measure(q0):
    r0 = get_readout_line(q0)
    readout(r0, square(200, 0.05))


def _run(kernel, n=1, target=None, **kwargs):
    lowered = on(target or _target(n), kernel, *[Q[i] for i in range(n)])
    return pulse.simulate(lowered, **kwargs)


def _rabi(amplitude):
    @pulse.kernel
    def rabi(q0):
        d0 = get_drive_line(q0)
        drive(d0, square(40, amplitude))

    return rabi


def test_rabi_matches_the_analytic_rotation():
    for amplitude in (_HALF_PI, _PI, 0.03):
        rho = _run(_rabi(amplitude)).density_matrix
        assert rho[1, 1].real == pytest.approx(math.sin(amplitude * 20.0 / 2) ** 2, abs=1e-8)


def test_simulate_takes_the_kernel_and_its_arguments():
    @pulse.kernel
    def rabi(q0, amplitude: float):
        d0 = get_drive_line(q0)
        drive(d0, square(40, amplitude))

    dev = pulse.device({"target": _target()})
    q = dev.qudit_handles[0]
    expected = pulse.simulate(rabi.lower(q, _PI)).density_matrix
    np.testing.assert_array_equal(pulse.simulate(rabi, q, _PI).density_matrix, expected)
    compiled = pulse.compile(rabi)
    np.testing.assert_array_equal(
        pulse.simulate(compiled, q, _PI, shots=2, seed=1).density_matrix, expected
    )
    with pytest.raises(pulse.InvalidTypeError, match="module of a call, which has its arguments"):
        pulse.simulate(rabi.lower(q, _PI), q)
    with pytest.raises(
        pulse.InvalidTypeError, match=r"expects a kernel and its arguments.*got str"
    ):
        pulse.simulate("rabi", q)  # type: ignore[arg-type]


@pulse.kernel
def _rabi_with_amplitude(q0, amplitude: float):
    d0 = get_drive_line(q0)
    drive(d0, square(40, amplitude))


@pytest.mark.parametrize("kind", ["kernel", "compiled", "with_parameters", "bound", "call"])
def test_simulate_takes_a_kernel_or_a_program_compiled_already(kind):
    dev = pulse.device({"target": _target()})
    q = dev.qudit_handles[0]
    compiled = pulse.compile(_rabi_with_amplitude)
    generic = compiled.lower(q, _PI).generic
    assert generic is not None and generic.parameters == ("amplitude",)
    program, args = {
        "kernel": (_rabi_with_amplitude, (q, _PI)),
        "compiled": (compiled, (q, _PI)),
        "with_parameters": (generic, (_PI,)),
        "bound": (generic.bind([_PI]), ()),
        "call": (compiled.lower(q, _PI), ()),
    }[kind]
    rho = pulse.simulate(program, *args).density_matrix
    assert rho[1, 1].real == pytest.approx(math.sin(_PI * 20.0 / 2) ** 2, abs=1e-8)


def test_simulate_binds_a_program_s_parameters_as_bind_does():
    @pulse.kernel
    def wait_for(q0, gap: float):
        d0 = get_drive_line(q0)
        wait(d0, gap, bounds=(0, 100))
        drive(d0, square(40, _PI))

    dev = pulse.device({"target": _target()})
    q = dev.qudit_handles[0]
    generic = pulse.compile(wait_for).lower(q, 10.0).generic
    assert generic is not None
    with pytest.raises(pulse.InvalidValueError, match=r"float parameters \(gap\); pass their"):
        pulse.simulate(generic)
    with pytest.raises(ValueError, match="bounds") as from_bind:
        generic.bind([500.0])
    with pytest.raises(type(from_bind.value), match="bounds"):
        pulse.simulate(generic, 500.0)
    with pytest.raises(pulse.InvalidTypeError, match="module of a call, which has its arguments"):
        pulse.simulate(generic.bind([10.0]), 10.0)


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
    d0 = get_drive_line(q0)
    r0 = get_readout_line(q0)
    drive(d0, square(40, _HALF_PI))
    for _ in range(3):
        sync(d0, r0)
        bit = readout(r0, square(200, 0.05))
        if bit:
            drive(d0, square(40, _PI))


def test_resets_in_a_retained_loop():
    assert "scf.for" in repeated_reset.lower(Q[0]).mlir()
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
        d0 = get_drive_line(q0)
        for _ in range(4):
            drive(d0, square(40, 0.02))
            shift_phase(d0.tone, 0.7)

    @pulse.kernel
    def unrolled(q0):
        d0 = get_drive_line(q0)
        for i in range(4):
            drive(d0, square(40, 0.02 + 0 * i))
            shift_phase(d0.tone, 0.7)

    a = _run(rolled).density_matrix
    b = _run(unrolled).density_matrix
    assert "scf.for" in rolled.lower(Q[0]).mlir()
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
