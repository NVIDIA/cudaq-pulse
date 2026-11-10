# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The CPU integrator's exact and kept propagators, and sampled paths.

Stretches with a constant Lindbladian (idle waits, square pulses in their
frame) are propagated exactly; driven stretches on small registers by their
RK4 map, built at once. Both must agree with stepping RK4 on the state, and
``branches="sampled"`` must follow only the branches shots take.
"""

from __future__ import annotations

import math
import time

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse import ControlSystem
from cudaq_pulse.classical import Link
from cudaq_pulse.quantum_systems import MicrowaveDrive, PhysicalQudit
from cudaq_pulse.runtime import _propagate
from cudaq_pulse.runtime import simulate as simulate_module

from tests._handles import Q
from tests.runtime._lowering import on

_PI = math.pi / 20.0


def _target(t1_us=30.0, t2_us=20.0, thermal=0.02, offset_hz=0.0):
    return pulse.quantum_system(
        name="propagation-test",
        qudits={
            0: PhysicalQudit(
                index=0,
                frequency_hz=5.0e9 + offset_hz,
                anharmonicity_hz=-200.0e6,
                t1_us=t1_us,
                t2_star_us=t2_us,
                drive=MicrowaveDrive(amplitude_scale_rad_per_ns=1.0),
                thermal_population=thermal,
            )
        },
    )


@pulse.kernel
def _mixed_pulses(q0):
    d = get_drive_line(q0)
    drive(d, gaussian(120, _PI, 30))
    # Short enough that stepping does not damp a qutrit's 0-2 coherence.
    wait(d, 40)
    shift_phase(d.tone, 0.3)
    drive(d, square(40, _PI / 2))
    drive(d, drag(60, _PI, 15, 0.4))
    shift_frequency(d.tone, 3e6)
    drive(d, square(80, _PI / 3))


def _simulated(lowered, monkeypatch, *, maps=True, exact=True, **kwargs):
    """The density matrix with kept RK4 maps and exact propagation on or
    off (off: every stretch stepped by RK4 on the state)."""
    with monkeypatch.context() as m:
        if not maps:
            m.setattr(_propagate, "MAX_MAPPED_DIM", 0)
        if not exact:
            m.setattr(simulate_module._Stepper, "_exact_pays", lambda self, n, h: False)
        _propagate.clear_cache()
        return pulse.simulate(lowered, **kwargs).density_matrix


@pytest.mark.parametrize("levels", [2, 3])
def test_kept_rk4_maps_give_what_stepping_gives(levels, monkeypatch):
    lowered = on(_target(), _mixed_pulses, Q[0])
    stepped = _simulated(lowered, monkeypatch, maps=False, exact=False, levels=levels)
    for _ in range(2):  # built, then kept
        with monkeypatch.context() as m:
            m.setattr(simulate_module._Stepper, "_exact_pays", lambda self, n, h: False)
            rho = pulse.simulate(lowered, levels=levels).density_matrix
        np.testing.assert_allclose(rho, stepped, atol=1e-10)


@pytest.mark.parametrize("levels", [2, 3])
def test_exact_stretches_agree_with_stepping_within_its_error(levels, monkeypatch):
    lowered = on(_target(), _mixed_pulses, Q[0])
    errors = []
    for dt in (0.05, 0.025):
        exact = _simulated(lowered, monkeypatch, levels=levels, dt_ns=dt)
        stepped = _simulated(lowered, monkeypatch, exact=False, levels=levels, dt_ns=dt)
        errors.append(np.abs(exact - stepped).max())
    # They differ by RK4's error on the square pulses, which falls as dt^4.
    assert errors[0] < 1e-7
    assert errors[1] < errors[0] / 8 or errors[0] < 1e-12


@pulse.kernel
def _excite_and_wait(q0, wait_cycles: int):
    d = get_drive_line(q0)
    drive(d, square(40, _PI))
    wait(d, wait_cycles)


def test_a_long_idle_decays_exactly_and_quickly():
    target = _target(t1_us=30.0, thermal=0.0)
    short = pulse.simulate(on(target, _excite_and_wait, Q[0], 200_000), levels=2).density_matrix
    start = time.perf_counter()
    long = pulse.simulate(on(target, _excite_and_wait, Q[0], 400_000), levels=2).density_matrix
    elapsed = time.perf_counter() - start
    # 100 us more of pure decay.
    assert long[1, 1].real / short[1, 1].real == pytest.approx(math.exp(-100.0 / 30.0), rel=1e-9)
    # Stepping 0.05 ns at a time would take minutes.
    assert elapsed < 2.0


def test_a_detuned_plant_is_propagated_like_stepping(monkeypatch):
    lowered = on(_target(), _mixed_pulses, Q[0])
    plant = _target(offset_hz=250e3)
    stepped = _simulated(lowered, monkeypatch, maps=False, exact=False, quantum_system=plant)
    rho = _simulated(lowered, monkeypatch, quantum_system=plant)
    np.testing.assert_allclose(rho, stepped, atol=1e-7)


def test_expm_matches_a_diagonalizable_exponential():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    a = 40.0 * (a + a.conj().T)  # large norm: many squarings
    values, vectors = np.linalg.eigh(a)
    expected = vectors @ np.diag(np.exp(-1j * values)) @ vectors.conj().T
    np.testing.assert_allclose(_propagate.expm(-1j * a), expected, atol=1e-9)


@pytest.mark.parametrize(("rows", "columns"), [(1, 1), (2, 3), (4, 4)])
def test_kron_is_numpys_kron(rows, columns):
    rng = np.random.default_rng(rows * columns)
    a = rng.normal(size=(rows, columns)) + 1j * rng.normal(size=(rows, columns))
    for b in (rng.normal(size=(3, 2)), np.eye(columns), -a.T):
        for x, y in ((a, b), (b, a)):
            expected = np.kron(x, y)
            got = _propagate.kron(x, y)
            assert got.dtype == expected.dtype and got.tobytes() == expected.tobytes()


@pulse.kernel
def _three_measurements(q0):
    d = get_drive_line(q0)
    r = get_readout_line(q0)
    for _ in range(3):
        drive(d, square(40, _PI / 2))
        sync(d, r)
        readout(r, square(200, 0.05))
        sync(d, r)


def test_sampled_branches_have_the_frequencies_of_the_exact_branches():
    lowered = on(_target(t1_us=0.0, t2_us=0.0, thermal=0.0), _three_measurements, Q[0])
    exact = {b.levels: b.probability for b in pulse.simulate(lowered, levels=2).branches}
    shots_count = 4000
    sampled = pulse.simulate(lowered, shots_count=shots_count, seed=5, branches="sampled", levels=2)
    assert len(sampled.branches) <= shots_count
    assert sum(b.probability for b in sampled.branches) == pytest.approx(1.0)
    assert np.trace(sampled.density_matrix).real == pytest.approx(1.0)
    for branch in sampled.branches:
        p = exact[branch.levels]
        assert abs(branch.probability - p) < 5 * math.sqrt(p / shots_count)
    assert sampled.records is not None and sampled.records.shape == (shots_count, 3)


def test_sampled_branches_need_shots():
    lowered = on(_target(), _three_measurements, Q[0])
    with pytest.raises(ValueError, match="shots_count"):
        pulse.simulate(lowered, branches="sampled", levels=2)
    with pytest.raises(ValueError, match="unknown branches 'some'; choose 'all' or 'sampled'"):
        pulse.simulate(lowered, shots_count=1, branches="some", levels=2)


def test_branch_coverage_takes_its_members_or_their_strings():
    lowered = on(_target(), _three_measurements, Q[0])
    by_string = pulse.simulate(lowered, shots_count=50, seed=2, branches="sampled", levels=2)
    by_member = pulse.simulate(
        lowered, shots_count=50, seed=2, branches=pulse.BranchCoverage.SAMPLED, levels=2
    )
    assert by_member.branches == by_string.branches
    assert pulse.BranchCoverage("all") is pulse.BranchCoverage.ALL


_MACHINE = ControlSystem(links={"controller": Link(fixed_ns=4, feedback_ns=8)})


@pulse.classical(max_latency_ns=200, link="controller")
def _halve_or_double(bit: pulse.Bits, wait_cycles: float) -> float:
    return min(max(wait_cycles * (2.0 if bit[0] else 0.5), 20.0), 20000.0)


@pulse.kernel
def _adaptive_loop(q0, rounds: int):
    d = get_drive_line(q0)
    r = get_readout_line(q0)
    w = 2000.0
    drive(d, square(40, _PI))
    wait(d, 2000)
    for _ in range(rounds):
        sync(d, r)
        m = readout(r, square(200, 0.05))
        w = _halve_or_double.submit([m], w).result()
        sync(d, r)
        if m:
            drive(d, square(40, _PI))
        drive(d, square(40, _PI))
        wait(d, w, bounds=(20, 20000))


def test_a_long_adaptive_loop_follows_one_path_per_shot():
    dev = _target(t1_us=5.0)
    lowered = pulse.compile(_adaptive_loop, quantum_system=dev, control_system=_MACHINE).lower(
        dev.qudits[0], 60
    )
    result = pulse.simulate(lowered, shots_count=3, seed=1, branches="sampled", levels=2)
    assert 1 <= len(result.branches) <= 3
    assert all(len(branch.calls) == 60 for branch in result.branches)
    assert result.records is not None and result.records.shape == (3, 60)


def test_kept_propagators_are_keyed_compactly():
    @pulse.kernel(unroll_threshold=0)
    def train(q0):
        d = get_drive_line(q0)
        for _ in range(4):
            drive(d, gaussian(400, _PI, 100))

    _propagate.clear_cache()
    pulse.simulate(on(_target(), train, Q[0]), levels=2)
    sizes = [
        sum(len(part) for part in key[2:] if isinstance(part, bytes)) for key in _propagate._cache
    ]
    # A key names a stretch by a digest of its coefficients, whatever its
    # length: the cache would otherwise hold every stretch's samples.
    assert sizes and max(sizes) <= 1024
