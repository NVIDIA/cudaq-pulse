# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The calibration estimators, against exact posteriors and known answers."""

from __future__ import annotations

import math

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse import _controller_cost
from cudaq_pulse._units import as_nanoseconds
from cudaq_pulse.classical import calibration as cal

us, ns = pulse.us, pulse.ns

# Each estimator with its times as pulse.Time and as numbers of ns.
TIMED = [
    (cal.t1_update, (2.0, 4000.0, False, 3 * us), (2.0, 4000.0, False, 3000.0)),
    (cal.frequency_update, (1e5, 2e5, 1, 250 * ns, 0.3), (1e5, 2e5, 1, 250.0, 0.3)),
    (cal.ramsey_setting, (1e5, 2e5, 0.1, 0.2 * us), (1e5, 2e5, 0.1, 200.0)),
    (cal.three_point_decay, (0.9, 0.6, 0.45, 10 * us), (0.9, 0.6, 0.45, 10_000.0)),
    (
        cal.adaptive_t1_update,
        (3.0, 450_000.0, 1, 200 * us, 0.11, 0.14),
        (3.0, 450_000.0, 1, 200_000.0, 0.11, 0.14),
    ),
    (
        cal.binary_search_update,
        (0.0, 1e6, 1, 100 * ns, -0.02, 0.6, 10 * us),
        (0.0, 1e6, 1, 100.0, -0.02, 0.6, 10_000.0),
    ),
    (cal.binary_search_wait, (1e6, 10 * us), (1e6, 10_000.0)),
    (cal.binary_search_phase, (1e5, 120 * ns), (1e5, 120.0)),
    (cal.decay_time, (0.9, 0.6, 0.35, 5 * us, 50), (0.9, 0.6, 0.35, 5000.0, 50)),
    (cal.ramsey_detuning, (1e5, 3.0, 400 * ns), (1e5, 3.0, 400.0)),
]


@pytest.mark.parametrize(
    ("estimator", "timed", "in_ns"), TIMED, ids=[e.__name__ for e, *_ in TIMED]
)
def test_every_time_an_estimator_takes_may_be_a_time(estimator, timed, in_ns):
    assert estimator(*timed) == estimator(*in_ns)


def test_keyword_and_default_times_may_be_times():
    assert cal.ramsey_setting(1e5, 2e5, max_wait_ns=0.2 * us) == cal.ramsey_setting(
        1e5, 2e5, max_wait_ns=200.0
    )
    assert cal.ramsey_setting(1e5, 2e5, max_wait_ns=None) == cal.ramsey_setting(1e5, 2e5)
    assert cal.binary_search_update(0.0, 1e6, -1, 80 * ns, coherence_ns=4 * us) == (
        cal.binary_search_update(0.0, 1e6, -1, 80.0, coherence_ns=4000.0)
    )


def test_converting_a_time_costs_a_controller_nothing():
    def converts(t):
        return as_nanoseconds(t) * 2.0

    def takes(t):
        return t * 2.0

    cost = _controller_cost.analyze_callable(converts, {})
    assert cost.runs_on_processor, cost.violations
    assert cost.ops == _controller_cost.analyze_callable(takes, {}).ops


def _moments(grid, density):
    weight = density / np.trapezoid(density, grid)
    mean = np.trapezoid(grid * weight, grid)
    return mean, np.trapezoid((grid - mean) ** 2 * weight, grid)


@pytest.mark.parametrize("survived", [True, False])
@pytest.mark.parametrize(("alpha", "beta", "t"), [(2.0, 4000.0, 3000.0), (8.0, 80000.0, 500.0)])
def test_t1_update_matches_the_exact_posterior_moments(alpha, beta, t, survived):
    grid = np.linspace(1e-9, 40 * alpha / beta, 400001)
    prior = grid ** (alpha - 1) * np.exp(-beta * grid)
    likelihood = np.exp(-grid * t) if survived else 1 - np.exp(-grid * t)
    mean, variance = _moments(grid, prior * likelihood)
    a, b = cal.t1_update(alpha, beta, survived, t)
    assert a / b == pytest.approx(mean, rel=1e-6)
    assert a / b**2 == pytest.approx(variance, rel=1e-5)


def test_t1_tracking_converges_on_simulated_shots():
    rng = np.random.default_rng(3)
    true_t1 = 20_000.0
    alpha, beta = 1.0, 5_000.0
    for _ in range(400):
        t = cal.t1_wait(alpha, beta)
        alpha, beta = cal.t1_update(alpha, beta, rng.random() < math.exp(-t / true_t1), t)
    estimate = beta / alpha
    assert estimate == pytest.approx(true_t1, rel=0.15)
    assert estimate / math.sqrt(alpha) < 0.1 * true_t1  # the posterior has narrowed


@pytest.mark.parametrize("outcome", [0, 1])
@pytest.mark.parametrize(
    ("mean", "sigma", "t", "phase"), [(1e5, 5e5, 500.0, 0.3), (-2e6, 1e6, 80.0, 2.0)]
)
def test_frequency_update_matches_the_exact_posterior_moments(mean, sigma, t, phase, outcome):
    grid = np.linspace(mean - 12 * sigma, mean + 12 * sigma, 400001)
    prior = np.exp(-0.5 * ((grid - mean) / sigma) ** 2)
    p1 = 0.5 * (1 - np.cos(2 * np.pi * grid * t * 1e-9 + phase))
    exact_mean, exact_variance = _moments(grid, prior * (p1 if outcome else 1 - p1))
    m, s = cal.frequency_update(mean, sigma, outcome, t, phase)
    assert m == pytest.approx(exact_mean, abs=1e-6 * sigma)
    assert s**2 == pytest.approx(exact_variance, rel=1e-6)


def _search(seed, shots_count, detuning=1.234e6):
    rng = np.random.default_rng(seed)
    mean, sigma = 0.0, 4e6
    for _ in range(shots_count):
        t, phase = cal.ramsey_setting(mean, sigma)
        p1 = 0.5 * (1 - math.cos(2 * math.pi * detuning * t * 1e-9 + phase))
        mean, sigma = cal.frequency_update(mean, sigma, int(rng.random() < p1), t, phase)
    return mean - detuning, sigma


def test_the_frequency_binary_search_finds_the_detuning():
    error, sigma = _search(7, 60)
    assert sigma < 1e3  # from 4 MHz, about 14% narrower per shot
    assert abs(error) < 5 * sigma
    failures = sum(abs(e) > 5 * s for e, s in (_search(seed, 60) for seed in range(200)))
    assert failures <= 4


def test_three_point_decay_is_exact_for_any_amplitude_and_offset():
    for amplitude, offset, t1 in [(0.9, 0.05, 15.0), (-0.4, 0.7, 3.0)]:
        p = [amplitude * math.exp(-(2.0 + k * 4.0) / t1) + offset for k in range(3)]
        assert cal.three_point_decay(*p, spacing=4.0) == pytest.approx(t1)
    assert cal.three_point_decay(0.5, 0.5, 0.5, 1.0) == math.inf


def test_three_point_phase_is_exact_for_any_contrast():
    for a, b, error, theta, delta in [
        (0.5, 0.5, 0.07, math.pi / 2, 0.4),
        (0.45, 0.38, -0.3, math.pi, 0.2),
    ]:
        p = [a - b * math.cos(theta + k * delta + error) for k in (-1, 0, 1)]
        assert cal.three_point_phase(*p, theta=theta, delta=delta) == pytest.approx(error)


def test_golden_section_finds_a_minimum_in_logarithmic_steps():
    state = cal.GoldenSection.start(0.0, 1.0, tolerance=1e-4)
    x, done, evaluations = state.next_point(), False, 0
    while not done:
        state, x, done = cal.golden_section_step(state, (x - 0.3137) ** 2)
        evaluations += 1
        state = cal.GoldenSection.from_floats(state.as_floats())
    assert x == pytest.approx(0.3137, abs=1e-4)
    # (0.618)^n <= 1e-4 after about 20 interval reductions, one cost each.
    assert evaluations <= 23


def test_nelder_mead_minimizes_a_quadratic_and_its_state_packs():
    def cost(p):
        x, y = p
        return (x - 0.4) ** 2 + 3 * (y + 0.2) ** 2 + 0.5 * (x - 0.4) * (y + 0.2)

    state = cal.NelderMead.start([0.0, 0.0], step=0.3, tolerance=1e-5)
    point, done, steps = state.next_point(), False, 0
    while not done and steps < 500:
        state, point, done = cal.nelder_mead_step(state, cost(point))
        packed = state.as_floats()
        assert len(packed) == cal.NelderMead.size(2)
        state = cal.NelderMead.from_floats(packed, 2)
        steps += 1
    assert done
    assert point == pytest.approx((0.4, -0.2), abs=1e-4)


def test_nelder_mead_handles_rosenbrock():
    def cost(p):
        x, y = p
        return (1 - x) ** 2 + 100 * (y - x * x) ** 2

    state = cal.NelderMead.start([-1.2, 1.0], step=0.5, tolerance=1e-6)
    point, done, steps = state.next_point(), False, 0
    while not done and steps < 3000:
        state, point, done = cal.nelder_mead_step(state, cost(point))
        steps += 1
    assert done
    assert point == pytest.approx((1.0, 1.0), abs=1e-3)
