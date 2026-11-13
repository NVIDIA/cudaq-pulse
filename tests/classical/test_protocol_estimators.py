# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The on-controller calibration estimators against exact results and the
papers' numbers.

Marciniak et al. (arXiv:2602.11912) report T1 = 18.3 +- 4.7 us from the
analytical decay estimator with 50 shots per delay; Berritta et al.
(arXiv:2506.09576) report T1 = 159 us, 68% interval [131, 202] us, after 50
adaptive shots from the prior (3, 450 us) with assignment errors 0.11 and
0.14. The binary search's posterior narrows by the factor its equations
give per shot: from 1 MHz to about 0.37 MHz in 15 shots.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from cudaq_pulse.classical import calibration as cal
from cudaq_pulse.controller import OpCounts

# ── Adaptive T1 ─────────────────────────────────────────────────────────────

_K0, _THETA0 = 3.0, 450_000.0  # shape, rate (ns)
_ALPHA, _BETA = 0.11, 0.14


def _p_excited(wait, t1):
    return _BETA + (1 - _ALPHA - _BETA) * math.exp(-wait / t1)


@pytest.mark.parametrize("excited", [0, 1])
def test_the_t1_update_matches_the_exact_posterior_moments(excited):
    wait = 200_000.0
    rates = np.linspace(1e-9, 80e-6, 400_001)
    prior = rates ** (_K0 - 1) * np.exp(-_THETA0 * rates)
    likely = _BETA + (1 - _ALPHA - _BETA) * np.exp(-rates * wait)
    posterior = prior * (likely if excited else 1 - likely)
    mean = np.sum(rates * posterior) / np.sum(posterior)
    variance = np.sum(rates**2 * posterior) / np.sum(posterior) - mean**2
    shape, rate = cal.adaptive_t1_update(_K0, _THETA0, excited, wait, _ALPHA, _BETA)
    assert shape / rate == pytest.approx(mean, rel=1e-6)
    assert shape / rate**2 == pytest.approx(variance, rel=1e-5)


def test_adaptive_t1_reaches_the_papers_interval_in_fifty_shots():
    rng = np.random.default_rng(11)
    rows = 3000
    estimates, shapes, rates = np.empty(rows), np.empty(rows), np.empty(rows)
    for i in range(rows):
        shape, rate = _K0, _THETA0
        for _ in range(50):
            wait = cal.adaptive_t1_wait(shape, rate, 0.51)
            bit = int(rng.random() < _p_excited(wait, 160_000.0))
            shape, rate = cal.adaptive_t1_update(shape, rate, bit, wait, _ALPHA, _BETA)
        estimates[i], shapes[i], rates[i] = rate / shape / 1e3, shape, rate
    median = int(np.argsort(estimates)[rows // 2])
    draws = 1e-3 / rng.gamma(shapes[median], 1.0 / rates[median], 100_000)
    lo, hi = np.quantile(draws, [0.16, 0.84])
    # The paper: 159 us, [131, 202] us; 156 us, [125, 209] us for the
    # reference implementation of the same equations at a true 160 us.
    assert 150 < np.median(estimates) < 163
    assert 118 < lo < 136 and 195 < hi < 222
    assert 0.2 < np.sqrt(np.mean((estimates / 160.0 - 1) ** 2)) < 0.3


# ── Frequency binary search ─────────────────────────────────────────────────


@pytest.mark.parametrize("outcome", [-1, 1])
def test_the_binary_search_update_matches_the_exact_posterior_moments(outcome):
    mean0, sigma0, wait = 30e3, 400e3, 300.0
    offsets = np.linspace(mean0 - 8 * sigma0, mean0 + 8 * sigma0, 200_001)
    prior = np.exp(-0.5 * ((offsets - mean0) / sigma0) ** 2)
    t = wait * 1e-9
    plus = 0.5 + 0.5 * (
        -0.02 + 0.6 * math.exp(-wait / 1e4) * np.sin(2 * np.pi * (offsets - mean0) * t)
    )
    posterior = prior * (plus if outcome == 1 else 1 - plus)
    mean = np.sum(offsets * posterior) / np.sum(posterior)
    variance = np.sum(offsets**2 * posterior) / np.sum(posterior) - mean**2
    found, sigma = cal.binary_search_update(mean0, sigma0, outcome, wait)
    assert found == pytest.approx(mean, rel=1e-6)
    assert sigma**2 == pytest.approx(variance, rel=1e-6)


def test_the_binary_search_wait_and_phase():
    sigma, coherence = 1e6, 10_000.0
    s2 = sigma**2
    expected = (math.sqrt(16 * math.pi**2 * s2 + 1e18 / coherence**2) - 1e9 / coherence) / (
        8 * math.pi**2 * s2
    )
    assert cal.binary_search_wait(sigma, coherence) == pytest.approx(expected * 1e9)
    # The phase puts the estimate on the fringe's steepest point.
    wait = 250.0
    phase = cal.binary_search_phase(40e3, wait)
    assert math.cos(2 * math.pi * 40e3 * wait * 1e-9 + phase) == pytest.approx(0.0, abs=1e-12)


def test_the_binary_search_narrows_as_its_equations_give():
    rng = np.random.default_rng(12)
    rows = 3000
    sigmas, errors = np.empty(rows), np.empty(rows)
    for i in range(rows):
        truth, mean, sigma = rng.normal(0.0, 1e6), 0.0, 1e6
        for _ in range(15):
            wait = cal.binary_search_wait(sigma)
            p = 0.5 + 0.5 * (
                -0.02
                + 0.6 * math.exp(-wait / 1e4) * math.sin(2 * math.pi * (truth - mean) * wait * 1e-9)
            )
            outcome = 1 if rng.random() < p else -1
            mean, sigma = cal.binary_search_update(mean, sigma, outcome, wait)
        sigmas[i], errors[i] = sigma, mean - truth
    # The reference implementation of the same equations: 366 kHz, and an
    # rms error of 647 kHz (the Gaussian posterior's tails are too thin).
    assert 355e3 < np.median(sigmas) < 378e3
    assert 560e3 < np.sqrt(np.mean(errors**2)) < 720e3


# ── Analytical decay estimator ──────────────────────────────────────────────


@pytest.mark.parametrize(("t1", "spacing"), [(18.3e3, 18.3e3), (30e3, 12e3), (5e3, 9e3)])
@pytest.mark.parametrize(("amplitude", "offset"), [(0.9, 0.04), (-0.5, 0.7)])
def test_the_decay_estimator_is_exact_whatever_the_amplitude_and_offset(
    t1, spacing, amplitude, offset
):
    t0 = 16.0
    p = [offset + amplitude * math.exp(-t / t1) for t in (t0, t0 + spacing, t0 + 3 * spacing)]
    estimate, _, ok = cal.decay_time(*p, spacing, 50)
    assert ok and estimate == pytest.approx(t1, rel=1e-12)


def test_the_decay_estimator_reports_points_that_do_not_decay():
    assert cal.decay_time(0.5, 0.5, 0.3, 100.0, 50)[2] is False
    assert cal.decay_time(0.9, 0.5, 0.95, 100.0, 50)[2] is False
    assert cal.decay_time(0.9, 0.5, -1.0, 100.0, 50)[2] is False


def test_the_decay_estimators_spread_matches_the_papers_and_its_own_error():
    rng = np.random.default_rng(13)
    t1 = spacing = 18.3e3
    ts = np.array([16.0, 16.0 + spacing, 16.0 + 3 * spacing])
    p = 0.05 + 0.85 * np.exp(-ts / t1)
    counts = rng.binomial(50, p, (20_000, 3)) / 50
    found = [cal.decay_time(*row, spacing, 50) for row in counts]
    estimates = np.array([t for t, _, ok in found if ok])
    errors = np.array([s for _, s, ok in found if ok])
    assert len(estimates) > 0.99 * len(found)
    lo, median, hi = np.quantile(estimates, [0.16, 0.5, 0.84])
    # The paper's single estimate: 18.3 +- 4.7 us.
    assert 17e3 < median < 19.5e3
    assert 3.5e3 < (hi - lo) / 2 < 7e3
    # The propagated error tracks the spread around the median.
    assert 0.8 < np.median(errors) / ((hi - lo) / 2) < 1.25


# ── Sparse phase estimator ──────────────────────────────────────────────────


@pytest.mark.parametrize("theta0", [-2.5, -0.3, 0.0, 0.7, 2.9])
def test_the_sparse_phase_estimator_is_exact_whatever_the_amplitude_and_offset(theta0):
    a, c = 0.37, 0.48
    p = [a * math.cos(theta0 + d) + c for d in (-math.pi / 2, 0.0, math.pi / 2)]
    theta, _ = cal.sparse_phase(*p, 100)
    assert theta == pytest.approx(theta0, abs=1e-12)


def test_the_sparse_phase_error_matches_its_spread():
    rng = np.random.default_rng(14)
    theta0, shots_count = 0.4, 64
    p = np.array([0.5 - 0.45 * math.cos(theta0 + d) for d in (-math.pi / 2, 0.0, math.pi / 2)])
    counts = rng.binomial(shots_count, p, (20_000, 3)) / shots_count
    found = [cal.sparse_phase(*row, shots_count) for row in counts]
    thetas = np.array([t for t, _ in found])
    assert np.std(thetas) == pytest.approx(np.median([s for _, s in found]), rel=0.15)


def _train_estimate(error, turns):
    # P(1) = (1 - cos R) / 2 for a train that rotates by R.
    rotation = turns * math.pi * (1 + error)
    p = [0.5 - 0.5 * math.cos(rotation * s) for s in (1 - 0.5 / turns, 1.0, 1 + 0.5 / turns)]
    theta, _ = cal.sparse_phase(*p, 64)
    return cal.pi_train_amplitude(1.0, theta, turns)


@pytest.mark.parametrize("error", [-0.02, -0.005, 0.0, 0.01, 0.02])
@pytest.mark.parametrize("turns", [21, 41])
def test_a_pulse_train_corrects_the_amplitude(error, turns):
    assert _train_estimate(error, turns) == pytest.approx(1 / (1 + error), abs=2e-4)


def test_a_pulse_train_captures_errors_below_one_over_its_turns():
    # 41 turns at 2.5% puts the train's phase past pi: the estimate aliases.
    assert abs(_train_estimate(0.025, 41) - 1 / 1.025) > 0.01
    assert _train_estimate(0.025, 21) == pytest.approx(1 / 1.025, abs=5e-4)


@pytest.mark.parametrize("detuning", [-180e3, -60e3, 0.0, 25e3, 200e3])
def test_ramsey_phases_give_the_detuning(detuning):
    wait, mean = 2000.0, 10e3
    phase0 = -2 * math.pi * mean * wait * 1e-9
    p = [
        (1 - math.cos(2 * math.pi * detuning * wait * 1e-9 + phase0 + d)) / 2
        for d in (-math.pi / 2, 0.0, math.pi / 2)
    ]
    theta, _ = cal.sparse_phase(*p, 64)
    assert cal.ramsey_detuning(mean, theta, wait) == pytest.approx(detuning, abs=1e-6)


def test_counting_ones():
    assert cal.count_ones([1, 0, 1, 1, 0]) == 3
    assert cal.count_ones.controller_cost(150) == OpCounts(popcount=5, add=5)
