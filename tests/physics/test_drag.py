# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""DRAG leakage against beta, with beta in clock cycles."""

import math

import cudaq_pulse as pulse
import numpy as np
import pytest

from tests.physics._sim import CLOCK_GHZ, cycles, device, run
from tests.physics.conftest import ref

ENTRY = ref("drag_leakage_vs_beta")
P = ENTRY["params"]


def _pi_pulse(beta_cycles):
    duration, sigma = cycles(P["duration_ns"]), P["sigma_ns"] * CLOCK_GHZ
    amplitude = P["amplitude_rad_per_ns"]

    @pulse.kernel
    def pi_pulse(q, beta: float):
        d = get_drive_line(q)
        drive(d, drag(duration, amplitude, sigma, beta))

    result = run(
        device([P["f_q_hz"]], [P["alpha_hz"]]),
        pi_pulse,
        beta_cycles,
        levels=3,
        dt_ns=ENTRY["tolerance"]["dt_ns"],
    )
    return result.density_matrix.diagonal().real


@pytest.mark.parametrize(
    "point", ENTRY["expected"]["points"], ids=lambda p: f"{p['beta_ns']:.3f}ns"
)
def test_leakage_and_transfer_against_beta(point):
    populations = _pi_pulse(point["beta_ns"] * CLOCK_GHZ)
    tolerance = ENTRY["tolerance"]["abs"]
    assert populations[2] == pytest.approx(point["P2"], abs=tolerance)
    assert populations[1] == pytest.approx(point["P1"], abs=tolerance)


def test_leakage_is_smallest_near_minus_one_over_alpha_in_cycles():
    """Motzoi et al. 2009: leakage ~ |1 + beta alpha|^2 at first order, so
    beta* = -1 / alpha, with alpha in radians per clock cycle."""
    alpha_per_cycle = 2.0 * math.pi * P["alpha_hz"] * 1e-9 / CLOCK_GHZ
    step = ENTRY["tolerance"]["beta_grid_ns"] * CLOCK_GHZ
    grid = np.arange(0.4, 0.85, ENTRY["tolerance"]["beta_grid_ns"]) * CLOCK_GHZ
    leakage = [_pi_pulse(beta)[2] for beta in grid]
    best = grid[int(np.argmin(leakage))]
    assert best == pytest.approx(ENTRY["expected"]["leakage_min_beta_ns"] * CLOCK_GHZ, abs=step / 2)
    # The first-order optimum, 1.27 cycles here, lies within the grid step of
    # the minimum; edge truncation of the 8-sigma window moves it slightly.
    assert best == pytest.approx(-1.0 / alpha_per_cycle, abs=step)
