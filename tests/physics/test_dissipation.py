# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""T1 and T2 of a qubit and a qutrit, and the time step a qutrit needs."""

import math

import cudaq_pulse as pulse
import numpy as np
import pytest

from tests.physics._sim import basis, cycles, device, run
from tests.physics.conftest import ref


def _idle(duration):
    @pulse.kernel
    def idle(q):
        d, _ = get_drive_line(q)
        wait(d, duration)

    return idle


def _decay(entry_id, levels, initial, dt_ns, t_ns=None):
    p = ref(entry_id)["params"]
    t_ns = p["T_ns"] if t_ns is None else t_ns
    dev = device([p["f_q_hz"]], [p["alpha_hz"]], t1_us=p["T1_us"], t2_us=p["T2_us"])
    return run(
        dev, _idle(cycles(t_ns)), levels=levels, initial_state=initial, dt_ns=dt_ns
    ).density_matrix


def test_t1_of_a_qubit():
    entry = ref("t1_qubit")
    rho = _decay("t1_qubit", 2, basis(2, 1), entry["tolerance"]["dt_ns"])
    assert rho[1, 1].real == pytest.approx(
        entry["expected"]["analytic_P1"], rel=entry["tolerance"]["rel"]
    )


def test_t2_of_a_qubit():
    entry = ref("t2_qubit")
    plus = (basis(2, 0) + basis(2, 1)) / math.sqrt(2.0)
    rho = _decay("t2_qubit", 2, plus, entry["tolerance"]["dt_ns"])
    assert abs(rho[0, 1]) == pytest.approx(
        entry["expected"]["analytic_abs_rho01"], rel=entry["tolerance"]["rel"]
    )


def test_t1_of_a_qutrit_from_2_follows_the_harmonic_ladder():
    """|2> decays at 2 / T1. Populations alone carry no coherence that the
    anharmonicity turns, so a 1 ns step resolves them."""
    entry = ref("t1_qutrit_from_2")
    rho = _decay("t1_qutrit_from_2", 3, basis(3, 2), dt_ns=1.0)
    assert rho[2, 2].real == pytest.approx(
        entry["expected"]["harmonic_ladder_P2"], rel=entry["tolerance"]["rel"]
    )


def _rho02(t_ns, t1_us, t2_us):
    gamma1 = 1.0 / (t1_us * 1e3)
    gamma_phi = 1.0 / (t2_us * 1e3) - gamma1 / 2.0
    return 0.5 * math.exp(-(gamma1 + 4.0 * gamma_phi) * t_ns)


def test_a_qutrit_coherence_needs_a_step_that_resolves_the_anharmonicity():
    """rho02 turns at the anharmonicity, 1.26 rad/ns here: RK4 damps it
    unless |alpha| dt << 1, and converges to the analytic decay as dt
    shrinks."""
    entry = ref("t2_qutrit_02")
    p = entry["params"]
    psi = (basis(3, 0) + basis(3, 2)) / math.sqrt(2.0)
    t_ns = 500.0
    exact = _rho02(t_ns, p["T1_us"], p["T2_us"])
    errors = [
        abs(abs(_decay("t2_qutrit_02", 3, psi, dt, t_ns)[0, 2]) - exact) / exact
        for dt in (0.2, 0.1, 0.05)
    ]
    assert errors[0] > 1e-4
    # Fourth order: halving the step cuts the error about 2^4 = 16 times.
    assert errors[0] / errors[1] > 10 and errors[1] / errors[2] > 10
    assert errors[2] < entry["tolerance"]["rel"] * 10


@pytest.mark.nightly
def test_t2_of_a_qutrit_02():
    """Over 5 us, RK4 damps rho02 by (alpha dt)^6 / 144 per step: 4e-5 at
    the entry's 0.05 ns, 4e-7 at 0.02 ns."""
    entry = ref("t2_qutrit_02")
    psi = (basis(3, 0) + basis(3, 2)) / math.sqrt(2.0)
    rho = _decay("t2_qutrit_02", 3, psi, dt_ns=0.02)
    assert abs(rho[0, 2]) == pytest.approx(
        entry["expected"]["n_dephasing_abs_rho02"], rel=entry["tolerance"]["rel"]
    )
    assert np.trace(rho).real == pytest.approx(1.0, abs=1e-9)
