# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Thermal populations and Hahn-echo dephasing, against detailed balance.

A qudit with excited-state population p at equilibrium relaxes down at
(1 - p) / T1 and up at p / T1, so populations relax to p at the measured
rate 1 / T1 and coherences still decay at 1 / T2.
"""

import math

import cudaq_pulse as pulse
import numpy as np
import pytest

from tests.physics._sim import basis, cycles, device, run

P, T1_US, T2_US, T2_ECHO_US = 0.08, 10.0, 12.0, 18.0


def _idle(duration):
    @pulse.kernel
    def idle(q):
        d = get_drive_line(q)
        wait(d, duration)

    return idle


def _thermal_qubit(**qubit):
    return device([5.0e9], [-200e6], t1_us=T1_US, t2_us=T2_US, thermal_population=P, **qubit)


def test_a_qubit_starts_in_its_thermal_mixture_and_stays_there():
    rho = run(_thermal_qubit(), _idle(cycles(2000.0)), dt_ns=1.0).density_matrix
    assert np.allclose(rho, np.diag([1.0 - P, P]), atol=1e-12)


def test_a_qutrit_starts_in_its_thermal_mixture_and_stays_there():
    """Higher levels follow the Boltzmann ratio p / (1 - p) per level, which
    the ladder's downward and upward jumps keep in detailed balance."""
    r = P / (1.0 - P)
    weights = np.array([1.0, r, r * r]) / (1.0 + r + r * r)
    rho = run(_thermal_qubit(), _idle(cycles(2000.0)), levels=3, dt_ns=0.05).density_matrix
    assert np.allclose(rho, np.diag(weights), atol=1e-10)


@pytest.mark.parametrize("start", [0, 1])
def test_populations_relax_to_the_thermal_population_at_1_over_t1(start):
    """An explicit initial state replaces the thermal one; from it P1 relaxes
    as p + (P1(0) - p) exp(-t / T1)."""
    t_ns = 5000.0
    rho = run(
        _thermal_qubit(), _idle(cycles(t_ns)), initial_state=basis(2, start), dt_ns=1.0
    ).density_matrix
    expected = P + (start - P) * math.exp(-t_ns / (T1_US * 1e3))
    assert rho[1, 1].real == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize(("dephasing", "t2_us"), [("t2_star", T2_US), ("t2_echo", T2_ECHO_US)])
def test_coherence_decays_at_the_chosen_t2(dephasing, t2_us):
    t_ns = 4000.0
    plus = (basis(2, 0) + basis(2, 1)) / math.sqrt(2.0)
    rho = run(
        _thermal_qubit(t2_echo_us=T2_ECHO_US),
        _idle(cycles(t_ns)),
        initial_state=plus,
        dt_ns=1.0,
        dephasing=dephasing,
    ).density_matrix
    assert abs(rho[0, 1]) == pytest.approx(0.5 * math.exp(-t_ns / (t2_us * 1e3)), rel=1e-9)


def test_echo_dephasing_needs_an_echo_time():
    with pytest.raises(ValueError, match="Hahn-echo T2"):
        run(_thermal_qubit(), _idle(cycles(10.0)), dephasing="t2_echo")
