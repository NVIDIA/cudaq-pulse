# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Krinner et al. (2022) target against the paper's published structure."""

from __future__ import annotations

import cmath
import math
import statistics

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.targets import (
    Coupling,
    DeviceDescription,
    Qubit,
    transmon_17q,
    transmon_generic,
)
from cudaq_pulse.targets import _transmon_17q as _k
from cudaq_pulse.targets.transmon import CZ_LAYERS, transmon_17q_idle_zz_hz

from tests._handles import Q
from tests.runtime._lowering import on

# Tab. II: the data qubits of each stabilizer.
_STABILIZERS = {
    "Z1": {"D1", "D4"},
    "Z2": {"D4", "D5", "D7", "D8"},
    "Z3": {"D2", "D3", "D5", "D6"},
    "Z4": {"D6", "D9"},
    "X1": {"D2", "D3"},
    "X2": {"D1", "D2", "D4", "D5"},
    "X3": {"D5", "D6", "D8", "D9"},
    "X4": {"D7", "D8"},
}
# Main text: Z_L = Z1 Z2 Z3 and X_L = X1 X4 X7.
_Z_LOGICAL = {"D1", "D2", "D3"}
_X_LOGICAL = {"D1", "D4", "D7"}


@pytest.fixture(scope="module")
def target():
    return transmon_17q()


def _label(target, index):
    return target.qubits[index].label


def test_qudits_are_in_the_order_of_table_one(target):
    labels = [_label(target, i) for i in range(17)]
    assert labels == [f"D{i}" for i in range(1, 10)] + [
        f"{k}{i}" for k in "XZ" for i in range(1, 5)
    ]
    assert target.attribution.count("Nature 605, 669 (2022)") == 1


def test_frequency_bands_and_anharmonicities(target):
    for q in target.qubits.values():
        band = (3.7e9, 4.15e9) if q.label.startswith("D") else (5.88e9, 6.33e9)
        assert band[0] <= q.frequency_hz <= band[1]
        assert -186e6 <= q.anharmonicity_hz <= -163e6
        assert q.t1_us > 0 and q.t2_star_us > 0
    # Data qubits about 2 GHz below the auxiliary qubits (Methods A).
    assert max(q.frequency_hz for q in target.qubits.values() if q.label[0] == "D") < min(
        q.frequency_hz for q in target.qubits.values() if q.label[0] != "D"
    )


def edges_of(target, c):
    return tuple(_label(target, q) for q in c.qudits)


def test_couplings_are_the_stabilizer_supports(target):
    edges = {edges_of(target, c) for c in target.couplings}
    assert len(target.couplings) == 24
    assert edges == {(a, d) for a, data in _STABILIZERS.items() for d in data}
    # The published 7 MHz (7.4 MHz measured on X2-D4) is the |11> <-> |20>
    # coupling; the calibration's charge coupling makes the idle exchange
    # frequency-dependent, near 7 MHz / sqrt(2).
    for c in target.couplings:
        g_11_20 = 7.4e6 if edges_of(target, c) == ("X2", "D4") else 7.0e6
        assert _k.coupling_11_20_hz(*edges_of(target, c)) == g_11_20
        assert 5.6e6 < c.strength_hz < 6.1e6
    assert len({round(c.strength_hz) for c in target.couplings}) == 24
    assert not target.residual_zz


def test_without_a_calibration_the_exchange_is_the_published_value_over_sqrt2():
    published = transmon_17q(calibration=None)
    for c in published.couplings:
        g_11_20 = _k.coupling_11_20_hz(*edges_of(published, c))
        assert c.strength_hz == pytest.approx(g_11_20 / math.sqrt(2.0))


def test_stabilizers_and_logicals_commute_as_a_surface_code():
    z = {a: s for a, s in _STABILIZERS.items() if a[0] == "Z"}
    x = {a: s for a, s in _STABILIZERS.items() if a[0] == "X"}
    assert all(len(zs & xs) % 2 == 0 for zs in z.values() for xs in x.values())
    assert all(len(_Z_LOGICAL & xs) % 2 == 0 for xs in x.values())
    assert all(len(_X_LOGICAL & zs) % 2 == 0 for zs in z.values())
    assert len(_Z_LOGICAL & _X_LOGICAL) == 1


def test_cz_layers_run_three_disjoint_gates_per_step(target):
    assert len(CZ_LAYERS) == 8
    for step, layer in enumerate(CZ_LAYERS, 1):
        assert len(layer) == 3 and len({q for pair in layer for q in pair}) == 6
        # Steps 1-4 measure the Z stabilizers, 5-8 the X stabilizers.
        assert {_label(target, a)[0] for a, _ in layer} == {"Z" if step <= 4 else "X"}
    # Every coupling once, as its (auxiliary, data) qudits, in the published step.
    pairs = [pair for layer in CZ_LAYERS for pair in layer]
    assert sorted(pairs) == sorted(c.qudits for c in target.couplings)
    published = [
        [(_k.INDEX[e.auxiliary], _k.INDEX[e.data]) for e in _k.EDGES if e.cz_step == s]
        for s in range(1, 9)
    ]
    assert [list(layer) for layer in CZ_LAYERS] == published
    for c in target.couplings:
        cz = c.get_calibration(pulse.gates.CZ)
        assert cz.interaction_frequency_hz is not None
        assert 4.3e9 <= cz.interaction_frequency_hz <= 5.61e9
        assert cz.duration_ns == 98.0 and cz.buffer_ns == 15.0


def test_gate_errors_reproduce_the_published_statistics():
    errors = [100 * e.cz_error for e in _k.EDGES]
    # 1.5(1.0) % mean, 0.6 % best and 5.4 % worst (App. B, Fig. 5).
    assert statistics.mean(errors) == pytest.approx(1.5, abs=0.1)
    assert min(errors) == pytest.approx(0.6) and max(errors) == pytest.approx(5.4)


def test_idle_zz_is_a_few_kilohertz(target):
    zz = transmon_17q_idle_zz_hz()
    assert set(zz) == {c.qudits for c in target.couplings}
    # Calculated idle ZZ "lower than ~8 kHz" (Methods A).
    assert all(-7.5e3 < v < -4.4e3 for v in zz.values())


@pytest.mark.slow
@pytest.mark.parametrize("pair", [("X2", "D2"), ("Z1", "D1"), ("X2", "D4")])
def test_a_coupled_pair_idles_with_its_dispersive_zz(target, pair):
    index = {q.label: i for i, q in target.qubits.items()}
    a, d = (index[label] for label in pair)
    qubits = {
        k: Qubit(k, q.frequency_hz, q.anharmonicity_hz, 0.0, 0.0)
        for k, q in enumerate((target.qubits[a], target.qubits[d]))
    }
    (g_hz,) = (c.strength_hz for c in target.couplings if c.qudits == (a, d))
    pair_target = DeviceDescription(
        name="transmon-17q-pair",
        qubits=qubits,
        couplings=[Coupling((0, 1), strength_hz=g_hz)],
    )

    @pulse.kernel
    def idle(q0, q1):
        d0 = get_drive_line(q0)
        get_drive_line(q1)  # simulate the idle partner
        drive(d0, square(2, 0.0))

    lowered = on(pair_target, idle, Q[0], Q[1])
    t_ns, levels = 1500.0, 3
    phases = []
    for k in (0, 1):
        psi = np.zeros(levels * levels, dtype=complex)
        psi[levels * k] = psi[1 + levels * k] = 1.0 / math.sqrt(2.0)
        rho = pulse.simulate(
            lowered, levels=levels, initial_state=psi, t_end=t_ns, dt_ns=0.02
        ).density_matrix
        phases.append(cmath.phase(rho[levels * k, 1 + levels * k]))
        # The excitation stays on the auxiliary qubit, up to the weak
        # hybridization of |11> with |20> and |02>.
        assert rho[1 + levels * k, 1 + levels * k].real == pytest.approx(0.5, abs=5e-4)
    zeta = (phases[1] - phases[0]) / (2.0 * math.pi * t_ns * 1e-9)
    # The simulation's Duffing ladder with exchange J reproduces the
    # second-order zeta = 2 J^2 (a_a + a_d) / ((D + a_d)(D - a_a)), and the
    # charge model within its n_12 / n_01 and counter-rotating corrections
    # (2-6 % on these pairs).
    alpha_a, alpha_d = (target.qubits[i].anharmonicity_hz for i in (a, d))
    detuning = target.qubits[d].frequency_hz - target.qubits[a].frequency_hz
    second_order = 2 * g_hz**2 * (alpha_a + alpha_d) / ((detuning + alpha_d) * (detuning - alpha_a))
    assert zeta == pytest.approx(second_order, rel=2e-2)
    assert zeta == pytest.approx(transmon_17q_idle_zz_hz()[(a, d)], rel=7e-2)


def test_echo_t2_and_thermal_populations_are_on_the_target():
    target = transmon_17q()
    echo = target.t2_echo_times
    thermal = target.thermal_populations
    assert all(t is not None and t > 0 for t in echo.values())
    assert all(0.0 <= p < 0.1 for p in thermal.values())
    assert echo[0] == _k.QUBITS[0].t2_echo_us
    assert thermal[0] == _k.QUBITS[0].thermal_population


def test_the_time_step_resolves_the_fastest_coupled_detuning():
    from cudaq_pulse.runtime.simulate import _default_step

    # X1 and D2 with their published frequencies and coupling, as qudits 0 and 1.
    x1, d2 = _k.QUBITS[_k.INDEX["X1"]], _k.QUBITS[_k.INDEX["D2"]]
    detuning = x1.idle_frequency_hz - d2.idle_frequency_hz
    assert detuning > 2.0e9
    pair = DeviceDescription(
        name="x1-d2",
        qubits={
            i: Qubit(i, q.idle_frequency_hz, q.anharmonicity_hz, 0.0, 0.0)
            for i, q in enumerate((x1, d2))
        },
        couplings=[Coupling((0, 1), _k.COUPLING_11_20_HZ / math.sqrt(2.0))],
    )
    dev = pulse.device(pair)

    @pulse.kernel
    def flip_and_idle(qa, qb):
        da = get_drive_line(qa)
        db = get_drive_line(qb)
        drive(da, square(40, math.pi / 20))
        sync(da, db)
        wait(da, 400)

    lowered = flip_and_idle.lower(dev.qudits[0], dev.qudits[1])
    step = _default_step(lowered)
    assert step <= 1e9 / (20 * detuning) and (0.5 / step) == pytest.approx(round(0.5 / step))
    # One qudit alone keeps the default step.
    assert _default_step(x90_alone.lower(dev.qudits[0])) == pytest.approx(0.05)
    automatic = pulse.simulate(lowered).density_matrix
    fine = pulse.simulate(lowered, dt_ns=0.0125).density_matrix
    with pytest.warns(pulse.PulseWarning, match="qudits 0 and 1 turns at their detuning"):
        coarse = pulse.simulate(lowered, dt_ns=0.05).density_matrix
    assert np.abs(automatic - fine).max() < np.abs(coarse - fine).max() / 10


@pulse.kernel
def x90_alone(q):
    d = get_drive_line(q)
    drive(d, square(40, 0.02))


def test_a_generic_coupling_makes_no_gate_claims():
    assert not Coupling((0, 1), strength_hz=5e6).calibrations
    assert all(not c.calibrations for c in transmon_generic(3).couplings)
