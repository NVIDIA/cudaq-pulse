# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The Krinner et al. (2022) target against the paper's published structure."""

from __future__ import annotations

import cmath
import math
import statistics

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.targets import Coupling, Qubit, Target, transmon_krinner_17q
from cudaq_pulse.targets.transmon import krinner_idle_zz_hz

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
    return transmon_krinner_17q()


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


def test_couplings_are_the_stabilizer_supports(target):
    edges = {(_label(target, c.qubit_a), _label(target, c.qubit_b)) for c in target.couplings}
    assert len(target.couplings) == 24
    assert edges == {(a, d) for a, data in _STABILIZERS.items() for d in data}
    assert all(c.coupling_strength_hz == 7.0e6 for c in target.couplings)
    assert not target.crosstalk


def test_stabilizers_and_logicals_commute_as_a_surface_code():
    z = {a: s for a, s in _STABILIZERS.items() if a[0] == "Z"}
    x = {a: s for a, s in _STABILIZERS.items() if a[0] == "X"}
    assert all(len(zs & xs) % 2 == 0 for zs in z.values() for xs in x.values())
    assert all(len(_Z_LOGICAL & xs) % 2 == 0 for xs in x.values())
    assert all(len(_X_LOGICAL & zs) % 2 == 0 for zs in z.values())
    assert len(_Z_LOGICAL & _X_LOGICAL) == 1


def test_cz_schedule_runs_three_disjoint_gates_per_step(target):
    steps: dict[int, list[Coupling]] = {}
    for c in target.couplings:
        steps.setdefault(int(c.gate_params["cz_step"]), []).append(c)
    assert sorted(steps) == list(range(1, 9))
    for step, gates in steps.items():
        assert len(gates) == 3
        qudits = [q for c in gates for q in (c.qubit_a, c.qubit_b)]
        assert len(set(qudits)) == 6
        # Steps 1-4 measure the Z stabilizers, 5-8 the X stabilizers.
        assert {_label(target, c.qubit_a)[0] for c in gates} == {"Z" if step <= 4 else "X"}
    for c in target.couplings:
        assert 4.3e9 <= c.gate_params["interaction_frequency_hz"] <= 5.61e9
        assert c.gate_duration_ns == 98.0 and c.gate_buffer_ns == 15.0


def test_gate_errors_reproduce_the_published_statistics(target):
    errors = [100 * (1 - c.gate_fidelity) for c in target.couplings]
    # 1.5(1.0) % mean, 0.6 % best and 5.4 % worst (App. B, Fig. 5).
    assert statistics.mean(errors) == pytest.approx(1.5, abs=0.1)
    assert min(errors) == pytest.approx(0.6) and max(errors) == pytest.approx(5.4)


def test_idle_zz_is_a_few_kilohertz(target):
    zz = krinner_idle_zz_hz()
    assert set(zz) == {(c.qubit_a, c.qubit_b) for c in target.couplings}
    # Calculated idle ZZ "lower than ~8 kHz" (Methods A).
    assert all(-10e3 < v < -5e3 for v in zz.values())


@pytest.mark.parametrize("pair", [("X2", "D2"), ("Z1", "D1")])
def test_a_coupled_pair_idles_with_its_dispersive_zz(target, pair):
    index = {q.label: i for i, q in target.qubits.items()}
    a, d = (index[label] for label in pair)
    qubits = {
        k: Qubit(k, q.frequency_hz, q.anharmonicity_hz, 0.0, 0.0)
        for k, q in enumerate((target.qubits[a], target.qubits[d]))
    }
    pair_target = Target(
        name="krinner-pair", qubits=qubits, couplings=[Coupling(0, 1, coupling_strength_hz=7.0e6)]
    )

    @pulse.kernel
    def idle(q0, q1):
        d0, t0 = get_drive_line(q0)
        drive(d0, square(2, 0.0), t0)

    lowered = on(pair_target, idle, pulse.qudit_ref(), pulse.qudit_ref())
    t_ns, levels = 1500.0, 3
    phases = []
    for k in (0, 1):
        psi = np.zeros(levels * levels, dtype=complex)
        psi[levels * k] = psi[1 + levels * k] = 1.0 / math.sqrt(2.0)
        rho = pulse.simulate(
            lowered, levels=levels, initial_state=psi, t_end=t_ns, dt_ns=0.025
        ).density_matrix
        phases.append(cmath.phase(rho[levels * k, 1 + levels * k]))
        # The excitation stays on the auxiliary qubit, up to the weak
        # hybridization of |11> with |20> and |02>.
        assert rho[1 + levels * k, 1 + levels * k].real == pytest.approx(0.5, abs=5e-4)
    zeta = (phases[1] - phases[0]) / (2.0 * math.pi * t_ns * 1e-9)
    assert zeta == pytest.approx(krinner_idle_zz_hz()[(a, d)], rel=2e-2)
