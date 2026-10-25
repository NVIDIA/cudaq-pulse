# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The in-model CZ knobs of the shipped Krinner calibration.

The sudden-net-zero pulse (Negirneac et al., PRL 126, 220502 (2021)): the
data qubit at the interaction frequency and the auxiliary's 1-2 transition
at that frequency plus the amplitude offset, for t_p each in two plateaus
around a t_mid transition section in which the data qubit sits at the flux
Phi_idle + B (Phi_A - Phi_idle) and the auxiliary at idle; Gaussian edges
of sigma = 0.5 ns. Evolved with the charge-coupled pair of the transmon
model and the coupling recorded with the knobs, it reproduces the
calibrated conditional phase.
"""

import math
import statistics

import numpy as np
import pytest
from cudaq_pulse.targets import _krinner17 as _k
from cudaq_pulse.targets.calibration import Calibration

SIGMA_NS = 0.5
BUFFER_NS = 3.0
LEVELS = 5


@pytest.fixture(scope="module")
def cal():
    return Calibration.load("krinner17")


def _knobs(cal):
    return cal._data["cz"]["knobs"]["per_pair"]


def test_every_pair_has_its_knobs_with_provenance(cal):
    knobs = _knobs(cal)
    assert set(knobs) == {f"{e.auxiliary}-{e.data}" for e in _k.EDGES}
    for pair, record in knobs.items():
        for name in (
            "duration_plateau",
            "duration_transition",
            "transition_amplitude_B",
            "amplitude_offset_detuning",
            "predicted_phi_c",
            "predicted_leakage",
        ):
            entry = cal.entry("cz", "knobs", "per_pair", pair, name)
            assert entry["kind"] == "fitted", (pair, name)
            assert entry["cooldown"] in cal.cooldowns, (pair, name)
        assert abs(record["predicted_phi_c"]["value"] - 180.0) <= 1.0, pair
        assert isinstance(record["flags"], list)
        assert record["coupling_used"]["g_c_mhz"] > 0


def test_durations_reproduce_the_published_mean(cal):
    # 2 t_p + t_mid: 68(7) ns on average (Krinner et al., App. B).
    totals = [r["flux_pulse_total"]["value"] * 1e9 for r in _knobs(cal).values()]
    assert statistics.mean(totals) == pytest.approx(65.0, abs=0.1)
    assert abs(statistics.mean(totals) - 68.0) < 7.0


def test_the_flagged_pairs_keep_their_labels_and_alternatives(cal):
    knobs = _knobs(cal)
    # Leakage of at least 1e-4 is limited by the 0.417 ns time grid; flagged.
    leaky = {p for p, r in knobs.items() if r["predicted_leakage"]["value"] >= 1e-4}
    flagged = {p for p, r in knobs.items() if any("leakage" in f for f in r["flags"])}
    assert leaky == flagged and len(leaky) == 9
    z2d7 = knobs["Z2-D7"]
    assert z2d7["transition_amplitude_B"]["value"] == pytest.approx(1.934, abs=1e-3)
    assert z2d7["label"].startswith("unphysical")
    assert z2d7["default_option"] and z2d7["alternatives"]


class _Table:
    """Levels and charge matrix of a transmon against f01, tabulated over flux
    and interpolated linearly (f01 is monotonic in flux)."""

    def __init__(self, circuit, points=400):
        fluxes = np.linspace(0.0, 0.5, points)
        systems = [circuit.eigensystem(f, LEVELS) for f in fluxes]
        order = np.argsort([e[1] for e, _ in systems])
        self.e = np.array([systems[i][0] for i in order]) * 1e-9
        self.n = np.array([systems[i][1] for i in order])
        self.f01 = self.e[:, 1]
        self.f12 = self.e[:, 2] - self.e[:, 1]
        self.circuit = circuit

    def at(self, f01_ghz):
        k = int(np.clip(np.searchsorted(self.f01, f01_ghz), 1, len(self.f01) - 1))
        w = (f01_ghz - self.f01[k - 1]) / (self.f01[k] - self.f01[k - 1])
        return (1 - w) * self.e[k - 1] + w * self.e[k], (1 - w) * self.n[k - 1] + w * self.n[k]

    def f01_for_f12(self, f12_ghz):
        order = np.argsort(self.f12)
        return float(np.interp(f12_ghz, self.f12[order], self.f01[order]))


def _conditional_phase(cal, pair, dt_ns=0.01):
    auxiliary, data = pair.split("-")
    record = _knobs(cal)[pair]
    aux, dat = _Table(cal.circuit(auxiliary)), _Table(cal.circuit(data))
    g_c = record["coupling_used"]["g_c_mhz"] * 1e-3
    w_int = cal.value("cz", "edges", pair, "interaction_frequency") * 1e-9
    t_p = record["duration_plateau"]["value"] * 1e9
    t_mid = record["duration_transition"]["value"] * 1e9
    b = record["transition_amplitude_B"]["value"]
    delta = record["amplitude_offset_detuning"]["value"] * 1e-9

    fa_i = cal.circuit(auxiliary).f01_hz(cal.idle_flux(auxiliary)) * 1e-9
    fd_i = cal.circuit(data).f01_hz(cal.idle_flux(data)) * 1e-9
    fa_p, fd_p = aux.f01_for_f12(w_int + delta), w_int
    phi_idle = cal.idle_flux(data)
    phi_a = cal.circuit(data).flux_for_f01(w_int * 1e9)
    fd_m = cal.circuit(data).f01_hz(phi_idle + b * (phi_a - phi_idle)) * 1e-9

    eye = np.eye(LEVELS)

    def hamiltonian(fa, fd):
        ea, na = aux.at(fa)
        ed, nd = dat.at(fd)
        return np.kron(np.diag(ea), eye) + np.kron(eye, np.diag(ed)) + g_c * np.kron(na, nd)

    t1 = BUFFER_NS
    edges = [t1, t1 + t_p, t1 + t_p + t_mid, t1 + 2 * t_p + t_mid]
    total = 2 * BUFFER_NS + 2 * t_p + t_mid

    def step(x):
        return 0.5 * (1 + math.erf(x / (math.sqrt(2) * SIGMA_NS)))

    def trajectory(t, idle, plateau, mid):
        s = [step(t - e) for e in edges]
        return idle + (plateau - idle) * (s[0] - s[1] + s[2] - s[3]) + (mid - idle) * (s[1] - s[2])

    def h_at(t):
        return hamiltonian(trajectory(t, fa_i, fa_p, fa_i), trajectory(t, fd_i, fd_p, fd_m))

    def propagator(h, duration):
        w, v = np.linalg.eigh(h)
        return (v * np.exp(-2j * np.pi * w * duration)) @ v.T

    windows = []
    for lo, hi in sorted((max(0.0, e - 4 * SIGMA_NS), min(total, e + 4 * SIGMA_NS)) for e in edges):
        if windows and lo <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(hi, windows[-1][1]))
        else:
            windows.append((lo, hi))
    u = np.eye(LEVELS * LEVELS, dtype=complex)
    t = 0.0
    for lo, hi in [*windows, (total, total)]:
        if lo > t:
            u = propagator(h_at(0.5 * (t + lo)), lo - t) @ u
        n = max(1, math.ceil((hi - lo) / dt_ns))
        for k in range(n):
            u = propagator(h_at(lo + (k + 0.5) * (hi - lo) / n), (hi - lo) / n) @ u
        t = hi

    # Conditional phase in the dressed idle basis, auxiliary first.
    w, v = np.linalg.eigh(hamiltonian(fa_i, fd_i))
    owner = np.argmax(np.abs(v) ** 2, axis=1)
    ud = v.conj().T @ u @ v
    comp = [owner[i * LEVELS + j] for i in (0, 1) for j in (0, 1)]
    phases = [np.angle(ud[c, c]) for c in comp]
    phi = math.degrees((phases[3] - phases[2] - phases[1] + phases[0]) % (2 * math.pi))
    leakage = 1 - sum(abs(ud[c, comp[3]]) ** 2 for c in comp)
    return phi, leakage


@pytest.mark.parametrize("pair", ["X2-D4", "Z1-D1"])
def test_the_knobs_reproduce_a_cz(cal, pair):
    record = _knobs(cal)[pair]
    phi, leakage = _conditional_phase(cal, pair)
    assert phi == pytest.approx(record["predicted_phi_c"]["value"], abs=1.0)
    assert phi == pytest.approx(180.0, abs=1.5)
    assert leakage < 1e-3
