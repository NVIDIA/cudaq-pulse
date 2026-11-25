# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dressed frames: in a system with couplings, each qudit's drive tone plays
at its dressed 0-1 frequency, the bare frequency_hz moved by its couplings,
which the Hamiltonian keeps bare."""

from __future__ import annotations

import math

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.quantum_systems.base import (
    _exchange_pairs,
    _single_excitation,
    dressed_frequencies,
    layout_frequencies,
)


@pytest.fixture(scope="module")
def surface17():
    return pulse.quantum_system("transmon_surface17")


def _shift_khz(qs, label):
    q = qs.qudit(label)
    return (q.dressed_frequency_hz - qs._description.qudits[q.index].frequency_hz) / 1e3


@pytest.mark.parametrize(
    ("label", "khz"),
    [("D1", -31.1), ("D5", -63.0), ("D9", -38.5), ("X3", +66.5), ("Z2", +64.0), ("Z4", +33.5)],
)
def test_a_coupled_qudit_is_dressed_by_all_its_couplings(surface17, label, khz):
    assert _shift_khz(surface17, label) == pytest.approx(khz, abs=0.05)


def test_the_dressed_frequency_is_the_exact_single_excitation_eigenvalue():
    # Two transmons and their exchange, diagonalised in full at 6 levels each:
    # D5's dressed 0-1 frequency is E(1, 0) - E(0, 0), whatever the levels.
    full = pulse.quantum_system("transmon_surface17")._description
    d, z = full.qudits[4], full.qudits[14]
    (pair,) = [c for c in full.couplings if set(c.qudits) == {4, 14}]
    g, levels = pair.strength_hz, 6
    a = np.diag(np.sqrt(np.arange(1, levels)), 1)
    eye = np.eye(levels)

    def duffing(p):
        n = np.arange(levels)
        return np.diag(n * p.frequency_hz + n * (n - 1) / 2 * p.anharmonicity_hz)

    h = np.kron(duffing(d), eye) + np.kron(eye, duffing(z))
    h += g * (np.kron(a.T, a) + np.kron(a, a.T))
    values, vectors = np.linalg.eigh(h)

    def energy(i, j):
        bare = np.zeros(levels * levels)
        bare[i * levels + j] = 1.0
        return values[np.argmax((vectors.T @ bare) ** 2)]

    reference = energy(1, 0) - energy(0, 0)
    alone = pulse.quantum_system(name="pair", qudits={4: d, 14: z}, couplings=[pair])
    assert alone.qudits[0].dressed_frequency_hz == pytest.approx(reference, abs=0.01)
    assert reference - d.frequency_hz == pytest.approx(-15_151.3, abs=0.1)
    # Second order: J^2 / Delta.
    assert reference - d.frequency_hz == pytest.approx(
        g**2 / (d.frequency_hz - z.frequency_hz), rel=1e-3
    )


def test_a_system_without_couplings_is_left_as_it_is():
    qs = pulse.quantum_system("transmon_lattice:1")
    description = qs._description
    assert not description.couplings
    assert qs._dressed is None and "_dressed" not in vars(qs)
    # Its line tones are the class's own method, not a dressed one.
    assert "_line_tones" not in vars(qs)
    assert qs.frequencies == description.frequencies
    assert qs.qudits[0].dressed_frequency_hz == description.qudits[0].frequency_hz


def test_a_drive_tone_plays_at_the_dressed_frequency(surface17):
    d5 = surface17.qudit("D5")
    default, tones = surface17._line_tones(d5.index, False)
    assert default == (0, 1)
    assert tones[0, 1] == d5.dressed_frequency_hz
    assert surface17.frequencies[d5.index] == d5.dressed_frequency_hz
    # Its readout tone is the resonator's, not moved.
    _, readout = surface17._line_tones(d5.index, True)
    assert readout[0, 1] == surface17._description.qudits[d5.index].readout.frequency_hz


def test_the_dressed_table_is_computed_once_per_system(surface17):
    description = surface17._description
    assert dressed_frequencies(description) is dressed_frequencies(description)


def test_a_simulated_part_of_the_system_is_dressed_as_on_the_whole(surface17):
    # D5 and Z2 alone keep their one coupling: given these frequencies, that
    # coupling dresses them to their frequencies on the whole system.
    description = surface17._description
    given = layout_frequencies(description, (4, 14))
    within = {**{q: description.qudits[q].frequency_hz for q in (4, 14)}, **given}
    pairs = [(q, s) for q, s in _exchange_pairs(description) if set(q) <= {4, 14}]
    dressed = dressed_frequencies(description)
    for q, hz in _single_excitation(within, pairs).items():
        assert hz == pytest.approx(dressed[q], abs=1e-3)
    # The whole system, simulated, needs no moving.
    assert layout_frequencies(description, tuple(description.qudits)) == {}


@pulse.kernel
def _ramsey(d, z, t: int, quarter: int):
    dl = get_drive_line(d)
    zl = get_drive_line(z)
    # Z2 is in the program, so the simulation keeps it and its coupling.
    sync(dl, zl)
    drive(dl, dl.get_calibration(pulse.gates.X90))
    wait(dl, t)
    for _ in range(quarter):
        shift_phase(dl.tone, math.pi / 2)
    drive(dl, dl.get_calibration(pulse.gates.X90))


def test_a_ramsey_on_a_coupled_qudit_shows_no_offset(surface17):
    # D5 beside Z2 (in its ground level), the rest of surface17 left out, no
    # decoherence: at its default frame D5 no longer precesses at its dressed
    # shift (-15 kHz from Z2, -63 kHz from all its neighbours, which bare
    # frames showed).
    d5, z2 = surface17.qudit("D5").index, surface17.qudit("Z2").index
    base = pulse.quantum_system("transmon_surface17", layout=[d5, z2])
    quiet = {
        i: q.replace(t1_us=0, t2_star_us=0, t2_echo_us=0) for i, q in base.physical_qudits.items()
    }
    qs = base.replace(qudits=quiet)
    d, z = qs.qudit("D5"), qs.qudit("Z2")
    cycle_ns = 1 / qs.clock_ghz

    def phase(t_ns):
        population = []
        for quarter in (0, 1):
            program = pulse.compile(_ramsey, quantum_system=qs).lower(
                d, z, round(t_ns / cycle_ns), quarter
            )
            result = pulse.simulate(program, quantum_system=qs, dt_ns=1.0, levels=2)
            population.append(result.level_population(1, qudit=d.index))
        return math.atan2(2 * population[1] - 1, 2 * population[0] - 1)

    turned = math.remainder(phase(12_000) - phase(2_000), 2 * math.pi)
    assert abs(turned / (2 * math.pi * 10e-6)) < 100


def test_bare_frames_keep_the_drive_tones_at_frequency_hz():
    bare = pulse.quantum_system("transmon_surface17", frames="bare")
    dressed = pulse.quantum_system("transmon_surface17")
    d5 = bare.qudit("D5")
    assert bare._dressed is None and "_line_tones" not in vars(bare)
    assert d5.dressed_frequency_hz == bare.physical_qudits[d5.index].frequency_hz
    assert bare._line_tones(d5.index, False)[1][0, 1] == d5.dressed_frequency_hz
    # A different system: another fingerprint, as its programs differ.
    assert bare.fingerprint != dressed.fingerprint and bare != dressed
    # The registered name stays, and a system file says which frames.
    data = bare.as_dict()
    assert data["name"] == "transmon_surface17" and data["frames"] == "bare"
    assert "frames" not in dressed.as_dict()
    assert pulse.QuantumSystem.from_dict(data) == bare


def test_frames_is_a_keyword_of_a_built_system_and_of_replace():
    surface = pulse.quantum_system("transmon_surface17")
    built = pulse.quantum_system(
        name="pair",
        qudits={4: surface.physical_qudits[4], 14: surface.physical_qudits[14]},
        couplings=[c for c in surface.couplings if set(c.qudits) == {4, 14}],
        frames="bare",
    )
    assert built._dressed is None
    assert built.replace(frames="dressed")._dressed is not None
    with pytest.raises(pulse.QuantumSystemDefinitionError, match="frames"):
        pulse.quantum_system(name="x", qudits=dict(built.physical_qudits), frames="lab")


def test_dressed_frames_keep_the_fingerprints_systems_had():
    # The default adds nothing to what a fingerprint hashes.
    plain = pulse.quantum_system("transmon_lattice:1")
    assert (
        plain.fingerprint == pulse.quantum_system("transmon_lattice:1", frames="bare").fingerprint
    )
