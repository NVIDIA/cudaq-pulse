# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Readout crosstalk declared on a quantum system: ReadoutCrosstalk entries,
their checks, the fingerprint, pickling and the system file."""

from __future__ import annotations

import pickle

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.quantum_systems import ReadoutCrosstalk
from cudaq_pulse.quantum_systems.base import Offset

Z2_ON_D5 = ReadoutCrosstalk(resonator_of=14, qudit=4, chi_hz=-20e3)


def _surface():
    return pulse.quantum_system("transmon_surface17")


def test_a_system_declares_its_readout_crosstalk_with_replace():
    qs = _surface()
    assert qs.readout_crosstalk == ()
    crossed = qs.replace(readout_crosstalk=[Z2_ON_D5])
    assert crossed.readout_crosstalk == (Z2_ON_D5,)
    # A variant has no calibration of its own: compare with one.
    plain = qs.replace()
    assert crossed.fingerprint != plain.fingerprint
    # Other changes keep the entries.
    assert crossed.replace(attribution="moved").readout_crosstalk == (Z2_ON_D5,)
    assert crossed.replace(readout_crosstalk=[]).fingerprint == plain.fingerprint


def test_an_entry_prints_and_takes_a_shift_per_level():
    assert repr(Z2_ON_D5) == "ReadoutCrosstalk(resonator_of=14, qudit=4, chi_hz=-20000.0)"
    by_level = ReadoutCrosstalk(resonator_of=14, qudit=4, chi_hz=[-20e3, -26e3])
    assert by_level.chi_hz == (-20e3, -26e3)
    qs = _surface().replace(readout_crosstalk=[by_level])
    assert qs.readout_crosstalk[0].chi_hz == (-20e3, -26e3)


def test_equal_entries_give_one_fingerprint():
    a = _surface().replace(readout_crosstalk=[Z2_ON_D5])
    b = _surface().replace(readout_crosstalk=[ReadoutCrosstalk(14, 4, -20e3)])
    c = _surface().replace(readout_crosstalk=[ReadoutCrosstalk(14, 4, -21e3)])
    assert a.fingerprint == b.fingerprint != c.fingerprint
    assert a == b and a != c


def test_bare_frames_and_crosstalk_together():
    qs = _surface().replace(frames="bare", readout_crosstalk=[Z2_ON_D5])
    assert qs._description.frames == "bare"
    assert qs.readout_crosstalk == (Z2_ON_D5,)
    assert qs.fingerprint not in {
        _surface().replace(frames="bare").fingerprint,
        _surface().replace(readout_crosstalk=[Z2_ON_D5]).fingerprint,
        _surface().replace().fingerprint,
    }
    assert qs.replace(frames="dressed").readout_crosstalk == (Z2_ON_D5,)


def test_every_problem_is_listed_at_its_path():
    bad = [
        ReadoutCrosstalk(resonator_of=14, qudit=14, chi_hz=-20e3),
        ReadoutCrosstalk(resonator_of=99, qudit="4", chi_hz=float("nan")),
        ReadoutCrosstalk(resonator_of=14, qudit=4, chi_hz=()),
        ReadoutCrosstalk(resonator_of=13, qudit=4, chi_hz=(-20e3, float("inf"))),
        Z2_ON_D5,
        Z2_ON_D5,
        (14, 4, -20e3),
    ]
    with pytest.raises(pulse.QuantumSystemDefinitionError) as caught:
        _surface().replace(readout_crosstalk=bad)
    found = {(p.path, p.message) for p in caught.value.problems}
    assert found == {
        (
            "readout_crosstalk[0]",
            "names qudit 14 as both the qudit read out and the one shifted",
        ),
        ("readout_crosstalk[1].resonator_of", "99 is not a qudit of the quantum system"),
        ("readout_crosstalk[1].qudit", "must be a qudit index (an int), got str"),
        ("readout_crosstalk[1].chi_hz", "nan is not a finite number of Hz per photon"),
        ("readout_crosstalk[2].chi_hz", "is an empty tuple"),
        ("readout_crosstalk[3].chi_hz[1]", "inf is not a finite number of Hz per photon"),
        ("readout_crosstalk[4]", "names the same qudits as readout_crosstalk[2]"),
        ("readout_crosstalk[5]", "names the same qudits as readout_crosstalk[2]"),
        ("readout_crosstalk[6]", "must be a ReadoutCrosstalk, got tuple"),
    }


def test_a_list_is_needed():
    with pytest.raises(pulse.InvalidTypeError, match="must be a list of ReadoutCrosstalk"):
        _surface().replace(readout_crosstalk=Z2_ON_D5)


def test_it_pickles():
    qs = _surface().replace(frames="bare", readout_crosstalk=[Z2_ON_D5])
    back = pickle.loads(pickle.dumps(qs))
    assert back.readout_crosstalk == (Z2_ON_D5,)
    assert back.fingerprint == qs.fingerprint
    description = pickle.loads(pickle.dumps(qs._description))
    assert description == qs._description and type(description) is type(qs._description)


def test_a_system_file_carries_it():
    qs = _surface().replace(readout_crosstalk=[ReadoutCrosstalk(14, 4, (-20e3, -26e3))])
    data = qs.as_dict()
    assert data["readout_crosstalk"] == [
        {"resonator_of": 14, "qudit": 4, "chi_hz": [-20000.0, -26000.0]}
    ]
    back = pulse.QuantumSystem.from_dict(data)
    assert back.readout_crosstalk == qs.readout_crosstalk
    assert back.fingerprint == qs.fingerprint
    # A system without any writes nothing.
    assert "readout_crosstalk" not in _surface().as_dict()


def test_a_system_file_with_problems_names_them():
    data = _surface().as_dict()
    data["readout_crosstalk"] = [
        {"resonator_of": 14, "qudit": 4},
        {"resonator_of": 14, "qudit": 4, "chi_hz": -1.0, "kappa": 2.0},
        7,
        {"resonator_of": 14, "qudit": 40, "chi_hz": -1.0},
    ]
    with pytest.raises(pulse.QuantumSystemDefinitionError) as caught:
        pulse.QuantumSystem.from_dict(data)
    found = {(p.path, p.message) for p in caught.value.problems}
    assert found == {
        ("readout_crosstalk[0]", "chi_hz is missing"),
        ("readout_crosstalk[1].kappa", "'kappa' is not a field of a readout crosstalk entry"),
        (
            "readout_crosstalk[2]",
            "must be an object of resonator_of, qudit and chi_hz, got int",
        ),
    }
    data["readout_crosstalk"] = [{"resonator_of": 14, "qudit": 40, "chi_hz": -1.0}]
    with pytest.raises(pulse.QuantumSystemDefinitionError) as caught:
        pulse.QuantumSystem.from_dict(data)
    assert [(p.path, p.message) for p in caught.value.problems] == [
        ("readout_crosstalk[0].qudit", "40 is not a qudit of the quantum system")
    ]


def test_a_system_without_any_is_built_as_before():
    from cudaq_pulse.quantum_systems.base import _Description

    qs = _surface()
    assert type(qs._description) is _Description
    assert type(qs.replace(attribution="x")._description) is _Description


def _offsets(frequencies, couplings=()):
    # Two qudits whose frequencies are Offsets from a shared reference (or
    # unstated, with None), each with a dispersive readout of no stated
    # frequency.
    from cudaq_pulse.quantum_systems import DispersiveReadout, PhysicalQudit
    from cudaq_pulse.quantum_systems.base import Offset

    qudits = {
        i: PhysicalQudit(
            i,
            None if f is None else Offset(f),
            50.0,
            30.0,
            anharmonicity_hz=-2.0e8,
            readout=DispersiveReadout(dispersive_shift_hz=-2.9e6, linewidth_hz=12.2e6),
        )
        for i, f in enumerate(frequencies)
    }
    return pulse.quantum_system(name="offsets", qudits=qudits, couplings=list(couplings))


@pytest.mark.parametrize("frames", ["dressed", "bare"])
def test_crosstalk_on_offsets_keeps_the_frame_and_frames(frames):
    from cudaq_pulse.quantum_systems import Coupling

    qs = _offsets((0.0, 1.5e8), [Coupling((0, 1), strength_hz=5e6)]).replace(frames=frames)
    entry = ReadoutCrosstalk(resonator_of=1, qudit=0, chi_hz=-20e3)
    crossed = qs.replace(readout_crosstalk=[entry])
    assert crossed.frequency_frame == "shared_reference"
    assert crossed._description.frames == frames
    assert crossed.readout_crosstalk == (entry,)
    assert crossed.replace(attribution="x").frequency_frame == "shared_reference"
    back = pickle.loads(pickle.dumps(crossed._description))
    assert back == crossed._description and type(back) is type(crossed._description)
    again = pickle.loads(pickle.dumps(crossed))
    assert again.fingerprint == crossed.fingerprint
    assert crossed.replace(readout_crosstalk=[]).frequency_frame == "shared_reference"
    assert crossed.replace(readout_crosstalk=[])._description.frames == frames


def test_crosstalk_without_frequencies():
    qs = _offsets((None, None)).replace(
        readout_crosstalk=[ReadoutCrosstalk(resonator_of=1, qudit=0, chi_hz=-20e3)]
    )
    assert qs.frequency_frame == "own_carrier"
    assert qs.readout_crosstalk[0].qudit == 0
    back = pickle.loads(pickle.dumps(qs))
    assert back.fingerprint == qs.fingerprint and back.frequency_frame == "own_carrier"


def test_equal_shifts_spelled_as_ints_are_the_same_entry():
    assert ReadoutCrosstalk(14, 4, -20000) == ReadoutCrosstalk(14, 4, -20e3)
    a = _surface().replace(readout_crosstalk=[ReadoutCrosstalk(14, 4, (-20000, -26000))])
    b = _surface().replace(readout_crosstalk=[ReadoutCrosstalk(14, 4, (-20e3, -26e3))])
    assert a.fingerprint == b.fingerprint


@pytest.mark.parametrize("given", [None, 0, "x"])
def test_readout_crosstalk_that_is_no_list_is_refused(given):
    with pytest.raises(pulse.InvalidTypeError, match="must be a list of ReadoutCrosstalk"):
        _surface().replace(readout_crosstalk=given)


@pytest.mark.parametrize("chi", [Offset(5.0), (-20e3, Offset(5.0)), b"ab", True])
def test_a_shift_that_is_no_plain_number_is_refused(chi):
    with pytest.raises(pulse.QuantumSystemDefinitionError, match="chi_hz"):
        _surface().replace(readout_crosstalk=[ReadoutCrosstalk(14, 4, chi)])


def test_a_registered_factory_may_give_a_system_with_readout_crosstalk():
    from cudaq_pulse.quantum_systems import register_quantum_system

    crossed = _surface().replace(readout_crosstalk=[Z2_ON_D5])
    register_quantum_system("crosstalk-factory-test", lambda arg: crossed._description)
    made = pulse.quantum_system("crosstalk-factory-test")
    assert made.readout_crosstalk == (Z2_ON_D5,)
