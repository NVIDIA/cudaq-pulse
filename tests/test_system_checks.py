# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A quantum system's couplings, their CZ calibrations and residual ZZ entries.

A field left out in Python is Python's own TypeError at the constructor; a
system file or config reports each missing key, at its path and (for a file)
its line, under the file's key names.
"""

from __future__ import annotations

import dataclasses
import json
import math

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import QuantumSystemDefinitionError
from cudaq_pulse.quantum_systems import (
    Coupling,
    DeviceDescription,
    FluxCZ,
    PhysicalQudit,
    ResidualZZ,
    SimulationModel,
    TransmonModel,
)
from cudaq_pulse.quantum_systems.base import _COUPLING_OPERATORS, _coupling_operator


def _qudits(n=3):
    return {i: PhysicalQudit(i, 5.0e9 + 1.0e8 * i, -2.5e8, 50.0, 40.0) for i in range(n)}


def _system(**fields):
    return DeviceDescription(name="checked", qudits=_qudits(), **fields)


def _cz(duration_ns, buffer_ns=None, interaction_frequency_hz=None):
    """The calibrations of a coupling with a CZ."""
    cz = FluxCZ(
        duration_ns=duration_ns,
        buffer_ns=buffer_ns,
        interaction_frequency_hz=interaction_frequency_hz,
    )
    return {pulse.gates.CZ: cz}


def _file_problems(tmp_path, data):
    path = tmp_path / "qpu.json"
    text = json.dumps(data, indent=2)
    path.write_text(text)
    with pytest.raises(QuantumSystemDefinitionError) as raised:
        pulse.QuantumSystem.load(path)
    assert str(raised.value).startswith(f"{len(raised.value.problems)} problem")
    return raised.value.problems, text.splitlines()


def _system_data(**changes):
    data = pulse.quantum_system(_system()).as_dict()
    data["quantum_system"].update(changes)
    return data


def _line_of(lines, needle, after=0):
    return next(i for i, line in enumerate(lines, start=1) if i > after and needle in line)


# ── The coupling operator comes from the model ─────────────────────────────


def test_a_system_names_no_architecture():
    assert not hasattr(pulse.quantum_systems, "Architecture")
    assert "architecture" not in _system_data()["quantum_system"]


def test_a_system_gives_no_hamiltonian_term_dicts():
    assert not hasattr(DeviceDescription, "hamiltonian_terms")
    assert not hasattr(DeviceDescription, "dissipator_terms")


def test_the_model_decides_the_coupling_operator():
    assert dict(_COUPLING_OPERATORS) == {TransmonModel: "exchange"}
    assert _coupling_operator(TransmonModel()) == "exchange"
    assert _coupling_operator(TransmonModel(levels={0: 4})) == "exchange"

    class Tuned(TransmonModel):
        __slots__ = ()

    assert _coupling_operator(Tuned()) == "exchange"
    assert _coupling_operator(SimulationModel()) is None
    assert _coupling_operator(None) is None
    assert _system().model == TransmonModel()


# ── Couplings, CZ calibrations and residual ZZ ─────────────────────────────


@pulse.kernel
def _pair_program(q0, q1):
    d0 = get_drive_line(q0)
    d1 = get_drive_line(q1)
    drive(d0, square(8, 0.1))
    sync(d0, d1)
    drive(d1, square(8, 0.1))


def _unmeasured_system():
    return _system(
        couplings=[Coupling((0, 1), None), Coupling((1, 2), 3.0e6), Coupling((2, 0), None)]
    )


def test_a_coupling_not_measured_yet_compiles_schedules_and_exports():
    dev = pulse.quantum_system(_unmeasured_system())
    lowered = pulse.compile(_pair_program).lower(*dev.qudits[:2])
    assert [e.start_cycles for e in lowered.schedule] == [0, 8]
    assert lowered.export_controller() and lowered.export_emulation()
    assert "pulse.coupling" not in lowered.mlir()
    data = dev.as_dict()
    assert [c["strength_hz"] for c in data["quantum_system"]["couplings"]] == [None, 3.0e6, None]
    assert pulse.QuantumSystem.from_dict(json.loads(json.dumps(data))) == dev


def test_simulate_refuses_a_coupling_not_measured_yet_by_name():
    dev = pulse.quantum_system(_unmeasured_system())
    lowered = pulse.compile(_pair_program).lower(*dev.qudits[[2, 0]])
    with pytest.raises(pulse.InvalidValueError) as raised:
        pulse.simulate(lowered, levels=2)
    assert str(raised.value) == (
        "simulating needs the strength of every coupling of the program's qudits; "
        "couplings[2] (qudits 2 and 0) of 'checked' has strength_hz None, not measured yet: "
        "give the strength in Hz, or leave those qudits out of the program"
    )
    everything = pulse.compile(_three_qudits).lower(*dev.qudits)
    with pytest.raises(pulse.InvalidValueError, match=r"couplings\[0\] \(qudits 0 and 1\) and "):
        pulse.simulate(everything, levels=2)
    # A program on measured couplings only simulates.
    pulse.simulate(pulse.compile(_pair_program).lower(*dev.qudits[1:]), levels=2)


@pulse.kernel
def _three_qudits(q0, q1, q2):
    d0 = get_drive_line(q0)
    get_drive_line(q1)
    get_drive_line(q2)
    drive(d0, square(8, 0.1))


def test_residual_zz_is_a_static_zz_term():
    from cudaq_pulse.runtime._attributes import _target_attributes

    system = _system(residual_zz=[ResidualZZ((2, 0), -4.5e4), ResidualZZ((1, 2), 1.0e3)])
    attrs = _target_attributes(system, (0, 2), integrator="rk4")
    assert "crosstalk" not in attrs
    assert attrs.endswith(
        "pulse.static_terms = [#pulse.static_term<zz, qudits = [2, 0], "
        "rate_hz = -4.50000000000000000e+04>]"
    )


def test_a_cz_calibration_belongs_to_its_coupling_and_is_read_only():
    calibrations = _cz(98.0, 15.0, 4.4e9)
    coupling = Coupling([3, 1], 5.0e6, calibrations=calibrations)
    assert coupling.qudits == (3, 1)
    assert coupling == Coupling((3, 1), 5.0e6, calibrations=_cz(98.0, 15.0, 4.4e9))
    assert hash(coupling) == hash(Coupling((3, 1), 5.0e6, calibrations=calibrations))
    cz = coupling.get_calibration(pulse.gates.CZ)
    assert cz == FluxCZ(duration_ns=98.0, buffer_ns=15.0, interaction_frequency_hz=4.4e9)
    with pytest.raises(AttributeError):
        cz.duration_ns = 4.5e9  # type: ignore[misc]
    with pytest.raises(TypeError, match="read-only"):
        coupling.calibrations[pulse.gates.CZ] = cz  # type: ignore[index]
    assert not hasattr(cz, "fidelity") and not hasattr(coupling, "gate")
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument: 'duration_ns'"):
        FluxCZ()  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'strength_hz'"):
        Coupling((0, 1))  # type: ignore[call-arg]
    assert {f for f in Coupling.__dataclass_fields__} == {"qudits", "strength_hz", "calibrations"}


def test_a_coupling_has_only_a_couplings_calibrations():
    with pytest.raises(pulse.InvalidValueError, match="pulse.gates.X is a qudit's calibration"):
        Coupling((0, 1), 5.0e6, calibrations={pulse.gates.X: FluxCZ(duration_ns=1.0)})
    with pytest.raises(pulse.InvalidTypeError, match="CZ calibration .* must be a FluxCZ"):
        Coupling((0, 1), 5.0e6, calibrations={pulse.gates.CZ: 98.0})
    with pytest.raises(pulse.InvalidValueError, match="did you mean pulse.gates.CZ"):
        Coupling((0, 1), 5.0e6, calibrations={"czz": FluxCZ(duration_ns=1.0)})
    plain = Coupling((0, 1), 5.0e6)
    assert plain.calibrations == {}
    with pytest.raises(pulse.NotFoundError, match="qudits 0 and 1 publishes no pulse.gates.CZ"):
        plain.get_calibration(pulse.gates.CZ)
    with pytest.raises(pulse.InvalidValueError, match="calibrated on a qudit"):
        plain.get_calibration(pulse.gates.X)


def test_residual_zz_is_its_qudits_and_rate():
    entry = ResidualZZ([0, 1], -2.0e4)
    assert (entry.qudits, entry.rate_hz) == ((0, 1), -2.0e4)
    assert [f.name for f in dataclasses.fields(ResidualZZ)] == ["qudits", "rate_hz"]
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'rate_hz'"):
        ResidualZZ((0, 1))  # type: ignore[call-arg]
    assert _system(residual_zz=[entry]).residual_zz == (entry,)


def test_a_system_file_holds_couplings_gates_and_residual_zz():
    target = _system(
        couplings=[
            Coupling((1, 0), 5.7e6, calibrations=_cz(98.0, 15.0, 4.4e9)),
            Coupling((1, 2), None),
        ],
        residual_zz=[ResidualZZ((0, 2), 1.5e4)],
    )
    dev = pulse.quantum_system(target)
    system = dev.as_dict()["quantum_system"]
    assert system["couplings"] == [
        {
            "qudits": [1, 0],
            "strength_hz": 5.7e6,
            "calibrations": [
                {
                    "gate": "cz",
                    "duration_ns": 98.0,
                    "buffer_ns": 15.0,
                    "interaction_frequency_hz": 4.4e9,
                }
            ],
        },
        {"qudits": [1, 2], "strength_hz": None},
    ]
    assert system["residual_zz"] == [{"qudits": [0, 2], "rate_hz": 1.5e4}]
    assert "crosstalk" not in system
    back = pulse.QuantumSystem.from_dict(json.loads(json.dumps(dev.as_dict())))
    assert back == dev and back._description == target
    assert back.couplings[0].qudits == (1, 0)


def test_a_key_that_is_no_field_is_an_unknown_key(tmp_path):
    cz = {"gate": "cz", "duration_ns": 98.0, "fidelity": 0.99, "params": {"cz_step": 1.0}}
    data = _system_data(
        couplings=[{"qudits": [0, 1], "strength_hz": 1.0e6, "calibrations": [cz], "gate": cz}],
        crosstalk=[{"qudits": [0, 1], "static_zz_hz": 1.0e4}],
        residual_zz=[{"qudits": [0, 2], "static_zz_hz": 1.0e4}],
    )
    problems, _ = _file_problems(tmp_path, data)
    assert sorted((p.path, p.message) for p in problems) == [
        ("quantum_system.couplings[0].calibrations[0].fidelity", "is not a known key"),
        ("quantum_system.couplings[0].calibrations[0].params", "is not a known key"),
        ("quantum_system.couplings[0].gate", "is not a known key"),
        ("quantum_system.crosstalk", "is not a known key"),
        ("quantum_system.residual_zz[0].rate_hz", "is missing"),
        ("quantum_system.residual_zz[0].static_zz_hz", "is not a known key"),
    ]


def test_the_fingerprint_holds_the_whole_cz_calibration():
    def fingerprint(calibrations):
        return pulse.quantum_system(
            _system(couplings=[Coupling((0, 1), 5e6, calibrations)])
        ).fingerprint

    prints = {
        fingerprint({}),
        fingerprint(_cz(98.0)),
        fingerprint(_cz(99.0)),
        fingerprint(_cz(98.0, buffer_ns=15.0)),
        fingerprint(_cz(98.0, interaction_frequency_hz=4.4e9)),
    }
    assert len(prints) == 5


def test_a_calibration_matches_a_pair_named_in_either_order():
    from cudaq_pulse.quantum_systems import transmon_17q
    from cudaq_pulse.quantum_systems.calibration import shipped_calibration

    published = transmon_17q(calibration=None)
    calibration = shipped_calibration("transmon_17q")
    reversed_ = dataclasses.replace(
        published,
        couplings=[
            Coupling(c.qudits[::-1], c.strength_hz, c.calibrations) for c in published.couplings
        ],
    )
    forward = calibration.apply(published).couplings
    backward = calibration.apply(reversed_).couplings
    assert [c.strength_hz for c in backward] == [c.strength_hz for c in forward]
    assert [c.qudits for c in backward] == [c.qudits[::-1] for c in forward]
    assert {c.strength_hz for c in forward}.isdisjoint(c.strength_hz for c in published.couplings)


# ── Value checks ────────────────────────────────────────────────────────────


def _problems(**fields):
    with pytest.raises(QuantumSystemDefinitionError) as raised:
        _system(**fields)
    return [str(p) for p in raised.value.problems]


def test_every_problem_of_every_coupling_is_listed_at_its_path():
    problems = _problems(
        couplings=[
            Coupling((0, 7), 1.0e6),
            Coupling((1, 1), 1.0e6),
            Coupling((0, 1), math.nan),
            Coupling((2, True), math.inf),  # type: ignore[arg-type]
        ]
    )
    assert problems == [
        "couplings[0].qudits[1]: 7 is not a qudit of the quantum system; its qudits are 0..2",
        "couplings[1].qudits: names qudit 1 twice; a coupling is between two distinct qudits",
        "couplings[2].strength_hz: nan is not a finite number of Hz; give the strength in Hz, "
        "or None (null in a file) if it is not measured yet",
        "couplings[3].qudits[1]: must be a qudit index (an int), got bool",
        "couplings[3].strength_hz: inf is not a finite number of Hz; give the strength in Hz, "
        "or None (null in a file) if it is not measured yet",
    ]


def test_zero_negative_and_unmeasured_strengths_are_allowed():
    target = _system(
        couplings=[Coupling((0, 1), 0.0), Coupling((1, 2), -2.5e6), Coupling((0, 2), None)]
    )
    assert [c.strength_hz for c in target.couplings] == [0.0, -2.5e6, None]
    assert _system(couplings=[Coupling((0, 1), 3)]).couplings[0].strength_hz == 3


def test_a_pair_has_one_coupling_in_either_order():
    problems = _problems(
        couplings=[Coupling((0, 1), 1.0e6), Coupling((0, 1), 2.0e6), Coupling((1, 0), 3.0e6)]
    )
    assert problems == [
        "couplings[1].qudits: names the same pair of qudits as couplings[0]; give each pair "
        "one coupling; (0, 1) and (1, 0) are the same pair",
        "couplings[2].qudits: names the same pair of qudits as couplings[0]; give each pair "
        "one coupling; (1, 0) and (0, 1) are the same pair",
    ]
    # A coupling and a residual ZZ entry of one pair are two different terms.
    _system(couplings=[Coupling((0, 1), 1.0e6)], residual_zz=[ResidualZZ((1, 0), 1.0e4)])


def test_a_coupling_is_between_two_qudits_in_this_release():
    problems = _problems(
        couplings=[Coupling((0, 1, 2), 1.0e6), Coupling((0,), 1.0e6), Coupling(0, 1.0e6)]  # type: ignore[arg-type]
    )
    assert problems == [
        "couplings[0].qudits: names 3 qudits; a coupling is between two qudits in this "
        "release; give the indices of its two qudits, as (0, 1)",
        "couplings[1].qudits: names 1 qudit; a coupling is between two qudits in this "
        "release; give the indices of its two qudits, as (0, 1)",
        "couplings[2].qudits: must be a tuple of qudit indices, got int; give the indices of "
        "its two qudits, as (0, 1)",
    ]


def test_a_cz_calibration_is_checked_with_its_coupling():
    problems = _problems(
        couplings=[
            Coupling((0, 1), 1.0e6, calibrations=_cz(-1.0, buffer_ns=math.nan)),
            Coupling((1, 2), 1.0e6, calibrations=_cz(98.0, interaction_frequency_hz="4.4e9")),
        ]
    )
    assert problems == [
        "couplings[0].calibrations[CZ].duration_ns: -1.0 is not a finite time of 0 ns or more; "
        "give it in ns",
        "couplings[0].calibrations[CZ].buffer_ns: nan is not a finite time of 0 ns or more; "
        "give it in ns, or None",
        "couplings[1].calibrations[CZ].interaction_frequency_hz: '4.4e9' is not a finite "
        "frequency above 0 Hz; give it in Hz, or None",
    ]
    assert _problems(couplings=[Coupling((0, 1), 1.0e6, calibrations=_cz(98.0, None, 0.0))]) == [
        "couplings[0].calibrations[CZ].interaction_frequency_hz: 0.0 is not a finite frequency "
        "above 0 Hz; give it in Hz, or None",
    ]
    checked = _system(couplings=[Coupling((0, 1), 1.0e6, calibrations=_cz(0, 0, 4))])
    cz = checked.couplings[0].get_calibration(pulse.gates.CZ)
    assert (cz.duration_ns, cz.interaction_frequency_hz) == (0, 4)


def test_residual_zz_has_the_qudit_checks_of_couplings():
    problems = _problems(
        residual_zz=[
            ResidualZZ((0, 1), 1.0e4),
            ResidualZZ((1, 0), math.inf),
            ResidualZZ((2, 2), 1.0e4),
            Coupling((1, 2), 1.0e6),
        ]
    )
    assert problems == [
        "residual_zz[1].qudits: names the same pair of qudits as residual_zz[0]; give each pair "
        "one residual ZZ entry; (1, 0) and (0, 1) are the same pair",
        "residual_zz[1].rate_hz: inf is not a finite number of Hz; give the ZZ rate in Hz",
        "residual_zz[2].qudits: names qudit 2 twice; a residual ZZ entry is between two "
        "distinct qudits",
        "residual_zz[3]: must be a ResidualZZ, got Coupling",
    ]


def test_coupling_and_residual_zz_problems_come_in_one_error():
    with pytest.raises(QuantumSystemDefinitionError) as raised:
        _system(couplings=[Coupling((0, 9), 1.0e6)], residual_zz=[ResidualZZ((1, 1), 1.0e4)])
    assert str(raised.value).splitlines() == [
        "2 problems in the quantum system 'checked':",
        "  couplings[0].qudits[1]: 9 is not a qudit of the quantum system; its qudits are 0..2",
        "  residual_zz[0].qudits: names qudit 1 twice; a residual ZZ entry is between two "
        "distinct qudits",
    ]


def test_a_calibration_keeps_the_checks():
    from cudaq_pulse.quantum_systems import transmon_17q

    target = transmon_17q()
    assert all(pulse.gates.CZ in c.calibrations for c in target.couplings)
    with pytest.raises(
        QuantumSystemDefinitionError, match=r"couplings\[1\].qudits: names the same pair"
    ):
        dataclasses.replace(target, couplings=[target.couplings[0]] * 2)


# ── System files: missing keys and value problems at their lines ───────────


def _coupled_data():
    target = _system(
        couplings=[Coupling((0, 1), 5.0e6, _cz(98.0)), Coupling((1, 2), 4.0e6)],
        residual_zz=[ResidualZZ((0, 2), 1.0e4)],
    )
    return pulse.quantum_system(target).as_dict()


def test_a_system_file_names_each_missing_key_by_its_file_name_and_line(tmp_path):
    data = _coupled_data()
    system = data["quantum_system"]
    del system["couplings"][0]["strength_hz"]
    del system["couplings"][0]["calibrations"][0]["duration_ns"]
    del system["couplings"][1]["qudits"]
    del system["residual_zz"][0]["rate_hz"]
    problems, lines = _file_problems(tmp_path, data)
    assert [(p.path, p.message, p.fix) for p in problems] == [
        (
            "quantum_system.couplings[0].strength_hz",
            "is missing",
            "give the strength in Hz, or null if it is not measured yet",
        ),
        (
            "quantum_system.couplings[1].qudits",
            "is missing",
            "give the indices of the two qudits, as [0, 1]",
        ),
        ("quantum_system.residual_zz[0].rate_hz", "is missing", "give the ZZ rate in Hz"),
    ]
    first = _line_of(lines, '"couplings"')
    assert problems[0].line == _line_of(lines, "{", after=first)
    assert problems[2].line == _line_of(lines, "{", after=_line_of(lines, '"residual_zz"'))
    assert all(p.file is not None and p.file.endswith("qpu.json") for p in problems)


def test_a_config_mapping_lists_every_missing_key_and_value_problem():
    data = _coupled_data()
    system = data["quantum_system"]
    del system["couplings"][0]["calibrations"][0]["gate"]
    system["couplings"][1]["strength_hz"] = math.nan
    system["residual_zz"][0]["qudits"] = [2, 2]
    with pytest.raises(QuantumSystemDefinitionError) as raised:
        pulse.QuantumSystem.from_dict(data)
    assert [(p.path, p.message, p.file) for p in raised.value.problems] == [
        ("quantum_system.couplings[0].calibrations[0].gate", "is missing", None),
        ("quantum_system.couplings[1].strength_hz", "nan is not a finite number of Hz", None),
        ("quantum_system.residual_zz[0].qudits", "names qudit 2 twice", None),
    ]


def test_a_missing_calibration_key_is_named_inside_the_calibration(tmp_path):
    data = _coupled_data()
    del data["quantum_system"]["couplings"][0]["calibrations"][0]["duration_ns"]
    data["quantum_system"]["couplings"][0]["calibrations"][0]["interaction_frequency"] = 4.4e9
    problems, lines = _file_problems(tmp_path, data)
    assert [str(p).split(": ", 1)[1] for p in problems] == [
        "quantum_system.couplings[0].calibrations[0].interaction_frequency: is not a known key; "
        "did you mean 'interaction_frequency_hz'?",
        "quantum_system.couplings[0].calibrations[0]: FluxCZ.__init__() missing 1 required "
        "keyword-only argument: 'duration_ns'",
    ]
    assert problems[1].line == _line_of(lines, "{", after=_line_of(lines, '"calibrations"'))


def test_a_coupling_of_unknown_keys_names_each_and_its_missing_qudits(tmp_path):
    data = _coupled_data()
    data["quantum_system"]["couplings"][1] = {"a": 1, "b": 2, "strength_hz": 4.0e6}
    problems, _ = _file_problems(tmp_path, data)
    assert [(p.path, p.message) for p in problems] == [
        ("quantum_system.couplings[1].a", "is not a known key"),
        ("quantum_system.couplings[1].b", "is not a known key"),
        ("quantum_system.couplings[1].qudits", "is missing"),
    ]


def test_a_value_problem_in_a_file_has_its_file_path_and_line(tmp_path):
    data = _coupled_data()
    system = data["quantum_system"]
    del system["couplings"][0]["strength_hz"]
    system["couplings"][1]["qudits"] = [1, 9]
    system["residual_zz"][0]["rate_hz"] = "1e4"
    problems, lines = _file_problems(tmp_path, data)
    assert [(p.path, p.message) for p in problems] == [
        ("quantum_system.couplings[0].strength_hz", "is missing"),
        ("quantum_system.couplings[1].qudits[1]", "9 is not a qudit of the quantum system"),
        ("quantum_system.residual_zz[0].rate_hz", "'1e4' is not a finite number of Hz"),
    ]
    first = _line_of(lines, '"qudits"', after=_line_of(lines, '"couplings"'))
    assert problems[1].line == _line_of(lines, "9", after=_line_of(lines, '"qudits"', after=first))
    assert problems[2].line == _line_of(lines, '"rate_hz"')


def test_a_cz_calibration_is_read_only_and_pickles_by_content():
    import pickle

    cz = FluxCZ(duration_ns=98.0, buffer_ns=15.0, interaction_frequency_hz=4.4e9)
    with pytest.raises(AttributeError):
        cz.buffer_ns = 1.0  # type: ignore[misc]
    assert repr(cz) == (
        "FluxCZ(duration_ns=98.0, buffer_ns=15.0, interaction_frequency_hz=4400000000.0)"
    )
    back = pickle.loads(pickle.dumps(cz))
    assert back == cz and hash(back) == hash(cz)
    assert cz != FluxCZ(duration_ns=98.0, buffer_ns=15.0) and cz != (98.0, 15.0, 4.4e9)
    coupling = Coupling((0, 1), 1.0e6, _cz(98.0))
    back = pickle.loads(pickle.dumps(coupling))
    assert back == coupling and back.calibrations.__class__ is coupling.calibrations.__class__


def test_a_system_made_from_a_checked_one_is_checked_for_what_changed():
    checked = _system(
        couplings=[Coupling((0, 1), 1.0e6, _cz(98.0)), Coupling((1, 2), 2.0e6)],
        residual_zz=[ResidualZZ((0, 2), 1.0e4)],
    )
    assert dataclasses.replace(checked, clock_ghz=1.0).couplings is checked.couplings
    fewer = {i: q for i, q in checked.qudits.items() if i < 2}
    with pytest.raises(QuantumSystemDefinitionError, match=r"couplings\[1\].qudits\[1\]: 2 is not"):
        dataclasses.replace(checked, qudits=fewer)
    restrengthened = [Coupling(c.qudits, math.nan, c.calibrations) for c in checked.couplings]
    problems = [str(p) for p in _problems_of(checked, restrengthened)]
    assert [p.split(":")[0] for p in problems] == [
        "couplings[0].strength_hz",
        "couplings[1].strength_hz",
    ]
    recalibrated = [Coupling(c.qudits, c.strength_hz, _cz(-1.0)) for c in checked.couplings]
    problems = [str(p) for p in _problems_of(checked, recalibrated)]
    assert [p.split(":")[0] for p in problems] == [
        "couplings[0].calibrations[CZ].duration_ns",
        "couplings[1].calibrations[CZ].duration_ns",
    ]


def _problems_of(target, couplings):
    with pytest.raises(QuantumSystemDefinitionError) as raised:
        dataclasses.replace(target, couplings=couplings)
    return raised.value.problems


def test_values_that_are_not_floats_take_the_complete_checks():
    target = _system(
        couplings=[
            Coupling((0, 1), 5, _cz(98, buffer_ns=0, interaction_frequency_hz=4)),
            Coupling((1, 2), 0.0, _cz(0.0, buffer_ns=0.0)),
        ],
        residual_zz=[ResidualZZ((2, 0), 0)],
    )
    assert [c.strength_hz for c in target.couplings] == [5, 0.0]
    with pytest.raises(QuantumSystemDefinitionError, match="names the same pair of qudits"):
        dataclasses.replace(target, couplings=[*target.couplings, Coupling((2, 1), 7)])
