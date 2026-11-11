# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``pulse.quantum_system()``, ``QuantumSystem.load`` and ``QuantumSystem.from_dict`` report every
problem of a definition at once."""

import json
import pickle
from pathlib import Path

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.quantum_systems import Calibration, Coupling, PhysicalQudit

TRANSMON_SURFACE17 = "transmon_surface17"


def _pair(**parts):
    qudits = {
        i: PhysicalQudit(
            index=i, frequency_hz=5e9, anharmonicity_hz=-2e8, t1_us=0.0, t2_star_us=0.0
        )
        for i in (0, 1)
    }
    return pulse.quantum_system(name="pair", qudits=qudits, **parts)


def _problems(given):
    """The error of the system file *given* (a path) or of its data."""
    with pytest.raises(pulse.QuantumSystemDefinitionError) as info:
        if isinstance(given, Path):
            pulse.QuantumSystem.load(given)
        else:
            pulse.QuantumSystem.from_dict(given)
    return info.value


def test_every_problem_of_a_config_is_reported_in_its_key_order():
    error = _problems(
        {
            "layout": [0, 99, 0],
            "cooldown": "x",
            "name": TRANSMON_SURFACE17,
            "laout": [],
            "zones": [],
        }
    )
    assert [(p.path, p.value) for p in error.problems] == [
        ("layout[1]", 99),
        ("layout[2]", 0),
        ("cooldown", "x"),
        ("laout", "laout"),
        ("zones", pulse.Problem.UNSET),
    ]
    assert str(error).startswith("5 problems in the quantum system data:\n  layout[1]: carrier 99")
    assert "(it has 0..16)" in str(error)
    assert error.problems[1].message == "carrier 0 is already named by layout[0]"
    assert error.problems[3].fix == "did you mean 'layout'?"
    assert error.location is None


def test_a_device_definition_error_is_a_value_error_and_a_pulse_error():
    error = _problems({"name": TRANSMON_SURFACE17, "cooldown": "x"})
    assert isinstance(error, ValueError) and isinstance(error, pulse.PulseError)
    assert not isinstance(error, TypeError)


@pytest.mark.parametrize(
    ("config", "path", "text"),
    [
        ({}, "name", "is missing; give a registered name"),
        ({"name": 4}, "name", "must be a registered name, got int"),
        ({"name": "transmon_17"}, "name", "did you mean 'transmon_surface17'?"),
        ({"name": "transmon_lattice"}, "name", "needs a qudit count"),
        (
            {"name": TRANSMON_SURFACE17, "couplings": []},
            "couplings",
            'builds a quantum system, with "qudits"',
        ),
        ({"qudits": []}, "name", "is missing; give the quantum system's name"),
        ({"name": "chip", "qudits": 3}, "qudits", "must be a list of qudits, got int"),
        (
            {"name": TRANSMON_SURFACE17, "target": TRANSMON_SURFACE17},
            "target",
            "is not a key of a system file",
        ),
        (
            {"name": TRANSMON_SURFACE17, "layout": "0,1"},
            "layout",
            "must be a list of carrier indices",
        ),
        (
            {"name": TRANSMON_SURFACE17, "layout": [0, True]},
            "layout[1]",
            "must be a carrier index",
        ),
        (
            {"name": TRANSMON_SURFACE17, "layout": []},
            "layout",
            "must name at least one carrier",
        ),
        (
            {"name": TRANSMON_SURFACE17, "layout": ()},
            "layout",
            "must name at least one carrier",
        ),
        (
            {"name": TRANSMON_SURFACE17, "calibration": "nope"},
            "calibration",
            "no calibration 'nope'",
        ),
        (
            {"name": TRANSMON_SURFACE17, "calibration": 3},
            "calibration",
            "must be a shipped calibration",
        ),
        (
            {"name": TRANSMON_SURFACE17, "calibration": "transmon_surface17", "cooldown": "nope"},
            "cooldown",
            "has no cooldown 'nope'",
        ),
        ({"name": TRANSMON_SURFACE17, "transport": {}}, "transport", "is reserved"),
    ],
)
def test_each_check_names_its_path(config, path, text):
    (problem,) = _problems(config).problems
    assert problem.path == path
    assert text in str(problem)


def test_a_calibration_loaded_for_another_cooldown():
    calibration = Calibration.load("transmon_surface17", cooldown="2025-flux-tuning")
    (problem,) = _problems(
        {"name": TRANSMON_SURFACE17, "calibration": calibration, "cooldown": "2022-main"}
    ).problems
    assert problem.path == "cooldown"
    assert problem.fix == "pass Calibration.load(..., cooldown='2022-main')"


def test_a_built_quantum_system_is_checked_like_a_registered_one():
    (problem,) = _problems(_pair().as_dict() | {"layout": [2]}).problems
    assert problem.message == "carrier 2 is not in quantum system 'pair' (it has 0..1)"
    assert _pair(layout=[1]).layout == (1,)
    with pytest.raises(pulse.QuantumSystemDefinitionError) as info:
        _pair(layout=[2])
    assert [p.path for p in info.value.problems] == ["layout[0]"]


def test_a_quantum_system_is_built_from_its_keywords_and_reads_back_from_its_data():
    pair = _pair(couplings=[Coupling((0, 1), 5e6)], attribution="lab")
    assert (
        pair.name == "pair" and pair.attribution == "lab" and pair.couplings[0].strength_hz == 5e6
    )
    data = pair.as_dict()
    assert data["name"] == "pair" and [q["index"] for q in data["qudits"]] == [0, 1]
    assert "quantum_system" not in data
    assert pulse.QuantumSystem.from_dict(data) == pair
    with pytest.raises(pulse.QuantumSystemDefinitionError) as info:
        _pair(couplings=[Coupling((0, 7), 5e6)])
    assert [p.path for p in info.value.problems] == ["couplings[0].qudits[1]"]
    data["couplings"] = [{"qudits": [0, 7], "strength_hz": 5e6}]
    assert [p.path for p in _problems(data).problems] == ["couplings[0].qudits[1]"]


def test_a_registered_name_takes_no_keywords_that_build_a_system():
    for given in ({"couplings": [Coupling((0, 1), 5e6)]}, {"model": None}, {"attribution": "x"}):
        (keyword,) = given
        with pytest.raises(TypeError, match=f"quantum_system's {keyword} builds a quantum system"):
            pulse.quantum_system(TRANSMON_SURFACE17, **given)
    with pytest.raises(TypeError, match="builds a quantum system and needs its name="):
        pulse.quantum_system(qudits={})


def test_the_device_constructor_checks_its_layout():
    target = _pair()._description
    with pytest.raises(
        pulse.QuantumSystemDefinitionError, match=r"layout\[1\]: carrier 0 is already"
    ):
        pulse.QuantumSystem(target, name="pair", layout=[0, 0])
    with pytest.raises(pulse.QuantumSystemDefinitionError) as info:
        pulse.QuantumSystem(target, name="pair", layout=[])
    (problem,) = info.value.problems
    assert str(problem) == (
        "layout: must name at least one carrier, got an empty list; "
        "leave layout out for every carrier"
    )
    with pytest.raises(pulse.QuantumSystemDefinitionError) as info:
        pulse.QuantumSystem(target, name="pair", layout=["0"])
    (problem,) = info.value.problems
    assert problem.path == "layout[0]"
    assert pulse.QuantumSystem(target, name="pair", layout=[1]).layout == (1,)


def test_a_file_problem_gives_its_file_line_and_column(tmp_path: Path):
    path = tmp_path / "lab.json"
    path.write_text(
        '{\n  "name": "transmon_surface17",\n  "layout": [4, 5, 99],\n  "cooldown": "x"\n}\n'
    )
    error = _problems(path)
    assert [(p.file, p.line, p.column, p.path) for p in error.problems] == [
        (str(path), 3, 20, "layout[2]"),
        (str(path), 4, 15, "cooldown"),
    ]
    assert str(error).splitlines() == [
        f"2 problems in quantum system file {path}:",
        f"  {path}:3:20: layout[2]: carrier 99 is not in quantum system 'transmon_surface17' "
        "(it has 0..16)",
        f"  {path}:4:15: cooldown: 'x' is given without a calibration; add "
        '"calibration", a shipped name or a file path, or remove "cooldown"',
    ]
    assert error.location == error.problems[0]


def test_a_file_that_is_not_json_or_not_an_object(tmp_path: Path):
    path = tmp_path / "lab.json"
    path.write_text('{"name": "transmon_surface17",\n  , }')
    (problem,) = _problems(path).problems
    assert (problem.line, problem.column) == (2, 3)
    assert problem.message.startswith("not valid JSON: ")
    path.write_text("[1, 2]")
    (problem,) = _problems(path).problems
    assert problem.message == "the file must hold a JSON object, got list"
    assert (problem.line, problem.column) == (1, 1)


def test_a_calibration_file_that_is_not_json(tmp_path: Path):
    path = tmp_path / "cal.json"
    path.write_text('{"cooldowns": ')
    (problem,) = _problems({"name": TRANSMON_SURFACE17, "calibration": str(path)}).problems
    assert problem.path == "calibration"
    assert problem.message.startswith("the calibration file is not valid JSON: ")


@pytest.mark.parametrize(
    ("spec", "topology", "coupled"),
    [
        ("transmon_lattice:4", "linear", [(0, 1), (1, 2), (2, 3)]),
        ("transmon_lattice:4:ring", "ring", [(0, 1), (1, 2), (2, 3), (3, 0)]),
        ("transmon_lattice:4:grid", "grid", [(0, 1), (0, 2), (1, 3), (2, 3)]),
    ],
)
def test_a_lattice_is_named_as_it_is_registered(spec, topology, coupled):
    from cudaq_pulse.quantum_systems import transmon_lattice

    built = transmon_lattice(4, topology=topology)
    registered = pulse.quantum_system(spec)
    assert built.name == registered.name == spec
    assert built.coupling_map == registered.coupling_map == coupled


def test_a_lattice_spans_its_frequency_range():
    from cudaq_pulse.quantum_systems import transmon_lattice

    qs = transmon_lattice(5, base_frequency_hz=5.0e9, frequency_range_hz=200e6)
    assert [qs.physical_qudits[i].frequency_hz for i in range(5)] == pytest.approx(
        [5.0e9, 5.05e9, 5.1e9, 5.15e9, 5.2e9]
    )


def test_replace_makes_a_variant_of_a_quantum_system():
    qs = pulse.quantum_system(TRANSMON_SURFACE17, layout=[4, 5, 6])
    q4 = qs.physical_qudits[4]
    detuned = qs.replace(qudits={**qs.physical_qudits, 4: q4.replace(frequency_hz=5.0e9)})
    assert detuned.name == qs.name and detuned.layout == (4, 5, 6)
    assert detuned.physical_qudits[4].frequency_hz == 5.0e9
    assert detuned.couplings == qs.couplings and detuned.calibration is None
    assert detuned.qudit_freq[0] == 5.0e9 and detuned.qudit_freq[1] == qs.qudit_freq[1]
    assert qs.replace(layout=[0, 1]).layout == (0, 1)
    assert qs.replace(name="copy").name == "copy"
    with pytest.raises(
        TypeError, match=r"unexpected keyword argument 'calibration'; it takes name"
    ):
        qs.replace(calibration=None)


@pytest.mark.parametrize("keywords", [{}, {"layout": [0, 1]}, {"couplings": []}, {"qudits": {}}])
def test_a_quantum_system_passed_to_quantum_system_says_to_replace_it(keywords):
    qs = pulse.quantum_system(TRANSMON_SURFACE17)
    with pytest.raises(TypeError) as info:
        pulse.quantum_system(qs, **keywords)
    assert str(info.value) == (
        "pulse.quantum_system takes a registered name or the keywords that build a quantum "
        "system, not a QuantumSystem; qs.replace(...) makes a variant of one, as "
        "qs.replace(layout=[0, 1]) for 'transmon_surface17' on its first two qudits"
    )


def test_a_missing_file_is_an_os_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        pulse.QuantumSystem.load(tmp_path / "absent.json")


def test_a_target_name_is_a_device(tmp_path: Path, monkeypatch):
    dev = pulse.quantum_system(TRANSMON_SURFACE17)
    assert dev == pulse.quantum_system(TRANSMON_SURFACE17) and dev.name == TRANSMON_SURFACE17
    assert len(pulse.quantum_system("transmon_lattice:4").qudits) == 4
    with pytest.raises(pulse.NotFoundError, match=r"did you mean 'transmon_surface17'\?"):
        pulse.quantum_system("transmon_transmon_surface17s")
    with pytest.raises(FileNotFoundError):
        pulse.QuantumSystem.load("absent.json")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "lab.json").write_text(json.dumps({"name": TRANSMON_SURFACE17, "layout": [4]}))
    assert pulse.QuantumSystem.load("lab.json").layout == (4,)
    with pytest.raises(pulse.NotFoundError) as raised:
        pulse.quantum_system("lab.json")
    assert (
        "a string is a registered quantum system name: to read a system file, use "
        "pulse.QuantumSystem.load(" in str(raised.value)
    )


def test_no_argument_is_the_process_target(monkeypatch):
    monkeypatch.setenv("CUDAQ_PULSE_QUANTUM_SYSTEM", "transmon_lattice:3")
    assert len(pulse.quantum_system().qudits) == 3
    monkeypatch.delenv("CUDAQ_PULSE_QUANTUM_SYSTEM")
    with pytest.raises(
        pulse.NoProcessQuantumSystemError,
        match=r'name one, as quantum_system\("transmon_surface17',
    ):
        pulse.quantum_system()


def test_data_that_is_not_a_mapping_is_a_type_error():
    with pytest.raises(TypeError, match="quantum system data must be a mapping, got list"):
        pulse.QuantumSystem.from_dict([1])  # type: ignore[arg-type]
    with pytest.raises(
        TypeError, match="takes a registered name, or name= and qudits= to build one"
    ):
        pulse.quantum_system([1])  # type: ignore[arg-type]
    with pytest.raises(
        TypeError, match=r"QuantumSystem.from_dict\(data\) reads a system file's data"
    ):
        pulse.quantum_system({"name": TRANSMON_SURFACE17})  # type: ignore[arg-type]
    for keyword in ("control_system", "calibration"):
        with pytest.raises(TypeError, match=f"pulse.quantum_system's {keyword} is a .*, not data"):
            pulse.quantum_system(TRANSMON_SURFACE17, **{keyword: {}})


def test_the_arguments_of_device_are_checked_at_their_names():
    with pytest.raises(pulse.QuantumSystemDefinitionError) as info:
        pulse.quantum_system(TRANSMON_SURFACE17, layout=[0, 99], cooldown="x", clock_ghz=-1.0)
    assert str(info.value).startswith("3 problems in pulse.quantum_system():")
    assert [p.path for p in info.value.problems] == ["layout[1]", "cooldown", "clock_ghz"]
    with pytest.raises(TypeError):
        pulse.quantum_system(description=TRANSMON_SURFACE17)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        pulse.quantum_system(TRANSMON_SURFACE17, [4])  # type: ignore[call-arg]


def test_the_keyword_form_is_the_device_of_the_same_values(monkeypatch):
    dev = pulse.quantum_system(TRANSMON_SURFACE17, layout=[4, 5], calibration="transmon_surface17")
    data = {"name": TRANSMON_SURFACE17, "layout": [4, 5], "calibration": "transmon_surface17"}
    assert (
        dev == pulse.QuantumSystem.from_dict(data) == pulse.QuantumSystem.from_dict(dev.as_dict())
    )
    monkeypatch.setenv("CUDAQ_PULSE_QUANTUM_SYSTEM", "transmon_lattice:3")
    assert pulse.quantum_system(layout=[2]).layout == (2,)


def test_problems_survive_pickling(tmp_path: Path):
    path = tmp_path / "lab.json"
    path.write_text(json.dumps({"name": TRANSMON_SURFACE17, "layout": [99]}))
    error = _problems(path)
    copy = pickle.loads(pickle.dumps(error))
    assert type(copy) is pulse.QuantumSystemDefinitionError
    assert copy.problems == error.problems and str(copy) == str(error)
    assert copy.problems[0].value == 99


def test_a_long_list_is_cut_at_ten():
    error = _problems({"name": TRANSMON_SURFACE17, "layout": list(range(100, 112))})
    assert len(error.problems) == 12
    lines = str(error).splitlines()
    assert lines[0] == "12 problems in the quantum system data:"
    assert len(lines) == 12 and lines[-1] == "  ... and 2 more"


def test_a_problem_reads_as_where_what_and_fix():
    problem = pulse.Problem("is wrong", path="a[0]", fix="use 1", file="f.json", line=2, column=5)
    assert str(problem) == "f.json:2:5: a[0]: is wrong; use 1"
    assert str(pulse.Problem("is wrong")) == "is wrong"
    assert str(pulse.Problem("is wrong", file="f.json")) == "f.json: is wrong"


def test_an_error_built_from_problems_alone_has_a_header():
    error = pulse.QuantumSystemDefinitionError(problems=[pulse.Problem("is wrong", path="x")])
    assert str(error) == "1 problem in the quantum system definition:\n  x: is wrong"
