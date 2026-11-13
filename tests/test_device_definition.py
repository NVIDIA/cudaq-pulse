# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``pulse.device()``, ``Device.load`` and ``Device.from_dict`` report every
problem of a definition at once."""

import json
import pickle
from pathlib import Path

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.targets import Calibration, DeviceDescription, PhysicalQudit

TRANSMON_17Q = "transmon_17q"


def _pair():
    qudits = {
        i: PhysicalQudit(
            index=i, frequency_hz=5e9, anharmonicity_hz=-2e8, t1_us=0.0, t2_star_us=0.0
        )
        for i in (0, 1)
    }
    return DeviceDescription(name="pair", qudits=qudits)


def _problems(given):
    """The error of the system file *given* (a path) or of its data."""
    with pytest.raises(pulse.DeviceDefinitionError) as info:
        if isinstance(given, Path):
            pulse.Device.load(given)
        else:
            pulse.Device.from_dict(given)
    return info.value


def test_every_problem_of_a_config_is_reported_in_its_key_order():
    error = _problems(
        {
            "layout": [0, 99, 0],
            "cooldown": "x",
            "quantum_system": TRANSMON_17Q,
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
    assert str(error).startswith("5 problems in the device data:\n  layout[1]: carrier 99")
    assert "(it has 0..16)" in str(error)
    assert error.problems[1].message == "carrier 0 is already named by layout[0]"
    assert error.problems[3].fix == "did you mean 'layout'?"
    assert error.location is None


def test_a_device_definition_error_is_a_value_error_and_a_pulse_error():
    error = _problems({"quantum_system": TRANSMON_17Q, "cooldown": "x"})
    assert isinstance(error, ValueError) and isinstance(error, pulse.PulseError)
    assert not isinstance(error, TypeError)


@pytest.mark.parametrize(
    ("config", "path", "text"),
    [
        ({}, "quantum_system", "is missing; give a registered target name"),
        (
            {"quantum_system": 4},
            "quantum_system",
            "must be a registered target name or a DeviceDescription",
        ),
        ({"quantum_system": "transmon_17"}, "quantum_system", "did you mean 'transmon_17q'?"),
        ({"quantum_system": "transmon_generic"}, "quantum_system", "needs a qudit count"),
        (
            {"quantum_system": TRANSMON_17Q, "target": TRANSMON_17Q},
            "target",
            "is not a key of a system file",
        ),
        (
            {"quantum_system": TRANSMON_17Q, "layout": "0,1"},
            "layout",
            "must be a list of carrier indices",
        ),
        (
            {"quantum_system": TRANSMON_17Q, "layout": [0, True]},
            "layout[1]",
            "must be a carrier index",
        ),
        (
            {"quantum_system": TRANSMON_17Q, "layout": []},
            "layout",
            "must name at least one carrier",
        ),
        (
            {"quantum_system": TRANSMON_17Q, "layout": ()},
            "layout",
            "must name at least one carrier",
        ),
        (
            {"quantum_system": TRANSMON_17Q, "calibration": "nope"},
            "calibration",
            "no calibration 'nope'",
        ),
        (
            {"quantum_system": TRANSMON_17Q, "calibration": 3},
            "calibration",
            "must be a shipped calibration",
        ),
        (
            {"quantum_system": TRANSMON_17Q, "calibration": "transmon_17q", "cooldown": "nope"},
            "cooldown",
            "has no cooldown 'nope'",
        ),
        ({"quantum_system": TRANSMON_17Q, "transport": {}}, "transport", "is reserved"),
    ],
)
def test_each_check_names_its_path(config, path, text):
    (problem,) = _problems(config).problems
    assert problem.path == path
    assert text in str(problem)


def test_a_calibration_loaded_for_another_cooldown():
    calibration = Calibration.load("transmon_17q", cooldown="2025-flux-tuning")
    (problem,) = _problems(
        {"quantum_system": TRANSMON_17Q, "calibration": calibration, "cooldown": "2022-main"}
    ).problems
    assert problem.path == "cooldown"
    assert problem.fix == "pass Calibration.load(..., cooldown='2022-main')"


def test_a_target_description_is_checked_like_a_name():
    target = _pair()
    (problem,) = _problems({"quantum_system": target, "layout": [2]}).problems
    assert problem.message == "carrier 2 is not in target 'pair' (it has 0..1)"
    assert pulse.device(target, layout=[1]).layout == (1,)


def test_the_device_constructor_checks_its_layout():
    target = _pair()
    with pytest.raises(pulse.DeviceDefinitionError, match=r"layout\[1\]: carrier 0 is already"):
        pulse.Device(target, name="pair", layout=[0, 0])
    with pytest.raises(pulse.DeviceDefinitionError) as info:
        pulse.Device(target, name="pair", layout=[])
    (problem,) = info.value.problems
    assert str(problem) == (
        "layout: must name at least one carrier, got an empty list; "
        "leave layout out for every carrier of the target"
    )
    with pytest.raises(pulse.DeviceDefinitionError) as info:
        pulse.Device(target, name="pair", layout=["0"])
    (problem,) = info.value.problems
    assert problem.path == "layout[0]"
    assert pulse.Device(target, name="pair", layout=[1]).layout == (1,)


def test_a_file_problem_gives_its_file_line_and_column(tmp_path: Path):
    path = tmp_path / "lab.json"
    path.write_text(
        '{\n  "quantum_system": "transmon_17q",\n  "layout": [4, 5, 99],\n  "cooldown": "x"\n}\n'
    )
    error = _problems(path)
    assert [(p.file, p.line, p.column, p.path) for p in error.problems] == [
        (str(path), 3, 20, "layout[2]"),
        (str(path), 4, 15, "cooldown"),
    ]
    assert str(error).splitlines() == [
        f"2 problems in device file {path}:",
        f"  {path}:3:20: layout[2]: carrier 99 is not in target 'transmon_17q' (it has 0..16)",
        f"  {path}:4:15: cooldown: 'x' is given without a calibration; add "
        '"calibration", a shipped name or a file path, or remove "cooldown"',
    ]
    assert error.location == error.problems[0]


def test_a_file_that_is_not_json_or_not_an_object(tmp_path: Path):
    path = tmp_path / "lab.json"
    path.write_text('{"quantum_system": "transmon_17q",\n  , }')
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
    (problem,) = _problems({"quantum_system": TRANSMON_17Q, "calibration": str(path)}).problems
    assert problem.path == "calibration"
    assert problem.message.startswith("the calibration file is not valid JSON: ")


def test_a_missing_file_is_an_os_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        pulse.Device.load(tmp_path / "absent.json")


def test_a_target_name_is_a_device(tmp_path: Path, monkeypatch):
    dev = pulse.device(TRANSMON_17Q)
    assert dev == pulse.device(TRANSMON_17Q) and dev.name == TRANSMON_17Q
    assert len(pulse.device("transmon_generic:4").qudits) == 4
    with pytest.raises(pulse.NotFoundError, match=r"did you mean 'transmon_17q'\?"):
        pulse.device("transmon_transmon_17qq")
    with pytest.raises(FileNotFoundError):
        pulse.Device.load("absent.json")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "lab.json").write_text(json.dumps({"quantum_system": TRANSMON_17Q, "layout": [4]}))
    assert pulse.Device.load("lab.json").layout == (4,)
    with pytest.raises(pulse.NotFoundError) as raised:
        pulse.device("lab.json")
    assert "a string is a target name: to read a system file, use pulse.Device.load(" in str(
        raised.value
    )


def test_no_argument_is_the_process_target(monkeypatch):
    monkeypatch.setenv("CUDAQ_PULSE_QUANTUM_SYSTEM", "transmon_generic:3")
    assert len(pulse.device().qudits) == 3
    monkeypatch.delenv("CUDAQ_PULSE_QUANTUM_SYSTEM")
    with pytest.raises(
        pulse.NoProcessQuantumSystemError, match=r'name a target, as device\("transmon_17q'
    ):
        pulse.device()


def test_data_that_is_not_a_mapping_is_a_type_error():
    with pytest.raises(TypeError, match="device data must be a mapping, got list"):
        pulse.Device.from_dict([1])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="takes a registered target name or a DeviceDescription"):
        pulse.device([1])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"Device.from_dict\(data\) reads a system file's data"):
        pulse.device({"quantum_system": TRANSMON_17Q})  # type: ignore[arg-type]
    for keyword in ("control_system", "calibration"):
        with pytest.raises(TypeError, match=f"pulse.device's {keyword} is a .*, not data"):
            pulse.device(TRANSMON_17Q, **{keyword: {}})


def test_the_arguments_of_device_are_checked_at_their_names():
    with pytest.raises(pulse.DeviceDefinitionError) as info:
        pulse.device(TRANSMON_17Q, layout=[0, 99], cooldown="x", clock_ghz=-1.0)
    assert str(info.value).startswith("3 problems in pulse.device():")
    assert [p.path for p in info.value.problems] == ["layout[1]", "cooldown", "clock_ghz"]
    with pytest.raises(TypeError):
        pulse.device(description=TRANSMON_17Q)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        pulse.device(TRANSMON_17Q, [4])  # type: ignore[call-arg]


def test_the_keyword_form_is_the_device_of_the_same_values(monkeypatch):
    dev = pulse.device(TRANSMON_17Q, layout=[4, 5], calibration="transmon_17q")
    data = {"quantum_system": TRANSMON_17Q, "layout": [4, 5], "calibration": "transmon_17q"}
    assert dev == pulse.Device.from_dict(data) == pulse.Device.from_dict(dev.as_dict())
    monkeypatch.setenv("CUDAQ_PULSE_QUANTUM_SYSTEM", "transmon_generic:3")
    assert pulse.device(layout=[2]).layout == (2,)


def test_problems_survive_pickling(tmp_path: Path):
    path = tmp_path / "lab.json"
    path.write_text(json.dumps({"quantum_system": TRANSMON_17Q, "layout": [99]}))
    error = _problems(path)
    copy = pickle.loads(pickle.dumps(error))
    assert type(copy) is pulse.DeviceDefinitionError
    assert copy.problems == error.problems and str(copy) == str(error)
    assert copy.problems[0].value == 99


def test_a_long_list_is_cut_at_ten():
    error = _problems({"quantum_system": TRANSMON_17Q, "layout": list(range(100, 112))})
    assert len(error.problems) == 12
    lines = str(error).splitlines()
    assert lines[0] == "12 problems in the device data:"
    assert len(lines) == 12 and lines[-1] == "  ... and 2 more"


def test_a_problem_reads_as_where_what_and_fix():
    problem = pulse.Problem("is wrong", path="a[0]", fix="use 1", file="f.json", line=2, column=5)
    assert str(problem) == "f.json:2:5: a[0]: is wrong; use 1"
    assert str(pulse.Problem("is wrong")) == "is wrong"
    assert str(pulse.Problem("is wrong", file="f.json")) == "f.json: is wrong"


def test_an_error_built_from_problems_alone_has_a_header():
    error = pulse.DeviceDefinitionError(problems=[pulse.Problem("is wrong", path="x")])
    assert str(error) == "1 problem in the device definition:\n  x: is wrong"
