# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coupling lines of a controller: the qudits a ``CouplingLine`` acts on,
its conversion, limit and delay, its checks against the quantum system, and
that a controller without them is what it was."""

from __future__ import annotations

import dataclasses
import json
import pickle
import subprocess
import sys

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.controller import CouplingLine
from cudaq_pulse.quantum_systems import (
    ChargeCoupling,
    Coupling,
    ExchangeControl,
    FluxTuning,
    PhysicalQudit,
)

_TUNING = FluxTuning(ec_hz=1.65e8, ej_max_hz=2.6e10, asymmetry=0.47, idle_flux=0.0)
_EXCHANGE = ExchangeControl(
    unit="flux_quantum", amplitudes=(-0.2, 0.0, 0.2), exchange_hz=(9.0e6, 4.0e6, 9.0e6)
)


def _system(controllers, tuned=(0, 3, 4, 5)):
    """Six qudits: *tuned* have a tuning; 1-2 has its own control, 0-3 an
    interaction made through its qudits, 4-5 is always on."""
    return pulse.quantum_system(
        name="coupled",
        qudits={
            i: PhysicalQudit(
                i,
                5.0e9 + 1.0e8 * i,
                50.0,
                30.0,
                anharmonicity_hz=-2.0e8,
                tuning=_TUNING if i in tuned else None,
            )
            for i in range(6)
        },
        couplings=[
            Coupling((1, 2), 4.0e6, control=_EXCHANGE),
            Coupling(
                (0, 3),
                5.0e6,
                interaction=ChargeCoupling(charge_coupling_hz=3e6, tunings=(_TUNING, _TUNING)),
            ),
            Coupling((4, 5), 1.0e6),
        ],
        control_system=pulse.ControlSystem(controllers=controllers),
    )


def test_a_line_is_named_by_the_set_of_its_qudits():
    own = CouplingLine(0, amplitude_per_volt=0.52, max_volts=1.0)
    assert own.qudits == (0,)
    assert own.amplitude_per_volt == 0.52 and own.max_volts == 1.0 and own.delay_ns == 0.0
    assert repr(own) == "CouplingLine(0, amplitude_per_volt=0.52, max_volts=1.0)"
    pair = CouplingLine((2, 1), delay_ns=2 * pulse.ns)
    assert pair.qudits == (1, 2) and pair.amplitude_per_volt is None and pair.max_volts is None
    assert repr(pair) == "CouplingLine((1, 2), delay_ns=2.0)"
    # A tuple, a list, a set and a range mean the same, in any order.
    assert pair == CouplingLine([1, 2], delay_ns=2.0) == CouplingLine({2, 1}, delay_ns=2.0)
    assert hash(pair) == hash(CouplingLine([2, 1], delay_ns=2.0))
    group = CouplingLine([5, 3, 4])
    assert group.qudits == (3, 4, 5) and group == CouplingLine(range(3, 6))
    assert repr(group) == "CouplingLine((3, 4, 5))"
    assert CouplingLine([3]) == CouplingLine(3) == CouplingLine((3,))
    assert CouplingLine(0) != 0 and CouplingLine(0) != CouplingLine(0, max_volts=1.0)
    for line in (own, pair, group):
        assert pickle.loads(pickle.dumps(line)) == line
        assert eval(repr(line), {"CouplingLine": CouplingLine}) == line


def test_a_line_is_read_only():
    line = CouplingLine(0)
    with pytest.raises(AttributeError, match="read-only"):
        line.delay_ns = 1.0
    with pytest.raises(AttributeError, match="read-only"):
        del line.qudits


@pytest.mark.parametrize(
    ("make", "error", "match"),
    [
        (lambda: CouplingLine("01"), pulse.InvalidTypeError, "acts on a qudit or on several"),
        (lambda: CouplingLine(1.5), pulse.InvalidTypeError, "acts on a qudit or on several"),
        (lambda: CouplingLine(True), pulse.InvalidTypeError, "acts on a qudit or on several"),
        (lambda: CouplingLine([(1, 2)]), pulse.InvalidTypeError, r"got \(1, 2\) in"),
        (lambda: CouplingLine(-1), pulse.InvalidValueError, "negative carrier index -1"),
        (lambda: CouplingLine(()), pulse.InvalidValueError, "acts on no qudit"),
        (lambda: CouplingLine((1, 1)), pulse.InvalidValueError, "lists a qudit twice"),
        (
            lambda: CouplingLine(0, amplitude_per_volt=0),
            pulse.InvalidValueError,
            "a finite number other than 0, or None; got 0",
        ),
        (
            lambda: CouplingLine(0, amplitude_per_volt=10**400),
            pulse.InvalidValueError,
            "amplitude_per_volt",
        ),
        (
            lambda: CouplingLine(0, max_volts=-1.0),
            pulse.InvalidValueError,
            r"CouplingLine\(0\).max_volts is the most volts the output plays",
        ),
        (
            lambda: CouplingLine(0, delay_ns=-1.0),
            pulse.InvalidValueError,
            r"CouplingLine\(0\).delay_ns is -1.0 ns; delays cannot be negative",
        ),
    ],
)
def test_a_line_names_what_is_wrong(make, error, match):
    with pytest.raises(error, match=match):
        make()


def test_a_controller_keeps_its_coupling_lines():
    lines = [CouplingLine(0, amplitude_per_volt=0.52), CouplingLine((1, 2))]
    controller = pulse.Controller(coupling_lines=lines)
    assert controller.coupling_lines == tuple(lines)
    assert hash(controller) == hash(pulse.Controller(coupling_lines=tuple(lines)))
    assert controller == pickle.loads(pickle.dumps(controller))
    assert controller != pulse.Controller()
    assert dataclasses.replace(controller, clock_ghz=1.0).coupling_lines == tuple(lines)
    # The exports hold a controller's as_dict as its limits: no coupling
    # lines there. A system file lists them.
    assert controller.as_dict() == pulse.Controller().as_dict()
    system = pulse.ControlSystem(controllers=[controller])
    assert system.as_dict()["controllers"][0]["coupling_lines"] == [
        {"qudits": [0], "amplitude_per_volt": 0.52},
        {"qudits": [1, 2]},
    ]


def test_a_controller_without_coupling_lines_is_what_it_was():
    plain = pulse.Controller(drive_lines=[0, 1], readout_lines=[0, 1])
    assert "coupling_lines" not in plain.as_dict()
    assert plain.coupling_lines == ()
    # It prints as before coupling lines existed, and so does one with them.
    assert "coupling_lines" not in repr(plain)
    system = pulse.ControlSystem(controllers=[plain])
    assert "coupling_lines" not in json.dumps(system.as_dict())
    # It pickles as it did before coupling lines existed.
    assert b"coupling_lines" not in pickle.dumps(plain)
    assert pickle.loads(pickle.dumps(plain)) == plain


def test_the_module_loads_only_when_coupling_lines_are_used():
    code = (
        "import sys, cudaq_pulse as pulse\n"
        "c = pulse.Controller(drive_lines=[0], readout_lines=[0])\n"
        "pulse.ControlSystem(controllers=[c]).as_dict()\n"
        "s = pulse.ControlSystem(controllers=[c])\n"
        "qs = pulse.quantum_system('transmon_lattice:2', control_system=s)\n"
        "print('cudaq_pulse._coupling_lines' in sys.modules)\n"
        "pulse.controller.CouplingLine(0)\n"
        "print('cudaq_pulse._coupling_lines' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "True"]


def test_a_quantum_system_takes_the_lines_it_has_controls_for():
    lines = [
        CouplingLine(0, amplitude_per_volt=0.52, max_volts=1.0),
        CouplingLine((1, 2), amplitude_per_volt=0.31),
        CouplingLine(3, delay_ns=2.0),
    ]
    qs = _system([pulse.Controller(coupling_lines=lines)])
    assert qs.control_system.controllers[0].coupling_lines == tuple(lines)
    assert pickle.loads(pickle.dumps(qs)).control_system == qs.control_system
    data = json.loads(json.dumps(qs.as_dict()))
    assert data["control_system"]["controllers"][0]["coupling_lines"] == [
        {"qudits": [0], "amplitude_per_volt": 0.52, "max_volts": 1.0},
        {"qudits": [1, 2], "amplitude_per_volt": 0.31},
        {"qudits": [3], "delay_ns": 2.0},
    ]
    loaded = pulse.QuantumSystem.from_dict(data)
    assert loaded.control_system.controllers[0].coupling_lines == tuple(lines)
    assert loaded == qs


def test_the_shipped_system_takes_lines_to_the_tunings_its_calibration_gives():
    lines = [CouplingLine(3, amplitude_per_volt=0.5), CouplingLine(10)]
    qs = pulse.quantum_system(
        "transmon_surface17",
        control_system=pulse.ControlSystem(controllers=[pulse.Controller(coupling_lines=lines)]),
    )
    assert qs.control_system.controllers[0].coupling_lines == tuple(lines)
    # X2-D4 (10, 3) interacts through its qudits' tunings: no line of its own.
    with pytest.raises(pulse.QuantumSystemDefinitionError) as raised:
        pulse.quantum_system(
            "transmon_surface17",
            control_system=pulse.ControlSystem(
                controllers=[pulse.Controller(coupling_lines=[*lines, CouplingLine((10, 3))])]
            ),
        )
    (problem,) = raised.value.problems
    assert problem.message == (
        "the coupling of qudits 3 and 10 has no control of its own: its interaction is made "
        "through its qudits' tunings"
    )
    assert problem.fix == (
        "the lines that reach its qudits are CouplingLine(3, amplitude_per_volt=0.5), "
        "CouplingLine(10)"
    )


@pytest.mark.parametrize(
    ("line", "message", "fix"),
    [
        (CouplingLine(7), "qudit 7 is not in the quantum system", "it has qudits 0..5"),
        (
            CouplingLine(1),
            "qudit 1 has no tuning, so no coupling line reaches it",
            "give PhysicalQudit(1, ..., tuning=...)",
        ),
        (
            CouplingLine((1, 3)),
            "qudits 1 and 3 are not coupled, so no coupling line reaches their control",
            "list the pair in the quantum system's couplings",
        ),
        (
            CouplingLine((4, 5)),
            "the coupling of qudits 4 and 5 has no control: it is always on",
            "give Coupling(..., control=...)",
        ),
        (
            CouplingLine((0, 3)),
            "the coupling of qudits 0 and 3 has no control of its own: its interaction is made "
            "through its qudits' tunings",
            "the lines that reach its qudits are CouplingLine(0)",
        ),
        (
            CouplingLine([3, 4, 5]),
            "acts on qudits 3, 4 and 5; a quantum system declares the control of one qudit "
            "(its tuning) or of the coupling of two in this release, so nothing says what a "
            "line of three or more does",
            "CouplingLine((0, 1))",
        ),
    ],
)
def test_a_line_that_reaches_no_control_is_refused_at_its_path(line, message, fix):
    with pytest.raises(pulse.QuantumSystemDefinitionError) as raised:
        _system([pulse.Controller(coupling_lines=[CouplingLine(0), line])])
    (problem,) = raised.value.problems
    assert problem.path == "control_system.controllers[0].coupling_lines[1]"
    assert problem.message == message and fix in problem.fix


@pytest.mark.parametrize(
    ("given", "error", "match"),
    [
        ([0], pulse.InvalidTypeError, "holds CouplingLines, as CouplingLine"),
        (CouplingLine(0), pulse.InvalidTypeError, "is a sequence of CouplingLine"),
        (5, pulse.InvalidTypeError, "is a sequence of CouplingLine"),
        (
            [CouplingLine(3), CouplingLine([3])],
            pulse.InvalidValueError,
            r"qudit 3's tuning is reached by two coupling lines, CouplingLine\(3\) and "
            r"CouplingLine\(3\)",
        ),
    ],
)
def test_a_controllers_lines_are_checked_where_it_is_built(given, error, match):
    with pytest.raises(error, match=match):
        pulse.Controller(coupling_lines=given)
    line = CouplingLine(0)
    with pytest.raises(pulse.InvalidValueError, match="reached by two coupling lines"):
        pulse.Controller(coupling_lines=[line, line])


def test_controllers_compare_by_their_lines_however_they_are_given():
    lines = [CouplingLine(0), CouplingLine((1, 2))]
    as_list, as_tuple = (
        pulse.Controller(coupling_lines=lines),
        pulse.Controller(coupling_lines=tuple(lines)),
    )
    assert as_list == as_tuple and hash(as_list) == hash(as_tuple)
    assert as_list.coupling_lines == tuple(lines)
    assert pulse.Controller(coupling_lines=iter(lines)) == as_tuple
    assert pulse.Controller(coupling_lines=[]) == pulse.Controller()
    assert hash(pulse.Controller(coupling_lines=[])) == hash(pulse.Controller())


def test_a_control_is_reached_from_one_controller():
    first = pulse.Controller(
        drive_lines=[0], coupling_lines=[CouplingLine(0), CouplingLine((1, 2))]
    )
    second = pulse.Controller(drive_lines=[1], coupling_lines=[CouplingLine([2, 1])])
    with pytest.raises(pulse.QuantumSystemDefinitionError) as raised:
        _system([first, second])
    (problem,) = raised.value.problems
    assert problem.path == "control_system.controllers[1].coupling_lines[0]"
    assert problem.message == (
        "the control of qudits 1 and 2 is also reached by controllers[0].coupling_lines[1]"
    )
    assert problem.fix == "a control is reached by one coupling line"


def test_a_system_file_names_a_coupling_line_that_is_not_one():
    qs = _system([pulse.Controller(coupling_lines=[CouplingLine(0)])])
    data = json.loads(json.dumps(qs.as_dict()))
    lines = data["control_system"]["controllers"][0]["coupling_lines"]
    for entry, match in (
        ({"amplitude_per_volt": 0.5}, "is missing"),
        ({"qudits": 0}, "must be the list of the qudits the line acts on"),
        ({"qudits": [0], "target": [0]}, "target"),
        ({"qudits": [0], "amplitude_per_volt": 0}, "a finite number other than 0"),
    ):
        lines[0] = entry
        with pytest.raises(pulse.QuantumSystemDefinitionError, match=match) as raised:
            pulse.QuantumSystem.from_dict(data)
        assert any(
            p.path.startswith("control_system.controllers[0].coupling_lines[0]")
            for p in raised.value.problems
        ), str(raised.value)
