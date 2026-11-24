# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The table of a compiled kernel's lines, ``lowered.lines``: each line's
role and qudits in the order events number them, the controller that plays
it, and what a coupling line reaches."""

from __future__ import annotations

import pickle
import subprocess
import sys

import cudaq_pulse as pulse
import pytest
from cudaq_pulse._lines import table
from cudaq_pulse.compiler import Line
from cudaq_pulse.controller import CouplingLine, DriveLine, ReadoutLine
from cudaq_pulse.ops import (
    drive,
    gaussian,
    get_coupling_line,
    get_drive_line,
    get_readout_line,
    readout,
    square,
    sync,
    wait,
)
from cudaq_pulse.quantum_systems import Coupling, PhysicalQudit
from cudaq_pulse.quantum_systems.tuning import ExchangeControl, FluxTuning

_TUNING = FluxTuning(ec_hz=1.65e8, ej_max_hz=2.6e10, asymmetry=0.47, idle_flux=0.0)
_EXCHANGE = ExchangeControl(
    unit="flux_quantum", amplitudes=(-0.4, 0.0, 0.4), exchange_hz=(9.0e6, 4.0e6, 9.0e6)
)


def _system(control_system=None):
    """Three qudits, 0 and 1 with a tuning; 0-1 has its own control."""
    return pulse.quantum_system(
        name="tuned",
        qudits={
            i: PhysicalQudit(i, 5.0e9 + 1.0e8 * i, 50.0, 30.0, anharmonicity_hz=-3.0e8)
            for i in range(3)
        },
        couplings=[Coupling((0, 1), 4.0e6), Coupling((1, 2), 4.0e6)],
        tunings={0: _TUNING, 1: _TUNING},
        controls={(0, 1): _EXCHANGE},
        control_system=control_system,
    )


QS = _system()
A, D, E = QS.qudits[0], QS.qudits[1], QS.qudits[2]


@pulse.kernel
def detune(a, d, amplitude: float):
    da = get_drive_line(a)
    pair = get_coupling_line(d, a)
    ra = get_readout_line(a)
    ca = get_coupling_line(a)
    drive(ca, square(60 * pulse.ns, amplitude))
    drive(da, gaussian(40, 0.2, 10.0))
    wait(pair, 20 * pulse.ns)
    sync(ca, pair, da)
    drive(pair, square(40, -0.1))
    readout(ra, square(100, 0.2))


@pulse.kernel
def plain(a, d):
    drive(get_drive_line(d), gaussian(40, 0.2, 10.0))
    drive(get_drive_line(a), gaussian(40, 0.2, 10.0))
    readout(get_readout_line(d), square(100, 0.2))


def test_the_table_lists_each_line_in_the_order_events_number_them():
    lowered = detune.lower(A, D, 0.31)
    lines = lowered.lines
    assert isinstance(lines, tuple) and all(type(line) is Line for line in lines)
    assert [(line.index, line.role, line.qudits) for line in lines] == [
        (0, "drive", (0,)),
        (1, "coupling", (0, 1)),
        (2, "readout", (0,)),
        (3, "coupling", (0,)),
    ]
    # Each event names its line by that index: a drive on a coupling line
    # and a drive on a drive line are told apart by the line's role.
    seen = [(e.kind, lines[e.line].role, lines[e.line].qudits) for e in lowered.schedule]
    assert seen == [
        ("drive", "coupling", (0,)),
        ("drive", "drive", (0,)),
        ("wait", "coupling", (0, 1)),
        ("drive", "coupling", (0, 1)),
        ("readout", "readout", (0,)),
    ]
    # The lowest qudit of the line is the event's.
    assert all(e.qudit == lines[e.line].qudits[0] for e in lowered.schedule)
    # Asked again, the table is made again and is equal.
    assert lowered.lines == lines and lowered.lines is not lines


def test_a_coupling_line_shows_what_it_reaches_in_the_quantum_system():
    by_role = {(line.role, line.qudits): line for line in detune.lower(A, D, 0.31).lines}
    own, pair = by_role["coupling", (0,)], by_role["coupling", (0, 1)]
    assert own.component is QS.tuning(0) and own.component == _TUNING
    assert pair.component is QS.control(0, 1) and pair.component == _EXCHANGE
    # Without a control system nothing plays the lines or declares their volts.
    for line in by_role.values():
        assert line.controller is None and line.delay_ns is None
        assert line.amplitude_per_volt is None and line.max_volts is None
    assert by_role["drive", (0,)].component is None
    assert by_role["readout", (0,)].component is None


def test_the_component_is_read_from_the_quantum_system_not_kept_with_the_program():
    # Nothing a tuning gives is in the compiled program: the module lists
    # roles and qudits only, and the table asks the quantum system it is
    # given, when it is made.
    lowered = detune.lower(A, D, 0.31)
    rows = lowered._module.lines()
    assert rows == [("drive", (0,)), ("coupling", (0, 1)), ("readout", (0,)), ("coupling", (0,))]
    other = FluxTuning(ec_hz=1.7e8, ej_max_hz=2.5e10, asymmetry=0.4, idle_flux=0.0)
    retuned = QS.replace(tunings={0: other, 1: _TUNING})
    assert table(rows, retuned, lowered.options.machine)[3].component == other
    assert table(rows, QS, lowered.options.machine)[3].component == _TUNING
    assert lowered.lines[3].component is lowered.quantum_system.tuning(0)


def test_a_line_is_read_only_and_shows_its_fields():
    line = detune.lower(A, D, 0.31).lines[1]
    with pytest.raises(AttributeError):
        line.role = "drive"
    assert not hasattr(line, "__dict__")
    assert repr(line) == (f"Line(index=1, role='coupling', qudits=(0, 1), component={_EXCHANGE!r})")
    assert pulse.compiler.Line is Line


def test_a_call_with_new_values_and_a_pickled_program_give_the_same_table():
    lowered = detune.lower(A, D, 0.31)
    assert detune.lower(A, D, -0.2).lines == lowered.lines
    assert lowered.generic.lines == lowered.lines
    assert pickle.loads(pickle.dumps(lowered)).lines == lowered.lines


def test_a_call_whose_values_set_a_wait_reads_the_module_its_schedule_reads():
    @pulse.kernel
    def spaced(a, d, gap: float):
        ca = get_coupling_line(a)
        da = get_drive_line(d)
        drive(ca, square(40, 0.1))
        wait(ca, gap * pulse.ns)
        drive(da, gaussian(40, 0.2, 10.0))
        sync(ca, da)
        drive(ca, square(40, -0.1))

    lowered = spaced.lower(A, D, 24.0)
    assert lowered.rescheduled and lowered._concrete_module is None
    lines = lowered.lines
    # The table was read from the module with the call's values bound.
    assert lowered._concrete_module is not None
    assert [(line.role, line.qudits) for line in lines] == [("coupling", (0,)), ("drive", (1,))]
    assert [lines[e.line].role for e in lowered.schedule] == [
        "coupling",
        "coupling",
        "drive",
        "coupling",
    ]


def test_lines_fetched_in_a_loop_and_in_a_helper_are_listed_once():
    @pulse.kernel
    def helper(line):
        drive(line, square(20, 0.1))

    @pulse.kernel
    def looped(a, d):
        for _ in range(3):
            helper(get_coupling_line(a))
            drive(get_drive_line(d), gaussian(40, 0.2, 10.0))
        helper(get_coupling_line(a, d))

    lowered = looped.lower(A, D)
    lines = lowered.lines
    assert [(line.role, line.qudits) for line in lines] == [
        ("coupling", (0,)),
        ("drive", (1,)),
        ("coupling", (0, 1)),
    ]
    assert {e.line for e in lowered.schedule} == {0, 1, 2}
    assert len(lowered.schedule) == 7


def test_a_program_without_coupling_lines_has_its_table_too():
    lines = pulse.compile(plain, quantum_system=QS).lower(A, D).lines
    assert lines == (Line(0, "drive", (1,)), Line(1, "drive", (0,)), Line(2, "readout", (1,)))


def test_the_controller_that_plays_each_line_and_its_delay():
    first = pulse.Controller(
        drive_lines=[DriveLine(0, delay_ns=4.0)], readout_lines=[ReadoutLine(0)]
    )
    second = pulse.Controller(
        drive_lines=[DriveLine(1)], readout_lines=[ReadoutLine(1, delay_ns=8.0)]
    )
    qs = _system(pulse.ControlSystem(controllers=[first, second], reference_clock_ghz=1.0))
    lines = pulse.compile(plain, quantum_system=qs).lower(qs.qudits[0], qs.qudits[1]).lines
    assert lines == (
        Line(0, "drive", (1,), controller=1, output_qudits=(1,), delay_ns=0.0),
        Line(1, "drive", (0,), controller=0, output_qudits=(0,), delay_ns=4.0),
        Line(2, "readout", (1,), controller=1, output_qudits=(1,), delay_ns=8.0),
    )
    # One controller, which plays every line.
    alone = _system(pulse.ControlSystem(controllers=[pulse.Controller()]))
    lines = pulse.compile(plain, quantum_system=alone).lower(*alone.qudits[:2]).lines
    assert [(line.controller, line.delay_ns) for line in lines] == [(0, 0.0)] * 3
    assert [line.output_qudits for line in lines] == [(1,), (0,), (1,)]
    # One controller that lists its lines plays those.
    listed = pulse.ControlSystem(
        controllers=[
            pulse.Controller(drive_lines=[0, DriveLine(1, delay_ns=2.0)], readout_lines=[1])
        ]
    )
    assert table([("drive", (1,)), ("readout", (0,))], QS, listed._machine) == (
        Line(0, "drive", (1,), 0, (1,), 2.0),
        Line(1, "readout", (0,)),
    )
    # A kernel compiled for one Controller.
    bare = pulse.Controller(min_pulse_cycles=4)
    lines = pulse.compile(plain, quantum_system=QS, control_system=bare).lower(A, D).lines
    assert [line.controller for line in lines] == [0, 0, 0]


def test_the_lines_of_a_shared_output_each_name_every_qudit_it_reaches():
    shared = pulse.Controller(
        drive_lines=[DriveLine([1, 0], max_concurrent_tones=2, delay_ns=2.0), 2],
        readout_lines=[ReadoutLine([0, 1, 2], max_concurrent_tones=3)],
    )
    qs = _system(pulse.ControlSystem(controllers=[shared]))
    lowered = pulse.compile(plain, quantum_system=qs).lower(qs.qudits[0], qs.qudits[1])
    lines = lowered.lines
    # One row per line as the events number them; the two drive rows are
    # one output, which their controller, role and output_qudits say.
    assert lines == (
        Line(0, "drive", (1,), 0, (0, 1), 2.0),
        Line(1, "drive", (0,), 0, (0, 1), 2.0),
        Line(2, "readout", (1,), 0, (0, 1, 2), 0.0),
    )
    assert repr(lines[0]) == (
        "Line(index=0, role='drive', qudits=(1,), controller=0, output_qudits=(0, 1), delay_ns=2.0)"
    )
    assert [e.line for e in lowered.schedule] == [0, 1, 2]
    # On the second of two controllers, beside a line of its own, with two
    # qudits read out through one line.
    first = pulse.Controller(drive_lines=[2], readout_lines=[2])
    second = pulse.Controller(
        drive_lines=[DriveLine([0, 1], max_concurrent_tones=2)],
        readout_lines=[ReadoutLine([1, 0], max_concurrent_tones=2)],
    )
    system = pulse.ControlSystem(controllers=[first, second], reference_clock_ghz=1.0)
    rows = [("drive", (2,)), ("drive", (1,)), ("readout", (0,)), ("readout", (1,))]
    assert table(rows, _system(system), system._machine) == (
        Line(0, "drive", (2,), 0, (2,), 0.0),
        Line(1, "drive", (1,), 1, (0, 1), 0.0),
        Line(2, "readout", (0,), 1, (0, 1), 0.0),
        Line(3, "readout", (1,), 1, (0, 1), 0.0),
    )
    # A plain line's repr leaves out what is None; the fields are there.
    bare = pulse.compile(plain, quantum_system=QS).lower(A, D).lines[0]
    assert repr(bare) == "Line(index=0, role='drive', qudits=(1,))"
    assert bare.controller is None and bare.output_qudits is None and bare.component is None


def test_what_the_control_system_declares_of_a_coupling_line():
    # A program with a coupling line does not compile for a control system
    # yet, so the rows of one are given here as the module lists them.
    rows = [("drive", (0,)), ("coupling", (1, 0)), ("coupling", (1,)), ("coupling", (0,))]
    wired = pulse.Controller(drive_lines=[0], readout_lines=[0]).with_coupling_lines(
        [CouplingLine((0, 1), amplitude_per_volt=0.5, max_volts=1.2, delay_ns=6.0)]
    )
    tuner = pulse.Controller(drive_lines=[1], readout_lines=[1]).with_coupling_lines(
        [CouplingLine(1, amplitude_per_volt=0.25)]
    )
    qs = _system(pulse.ControlSystem(controllers=[tuner, wired], reference_clock_ghz=1.0))
    lines = table(rows, qs, qs.control_system._machine)
    assert lines == (
        Line(0, "drive", (0,), controller=1, output_qudits=(0,), delay_ns=0.0),
        Line(1, "coupling", (0, 1), 1, (0, 1), 6.0, _EXCHANGE, 0.5, 1.2),
        Line(2, "coupling", (1,), 0, (1,), 0.0, _TUNING, 0.25, None),
        # A line no controller of the control system plays.
        Line(3, "coupling", (0,), component=_TUNING),
    )
    # A qudit that a call has yet to bind: nothing is known of its line,
    # with a control system or without.
    rows = [("coupling", (-1,)), ("drive", (-1,)), ("coupling", (-1, 1))]
    expected = (Line(0, "coupling", (-1,)), Line(1, "drive", (-1,)), Line(2, "coupling", (-1, 1)))
    assert table(rows, QS, QS_MACHINE) == expected
    assert table(rows, qs, qs.control_system._machine) == expected
    # A coupling line of more qudits, or of a pair the system does not
    # couple, reaches no component the system names.
    rows = [("coupling", (0, 1, 2)), ("coupling", (0, 2))]
    assert [line.component for line in table(rows, QS, QS_MACHINE)] == [None, None]
    # A line with its component hashes.
    assert len(set(lines)) == 4


QS_MACHINE = pulse.compile(plain, quantum_system=QS).lower(A, D).options.machine


def test_a_qudit_that_a_call_binds_is_its_carrier_in_the_call():
    @pulse.kernel
    def bound(a: pulse.Qudit):
        drive(get_drive_line(a), gaussian(40, 0.2, 10.0))

    lowered = bound.lower(E)
    assert [(line.role, line.qudits) for line in lowered.lines] == [("drive", (2,))]
    assert [e.qudit for e in lowered.schedule] == [2]
    assert [line.qudits for line in lowered.generic.lines] == [(-1,)]


def test_a_program_whose_lines_are_never_asked_for_loads_nothing_for_them():
    script = (
        "import sys\n"
        "import cudaq_pulse as pulse\n"
        "from cudaq_pulse.ops import drive, gaussian, get_drive_line\n"
        "@pulse.kernel\n"
        "def k(a):\n"
        "    drive(get_drive_line(a), gaussian(40, 0.2, 10.0))\n"
        "qs = pulse.quantum_system('transmon_surface17')\n"
        "lowered = k.lower(qs.qudits[0])\n"
        "events = lowered.schedule\n"
        "assert 'cudaq_pulse._lines' not in sys.modules\n"
        "assert lowered.lines[0].role == 'drive'\n"
        "assert 'cudaq_pulse._lines' in sys.modules\n"
    )
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_the_native_list_refuses_a_module_that_was_not_constructed():
    from cudaq_pulse._native._cudaq_pulse_native import PulseModule

    with pytest.raises(TypeError, match="was not constructed"):
        PulseModule.__new__(PulseModule).lines()
    with pytest.raises(TypeError):
        PulseModule.lines(object())
