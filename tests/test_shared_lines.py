# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared lines: a DriveLine or ReadoutLine reaching several qudits, the
tones it plays at once, and what the checks and the exports make of it."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.controller import DriveLine, ReadoutLine


def _system(readout_lines, drive_lines=(0, 1, 2, 3), **limits):
    controller = pulse.Controller(
        drive_lines=list(drive_lines), readout_lines=readout_lines, **limits
    )
    qudits = sorted(
        {q for line in controller.readout_lines for q in line.qudits}
        | {q for line in controller.drive_lines for q in line.qudits}
    )
    return pulse.quantum_system(
        "transmon_surface17",
        layout=qudits,
        control_system=pulse.ControlSystem(controllers=[controller]),
    )


def _feedline(tones, qudits=(0, 1, 2, 3), **limits):
    return _system([ReadoutLine(list(qudits), max_concurrent_tones=tones)], **limits)


@pulse.kernel
def multiplexed(a, b, c, d):
    ra = get_readout_line(a)
    rb = get_readout_line(b)
    rc = get_readout_line(c)
    rd = get_readout_line(d)
    sync(ra, rb, rc, rd)
    readout(ra, square(200, 0.05))
    readout(rb, square(200, 0.05))
    readout(rc, square(200, 0.05))
    readout(rd, square(200, 0.05))


@pulse.kernel
def in_turn(a, b):
    ra = get_readout_line(a)
    rb = get_readout_line(b)
    readout(ra, square(200, 0.05))
    sync(ra, rb)
    readout(rb, square(200, 0.05))


@pulse.kernel
def together(a, b):
    ra = get_readout_line(a)
    rb = get_readout_line(b)
    sync(ra, rb)
    readout(ra, square(200, 0.05))
    readout(rb, square(200, 0.05))


@pulse.kernel
def rolled_together(a, b):
    ra = get_readout_line(a)
    rb = get_readout_line(b)
    for _ in range(500):
        sync(ra, rb)
        readout(ra, square(100, 0.05))
        readout(rb, square(100, 0.05))


@pulse.kernel
def rolled_in_turn(a, b):
    ra = get_readout_line(a)
    rb = get_readout_line(b)
    for _ in range(500):
        readout(ra, square(100, 0.05))
        sync(ra, rb)
        readout(rb, square(100, 0.05))
        sync(ra, rb)


def _lowered(kernel, qs, *qudits, **options):
    return pulse.compile(kernel, quantum_system=qs, **options).lower(
        *(qs.qudits[q] for q in qudits)
    )


# ── The line ────────────────────────────────────────────────────────────────


def test_a_line_reaches_one_qudit_or_several():
    dedicated = ReadoutLine(5, delay_ns=12.0)
    assert dedicated.qudit == 5
    assert dedicated.qudits == (5,)
    assert dedicated.max_concurrent_tones == 1
    assert repr(dedicated) == "ReadoutLine(5, delay_ns=12.0)"
    shared = ReadoutLine([0, 1, 2, 3], delay_ns=12.0, max_concurrent_tones=4)
    assert shared.qudits == (0, 1, 2, 3)
    assert shared.delay_ns == 12.0
    assert shared.max_concurrent_tones == 4
    assert repr(shared) == "ReadoutLine([0, 1, 2, 3], delay_ns=12.0, max_concurrent_tones=4)"
    assert repr(DriveLine(range(3))) == "DriveLine([0, 1, 2])"
    # A list of one qudit is that qudit's dedicated line.
    assert DriveLine([4], delay_ns=2.0) == DriveLine(4, delay_ns=2.0)
    assert DriveLine([4]).qudit == 4
    assert shared == ReadoutLine((0, 1, 2, 3), 12.0, max_concurrent_tones=4)
    assert shared != ReadoutLine([0, 1, 2, 3], 12.0, max_concurrent_tones=3)
    assert shared != DriveLine([0, 1, 2, 3], 12.0, max_concurrent_tones=4)
    assert hash(shared) == hash(ReadoutLine([0, 1, 2, 3], 12.0, max_concurrent_tones=4))
    for line in (dedicated, shared, DriveLine([0, 1])):
        assert pickle.loads(pickle.dumps(line)) == line
        assert type(pickle.loads(pickle.dumps(line))) is type(line)


def test_a_shared_line_has_no_one_qudit():
    with pytest.raises(
        pulse.InvalidValueError,
        match=r"reaches qudits 0\.\.3, so it has no one qudit; read its \.qudits",
    ):
        _ = ReadoutLine([0, 1, 2, 3]).qudit


@pytest.mark.parametrize(
    ("make", "error", "match"),
    [
        (
            lambda: ReadoutLine([0, 1], max_concurrent_tones=3),
            pulse.InvalidValueError,
            r"ReadoutLine\(\[0, 1\]\) reaches 2 qudits, so it plays 1 to 2 tones at once; "
            "got max_concurrent_tones=3",
        ),
        (
            lambda: ReadoutLine([0, 1], max_concurrent_tones=0),
            pulse.InvalidValueError,
            "plays 1 to 2 tones at once; got max_concurrent_tones=0",
        ),
        (
            lambda: DriveLine(3, max_concurrent_tones=2),
            pulse.InvalidValueError,
            r"DriveLine\(3\) reaches one qudit, so it plays one tone at once; "
            "got max_concurrent_tones=2",
        ),
        (
            lambda: DriveLine([3], max_concurrent_tones=2),
            pulse.InvalidValueError,
            "reaches one qudit, so it plays one tone at once",
        ),
        (
            lambda: DriveLine([3, 4], max_concurrent_tones=True),
            pulse.InvalidTypeError,
            "DriveLine.max_concurrent_tones is an int, got True",
        ),
        (
            lambda: DriveLine([3, 4], max_concurrent_tones=1.0),
            pulse.InvalidTypeError,
            "max_concurrent_tones is an int, got 1.0",
        ),
        (
            lambda: DriveLine(3, max_concurrent_tones=1.0),
            pulse.InvalidTypeError,
            "max_concurrent_tones is an int",
        ),
        (
            lambda: DriveLine([]),
            pulse.InvalidValueError,
            "DriveLine reaches no qudit; give it at least one",
        ),
        (
            lambda: DriveLine([1, 1]),
            pulse.InvalidValueError,
            r"DriveLine\(\[1, 1\]\) lists a qudit twice",
        ),
        (
            lambda: DriveLine([1, -1]),
            pulse.InvalidValueError,
            "DriveLine has a negative carrier index -1",
        ),
        (
            lambda: DriveLine([1, "2"]),
            pulse.InvalidTypeError,
            r"reaches qudits by carrier index, got '2'",
        ),
        (
            lambda: DriveLine([1, True]),
            pulse.InvalidTypeError,
            "reaches qudits by carrier index, got True",
        ),
        (
            lambda: DriveLine("01"),
            pulse.InvalidTypeError,
            "DriveLine reaches a qudit or a list of qudits, by carrier index, got '01'",
        ),
        (
            lambda: ReadoutLine([0, 1], delay_ns=-1.0),
            pulse.InvalidValueError,
            "delays cannot be negative",
        ),
    ],
)
def test_a_line_is_checked(make, error, match):
    with pytest.raises(error, match=match):
        make()


def test_a_controller_reaches_a_qudit_by_one_line_of_a_kind():
    with pytest.raises(
        pulse.InvalidValueError,
        match=r"qudit 1 is reached by two readout lines, ReadoutLine\(\[0, 1\]\) and "
        r"ReadoutLine\(1\); "
        "choosing between lines is not supported yet: reach each qudit by one readout line",
    ):
        pulse.Controller(readout_lines=[ReadoutLine([0, 1]), 1])
    with pytest.raises(pulse.InvalidValueError, match="qudit 2 is reached by two drive lines"):
        pulse.Controller(drive_lines=[DriveLine([0, 1, 2]), DriveLine([2, 3])])
    # Two dedicated lines of one qudit are the carrier listed twice, as before.
    with pytest.raises(pulse.InvalidValueError, match=r"lists a carrier twice: \[1, 1\]"):
        pulse.Controller(readout_lines=[ReadoutLine(1), 1])
    # A drive line and a readout line may reach the same qudits.
    both = pulse.Controller(drive_lines=[DriveLine([0, 1])], readout_lines=[ReadoutLine([0, 1])])
    assert both.drive_lines == (DriveLine([0, 1]),)


def test_a_bare_list_is_not_a_line():
    with pytest.raises(
        pulse.InvalidTypeError,
        match=r"or a line reaching several qudits as DriveLine\(\[0, 1, 2, 3\], "
        r"max_concurrent_tones=\.\.\.\), got \[0, 1\]",
    ):
        pulse.Controller(drive_lines=[[0, 1]])


def test_a_control_system_reaches_a_qudit_by_one_line_of_a_kind():
    first = pulse.Controller(readout_lines=[ReadoutLine([0, 1, 2], max_concurrent_tones=2)])
    second = pulse.Controller(readout_lines=[2])
    with pytest.raises(pulse.QuantumSystemDefinitionError) as caught:
        pulse.ControlSystem(controllers=[first, second])
    (problem,) = caught.value.problems
    assert problem.path == "controllers[1].readout_lines[0]"
    assert problem.message == (
        "qudit 2 is reached by two readout lines, ReadoutLine([0, 1, 2], "
        "max_concurrent_tones=2) on controllers[0] and ReadoutLine(2); choosing between "
        "lines is not supported yet"
    )
    assert problem.fix == "reach each qudit by one readout line"


def test_every_qudit_of_a_line_is_in_the_quantum_system():
    controller = pulse.Controller(readout_lines=[ReadoutLine([0, 1, 30])])
    with pytest.raises(
        pulse.QuantumSystemDefinitionError, match="qudit 30 is not in the quantum system"
    ):
        pulse.quantum_system(
            "transmon_surface17", control_system=pulse.ControlSystem(controllers=[controller])
        )


def test_a_system_file_holds_a_shared_line():
    qs = _system(
        [ReadoutLine([0, 1, 2], delay_ns=12.0, max_concurrent_tones=2), 3],
        drive_lines=[DriveLine([0, 1]), 2, DriveLine(3, delay_ns=4.0)],
    )
    data = qs.as_dict()
    controller = data["control_system"]["controllers"][0]
    assert controller["readout_lines"] == [
        {"qudits": [0, 1, 2], "max_concurrent_tones": 2, "delay_ns": 12.0},
        3,
    ]
    assert controller["drive_lines"] == [{"qudits": [0, 1]}, 2, {"qudit": 3, "delay_ns": 4.0}]
    back = pulse.QuantumSystem.from_dict(json.loads(json.dumps(data)))
    assert back == qs
    assert back.control_system.controllers[0].readout_lines[0] == ReadoutLine(
        [0, 1, 2], 12.0, max_concurrent_tones=2
    )


@pytest.mark.parametrize(
    ("entry", "path", "message"),
    [
        (
            {"qudits": [0, 1, 2], "max_concurrent_tones": 5},
            "readout_lines[0]",
            "reaches 3 qudits, so it plays 1 to 3 tones at once; got max_concurrent_tones=5",
        ),
        ({"qudits": [0, 1], "tones": 2}, "readout_lines[0].tones", "is not a known key"),
        ({"qudits": []}, "readout_lines[0]", "reaches no qudit"),
    ],
)
def test_a_system_file_shared_line_is_checked(entry, path, message):
    data = _feedline(2, (0, 1, 2)).as_dict()
    data["control_system"]["controllers"][0]["readout_lines"] = [entry]
    with pytest.raises(pulse.QuantumSystemDefinitionError) as caught:
        pulse.QuantumSystem.from_dict(data)
    (problem,) = caught.value.problems
    assert problem.path == f"control_system.controllers[0].{path}"
    assert message in problem.message


def test_a_line_s_delay_reaches_each_of_its_qudits():
    qs = _system([ReadoutLine([0, 1], delay_ns=10.0, max_concurrent_tones=2)], drive_lines=())
    control = qs.control_system
    assert dict(control._line_delays) == {("readout", 0): 10.0, ("readout", 1): 10.0}
    assert control._undelayed().controllers[0].readout_lines == (
        ReadoutLine([0, 1], max_concurrent_tones=2),
    )


# ── Checks ──────────────────────────────────────────────────────────────────


def test_a_feedline_reads_out_its_qudits_together():
    lowered = _lowered(multiplexed, _feedline(4), 0, 1, 2, 3)
    assert lowered.export_to_controller(version=2)["lines"][0] == {
        "line": 0,
        "role": "readout",
        "qudits": [0, 1, 2, 3],
        "max_concurrent_tones": 4,
    }


def test_a_feedline_plays_at_most_its_tones_at_once():
    with pytest.raises(pulse.controller.ControllerLimitError) as caught:
        _lowered(multiplexed, _feedline(2), 0, 1, 2, 3)
    assert list(caught.value.violations) == [
        "the readout line of qudits 0..3 (controllers[0].readout_lines[0]) plays 3 tones at once "
        "from cycle 0 (the readouts of qudits 0, 1, 2), and does so 1 more time; it plays at most "
        "2 (max_concurrent_tones): play some after the others end (a sync of their lines between "
        "them), or raise its max_concurrent_tones"
    ]


def test_a_one_tone_line_is_checked_too():
    # Two handles of a feedline that plays one tone at once must not overlap.
    with pytest.raises(
        pulse.controller.ControllerLimitError, match="plays 2 tones at once from cycle 0"
    ):
        _lowered(together, _feedline(1, (0, 1)), 0, 1)
    lowered = _lowered(in_turn, _feedline(1, (0, 1)), 0, 1)
    events = lowered.export_to_controller(version=2)["events"]
    assert [
        (e["kind"], e.get("tone_slot"), e.get("qudits")) for e in events if e["kind"] == "readout"
    ] == [
        ("readout", 0, [0]),
        ("readout", 0, [1]),
    ]


def test_one_handle_of_a_shared_line_needs_no_tone_check():
    @pulse.kernel
    def alone(a):
        r = get_readout_line(a)
        readout(r, square(200, 0.05))
        readout(r, square(200, 0.05))

    lowered = _lowered(alone, _feedline(1, (0, 1)), 0)
    assert lowered.export_to_controller(version=2)["events"][0]["tone_slot"] == 0


def test_dedicated_lines_never_load_the_shared_line_code():
    code = (
        "import sys\n"
        "import cudaq_pulse as pulse\n"
        "controller = pulse.Controller(drive_lines=[0, 1], readout_lines=[0, 1])\n"
        "qs = pulse.quantum_system('transmon_surface17', layout=[0, 1],\n"
        "    control_system=pulse.ControlSystem(controllers=[controller]))\n"
        "@pulse.kernel\n"
        "def k(a, b):\n"
        "    ra = get_readout_line(a)\n"
        "    rb = get_readout_line(b)\n"
        "    sync(ra, rb)\n"
        "    readout(ra, square(200, 0.05))\n"
        "    readout(rb, square(200, 0.05))\n"
        "low = pulse.compile(k, quantum_system=qs).lower(*qs.qudits[:2])\n"
        "low.export_to_controller()\n"
        "assert 'cudaq_pulse._shared_lines' not in sys.modules, 'loaded'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_branches_that_never_play_together_share_a_tone():
    @pulse.kernel
    def either(a, b):
        ra = get_readout_line(a)
        rb = get_readout_line(b)
        m = readout(ra, square(200, 0.05))
        sync(ra, rb)
        if m:
            readout(ra, square(200, 0.05))
        else:
            readout(rb, square(200, 0.05))

    _lowered(either, _feedline(1, (0, 1)), 0, 1, schedule="rcp")


def test_tone_frequencies_count_per_physical_line():
    qs = _feedline(4, max_tones_per_line=2)
    with pytest.raises(pulse.controller.ControllerLimitError) as caught:
        _lowered(multiplexed, qs, 0, 1, 2, 3)
    assert list(caught.value.violations) == [
        "the readout line of qudits 0..3 (controllers[0].readout_lines[0]): 4 tone frequencies; "
        "the controller holds 2 per line"
    ]


def test_waveform_memory_counts_per_physical_line():
    @pulse.kernel
    def shapes(a, b):
        ra = get_readout_line(a)
        rb = get_readout_line(b)
        sync(ra, rb)
        readout(ra, square(200, 0.05))
        readout(rb, gaussian(200, 0.05, 40.0))

    # Each handle stores one 200-sample shape; the line stores both.
    qs = _feedline(2, (0, 1), waveform_memory_samples=300)
    with pytest.raises(pulse.controller.ControllerLimitError) as caught:
        _lowered(shapes, qs, 0, 1)
    assert list(caught.value.violations) == [
        "the readout line of qudits 0..1 (controllers[0].readout_lines[0]): its waveforms take "
        "400 samples; the controller's waveform memory holds 300 per line"
    ]
    _lowered(shapes, _feedline(2, (0, 1), waveform_memory_samples=400), 0, 1)


def test_spacing_and_amplitude_are_per_handle():
    # Each handle is a tone slot of its own sequencer: two handles start
    # together however many cycles one instruction takes, and the amplitude
    # limit is each tone's (the sum of concurrent tones is not checked).
    @pulse.kernel
    def loud(a, b):
        ra = get_readout_line(a)
        rb = get_readout_line(b)
        sync(ra, rb)
        readout(ra, square(200, 0.6))
        readout(rb, square(200, 0.6))

    qs = _feedline(2, (0, 1), instruction_cycles=8, max_amplitude=1.0)
    _lowered(loud, qs, 0, 1)
    with pytest.raises(pulse.controller.ControllerLimitError, match="peaks at amplitude 0.6"):
        _lowered(loud, _feedline(2, (0, 1), max_amplitude=0.5), 0, 1)


def test_a_rolled_loop_on_a_feedline_is_checked_every_iteration():
    qs = _feedline(1, (0, 1))
    with pytest.raises(pulse.controller.ControllerLimitError) as caught:
        _lowered(rolled_together, qs, 0, 1, unroll_threshold=10)
    (message,) = caught.value.violations
    assert "plays 2 tones at once from cycle 0" in message
    assert "and does so 499 more times" in message
    _lowered(rolled_together, _feedline(2, (0, 1)), 0, 1, unroll_threshold=10)
    _lowered(rolled_in_turn, qs, 0, 1, unroll_threshold=10)


def test_a_program_is_checked_against_a_controller_with_a_feedline():
    lowered = _lowered(together, _system([0, 1], drive_lines=(0, 1)), 0, 1)
    feedline = pulse.Controller(readout_lines=[ReadoutLine([0, 1], max_concurrent_tones=1)])
    (message,) = pulse.controller.check(lowered, feedline)
    assert message.startswith(
        "the readout line of qudits 0..1 (controllers[0].readout_lines[0]) plays 2 tones at once"
    )
    roomy = pulse.Controller(readout_lines=[ReadoutLine([0, 1], max_concurrent_tones=2)])
    assert list(pulse.controller.check(lowered, roomy)) == []


def test_runtime_qudits_on_a_feedline_are_checked_when_bound():
    @pulse.kernel
    def pair(a: pulse.Qudit, b: pulse.Qudit):
        ra = get_readout_line(a)
        rb = get_readout_line(b)
        sync(ra, rb)
        readout(ra, square(200, 0.05))
        readout(rb, square(200, 0.05))

    qs = _system([ReadoutLine([0, 1], max_concurrent_tones=1), 2, 3])
    compiled = pulse.compile(pair, quantum_system=qs)
    compiled.lower(qs.qudits[2], qs.qudits[3])
    compiled.lower(qs.qudits[0], qs.qudits[2])
    with pytest.raises(pulse.controller.ControllerLimitError, match="plays 2 tones at once"):
        compiled.lower(qs.qudits[0], qs.qudits[1])


# ── Exports ─────────────────────────────────────────────────────────────────


def _played(events):
    for event in events:
        if event["kind"] == "loop":
            yield from _played(event["body"])
        else:
            yield event


def test_format_2_gives_each_tone_its_slot_and_qudit():
    lowered = _lowered(multiplexed, _feedline(4), 0, 1, 2, 3)
    program = lowered.export_to_controller(version=2)
    assert program["lines"] == [
        {"line": 0, "role": "readout", "qudits": [0, 1, 2, 3], "max_concurrent_tones": 4}
    ]
    assert [
        {k: e[k] for k in ("kind", "line", "tone_slot", "qudits", "start_cycles") if k in e}
        for e in program["events"]
    ] == [
        {"kind": "sync", "start_cycles": 0},
        {"kind": "readout", "line": 0, "tone_slot": 0, "qudits": [0], "start_cycles": 0},
        {"kind": "readout", "line": 0, "tone_slot": 1, "qudits": [1], "start_cycles": 0},
        {"kind": "readout", "line": 0, "tone_slot": 2, "qudits": [2], "start_cycles": 0},
        {"kind": "readout", "line": 0, "tone_slot": 3, "qudits": [3], "start_cycles": 0},
    ]
    # The keys follow "line", in the order of every other event's.
    assert list(program["events"][1])[:5] == ["kind", "line", "tone_slot", "qudits", "start_cycles"]
    assert program["events"][0]["lines"] == [0]
    again = _lowered(multiplexed, _feedline(4), 0, 1, 2, 3).export_to_controller(version=2)
    assert json.dumps(again, sort_keys=False) == json.dumps(program, sort_keys=False)


def test_format_2_numbers_physical_lines_by_first_use():
    @pulse.kernel
    def mixed(a, b):
        da = get_drive_line(a)
        ra = get_readout_line(a)
        rb = get_readout_line(b)
        drive(da, gaussian(40, 0.5, 10.0))
        sync(da, ra, rb)
        readout(ra, square(200, 0.05))
        readout(rb, square(200, 0.05))

    lowered = _lowered(mixed, _feedline(2, (0, 1)), 0, 1)
    program = lowered.export_to_controller(version=2)
    assert program["lines"] == [
        {"line": 0, "role": "drive", "qudits": [0]},
        {"line": 1, "role": "readout", "qudits": [0, 1], "max_concurrent_tones": 2},
    ]
    drive_event = next(e for e in program["events"] if e["kind"] == "play")
    assert "tone_slot" not in drive_event and "qudits" not in drive_event
    sync_event = next(e for e in program["events"] if e["kind"] == "sync")
    assert sync_event["lines"] == [0, 1]


def test_a_program_on_dedicated_lines_exports_as_before():
    @pulse.kernel
    def drives(a, b):
        da = get_drive_line(a)
        db = get_drive_line(b)
        drive(da, gaussian(40, 0.5, 10.0))
        sync(da, db)
        drive(db, gaussian(40, 0.5, 10.0))

    # The same program on a system whose readout feedline it does not play.
    plain = _lowered(drives, _system([0, 1], drive_lines=(0, 1)), 0, 1)
    shared = _lowered(
        drives, _system([ReadoutLine([0, 1], max_concurrent_tones=2)], drive_lines=(0, 1)), 0, 1
    )
    for version in (1, 2):
        a = plain.export_to_controller(version=version)
        b = shared.export_to_controller(version=version)
        key = "limits" if version == 1 else "controller"
        a.pop(key), b.pop(key)
        assert json.dumps(a) == json.dumps(b)


def test_format_2_relines_frame_updates():
    @pulse.kernel
    def framed(a, b):
        ra = get_readout_line(a)
        rb = get_readout_line(b)
        sync(ra, rb)
        readout(ra, square(200, 0.05))
        readout(rb, square(200, 0.05))
        shift_phase(rb.tone, 0.5)

    program = _lowered(framed, _feedline(2, (0, 1)), 0, 1).export_to_controller(version=2)
    update = next(e for e in program["events"] if e["kind"] == "frame_update")
    assert (update["line"], update["tone_slot"], update["qudits"]) == (0, 1, [1])


def test_format_2_gives_handles_taking_turns_the_lowest_free_slot():
    @pulse.kernel
    def three(a, b, c):
        ra = get_readout_line(a)
        rb = get_readout_line(b)
        rc = get_readout_line(c)
        sync(ra, rb, rc)
        readout(ra, square(100, 0.05))
        readout(rb, square(100, 0.05))
        sync(ra, rb, rc)
        readout(rc, square(100, 0.05))
        readout(ra, square(100, 0.05))
        sync(ra, rb, rc)
        readout(rb, square(100, 0.05))
        readout(rc, square(100, 0.05))

    # Each pair of the three handles overlaps somewhere, so they take turns
    # at the line's two slots, tone by tone: in start order, then the order
    # of the handles' first use, each the lowest free slot.
    lowered = _lowered(three, _feedline(2, (0, 1, 2)), 0, 1, 2, schedule="asap")
    events = [
        e for e in lowered.export_to_controller(version=2)["events"] if e["kind"] == "readout"
    ]
    assert [(e["qudits"][0], e["start_cycles"], e["tone_slot"]) for e in events] == [
        (0, 0, 0),
        (1, 0, 1),
        (2, 100, 1),
        (0, 100, 0),
        (1, 200, 0),
        (2, 200, 1),
    ]


def test_format_2_refuses_handles_taking_turns_in_a_rolled_loop():
    @pulse.kernel
    def three(a, b, c):
        ra = get_readout_line(a)
        rb = get_readout_line(b)
        rc = get_readout_line(c)
        for _ in range(100):
            sync(ra, rb, rc)
            readout(ra, square(100, 0.05))
            readout(rb, square(100, 0.05))
            sync(ra, rb, rc)
            readout(rc, square(100, 0.05))
            readout(ra, square(100, 0.05))
            sync(ra, rb, rc)
            readout(rb, square(100, 0.05))
            readout(rc, square(100, 0.05))
            sync(ra, rb, rc)

    lowered = _lowered(
        three, _feedline(2, (0, 1, 2)), 0, 1, 2, schedule="asap", unroll_threshold=10
    )
    with pytest.raises(
        pulse.UnsupportedError, match="format /2 gives each event of a rolled loop one tone slot"
    ):
        lowered.export_to_controller(version=2)


def test_format_1_has_one_qudit_per_line():
    lowered = _lowered(multiplexed, _feedline(4), 0, 1, 2, 3)
    with pytest.raises(
        pulse.UnsupportedError,
        match=r"format /1 has one qudit per line, and the readout line of qudits 0\.\.3 "
        r"\(controllers\[0\]\.readout_lines\[0\]\) reaches several: export version 2",
    ):
        lowered.export_to_controller()


def test_a_program_on_a_delayed_shared_line_does_not_export_yet():
    qs = _system([ReadoutLine([0, 1], delay_ns=12.0, max_concurrent_tones=2)], drive_lines=(0, 1))
    lowered = _lowered(together, qs, 0, 1)
    pulse.simulate(lowered, shots_count=1)
    with pytest.raises(
        pulse.UnsupportedError,
        match="reaches several; format /2, which has shared lines, does not take line delays "
        "yet: a program on delayed shared lines does not export yet",
    ):
        lowered.export_to_controller()
    with pytest.raises(pulse.UnsupportedError, match="without line delays"):
        lowered.export_to_controller(version=2)


def test_format_1_of_several_controllers_has_one_qudit_per_line():
    left = pulse.Controller(
        drive_lines=[0, 1], readout_lines=[ReadoutLine([0, 1], max_concurrent_tones=2)]
    )
    right = pulse.Controller(drive_lines=[2, 3], readout_lines=[2, 3])
    qs = pulse.quantum_system(
        "transmon_surface17",
        layout=[0, 1, 2, 3],
        control_system=pulse.ControlSystem(controllers=[left, right]),
    )
    lowered = _lowered(together, qs, 0, 1)
    with pytest.raises(pulse.UnsupportedError, match="format /1 has one qudit per line"):
        lowered.export_to_controller(controller=0)


def test_a_shared_line_simulates_as_its_qudits_own_lines():
    shared = pulse.simulate(_lowered(together, _feedline(2, (0, 1)), 0, 1), shots_count=10)
    plain = pulse.simulate(
        _lowered(together, _system([0, 1], drive_lines=(0, 1)), 0, 1), shots_count=10
    )
    np.testing.assert_allclose(shared.density_matrix, plain.density_matrix)
