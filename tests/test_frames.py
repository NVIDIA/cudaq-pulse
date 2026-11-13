# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frame changes on the timeline: their stamps, the changes no operation
takes, which the export keeps as frame updates, and the frames the
simulator plays each drive with against the controller export."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import types
import warnings

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import Controller, ControlSystem, _validation
from cudaq_pulse.compiler import DEFAULT_PASSES, Pass
from cudaq_pulse.controller import unroll
from cudaq_pulse.quantum_systems import PhysicalQudit

from tests import _child
from tests._work import python_work

_FREQUENCY_HZ = 5.0e9
_CLOCK_GHZ = 2.0
_TARGET = pulse.quantum_system(
    name="frames",
    clock_ghz=_CLOCK_GHZ,
    qudits={
        i: PhysicalQudit(i, _FREQUENCY_HZ + 1.0e7 * i, 50.0, 30.0, anharmonicity_hz=-2.0e8)
        for i in range(2)
    },
)
_DEV = _TARGET
Q = _DEV.qudits
_PERSISTENT = ControlSystem(controllers=[Controller(persistent_frames=True)])


@pulse.kernel
def straight(q):
    d = get_drive_line(q)
    shift_phase(d.tone, 0.3)
    drive(d, square(16, 0.1))
    shift_frequency(d.tone, 2.0e6)
    drive(d, square(16, 0.1))
    wait(d, 40)
    drive(d, square(16, 0.1))
    set_phase(d.tone, 1.0)
    drive(d, square(16, 0.1))
    set_frequency(d.tone, 5.001e9)
    drive(d, square(16, 0.1))


@pulse.kernel
def looped(q):
    d = get_drive_line(q)
    shift_frequency(d.tone, 1.0e6)
    for _ in range(6):
        drive(d, square(16, 0.1))
        shift_phase(d.tone, 0.25)
        wait(d, 8)


@pulse.kernel
def nested(q):
    d = get_drive_line(q)
    set_frequency(d.tone, 5.003e9)
    for _ in range(3):
        shift_phase(d.tone, 0.5)
        for _ in range(4):
            drive(d, square(16, 0.1))
            shift_phase(d.tone, 0.125)
        wait(d, 24)


@pulse.kernel
def branched(q):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    shift_phase(d.tone, 0.1)
    m = readout(ro, square(100, 0.05))
    sync(d, ro)
    if m:
        drive(d, square(16, 0.1))
        shift_phase(d.tone, 0.5)
    else:
        set_phase(d.tone, 0.2)
    drive(d, square(16, 0.1))
    shift_phase(d.tone, 0.7)
    drive(d, square(16, 0.1))


@pulse.kernel
def trailing(q):
    d = get_drive_line(q)
    shift_frequency(d.tone, 3.0e6)
    drive(d, square(16, 0.1))
    wait(d, 40)
    shift_phase(d.tone, 0.4)
    set_frequency(d.tone, 5.002e9)


@pulse.kernel
def untrailed(q):
    d = get_drive_line(q)
    shift_frequency(d.tone, 3.0e6)
    drive(d, square(16, 0.1))
    wait(d, 40)


@pulse.kernel
def one_arm(q):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    drive(d, square(16, 0.1))
    shift_phase(d.tone, 0.5)
    m = readout(ro, square(100, 0.05))
    sync(d, ro)
    if m:
        drive(d, square(16, 0.1))


@pulse.kernel
def other_arm(q):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    shift_frequency(d.tone, 2.0e6)
    m = readout(ro, square(100, 0.05))
    sync(d, ro)
    if m:
        wait(d, 8)
    else:
        drive(d, square(16, 0.1))


@pulse.kernel
def branched_loop(q):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    for _ in range(3):
        shift_phase(d.tone, 0.3)
        m = readout(ro, square(100, 0.05))
        sync(d, ro)
        if m:
            drive(d, square(16, 0.1))


@pulse.kernel
def measured_rounds(q, r):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    e = get_drive_line(r)
    for _ in range(2):
        drive(d, gaussian(16, 0.1, 4.0))
        shift_phase(d.tone, 0.3)
        sync(d, ro, e)
        m = readout(ro, square(40, 0.05))
        sync(d, ro, e)
        if m:
            drive(e, drag(16, 0.2, 4.0, 0.1))
        else:
            drive(d, square(8, 0.1))


@pulse.kernel
def idle(q, r):
    d = get_drive_line(q)
    e = get_drive_line(r)
    drive(d, square(16, 0.1))
    for _ in range(4):
        drive(e, square(16, 0.1))
        shift_phase(d.tone, 0.25)


@pulse.kernel
def idle_unchanged(q, r):
    d = get_drive_line(q)
    e = get_drive_line(r)
    drive(d, square(16, 0.1))
    for _ in range(4):
        drive(e, square(16, 0.1))


def _compile(kernel, **options):
    return pulse.compile(kernel, quantum_system=_DEV, **options).lower(Q[0])


def _wrapped(phase):
    return math.remainder(phase, 2.0 * math.pi)


def _simulated(lowered, outcome=-1):
    """(start, iteration, frequency, phase) of each drive the simulator plays,
    every measurement giving *outcome*."""
    drives, _ = lowered._concrete().drive_frames([], outcome)
    return sorted((start, list(it), f, _wrapped(p)) for _, it, start, f, p in drives)


def _approx(drives):
    """*drives* as from _simulated, the frequency and phase approximately."""
    return [(start, it, pytest.approx(f), pytest.approx(p)) for start, it, f, p in drives]


def _exported(program, outcome=-1):
    """The same from a format /1 *program*, applying the frame changes of a
    tone tracked at run time as a controller does."""

    def taken(conditions):
        return all(c["test"] == "is_set" and c["arm"] == bool(outcome) for c in conditions)

    phases: dict[int, float] = {}
    applied: set[int] = set()
    found = []
    for event in program["events"]:
        if event["kind"] != "drive" or not taken(event["conditions"]):
            continue
        frequency, phase = event["frequency_hz"], event["phase_rad"]
        if "tone" in event:
            phase = phases.get(event["tone"], 0.0)
            for update in event["frame_updates"]:
                if update["id"] in applied or not taken(update["conditions"]):
                    continue
                applied.add(update["id"])
                assert update["op"] in ("shift_phase", "set_phase")
                phase = update["value"] + (phase if update["op"] == "shift_phase" else 0.0)
            phases[event["tone"]] = phase
            phase += event.get("phase_offset_rad", 0.0)
            frequency = _FREQUENCY_HZ
        found.append((event["start_cycles"], event["iteration"], frequency, _wrapped(phase)))
    return sorted(found)


_EQUIVALENT = [
    pytest.param(straight, {}, id="straight"),
    pytest.param(looped, {"unroll_threshold": 0}, id="rolled"),
    pytest.param(looped, {}, id="unrolled"),
    pytest.param(nested, {"unroll_threshold": 0}, id="nested_rolled"),
    pytest.param(nested, {}, id="nested_unrolled"),
]


@pytest.mark.parametrize(("kernel", "options"), _EQUIVALENT)
def test_the_simulator_plays_each_drive_with_the_frame_the_export_gives(kernel, options):
    lowered = _compile(kernel, **options)
    simulated = _simulated(lowered)
    assert simulated
    for program in (
        lowered.export_to_controller(),
        unroll(lowered.export_to_controller(version=2)),
    ):
        assert _exported(program) == _approx(simulated)


@pytest.mark.parametrize("outcome", [0, 1])
def test_a_frame_changed_in_a_branch_plays_as_the_controller_tracks_it(outcome):
    lowered = _compile(branched)
    simulated = _simulated(lowered, outcome)
    assert len(simulated) == 2 + outcome
    for program in (
        lowered.export_to_controller(),
        unroll(lowered.export_to_controller(version=2)),
    ):
        assert _exported(program, outcome) == _approx(simulated)


def test_a_frame_change_carried_to_the_next_iteration_is_a_phase_step():
    structured = _compile(looped, unroll_threshold=0).export_to_controller(version=2)
    [loop] = (e for e in structured["events"] if e["kind"] == "loop")
    [play] = (e for e in loop["body"] if e["kind"] == "play")
    [tone] = play["tones"]
    detuning = 2.0 * math.pi * 1.0e6 * 1.0e-9 * loop["period_cycles"] / _CLOCK_GHZ
    assert tone["phase_step_rad"] == [pytest.approx(0.25 + detuning)]


def test_frame_changes_are_stamped_on_the_timeline_with_no_duration():
    text = _compile(looped, unroll_threshold=0).mlir()
    [before] = re.findall(r"pulse\.shift_frequency [^<]*(<\{.*?\}>)", text)
    assert before == "<{duration_cycles = 0 : i64, start_cycles = 0 : i64}>"
    # The shift after the drive in the loop's body: at its end, every period;
    # the last iteration's is left to nothing, so the loop lists it.
    [carried] = re.findall(r"pulse\.shift_phase [^<]*(<\{.*?\}> \{.*?\})", text)
    assert carried == (
        "<{duration_cycles = 0 : i64, period_cycles = array<i64: 24>, start_cycles = 16 : i64}> "
        "{pulse.carried = 0 : i64}"
    )
    assert "pulse.exit_updates = array<i64: 0>" in text
    text = _compile(straight).mlir()
    assert re.findall(r"pulse\.(\w+_\w+) .*start_cycles = (\d+)", text) == [
        ("shift_frequency", "16"),
        ("set_phase", "88"),
        ("set_frequency", "104"),
    ]


def test_a_frame_change_no_operation_takes_is_kept_as_a_frame_update():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lowered = pulse.compile(trailing, quantum_system=_DEV).lower(Q[0])
    text = lowered.mlir()
    for op in ("shift_phase", "set_frequency"):
        assert re.search(rf"pulse\.{op} .*pulse\.unconsumed", text)
    flat = lowered.export_to_controller()
    kept = [e for e in flat["events"] if e["kind"] == "frame_update"]
    # A change is placed where its tone's previous drive ends; a wait on the
    # line does not move the tone. Each has the tone's frame after it.
    _, [(_, _, frequency, phase, time_ns)] = lowered._concrete().drive_frames([])
    assert time_ns == 16 / _CLOCK_GHZ
    detuned = 2.0 * math.pi * 3.0e6 * 1.0e-9 * time_ns
    assert [
        (e["start_cycles"], e["duration_cycles"], e["op"], e["value"], e["frequency_hz"])
        for e in kept
    ] == [
        (16, 0, "shift_phase", 0.4, _FREQUENCY_HZ + 3.0e6),
        (16, 0, "set_frequency", 5.002e9, 5.002e9),
    ]
    assert [e["phase_rad"] for e in kept] == [pytest.approx(detuned + 0.4)] * 2
    assert (kept[-1]["frequency_hz"], kept[-1]["phase_rad"]) == (frequency, pytest.approx(phase))
    assert "frame_state" not in flat
    structured = lowered.export_to_controller(version=2)
    assert unroll(structured) == flat
    # Every other event is as without the changes.
    plain = _compile(untrailed)
    assert [e for e in flat["events"] if e not in kept] == plain.export_to_controller()["events"]
    assert [e for e in structured["events"] if e["kind"] != "frame_update"] == (
        plain.export_to_controller(version=2)["events"]
    )


@pytest.mark.parametrize("kernel", [straight, untrailed])
def test_a_program_whose_frame_changes_are_all_taken_has_no_frame_update(kernel):
    lowered = _compile(kernel, unroll_threshold=0)
    assert "pulse.unconsumed" not in lowered.mlir()
    assert "pulse.carried" not in lowered.mlir()
    for version in (1, 2):
        assert "frame_update" not in json.dumps(lowered.export_to_controller(version=version))


def _updates(events):
    return [e for e in events if e["kind"] == "frame_update"]


def _close(a, b):
    """Equal, floats to rounding: a rolled loop moves a phase by its step
    times the iterations, where a walk of every iteration adds the step."""
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)
    if isinstance(a, dict) and isinstance(b, dict):
        return list(a) == list(b) and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b, strict=True))
    return a == b


@pytest.mark.parametrize("threshold", [0, None])
@pytest.mark.parametrize("kernel", [looped, nested])
def test_the_changes_a_loop_carries_out_of_its_last_iteration_follow_it(kernel, threshold):
    options = {} if threshold is None else {"unroll_threshold": threshold}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lowered = _compile(kernel, **options)
    flat = lowered.export_to_controller()
    # The last iteration's shift, after the loop: the frame the shot ends in.
    [update] = _updates(flat["events"])
    assert update is flat["events"][-1]
    assert (update["op"], update["value"]) == ("shift_phase", 0.125 if kernel is nested else 0.25)
    _, [(_, _, frequency, phase, time_ns)] = lowered._concrete().drive_frames([])
    assert update["start_cycles"] == round(time_ns * _CLOCK_GHZ)
    assert update["frequency_hz"] == frequency
    assert _wrapped(update["phase_rad"]) == pytest.approx(_wrapped(phase))
    structured = lowered.export_to_controller(version=2)
    assert _close(unroll(structured), flat)
    if threshold == 0:
        assert [e["kind"] for e in structured["events"]] == ["loop", "frame_update"]
    # The drives play as before.
    assert _exported(flat) == _approx(_simulated(lowered))


@pytest.mark.parametrize(("kernel", "arm"), [(one_arm, False), (other_arm, True)])
def test_a_change_before_a_branch_one_arm_plays_is_a_frame_update_in_the_other(kernel, arm):
    lowered = _compile(kernel)
    flat = lowered.export_to_controller()
    [update] = _updates(flat["events"])
    [condition] = update["conditions"]
    assert condition["arm"] is arm
    [played] = [e for e in flat["events"] if e["kind"] == "drive" and e["conditions"]]
    assert played["conditions"][0]["arm"] is not arm
    if kernel is one_arm:
        # Where the change is placed: after the first drive.
        assert (update["start_cycles"], update["op"], update["value"]) == (100, "shift_phase", 0.5)
        assert update["phase_rad"] == pytest.approx(0.5)
        assert played["phase_rad"] == pytest.approx(0.5)
    else:
        assert (update["op"], update["frequency_hz"]) == ("shift_frequency", _FREQUENCY_HZ + 2e6)
        assert played["frequency_hz"] == _FREQUENCY_HZ + 2e6
    assert unroll(lowered.export_to_controller(version=2)) == flat
    for outcome in (0, 1):
        assert _exported(flat, outcome) == _approx(_simulated(lowered, outcome))


@pytest.mark.parametrize("outcome", [0, 1])
def test_a_change_a_loop_carries_through_a_branch_is_a_frame_update_at_each_execution(outcome):
    lowered = _compile(branched_loop, unroll_threshold=0)
    assert "pulse.unconsumed" in lowered.mlir()
    flat = lowered.export_to_controller()
    updates = _updates(flat["events"])
    assert [(e["op"], e["conditions"]) for e in updates] == [("shift_phase", [])] * 3
    assert [e["phase_rad"] for e in updates] == pytest.approx([0.3, 0.6, 0.9])
    # A drive that plays with them has them in its frame as well.
    assert _exported(flat, outcome) == _approx(_simulated(lowered, outcome))
    assert _close(unroll(lowered.export_to_controller(version=2)), flat)


@pytest.mark.parametrize("threshold", [None, 0])
def test_a_change_after_a_drive_in_a_loop_is_where_the_drive_ends(threshold):
    # With a threshold of 0 the second iteration is a loop of one, whose
    # drive repeats with no period.
    lowered = pulse.compile(measured_rounds, quantum_system=_DEV, unroll_threshold=threshold)
    flat = lowered.lower(Q[0], Q[1]).export_to_controller()
    [update] = _updates(flat["events"])
    drives = [e for e in flat["events"] if e["kind"] == "drive" and e["line"] == update["line"]]
    last = max(d["end_cycles"] for d in drives if not d.get("conditions"))
    assert update["start_cycles"] == last


def test_a_change_in_a_loop_on_a_tone_it_does_not_play_keeps_the_loop_rolled():
    lowered = pulse.compile(idle, quantum_system=_DEV, unroll_threshold=0).lower(Q[0], Q[1])
    plain = pulse.compile(idle_unchanged, quantum_system=_DEV, unroll_threshold=0).lower(Q[0], Q[1])
    # Each iteration's change at the start of the iteration.
    [stamp] = re.findall(r"pulse\.shift_phase [^<]*(<\{.*?\}> \{.*?\})", lowered.mlir())
    assert stamp == (
        "<{duration_cycles = 0 : i64, period_cycles = array<i64: 16>, start_cycles = 0 : i64}> "
        "{pulse.unconsumed}"
    )
    structured = lowered.export_to_controller(version=2)
    unchanged = plain.export_to_controller(version=2)
    assert [e["kind"] for e in structured["events"]] == ["play", "loop"]
    loop, before = structured["events"][1], unchanged["events"][1]
    assert [e["kind"] for e in loop["body"]] == ["play", "frame_update"]
    # The program is the one without the change, and the frame update.
    assert loop["duration_cycles"] == before["duration_cycles"] == 64
    assert {**loop, "body": loop["body"][:1]} == before
    rest = {**structured, "events": [structured["events"][0], before]}
    assert json.dumps(rest) == json.dumps(unchanged)
    assert lowered.duration == plain.duration
    flat = lowered.export_to_controller()
    assert [e["start_cycles"] for e in _updates(flat["events"])] == [0, 16, 32, 48]
    assert [e["phase_rad"] for e in _updates(flat["events"])] == pytest.approx(
        [0.25, 0.5, 0.75, 1.0]
    )
    assert unroll(structured) == flat


def test_frames_that_persist_keep_the_last_changes_and_export_the_final_frame():
    lowered = pulse.compile(trailing, quantum_system=_DEV, control_system=_PERSISTENT).lower(Q[0])
    assert "pulse.shift_phase" in lowered.mlir()
    flat = lowered.export_to_controller()
    structured = lowered.export_to_controller(version=2)
    assert flat["limits"]["persistent_frames"] is True
    assert structured["controller"]["persistent_frames"] is True
    _, finals = lowered._concrete().drive_frames([])
    [(qudit, role, frequency, phase, time_ns)] = finals
    assert (qudit, role, frequency) == (0, "drive", 5.002e9)
    end_ns = max(e["end_cycles"] for e in flat["events"]) / _CLOCK_GHZ
    phase += 2.0 * math.pi * (frequency - _FREQUENCY_HZ) * 1.0e-9 * (end_ns - time_ns)
    [frame] = flat["frame_state"]
    assert frame == {"line": 0, "frequency_hz": 5.002e9, "phase_rad": pytest.approx(phase)}
    assert structured["frame_state"] == flat["frame_state"]
    assert unroll(structured)["frame_state"] == flat["frame_state"]
    # The drives play as without the declaration.
    plain = pulse.compile(untrailed, quantum_system=_DEV).lower(Q[0]).export_to_controller()
    assert flat["events"] == plain["events"]


def test_persistent_frames_is_exported_only_when_declared():
    assert "persistent_frames" not in Controller().as_dict()
    assert Controller(persistent_frames=True).as_dict()["persistent_frames"] is True
    with pytest.raises(pulse.InvalidValueError, match="persistent_frames must be a bool"):
        Controller(persistent_frames=1)


_FRAME_CHANGES = ("shift_phase", "set_phase", "shift_frequency", "set_frequency")


def _kept(tone, value):
    return tone


def _without_frame_changes(kernel):
    """*kernel* with each of its frame changes leaving the tone as it is."""
    fn = kernel.__wrapped__
    names = {**fn.__globals__, **dict.fromkeys(_FRAME_CHANGES, _kept)}
    bare = types.FunctionType(fn.__code__, names, fn.__name__, fn.__defaults__, fn.__closure__)
    bare.__qualname__ = f"{fn.__qualname__}_without_frame_changes"
    return pulse.kernel(bare)


def _timing(events):
    """The timing of every event but the frame updates, loops' bodies
    included, in order."""
    timed = []
    for event in events:
        if event["kind"] == "frame_update":
            continue
        timed.append(
            tuple(event.get(key) for key in ("kind", "line", "start_cycles", "duration_cycles"))
            + (event.get("end_cycles"), event.get("period_cycles"), event.get("trips"))
        )
        timed.extend(_timing(event.get("body", ())))
    return timed


_TIMING_CORPUS = [
    *(
        (kind, kernel, _validation._device(ControlSystem()))
        for kind, kernel in _validation._corpus().items()
    ),
    *(
        (kernel.__wrapped__.__name__, kernel, _DEV)
        for kernel in (
            straight,
            looped,
            nested,
            branched,
            trailing,
            untrailed,
            one_arm,
            other_arm,
            branched_loop,
            measured_rounds,
            idle,
        )
    ),
]


_UNFUSED = [p for p in DEFAULT_PASSES if p != Pass.FUSE_PULSES]


@pytest.mark.parametrize("threshold", [None, 0], ids=["default", "rolled"])
@pytest.mark.parametrize(
    ("name", "kernel", "quantum_system"), _TIMING_CORPUS, ids=[n for n, _, _ in _TIMING_CORPUS]
)
def test_keeping_frame_changes_moves_and_lengthens_nothing(name, kernel, quantum_system, threshold):
    qudits = quantum_system.qudits[: kernel.__wrapped__.__code__.co_argcount]
    options = {} if threshold is None else {"unroll_threshold": threshold}
    bare_kernel = _without_frame_changes(kernel)

    def lower(kernel, **more):
        return pulse.compile(kernel, quantum_system=quantum_system, **options, **more).lower(
            *qudits
        )

    assert lower(kernel).duration == lower(bare_kernel).duration
    # Without fusion, which a change between two drives keeps apart, every
    # event is where it is without the changes.
    kept, bare = lower(kernel, passes=_UNFUSED), lower(bare_kernel, passes=_UNFUSED)
    assert not any(f"pulse.{op}" in bare.mlir() for op in _FRAME_CHANGES)
    assert kept.duration == bare.duration
    assert [(e.start_cycles, e.duration_cycles) for e in kept.schedule] == [
        (e.start_cycles, e.duration_cycles) for e in bare.schedule
    ]
    end = max((e.start_cycles + e.duration_cycles for e in bare.schedule), default=0)
    for version in (1, 2):
        exported = kept.export_to_controller(version=version)
        assert _timing(exported["events"]) == _timing(
            bare.export_to_controller(version=version)["events"]
        )
        for update in _updates(unroll(exported)["events"]):
            assert update["duration_cycles"] == 0 and 0 <= update["start_cycles"] <= end


def _branches(n):
    @pulse.kernel(unroll_threshold=10**9)
    def branches(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        m = readout(r, square(40, 0.05))
        sync(d, r)
        for _ in range(n):
            if m:
                shift_frequency(d.tone, 1.0e6)
                drive(d, square(8, 0.1))

    return branches


@pytest.mark.slow
def test_a_run_of_branches_places_its_frame_changes_in_linear_time():
    # Each branch passes its tone through the arm that does not play, so a
    # change is free where the branches before it, all of them, leave its
    # tone. The Python work is counted; the schedule, native, is the best of
    # five times.
    work, timings = {}, {}
    for n in (1000, 4000):
        kernel = _branches(n)

        def run(kernel=kernel):
            return pulse.compile(kernel).lower(Q[0])

        work[n] = python_work(run)
        timings[n] = min(run().metrics.schedule_ms for _ in range(5))
    # 4x the branches in well under 16x the time (quadratic would be 16x).
    assert all(work[4000][count] <= 4 * work[1000][count] for count in work[1000]), work
    assert timings[4000] < 8 * timings[1000], timings


_LONG_RUN = """
import cudaq_pulse as pulse
from tests._handles import Q

@pulse.kernel(unroll_threshold=10**9)
def chain(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    m = readout(r, square(40, 0.05))
    sync(d, r)
    for _ in range(100_000):
        if m:
            drive(d, square(8, 0.1))
    shift_frequency(d.tone, 1.0e6)
    drive(d, square(8, 0.1))

print(pulse.compile(chain).lower(Q[0]).metrics.op_count)
"""


@pytest.mark.slow
def test_a_frame_change_after_a_run_of_branches_as_long_as_the_program_is_placed():
    # Finding where the tone is free goes back through every branch, which
    # a recursive walk did one stack frame each: it overflowed the stack.
    code = _child.limits(60, 3000) + _LONG_RUN
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=root, env=env, timeout=240
    )
    assert out.returncode == 0, f"the child exited {out.returncode}: {out.stderr[-2000:]}"
    assert int(out.stdout.split()[-1]) > 100_000
