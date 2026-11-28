# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The structured export format (/2) and limit checks that keep loops rolled."""

from __future__ import annotations

import copy
import json
import math
import sys
import warnings

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import Controller, ControlSystem, backends
from cudaq_pulse.classical import Link
from cudaq_pulse.controller import FORMAT, ControllerLimitError, check, unroll
from cudaq_pulse.quantum_systems import PhysicalQudit

from tests.test_controller import answered, chained, chosen_by_readout, frame_in_arms, settings

_TARGET = pulse.quantum_system(
    name="structured",
    clock_ghz=2.0,
    qudits={
        i: PhysicalQudit(i, 5.0e9 + 1.0e7 * i, 50.0, 30.0, anharmonicity_hz=-2.0e8)
        for i in range(4)
    },
)
_DEV = _TARGET
Q = _DEV.qudits


@pulse.classical(max_latency_ns=200, link="ctl")
def _decide(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]


_MACHINE = ControlSystem(links={"ctl": Link(fixed_ns=4, feedback_ns=8)})


def _sampled(per_cycle, **limits):
    """Limits of a controller whose lines play *per_cycle* samples per cycle."""
    return Controller(sample_rate_ghz=per_cycle * _TARGET.clock_ghz, **limits)


def _rounds(n):
    @pulse.kernel
    def rounds(a, b):
        d = get_drive_line(a)
        r = get_readout_line(a)
        e = get_drive_line(b)
        for _ in range(n):
            drive(d, gaussian(16, 0.1, 4.0))
            shift_phase(d.tone, 0.3)
            sync(d, r, e)
            m = readout(r, square(40, 0.05))
            sync(d, r, e)
            if m:
                drive(e, drag(16, 0.2, 4.0, 0.1))
            else:
                drive(d, square(8, 0.1))

    return rounds


@pulse.kernel
def _detuned(a):
    d = get_drive_line(a)
    shift_frequency(d.tone, 1.0e6)
    for _ in range(4):
        for _ in range(3):
            drive(d, gaussian(16, 0.1, 4.0))
            shift_phase(d.tone, 0.125)
            wait(d, 7)
        drive(d, cosine(12, 0.3))


@pulse.kernel
def _chirp(a):
    d = get_drive_line(a)
    for _ in range(6):
        shift_frequency(d.tone, 2.0e5)
        drive(d, square(8, 0.1))


@pulse.kernel
def _kept_frames(a):
    d = get_drive_line(a)
    ro = get_readout_line(a)
    for _ in range(3):
        drive(d, gaussian(16, 0.1, 4.0))
        shift_phase(ro.tone, 0.2)
    shift_frequency(d.tone, 2.0e5)


@pulse.kernel
def _first_differs(a):
    d = get_drive_line(a)
    for _ in range(5):
        set_frequency(d.tone, 5.01e9)
        drive(d, square(8, 0.1))
        wait(d, 3)


@pulse.kernel
def _arithmetic(a, b):
    d = get_drive_line(a)
    e = get_drive_line(b)
    for _ in range(3):
        drive(d, wf_add(gaussian(16, 0.1, 4.0), square(16, 0.05)))
        drive(e, custom_samples([0.0, 0.5, 1.0, 0.5]))
        drive(d, wf_neg(gaussian_square(24, 0.2, 4.0, 8)))
        drive(e, tanh_ramp(16, 0.3, 2.0))


@pulse.kernel
def _calls(a):
    d = get_drive_line(a)
    r = get_readout_line(a)
    for _ in range(4):
        m = readout(r, square(40, 0.05))
        bit = _decide.submit([m]).result()
        sync(d, r)
        if bit:
            drive(d, square(16, 0.1))
        drive(d, gaussian(16, 0.1, 4.0))


_CASES = {
    "branches": (_rounds(5), (Q[0], Q[1]), {}, {}),
    "unrolled": (_rounds(5), (Q[0], Q[1]), {"schedule": "alap", "unroll_threshold": 10**6}, {}),
    "detuned": (_detuned, (Q[0],), {}, {}),
    "chirp": (_chirp, (Q[0],), {}, {}),
    "kept_frames": (_kept_frames, (Q[0],), {}, {}),
    "first_differs": (_first_differs, (Q[0],), {}, {}),
    "arithmetic": (_arithmetic, (Q[0], Q[1]), {}, {}),
    "samples": (_arithmetic, (Q[0], Q[1]), {}, {"sample_rate_ghz": 4.0}),
    "calls": (_calls, (Q[0],), {"control_system": _MACHINE}, {}),
}


def _lower(name):
    kernel, args, options, export = _CASES[name]
    return pulse.compile(kernel, **{"unroll_threshold": 0, **options}).lower(*args), export


def _canonical(program):
    """Branches renumbered in the order they first appear."""
    ids: dict[int, int] = {}

    def renumber(conditions):
        return [{**c, "branch": ids.setdefault(c["branch"], len(ids))} for c in conditions]

    events = []
    for event in program["events"]:
        event = {**event, "conditions": renumber(event["conditions"])}
        if "frame_updates" in event:
            event["frame_updates"] = [
                {**u, "conditions": renumber(u["conditions"])} for u in event["frame_updates"]
            ]
        events.append(event)
    return {**program, "events": events}


def _differences(a, b, path=""):
    if isinstance(a, float) and isinstance(b, float):
        return [] if math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9) else [(path, a, b)]
    if isinstance(a, dict) and isinstance(b, dict):
        if list(a) != list(b):
            return [(path, list(a), list(b))]
        return [d for k in a for d in _differences(a[k], b[k], f"{path}.{k}")]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [(path, len(a), len(b))]
        return [
            d
            for i, (x, y) in enumerate(zip(a, b, strict=True))
            for d in _differences(x, y, f"{path}[{i}]")
        ]
    return [] if a == b else [(path, a, b)]


def _loops(events):
    for event in events:
        if event["kind"] == "loop":
            yield event
            yield from _loops(event["body"])


@pytest.mark.parametrize("name", sorted(_CASES))
def test_unrolling_the_structured_export_gives_the_flat_one(name):
    lowered, export = _lower(name)
    flat = lowered.export_to_controller(**export)
    structured = lowered.export_to_controller(**export, version=2)
    assert structured["format"] == "cudaq-pulse.controller/2"
    assert structured["programs"] == {} and structured["bindings"] == {}
    assert _differences(_canonical(flat), _canonical(unroll(structured))) == []
    assert len(json.dumps(structured)) < len(json.dumps(flat))


@pulse.kernel
def _retuned_after_rolled(a, b):
    d = get_drive_line(a)
    e = get_drive_line(b)
    shift_frequency(d.tone, 3.0e6)
    drive(e, square(40, 0.1))
    for _ in range(3):
        drive(e, square(16, 0.1))
        shift_phase(d.tone, 0.37)
    shift_frequency(d.tone, 3.0e6)
    drive(d, square(8, 0.1))


@pytest.mark.parametrize("schedule", ["asap", "alap", "rcp", "alap_rcp"])
def test_a_frame_retuned_after_a_rolled_loop_keeps_the_last_iterations_time(schedule):
    """A rolled loop leaves its frames as its last iteration does, at the time
    of their last change in it. Moved on from the state before the loop, whose
    last change is not one period earlier, a frame's time overshoots, and a
    retune between the two times shifts the phase (by 0.45 rad under asap)."""
    lowered = pulse.compile(_retuned_after_rolled, unroll_threshold=0, schedule=schedule).lower(
        Q[0], Q[1]
    )
    flat = lowered.export_to_controller()
    structured = lowered.export_to_controller(version=2)
    assert any(event["kind"] == "loop" for event in structured["events"])

    def split(event):
        """The event without its phases, and its phases."""
        tones = [{k: v for k, v in t.items() if k != "phase_rad"} for t in event.get("tones", [])]
        rest = {k: v for k, v in event.items() if k != "phase_rad"} | {"tones": tones}
        return rest, [event.get("phase_rad")] + [t["phase_rad"] for t in event.get("tones", [])]

    unrolled = unroll(structured)
    assert len(unrolled["events"]) == len(flat["events"])
    for ours, theirs in zip(flat["events"], unrolled["events"], strict=True):
        (rest, phases), (other, others) = split(ours), split(theirs)
        assert rest == other
        for phase, unrolled_phase in zip(phases, others, strict=True):
            assert (phase is None) == (unrolled_phase is None)
            assert phase is None or abs(phase - unrolled_phase) <= 1e-9, (ours, theirs)


def _kinds(events):
    for event in events:
        yield event["kind"]
        if event["kind"] == "loop":
            yield from _kinds(event["body"])


def test_every_structured_event_kind_is_an_event_and_exported_as_a_plain_string():
    seen = set()
    for name in _CASES:
        lowered, export = _lower(name)
        program = json.loads(json.dumps(lowered.export_to_controller(**export, version=2)))
        seen.update(_kinds(program["events"]))
    assert seen <= set(pulse.Event)
    assert {"play", "readout", "sync", "loop", "call", "answer"} <= seen
    assert pulse.Event.PLAY == "play" and pulse.Event("readout") is pulse.Event.READOUT


# Each kind of /2 event, its string, and the /1 kind it unrolls to.
_EVENT_TABLE = [
    (pulse.Event.PLAY, "play", "drive"),
    (pulse.Event.READOUT, "readout", "readout"),
    (pulse.Event.WAIT, "wait", "wait"),
    (pulse.Event.SYNC, "sync", "sync"),
    (pulse.Event.LOOP, "loop", None),
    (pulse.Event.CALL, "call", "submit"),
    (pulse.Event.ANSWER, "answer", "answer"),
    (pulse.Event.FRAME_UPDATE, "frame_update", "frame_update"),
]


def _executions(events, times, counts):
    for event in events:
        if event["kind"] == "loop":
            _executions(event["body"], times * event["trips"], counts)
        else:
            counts[event["kind"]] = counts.get(event["kind"], 0) + times
    return counts


@pytest.mark.parametrize(
    ("member", "text", "flat"), _EVENT_TABLE, ids=[t for _, t, _ in _EVENT_TABLE]
)
def test_each_event_kind_is_its_string_and_unrolls_to_its_flat_kind(member, text, flat):
    assert [m for m, _, _ in _EVENT_TABLE] == list(pulse.Event)
    assert member == text and str(member) == text and pulse.Event(text) is member
    exported = 0
    for name in _CASES:
        lowered, export = _lower(name)
        structured = lowered.export_to_controller(**export, version=2)
        if member is pulse.Event.LOOP:
            exported += len(list(_loops(structured["events"])))
            continue
        times = _executions(structured["events"], 1, {}).get(text, 0)
        flat_events = lowered.export_to_controller(**export)["events"]
        assert times == sum(e["kind"] == flat for e in flat_events), name
        exported += times
    assert exported, f"no case exports a {text!r} event"


def test_unrolling_refuses_an_event_kind_the_format_does_not_have():
    lowered, _ = _lower("branches")
    program = lowered.export_to_controller(version=2)
    program["events"].append({**program["events"][0], "kind": "teleport"})
    with pytest.raises(ValueError, match="cannot unroll events of kind 'teleport'"):
        unroll(program)


def test_a_program_states_its_sample_rate_beside_its_clock():
    kernel = _rounds(2)
    lowered = pulse.compile(kernel, unroll_threshold=0).lower(Q[0], Q[1])
    plain = lowered.export_to_controller(version=2)
    assert list(plain)[2:4] == ["clock_ghz", "sample_rate_ghz"]
    assert plain["sample_rate_ghz"] == _TARGET.clock_ghz
    assert "samples_per_cycle" not in plain
    assert unroll(plain) == lowered.export_to_controller()
    whole = lowered.export_to_controller(version=2, sample_rate_ghz=2 * _TARGET.clock_ghz)
    assert unroll(whole) == lowered.export_to_controller(sample_rate_ghz=2 * _TARGET.clock_ghz)

    # 1.5 samples per cycle: every waveform here is a whole number of them.
    sampled = lowered.export_to_controller(version=2, sample_rate_ghz=3.0)
    rate, clock = sampled["sample_rate_ghz"], sampled["clock_ghz"]
    assert rate == 3.0
    played = list(_played(sampled["events"]))
    assert played
    for event in played:
        samples = sampled["samples"][event["waveform"]]
        assert len(samples["re"]) == event["duration_cycles"] * rate / clock
    with pytest.raises(ValueError, match="whole multiple of the clock"):
        unroll(sampled)
    with pytest.raises(ValueError, match="16 cycles at 1.1 samples per cycle is 17.6 samples"):
        lowered.export_to_controller(version=2, sample_rate_ghz=2.2)

    machine = ControlSystem(controllers=[_sampled(4)])
    lowered = pulse.compile(kernel, control_system=machine, unroll_threshold=0).lower(Q[0], Q[1])
    program = lowered.export_to_controller(version=2)
    assert program["sample_rate_ghz"] == program["controller"]["sample_rate_ghz"] == 8.0
    assert unroll(program) == lowered.export_to_controller()
    with pytest.raises(ValueError, match="the controller samples at 8.0 GHz"):
        lowered.export_to_controller(version=2, sample_rate_ghz=6.0)


def test_unrolling_a_flat_export_returns_it():
    lowered, _ = _lower("branches")
    flat = lowered.export_to_controller()
    assert unroll(flat) == flat and flat["format"] == FORMAT


def _shape(value):
    """`value` with every number replaced by its type: what is left is the
    program's structure (events, their kinds and lines, nesting, keys)."""
    if isinstance(value, dict):
        return {
            key: (item if key in ("kind", "op") else _shape(item)) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_shape(item) for item in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return type(value).__name__
    return value


def test_a_repeating_loop_stays_one_loop_whatever_its_trip_count():
    shapes = {}
    for n in (8, 4000):
        program = (
            pulse.compile(_rounds(n), unroll_threshold=0)
            .lower(Q[0], Q[1])
            .export_to_controller(version=2)
        )
        (loop,) = [e for e in program["events"] if e["kind"] == "loop"]
        assert loop["trips"] + loop.get("first_iteration", 0) <= n
        assert all(event["iteration"] == [None] for event in loop["body"])
        shapes[n] = _shape(program)
    # Only the numbers differ (times, the trip count, phases): the export's
    # events, their kinds, nesting and keys are the same at any trip count.
    assert shapes[4000] == shapes[8]


def test_a_loop_body_carries_its_period_and_phase_steps():
    program = _lower("detuned")[0].export_to_controller(version=2)
    flat = unroll(program)
    outer = [e for e in _loops(program["events"])][0]
    assert outer["period_cycles"] > 0
    played = [e for e in flat["events"] if e["kind"] == "drive"]
    assert len(played) == 16
    in_loops = [e["tones"] for loop in _loops(program["events"]) for e in _played(loop["body"])]
    assert in_loops and all(len(t) == 1 and "phase_step_rad" in t[0] for t in in_loops)
    outside = [e["tones"] for e in program["events"] if e["kind"] == "play"]
    assert all("phase_step_rad" not in t[0] for t in outside)


def _played(events):
    for event in events:
        if event["kind"] == "loop":
            yield from _played(event["body"])
        elif event["kind"] in ("play", "readout"):
            yield event


def test_a_loop_whose_frequency_changes_every_iteration_is_listed_per_iteration():
    program = _lower("chirp")[0].export_to_controller(version=2)
    assert not list(_loops(program["events"]))
    frequencies = [e["tones"][0]["frequency_hz"] for e in program["events"]]
    assert len(set(frequencies)) == 6


def test_a_first_iteration_that_differs_is_exported_before_its_loop():
    program = _lower("first_differs")[0].export_to_controller(version=2)
    kinds = [e["kind"] for e in program["events"]]
    assert kinds[-1] == "loop" and "play" in kinds[:-1]
    assert program["events"][-1]["first_iteration"] == 1


@pulse.kernel
def _first_in_loop(q):
    # The drive line plays first in the loop; the sync after it waits for it.
    d = get_drive_line(q)
    r = get_readout_line(q)
    readout(r, square(10, 0.05))
    for _ in range(6):
        drive(d, square(16, 0.1))
        wait(d, 4)
    sync(d, r)
    readout(r, square(20, 0.05))


@pytest.mark.parametrize("schedule", ["asap", "alap"])
def test_a_line_first_played_in_a_rolled_loop_is_free_after_its_last_iteration(schedule):
    lowered = pulse.compile(_first_in_loop, schedule=schedule, unroll_threshold=0).lower(Q[0])
    structured = lowered.export_to_controller(version=2)
    assert list(_loops(structured["events"]))
    assert unroll(structured)["events"] == lowered.export_to_controller()["events"]


def test_waveforms_are_listed_once_and_lines_list_their_qudits():
    lowered, _ = _lower("unrolled")
    program = lowered.export_to_controller(version=2)
    played = list(_played(program["events"]))
    # gaussian, square(40), drag and square(8), each listed once.
    assert len(program["waveforms"]) == 4 < len(played)
    assert {e["waveform"] for e in played} == {0, 1, 2, 3}
    flat_lines = lowered.export_to_controller()["lines"]
    assert [line["qudits"] for line in program["lines"]] == [[line["qudit"]] for line in flat_lines]
    assert all("index" not in e and "qudit" not in e and "end_cycles" not in e for e in played)


def test_the_structured_format_is_refused_by_an_unknown_version():
    lowered, _ = _lower("branches")
    with pytest.raises(ValueError, match="version must be 1 or 2"):
        lowered.export_to_controller(version=3)
    with pytest.raises(ValueError, match="cannot unroll"):
        unroll({"format": "cudaq-pulse.controller/9"})


# ── Limit checks on rolled loops ────────────────────────────────────────


# What a controller that plays rolled loops unrolled supports.
_NO_LOOP = [c for c in pulse.Capability if c is not pulse.Capability.LOOP]
# Built per test, so that a limit the controller no longer takes fails its
# own cases only.
_LIMITS = {
    "issue": lambda: Controller(instruction_cycles=3, min_idle_cycles=5),
    "tones": lambda: Controller(max_frequencies_per_line=1),
    "waveform_memory_samples": lambda: _sampled(2, waveform_memory_samples=60),
    "shared_memory": lambda: _sampled(2, waveform_memory_samples=60, waveform_memory="shared"),
    "detectors": lambda: Controller(detectors={0: 0, 1: 0}, detector_recovery_cycles=500),
    "feedback": lambda: Controller(feedback_latency_cycles=50),
    "max_pulses": lambda: Controller(max_pulses=7, supports=_NO_LOOP),
}


@pytest.mark.parametrize("name", list(_LIMITS))
@pytest.mark.parametrize("n", [3, 40])
def test_a_rolled_loop_is_checked_like_the_same_loop_unrolled(name, n):
    limits = _LIMITS[name]()
    rolled = pulse.compile(_rounds(n), unroll_threshold=0).lower(Q[0], Q[1])
    unrolled = pulse.compile(_rounds(n), unroll_threshold=10**6).lower(Q[0], Q[1])
    assert rolled.schedule == unrolled.schedule
    found, every = check(rolled, limits), check(unrolled, limits)
    assert bool(found) == bool(every)
    # Only the rolled loop is a loop the event count says is unrolled.
    assert {m.split(" (in each of")[0].split(". It does not")[0] for m in found} <= set(every)


def test_a_violation_in_every_iteration_is_reported_for_each_iteration():
    @pulse.kernel
    def pairs(q):
        d = get_drive_line(q)
        for _ in range(1000):
            drive(d, square(8, 0.1))
            drive(d, square(8, 0.2))
            wait(d, 20)

    limits = Controller(instruction_cycles=3, min_idle_cycles=6, max_pulses=10, supports=_NO_LOOP)
    lowered = pulse.compile(pairs, unroll_threshold=0).lower(Q[0])
    unrolled = pulse.compile(pairs, unroll_threshold=10**6).lower(Q[0])
    found, every = check(lowered, limits), check(unrolled, limits)
    assert found[0] == (
        "2000 drives and readouts; the controller plays at most 10. It does not support "
        "'loop', so rolled loops are unrolled for it: the loop of 1000 iterations from "
        "cycle 0 plays 1998 more"
    )
    assert every[0] == "2000 drives and readouts; the controller plays at most 10"
    assert len(found) == len(every) == 1001
    assert list(found[1:]) == list(every[1:])
    assert found[-1] == (
        "line 0: drive at 35972 starts 8 cycles after the drive at 35964; the controller "
        "needs exactly 3 or at least 9"
    )
    machine = ControlSystem(controllers=[Controller(instruction_cycles=3, min_idle_cycles=6)])
    spaced = check(unrolled, machine.controllers[0])
    assert len(spaced) == 1000
    with pytest.raises(ControllerLimitError) as error:
        pulse.compile(pairs, control_system=machine, unroll_threshold=0).lower(Q[0])
    assert list(error.value.violations) == list(spaced)


def test_a_backend_reading_only_the_structured_format_is_accepted():
    class Structured:
        name = "structured"
        formats = ("cudaq-pulse.controller/2",)

        def control_system(self):
            return ControlSystem()

        def emit(self, program):
            return program

        def run(self, artifact, *, shots_count=1):
            return [artifact] * shots_count

    class Unknown(Structured):
        formats = ("cudaq-pulse.controller/9",)

    backends.register("structured_only", Structured)
    backends.register("unknown_format", Unknown)
    try:
        assert backends.get("structured_only").name == "structured"
        with pytest.raises(backends.IncompatibleBackendError, match="controller/2"):
            backends.get("unknown_format")
    finally:
        backends.unregister("structured_only")
        backends.unregister("unknown_format")


# ── Values known at run time ────────────────────────────────────────────────


@pulse.kernel
def _fed_back(q):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    for _ in range(50):
        m = readout(ro, square(100, 0.05))
        amp, detuning, delay = settings.submit([m]).result()
        drive(d, square(20, amp * 0.5))
        sync(d, ro)


@pulse.kernel
def _tracked_in_loop(q):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    for _ in range(6):
        m = readout(ro, square(100, 0.05))
        amp, detuning, delay = settings.submit([m]).result()
        shift_phase(d.tone, amp)
        drive(d, square(20, 0.2))
        sync(d, ro)


@pulse.classical(max_latency_ns=200)
def _amp_and_phase(bits: pulse.Bits) -> tuple[pulse.Bits, float, float]:
    return [bits[0]], 0.5, 12.0


@pulse.kernel
def _answer_amplitude(q):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    m = readout(ro, square(100, 0.05))
    amp, phase = _amp_and_phase.submit([m]).result()[1:]
    drive(d, square(20, amp))
    shift_phase(d.tone, phase)


@pulse.kernel
def _answer_amplitude_unshifted(q):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    m = readout(ro, square(100, 0.05))
    amp, phase = _amp_and_phase.submit([m]).result()[1:]
    drive(d, square(20, amp))


@pulse.kernel
def _answer_phase_then_drive(q):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    m = readout(ro, square(100, 0.05))
    amp, phase = _amp_and_phase.submit([m]).result()[1:]
    drive(d, square(20, amp))
    shift_phase(d.tone, phase)
    drive(d, square(20, 0.1))


_RUN_TIME = {
    "answer_amplitude": (_answer_amplitude, (Q[0],)),
    "answer_phase_then_drive": (_answer_phase_then_drive, (Q[0],)),
    "answered": (answered, (Q[0],)),
    "chained": (chained, (Q[0],)),
    "frame_in_arms": (frame_in_arms, (Q[0], Q[1])),
    "fed_back": (_fed_back, (Q[0],)),
    "tracked_in_loop": (_tracked_in_loop, (Q[0],)),
}
_RUN_TIME_KEYS = {"tone", "frame_updates", "phase_offset_rad", "length", "operands"}


def _run_time_keys(program):
    """The keys of values known only at run time in *program*, in either
    format; ``runtime`` stands for a waveform operand computed at run time."""

    def operand(node):
        if not isinstance(node, dict):
            return False
        params = node.get("params", {}).values()
        return (
            "runtime" in node
            or any(isinstance(v, dict) for v in params)
            or any(operand(child) for child in node.get("of", ()))
        )

    def visit(events):
        for event in events:
            found.update(_RUN_TIME_KEYS & event.keys())
            waveform = event.get("waveform")
            if isinstance(waveform, int):
                waveform = program["waveforms"][waveform]
            if operand(waveform):
                found.add("runtime")
            if event["kind"] == "loop":
                visit(event["body"])

    found = set()
    if "frame_state" in program:
        found.add("frame_state")
    visit(program["events"])
    return found


def _amplitude_of(structured, event):
    return structured["waveforms"][event["waveform"]]["params"]["amplitude"]


@pytest.mark.parametrize("name", sorted(_RUN_TIME))
def test_run_time_values_in_format_2_unroll_to_those_of_format_1(name):
    kernel, args = _RUN_TIME[name]
    lowered = pulse.compile(kernel, unroll_threshold=0).lower(*args)
    flat = lowered.export_to_controller()
    structured = lowered.export_to_controller(version=2)
    assert _run_time_keys(flat) == _run_time_keys(structured) != set()
    assert _differences(_canonical(flat), _canonical(unroll(structured))) == []
    assert json.loads(json.dumps(structured)) == structured
    assert "null" not in json.dumps(structured["waveforms"])


def test_a_value_a_readout_selects_unrolls_to_its_select(monkeypatch):
    monkeypatch.setattr(pulse.compiler, "_can_select_at_runtime", lambda target, kind: True)
    lowered = pulse.compile(chosen_by_readout, unroll_threshold=0).lower(Q[0])
    flat = lowered.export_to_controller()
    structured = lowered.export_to_controller(version=2)
    [play] = (e for e in structured["events"] if e["kind"] == "play")
    assert _amplitude_of(structured, play)["op"] == "select"
    assert _differences(_canonical(flat), _canonical(unroll(structured))) == []


def test_an_amplitude_and_a_phase_from_an_answer_are_exported_with_their_source():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lowered = pulse.compile(_answer_amplitude, unroll_threshold=0).lower(Q[0])
    structured = lowered.export_to_controller(version=2)
    answer = next(i for i, e in enumerate(structured["events"]) if e["kind"] == "answer")
    [play] = (e for e in structured["events"] if e["kind"] == "play")
    assert _amplitude_of(structured, play) == {"answer": answer, "result": 1}
    # The phase shift comes after the last drive on its tone: it is kept, as
    # a frame update of its own where the drive ends.
    phase = {"answer": answer, "result": 2}
    end = play["start_cycles"] + play["duration_cycles"]
    update = {"id": 0, "op": "shift_phase", "start_cycles": end, "value": phase}
    assert (play["tone"], play["frame_updates"]) == (0, [])
    [kept] = (e for e in structured["events"] if e["kind"] == "frame_update")
    assert kept == {
        "kind": "frame_update",
        "line": 0,
        "start_cycles": end,
        "duration_cycles": 0,
        "op": "shift_phase",
        "value": phase,
        "tones": [{"frequency_hz": 5.0e9, "phase_rad": None}],
        "tone": 0,
        "frame_updates": [update],
    }
    assert "frame_state" not in structured
    unshifted = pulse.compile(_answer_amplitude_unshifted, unroll_threshold=0).lower(Q[0])
    assert unshifted.export_to_controller(version=2)["events"] == [
        {k: v for k, v in e.items() if k not in ("tone", "frame_updates")}
        for e in structured["events"]
        if e is not kept
    ]

    # Frames that persist keep it, as the frame the tone ends the shot in.
    persistent = ControlSystem(controllers=[Controller(persistent_frames=True)])
    lowered = pulse.compile(_answer_amplitude, unroll_threshold=0, control_system=persistent).lower(
        Q[0]
    )
    structured = lowered.export_to_controller(version=2)
    [play] = (e for e in structured["events"] if e["kind"] == "play")
    assert (play["tone"], play["frame_updates"]) == (0, [])
    phase = {"answer": answer, "result": 2}
    drive_frame, readout_frame = structured["frame_state"]
    end = play["start_cycles"] + play["duration_cycles"]
    update = {"id": 0, "op": "shift_phase", "start_cycles": end, "value": phase}
    assert drive_frame == {"line": 0, "tone": 0, "frame_updates": [update]}
    assert readout_frame == {"line": 1, "frequency_hz": 5.0e9, "phase_rad": 0.0}
    flat = lowered.export_to_controller()
    keys = list(flat)
    assert keys[keys.index("events") + 1] == "frame_state"
    assert flat["frame_state"] == [
        {**drive_frame, "frame_updates": [{**update, "conditions": []}]},
        readout_frame,
    ]
    assert unroll(structured)["frame_state"] == flat["frame_state"]

    structured = (
        pulse.compile(_answer_phase_then_drive, unroll_threshold=0)
        .lower(Q[0])
        .export_to_controller(version=2)
    )
    assert "frame_updates" not in structured
    first, second = (e for e in structured["events"] if e["kind"] == "play")
    assert first["frame_updates"] == []
    [update] = second["frame_updates"]
    assert (update["op"], update["value"]) == ("shift_phase", phase)
    assert second["tone"] == first["tone"]


def test_a_loop_whose_calls_and_amplitudes_come_from_answers_stays_rolled():
    structured = (
        pulse.compile(_fed_back, unroll_threshold=0).lower(Q[0]).export_to_controller(version=2)
    )
    [loop] = (e for e in structured["events"] if e["kind"] == "loop")
    assert loop["trips"] == 50
    body = loop["body"]
    place = {e["kind"]: structured["events"].index(loop) + 1 + i for i, e in enumerate(body)}
    by_kind = {e["kind"]: e for e in body}
    # References are places in the export: the body's own events.
    assert by_kind["call"]["operands"] == [{"readout": place["readout"]}]
    assert _amplitude_of(structured, by_kind["play"]) == {
        "of": [{"answer": place["answer"], "result": 0}, {"const": 0.5}],
        "op": "mul",
        "type": "float",
    }


def test_a_loop_that_changes_a_tracked_frame_is_listed_per_iteration():
    structured = (
        pulse.compile(_tracked_in_loop, unroll_threshold=0)
        .lower(Q[0])
        .export_to_controller(version=2)
    )
    assert not any(e["kind"] == "loop" for e in structured["events"])
    ids = [u["id"] for e in structured["events"] for u in e.get("frame_updates", ())]
    assert len(ids) == len(set(ids)) == 6


def test_an_amplitude_over_the_limit_in_a_rolled_loop_is_reported_for_each_iteration():
    @pulse.kernel
    def strong(q):
        d = get_drive_line(q)
        for _ in range(1000):
            drive(d, square(16, 0.9))
            drive(d, square(16, 0.2))

    lowered = pulse.compile(strong, unroll_threshold=0).lower(Q[0])
    unrolled = pulse.compile(strong, unroll_threshold=10**6).lower(Q[0])
    limits = Controller(max_amplitude=0.5)
    found = check(lowered, limits)
    assert list(found) == list(check(unrolled, limits))
    assert len(found) == 1000
    assert found[0] == (
        "line 0 (drive of qudit 0): the drive at 0 peaks at amplitude 0.9; the controller "
        "plays at most 0.5 (Controller.max_amplitude)"
    )
    assert found[-1] == found[0].replace("at 0 peaks", "at 31968 peaks")
    assert check(lowered, Controller(max_amplitude=1.0)) == []


@pulse.kernel
def _loop_of_loops(q):
    d = get_drive_line(q)
    for _ in range(6):
        for _ in range(4):
            drive(d, gaussian(16, 0.1, 4.0))
            shift_phase(d.tone, 0.125)
            wait(d, 8)
        drive(d, square(16, 0.1))
        wait(d, 8)


def _calls_of(target, fn):
    """The calls of *target* that Python code makes while *fn* runs."""
    monitoring = sys.monitoring
    tool = next((i for i in range(6) if monitoring.get_tool(i) is None), None)
    if tool is None:
        pytest.skip("every sys.monitoring tool id is in use")
    calls = 0

    def call(code, offset, called, first):
        nonlocal calls
        calls += called is target

    monitoring.use_tool_id(tool, "cudaq-pulse tests")
    try:
        monitoring.register_callback(tool, monitoring.events.CALL, call)
        monitoring.set_events(tool, monitoring.events.CALL)
        fn()
    finally:
        monitoring.set_events(tool, monitoring.events.NO_EVENTS)
        monitoring.register_callback(tool, monitoring.events.CALL, None)
        monitoring.free_tool_id(tool)
    return calls


def _shifted(structured, by):
    """*structured* as another exporter may write it: *by* more cycles in
    every loop period."""
    program = copy.deepcopy(structured)
    for loop in _loops(program["events"]):
        loop["period_cycles"] += by
    return program


def test_unrolling_carries_the_time_of_whole_periods_from_loop_to_event():
    lowered = pulse.compile(_loop_of_loops, unroll_threshold=0).lower(Q[0])
    structured = lowered.export_to_controller(version=2)
    outer, inner = _loops(structured["events"])
    assert inner in outer["body"] and (outer["trips"], inner["trips"]) == (6, 4)
    flat = unroll(structured)
    assert _differences(_canonical(lowered.export_to_controller()), _canonical(flat)) == []
    assert all(type(e["start_cycles"]) is int for e in flat["events"])
    # An event's segment, and a tone's phase, are sums over the loops around
    # it; its time is the loops' own, added up once per iteration.
    toned = sum(e["kind"] in ("drive", "readout", "frame_update") for e in flat["events"])
    assert _calls_of(sum, lambda: unroll(structured)) <= len(flat["events"]) + toned


@pytest.mark.parametrize("by", [0.25, 0.1, 1.0])
def test_unrolling_adds_up_periods_that_are_not_integers_as_they_are_written(by):
    lowered = pulse.compile(_loop_of_loops, unroll_threshold=0).lower(Q[0])
    structured = lowered.export_to_controller(version=2)
    flat = unroll(structured)
    moved = unroll(_shifted(structured, by))
    whole = [loop["period_cycles"] for loop in _loops(structured["events"])]
    periods = [period + by for period in whole]
    assert [e["iteration"] for e in moved["events"]] == [e["iteration"] for e in flat["events"]]
    for was, event in zip(flat["events"], moved["events"], strict=True):
        # The first iteration's time, plus one sum of the loops' periods.
        j = event["iteration"]
        first = was["start_cycles"] - sum(k * p for k, p in zip(j, whole, strict=False))
        start = first + sum(k * p for k, p in zip(j, periods, strict=False))
        assert type(event["start_cycles"]) is float
        assert event["start_cycles"] == start
        assert event["end_cycles"] == start + event["duration_cycles"]
    # Only the times move: branches, references and frames are as they were.
    assert [{**e, "start_cycles": 0, "end_cycles": 0} for e in moved["events"]] == [
        {**e, "start_cycles": 0, "end_cycles": 0} for e in flat["events"]
    ]


@pytest.mark.parametrize("moved", ["outer", "inner"])
def test_unrolling_one_loop_whose_period_is_not_an_integer_keeps_the_others_whole(moved):
    lowered = pulse.compile(_loop_of_loops, unroll_threshold=0).lower(Q[0])
    structured = lowered.export_to_controller(version=2)
    flat = unroll(structured)
    program = copy.deepcopy(structured)
    outer, inner = _loops(program["events"])
    (outer if moved == "outer" else inner)["period_cycles"] += 0.5
    periods = [outer["period_cycles"], inner["period_cycles"]]
    whole = [round(period - 0.25) for period in periods]
    for was, event in zip(flat["events"], unroll(program)["events"], strict=True):
        j = event["iteration"]
        first = was["start_cycles"] - sum(k * p for k, p in zip(j, whole, strict=False))
        start = first + sum(k * p for k, p in zip(j, periods, strict=False))
        # Outside the inner loop, an event after whole periods keeps a whole time.
        assert type(event["start_cycles"]) is (int if moved == "inner" and len(j) == 1 else float)
        assert event["start_cycles"] == start


def test_unrolling_reads_a_loop_it_cannot_add_up_as_it_did():
    structured = (
        pulse.compile(_loop_of_loops, unroll_threshold=0)
        .lower(Q[0])
        .export_to_controller(version=2)
    )

    def edited(edit):
        program = copy.deepcopy(structured)
        for loop in _loops(program["events"]):
            edit(loop)
        return program

    # A period that is no number stops the first event inside the loop.
    with pytest.raises(TypeError, match="unsupported operand"):
        unroll(edited(lambda loop: loop.update(period_cycles=None)))
    with pytest.raises(KeyError, match="period_cycles"):
        unroll(edited(lambda loop: loop.pop("period_cycles")))
    # A loop that runs nothing is not read for its period.
    assert unroll(edited(lambda loop: loop.update(period_cycles=None, body=[])))["events"] == []
    assert (
        unroll(edited(lambda loop: (loop.pop("period_cycles"), loop.update(trips=0))))["events"]
        == []
    )
    # True is one cycle, though no integer by type: added up at each event.
    one = unroll(edited(lambda loop: loop.update(period_cycles=True)))
    assert one["events"] == unroll(edited(lambda loop: loop.update(period_cycles=1)))["events"]
