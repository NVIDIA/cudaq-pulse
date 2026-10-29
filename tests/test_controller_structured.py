# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The structured export format (/2) and limit checks that keep loops rolled."""

from __future__ import annotations

import json
import math

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import backends
from cudaq_pulse.classical import Link
from cudaq_pulse.compiler import MachineModel
from cudaq_pulse.controller import FORMAT, ControllerLimitError, ControllerLimits, check, unroll
from cudaq_pulse.targets import Qubit, TargetDescription

_TARGET = TargetDescription(
    name="structured",
    clock_ghz=2.0,
    qubits={i: Qubit(i, 5.0e9 + 1.0e7 * i, -2.0e8, 50.0, 30.0) for i in range(4)},
)
_DEV = pulse.device({"target": _TARGET})
Q = _DEV.qudit_handles


@pulse.classical(max_latency_ns=200, link="ctl")
def _decide(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]


_MACHINE = MachineModel(links={"ctl": Link(fixed_ns=4, feedback_ns=8)})


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
    "first_differs": (_first_differs, (Q[0],), {}, {}),
    "arithmetic": (_arithmetic, (Q[0], Q[1]), {}, {}),
    "samples": (_arithmetic, (Q[0], Q[1]), {}, {"sample_rate_ghz": 4.0}),
    "calls": (_calls, (Q[0],), {"machine": _MACHINE}, {}),
}


def _lower(name):
    kernel, args, options, export = _CASES[name]
    return pulse.compile(kernel, **options).lower(*args), export


def _canonical(program):
    """Branches renumbered in the order they first appear."""
    ids: dict[int, int] = {}
    events = []
    for event in program["events"]:
        conditions = [
            {**c, "branch": ids.setdefault(c["branch"], len(ids))} for c in event["conditions"]
        ]
        events.append({**event, "conditions": conditions})
    return {**program, "events": events}


_RUN_TIME_KEYS = {
    "tone",
    "frame_updates",
    "phase_offset_rad",
    "length",
    "min_cycles",
    "max_cycles",
    "operands",
}


def _without_run_time(program):
    """*program* without the values known only at run time, which format /2
    does not have."""

    def node(tree):
        if not isinstance(tree, dict):
            return tree
        out = {k: v for k, v in tree.items() if k != "runtime"}
        if "of" in out:
            out["of"] = [node(child) for child in out["of"]]
        return out

    events = []
    for event in program["events"]:
        event = {k: v for k, v in event.items() if k not in _RUN_TIME_KEYS}
        if "waveform" in event:
            event["waveform"] = node(event["waveform"])
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
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _differences(x, y, f"{path}[{i}]")]
    return [] if a == b else [(path, a, b)]


def _loops(events):
    for event in events:
        if event["kind"] == "loop":
            yield event
            yield from _loops(event["body"])


@pytest.mark.parametrize("name", sorted(_CASES))
def test_unrolling_the_structured_export_gives_the_flat_one(name):
    lowered, export = _lower(name)
    flat = lowered.export_controller(**export)
    structured = lowered.export_controller(**export, version=2)
    assert structured["format"] == "cudaq-pulse.controller/2"
    assert structured["programs"] == {} and structured["bindings"] == {}
    expected = _canonical(_without_run_time(flat))
    assert _differences(expected, _canonical(unroll(structured))) == []
    assert len(json.dumps(structured)) < len(json.dumps(flat))


def test_unrolling_a_flat_export_returns_it():
    lowered, _ = _lower("branches")
    flat = lowered.export_controller()
    assert unroll(flat) == flat and flat["format"] == FORMAT


def test_a_repeating_loop_stays_one_loop_whatever_its_trip_count():
    sizes = {}
    for n in (8, 4000):
        program = pulse.compile(_rounds(n)).lower(Q[0], Q[1]).export_controller(version=2)
        (loop,) = [e for e in program["events"] if e["kind"] == "loop"]
        assert loop["trips"] + loop.get("first_iteration", 0) <= n
        assert all(event["iteration"] == [None] for event in loop["body"])
        sizes[n] = len(json.dumps(program))
    # Only the numbers grow: times and the trip count take more digits.
    assert sizes[4000] < sizes[8] + 64, sizes


def test_a_loop_body_carries_its_period_and_phase_steps():
    program = _lower("detuned")[0].export_controller(version=2)
    flat = unroll(program)
    outer = [e for e in _loops(program["events"])][0]
    assert outer["period_cycles"] > 0
    played = [e for e in flat["events"] if e["kind"] == "drive"]
    assert len(played) == 16
    in_loops = [e["tones"] for loop in _loops(program["events"]) for e in _played(loop["body"])]
    assert in_loops and all(len(t) == 1 and "phase_step_rad" in t[0] for t in in_loops)
    outside = [e["tones"] for e in program["events"] if e["kind"] == "drive"]
    assert all("phase_step_rad" not in t[0] for t in outside)


def _played(events):
    for event in events:
        if event["kind"] == "loop":
            yield from _played(event["body"])
        elif event["kind"] in ("drive", "readout"):
            yield event


def test_a_loop_whose_frequency_changes_every_iteration_is_listed_per_iteration():
    program = _lower("chirp")[0].export_controller(version=2)
    assert not list(_loops(program["events"]))
    frequencies = [e["tones"][0]["frequency_hz"] for e in program["events"]]
    assert len(set(frequencies)) == 6


def test_a_first_iteration_that_differs_is_exported_before_its_loop():
    program = _lower("first_differs")[0].export_controller(version=2)
    kinds = [e["kind"] for e in program["events"]]
    assert kinds[-1] == "loop" and "drive" in kinds[:-1]
    assert program["events"][-1]["first_iteration"] == 1


def test_waveforms_are_listed_once_and_lines_list_their_qudits():
    lowered, _ = _lower("unrolled")
    program = lowered.export_controller(version=2)
    played = list(_played(program["events"]))
    # gaussian, square(40), drag and square(8), each listed once.
    assert len(program["waveforms"]) == 4 < len(played)
    assert {e["waveform"] for e in played} == {0, 1, 2, 3}
    flat_lines = lowered.export_controller()["lines"]
    assert [line["qudits"] for line in program["lines"]] == [[line["qudit"]] for line in flat_lines]
    assert all("index" not in e and "qudit" not in e and "end_cycles" not in e for e in played)


def test_the_structured_format_is_refused_by_an_unknown_version():
    lowered, _ = _lower("branches")
    with pytest.raises(ValueError, match="version must be 1 or 2"):
        lowered.export_controller(version=3)
    with pytest.raises(ValueError, match="cannot unroll"):
        unroll({"format": "cudaq-pulse.controller/9"})


# ── Limit checks on rolled loops ────────────────────────────────────────


_LIMITS = [
    ControllerLimits(issue_cycles=3, min_wait_cycles=5),
    ControllerLimits(tones_per_line=1),
    ControllerLimits(sample_rate_ghz=2 * _TARGET.clock_ghz, waveform_samples=60),
    ControllerLimits(acquirers={0: 0, 1: 0}, acquirer_hold_cycles=500),
    ControllerLimits(feedback_latency_cycles=50),
    ControllerLimits(max_events=7),
]


@pytest.mark.parametrize(
    "limits", _LIMITS, ids=lambda limits: str(sorted(limits.as_dict().items()))
)
@pytest.mark.parametrize("n", [3, 40])
def test_a_rolled_loop_is_checked_like_the_same_loop_unrolled(limits, n):
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

    limits = ControllerLimits(issue_cycles=3, min_wait_cycles=6, max_events=10)
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
    machine = MachineModel(controller=ControllerLimits(issue_cycles=3, min_wait_cycles=6))
    spaced = check(unrolled, machine.controller)
    assert len(spaced) == 1000
    with pytest.raises(ControllerLimitError) as error:
        pulse.compile(pairs, machine=machine, unroll_threshold=0).lower(Q[0])
    assert list(error.value.violations) == list(spaced)


def test_a_backend_reading_only_the_structured_format_is_accepted():
    class Structured:
        name = "structured"
        formats = ("cudaq-pulse.controller/2",)

        def machine_model(self):
            return MachineModel()

        def emit(self, program):
            return program

        def run(self, artifact, *, shots=1):
            return [artifact] * shots

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
