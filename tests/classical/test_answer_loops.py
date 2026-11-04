# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Scheduling loops that branch on a classical call's answer.

A round that reads out, submits the outcome and branches on the answer
overlaps the next round's readout with the call, so its iterations repeat
only every few of them. The scheduler runs that many in each iteration of
the loop and keeps it rolled, so lowering costs the same at any trip count;
loops of 10**6 iterations lower in a child process with a CPU, wall and
memory limit. What it schedules is what scheduling every iteration does:
the same kernel with its loops unrolled before scheduling, event by event.
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
from typing import Any

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.classical import Link
from cudaq_pulse.compiler import MachineModel

from tests._handles import Q

POLICIES = ("asap", "alap", "rcp", "alap_rcp")


@pulse.classical(max_latency_ns=400)
def decide(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]


@pulse.classical(max_latency_ns=150, submit="lazy")
def decide_lazy(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]


@pulse.classical(max_latency_ns=1000, link=1)
def decide_slow(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]


@pulse.classical(max_latency_ns=600, latency_mode="timeout", timeout_ns=250)
def decide_timed(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]


@pulse.classical(max_latency_ns=300, latency_mode="blocking")
def decide_blocking(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]


@pulse.classical(max_latency_ns=80)
def settle(bits: pulse.Bits) -> float:
    return 20.0 * bits[0]


@pulse.classical(max_latency_ns=250)
def tune(bits: pulse.Bits) -> tuple[float, float]:
    return 0.1 * bits[0], 1.0e5 * bits[0]


_CALLS = {
    "decide": decide,
    "decide_lazy": decide_lazy,
    "decide_slow": decide_slow,
    "decide_timed": decide_timed,
}

# Calls whose state their values do not carry: a blocking answer holds back
# its lines until it arrives, as a wait of an answer's length does.
_STATEFUL = {**_CALLS, "decide_blocking": decide_blocking}


def _kernels(n: int) -> dict[str, Any]:
    @pulse.kernel
    def round_(q, r, t):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(100, 0.05))
            fix = decide.submit([m]).result()
            sync(d, ro)
            if fix[0]:
                drive(d, square(20, 0.3))

    @pulse.kernel
    def with_else(q, r, t):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(60, 0.05))
            sync(d, ro)
            if decide.submit([m]).result()[0]:
                drive(d, square(20, 0.3))
            else:
                wait(d, 7)
                drive(d, square(4, 0.1))

    @pulse.kernel
    def two_qudits(q, r, t):
        a = get_drive_line(q)
        b = get_drive_line(r)
        p = get_readout_line(q)
        s = get_readout_line(r)
        for _ in range(n):
            m0 = readout(p, square(80, 0.05))
            m1 = readout(s, square(50, 0.05))
            f0 = decide.submit([m0]).result()
            f1 = decide_lazy.submit([m1]).result()
            sync(a, p)
            sync(b, s)
            if f0[0]:
                drive(a, square(16, 0.1))
            if f1[0]:
                drive(b, square(30, 0.1))

    @pulse.kernel
    def slow_link(q, r, t):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(40, 0.05))
            sync(d, ro)
            if decide_slow.submit([m]).result()[0]:
                drive(d, square(10, 0.3))
            wait(ro, 5)

    @pulse.kernel
    def timed_out(q, r, t):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(40, 0.05))
            sync(d, ro)
            ok, fix = decide_timed.submit([m]).result()
            if fix[0]:
                drive(d, square(12, 0.3))

    @pulse.kernel
    def answer_and_outcome(q, r, t):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(70, 0.05))
            if m:
                drive(d, square(8, 0.1))
            if decide.submit([m]).result()[0]:
                drive(d, square(20, 0.3))

    @pulse.kernel
    def nested(q, r, t):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(n):
            for _ in range(2):
                m = readout(ro, square(30, 0.05))
                sync(d, ro)
                if decide.submit([m]).result()[0]:
                    drive(d, square(10, 0.3))
            drive(d, square(5, 0.1))

    @pulse.kernel
    def before_and_after(q, r, t):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        drive(d, square(300, 0.1))
        for _ in range(n):
            m = readout(ro, square(100, 0.05))
            sync(d, ro)
            if decide.submit([m]).result()[0]:
                drive(d, square(20, 0.3))
        sync(d, ro)
        drive(d, square(10, 0.1))

    @pulse.kernel
    def unread_answer(q, r, t):
        b = get_drive_line(r)
        c = get_drive_line(t)
        p = get_readout_line(q)
        for _ in range(n):
            m = readout(p, square(61, 0.05))
            f = decide_lazy.submit([m]).result()
            decide_slow.submit([m]).result()
            drive(b, square(52, 0.1))
            if f[0]:
                drive(c, square(32, 0.2))

    return {
        "round": round_,
        "with_else": with_else,
        "two_qudits": two_qudits,
        "slow_link": slow_link,
        "timed_out": timed_out,
        "answer_and_outcome": answer_and_outcome,
        "nested": nested,
        "before_and_after": before_and_after,
        "unread_answer": unread_answer,
    }


NAMES = list(_kernels(1))


def _saturating(n: int) -> Any:
    """Four requests an iteration, each in flight for many: from about the
    16th iteration on the link's 64 slots hold the requests back, further
    behind the readouts every iteration, so the loop never repeats."""

    @pulse.kernel
    def inner_calls(q, r, t):
        b = get_drive_line(r)
        c = get_drive_line(t)
        p = get_readout_line(q)
        s = get_readout_line(r)
        for _ in range(n):
            m = readout(s, square(28, 0.05))
            ok, fix = decide_timed.submit([m]).result()
            for _ in range(3):
                drive(c, square(45, 0.1))
                if decide.submit([m]).result()[0]:
                    sync(b, p)
            if fix[0]:
                drive(b, square(23, 0.2))

    return inner_calls


_DRIVES = ["a", "b", "c"]
_READOUTS = ["p", "s"]


def _statement(
    rng: random.Random,
    depth: int,
    scope: dict[str, list[str]],
    indent: int,
    stateful: bool = False,
    valued: bool = False,
) -> list[str]:
    pad = "    " * indent
    calls = _STATEFUL if stateful else _CALLS
    roll = rng.random()
    measured, answered = scope["measured"], scope["answered"]
    if valued and roll < 0.25 and measured:
        # Answers as values: amplitudes, phase and frequency shifts.
        tuned = scope.setdefault("tuned", [])
        if not tuned or rng.random() < 0.4:
            # Named by where it is taken, so an arm's never shadows one past it.
            name = f"t{len(tuned)}_{indent}"
            tuned.append(name)
            return [f"{pad}{name}, {name}f = tune.submit([{rng.choice(measured)}]).result()"]
        name, line = rng.choice(tuned), rng.choice(_DRIVES)
        return [
            rng.choice(
                (
                    f"{pad}drive({line}, square({rng.randint(4, 40)}, {name}))",
                    f"{pad}shift_phase({line}.tone, {name})",
                    f"{pad}shift_frequency({line}.tone, {name}f)",
                )
            )
        ]
    if roll < 0.2:
        return [f"{pad}drive({rng.choice(_DRIVES)}, square({rng.randint(1, 60)}, 0.1))"]
    if roll < 0.3:
        if stateful and depth == 0 and measured and rng.random() < 0.5:
            length = f"settle.submit([{rng.choice(measured)}]).result()"
            return [f"{pad}wait({rng.choice(_DRIVES)}, {length}, bounds=(8, 60))"]
        return [f"{pad}wait({rng.choice(_DRIVES + _READOUTS)}, {rng.randint(1, 30)})"]
    if roll < 0.45:
        name = f"m{len(measured)}"
        measured.append(name)
        width = rng.randint(10, 120)
        return [f"{pad}{name} = readout({rng.choice(_READOUTS)}, square({width}, 0.05))"]
    if roll < 0.6 and measured:
        call = rng.choice(list(calls))
        request = f"{call}.submit([{rng.choice(measured)}]).result()"
        if depth > 0:
            # The unrolled program cannot take an answer an arm assigns past
            # it, so an arm tests one it does not keep.
            bit = "[1][0]" if call == "decide_timed" else "[0]"
            inner = {"measured": list(measured), "answered": list(answered)}
            lines = [f"{pad}if {request}{bit}:"]
            lines += _statement(rng, 2, inner, indent + 1, stateful, valued)
            return lines
        name = f"f{len(answered)}"
        answered.append(name)
        result = "ok, " + name if call == "decide_timed" else name
        return [f"{pad}{result} = {request}"]
    if roll < 0.8 and (answered or measured) and depth < 2:
        if answered and (not measured or rng.random() < 0.75):
            condition = f"{rng.choice(answered)}[0]"
        else:
            condition = rng.choice(measured)
        inner = {"measured": list(measured), "answered": list(answered)}
        lines = [f"{pad}if {condition}:"]
        for _ in range(rng.randint(1, 2)):
            lines += _statement(rng, depth + 1, inner, indent + 1, stateful, valued)
        if rng.random() < 0.4:
            lines.append(f"{pad}else:")
            inner = {"measured": list(measured), "answered": list(answered)}
            for _ in range(rng.randint(1, 2)):
                lines += _statement(rng, depth + 1, inner, indent + 1, stateful, valued)
        return lines
    if roll < 0.95:
        first, second = rng.sample(_DRIVES + _READOUTS, 2)
        return [f"{pad}sync({first}, {second})"]
    return [f"{pad}drive({rng.choice(_DRIVES)}, gaussian({rng.randint(4, 40)}, 0.1, 2.0))"]


def _random_source(seed: int, trips: int, stateful: bool = False, valued: bool = False) -> str:
    """A random loop of *trips* iterations that reads out, calls and branches
    on answers, between a random prologue and epilogue; some have a loop of
    their own in the body. *stateful* adds blocking calls and waits of an
    answer's length, and answers an inner loop takes and need not test;
    *valued* answers that set amplitudes, phases and frequencies."""
    rng = random.Random(seed)
    calls = _STATEFUL if stateful else _CALLS
    lines = [
        "def k(q, r, t):",
        "    a = get_drive_line(q)",
        "    b = get_drive_line(r)",
        "    c = get_drive_line(t)",
        "    p = get_readout_line(q)",
        "    s = get_readout_line(r)",
    ]
    outside: dict[str, list[str]] = {"measured": [], "answered": []}
    for _ in range(rng.randint(0, 2)):
        lines += _statement(rng, 2, outside, 1)
    lines.append(f"    for _ in range({trips}):")
    # Each body measures, calls and tests the answer at least once.
    ro, line = rng.choice(_READOUTS), rng.choice(_DRIVES)
    call = rng.choice(list(calls))
    result = "ok, f0" if call == "decide_timed" else "f0"
    lines += [
        f"        m0 = readout({ro}, square({rng.randint(10, 120)}, 0.05))",
        f"        {result} = {call}.submit([m0]).result()",
    ]
    if rng.random() < 0.6:
        lines.append(f"        sync({line}, {ro})")
    test = ["        if f0[0]:", f"            drive({line}, square({rng.randint(4, 40)}, 0.2))"]
    scope = {"measured": ["m0"], "answered": ["f0"]}
    if valued:
        scope["tuned"] = ["t0"]
        lines.append("        t0, t0f = tune.submit([m0]).result()")
        test += _statement(rng, 1, {**scope, "tuned": ["t0"]}, 2, stateful, valued)
    extra: list[str] = []
    for _ in range(rng.randint(0, 4)):
        extra += _statement(rng, 0, scope, 2, stateful, valued)
    if rng.random() < 0.15:
        inner = {"measured": list(scope["measured"]), "answered": list(scope["answered"])}
        extra.append(f"        for _ in range({rng.randint(2, 3)}):")
        for _ in range(rng.randint(1, 3)):
            extra += _statement(rng, 0 if stateful else 1, inner, 3, stateful, valued)
    lines += test + extra if rng.random() < 0.5 else extra + test
    for _ in range(rng.randint(0, 2)):
        lines += _statement(rng, 2, {"measured": [], "answered": []}, 1)
    return "\n".join(lines) + "\n"


def _random_kernel(seed: int, trips: int, stateful: bool = False, valued: bool = False) -> Any:
    namespace: dict[str, Any] = dict(_STATEFUL, settle=settle, tune=tune)
    source = _random_source(seed, trips, stateful, valued)
    exec(compile(source, f"<answers {seed}>", "exec"), namespace)  # noqa: S102 - generated kernels
    return pulse.kernel(namespace["k"])


def _machines() -> dict[str, dict[str, Any]]:
    return {
        "default": {},
        "two_slots": {
            "machine": MachineModel(
                links={0: Link(fixed_ns=3, max_in_flight=2), 1: Link(feedback_ns=20)},
                max_concurrent_drives=2,
                readout_latency_cycles=9,
            )
        },
    }


def _events(kernel: Any, schedule: str, unroll_threshold: int, options: dict[str, Any]) -> Any:
    """The duration and events of *kernel*'s schedule, or the error that
    stops it, but for the op it notes (two slots can be too few under alap
    whether or not the loop is unrolled)."""
    program = pulse.compile(
        kernel, schedule=schedule, unroll_threshold=unroll_threshold, passes=(), **options
    )
    try:
        lowered = program.lower(Q[0], Q[1], Q[2])
    except RuntimeError as error:
        return str(error).splitlines()[:2]
    found = sorted((e.kind, e.line, e.start_cycles, e.duration_cycles) for e in lowered.schedule)
    return lowered.duration, found


def _same_as_unrolled(kernel: Any, schedule: str, options: dict[str, Any]) -> bool:
    return _events(kernel, schedule, 0, options) == _events(kernel, schedule, 10**6, options)


@pytest.mark.parametrize("schedule", POLICIES)
def test_a_loop_branching_on_answers_plays_every_event_as_unrolled(schedule):
    differ = [
        (name, trips, machine)
        for machine, options in _machines().items()
        for trips in (1, 2, 3, 5, 8, 13, 20, 41)
        for name, kernel in {**_kernels(trips), "saturating": _saturating(trips)}.items()
        if not _same_as_unrolled(kernel, schedule, options)
    ]
    assert differ == []


@pytest.mark.parametrize("schedule", POLICIES)
def test_random_loops_branching_on_answers_play_every_event_as_unrolled(schedule):
    differ = [
        (seed, trips, machine)
        for machine, options in _machines().items()
        for seed in range(40)
        for trips in (3, 14, 37)
        if not _same_as_unrolled(_random_kernel(seed, trips), schedule, options)
    ]
    assert differ == []


@pytest.mark.parametrize("schedule", POLICIES)
def test_random_loops_with_calls_that_keep_state_play_every_event_as_unrolled(schedule):
    differ = [
        (seed, trips, machine)
        for machine, options in _machines().items()
        for seed in range(40)
        for trips in (3, 14, 37)
        if not _same_as_unrolled(_random_kernel(seed, trips, True), schedule, options)
    ]
    assert differ == []


def _size(kernel: Any, schedule: str) -> int:
    lowered = pulse.compile(kernel, schedule=schedule, unroll_threshold=0).lower(Q[0], Q[1], Q[2])
    return lowered.mlir("pulse").count("start_cycles")


# Past its irregular first and last iterations, each loop is run a few
# iterations at a time and rolled, so its program is the same at more trips
# (840 more, so as many are left over past the last whole run). The nested
# loop's lines drift apart for its first 158 iterations, which are peeled.
@pytest.mark.parametrize("schedule", POLICIES)
def test_a_loop_branching_on_answers_lowers_to_the_same_program_at_more_trips(schedule):
    grow = [
        name
        for name in NAMES
        if _size(_kernels(941)[name], schedule) != _size(_kernels(1781)[name], schedule)
    ]
    assert grow == []


def _settled(n: int) -> dict[str, Any]:
    @pulse.kernel
    def early(q, r):
        line = get_drive_line(q)
        other = get_drive_line(r)
        probe = get_readout_line(q)
        m = readout(probe, square(40, 0.05))
        sync(probe, line)
        for _ in range(n):
            if m:
                drive(line, square(8, 0.1))
            wait(line, 3)
            drive(other, square(5, 0.1))

    @pulse.kernel
    def two_lines(q, r):
        a = get_drive_line(q)
        b = get_drive_line(r)
        for _ in range(n):
            drive(a, square(5, 0.1))
            drive(b, gaussian(8, 0.1, 2.0))

    @pulse.kernel
    def feedback(q, r):
        line = get_drive_line(q)
        probe = get_readout_line(r)
        for _ in range(n):
            m = readout(probe, square(40, 0.05))
            if m:
                drive(line, square(8, 0.1))
            else:
                drive(line, square(4, 0.2))

    return {"early": early, "two_lines": two_lines, "feedback": feedback}


# A loop that repeats every iteration, line by line or once its lanes
# settle, is extrapolated as it was before loops could be regrouped, to a
# program of the same size.
SETTLED = {
    ("early", "asap"): 4,
    ("early", "alap"): 4,
    ("early", "rcp"): 13,
    ("early", "alap_rcp"): 22,
    ("two_lines", "rcp"): 18,
    ("two_lines", "alap_rcp"): 14,
}


@pytest.mark.parametrize("trips", (64, 300))
def test_a_loop_that_repeats_every_iteration_is_not_regrouped(trips):
    kernels = _settled(trips)
    sizes = {
        (name, schedule): pulse.compile(kernels[name], schedule=schedule, unroll_threshold=0)
        .lower(Q[0], Q[1])
        .mlir("pulse")
        .count("start_cycles")
        for name, schedule in SETTLED
    }
    assert sizes == SETTLED


def _answered_feedback(n: int) -> Any:
    @pulse.kernel
    def answered(q, r):
        line = get_drive_line(q)
        probe = get_readout_line(r)
        for _ in range(n):
            m = readout(probe, square(40, 0.05))
            if decide.submit([m]).result()[0]:
                drive(line, square(8, 0.1))
            else:
                drive(line, square(4, 0.2))

    return answered


@pytest.mark.parametrize("schedule", ("rcp", "alap_rcp"))
def test_lanes_regroup_a_loop_of_calls_only_past_their_tries(schedule):
    # The readouts take four lanes in turn, so the lanes repeat only every
    # four iterations. Past the tries each lane has to repeat on its own, a
    # long loop of calls is run four iterations at a time (17 trips leave
    # too few); one without calls is left as it was, for the controller
    # check to summarize.
    def size(kernel: Any) -> int:
        lowered = pulse.compile(kernel, schedule=schedule, unroll_threshold=0).lower(Q[0], Q[1])
        return lowered.mlir("pulse").count("start_cycles")

    trips = (17, 64, 300)
    assert {n: size(_answered_feedback(n)) for n in trips} == {17: 5, 64: 20, 300: 20}
    assert {n: size(_settled(n)["feedback"]) for n in trips} == {17: 3, 64: 3, 300: 3}


@pytest.mark.parametrize("schedule", POLICIES)
def test_a_round_of_feedback_is_rolled_two_iterations_at_a_time(schedule):
    # The next readout overlaps the call, so each iteration repeats the one
    # two before it, 921 cycles later.
    lowered = pulse.compile(_kernels(1000)["round"], schedule=schedule, unroll_threshold=0)
    text = lowered.lower(Q[0], Q[1], Q[2]).mlir("pulse")
    assert text.count("scf.for") == 1
    assert "period_cycles = array<i64: 921>" in text
    assert text.count("start_cycles") < 40


def test_an_unrolled_loop_of_calls_verifies_and_schedules_in_linear_time():
    # A branch's condition is not computed from the requests before its own
    # on the link, so neither verifying nor scheduling the unrolled loop
    # walks them all. Four times the trips take about four times as long
    # natively (sixteen if quadratic); tracing is linear either way.
    timings = {}
    for n in (1000, 4000):
        kernel = _kernels(n)["round"]
        best = float("inf")
        for _ in range(3):
            program = pulse.compile(kernel, schedule="alap", unroll_threshold=10**6)
            metrics = program.lower(Q[0], Q[1], Q[2]).metrics
            best = min(best, (metrics.passes_ms + metrics.schedule_ms) / 1000)
        timings[n] = best
    assert timings[4000] < 7 * timings[1000] + 0.02, timings


@pytest.mark.parametrize("schedule", ("asap", "alap", "rcp"))
def test_a_loop_that_never_repeats_schedules_in_linear_time(schedule):
    # Its iterations are peeled around the copies of its inner loop, all of
    # them in each round, so the rounds do not grow with the trips.
    timings = {}
    for n in (250, 1000):
        program = pulse.compile(_saturating(n), schedule=schedule, unroll_threshold=0)
        timings[n] = program.lower(Q[0], Q[1], Q[2]).metrics.schedule_ms / 1000
    assert timings[1000] < 7 * timings[250] + 0.02, timings


@pytest.mark.parametrize("schedule", POLICIES)
def test_an_answer_is_never_taken_before_it_is_due_when_lanes_run_out_backward(schedule):
    # Under alap_rcp, handing out drive lanes in reverse moves the first
    # round's last drive, and the waits deciding the syncs before it, ahead
    # of the eager requests' answers: that schedule is left for the forward
    # one, as one needing time before zero is.
    for trips in (1, 3, 5, 8, 41):
        kernel = _saturating(trips)
        events = _events(kernel, schedule, 0, {})
        assert isinstance(events, tuple), (trips, events)
        assert events == _events(kernel, schedule, 10**6, {}), trips


def _blocking_after_a_readout(n: int, rolled: bool) -> Any:
    """A readout before a loop whose body measures into the same name, so the
    loop carries a measurement it never reads, then calls blocking: as
    written (*rolled*) or with the body written out *n* times."""
    if rolled:
        body = "    for _ in range({n}):\n{body}"
        indent = "        "
    else:
        body = "{body}" * n
        indent = "    "
    lines = [
        "m0 = readout(p, square(77, 0.05))",
        "f0 = decide_blocking.submit([m0]).result()",
        "sync(c, p)",
        "if f0[0]:",
        "    drive(a, gaussian(33, 0.1, 2.0))",
        "drive(a, square(7, 0.1))",
        "drive(b, square(57, 0.1))",
    ]
    source = "\n".join(
        [
            "def k(q, r, t):",
            "    a = get_drive_line(q)",
            "    b = get_drive_line(r)",
            "    c = get_drive_line(t)",
            "    p = get_readout_line(q)",
            "    s = get_readout_line(r)",
            "    m0 = readout(s, square(70, 0.05))",
            "    wait(c, 3)",
            body.format(n=n, body="".join(f"{indent}{line}\n" for line in lines)),
        ]
    )
    namespace: dict[str, Any] = dict(_STATEFUL)
    exec(compile(source, "<blocking after a readout>", "exec"), namespace)  # noqa: S102 - generated kernels
    return pulse.kernel(namespace["k"])


@pytest.mark.parametrize("schedule", POLICIES)
def test_a_measurement_a_loop_carries_unread_does_not_hold_back_its_line(schedule):
    # The readout before the loop is on a line the blocking call does not
    # wait for: what the loop carries in its place is never read, so it
    # plays where the program written out plays it.
    for trips in (1, 2, 5):
        for machine, options in _machines().items():
            written_out = _events(_blocking_after_a_readout(trips, False), schedule, 0, options)
            rolled = _blocking_after_a_readout(trips, True)
            assert _events(rolled, schedule, 0, options) == written_out, (trips, machine)
            assert _events(rolled, schedule, 10**6, options) == written_out, (trips, machine)


@pytest.mark.parametrize("schedule", POLICIES)
def test_random_loops_carrying_unread_measurements_past_blocking_calls_play_as_unrolled(schedule):
    differ = [
        (seed, trips, machine)
        for machine, options in _machines().items()
        for seed in (3058, 3070)
        for trips in (1, 2, 7)
        if not _same_as_unrolled(_random_kernel(seed, trips, True), schedule, options)
    ]
    assert differ == []


# The child's limits, far past what lowering every kernel below at 10**6
# trips under every policy takes: under 20 ms each and about 50 MB in all.
CPU_S = 30
RSS_MB = 150

_CHILD = """
import json, os, resource, signal, sys, threading, time
signal.alarm({wall})
resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
_scale = 1 if sys.platform == "darwin" else 1024
def _watch():
    while True:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _scale > {mb} << 20:
            os._exit(3)
        time.sleep(0.005)
threading.Thread(target=_watch, daemon=True).start()

import cudaq_pulse as pulse
from tests._handles import Q
from {module} import {factory} as factory

out = {{}}
kernels = factory({trips})
for name in kernels:
    for schedule in {policies!r}:
        started = time.process_time()
        program = pulse.compile(kernels[name], schedule=schedule, unroll_threshold=0)
        lowered = program.lower(Q[0], Q[1], Q[2])
        spent = time.process_time() - started
        out[f"{{name}}/{{schedule}}"] = [str(lowered.duration.exact_ns), spent]
print(json.dumps(out))
"""


def _in_a_child(
    trips: int, factory: str, module: str, policies: tuple[str, ...] = POLICIES
) -> dict[str, list[Any]]:
    """The duration of every kernel *factory* (in *module*) makes of *trips*
    iterations under *policies*, and the CPU time its lowering took, in a
    child process that is stopped past CPU_S seconds or RSS_MB MB."""
    code = _CHILD.format(
        wall=2 * CPU_S,
        cpu=CPU_S,
        mb=RSS_MB,
        trips=trips,
        factory=factory,
        module=module,
        policies=tuple(policies),
    )
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=root,
        env=env,
        timeout=4 * CPU_S,
    )
    assert out.returncode == 0, f"the child exited {out.returncode}: {out.stderr[-2000:]}"
    result: dict[str, list[Any]] = json.loads(out.stdout.splitlines()[-1])
    return result


def _lowers_as_extrapolated(
    factory: Any, many: int = 10**6, policies: tuple[str, ...] = POLICIES
) -> None:
    """Lowering *factory*'s kernels at *many* trips under *policies* in a
    child takes well under a second each, to the duration two shorter loops
    extrapolate to."""
    from fractions import Fraction

    found = _in_a_child(many, factory.__name__, factory.__module__, policies)
    small: dict[tuple[str, int], Fraction] = {}
    # As many trips past the peeled ones as *many* is, modulo every run of
    # iterations a loop repeats after (840 is a multiple of each).
    fewer = 1240 + (many - 1240) % 840
    more = fewer + 840
    for trips in (fewer, more):
        kernels = factory(trips)
        for kernel in kernels:
            for schedule in policies:
                program = pulse.compile(kernels[kernel], schedule=schedule, unroll_threshold=0)
                lowered = program.lower(Q[0], Q[1], Q[2])
                small[f"{kernel}/{schedule}", trips] = lowered.duration.exact_ns
    assert len(found) == len(factory(1)) * len(policies)
    for key, (duration, spent) in found.items():
        step = small[key, more] - small[key, fewer]
        expected = small[key, fewer] + (many - fewer) // (more - fewer) * step
        assert Fraction(duration) == expected, key
        assert spent < 0.25, key


def test_a_loop_of_10_6_iterations_branching_on_answers_lowers_in_bounded_time_and_memory():
    _lowers_as_extrapolated(_kernels)


def _valued(n: int) -> dict[str, Any]:
    """Loops that use answers as values: what they set moves with its own
    lines, not with the readout the call asks about."""

    @pulse.kernel
    def amplitude(q, r, t):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(100, 0.05))
            amp, _ = tune.submit([m]).result()
            drive(d, square(20, amp))

    @pulse.kernel
    def framed(q, r, t):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(100, 0.05))
            amp, detuning = tune.submit([m]).result()
            shift_frequency(d.tone, detuning)
            drive(d, square(20, amp))
            shift_phase(d.tone, amp)

    @pulse.kernel
    def with_a_branch(q, r, t):
        d = get_drive_line(q)
        e = get_drive_line(t)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(61, 0.05))
            fix = decide.submit([m]).result()
            amp, _ = tune.submit([m]).result()
            drive(e, square(52, amp))
            if fix[0]:
                drive(d, square(32, 0.2))

    @pulse.kernel
    def unplayed(q, r, t):
        d = get_drive_line(q)
        e = get_drive_line(r)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(40, 0.05))
            amp, _ = tune.submit([m]).result()
            shift_phase(e.tone, amp)
            drive(d, square(30, 0.1))

    return {
        "amplitude": amplitude,
        "framed": framed,
        "with_a_branch": with_a_branch,
        "unplayed": unplayed,
    }


def _valued_apart(n: int) -> dict[str, Any]:
    """Answers as values that each iteration's own readout does not pace:
    a drive slower than the readouts, one taking the answer of the
    iteration before, one whose level a branch picks."""

    @pulse.kernel
    def slow_drive(q, r, t):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(20, 0.05))
            amp, _ = tune.submit([m]).result()
            drive(d, square(300, amp))

    @pulse.kernel
    def carried(q, r, t):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        amp = 0.1
        for _ in range(n):
            drive(d, square(300, amp))
            m = readout(ro, square(20, 0.05))
            amp, _ = tune.submit([m]).result()

    @pulse.kernel
    def selected(q, r, t):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(20, 0.05))
            fix = decide.submit([m]).result()
            amp, _ = tune.submit([m]).result()
            level = 0.1
            if fix[0]:
                level = amp
            drive(d, square(300, level))

    return {"slow_drive": slow_drive, "carried": carried, "selected": selected}


@pytest.mark.parametrize("schedule", POLICIES)
def test_a_loop_using_answers_as_values_plays_every_event_as_unrolled(schedule):
    differ = [
        (name, trips, machine)
        for machine, options in _machines().items()
        for trips in (1, 2, 5, 64, 130)
        for name, kernel in {**_valued(trips), **_valued_apart(trips)}.items()
        if not _same_as_unrolled(kernel, schedule, options)
    ]
    assert differ == []


@pytest.mark.parametrize("schedule", POLICIES)
def test_random_loops_using_answers_as_values_play_every_event_as_unrolled(schedule):
    differ = [
        (seed, trips, machine)
        for machine, options in _machines().items()
        for seed in range(40)
        for trips in (3, 14, 100)
        if not _same_as_unrolled(
            _random_kernel(seed, trips, seed % 4 == 0, True), schedule, options
        )
    ]
    assert differ == []


def test_a_loop_of_10_6_iterations_using_answers_as_values_lowers_in_bounded_time_and_memory():
    # Under alap the drives an answer sets move by their own period, the
    # readouts and requests by theirs; each loop stays rolled.
    _lowers_as_extrapolated(_valued)


def _slack(n: int) -> dict[str, Any]:
    """Loops whose answers' slack changes from one iteration to the next
    under alap: a drive or branch slower than the readouts (its answers
    taken ever later after they are due), and lazy requests sent on a link
    right before an eager one, which bounds them."""

    @pulse.kernel
    def slow_drive(q, r, t):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(20, 0.05))
            amp, _ = tune.submit([m]).result()
            drive(d, square(300, amp))

    @pulse.kernel
    def slow_branch(q, r, t):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(20, 0.05))
            if decide.submit([m]).result()[0]:
                drive(d, square(300, 0.3))

    @pulse.kernel
    def lazy_slow(q, r, t):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(40, 0.05))
            f = decide_lazy.submit([m]).result()
            amp, _ = tune.submit([m]).result()
            shift_phase(d.tone, amp)
            drive(d, square(300, 0.1))
            if f[0]:
                drive(d, square(5, 0.3))

    return {"slow_drive": slow_drive, "slow_branch": slow_branch, "lazy_slow": lazy_slow}


def _linked(n: int) -> dict[str, Any]:
    """Lazy requests sent right before an eager one on their link."""

    @pulse.kernel
    def lazy(q, r, t):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(40, 0.05))
            f = decide_lazy.submit([m]).result()
            amp, _ = tune.submit([m]).result()
            shift_phase(d.tone, amp)
            drive(d, square(30, 0.1))
            if f[0]:
                drive(d, square(5, 0.3))

    @pulse.kernel
    def lazy_fast(q, r, t):
        d = get_drive_line(q)
        e = get_drive_line(t)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(200, 0.05))
            f = decide_lazy.submit([m]).result()
            amp, _ = tune.submit([m]).result()
            drive(e, square(10, amp))
            if f[0]:
                drive(d, square(5, 0.3))

    @pulse.kernel
    def lazy_gap(q, r, t):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        so = get_readout_line(r)
        for _ in range(n):
            m = readout(ro, square(20, 0.05))
            f = decide_lazy.submit([m]).result()
            m2 = readout(so, square(90, 0.05))
            amp, _ = tune.submit([m2]).result()
            drive(d, square(200, amp))
            if f[0]:
                drive(d, square(5, 0.3))

    return {"lazy": lazy, "lazy_fast": lazy_fast, "lazy_gap": lazy_gap}


def _held(n: int) -> dict[str, Any]:
    """Loops where what is decided does hold its condition's readouts back:
    a branch on a measurement, lazy requests on their own or with an eager
    one sent well after them, each deciding a drive slower than the
    readouts."""

    @pulse.kernel
    def measured(q, r, t):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(20, 0.05))
            if m:
                drive(d, square(300, 0.3))

    @pulse.kernel
    def lazy_alone(q, r, t):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(20, 0.05))
            if decide_lazy.submit([m]).result()[0]:
                drive(d, square(300, 0.3))

    @pulse.kernel
    def lazy_valued(q, r, t):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(20, 0.05))
            f = decide_lazy.submit([m]).result()
            amp, _ = tune.submit([m]).result()
            drive(d, square(300, amp))
            if f[0]:
                drive(d, square(5, 0.3))

    @pulse.kernel
    def eager_later(q, r, t):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        so = get_readout_line(r)
        for _ in range(n):
            m = readout(ro, square(20, 0.05))
            f = decide_lazy.submit([m]).result()
            m2 = readout(so, square(150, 0.05))
            amp, _ = tune.submit([m2]).result()
            drive(d, square(400, amp))
            if f[0]:
                drive(d, square(5, 0.3))

    return {
        "measured": measured,
        "lazy_alone": lazy_alone,
        "lazy_valued": lazy_valued,
        "eager_later": eager_later,
    }


@pulse.classical(max_latency_ns=150, submit="lazy")
def tune_lazy(bits: pulse.Bits) -> float:
    return 0.1 * bits[0]


def _bound_late(n: int, tail: int) -> dict[str, Any]:
    """Loops whose readouts are first held back by a line synced to them,
    which ends early before a long drive after the loop, until what their
    measurement or answer decides, a slower drive, catches up a few
    iterations back: a hold that binds only then."""

    def body(decide_with: str) -> str:
        return "\n".join(
            [
                "def k(q, r, t):",
                "    d = get_drive_line(q)",
                "    e = get_drive_line(t)",
                "    ro = get_readout_line(q)",
                f"    for _ in range({n}):",
                "        m = readout(ro, square(20, 0.05))",
                "        sync(ro, e)",
                "        drive(e, square(10, 0.1))",
                *(f"        {line}" for line in decide_with.splitlines()),
                f"    drive(e, square({tail}, 0.1))",
            ]
        )

    shapes = {
        "measured": "if m:\n    drive(d, square(300, 0.3))",
        "lazy": "if decide_lazy.submit([m]).result()[0]:\n    drive(d, square(300, 0.3))",
        "lazy_value": "amp = tune_lazy.submit([m]).result()\ndrive(d, square(300, amp))",
        "linked": (
            "f = decide_lazy.submit([m]).result()\n"
            "amp, unused = tune.submit([m]).result()\n"
            "drive(d, square(300, amp))\n"
            "if f[0]:\n    drive(d, square(5, 0.3))"
        ),
    }
    kernels = {}
    for name, decide_with in shapes.items():
        namespace: dict[str, Any] = dict(_STATEFUL, tune=tune, tune_lazy=tune_lazy)
        exec(compile(body(decide_with), f"<{name}>", "exec"), namespace)  # noqa: S102 - generated kernels
        kernels[name] = pulse.kernel(namespace["k"])
    return kernels


@pytest.mark.parametrize("schedule", POLICIES)
def test_a_hold_that_binds_a_few_iterations_back_plays_every_event_as_unrolled(schedule):
    differ = [
        (name, trips, tail, machine)
        for machine, options in _machines().items()
        for tail in (1000, 3000)
        for trips in (5, 20, 70, 130)
        for name, kernel in _bound_late(trips, tail).items()
        if not _same_as_unrolled(kernel, schedule, options)
    ]
    assert differ == []


@pytest.mark.parametrize("schedule", POLICIES)
def test_loops_whose_answers_slack_changes_play_every_event_as_unrolled(schedule):
    differ = [
        (name, trips, machine)
        for machine, options in _machines().items()
        for trips in (1, 2, 5, 64, 130, 1001)
        for name, kernel in {**_slack(trips), **_linked(trips), **_held(trips)}.items()
        if not _same_as_unrolled(kernel, schedule, options)
    ]
    assert differ == []


def test_a_loop_of_10_6_iterations_whose_answers_slack_changes_lowers_in_bounded_time():
    # Without resources a request keeps its forward time and no wait is
    # placed before its own, so how the slack moves bounds nothing; under
    # alap_rcp, whose lanes may move a wait earlier, these loops are
    # simulated iteration by iteration.
    _lowers_as_extrapolated(_slack, policies=("asap", "alap", "rcp"))


def _linked_tight(n: int) -> dict[str, Any]:
    kernels = _linked(n)
    del kernels["lazy_gap"]
    return kernels


def test_a_loop_of_10_6_iterations_with_lazy_requests_before_eager_ones_lowers_in_bounded_time():
    _lowers_as_extrapolated(_linked_tight)


@pulse.classical(max_latency_ns=100)
def ask(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]


@pulse.classical(max_latency_ns=100, submit="lazy")
def ask_lazy(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]


@pulse.classical(max_latency_ns=60, submit="lazy")
def ask_soon(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]


# Loop bodies sharing one link. A request holds its slot until its answer
# is due, whether or not it is awaited: the requests nobody awaits are still
# in flight when the next loop starts.
_ON_ONE_LINK = {
    "answer": [
        "sync(d, ro)",
        "m = readout(ro, square(40, 0.05))",
        "if ask.submit([m]).result()[0]:",
        "    drive(d, square(16, 0.1))",
    ],
    "unawaited": [
        "m = readout(ro, square(40, 0.05))",
        "ask.submit([m])",
        "drive(d, square(4, 0.1))",
    ],
    "spaced": ["m = readout(ro, square(60, 0.05))", "ask.submit([m])"],
    "nested": [
        "for _ in range(3):",
        "    m = readout(ro, square(60, 0.05))",
        "    ask.submit([m])",
        "drive(d, square(40, 0.1))",
    ],
    "lazy": [
        "m = readout(ro, square(30, 0.05))",
        "if ask_lazy.submit([m]).result()[0]:",
        "    drive(d, square(6, 0.1))",
    ],
    "soon": [
        "m = readout(ro, square(30, 0.05))",
        "if ask_soon.submit([m]).result()[0]:",
        "    drive(d, square(6, 0.1))",
    ],
    "drives": ["drive(d, square(8, 0.1))"],
}


def _one_after_another(n: int, *bodies: str, name: str | None = None) -> Any:
    """A kernel running a loop of *n* trips of each of *bodies* in turn, each
    over a name of its own, or all over *name*."""
    lines = ["def k(q):", "    d = get_drive_line(q)", "    ro = get_readout_line(q)"]
    for index, body in enumerate(bodies):
        lines.append(f"    for {name or f'i{index}'} in range({n}):")
        lines += [f"        {line}" for line in _ON_ONE_LINK[body]]
    namespace: dict[str, Any] = {"ask": ask, "ask_lazy": ask_lazy, "ask_soon": ask_soon}
    exec(compile("\n".join(lines), "<loops>", "exec"), namespace)  # noqa: S102 - generated kernels
    return pulse.kernel(namespace["k"])


def _link_slots(slots: int) -> MachineModel:
    return MachineModel(links={0: Link(fixed_ns=4, max_in_flight=slots)})


def _most_at_once(spans: list[tuple[int, int]]) -> int:
    edges = sorted(edge for start, end in spans for edge in ((start, 1), (end, -1)))
    held = most = 0
    for _, step in edges:  # a slot freed at a time is free for one taken then
        held += step
        most = max(most, held)
    return most


def _most_in_flight(lowered: Any) -> int:
    """The most requests in flight at once in *lowered*'s schedule, each from
    its submission until its answer is due, for calls of one budget."""
    budgets = set(map(int, re.findall(r"classical\.budget_cycles = (\d+)", lowered.mlir())))
    assert len(budgets) == 1
    budget = budgets.pop()
    return _most_at_once(
        [
            (e.start_cycles, e.start_cycles + e.duration_cycles + budget)
            for e in lowered.schedule
            if e.kind == "submit"
        ]
    )


def _most_in_flight_written_out(lowered: Any) -> int:
    """_most_in_flight for a program without loops, each request by its
    own budget."""
    program = lowered.mlir()
    assert "scf.for" not in program
    spans = []
    for line in program.splitlines():
        if "classical.submit" in line:
            found = dict(re.findall(r"(budget_cycles|duration_cycles|start_cycles) = (\d+)", line))
            start = int(found["start_cycles"])
            spans.append(
                (start, start + int(found["duration_cycles"]) + int(found["budget_cycles"]))
            )
    return _most_at_once(spans)


# Under alap a lazy request is placed before the eager requests ahead of it,
# which keep their forward starts: where it took the slot one of those still
# holds, alap keeps the forward schedule.
@pytest.mark.parametrize("schedule", ("alap", "alap_rcp"))
def test_eager_requests_before_lazy_ones_keep_within_the_slots_under_alap(schedule):
    over = []
    for slots in (2, 4, 6):
        for bodies in (("unawaited", "soon"), ("answer", "soon"), ("nested", "soon")):
            lowered = pulse.compile(
                _one_after_another(5, *bodies),
                schedule=schedule,
                unroll_threshold=10**6,
                machine=_link_slots(slots),
                passes=(),
            ).lower(Q[0])
            if _most_in_flight_written_out(lowered) > slots:
                over.append((bodies, slots))
    assert over == []


_LOOPS_ON_ONE_LINK = (
    ("answer", "unawaited"),
    ("drives", "answer", "unawaited"),
    ("answer", "spaced"),
    ("answer", "nested"),
    ("lazy", "unawaited"),
    ("lazy", "spaced"),
    ("unawaited", "lazy"),
    ("answer", "lazy"),
)


# A loop whose rest the scheduler extrapolates leaves the link's slots as
# its last iteration would: the next loop's first requests wait for the ones
# still in flight, as written out. Loops over one name too, as `for _ in`
# loops one after another are written.
@pytest.mark.parametrize("name", (None, "_"), ids=("own_names", "one_name"))
@pytest.mark.parametrize("schedule", POLICIES)
def test_loops_one_after_another_on_a_link_play_as_unrolled_within_its_slots(schedule, name):
    differ, over = [], []
    for slots in (1, 2, 4, 6):
        machine = _link_slots(slots)
        for trips in (5, 20, 65):
            for bodies in _LOOPS_ON_ONE_LINK:
                kernel = _one_after_another(trips, *bodies, name=name)
                lowered = [
                    pulse.compile(
                        kernel,
                        schedule=schedule,
                        unroll_threshold=unroll,
                        machine=machine,
                        passes=(),
                    ).lower(Q[0])
                    for unroll in (0, 10**6)
                ]
                rolled, unrolled = (
                    sorted(
                        (e.kind, e.line, e.start_cycles, e.duration_cycles) for e in low.schedule
                    )
                    for low in lowered
                )
                if rolled != unrolled:
                    differ.append((bodies, trips, slots))
                if _most_in_flight(lowered[0]) > slots:
                    over.append((bodies, trips, slots))
    assert differ == []
    assert over == []


def _link_events(bodies: tuple[str, ...], schedule: str, slots: int) -> Any:
    lowered = pulse.compile(
        _one_after_another(5, *bodies),
        schedule=schedule,
        unroll_threshold=0,
        machine=_link_slots(slots),
        passes=(),
    ).lower(Q[0])
    return sorted((e.kind, e.line, e.start_cycles, e.duration_cycles) for e in lowered.schedule)


# Under alap a request kept at its forward start needs a slot no later
# request took; the last requests, in flight past the end of the program,
# take none, and alap still places what comes before them late.
@pytest.mark.parametrize("slots", (1, 4, 6))
def test_requests_in_flight_past_the_end_leave_alap_its_schedule(slots):
    same = [
        bodies
        for bodies in (("unawaited",), ("drives", "unawaited"), ("answer", "spaced"))
        if _link_events(bodies, "alap", slots) == _link_events(bodies, "asap", slots)
    ]
    assert same == []


def _two_calls_a_round(n: int) -> Any:
    @pulse.kernel
    def rounds(q, r):
        a = get_drive_line(q)
        b = get_drive_line(r)
        p = get_readout_line(q)
        for _ in range(n):
            m = readout(p, square(19, 0.05))
            f = decide.submit([m]).result()
            shift, unused = tune.submit([m]).result()
            if f[0]:
                drive(b, square(22, 0.2))
            shift_phase(a.tone, shift)

    return rounds


# Slots handed out in reverse program order to requests holding theirs for
# different times (two calls of different latencies a round) can run out
# where the requests fit: alap keeps its schedule unless they do not.
@pytest.mark.parametrize("unroll_threshold", (0, 10**6))
def test_calls_holding_slots_for_different_times_leave_alap_its_schedule(unroll_threshold):
    events = {
        schedule: sorted(
            (e.kind, e.line, e.start_cycles, e.duration_cycles)
            for e in pulse.compile(
                _two_calls_a_round(37),
                schedule=schedule,
                unroll_threshold=unroll_threshold,
                passes=(),
            )
            .lower(Q[0], Q[1])
            .schedule
        )
        for schedule in ("alap", "asap")
    }
    assert events["alap"] != events["asap"]


def _link_size(n: int, bodies: tuple[str, ...], schedule: str, slots: int) -> int:
    lowered = pulse.compile(
        _one_after_another(n, *bodies),
        schedule=schedule,
        unroll_threshold=0,
        machine=_link_slots(slots),
        passes=(),
    ).lower(Q[0])
    return lowered.mlir("pulse").count("start_cycles")


# On one slot each loop's first request waits for the last of the loop
# before it, and then each for the one before it: the loops stay rolled
# however many trips they run.
@pytest.mark.parametrize("schedule", POLICIES)
def test_a_loop_waiting_for_the_slots_of_the_loop_before_stays_rolled(schedule):
    grow = [
        bodies
        for bodies in (
            ("answer", "unawaited"),
            ("answer", "spaced"),
            ("answer", "nested"),
            ("answer", "lazy"),
            ("drives", "answer", "unawaited"),
        )
        if _link_size(300, bodies, schedule, 1) != _link_size(1140, bodies, schedule, 1)
    ]
    assert grow == []
