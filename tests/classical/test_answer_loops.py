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
) -> list[str]:
    pad = "    " * indent
    calls = _STATEFUL if stateful else _CALLS
    roll = rng.random()
    measured, answered = scope["measured"], scope["answered"]
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
            lines += _statement(rng, 2, inner, indent + 1, stateful)
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
            lines += _statement(rng, depth + 1, inner, indent + 1, stateful)
        if rng.random() < 0.4:
            lines.append(f"{pad}else:")
            inner = {"measured": list(measured), "answered": list(answered)}
            for _ in range(rng.randint(1, 2)):
                lines += _statement(rng, depth + 1, inner, indent + 1, stateful)
        return lines
    if roll < 0.95:
        first, second = rng.sample(_DRIVES + _READOUTS, 2)
        return [f"{pad}sync({first}, {second})"]
    return [f"{pad}drive({rng.choice(_DRIVES)}, gaussian({rng.randint(4, 40)}, 0.1, 2.0))"]


def _random_source(seed: int, trips: int, stateful: bool = False) -> str:
    """A random loop of *trips* iterations that reads out, calls and branches
    on answers, between a random prologue and epilogue; some have a loop of
    their own in the body. *stateful* adds blocking calls and waits of an
    answer's length, and answers an inner loop takes and need not test."""
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
    extra: list[str] = []
    for _ in range(rng.randint(0, 4)):
        extra += _statement(rng, 0, scope, 2, stateful)
    if rng.random() < 0.15:
        inner = {"measured": list(scope["measured"]), "answered": list(scope["answered"])}
        extra.append(f"        for _ in range({rng.randint(2, 3)}):")
        for _ in range(rng.randint(1, 3)):
            extra += _statement(rng, 0 if stateful else 1, inner, 3, stateful)
    lines += test + extra if rng.random() < 0.5 else extra + test
    for _ in range(rng.randint(0, 2)):
        lines += _statement(rng, 2, {"measured": [], "answered": []}, 1)
    return "\n".join(lines) + "\n"


def _random_kernel(seed: int, trips: int, stateful: bool = False) -> Any:
    namespace: dict[str, Any] = dict(_STATEFUL, settle=settle)
    source = _random_source(seed, trips, stateful)
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



# The child's limits, far past what lowering every kernel below at 10**6
# trips under every policy takes: about 0.4 s and 300 MB each, nearly all of
# both in checking the calls' slots and deadlines, which grows with the trips.
CPU_S = 30
RSS_MB = 700

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
from tests.classical.test_answer_loops import POLICIES, _kernels

out = {{}}
kernels = _kernels({trips})
for name in kernels:
    for schedule in POLICIES:
        started = time.process_time()
        program = pulse.compile(kernels[name], schedule=schedule, unroll_threshold=0)
        lowered = program.lower(Q[0], Q[1], Q[2])
        spent = time.process_time() - started
        out[f"{{name}}/{{schedule}}"] = [str(lowered.duration.exact_ns), spent]
print(json.dumps(out))
"""


def _in_a_child(trips: int) -> dict[str, list[Any]]:
    """The duration of every kernel of *trips* iterations under every policy,
    and the CPU time its lowering took, in a child process that is stopped
    past CPU_S seconds or RSS_MB MB."""
    code = _CHILD.format(wall=2 * CPU_S, cpu=CPU_S, mb=RSS_MB, trips=trips)
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


def test_a_loop_of_10_6_iterations_branching_on_answers_lowers_in_bounded_time_and_memory():
    from fractions import Fraction

    many = 10**6
    found = _in_a_child(many)
    small: dict[tuple[str, int], Fraction] = {}
    # As many trips past the peeled ones as 10**6 is, modulo every run of
    # iterations a loop repeats after.
    fewer, more = 1240, 2080
    for trips in (fewer, more):
        kernels = _kernels(trips)
        for name in NAMES:
            for schedule in POLICIES:
                program = pulse.compile(kernels[name], schedule=schedule, unroll_threshold=0)
                lowered = program.lower(Q[0], Q[1], Q[2])
                small[f"{name}/{schedule}", trips] = lowered.duration.exact_ns
    assert len(found) == len(NAMES) * len(POLICIES)
    for key, (duration, spent) in found.items():
        step = small[key, more] - small[key, fewer]
        expected = small[key, fewer] + (many - fewer) // (more - fewer) * step
        assert Fraction(duration) == expected, key
        assert spent < 1.0, key
