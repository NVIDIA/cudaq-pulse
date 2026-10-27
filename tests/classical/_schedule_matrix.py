# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Programs that call classical functions from every place a kernel can, under
every schedule, and the properties every compiled one must have.

A case is one program: the schedule of the compiled kernel, the schedule of
the kernel it calls (if it calls one), where the call sits, the link it
travels on and how its answer is taken. ``program(case)`` builds it,
``expected_error(case)`` says how compiling it must fail, if it must, and
the ``check_*`` functions test a compiled one. Times are read only through
the accessors below, in clock cycles.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import cudaq_pulse as pulse
from cudaq_pulse.classical import Bits, Float, Link
from cudaq_pulse.compiler import MachineModel

CLOCK = 2.0  # GHz, the test device's clock
SPECTATOR, X90, RO, ARM, TAIL = 1200, 24, 200, 40, 30
LATENCY_NS = 150.0
TIMEOUT_NS = 200.0  # longer than the latency: in simulation the answer is in time
TAU_BOUNDS = (8, 400)
TAU_ANSWER = 120.0  # what a variable wait's length function answers
FLOAT_TAU = 37.4  # the kernels' float parameter: a wait length in cycles
LOOP = 3

LINKS = {
    "controller": Link(fixed_ns=4, feedback_ns=8, max_in_flight=4),
    "gpu": Link(fixed_ns=100, bytes_per_ns=10, feedback_ns=50),
}
MACHINE = MachineModel(links=dict(LINKS))

# The schedule of the compiled kernel: its decorator's ("default" is none,
# which compiles as ALAP), or one only compile() takes.
ROOTS = ("default", "asap", "alap", "rcp", "alap_rcp")
# The schedule of the kernel it calls: None schedules it with its caller.
CALLEES = (None, "asap", "alap")
TAKES = ("result", "timeout", "blocking")
PLACEMENTS = (
    "top",
    "static_loop",
    "rolled_loop",
    "branch",
    "before_measure",
    "straddle_measure",
    "float_wait",
    "variable_region",
    "mixed_wait",
    "both_waits",
    "across_variable",
    "loop_variable",
    "callee",
    "callee_loop",
    "callee_twice",
    "callee_feedback",
    "callee_float",
    "callee_variable",
    "future_out",
    "future_in",
)
CALLEE_PLACEMENTS = frozenset(p for p in PLACEMENTS if p.startswith(("callee", "future")))
# Where the answer is taken inside the called kernel.
TAKEN_IN_CALLEE = CALLEE_PLACEMENTS - {"future_out"}
# Called kernels whose events are only their own, so their shape can be
# compared with the kernel compiled alone.
SHAPED = frozenset({"callee", "callee_loop", "callee_twice", "callee_feedback", "callee_float"})


@dataclass(frozen=True)
class Case:
    root: str = "default"
    callee: str | None = None
    placement: str = "top"
    link: str = "controller"
    take: str = "result"
    flatten: bool = False

    @property
    def id(self) -> str:
        callee = self.callee or "inline"
        flat = "-flat" if self.flatten else ""
        return f"{self.root}-{callee}-{self.placement}-{self.link}-{self.take}{flat}"

    @property
    def blocks(self) -> bool:
        """Whether the program has a kernel scheduled on its own."""
        return self.placement in CALLEE_PLACEMENTS and self.callee is not None and not self.flatten

    @property
    def policy(self) -> str:
        return "alap" if self.root == "default" else self.root


def all_cases() -> list[Case]:
    """Every case, without duplicates: a callee's schedule and flattening
    matter only where there is a callee."""
    cases = []
    for root, callee, placement, link, take, flatten in itertools.product(
        ROOTS, CALLEES, PLACEMENTS, LINKS, TAKES, (False, True)
    ):
        if placement not in CALLEE_PLACEMENTS and (callee is not None or flatten):
            continue
        if callee is None and flatten:
            continue
        cases.append(Case(root, callee, placement, link, take, flatten))
    return cases


# ── Times, in clock cycles ───────────────────────────────────────────────────


class Call(NamedTuple):
    """The times of one call from ``classical_report()``."""

    function: str
    link: str | None
    submit: int
    submit_length: int
    due: int
    taken: int | None
    first_use: int | None
    deadline_slack: int | None


def calls(low: Any) -> list[Call]:
    return [
        Call(
            c.function,
            c.link_name,
            c.submit_cycles,
            c.submit_duration_cycles,
            c.due_cycles,
            c.taken_cycles,
            c.first_use_cycles,
            c.deadline_slack_cycles,
        )
        for c in low.classical_report()
    ]


def epoch_starts(low: Any) -> list[tuple[str, int, int]]:
    """Each epoch after the first: what opened it, on which line or link,
    and when it starts in the static schedule."""
    return [(e.opened_by, e.line, e.start_cycles) for e in low.epochs()[1:]]


def _export_times(event: dict[str, Any]) -> tuple[int, int]:
    return event["start_cycles"], event["duration_cycles"]


def timeline(low: Any) -> list[tuple[str, int, int, int]]:
    return [(e.kind, e.line, e.start, e.end) for e in low.schedule]


def relative(events: Iterable[Any]) -> list[tuple[str, int, int]]:
    events = list(events)
    origin = min(e.start for e in events)
    return [(e.kind, e.start - origin, e.end - origin) for e in events]


# ── Classical functions ──────────────────────────────────────────────────────


def _declare(link: str, take: str) -> tuple[Any, Any]:
    mode = "blocking" if take == "blocking" else "bounded"
    decode = pulse.classical(
        lambda bits: [bits[0]],
        name=f"decode_{link}_{take}",
        args=(Bits(1),),
        results=(Bits(1),),
        max_latency_ns=LATENCY_NS,
        latency_mode=mode,
        link=link,
    )
    tau = pulse.classical(
        lambda x: TAU_ANSWER,
        name=f"tau_{link}_{take}",
        args=(Float(),),
        results=(Float(),),
        max_latency_ns=LATENCY_NS,
        latency_mode=mode,
        link=link,
    )
    return decode, tau


FUNCTIONS = {(link, take): _declare(link, take) for link in LINKS for take in TAKES}


def budget(case: Case, function: Any) -> int:
    """An answer's budget after its submission: the latency (or the
    timeout) plus the link's feedback time."""
    link = LINKS[case.link]
    latency = TIMEOUT_NS if case.take == "timeout" else function.max_latency_ns
    return math.ceil(latency * CLOCK) + math.ceil(link.feedback_ns * CLOCK)


def barrier_lengths(case: Case) -> list[int]:
    """How long each variable wait of *case* runs, in program order."""
    answer = math.floor(TAU_ANSWER + 0.5)
    mixed = math.floor(TAU_ANSWER + FLOAT_TAU + 0.5)
    return {
        "variable_region": [answer],
        "across_variable": [answer],
        "callee_variable": [answer],
        "loop_variable": [answer] * LOOP,
        "mixed_wait": [mixed],
        "both_waits": [answer, mixed],
    }.get(case.placement, [])


# ── Programs ─────────────────────────────────────────────────────────────────


def _calls(case: Case) -> tuple[Any, Any, Any]:
    """How *case* calls: ``answer(future)`` takes the answer bit,
    ``length()`` asks for a wait's length, ``feedback(d, t, m)`` sends a
    measurement and plays a pulse if the answer is set."""
    decode, tau_fn = FUNCTIONS[(case.link, case.take)]
    take = case.take

    def answer(future):
        if take == "timeout":
            _, bits = future.result(timeout_ns=TIMEOUT_NS)
            return bits[0]
        return future.result()[0]

    def length():
        future = tau_fn.submit(1.0)
        if take == "timeout":
            _, x = future.result(timeout_ns=TIMEOUT_NS)
            return x
        return future.result()

    def feedback(d, t, m):
        if answer(decode.submit([m])):
            drive(d, square(ARM, 0.3), t)

    return answer, length, feedback


def called(case: Case, *, as_helper: bool = False) -> Any:
    """The kernel *case* calls, with the case's schedule; with *as_helper*
    a plain function instead, inlined while tracing."""
    decode, _ = FUNCTIONS[(case.link, case.take)]
    answer, length, feedback = _calls(case)

    def measured(q):
        d, t = get_drive_line(q)
        r, rt = get_readout_line(q)
        m = readout(r, square(RO, 0.05), rt)
        feedback(d, t, m)

    def measured_with_feedback(q):
        d, t = get_drive_line(q)
        r, rt = get_readout_line(q)
        m = readout(r, square(RO, 0.05), rt)
        if m:
            drive(d, square(ARM, 0.2), t)
        feedback(d, t, m)

    def measured_after_wait(q, tau: float):
        d, t = get_drive_line(q)
        r, rt = get_readout_line(q)
        wait(d, tau, bounds=(0, 400))
        sync(d, r)
        m = readout(r, square(RO, 0.05), rt)
        feedback(d, t, m)

    def measured_after_variable(q):
        d, t = get_drive_line(q)
        r, rt = get_readout_line(q)
        wait(d, length(), bounds=TAU_BOUNDS)
        m = readout(r, square(RO, 0.05), rt)
        feedback(d, t, m)

    def submitted(q):
        r, rt = get_readout_line(q)
        return decode.submit([readout(r, square(RO, 0.05), rt)])

    def taken(future, q):
        d, t = get_drive_line(q)
        if answer(future):
            drive(d, square(ARM, 0.3), t)

    body = {
        "callee": measured,
        "callee_loop": measured,
        "callee_twice": measured,
        "callee_feedback": measured_with_feedback,
        "callee_float": measured_after_wait,
        "callee_variable": measured_after_variable,
        "future_out": submitted,
        "future_in": taken,
    }[case.placement]
    return body if as_helper else pulse.kernel(schedule=case.callee)(body)


def program(case: Case, *, callee_as_helper: bool = False) -> Any:
    """The kernel of *case*: ``k(q, p, n: int, tau: float)``.

    ``q`` is measured and fed back to; ``p`` plays a long pulse first, so a
    schedule has room to move the call. With *callee_as_helper* the called
    kernel is a plain function, inlined while tracing.
    """
    decode, _ = FUNCTIONS[(case.link, case.take)]
    answer, length, feedback = _calls(case)
    callee = None
    if case.placement in CALLEE_PLACEMENTS:
        callee = called(case, as_helper=callee_as_helper)
    placement = case.placement

    def k(q, p, n: int, tau: float):
        d, t = get_drive_line(q)
        r, rt = get_readout_line(q)
        e, u = get_drive_line(p)
        drive(e, square(SPECTATOR, 0.05), u)
        drive(d, square(X90, 0.1), t)
        m = readout(r, square(RO, 0.05), rt)
        if placement == "top":
            feedback(d, t, m)
        elif placement in ("static_loop", "rolled_loop"):
            for _ in range(n):
                feedback(d, t, readout(r, square(RO, 0.05), rt))
        elif placement == "branch":
            if m:
                feedback(d, t, readout(r, square(RO, 0.05), rt))
        elif placement == "before_measure":
            bit = answer(decode.submit([m]))
            m2 = readout(r, square(RO, 0.05), rt)
            if bit:
                drive(d, square(ARM, 0.3), t)
            if m2:
                drive(d, square(ARM, 0.2), t)
        elif placement == "straddle_measure":
            future = decode.submit([m])
            m2 = readout(r, square(RO, 0.05), rt)
            if answer(future):
                drive(d, square(ARM, 0.3), t)
            if m2:
                drive(d, square(ARM, 0.2), t)
        elif placement == "float_wait":
            wait(d, tau, bounds=(0, 400))
            feedback(d, t, m)
        elif placement == "variable_region":
            wait(d, length(), bounds=TAU_BOUNDS)
            feedback(d, t, m)
        elif placement == "mixed_wait":
            wait(d, length() + tau, bounds=TAU_BOUNDS)
            feedback(d, t, m)
        elif placement == "both_waits":
            wait(d, tau, bounds=(0, 400))
            wait(d, length(), bounds=TAU_BOUNDS)
            feedback(d, t, m)
            wait(d, length() + tau, bounds=TAU_BOUNDS)
        elif placement == "across_variable":
            future = decode.submit([m])
            wait(d, length(), bounds=TAU_BOUNDS)
            if answer(future):
                drive(d, square(ARM, 0.3), t)
        elif placement == "loop_variable":
            for _ in range(n):
                wait(d, length(), bounds=TAU_BOUNDS)
                feedback(d, t, readout(r, square(RO, 0.05), rt))
        elif placement in ("callee", "callee_feedback", "callee_variable"):
            callee(q)
        elif placement == "callee_loop":
            for _ in range(n):
                callee(q)
        elif placement == "callee_twice":
            callee(q)
            drive(e, square(TAIL, 0.05), u)
            callee(q)
        elif placement == "callee_float":
            callee(q, tau)
        elif placement == "future_out":
            if answer(callee(q)):
                drive(d, square(ARM, 0.3), t)
        elif placement == "future_in":
            callee(decode.submit([m]), q)
        drive(e, square(TAIL, 0.05), u)

    k.__name__ = f"k_{case.placement}"
    return pulse.kernel(schedule=case.root if case.root in ("asap", "alap") else None)(k)


def alone(case: Case) -> Any:
    """A kernel that only calls *case*'s called kernel: its own timing."""
    callee = called(case)
    with_tau = case.placement == "callee_float"

    def w(q, p, n: int, tau: float):
        if with_tau:
            callee(q, tau)
        else:
            callee(q)

    return pulse.kernel(w)


def compile_case(case: Case, device: Any = None, kernel: Any = None, **options: Any) -> Any:
    """``pulse.compile`` of *case*'s program (or of *kernel*), as the case says."""
    settings: dict[str, Any] = {
        "machine": MACHINE,
        "flatten_schedules": case.flatten,
        "unroll_threshold": 10**6 if case.placement == "static_loop" else 0,
    }
    if case.root in ("rcp", "alap_rcp"):
        settings["schedule"] = case.root
    if device is not None:
        settings["device"] = device
    settings.update(options)
    return pulse.compile(kernel or program(case), **settings)


def lower_case(case: Case, qudits: Sequence[Any], **options: Any) -> Any:
    return compile_case(case, **options).lower(qudits[0], qudits[1], LOOP, FLOAT_TAU)


# ── Which cases are errors ───────────────────────────────────────────────────


def expected_error(case: Case) -> tuple[type[Exception], str] | None:
    """How compiling *case* must fail, with the message naming the fix, or
    None."""
    if case.policy in ("rcp", "alap_rcp") and case.blocks:
        return (
            pulse.CompilationError,
            r"remove schedule= from the called kernels.*flatten_schedules",
        )
    if case.take == "blocking" and case.placement == "branch":
        return RuntimeError, r"blocking wait cannot be inside one arm of an scf.if; wait before"
    if case.blocks and case.take == "blocking" and case.placement in TAKEN_IN_CALLEE:
        return RuntimeError, (
            r"blocking wait cannot be inside a pulse.schedule_block.*take the answer "
            r"after it returns, or call the kernel without schedule="
        )
    if case.blocks and case.placement == "callee_variable":
        return RuntimeError, (
            r"variable wait cannot be inside a pulse.schedule_block.*call the kernel "
            r"without schedule="
        )
    return None


# ── Properties ───────────────────────────────────────────────────────────────


def check_calls(case: Case, low: Any) -> None:
    """Latencies and link slots: every answer is budgeted exactly and taken
    no earlier than it is due, and no link holds more requests than its
    slots or sends two at once."""
    link = LINKS[case.link]
    functions = {fn.name: fn for fn in low.classical}
    made = calls(low)
    assert made, "the program makes no call"
    for call in made:
        assert call.link == case.link
        assert call.due == call.submit + call.submit_length + budget(case, functions[call.function])
        assert call.submit_length >= max(1, math.ceil(link.fixed_ns * CLOCK)), call
        assert call.taken is None or call.taken >= call.due, call
        assert call.first_use is None or call.first_use >= call.due, call
    spans = sorted((c.submit, c.due) for c in made)
    for start, _ in spans:
        assert sum(s <= start < e for s, e in spans) <= link.max_in_flight
    for (s0, _), (s1, _) in zip(spans, spans[1:]):
        assert s1 >= s0 + 1


def check_lines(low: Any) -> None:
    """No two operations overlap on a line, and nothing starts before 0."""
    by_line: dict[int, list[tuple[int, int]]] = {}
    for e in low.schedule:
        assert e.start >= 0, e
        if e.link is None:
            by_line.setdefault(e.line, []).append((e.start, e.end))
    for spans in by_line.values():
        spans.sort()
        for (_, end), (start, _) in zip(spans, spans[1:]):
            assert end <= start, spans


def check_barriers(case: Case, low: Any) -> None:
    """A variable or blocking wait is a barrier: every operation before it
    in program order ends by its start, every one after starts after it;
    there is one per variable wait and blocking answer."""
    events = low.schedule
    barriers: list[int] = []
    for opened_by, line, start in epoch_starts(low):
        for i, e in enumerate(events):
            if i in barriers:
                continue
            wait = opened_by == "wait" and e.kind == "wait" and e.link is None and e.end == start
            answer = opened_by == "answer" and e.kind == "answer" and e.start == start
            if (wait or answer) and e.line == line:
                barriers.append(i)
                break
        else:
            raise AssertionError(f"no barrier opens the epoch at {start}")
    for i in barriers:
        barrier = events[i]
        for j, e in enumerate(events):
            if j < i:
                assert e.end <= barrier.start or e.kind == "answer", (e, barrier)
            elif j > i:
                assert e.start >= barrier.end, (e, barrier)
    variable = len(barrier_lengths(case))
    if case.placement in ("loop_variable", "callee_loop") or case.take == "blocking":
        assert len(barriers) >= variable
    else:
        assert len(barriers) == variable


def check_reverify(low: Any) -> None:
    """The scheduled module, read back from its text, verifies again: its
    timing and its calls."""
    from cudaq_pulse._native import MLIRPipeline

    module = MLIRPipeline().parse_to_module(low.mlir())
    module.run_passes(["pulse-verify", "pulse-classical-check"])


def check_tight(case: Case, low: Any) -> None:
    """Nothing floats: under ASAP every operation starts when something it
    may wait for ends (or at 0); under ALAP every pulse and wait ends when
    something that may wait for it starts (or at the end)."""
    events = low.schedule
    barriers = {start for _, _, start in epoch_starts(low)}
    if case.policy == "asap":
        ready = {0} | {e.end for e in events} | {c.due for c in calls(low)} | barriers
        for e in events:
            assert e.start in ready, e
    else:
        makespan = max(e.end for e in events)
        needed = {makespan} | {e.start for e in events} | barriers
        for e in events:
            if e.kind in ("drive", "readout", "wait"):
                assert e.end in needed, e


def check_policy_order(case: Case, qudits: Sequence[Any]) -> None:
    """ASAP places every operation no later than ALAP does, and both take
    as long: ALAP moves work late within the same makespan."""
    asap = lower_case(case, qudits, schedule="asap").schedule
    alap = lower_case(case, qudits, schedule="alap").schedule
    assert [(e.kind, e.line) for e in asap] == [(e.kind, e.line) for e in alap]
    for early, late in zip(asap, alap):
        assert early.start <= late.start, (early, late)
    assert max(e.end for e in asap) == max(e.end for e in alap)


def callee_slices(low: Any, n_callee: int) -> list[list[Any]]:
    """The events of each call of the called kernel, in program order: the
    events of ``q`` and of the link after the program's own drive and
    readout of ``q`` (``p`` plays only the program's own pulses)."""
    spectator = low.qudits[1]
    events = [e for e in low.schedule if e.qubit != spectator][2:]
    return [events[k : k + n_callee] for k in range(0, len(events), n_callee)]


def check_callee_shape(low: Any, own: Any) -> None:
    """A kernel with its own schedule= keeps its internal timing wherever
    it is called: each call's events are the kernel compiled alone, moved."""
    shape = relative(own.schedule)
    slices = callee_slices(low, len(shape))
    assert slices
    for events in slices:
        assert relative(events) == shape


def export_text(low: Any) -> str:
    program = low.export_controller()
    text = json.dumps(program, sort_keys=True)
    assert json.loads(text) == program
    return text


def check_export(case: Case, low: Any, again: Any) -> None:
    """The controller export is JSON data, the same from a second
    compilation, and within limits that allow what the case does; limits
    without timeouts reject exactly the programs that take one."""
    from cudaq_pulse.controller import ControllerLimits, check

    assert export_text(low) == export_text(again)
    assert check(low, ControllerLimits(clock_ghz=CLOCK)) == []
    no_timeouts = check(low, ControllerLimits(answer_timeouts=False))
    if case.take == "timeout":
        assert no_timeouts
        assert all("take it with result() (no timeout_ns)" in v for v in no_timeouts)
    else:
        assert no_timeouts == []


# ── The explicit reference ───────────────────────────────────────────────────


def epoch_shifts(case: Case, low: Any) -> list[int]:
    """How much later each epoch starts at run time than in the static
    schedule, which places each variable wait at its lowest length."""
    runs = iter(barrier_lengths(case))
    shifts = [0]
    for opened_by, _, _ in epoch_starts(low):
        shifts.append(shifts[-1] + (next(runs) - TAU_BOUNDS[0] if opened_by == "wait" else 0))
    return shifts


def explicit_reference(low: Any, shifts: Sequence[int]) -> Any:
    """A kernel without calls that plays every pulse of *low* where *low*'s
    schedule puts it, each after a wait, and branches on the readout a
    call's answer comes from instead (the functions answer their input).

    *shifts* is how much later each epoch starts at run time.
    """
    exported = low.export_controller()
    events = exported["events"]
    lines = {line["line"]: line["qudit"] for line in exported["lines"]}
    qudits = sorted(set(lines.values()))
    submits: dict[str, list[int]] = {}
    answer_source: dict[int, int] = {}
    for event in events:
        if event["kind"] == "submit":
            measured = event["measurements"]
            submits.setdefault(event["function"], []).append(measured[0] if measured else -1)
        elif event["kind"] == "answer":
            answer_source[event["index"]] = submits[event["function"]].pop(0)
    plan = []
    for event in events:
        if event["kind"] not in ("drive", "readout"):
            continue
        conditions = []
        for condition in event["conditions"]:
            refs = list(condition["readouts"]) + [answer_source[a] for a in condition["answers"]]
            assert len(refs) == 1 and refs[0] >= 0 and condition["arm"], condition
            conditions.append(refs[0])
        start, duration = _export_times(event)
        plan.append(
            (
                event["index"],
                event["kind"],
                qudits.index(lines[event["line"]]),
                start + shifts[event["epoch"]],
                duration,
                event["waveform"]["params"]["amplitude"],
                tuple(conditions),
            )
        )

    # Each pulse waits on its line from where the line's previous pulse
    # ends; a branch ends with its arm, so this holds across branches too.
    cursor: dict[tuple[str, int], int] = {}
    timed = []
    for index, kind, which, start, duration, amplitude, conditions in plan:
        gap = start - cursor.get((kind, which), 0)
        cursor[(kind, which)] = start + duration
        timed.append((index, kind, which, gap, duration, amplitude, conditions))
    tree = _hoist(_nest(timed, 0))
    # The program runs until its last operation ends at run time, a wait
    # perhaps: the reference waits as long.
    end = max(e.end for e in low.schedule) + shifts[-1]
    last = max((cursor.get(key, 0), key) for key in cursor)
    if end > last[0]:
        tree.append(("wait", last[1], end - last[0]))

    def play(nodes, lines, measured):
        for node in nodes:
            if node[0] == "branch":
                if measured[node[1]]:
                    play(node[2], lines, dict(measured))
                continue
            if node[0] == "wait":
                wait(lines[node[1]][0], node[2])
                continue
            _, index, kind, which, gap, duration, amplitude = node
            line, tone = lines[(kind, which)]
            if gap > 0:
                wait(line, gap)
            if kind == "readout":
                measured[index] = readout(line, square(duration, amplitude), tone)
            else:
                drive(line, square(duration, amplitude), tone)

    def reference(q0, q1):
        lines = {}
        for which, q in enumerate((q0, q1)):
            lines[("drive", which)] = get_drive_line(q)
            lines[("readout", which)] = get_readout_line(q)
        play(tree, lines, {})

    # Without fusion, which would merge the pulses it places back to back.
    return pulse.compile(pulse.kernel(schedule="asap")(reference), passes=("verify",))


def _nest(steps: Sequence[tuple[Any, ...]], depth: int) -> list[tuple[Any, ...]]:
    """*steps*, which share their first *depth* conditions, as a tree: an
    op, or a branch on the next condition holding the steps under it."""
    nodes: list[tuple[Any, ...]] = []
    k = 0
    while k < len(steps):
        conditions = steps[k][6]
        if len(conditions) == depth:
            nodes.append(("op", *steps[k][:6]))
            k += 1
            continue
        end = k + 1
        while end < len(steps) and len(steps[end][6]) > depth:
            if steps[end][6][depth] != conditions[depth]:
                break
            end += 1
        nodes.append(("branch", conditions[depth], _nest(steps[k:end], depth + 1)))
        k = end
    return nodes


def _hoist(nodes: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    """Move the wait before the first op of each line in a branch to before
    the branch: in an arm it would start when the branch does, after the
    measurement, not when the line is free."""
    out: list[tuple[Any, ...]] = []
    for node in nodes:
        if node[0] != "branch":
            out.append(node)
            continue
        seen: set[tuple[str, int]] = set()

        def strip(children: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
            kept = []
            for child in children:
                if child[0] == "branch":
                    kept.append(("branch", child[1], strip(child[2])))
                    continue
                key = (child[2], child[3])
                if key not in seen:
                    seen.add(key)
                    if child[4] > 0:
                        out.append(("wait", key, child[4]))
                    child = (*child[:4], 0, *child[5:])
                kept.append(child)
            return kept

        out.append(("branch", node[1], strip(node[2])))
    return out


def pulses(low: Any, shifts: Sequence[int] = ()) -> list[tuple[str, int, int, int]]:
    """The drives and readouts, by qudit, with their timing at run time."""
    exported = low.export_controller()
    lines = {line["line"]: line["qudit"] for line in exported["lines"]}
    found = []
    for event in exported["events"]:
        if event["kind"] in ("drive", "readout"):
            start, duration = _export_times(event)
            shift = shifts[event["epoch"]] if shifts else 0
            found.append((event["kind"], lines[event["line"]], start + shift, duration))
    return sorted(found)
