# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Controller checks of rolled loops that call classical functions.

A loop whose answers are used as values, or not at all, stays rolled, and
the checks reason about it once per body as about any rolled loop; what
they report is what they report with every iteration played.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from typing import Any

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import Controller, _loops, controller
from cudaq_pulse.controller import _AMPLITUDES, _PARAMETERS, ControllerLimitError, check

from tests import _child
from tests._handles import Q, ladder_target
from tests.test_controller_rolled_loops import LIMITS, OWN_CLOCKS, _systems


@pulse.classical(max_latency_ns=400)
def check_bit(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]


@pulse.classical(max_latency_ns=600, latency_mode="timeout", timeout_ns=250)
def timed_bit(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]


@pulse.classical(max_latency_ns=200)
def settings(bits: pulse.Bits) -> tuple[float, float, float]:
    return 0.1 * bits[0], 1.0e5 * bits[0], 40.0 + bits[0]


def _calls(n: int) -> dict[str, Any]:
    """Loops of three qudits that call a classical function each iteration."""

    @pulse.kernel
    def answered(q, p, t):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(100, 0.05))
            _fix = check_bit.submit([m]).result()
            drive(d, square(20, 0.3))

    @pulse.kernel
    def synced(q, p, t):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        e = get_drive_line(t)
        for _ in range(n):
            m = readout(ro, square(100, 0.05))
            amp, detuning, delay = settings.submit([m]).result()
            sync(d, ro, e)
            drive(d, square(20, amp * 0.5))
            drive(e, square(20, 0.2))
            sync(d, ro, e)

    @pulse.kernel
    def unused(q, p, t):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(100, 0.05))
            check_bit.submit([m])
            drive(d, square(20, 0.3))

    @pulse.kernel
    def valued(q, p, t):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(100, 0.05))
            amp, detuning, delay = settings.submit([m]).result()
            shift_frequency(d.tone, detuning)
            drive(d, square(20, amp * 0.5))
            shift_phase(d.tone, amp)
            drive(d, square(20, 0.2))

    @pulse.kernel
    def before(q, p, t):
        d = get_drive_line(q)
        e = get_drive_line(t)
        ro = get_readout_line(q)
        m = readout(ro, square(100, 0.05))
        amp, detuning, delay = settings.submit([m]).result()
        for _ in range(n):
            drive(e, square(20, amp))
            wait(d, delay, bounds=(8, 400))
            shift_phase(d.tone, amp)
            drive(d, square(20, 0.2))

    @pulse.kernel
    def timed(q, p, t):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(100, 0.05))
            ok, fix = check_bit.submit([m]).result(timeout=300 * pulse.ns)
            drive(d, square(20, 0.3))

    return {
        "answered": answered,
        "synced": synced,
        "unused": unused,
        "valued": valued,
        "before": before,
        "timed": timed,
    }


def _lowered(kernel: Any, schedule: str, **options: Any) -> Any:
    return pulse.compile(kernel, schedule=schedule, unroll_threshold=0, **options).lower(
        Q[0], Q[1], Q[2]
    )


def _summarizing(lowered: Any, values: bool = False) -> tuple[list[Any], list[Any]]:
    repeats: list[Any] = []
    _, events, *_ = controller._walk(lowered, values=values, summary=(repeats, []))
    return events, repeats


def _every(lowered: Any) -> list[Any]:
    """The events of the check's walk of every iteration, without what only
    a walk that tracks the tones' frame changes has."""
    _, events = lowered._concrete().controller_events(_PARAMETERS, _AMPLITUDES)
    for event in events:
        for key in ("tone", "frame_updates", "phase_offset_rad"):
            event.pop(key, None)
    return events


def _played(walk: Any) -> Any:
    def played(*args: Any, summary: Any = None, **kwargs: Any) -> Any:
        return walk(*args, **kwargs)

    return played


def _outcome(make: Any) -> Any:
    try:
        return list(make())
    except (ControllerLimitError, ValueError) as error:
        return [type(error).__name__, str(error)]


_CALL_LIMITS = [*LIMITS, Controller(supports_timeouts=False, feedback_latency_cycles=500)]


@pytest.mark.parametrize("clock", [None, *range(len(OWN_CLOCKS))])
@pytest.mark.parametrize("trips", [65, 100])
@pytest.mark.parametrize("name", list(_calls(1)))
@pytest.mark.parametrize("schedule", ["asap", "alap", "rcp"])
def test_a_long_rolled_loop_with_calls_is_summarized_and_checked_as_every_iteration(
    monkeypatch, clock, trips, name, schedule
):
    lowered = _lowered(_calls(trips)[name], schedule)
    _, repeats = _summarizing(lowered)
    assert repeats
    for limits in _CALL_LIMITS:
        if clock is not None:
            limits = dataclasses.replace(limits, **OWN_CLOCKS[clock])
        found = _outcome(lambda li=limits: check(lowered, li))
        compiled = _outcome(lambda li=limits: controller._compile_check(lowered, li)[0])
        with monkeypatch.context() as patch:
            patch.setattr(controller, "_walk", _played(controller._walk))
            assert found == _outcome(lambda li=limits: check(lowered, li))
            assert compiled == _outcome(lambda li=limits: controller._compile_check(lowered, li)[0])


def _frames(n: int) -> dict[str, Any]:
    """Loops whose frame changes or lines vary what each iteration knows
    only at run time."""

    @pulse.kernel
    def pending(q, p, t):
        # Phase changes from answers pile up on a tone the loop never plays.
        d = get_drive_line(q)
        e = get_drive_line(t)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(100, 0.05))
            amp, detuning, delay = settings.submit([m]).result()
            shift_phase(d.tone, amp)
            drive(e, square(20, 0.2))
        drive(d, square(20, 0.1))

    @pulse.kernel
    def late(q, p, t):
        # Each iteration's last phase change is played by the next, the last
        # one's after the loop.
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(100, 0.05))
            amp, detuning, delay = settings.submit([m]).result()
            shift_phase(d.tone, amp)
            drive(d, square(20, 0.3))
            shift_phase(d.tone, delay)
        shift_phase(d.tone, 0.5)
        drive(d, square(20, 0.1))

    @pulse.kernel
    def armed(q, p, t):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(100, 0.05))
            drive(d, square(20, 0.3))
            if m:
                shift_phase(d.tone, 0.25)
        drive(d, square(20, 0.1))

    @pulse.kernel
    def detuned(q, p, t):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(100, 0.05))
            amp, detuning, delay = settings.submit([m]).result()
            shift_frequency(d.tone, detuning)
            drive(d, square(20, 0.3))
        drive(d, square(20, 0.1))

    @pulse.kernel
    def nested(q, p, t):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(3):
            for _ in range(n):
                m = readout(ro, square(100, 0.05))
                check_bit.submit([m])
                drive(d, square(20, 0.3))
            drive(d, square(24, 0.1))

    @pulse.kernel
    def waited(q, p, t):
        # Each iteration's wait for an answer opens a segment on one timeline.
        d = get_drive_line(q)
        e = get_drive_line(t)
        ro = get_readout_line(q)
        m = readout(ro, square(100, 0.05))
        amp, detuning, delay = settings.submit([m]).result()
        for _ in range(n):
            wait(d, delay, bounds=(8, 400))
            drive(d, square(20, 0.2))
        drive(e, square(20, 0.1))
        drive(d, square(20, 0.1))

    return {
        "pending": pending,
        "late": late,
        "armed": armed,
        "detuned": detuned,
        "nested": nested,
        "waited": waited,
    }


class _Expansion:
    """The events of every iteration, from a summarizing walk's: copy j of a
    repeat is its first moved by j times its difference from the copy before
    (each line's and call's own period, in a loop of several). A reference
    to an event of that copy or the one before moves by j copies (by the
    last copy's from after the loop); starts, ends, segments, iterations and
    branch ids move by j differences; all else is the same in every copy."""

    _MOVING = frozenset({"start_cycles", "end_cycles", "segment", "iteration", "branch"})

    def __init__(self, events: list[Any], repeats: list[Any]) -> None:
        self.events = events
        self.tree, _ = _loops._tree(events, repeats)
        self.reps: list[Any] = []
        self.size: dict[int, int] = {}
        self.first: dict[int, int] = {}
        self.out: list[Any] = []
        self._collect(self.tree)

    def _collect(self, nodes: list[Any]) -> int:
        total = 0
        for node in nodes:
            if type(node) is int:
                total += 1
            else:
                self.reps.append(node)
                self.size[id(node)] = self._collect(node.body)
                total += node.copies * self.size[id(node)]
        return total

    def run(self) -> list[Any]:
        self._emit(self.tree, ())
        return self.out

    def _emit(self, nodes: list[Any], copies: tuple[tuple[Any, int], ...]) -> None:
        for node in nodes:
            if type(node) is not int:
                for j in range(node.copies):
                    self._emit(node.body, (*copies, (node, j)))
                continue
            if all(j == 0 for _, j in copies):
                self.first[node] = len(self.out)
            before = [(self.events[node - (r.end - r.begin)], j) for r, j in copies if j]
            at = {id(r): j for r, j in copies}
            event = self._value(self.events[node], before, None, node, at)
            event["index"] = len(self.out)
            self.out.append(event)

    def _ref(self, ref: int, at: int, copies: dict[int, int]) -> int:
        full = self.first[ref]
        for rep in self.reps:
            if rep.begin - (rep.end - rep.begin) <= ref < rep.end:
                if id(rep) in copies:
                    full += copies[id(rep)] * self.size[id(rep)]
                elif at >= rep.end:
                    full += (rep.copies - 1) * self.size[id(rep)]
        return full

    def _value(self, value: Any, before: list[Any], key: Any, at: int, copies: Any) -> Any:
        if isinstance(value, dict):
            out = {}
            for k, v in value.items():
                if k in ("readout", "answer") and type(v) is int:
                    out[k] = self._ref(v, at, copies)
                elif k in ("readouts", "answers", "measurements"):
                    out[k] = [self._ref(r, at, copies) for r in v]
                else:
                    inner = [(b.get(k) if isinstance(b, dict) else None, j) for b, j in before]
                    out[k] = self._value(v, inner, k, at, copies)
            return out
        if isinstance(value, list):
            return [
                self._value(
                    v,
                    [(b[n] if isinstance(b, list) and n < len(b) else None, j) for b, j in before],
                    key,
                    at,
                    copies,
                )
                for n, v in enumerate(value)
            ]
        if key in self._MOVING and type(value) is int:
            return value + sum(j * (value - b) for b, j in before if type(b) is int)
        return value


@pytest.mark.parametrize("name", [*_calls(1), *_frames(1)])
@pytest.mark.parametrize("schedule", ["asap", "alap", "rcp"])
def test_a_summarizing_walk_has_every_iteration_s_events_copy_by_copy(name, schedule):
    lowered = _lowered({**_calls(65), **_frames(65)}[name], schedule)
    events, repeats = _summarizing(lowered)
    assert repeats
    got = json.dumps(_Expansion(events, repeats).run(), sort_keys=True)
    assert got == json.dumps(_every(lowered), sort_keys=True)


def _branches(n: int) -> dict[str, Any]:
    """Loops that branch on an answer, which the scheduler may keep rolled
    with several iterations in each of the loop's."""

    @pulse.kernel
    def branched(q, p, t):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(100, 0.05))
            fix = check_bit.submit([m]).result()
            sync(d, ro)
            if fix[0]:
                drive(d, square(20, 0.3))

    @pulse.kernel
    def either(q, p, t):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(60, 0.05))
            sync(d, ro)
            if check_bit.submit([m]).result()[0]:
                drive(d, square(20, 0.3))
            else:
                wait(d, 7)
                drive(d, square(4, 0.1))

    @pulse.kernel
    def overlapped(q, p, t):
        # The next readout overlaps the call; a value is taken too.
        d = get_drive_line(q)
        e = get_drive_line(t)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(61, 0.05))
            fix = check_bit.submit([m]).result()
            amp, detuning, delay = settings.submit([m]).result()
            drive(e, square(52, amp))
            if fix[0]:
                drive(d, square(32, 0.2))

    @pulse.kernel
    def three(q, p, t):
        # Three calls on one link: a value, two branches, each answer's
        # wait moving by a period of its own under alap.
        a = get_drive_line(q)
        b = get_drive_line(p)
        c = get_drive_line(t)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(70, 0.05))
            f = check_bit.submit([m]).result()
            amp, detuning, delay = settings.submit([m]).result()
            g = check_bit.submit([m]).result()
            drive(b, square(45, amp))
            if f[0]:
                drive(a, square(20, 0.2))
            if g[0]:
                drive(c, square(33, 0.1))

    @pulse.kernel
    def timed_value(q, p, t):
        a = get_drive_line(q)
        b = get_drive_line(p)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(55, 0.05))
            ok, f = timed_bit.submit([m]).result()
            amp, detuning, delay = settings.submit([m]).result()
            drive(b, square(29, amp))
            if f[0]:
                drive(a, square(17, 0.2))

    @pulse.kernel
    def timed_apart(q, p, t):
        # Two waits with a timeout, each answer moving with the line it
        # decides, neither with line 0's.
        a = get_drive_line(q)
        b = get_drive_line(p)
        c = get_drive_line(t)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(66, 0.05))
            ok, f = timed_bit.submit([m]).result()
            fine, g = timed_bit.submit([m]).result()
            amp, detuning, delay = settings.submit([m]).result()
            drive(a, square(13, amp))
            if f[0]:
                drive(b, square(41, 0.2))
            if g[0]:
                drive(c, square(27, 0.1))

    @pulse.kernel
    def nested_timed(q, p, t):
        a = get_drive_line(q)
        b = get_drive_line(p)
        ro = get_readout_line(q)
        for _ in range(n):
            for _ in range(2):
                m = readout(ro, square(30, 0.05))
                ok, f = timed_bit.submit([m]).result()
                amp, detuning, delay = settings.submit([m]).result()
                drive(b, square(25, amp))
                if f[0]:
                    drive(a, square(12, 0.2))
            drive(a, square(5, 0.1))

    return {
        "branched": branched,
        "either": either,
        "overlapped": overlapped,
        "three": three,
        "timed_value": timed_value,
        "timed_apart": timed_apart,
        "nested_timed": nested_timed,
    }


@pytest.mark.parametrize("name", list(_branches(1)))
@pytest.mark.parametrize("schedule", ["asap", "alap", "rcp", "alap_rcp"])
def test_a_rolled_loop_branching_on_answers_has_every_iteration_s_events(name, schedule):
    lowered = _lowered(_branches(1100)[name], schedule)
    events, repeats = _summarizing(lowered)
    if "scf.for" in lowered.mlir("pulse"):
        # Kept rolled, with at most 16 iterations in each of the loop's.
        assert repeats
    got = json.dumps(_Expansion(events, repeats).run(), sort_keys=True)
    assert got == json.dumps(_every(lowered), sort_keys=True)


@pytest.mark.parametrize("clock", [None, *range(len(OWN_CLOCKS))])
@pytest.mark.parametrize("name", list(_branches(1)))
@pytest.mark.parametrize("schedule", ["asap", "alap", "rcp", "alap_rcp"])
def test_a_rolled_loop_whose_answers_move_apart_is_checked_as_every_iteration(
    monkeypatch, clock, name, schedule
):
    lowered = _lowered(_branches(300)[name], schedule)
    for limits in _CALL_LIMITS:
        if clock is not None:
            limits = dataclasses.replace(limits, **OWN_CLOCKS[clock])
        found = _outcome(lambda li=limits: check(lowered, li))
        compiled = _outcome(lambda li=limits: controller._compile_check(lowered, li)[0])
        with monkeypatch.context() as patch:
            patch.setattr(controller, "_walk", _played(controller._walk))
            assert found == _outcome(lambda li=limits: check(lowered, li))
            assert compiled == _outcome(lambda li=limits: controller._compile_check(lowered, li)[0])


def test_answers_on_one_link_moving_by_different_periods_are_summarized():
    # Under alap the branched answer moves with its drive, the value's with
    # the other line's: the walk keeps a period for each.
    lowered = _lowered(_branches(300)["overlapped"], "alap")
    events, repeats = _summarizing(lowered)
    assert [r[3] for r in repeats] == [-1]
    answers = [e for e in events if e["kind"] == "answer"]
    starts = {e["start_cycles"] for e in answers}
    assert len(answers) > 2 and len(starts) == len(answers)
    # One timeout answer a body: its copies move by its own period.
    lowered = _lowered(_branches(300)["timed_value"], "alap")
    limits = Controller(supports_timeouts=False)
    every = check(lowered, limits)
    assert len(every) == 300
    with pytest.MonkeyPatch.context() as patch:
        made: list[Any] = []
        original = _loops.summarized_check

        def spied(*args: Any) -> Any:
            found = original(*args)
            made.append(found is not None)
            return found

        patch.setattr(_loops, "summarized_check", spied)
        assert controller._compile_check(lowered, limits)[0] == every
    assert made == [True]


def _on_system(kernel: Any, schedule: str, qpu: Any) -> Any:
    try:
        compiled = pulse.compile(kernel, quantum_system=qpu, schedule=schedule, unroll_threshold=0)
        return compiled.lower(*qpu.qudits[:3]).metrics.op_count
    except ControllerLimitError as error:
        return list(error.violations)


@pytest.mark.parametrize("name", list(_calls(1)))
@pytest.mark.parametrize("schedule", ["asap", "alap", "rcp"])
@pytest.mark.parametrize("system", list(_systems()))
def test_a_long_rolled_loop_with_calls_on_a_control_system_is_checked_as_every_iteration(
    monkeypatch, system, schedule, name
):
    from cudaq_pulse import _system

    qpu = ladder_target(4, control_system=_systems()[system])
    # Loops whose lines all move by one period are summarized on a system,
    # their answers' values and capabilities checked once. The scheduler may
    # run up to 16 iterations in each of the loop's, so that one is longer.
    summarizes = name == "synced" and system in ("one", "shared", "counted", "gridded")
    kernel = _calls(1100 if summarizes else 100)[name]
    kept: list[bool] = []
    summarized = _system.ControlSystem._summarized

    def spied(self: Any, *args: Any) -> Any:
        found = summarized(self, *args)
        kept.append(found is not None)
        return found

    with monkeypatch.context() as patch:
        patch.setattr(_system.ControlSystem, "_summarized", spied)
        found = _on_system(kernel, schedule, qpu)
    with monkeypatch.context() as patch:
        patch.setattr(controller, "_walk", _played(controller._walk))
        assert found == _on_system(kernel, schedule, qpu)
    if summarizes:
        assert kept and all(kept)


_PROGRAM = """
import json
import cudaq_pulse as pulse
from cudaq_pulse import ControlSystem, Controller
from tests._handles import Q

@pulse.classical(max_latency_ns=400)
def check_bit(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]

@pulse.classical(max_latency_ns=200)
def settings(bits: pulse.Bits) -> tuple[float, float, float]:
    return 0.1 * bits[0], 1.0e5 * bits[0], 40.0 + bits[0]

@pulse.kernel
def answered(q, t):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    for _ in range(10**5):
        m = readout(ro, square(100, 0.05))
        fix = check_bit.submit([m]).result()
        drive(d, square(20, 0.3))

@pulse.kernel
def overlapped(q, t):
    d = get_drive_line(q)
    e = get_drive_line(t)
    ro = get_readout_line(q)
    for _ in range(10**6):
        m = readout(ro, square(61, 0.05))
        fix = check_bit.submit([m]).result()
        amp, detuning, delay = settings.submit([m]).result()
        drive(e, square(52, amp))
        if fix[0]:
            drive(d, square(32, 0.2))

system = ControlSystem(controllers=[Controller(instruction_cycles=2, max_amplitude=0.5)])
kernel = {name}
program = pulse.compile(kernel, control_system=system, schedule="{schedule}", unroll_threshold=0)
print(json.dumps({{"ops": program.lower(Q[0], Q[1]).metrics.op_count}}))
"""


@pytest.mark.parametrize(
    ("name", "schedule", "rss_mb"),
    [("answered", "alap", 300), ("overlapped", "alap", 600)],
    ids=["10**5 calls", "10**6 answers moving apart"],
)
def test_a_rolled_loop_of_many_calls_is_checked_in_bounded_time_and_memory(name, schedule, rss_mb):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    source = _child.limits(30, rss_mb) + _PROGRAM.format(name=name, schedule=schedule)
    out = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        cwd=root,
        env=env,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert json.loads(out.stdout.splitlines()[-1])["ops"] > 0
