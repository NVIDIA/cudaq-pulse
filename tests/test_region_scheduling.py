# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scheduling a retained scf.for gives the same timeline as unrolling it."""

from __future__ import annotations

import cudaq_pulse as pulse
import pytest
from cudaq_pulse._native._cudaq_pulse_native import MLIRPipeline
from hypothesis import given, settings
from hypothesis import strategies as st

from tests._handles import Q, ladder_target
from tests.property_settings import examples

_DURATIONS = (20, 40, 60, 100)
_LINE = "!pulse.drive_line"
_TONE = "!pulse.tone"


class _Writer:
    """Writes one function, renaming each line and tone as ops advance it."""

    def __init__(self, line_count: int):
        self.n = line_count
        self.lines: list[str] = []
        self.counter = 0
        self.text = ["func.func @main() {"]
        for i in range(line_count):
            self.emit(f"%q{i} = pulse.qudit_alloc {{qudit = {i} : i64}} : !pulse.qudit")
            self.emit(
                f"%d{i}_0, %t{i}_0 = pulse.get_drive_line %q{i} : "
                f"(!pulse.qudit) -> ({_LINE}, {_TONE})"
            )
        self.tones = [f"%t{i}_0" for i in range(line_count)]
        self.lines = [f"%d{i}_0" for i in range(line_count)]
        self.emit("%amp = arith.constant 0.1 : f64")
        self.emit("%zero = arith.constant 0.0 : f64")
        for d in _DURATIONS:
            self.emit(f"%c{d} = arith.constant {d} : i64")
            self.emit(
                f"%wf{d} = pulse.square %c{d}, %amp, %zero : i64, f64, f64 -> !pulse.waveform"
            )
            self.emit(f"%w{d} = pulse.duration_from_int %c{d} : (i64) -> !pulse.duration")

    def fresh(self, stem: str) -> str:
        self.counter += 1
        return f"%{stem}{self.counter}"

    def emit(self, line: str) -> None:
        self.text.append("  " + line)

    def op(self, op: tuple, unroll: bool = False) -> None:
        kind = op[0]
        if kind == "loop":
            _, trips, body = op
            if unroll:
                for _ in range(trips):
                    for inner in body:
                        self.op(inner, unroll)
            else:
                self.loop(trips, body)
        elif kind == "drive":
            # On line i, with the tone line j holds (its own unless given).
            i, d = op[1], op[2]
            j = op[3] if len(op) > 3 else i
            line, tone = self.fresh("d"), self.fresh("t")
            self.emit(
                f"{line}, {tone} = pulse.drive {self.lines[i]}, %wf{d}, {self.tones[j]} : "
                f"{_LINE}, !pulse.waveform, {_TONE} -> {_LINE}, {_TONE}"
            )
            self.lines[i], self.tones[j] = line, tone
        elif kind == "phase":
            _, i, value = op
            tone = self.fresh("t")
            angle = f"%c{tone[1:]}"
            self.emit(f"{angle} = arith.constant {float(value)!r} : f64")
            self.emit(
                f"{tone} = pulse.shift_phase {self.tones[i]}, {angle} : {_TONE}, f64 -> {_TONE}"
            )
            self.tones[i] = tone
        elif kind == "rotate":
            # Each member's line takes the tone of the next member's.
            members = op[1]
            moved = [self.tones[i] for i in members[1:] + members[:1]]
            for i, tone in zip(members, moved, strict=True):
                self.tones[i] = tone
        elif kind == "wait":
            _, i, d = op
            line = self.fresh("d")
            self.emit(
                f"{line} = pulse.wait {self.lines[i]}, %w{d} : "
                f"({_LINE}, !pulse.duration) -> {_LINE}"
            )
            self.lines[i] = line
        else:
            members = op[1]
            outs = [self.fresh("d") for _ in members]
            ins = ", ".join(self.lines[i] for i in members)
            types = ", ".join(_LINE for _ in members)
            self.emit(f"{', '.join(outs)} = pulse.sync {ins} : {types} -> {types}")
            for i, out in zip(members, outs, strict=True):
                self.lines[i] = out

    def loop(self, trips: int, body: list[tuple]) -> None:
        args = [self.fresh("a") for _ in range(2 * self.n)]
        results = [self.fresh("r") for _ in range(2 * self.n)]
        inits = self.lines + self.tones
        types = ", ".join([_LINE] * self.n + [_TONE] * self.n)
        lb, ub, step = self.fresh("lb"), self.fresh("ub"), self.fresh("st")
        self.emit(f"{lb} = arith.constant 0 : index")
        self.emit(f"{ub} = arith.constant {trips} : index")
        self.emit(f"{step} = arith.constant 1 : index")
        iter_args = ", ".join(f"{a} = {v}" for a, v in zip(args, inits, strict=True))
        self.emit(
            f"{', '.join(results)} = scf.for %i{self.counter} = {lb} to {ub} step {step} "
            f"iter_args({iter_args}) -> ({types}) {{"
        )
        self.lines, self.tones = args[: self.n], args[self.n :]
        for op in body:
            self.op(op)
        self.emit(f"scf.yield {', '.join(self.lines + self.tones)} : {types}")
        self.emit("}")
        self.lines, self.tones = results[: self.n], results[self.n :]

    def finish(self) -> str:
        self.emit("return")
        return "\n".join(self.text) + "\n}\n"


def _programs(line_count, pre, trips, body, post):
    rolled = _Writer(line_count)
    unrolled = _Writer(line_count)
    for op in [*pre, ("loop", trips, body), *post]:
        rolled.op(op)
        unrolled.op(op, unroll=True)
    return rolled.finish(), unrolled.finish()


def _ops(line_count):
    line = st.integers(0, line_count - 1)
    drive = st.tuples(st.just("drive"), line, st.sampled_from(_DURATIONS))
    wait = st.tuples(st.just("wait"), line, st.sampled_from(_DURATIONS))
    options = [drive, wait]
    if line_count > 1:
        sync = st.tuples(
            st.just("sync"),
            st.lists(st.integers(0, line_count - 1), min_size=2, max_size=line_count, unique=True),
        )
        options.append(sync)
    return st.one_of(*options)


@st.composite
def _case(draw):
    n = draw(st.integers(1, 3))
    ops = _ops(n)
    nested = st.tuples(st.just("loop"), st.integers(0, 4), st.lists(ops, min_size=1, max_size=3))
    return (
        n,
        draw(st.lists(ops, max_size=4)),
        draw(st.integers(0, 6)),
        draw(st.lists(st.one_of(ops, nested), min_size=1, max_size=5)),
        draw(st.lists(ops, max_size=3)),
    )


def _timeline(text, policy, drives, latency, penalty):
    module = MLIRPipeline().parse_to_module(text)
    module.schedule(policy, drives, 1, latency, penalty)
    module.run_passes(["pulse-verify"])
    return sorted(module.events())


@pytest.mark.parametrize("policy", ["asap", "alap", "rcp", "alap_rcp"])
@given(case=_case(), drives=st.integers(1, 3), penalty=st.integers(0, 5))
@settings(max_examples=examples(60))
def test_retained_loop_schedules_like_its_unrolling(policy, case, drives, penalty):
    rolled, unrolled = _programs(*case)
    assert _timeline(rolled, policy, drives, 0, penalty) == _timeline(
        unrolled, policy, drives, 0, penalty
    )


# A loop that hands tones from one iteration argument to another: each
# argument holds another tone every trip. (lines, before, trips, body, after)
_ROTATIONS = {
    # Line 0 plays the tone of line 1, then lines 1-3 pass theirs on.
    "three_rolled": (
        4,
        [],
        7,
        [("drive", 0, 40, 1), ("rotate", (1, 2, 3))],
        [("drive", i, 40) for i in (1, 2, 3)],
    ),
    "two_rolled": (
        3,
        [("drive", 2, 40)],
        7,
        [("drive", 0, 40, 1), ("rotate", (1, 2))],
        [("drive", i, 40) for i in (1, 2)],
    ),
    # The tone of line 4 is busy until 140: the last trips are peeled.
    "four_peeled": (
        5,
        [("drive", 4, 100), ("drive", 4, 40)],
        5,
        [("drive", 0, 40, 1), ("rotate", (1, 2, 3, 4))],
        [("drive", i, 40) for i in (1, 2, 3, 4)],
    ),
    # Line 0's own tone is rotated with the others.
    "three_peeled": (
        3,
        [("drive", 1, 100), ("drive", 2, 100), ("drive", 2, 100)],
        5,
        [("drive", 0, 40), ("rotate", (0, 1, 2))],
        [("drive", i, 40) for i in (0, 1, 2)],
    ),
    "three_after_a_drive": (
        3,
        [("drive", 2, 100)],
        3,
        [("drive", 0, 40), ("rotate", (0, 1, 2))],
        [("drive", i, 40) for i in (0, 1, 2)],
    ),
    # The tone played in an inner loop, which the outer loop hands on.
    "three_inner_loop": (
        4,
        [],
        7,
        [("loop", 1, [("drive", 0, 40, 1)]), ("rotate", (1, 2, 3))],
        [("drive", i, 40) for i in (1, 2, 3)],
    ),
    "three_inner_loop_of_two": (
        4,
        [],
        7,
        [("loop", 2, [("drive", 0, 40, 1)]), ("rotate", (1, 2, 3))],
        [("drive", i, 40) for i in (1, 2, 3)],
    ),
}


@pytest.mark.parametrize("case", _ROTATIONS)
@pytest.mark.parametrize("policy", ["asap", "alap", "rcp", "alap_rcp"])
def test_tones_rotated_through_a_loop_schedule_like_its_unrolling(policy, case):
    rolled, unrolled = _programs(*_ROTATIONS[case])
    assert _timeline(rolled, policy, 3, 0, 0) == _timeline(unrolled, policy, 3, 0, 0)


# ── A measurement carried to the next iteration ──────────────────────


@pulse.kernel(unroll_threshold=0)
def _carried_round(qs):
    d1 = get_drive_line(qs[1])
    r0 = get_readout_line(qs[0])
    m = readout(r0, square(400, 0.05))
    for _ in range(40):
        if m:
            drive(d1, square(40, 0.5))
        wait(d1, 600)
        m = readout(r0, square(400, 0.05))


@pulse.kernel(unroll_threshold=0)
def _rus_long_arm(qs):
    d0 = get_drive_line(qs[0])
    r0 = get_readout_line(qs[0])
    m = readout(r0, square(400, 0.05))
    for _ in range(20):
        if m:
            drive(d0, square(360, 0.5))
            sync(d0, r0)
            m = readout(r0, square(400, 0.05))


@pulse.kernel(unroll_threshold=0)
def _carried_longdrive(qs):
    d1 = get_drive_line(qs[1])
    r0 = get_readout_line(qs[0])
    m = readout(r0, square(40, 0.05))
    for _ in range(40):
        if m:
            drive(d1, square(400, 0.5))
        m = readout(r0, square(40, 0.05))


@pulse.kernel(unroll_threshold=0)
def _readout_before_branch(qs):
    d1 = get_drive_line(qs[1])
    r0 = get_readout_line(qs[0])
    m = readout(r0, square(400, 0.05))
    for _ in range(20):
        m2 = readout(r0, square(400, 0.05))
        if m:
            drive(d1, square(40, 0.5))
        m = m2


@pulse.kernel(unroll_threshold=0)
def _nested_carry(qs):
    d1 = get_drive_line(qs[1])
    r0 = get_readout_line(qs[0])
    m = readout(r0, square(40, 0.05))
    for _ in range(3):
        for _ in range(4):
            if m:
                drive(d1, square(400, 0.5))
            m = readout(r0, square(40, 0.05))


_CARRIED = {
    "carried_round": _carried_round,
    "rus_long_arm": _rus_long_arm,
    "carried_longdrive": _carried_longdrive,
    "readout_before_branch": _readout_before_branch,
    "nested": _nested_carry,
}


def _latency_system(latency):
    controller = pulse.Controller(
        drive_lines=[0, 1], readout_lines=[0, 1], feedback_latency_cycles=latency
    )
    return ladder_target(2, control_system=pulse.ControlSystem(controllers=[controller]))


def _slack(compiled):
    """Each drive's start less the end of the readout it branches on: in
    every kernel here the k-th drive reads the k-th readout."""
    readouts = sorted(
        e.start_cycles + e.duration_cycles for e in compiled.schedule if e.kind == "readout"
    )
    drives = sorted(e.start_cycles for e in compiled.schedule if e.kind == "drive")
    assert len(readouts) == len(drives) + 1
    return [start - end for start, end in zip(drives, readouts[:-1], strict=True)]


@pytest.mark.parametrize("kernel", _CARRIED)
@pytest.mark.parametrize("policy", ["asap", "alap", "rcp", "alap_rcp", None])
def test_a_branch_on_a_carried_measurement_waits_for_its_readout(kernel, policy):
    options = {"schedule": policy} if policy else {}
    compiled = pulse.compile(_CARRIED[kernel], **options).lower(Q[0:2])
    assert "scf.for" in compiled.mlir()
    assert min(_slack(compiled)) >= 0


@pytest.mark.parametrize("kernel", _CARRIED)
@pytest.mark.parametrize("policy", ["rcp", "alap_rcp"])
def test_a_carried_measurement_leaves_the_latency_under_rcp(kernel, policy):
    qpu = _latency_system(400)
    compiled = pulse.compile(_CARRIED[kernel], quantum_system=qpu, schedule=policy).lower(
        qpu.qudits[0:2]
    )
    assert min(_slack(compiled)) >= 400


@pytest.mark.parametrize("policy", ["alap", "alap_rcp", None])
def test_alap_keeps_the_readout_before_the_loop_ahead_of_the_first_branch(policy):
    options = {"schedule": policy} if policy else {}
    events = pulse.compile(_rus_long_arm, **options).lower(Q[0:2]).schedule
    first = sorted(events, key=lambda e: e.start_cycles)[:2]
    assert [(e.kind, e.start_cycles, e.duration_cycles) for e in first] == [
        ("readout", 0, 400),
        ("drive", 400, 360),
    ]
