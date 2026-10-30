# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Scheduling a retained scf.for gives the same timeline as unrolling it."""

from __future__ import annotations

import pytest
from cudaq_pulse._native._cudaq_pulse_native import MLIRPipeline
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.property_settings import examples

_DURATIONS = (20, 40, 60, 100)
_LINE = "!pulse.drive_line"
_TONE = "!pulse.tone"


class _Writer:
    """Writes one function, renaming each line and tone as ops advance it."""

    def __init__(self, n_lines: int):
        self.n = n_lines
        self.lines: list[str] = []
        self.counter = 0
        self.text = ["func.func @main() {"]
        for i in range(n_lines):
            self.emit(f"%q{i} = pulse.qudit_alloc {{qubit = {i} : i64}} : !pulse.qref")
            self.emit(
                f"%d{i}_0, %t{i}_0 = pulse.get_drive_line %q{i} {{qubit = {i} : i64}} : "
                f"(!pulse.qref) -> ({_LINE}, {_TONE})"
            )
        self.tones = [f"%t{i}_0" for i in range(n_lines)]
        self.lines = [f"%d{i}_0" for i in range(n_lines)]
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
            _, i, d = op
            line, tone = self.fresh("d"), self.fresh("t")
            self.emit(
                f"{line}, {tone} = pulse.drive {self.lines[i]}, %wf{d}, {self.tones[i]} : "
                f"{_LINE}, !pulse.waveform, {_TONE} -> {_LINE}, {_TONE}"
            )
            self.lines[i], self.tones[i] = line, tone
        elif kind == "phase":
            _, i, value = op
            tone = self.fresh("t")
            angle = f"%c{tone[1:]}"
            self.emit(f"{angle} = arith.constant {float(value)!r} : f64")
            self.emit(
                f"{tone} = pulse.shift_phase {self.tones[i]}, {angle} : {_TONE}, f64 -> {_TONE}"
            )
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


def _programs(n_lines, pre, trips, body, post):
    rolled = _Writer(n_lines)
    unrolled = _Writer(n_lines)
    for op in [*pre, ("loop", trips, body), *post]:
        rolled.op(op)
        unrolled.op(op, unroll=True)
    return rolled.finish(), unrolled.finish()


def _ops(n_lines):
    line = st.integers(0, n_lines - 1)
    drive = st.tuples(st.just("drive"), line, st.sampled_from(_DURATIONS))
    wait = st.tuples(st.just("wait"), line, st.sampled_from(_DURATIONS))
    options = [drive, wait]
    if n_lines > 1:
        sync = st.tuples(
            st.just("sync"),
            st.lists(st.integers(0, n_lines - 1), min_size=2, max_size=n_lines, unique=True),
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
