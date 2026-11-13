# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What classical functions cost a controller, and the limits on it."""

from __future__ import annotations

import math
import re

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import Controller, ControlSystem, Processor, _controller_cost
from cudaq_pulse.classical import Link
from cudaq_pulse.classical import calibration as cal
from cudaq_pulse.controller import (
    ControllerLimitError,
    OpClass,
    OpCounts,
    OpCycles,
    analyze,
    resources,
)

from tests._handles import Q

_CYCLES = {"div": 8, "sqrt": 8, "exp": 12, "log": 12, "trig": 12}
N = 4
_SCALE = 2.0


@pulse.classical(max_latency_ns=100)
def _arithmetic(x: float, y: float, z: float) -> float:
    return x * y + z / y - math.sqrt(x)


@pulse.classical(max_latency_ns=100)
def _by_constants(x: float, y: float) -> float:
    return x / _SCALE + y / (2.0 * math.pi * 1e-9) + x / y


@pulse.classical(max_latency_ns=100)
def _looped(values: list[float]) -> float:
    total = 0.0
    for i in range(N):
        total += values[i] * _SCALE
    return total


@pulse.classical(max_latency_ns=100)
def _branched(x: float) -> float:
    if x > 0:
        y = math.exp(x) * 2.0
    else:
        y = x
    return y


def _square(x):
    return x * x


@pulse.classical(max_latency_ns=100)
def _with_helper(x: float) -> float:
    return _square(x) + _square(x + 1.0)


@pulse.classical(max_latency_ns=100)
def _counted(bits: pulse.Length[pulse.Bits, 50]) -> float:
    return cal.count_ones(bits[0:50]) / 50.0


def test_operations_are_counted_by_class():
    cost = analyze(_arithmetic).cost
    assert cost.ops == OpCounts(add=2, div=1, mul=1, sqrt=1)
    assert cost.runs_on_processor and cost.size == cost.total == 5


def test_constants_fold_and_dividing_by_a_constant_is_a_multiply():
    cost = analyze(_by_constants).cost
    # 2 pi 1e-9 folds; dividing by it or by _SCALE multiplies by a reciprocal.
    assert cost.ops == OpCounts(add=2, mul=2, div=1)


def test_a_loop_multiplies_its_body_by_its_trip_count():
    cost = analyze(_looped.specialize(values=N)).cost
    # Per trip: a load and a multiply, the accumulation, the counter and
    # the branch back.
    assert cost.ops["mul"] == N and cost.ops["load"] == N
    assert cost.ops["branch"] == N and cost.ops["add"] == 2 * N
    assert cost.loop_count == 1 and cost.runs_on_processor


def test_a_branch_costs_its_longer_arm_and_holds_both():
    cost = analyze(_branched).cost
    assert cost.ops == OpCounts(branch=1, compare=1, exp=1, mul=1)
    assert cost.size == cost.total


def test_helpers_are_inlined():
    cost = analyze(_with_helper).cost
    assert cost.ops["mul"] == 2 and cost.ops["add"] == 2
    assert any(h.endswith("_square") for h in cost.helpers)


def test_counting_measurement_bits_is_by_word():
    cost = analyze(_counted).cost
    assert cost.ops["popcount"] == 2 and cost.runs_on_processor


def test_a_function_of_call_lengths_is_analyzed_per_version():
    @pulse.classical(max_latency_ns=100)
    def flip(outcomes: pulse.Bits) -> pulse.Bits:
        return [1 - b for b in outcomes]

    message = (
        "flip's list lengths come from its calls (outcomes), so it has no single cost; analyze"
        " one version, e.g. analyze(flip.specialize(outcomes=3))"
    )
    with pytest.raises(TypeError, match=re.escape(message)):
        analyze(flip)
    row = analyze(flip.specialize(outcomes=3))
    assert (row.function, row.compiled_name) == ("flip", "flip__3")
    assert (analyze(_counted).function, analyze(_counted).compiled_name) == ("_counted",) * 2


def _bad(source_kind):
    if source_kind == "while":

        @pulse.classical(max_latency_ns=10)
        def f(bits: pulse.Bits) -> pulse.Bits:
            i = 0
            while i < 4:
                i += 1
            return (i > 2,)

    elif source_kind == "list":

        @pulse.classical(max_latency_ns=10)
        def f(bits: pulse.Bits) -> float:
            kept = [bits[0], bits[1]]
            return kept[0]

    elif source_kind == "comprehension":

        @pulse.classical(max_latency_ns=10)
        def f(bits: pulse.Bits) -> float:
            return max([b for b in bits])

    elif source_kind == "raise":

        @pulse.classical(max_latency_ns=10)
        def f(x: float) -> float:
            if x < 0:
                raise ValueError("negative")
            return x

    elif source_kind == "unknown call":

        @pulse.classical(max_latency_ns=10)
        def f(bits: pulse.Bits) -> float:
            return sum(bits)

    elif source_kind == "unbounded loop":

        @pulse.classical(max_latency_ns=10)
        def f(x: float) -> float:
            total = 0.0
            for _ in range(int(x)):
                total += 1.0
            return total

    else:

        def recurse(x):
            return x if x < 1 else recurse(x / 2)

        @pulse.classical(max_latency_ns=10)
        def f(x: float) -> float:
            return recurse(x)

    # A function of call lengths is analyzed one version at a time.
    return (
        f.specialize(bits=4)
        if source_kind in ("while", "list", "comprehension", "unknown call")
        else f
    )


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("while", "while loop"),
        ("list", "allocation"),
        ("comprehension", "comprehension"),
        ("raise", "exception"),
        ("unknown call", "cannot be followed"),
        ("unbounded loop", "trip count"),
        ("recursion", "recursion"),
    ],
)
def test_what_keeps_a_body_out_of_controller_form_is_named(kind, message):
    cost = analyze(_bad(kind)).cost
    assert not cost.runs_on_processor
    assert any(message in v for v in cost.violations), cost.violations


def test_the_protocol_estimators_are_in_controller_form():
    for helper in (
        cal.adaptive_t1_update,
        cal.adaptive_t1_wait,
        cal.binary_search_update,
        cal.binary_search_wait,
        cal.binary_search_phase,
        cal.decay_time,
        cal.sparse_phase,
        cal.pi_train_amplitude,
        cal.ramsey_detuning,
    ):
        cost = _controller_cost.analyze_callable(helper, {})
        assert cost.runs_on_processor, (helper.__name__, cost.violations)
    # The estimators of the previous release raise on bad input.
    assert not _controller_cost.analyze_callable(cal.t1_update, {}).runs_on_processor


def test_compute_time_follows_the_cycle_costs():
    row = analyze(_arithmetic)
    # On no processor; mul 1 + add 2 at one cycle, div 8, sqrt 8: 19 cycles.
    assert row.compute_cycles is None and row.compute_ns is None and row.margin_ns is None
    assert row.cost.processor_cycles(OpCycles(**_CYCLES), True) == 19
    assert row.state_bytes == 3 * 8 + 8  # three arguments, one result
    assert row.request_bytes == 24 + 24 and row.answer_bytes == 24 + 8


def test_unknown_operation_classes_are_rejected():
    with pytest.raises(TypeError, match="fma"):
        OpCycles(fma=1)  # type: ignore[call-arg]
    assert set(OpClass) >= set(_CYCLES)
    a = Processor(cycles_per_op=OpCycles(div=8))
    assert a == Processor(cycles_per_op=OpCycles(div=8)) and hash(a) == hash(
        Processor(cycles_per_op=OpCycles(div=8))
    )
    assert a != Processor(cycles_per_op=OpCycles(div=9))
    assert a.as_dict()["cycles_per_op"] == {"div": 8}


@pulse.classical(max_latency_ns=40, link="controller")
def _update(bit: pulse.Bits, x: float) -> float:
    return math.exp(x) * 0.5 + bit[0]


@pulse.classical(max_latency_ns=40, link="gpu")
def _remote(bit: pulse.Bits, x: float) -> float:
    total = 0.0
    while total < x:
        total += 1.0
    return total + bit[0]


@pulse.kernel
def _rounds(q, n: int):
    d = get_drive_line(q)
    r = get_readout_line(q)
    x = 0.1
    for _ in range(n):
        drive(d, square(40, x))
        sync(d, r)
        m = readout(r, square(200, 0.05))
        x = _update.submit([m], x).result()
        y = _remote.submit([m], x).result()
        sync(d, r)
        drive(d, square(40, y * 0.0))


def _machine(processor=None, **limits):
    links = {"controller": Link(fixed_ns=4, feedback_ns=8), "gpu": Link(fixed_ns=500)}
    if processor is not None:
        controller = Controller(processors=[Processor(**processor)], **limits)
        return pulse.ControlSystem(controllers=[controller], links=links)
    return ControlSystem(controllers=[Controller(**limits)] if limits else [], links=links)


def _lower(machine, n=3):
    return pulse.compile(_rounds, control_system=machine).lower(Q[0], n)


def test_a_function_that_computes_past_its_budget_fails_to_compile():
    # exp at 12 cycles, a multiply, a load and an add, at 4 ns per cycle.
    with pytest.raises(ControllerLimitError, match="_update__1 computes for about 60 ns"):
        _lower(_machine({"cycles_per_op": OpCycles(**_CYCLES), "clock_ghz": 0.25}))
    _lower(_machine({"cycles_per_op": OpCycles(**_CYCLES), "clock_ghz": 1.0}))


def test_only_functions_on_a_processor_are_checked():
    # _remote's while loop runs on the GPU link: reported, not checked.
    report = resources(_lower(_machine({"fixed_cost_only": True})))
    remote = next(row for row in report.functions if row.compiled_name == "_remote__1")
    assert remote.function == "_remote"
    assert not remote.on_controller and not remote.cost.runs_on_processor
    assert remote.compute_ns is None
    assert report.violations == ()
    with pytest.raises(ControllerLimitError, match="_remote__1 is not fixed-cost"):
        _lower(_machine({"fixed_cost_only": True, "links": ("controller", "gpu")}))


def test_per_shot_limits_count_calls_and_bytes():
    lowered = _lower(_machine(), n=5)
    report = resources(lowered)
    update = next(row for row in report.functions if row.compiled_name == "_update__1")
    assert update.calls_per_shot == 5 and report.calls_per_shot == 10
    per_call = update.request_bytes + update.answer_bytes  # 24 + 9, 24 + 8
    assert per_call == 65
    with pytest.raises(ControllerLimitError, match="makes 5 calls to it; it takes 4"):
        _lower(_machine({"max_calls_per_shot": 4}), n=5)
    with pytest.raises(ControllerLimitError, match=f"take {5 * per_call} bytes"):
        _lower(_machine({"max_bytes_per_shot": 5 * per_call - 1}), n=5)
    with pytest.raises(ControllerLimitError, match="bytes of state"):
        _lower(_machine({"max_state_bytes_per_function": 16}))
    with pytest.raises(ControllerLimitError, match="holds 4 operations"):
        _lower(_machine({"max_ops_per_function": 3}))


def test_the_report_is_a_table_of_functions_and_links():
    report = resources(_lower(_machine({"cycles_per_op": OpCycles(**_CYCLES), "clock_ghz": 1.0})))
    text = str(report)
    assert "_update" in text and "_remote" in text and "controller" in text
    assert "Links, per shot" in text and "Per shot: 6 calls" in text
    controller_link = next(u for u in report.links if u.link == "controller")
    assert controller_link.calls_per_shot == 3
    assert 1 <= controller_link.peak_slots_used <= controller_link.slots
    update = next(row for row in report.functions if row.compiled_name == "_update__1")
    # exp 12, a multiply, a load and an add, at 1 ns per cycle.
    assert update.on_controller and update.compute_ns == pytest.approx(15.0)


def test_without_compute_limits_bodies_are_not_read(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("a body was read")

    monkeypatch.setattr(_controller_cost, "analyze_callable", fail)
    _lower(_machine(max_pulses=10_000))
    _lower(_machine())
