# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Declaring classical functions and calling them from kernels."""

from __future__ import annotations

import math
import re

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import classical
from cudaq_pulse.classical import (
    IQ,
    Bits,
    Byte,
    Bytes,
    Float,
    Floats,
    Integer,
    Integers,
    Levels,
    Link,
)
from cudaq_pulse.compiler import MachineModel

from tests._handles import Q


@pulse.classical(args=(Bits(1),), results=(Bits(1),), max_latency_ns=500)
def echo(bits):
    return [bits[0]]


@pulse.classical(args=(Float(),), results=(Float(),), max_latency_ns=100, link=1)
def double(x):
    return 2 * x


@pulse.classical(args=(Levels(1),), results=(Integer(),), max_latency_ns=200)
def choose(levels):
    return levels[0]


_slow = pulse.classical(
    lambda: 1.0, name="slow", results=(Float(),), max_latency_ns=1, latency_mode="blocking"
)
_sink = classical.external("sink", args=(Float(),), max_latency_ns=1)


def _mlir(kernel, *args, **options):
    return pulse.compile(kernel, **options).lower(*args).mlir()


# ── Declaration ──────────────────────────────────────────────────────────────


def test_module_is_the_decorator():
    assert callable(pulse.classical)
    assert pulse.classical is classical
    fn = pulse.classical(lambda: None, name="anon", max_latency_ns=1)
    assert isinstance(fn, classical.ClassicalFunction)
    assert classical.function(lambda: None, name="anon", max_latency_ns=1).name == "anon"


def test_function_id_is_fnv1a_as_cuda_q_hashes_handler_names():
    # The ids CUDA-QX pins for its decoder messages.
    assert classical.function_id("enqueue_syndromes") == 0x7ED8BE82
    assert classical.function_id("get_corrections") == 0x882D5BA1
    assert classical.function_id("reset_decoder") == 0x977A59CF
    assert echo.id == classical.function_id("echo")


def test_declaration_keeps_the_schema_and_timing():
    fn = pulse.classical(
        lambda s: s,
        name="fit",
        args=(IQ(4), Floats(3, 32), Integers(2), Bytes(5), Byte, Float(32), Integer(32)),
        results=(Floats(2),),
        min_latency_ns=10,
        nominal_latency_ns=20,
        max_latency_ns=30,
        latency_mode="timeout",
        timeout_ns=40,
        submit="lazy",
        deadline_ns=50,
        link=3,
        id=7,
    )
    assert [a.descriptor for a in fn.args] == [
        "iq:4",
        "f32[3]",
        "i32[2]",
        "u8[5]",
        "u8",
        "f32",
        "i32",
    ]
    assert (fn.min_latency_ns, fn.nominal_latency_ns, fn.max_latency_ns) == (10, 20, 30)
    assert (fn.latency_mode, fn.timeout_ns, fn.submit_policy, fn.deadline_ns) == (
        "timeout",
        40,
        "lazy",
        50,
    )
    assert (fn.link, fn.id, fn.custom_id) == (3, 7, True)
    assert fn.nominal_latency_ns == 20
    assert "fit" in repr(fn)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"args": (object(),)}, "must be a type such as"),
        ({"args": (Bits(1),) * 9}, "at most 8 arguments"),
        ({"results": (Bits(1),) * 5}, "at most 4 results"),
        ({"max_latency_ns": -1}, "finite number >= 0"),
        ({"max_latency_ns": math.inf}, "finite number >= 0"),
        ({"min_latency_ns": 20, "max_latency_ns": 10}, "min <= nominal <= max"),
        ({"latency_mode": "soon"}, "latency_mode must be"),
        ({"latency_mode": "timeout"}, "timeout_ns goes with"),
        ({"timeout_ns": 5}, "timeout_ns goes with"),
        ({"latency_mode": "timeout", "timeout_ns": 0}, "timeout_ns must be positive"),
        ({"submit": "late"}, "submit must be"),
        ({"deadline_ns": 0}, "deadline_ns must be positive"),
        ({"link": -1}, "link must be"),
        ({"link": True}, "link must be"),
        ({"id": 2**32}, "32-bit"),
        ({"name": "not valid"}, "identifier"),
    ],
)
def test_declaration_errors(options, message):
    options = {"max_latency_ns": 1, **options}
    with pytest.raises((ValueError, TypeError), match=message):
        pulse.classical(lambda *a: None, name=options.pop("name", "f"), **options)


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: Bits(0), "positive whole number"),
        (lambda: Levels(1.5), "positive whole number"),
        (lambda: IQ(True), "positive whole number"),
        (lambda: Float(16), "64 or 32 bits"),
        (lambda: Floats(2, 8), "64 or 32 bits"),
        (lambda: Integer(8), "64 or 32 bits"),
    ],
)
def test_type_errors(make, message):
    with pytest.raises(ValueError, match=message):
        make()


def test_types_compare_by_descriptor():
    assert Bits(3) == Bits(3) and Bits(3) != Levels(3)
    assert len({Floats(2), Floats(2), Floats(2, 32)}) == 2
    assert (IQ(2).answer_count, IQ(2).wire_bytes) == (4, 16)
    assert Bits(9).wire_bytes == 2


def test_link_validation():
    assert Link().max_in_flight == 64
    for bad in (
        {"fixed_ns": -1},
        {"bytes_per_ns": math.nan},
        {"max_in_flight": 0},
        {"slot_bytes": 24},
    ):
        with pytest.raises(ValueError):
            Link(**bad)


# ── Calling outside a kernel ─────────────────────────────────────────────────


def test_calling_outside_a_kernel_runs_the_body():
    assert double(1.5) == 3.0
    answer = double.submit(2.0)
    assert answer.done() and answer.result() == 4.0
    assert answer.result(timeout=20) == (True, 4.0)
    with pytest.raises(TypeError, match="takes 1 arguments"):
        double(1, 2)


def test_external_function_has_no_body():
    fn = classical.external("remote", args=(Float(),), results=(), max_latency_ns=5)
    assert fn.implementation is None and fn.name == "remote"
    with pytest.raises(TypeError, match="no reference implementation"):
        fn(1.0)
    with pytest.raises(TypeError, match="needs a name"):
        classical.ClassicalFunction(None, max_latency_ns=1)


# ── Tracing ──────────────────────────────────────────────────────────────────


@pulse.kernel
def _reset(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    m = readout(r, square(200, 0.05))
    fix = echo.submit([m]).result()
    if fix[0]:
        drive(d, square(40, math.pi / 20))


def test_submit_and_result_become_classical_ops():
    text = _mlir(_reset, Q[0])
    assert re.search(r"classical.callable @echo \{.*args = \[\"bits:1\"\]", text)
    assert "classical.get_link {device = 0 : i64}" in text
    assert re.search(r"classical.submit %\S+, @echo\(%result\)", text)
    assert re.search(r"classical.wait %future, @echo .*: i1", text)
    assert re.search(r"scf.if %\d+ -> ", text)
    assert "pulse.is_set" not in text


def test_answers_set_pulse_parameters():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        a = double.submit(0.25).result()
        shift_phase(d.tone, a)
        shift_frequency(d.tone, 1e6 * a)
        drive(d, gaussian(40, a * 0.5 + 0.1, 10.0))

    text = _mlir(k, Q[0])
    assert "pulse.shift_phase" in text and "pulse.shift_frequency" in text
    assert "arith.mulf" in text and "arith.addf" in text


@pytest.mark.parametrize("compare", ["== 2", "!= 2", "< 2", "<= 2", "> 2", ">= 2"])
def test_branch_on_a_comparison_of_an_integer_answer(compare):
    namespace = {}
    exec(  # noqa: S102 - builds one kernel per comparison operator
        "def body(q):\n"
        "    d = get_drive_line(q)\n"
        "    r = get_readout_line(q)\n"
        "    m = readout(r, square(100, 0.05))\n"
        "    setting = choose.submit([m]).result()\n"
        f"    if setting {compare}:\n"
        "        drive(d, square(40, 0.1))\n",
        {"choose": choose},
        namespace,
    )
    text = _mlir(pulse.kernel(namespace["body"]), Q[0])
    assert "arith.cmpi" in text


def test_branch_on_a_float_answer_and_its_negation():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        x = double.submit(0.3).result()
        if x > 0.5:
            drive(d, square(40, 0.1))
        if not x:
            drive(d, square(40, 0.2))

    text = _mlir(k, Q[0])
    assert text.count("arith.cmpf") == 2


def test_not_on_an_answer_bit():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        m = readout(r, square(100, 0.05))
        fix = echo.submit([m]).result()
        if not fix[0]:
            drive(d, square(40, 0.1))

    assert "scf.if" in _mlir(k, Q[0])


def test_link_is_threaded_through_a_rolled_loop():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        for _ in range(50):
            m = readout(r, square(100, 0.05))
            if echo.submit([m]).result()[0]:
                drive(d, square(40, 0.1))

    text = _mlir(k, Q[0], unroll_threshold=0)
    assert re.search(r"scf.for .*iter_args\(.*\) -> \(.*!classical.link", text)
    assert text.count("classical.get_link") == 1


def test_link_is_threaded_through_branch_arms():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        m = readout(r, square(100, 0.05))
        if m:
            double.submit(1.0)
        else:
            drive(d, square(40, 0.1))
        a = double.submit(2.0).result()
        drive(d, square(40, a))

    text = _mlir(k, Q[0])
    assert re.search(r"scf.if .* -> \(.*!classical.link", text)
    assert text.count("classical.get_link") == 1


def test_a_kernel_with_its_own_schedule_returns_a_future():
    @pulse.kernel(schedule="alap")
    def measure_and_submit(r):
        m = readout(r, square(300, 0.05))
        return echo.submit([m])

    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        answer = measure_and_submit(r)
        if answer.result()[0]:
            drive(d, square(40, 0.1))

    low = pulse.compile(k).lower(Q[0])
    report = low.classical_report()
    assert len(report) == 1 and report[0].taken_cycles == report[0].due_cycles
    flat = pulse.compile(k, flatten_schedules=True).lower(Q[0]).classical_report()
    assert flat[0].due_cycles == report[0].due_cycles


def test_two_links_get_two_link_values():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        m = readout(r, square(100, 0.05))
        a = echo.submit([m])
        b = double.submit(0.5)
        if a.result()[0]:
            drive(d, square(40, b.result()))

    text = _mlir(k, Q[0])
    assert "classical.get_link {device = 0 : i64}" in text
    assert "classical.get_link {device = 1 : i64}" in text


def test_result_twice_is_one_wait():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        a = double.submit(0.5)
        drive(d, square(40, a.result()))
        drive(d, square(40, a.result()))

    assert _mlir(k, Q[0]).count("classical.wait") == 1


def test_timeout_at_the_call_gives_an_ok_bit():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        ok, value = double.submit(0.5).result(timeout=100)
        if ok:
            drive(d, square(40, value))

    text = _mlir(k, Q[0])
    assert re.search(r"classical.wait .*timeout_ns = 5.0+e\+01.*: i1, f64", text)


def test_several_results_come_back_as_a_tuple():
    fit = pulse.classical(
        lambda p: (0.0, [0.0, 0.0]),
        name="fit",
        args=(IQ(2),),
        results=(Float(), Floats(2)),
        max_latency_ns=10,
    )

    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        a = readout(r, square(100, 0.05))
        b = readout(r, square(100, 0.05))
        shift, (x, y) = fit.submit([a, b]).result()
        drive(d, square(40, shift + x + y))

    assert re.search(r"classical.wait .*: f64, f64, f64", _mlir(k, Q[0]))


# ── Errors ───────────────────────────────────────────────────────────────────


def _compile_error(body, match):
    with pytest.raises(pulse.CompilationError, match=match):
        pulse.compile(pulse.kernel(body)).lower(Q[0])


def test_a_duration_cannot_come_from_an_answer():
    def square_duration(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        n = choose.submit([readout(r, square(10, 0.1))]).result()
        drive(d, square(n, 0.1))

    def gaussian_duration(q):
        d = get_drive_line(q)
        x = double.submit(0.5).result()
        drive(d, gaussian(x * 10, 0.1, 4.0))

    # A wait may take its length from an answer (a variable wait); a
    # waveform's length is structural.
    _compile_error(square_duration, "a duration cannot depend on the answer")
    _compile_error(gaussian_duration, "a duration cannot depend on the answer")


def test_argument_errors():
    def wrong_count(q):
        echo.submit()

    def number_for_bits(q):
        echo.submit([1])

    def measurement_for_float(q):
        r = get_readout_line(q)
        double.submit(readout(r, square(10, 0.1)))

    def too_many_measurements(q):
        r = get_readout_line(q)
        m = readout(r, square(10, 0.1))
        echo.submit([m, m])

    def keyword(q):
        double.submit(x=1.0)

    _compile_error(wrong_count, "takes 1 arguments, got 0")
    _compile_error(number_for_bits, "takes measurement results")
    _compile_error(measurement_for_float, "takes numbers")
    _compile_error(too_many_measurements, "takes 1 measurement result, got 2")
    _compile_error(keyword, "positional arguments only")


def test_misuse_errors():
    def call_directly(q):
        double(1.0)

    def no_answer(q):
        _sink.submit(1.0).result()

    def other_timeout(q):
        a = double.submit(1.0)
        a.result()
        a.result(timeout=10)

    def blocking_timeout(q):
        _slow.submit().result(timeout=10)

    def symbolic_timeout(q, t: float):
        double.submit(1.0).result(timeout=t)

    def taken_in_a_loop(q):
        d = get_drive_line(q)
        a = double.submit(1.0)
        for _ in range(3):
            drive(d, square(40, a.result()))

    def taken_outside_its_arm(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        a = double.submit(1.0)
        if readout(r, square(10, 0.1)):
            b = double.submit(2.0)
            drive(d, square(40, a.result()))
        else:
            b = double.submit(3.0)
        b.result()

    _compile_error(call_directly, r"call double.submit\(...\)")
    _compile_error(no_answer, "returns nothing")
    _compile_error(other_timeout, "an answer is taken once")
    _compile_error(blocking_timeout, "has no timeout")
    with pytest.raises(pulse.CompilationError, match="known when compiling"):
        pulse.compile(pulse.kernel(symbolic_timeout)).lower(Q[0], 1.0)
    with pytest.raises(pulse.CompilationError, match="another loop iteration"):
        pulse.compile(pulse.kernel(taken_in_a_loop), unroll_threshold=0).lower(Q[0])
    _compile_error(taken_outside_its_arm, "would depend on a measurement result|another loop")


def test_an_answer_cannot_leave_its_kernel_with_its_own_schedule():
    @pulse.kernel(schedule="asap")
    def inner(q):
        return double.submit(1.0).result()

    def outer(q):
        d = get_drive_line(q)
        a = inner(q)
        drive(d, square(40, a))

    _compile_error(outer, "used outside the loop, branch or kernel")


def test_two_functions_with_one_name():
    a = pulse.classical(lambda: 1.0, name="same", results=(Float(),), max_latency_ns=1)
    b = pulse.classical(lambda: 2.0, name="same", results=(Float(),), max_latency_ns=1)

    def k(q):
        a.submit()
        b.submit()

    _compile_error(k, "two different classical functions are named 'same'")


# ── Links and the report ─────────────────────────────────────────────────────


def test_links_reach_the_module_and_the_schedule():
    machine = MachineModel(links={0: Link(fixed_ns=50, bytes_per_ns=1, feedback_ns=100)})
    low = pulse.compile(_reset, machine=machine).lower(Q[0])
    assert "fixed_ns = 5.000000e+01" in low.mlir()
    (call,) = low.classical_report()
    clock = low.target.clock_ghz
    # 24 header bytes + 1 payload byte at 1 byte/ns, plus the fixed 50 ns.
    assert call.submit_duration_cycles == math.ceil((50 + 25) * clock)
    assert call.due_cycles == call.submit_cycles + call.submit_duration_cycles + math.ceil(
        (500 + 100) * clock
    )
    assert hash(machine) == hash(
        MachineModel(links={0: Link(fixed_ns=50, bytes_per_ns=1, feedback_ns=100)})
    )
    with pytest.raises(TypeError, match="read-only"):
        machine.links[1] = Link()  # type: ignore[index]
    with pytest.raises(ValueError, match="link ids"):
        MachineModel(links={-1: Link()})
    with pytest.raises(TypeError, match="must be a Link"):
        MachineModel(links={0: 5})  # type: ignore[dict-item]


def test_report_and_schedule_events():
    low = pulse.compile(_reset, schedule="asap").lower(Q[0])
    (call,) = low.classical_report()
    clock = low.target.clock_ghz
    assert call.function == "echo" and call.link == 0 and call.iteration == ()
    assert call.submit_cycles == 200 and call.submit_duration_cycles == 1
    assert call.due_cycles == 201 + math.ceil(500 * clock)
    assert call.taken_cycles == call.due_cycles == call.first_use_cycles
    assert call.slack_cycles == 0 and call.deadline_cycles is None and call.epoch == 0
    kinds = [(e.kind, e.link) for e in low.schedule if e.link is not None]
    assert kinds == [("submit", 0), ("answer", 0)]
    assert all(e.link is None for e in low.schedule if e.kind == "drive")


def test_report_entries_are_what_their_constructor_builds():
    from cudaq_pulse.compiler import ClassicalCall

    assert "__post_init__" not in vars(ClassicalCall)
    (call,) = pulse.compile(_reset, schedule="asap").lower(Q[0]).classical_report()
    fields = ClassicalCall.__dataclass_fields__
    built = ClassicalCall(**{name: getattr(call, name) for name in fields})
    assert built == call and hash(built) == hash(call) and repr(built) == repr(call)
    with pytest.raises(AttributeError):
        call.link = 1  # type: ignore[misc]


def test_programs_without_calls_are_unchanged():
    @pulse.kernel
    def plain(q):
        d = get_drive_line(q)
        drive(d, gaussian(40, 0.1, 10.0))

    text = _mlir(plain, Q[0])
    assert "classical" not in text
    assert pulse.compile(plain).lower(Q[0]).classical_report() == []
