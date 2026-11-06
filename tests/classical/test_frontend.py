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
import typing
from collections.abc import Sequence

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import classical
from cudaq_pulse.classical import Link, wire
from cudaq_pulse.compiler import MachineModel, _dispatch_tables

from tests._handles import Q


@pulse.classical(max_latency_ns=500)
def echo(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]


@pulse.classical(max_latency_ns=100, link=1)
def double(x: float) -> float:
    return 2 * x


@pulse.classical(max_latency_ns=200)
def choose(levels: list[int]) -> int:
    return levels[0]


@pulse.classical(name="slow", max_latency_ns=1, latency_mode="blocking")
def _slow() -> float:
    return 1.0


@classical.external(name="sink", max_latency_ns=1)
def _sink(x: float) -> None: ...


def _mlir(kernel, *args, **options):
    return pulse.compile(kernel, **options).lower(*args).mlir()


# ── Declaration ──────────────────────────────────────────────────────────────


def test_module_is_the_decorator():
    assert callable(pulse.classical)
    assert pulse.classical is classical

    def nothing() -> None:
        pass

    fn = pulse.classical(nothing, name="anon", max_latency_ns=1)
    assert isinstance(fn, classical.ClassicalFunction)
    assert classical.function(nothing, name="anon", max_latency_ns=1).name == "anon"
    assert pulse.Bits is classical.Bits


def test_function_id_is_fnv1a_as_cuda_q_hashes_handler_names():
    # The keys CUDA-QX pins for its decoder messages.
    assert classical.function_id("enqueue_syndromes") == 0x7ED8BE82
    assert classical.function_id("get_corrections") == 0x882D5BA1
    assert classical.function_id("reset_decoder") == 0x977A59CF


def test_declaration_keeps_the_schema_and_timing():
    def fit(
        points: pulse.Length[list[complex], 4],
        xs: pulse.Length[list[float], 3],
        counts: pulse.Length[list[int], 2],
        n: int,
    ) -> pulse.Length[list[float], 2]:
        return xs[:2]

    fn = pulse.classical(
        fit,
        min_latency_ns=10,
        nominal_latency_ns=20,
        max_latency_ns=30,
        latency_mode="timeout",
        timeout_ns=40,
        submit="lazy",
        deadline_ns=50,
        link=3,
    )
    assert [a.descriptor for a in fn.args] == ["iq:4", "f64[3]", "i32[2]", "i64"]
    assert [r.descriptor for r in fn.results] == ["f64[2]"]
    assert (fn.min_latency_ns, fn.nominal_latency_ns, fn.max_latency_ns) == (10, 20, 30)
    assert (fn.latency_mode, fn.timeout_ns, fn.submit_policy, fn.deadline_ns) == (
        "timeout",
        40,
        "lazy",
        50,
    )
    assert fn.link == 3 and not hasattr(fn, "id")
    assert fn.nominal_latency_ns == 20
    assert "fit" in repr(fn)


def test_hints_map_to_wire_types():
    @pulse.classical(max_latency_ns=1)
    def scalars(x: float, n: int, point: complex) -> tuple[float, int, bool]:
        return x, n, bool(point.real)

    assert [a.descriptor for a in scalars.args] == ["f64", "i64", "iq:1"]
    assert [r.descriptor for r in scalars.results] == ["f64", "i64", "bits:1"]
    assert all(a.is_scalar for a in scalars.args) and scalars.results[2].is_scalar


@pytest.mark.parametrize(
    ("options", "message"),
    [
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
        ({"id": 0x51}, "unexpected keyword argument 'id'"),
        ({"name": "not valid"}, "identifier"),
    ],
)
def test_declaration_errors(options, message):
    def f() -> None:
        pass

    options = {"max_latency_ns": 1, **options}
    with pytest.raises((ValueError, TypeError), match=message):
        pulse.classical(f, **options)


def test_declaring_takes_a_def_with_hints():
    with pytest.raises(TypeError, match="a function defined with def"):
        pulse.classical(lambda x: x, name="f", max_latency_ns=1)

    def f() -> None:
        pass

    with pytest.raises(TypeError, match="unexpected keyword argument 'args'"):
        pulse.classical(f, max_latency_ns=1, args=(), results=())  # remnant-ok


def _nine(
    a: float, b: float, c: float, d: float, e: float, f: float, g: float, h: float, i: float
) -> None:
    pass


def _five() -> tuple[float, float, float, float, float]:
    return 0.0, 0.0, 0.0, 0.0, 0.0


def _no_hint(x) -> float:  # noqa: ANN001
    return x


def _no_return(x: float):  # noqa: ANN202
    return x


def _flag(x: bool) -> float:
    return float(x)


def _measurement(m: pulse.ops.MeasurementResult) -> float:
    return float(m)


def _dict(x: dict[str, int]) -> float:
    return 0.0


def _bools(x: list[bool]) -> float:
    return 0.0


def _complex_out(x: float) -> complex:
    return complex(x)


def _keyword(*, x: float) -> float:
    return x


def _scalar_length(alpha: pulse.Length[float, 3]) -> float:
    return alpha


def _zero(index: pulse.Length[pulse.Bits, 0]) -> float:
    return 0.0


# Lengths that aren't whole numbers, set after the def: in a signature a
# checker would read the strings as names.
def _named_result(outcomes: pulse.Bits) -> pulse.Bits:
    return outcomes


_named_result.__annotations__["return"] = pulse.Length[pulse.Bits, "outcomes"]  # remnant-ok


def _named_parameter(x: pulse.Bits) -> float:
    return 0.0


_named_parameter.__annotations__["x"] = pulse.Length[pulse.Bits, "x"]  # remnant-ok


def _numbers_flag(x: list[int]) -> float:
    return 0.0


_numbers_flag.__annotations__["x"] = pulse.Length[list[int], 4, "numbers"]  # remnant-ok


def _no_length(x: list[int]) -> float:
    return 0.0


_no_length.__annotations__["x"] = pulse.Length[list[int], "numbers"]  # remnant-ok


def _true_length(x: pulse.Length[pulse.Bits, True]) -> float:
    return 0.0


def _two_lengths(x: pulse.Length[pulse.Bits, 2, 3]) -> float:
    return 0.0


def _scalar_result(x: pulse.Bits) -> pulse.Length[float, 2]:
    return 0.0


@pytest.mark.parametrize(
    ("fn", "message"),
    [
        (_nine, "at most 8 arguments"),
        (_five, "at most 4 results"),
        (_no_hint, "parameter 'x' needs a type"),
        (_no_return, "annotate what it returns"),
        (_flag, "a measured bit is pulse.Bits"),
        (_measurement, "say how to read it"),
        (_dict, "can't be dict"),
        (_bools, "lists hold float, int or complex"),
        (_complex_out, "return tuple\\[float, float\\] instead"),
        (_keyword, "only positional parameters"),
        (
            _scalar_length,
            "parameter 'alpha' is float; pulse.Length applies to pulse.Bits and lists",
        ),
        (
            _zero,
            re.escape("pulse.Length[pulse.Bits, 0]: a length is a positive whole number, got 0"),
        ),
        (
            _named_result,
            re.escape(
                'pulse.Length[pulse.Bits, "outcomes"]: a length is'  # remnant-ok
                " a positive whole number, got 'outcomes'; a list result without one takes"
                " its length from the body: -> pulse.Bits"
            ),
        ),
        (
            _named_parameter,
            re.escape(
                "got 'x'; a list parameter without one takes its length from the call: pulse.Bits"
            ),
        ),
        (
            _numbers_flag,
            re.escape(
                'pulse.Length[list[int], 4, "numbers"]: state one length, such as'  # remnant-ok
                " pulse.Length[list[int], 8]"
            ),
        ),
        (_no_length, "a length is a positive whole number, got 'numbers'"),
        (_true_length, "a length is a positive whole number, got True"),
        (_two_lengths, "state one length"),
        (_scalar_result, "its result is float; pulse.Length applies"),
    ],
)
def test_hint_errors_are_raised_where_the_function_is_used(fn, message):
    declared = pulse.classical(fn, max_latency_ns=1)
    with pytest.raises(TypeError, match=message):
        len(declared.args)
    with pytest.raises(TypeError, match=message):  # kept, raised again
        wire.schema(declared)

    def k(q):
        declared.submit()

    _compile_error(k, message)


def test_sequence_hints_are_lists():
    @pulse.classical(max_latency_ns=1)
    def total(
        xs: pulse.Length[Sequence[float], 2], counts: pulse.Length[Sequence[int], 3]
    ) -> float:
        return sum(xs) + sum(counts)

    assert [a.descriptor for a in total.args] == ["f64[2]", "i32[3]"]


def test_typing_sequence_hints_are_lists():
    @pulse.classical(max_latency_ns=1)
    def total(
        xs: pulse.Length[typing.Sequence[float], 2],  # noqa: UP006
        counts: pulse.Length[typing.Sequence[int], 3],  # noqa: UP006
    ) -> pulse.Length[typing.Sequence[int], 2]:  # noqa: UP006
        return [len(xs), len(counts)]

    assert [a.descriptor for a in total.args] == ["f64[2]", "i32[3]"]
    assert [r.descriptor for r in total.results] == ["i32[2]"]

    @pulse.classical(max_latency_ns=1)
    def scaled(xs: typing.Sequence[float]) -> typing.Sequence[float]:  # noqa: UP006
        return [2 * x for x in xs]

    assert scaled.name == "scaled"


def test_stated_lengths():
    @pulse.classical(max_latency_ns=1)
    def scale(
        counts: pulse.Length[list[int], 3],
        outcomes: pulse.Length[pulse.Bits, 2],
    ) -> tuple[pulse.Length[list[int], 3], pulse.Length[pulse.Bits, 4], float]:
        return counts, [0] * 4, 0.0

    assert [a.descriptor for a in scale.args] == ["i32[3]", "bits:2"]
    assert [r.descriptor for r in scale.results] == ["i32[3]", "bits:4", "f64"]
    assert scale._shape is None and scale.name == "scale"

    @pulse.classical(max_latency_ns=1)
    def flip(outcomes: pulse.Bits) -> pulse.Bits:
        return [1 - b for b in outcomes]

    assert [r.descriptor for r in flip.specialize(outcomes=5).results] == ["bits:5"]

    @pulse.classical(max_latency_ns=1)
    def pinned(syndrome: pulse.Length[pulse.Bits, 6]) -> pulse.Bits:
        return syndrome[1:]

    assert [r.descriptor for r in pinned.results] == ["bits:5"]


def test_a_length_the_body_cant_give_is_rejected():
    @pulse.classical(max_latency_ns=1)
    def fired(syndrome: pulse.Length[pulse.Bits, 6]) -> list[int]:
        return [i for i, s in enumerate(syndrome) if s]

    with pytest.raises(TypeError) as error:
        len(fired.results)
    assert "depends on the values" in str(error.value)
    assert str(error.value).endswith(
        "Write it in plain Python, or give it a fixed size: -> pulse.Length[list[int], 8]."
    )


def test_lengths_option_is_gone():
    def f(x: pulse.Bits) -> float:
        return 0.0

    with pytest.raises(TypeError, match="unexpected keyword argument 'lengths'"):
        pulse.classical(f, max_latency_ns=1, lengths={"x": 2})  # type: ignore[call-overload]  # remnant-ok


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
    with pytest.raises(TypeError, match="takes 1 positional arguments"):
        double(1, 2)
    assert echo([1]) == [1]


def test_external_function_has_no_body():
    @classical.external(max_latency_ns=5, link="feedback")
    def remote(syndrome: pulse.Length[pulse.Bits, 4], gain: float) -> pulse.Length[pulse.Bits, 2]:
        return pulse.Bits()

    assert remote.implementation is None and remote.name == "remote"
    assert [a.descriptor for a in remote.args] == ["bits:4", "f64"]
    assert [r.descriptor for r in remote.results] == ["bits:2"]
    message = (
        "remote runs on link 'feedback'; it is declared with @pulse.classical.external, so it"
        " has no body to run here. Give it a handler to simulate it:"
        ' pulse.simulate(..., handlers={"remote": fn})'
    )
    with pytest.raises(TypeError, match=re.escape(message)):
        remote([0, 1, 0, 1], 1.0)
    with pytest.raises(TypeError, match="needs a name"):
        classical.ClassicalFunction(None, max_latency_ns=1)
    assert pulse.Bits() == [] and isinstance(pulse.Bits(), list)


def test_an_external_placeholder_gives_its_result_shape():
    @classical.external(max_latency_ns=5)
    def table(index: pulse.Length[pulse.Bits, 3]) -> float:
        return float()

    @classical.external(max_latency_ns=5)
    def decode(syndrome: pulse.Bits) -> tuple[float, pulse.Bits]:
        return (float(), syndrome[:2])

    @classical.external(max_latency_ns=5)
    def echo(outcomes: pulse.Bits) -> pulse.Bits:
        return outcomes

    @classical.external(max_latency_ns=5)
    def anything(index: pulse.Length[pulse.Bits, 3]) -> pulse.Length[list[float], 2]:
        raise NotImplementedError  # remnant-ok: any body is accepted

    assert [a.descriptor for a in table.args] == ["bits:3"]
    version = decode.specialize(syndrome=6)
    assert version.name == "decode__6" and version.implementation is None
    assert [r.descriptor for r in version.results] == ["f64", "bits:2"]
    assert [r.descriptor for r in echo.specialize(outcomes=3).results] == ["bits:3"]
    assert [r.descriptor for r in anything.results] == ["f64[2]"]
    assert [a.descriptor for a in _sink.args] == ["f64"] and _sink.results == ()


def test_an_external_list_result_needs_a_length():
    advice = (
        "An external's body is a placeholder: return index, or a slice such as index[:2], for"
        " a length the input gives, or give it a fixed size: -> pulse.Length[pulse.Bits, 8]."
    )
    with pytest.raises(TypeError, match="the placeholder says only its type") as error:

        @classical.external(max_latency_ns=5)
        def lookup(index: pulse.Length[pulse.Bits, 2]) -> pulse.Bits:
            return pulse.Bits()

    assert str(error.value).endswith(advice)
    with pytest.raises(TypeError, match="it always raises an exception") as error:

        @classical.external(max_latency_ns=5)
        def table(index: pulse.Length[pulse.Bits, 2]) -> pulse.Bits:
            raise NotImplementedError  # remnant-ok

    assert str(error.value).endswith(advice)


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
    assert re.search(r"classical.callable @echo__1 \{.*args = \[\"bits:1\"\]", text)
    assert "classical.get_link {device = 0 : i64}" in text
    assert re.search(r"classical.submit %\S+, @echo__1\(%result\)", text)
    assert re.search(r"classical.wait %future, @echo__1 .*: i1", text)
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
        ok, value = double.submit(0.5).result(timeout=50)
        if ok:
            drive(d, square(40, value))

    text = _mlir(k, Q[0])
    assert re.search(r"classical.wait .*timeout_ns = 5.0+e\+01.*: i1, f64", text)


def test_several_results_come_back_as_a_tuple():
    @pulse.classical(max_latency_ns=10)
    def fit(p: list[complex]) -> tuple[float, list[float]]:
        return (0.0, [0.0, 0.0])

    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        a = readout(r, square(100, 0.05))
        b = readout(r, square(100, 0.05))
        shift, (x, y) = fit.submit([a, b]).result()
        drive(d, square(40, shift + x + y))

    assert re.search(r"classical.wait .*: f64, f64, f64", _mlir(k, Q[0]))


def _fixed_machine(**keys):
    cores = pulse.Processor(links=["controller"], dispatch_keys=keys)
    return pulse.ControlSystem(
        controllers=[pulse.Controller(model="vendor.fpga", processors=[cores])],
        links={"controller": Link(), "gpu": Link()},
    )


def _program(*functions, machine=None):
    @pulse.kernel
    def k(q):
        r = get_readout_line(q)
        m = readout(r, square(100, 0.05))
        for f in functions:
            f.submit([m]).result()

    return pulse.compile(k, machine=machine).lower(Q[0])


def _one(name: str, link: int | str = "controller"):
    def body(o: pulse.Length[pulse.Bits, 1]) -> bool:
        return bool(o[0])

    body.__name__ = body.__qualname__ = name
    return pulse.classical(body, max_latency_ns=10, link=link)


def test_a_call_is_dispatched_by_the_hash_of_its_compiled_name():
    @pulse.classical(max_latency_ns=10)
    def first(o: pulse.Bits) -> bool:
        return bool(o[0])

    low = _program(first)
    (call,) = low.classical_report()
    assert (call.compiled_name, call.dispatch_key) == (
        "first__1",
        classical.function_id("first__1"),
    )
    (entry,) = low.export_emulation()["manifest"]["functions"]
    assert entry["dispatch_key"] == call.dispatch_key and "dispatch_key_fixed_by" not in entry
    assert "function_id" not in entry and "dispatch_key" not in low.export_emulation()["calls"][0]


def test_a_function_on_a_processor_that_fixes_keys_must_be_listed():
    with pytest.raises(pulse.CompilationError) as error:
        _program(_one("decode_patch"), machine=_fixed_machine(decode_pach=1, table=3))
    assert str(error.value) == (
        "decode_patch runs on controllers[0] (vendor.fpga).processors[0], whose dispatch_keys"
        " have no 'decode_patch' (did you mean 'decode_pach'?); its firmware answers"
        " decode_pach, table. Add it to the processor's dispatch_keys, or call it on another link"
    )


def test_two_names_that_hash_alike_are_an_error():
    a, b = _one("fit_332789", link=0), _one("fit_529192", link=0)
    with pytest.raises(pulse.CompilationError) as error:
        _program(a, b)
    assert str(error.value).endswith(
        "hash to the same dispatch key 0x54b57d4b on link 0; rename one of them"
    )


def test_a_name_that_hashes_to_a_reserved_key_is_an_error():
    with pytest.raises(pulse.CompilationError, match="which cudaq_pulse.classical.qec uses"):
        _program(_one("enqueue_syndromes", link=0))


def test_the_reserved_keys_are_the_keys_of_the_decoder_protocol():
    from cudaq_pulse.classical import _RESERVED_KEYS, qec

    assert set(_RESERVED_KEYS) == {
        qec.ENQUEUE_SYNDROMES_ID,
        qec.GET_CORRECTIONS_ID,
        qec.RESET_DECODER_ID,
    }


def test_a_template_on_a_processor_that_fixes_keys_needs_a_length():
    @pulse.classical(max_latency_ns=10, link="controller")
    def table(o: pulse.Bits) -> bool:
        return bool(o[0])

    with pytest.raises(pulse.CompilationError) as error:
        _program(table, machine=_fixed_machine(table=3))
    assert str(error.value).endswith(
        "its parameter 'o' is pulse.Bits with no length. Give it a fixed size:"
        " o: pulse.Length[pulse.Bits, 8]"
    )


def test_a_processor_fixes_the_keys_of_the_functions_it_runs():
    cores = pulse.Processor(links=["controller"], dispatch_keys={"table": 5})
    box = pulse.Controller(model="vendor.fpga", processors=[cores])
    system = pulse.ControlSystem(controllers=[box], links={"controller": Link(), "gpu": Link()})
    low = _program(_one("table"), _one("decode", link="gpu"), machine=system)
    assert [(c.compiled_name, c.dispatch_key) for c in low.classical_report()] == [
        ("table", 5),
        ("decode", classical.function_id("decode")),
    ]
    entries = {e["name"]: e for e in low.export_emulation()["manifest"]["functions"]}
    assert entries["table"]["dispatch_key_fixed_by"] == "controllers[0] (vendor.fpga).processors[0]"
    with pytest.raises(pulse.CompilationError, match="processors\\[0\\]"):
        _program(_one("vote"), machine=system)


def test_two_controllers_on_one_link_agree_on_its_keys():
    def box(lines, keys, **fields):
        cores = pulse.Processor(links=["controller"], dispatch_keys=keys)
        return pulse.Controller(drive_lines=lines, processors=[cores], **fields)

    first = box([0], {"a": 3}, model="vendor.fpga")
    served = "controllers[0] (vendor.fpga).processors[0] also serves link 'controller'"
    for keys, message in (
        ({"a": 4}, f"gives a 4, but {served} and gives it 3"),
        ({"b": 3}, f"gives b 3, but {served} and gives that key to a"),
    ):
        second = box([1], keys)
        links = {"controller": Link(per_controller=True)}
        system = pulse.ControlSystem(controllers=[first, second], links=links)
        with pytest.raises(pulse.CompilationError, match=re.escape(message)):
            _dispatch_tables(system._machine)


# ── Errors ───────────────────────────────────────────────────────────────────


def _compile_error(body, match):
    with pytest.raises(pulse.CompilationError, match=match):
        pulse.compile(pulse.kernel(body)).lower(Q[0])


@pulse.classical(max_latency_ns=10)
def _fixed_pair(outcomes: pulse.Length[pulse.Bits, 1], k: float) -> bool:
    return bool(outcomes[0])


@pulse.classical(max_latency_ns=10)
def _one_count(n: pulse.Length[list[int], 1]) -> bool:
    return bool(n[0])


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda m: echo.submit(m),
            "echo.submit(): bits is pulse.Bits (a list of outcomes) but m is one readout."
            " Write echo.submit([m]).",
        ),
        (
            lambda m: _fixed_pair.submit(m, 1.0),
            "_fixed_pair.submit(): outcomes is pulse.Length[pulse.Bits, 1] (a list of outcomes)"
            " but m is one readout. Write _fixed_pair.submit([m], k).",
        ),
        (
            lambda m: _one_count.submit(m),
            "_one_count.submit(): n is pulse.Length[list[int], 1] (a list of numbers) but m is"
            " one readout. Write _one_count.submit([m]).",
        ),
        (
            lambda m: choose.submit(m),
            "choose.submit(): levels is list[int] (a list of outcomes or numbers) but m is one"
            " readout. Write choose.submit([m]).",
        ),
    ],
)
def test_one_readout_for_a_list_parameter_is_an_error(call, message):
    def k(q):
        r = get_readout_line(q)
        m = readout(r, square(10, 0.1))
        call(m)

    _compile_error(k, re.escape(message))


@classical.external(max_latency_ns=50)
def _table(index: pulse.Length[pulse.Bits, 3]) -> float:
    return float()


@classical.external(max_latency_ns=50)
def _syndromes(syndrome: pulse.Bits) -> tuple[float, pulse.Bits]:
    return (float(), syndrome[:1])


@classical.external(max_latency_ns=50)
def _decode_surface(syndrome: pulse.Bits) -> pulse.Length[pulse.Bits, 2]:
    return pulse.Bits()


def test_a_call_of_the_wrong_length_names_the_fixed_size():
    def k(q):
        r = get_readout_line(q)
        m = readout(r, square(10, 0.1))
        _table.submit([m, m]).result()

    _compile_error(
        k, re.escape("_table.submit(): 'index' takes 3 bits (pulse.Length[pulse.Bits, 3]), got 2")
    )


def test_an_external_of_unsized_inputs_has_a_version_per_length():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        m = readout(r, square(10, 0.1))
        _, a = _syndromes.submit([m, m]).result()
        _, b = _syndromes.submit([m, m, m]).result()
        c = _decode_surface.submit([m, m, m, m]).result()
        sync(d, r)
        if a[0] or b[0] or c[1]:
            drive(d, square(40, 0.1))

    low = pulse.compile(k, schedule="asap").lower(Q[0])
    names = [(c.function, c.compiled_name) for c in low.classical_report()]
    assert names == [
        ("_syndromes", "_syndromes__2"),
        ("_syndromes", "_syndromes__3"),
        ("_decode_surface", "_decode_surface__4"),
    ]
    entries = {e["name"]: e for e in low.export_emulation()["manifest"]["functions"]}
    assert set(entries) == {"_syndromes__2", "_syndromes__3", "_decode_surface__4"}
    assert entries["_syndromes__3"]["dispatch_key"] == classical.function_id("_syndromes__3")
    assert entries["_syndromes__3"]["schema"] == wire.schema(_syndromes.specialize(syndrome=3))
    key = classical.function_id("_decode_surface__4")
    assert entries["_decode_surface__4"]["dispatch_key"] == key


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

    def one_measurement_for_bits(q):
        r = get_readout_line(q)
        echo.submit(readout(r, square(10, 0.1)))

    def mixed_levels(q):
        r = get_readout_line(q)
        choose.submit([readout(r, square(10, 0.1)), 1])

    def keyword(q):
        double.submit(x=1.0)

    _compile_error(wrong_count, "takes 1 arguments, got 0")
    _compile_error(number_for_bits, "takes measurement results")
    _compile_error(measurement_for_float, "takes numbers")
    _compile_error(
        one_measurement_for_bits,
        re.escape(
            "echo.submit(): bits is pulse.Bits (a list of outcomes) but the argument is one"
            " readout. Write echo.submit([m])."
        ),
    )
    _compile_error(mixed_levels, "all measurements .sent as levels. or all numbers")
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
    @pulse.classical(name="same", max_latency_ns=1)
    def a() -> float:
        return 1.0

    @pulse.classical(name="same", max_latency_ns=1)
    def b() -> float:
        return 2.0

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
    assert call.function == "echo" and call.compiled_name == "echo__1"
    assert call.link == 0 and call.iteration == ()
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
