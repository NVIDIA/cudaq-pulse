# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The values a controller program computes at run time are exported as
expressions in one vocabulary of terms, ``EXPRESSION_TERMS``, whatever
operations the compiler computes them with."""

from __future__ import annotations

import math
import re
import warnings

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import _validation
from cudaq_pulse._errors import ExecutionError, InvalidValueError, UnsupportedError
from cudaq_pulse._native._cudaq_pulse_native import MLIRPipeline
from cudaq_pulse.compiler import MachineModel
from cudaq_pulse.controller import (
    EXPRESSION_TERMS,
    FORMAT,
    ControllerLimits,
    evaluate_expression,
    unroll,
)

from tests._handles import Q


@pulse.classical(max_latency_ns=200, name="expression_inputs")
def _inputs(bits: pulse.Bits) -> tuple[float, int]:
    return 0.25 * bits[0], 3 + bits[0]


_HEAD = (
    "    d = get_drive_line(q)\n"
    "    ro = get_readout_line(q)\n"
    "    m = readout(ro, square(100, 0.05))\n"
)


def _kernel(body, *, bound=False):
    """A kernel running *body* on a float ``x`` and an int ``n``: answers of
    a call, or (*bound*) parameters, which the compiler folds."""
    source = "def k(q, x: float, n: int):\n" if bound else "def k(q):\n"
    source += _HEAD
    if not bound:
        source += "    x, n = _inputs.submit([m]).result()\n"
    source += "".join(f"    {line}\n" for line in body)
    namespace = {"_inputs": _inputs}
    exec(source, namespace)  # noqa: S102
    return pulse.kernel(namespace["k"])


def _value(program):
    """The phase the program shifts by, or the length it waits."""
    for event in program["events"] if program["format"] == FORMAT else unroll(program)["events"]:
        if event["kind"] == "wait":
            return event.get("length") or event["duration_cycles"]
        if event.get("op") == "shift_phase":
            return event["value"]
        for update in event.get("frame_updates", ()):
            if update["op"] == "shift_phase":
                return update["value"]
    raise AssertionError("no phase shift or wait")


def _answer(program):
    return next(e["index"] for e in program["events"] if e["kind"] == "answer")


def _ops(expression):
    if not isinstance(expression, dict):
        return set()
    found = {expression["op"]} if "op" in expression else set()
    for child in expression.get("of", ()):
        found |= _ops(child)
    return found


@pytest.fixture(autouse=True)
def _selects(monkeypatch):
    monkeypatch.setattr(pulse.compiler, "_can_select_at_runtime", lambda target, kind: True)


def _if(condition, then, other, use="shift_phase(d.tone, y)"):
    return [f"if {condition}:", f"    y = {then}", "else:", f"    y = {other}", use]


_PHASES = {
    "add": ["shift_phase(d.tone, x + 0.5)"],
    "sub": ["shift_phase(d.tone, 0.5 - x)"],
    "mul": ["shift_phase(d.tone, x * 3.0)"],
    "div": ["shift_phase(d.tone, x / 3.0)"],
    "neg": ["shift_phase(d.tone, -x)"],
    "to_float": ["shift_phase(d.tone, n * 0.5)"],
    "max": _if("x > 0.1", "x", "0.1"),
    "min": _if("x < 0.1", "x", "0.1"),
    "ge": _if("x >= 0.1", "x", "0.1"),
    "le": _if("x <= 0.1", "0.3", "0.1"),
    "lt": _if("n < 4", "0.3", "0.1"),
    "gt": _if("n > 3", "0.3", "0.1"),
    "eq": _if("n == 3", "0.3", "0.1"),
    "ne": _if("x != 0.25", "0.3", "0.1"),
    "select": _if("n == 3", "x", "0.1"),
}
_WAITS = {
    "add": ["wait(d, n + 4)"],
    "sub": ["wait(d, 10 - n)"],
    "mul": ["wait(d, n * 4)"],
    "floor_div": ["wait(d, n // 2 + 1)"],
    "rem": ["wait(d, n % 2 + 4)"],
    "to_int": ["wait(d, int(x * 80.0))"],
}
_INPUTS = [(0.05, 2), (0.1, 3), (0.25, 5), (0.3, 4)]


def _folded(body, x, n, machine=None):
    """What the compiler computes for *body* at x and n."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compiled = pulse.compile(_kernel(body, bound=True), machine=machine)
        return _value(compiled.lower(Q[0], x, n).export_controller())


@pytest.mark.parametrize(
    ("term", "body"),
    [*_PHASES.items(), *((f"wait {term}", body) for term, body in _WAITS.items())],
)
def test_each_term_computes_what_the_compiler_computes(term, body):
    lowered = pulse.compile(_kernel(body)).lower(Q[0])
    program = lowered.export_controller()
    expression = _value(program)
    assert term.removeprefix("wait ") in _ops(expression), expression
    answer = _answer(program)
    for x, n in _INPUTS:
        got = evaluate_expression(expression, answers={answer: [x, n]})
        assert got == _folded(body, x, n), (x, n)
    # Format /2 exports the same expression.
    assert _value(lowered.export_controller(version=2)) == expression


@pytest.mark.parametrize("term", sorted(_WAITS))
def test_each_term_waits_as_long_as_the_emulated_program(term):
    lowered = pulse.compile(_kernel(_WAITS[term])).lower(Q[0])
    expression = _value(program := lowered.export_controller())
    args = lowered.options.schedule_args()
    module = lowered._concrete().simulation_module(*args)
    readout, iteration = module.next_measurement([])[:2]
    call = module.next_call([[readout, *iteration, 1]])[0]
    for x, n in _INPUTS:
        bound = module.bind_call(call, [x, float(n)], *args)
        [wait] = (e for e in bound.events() if e[0] == "wait")
        assert evaluate_expression(expression, answers={_answer(program): [x, n]}) == wait[4]


def test_a_readout_is_set_when_it_observed_any_level_but_0():
    program = pulse.compile(_kernel(_if("m", "0.3", "0.1"))).lower(Q[0]).export_controller()
    readout = next(e["index"] for e in program["events"] if e["kind"] == "readout")
    expression = _value(program)
    assert expression["of"][0] == {
        "op": "ne",
        "type": "int",
        "of": [{"readout": readout}, {"const": 0}],
    }
    levels = {0: 0.1, 1: 0.3, 2: 0.3}
    for level, phase in levels.items():
        assert evaluate_expression(expression, readouts={readout: level}) == phase


def test_a_controller_clock_clamps_and_converts_a_run_time_length():
    machine = MachineModel(controller=ControllerLimits(clock_ghz=2.0, cycles_per_timestep=4))
    body = ["wait(d, n * 4, bounds=(8, 400))"]
    program = pulse.compile(_kernel(body), machine=machine).lower(Q[0]).export_controller()
    expression = _value(program)
    assert {"controller_cycles", "max", "min"} <= _ops(expression)
    answer = _answer(program)
    for n in (2, 7, 100):
        got = evaluate_expression(expression, answers={answer: [0.0, n]})
        assert got == _folded(body, 0.0, n, machine) == 4 * n
    # Outside its bounds, which a kernel cannot wait for, the run clamps it.
    for n, cycles in ((1, 8), (150, 400)):
        assert evaluate_expression(expression, answers={answer: [0.0, n]}) == cycles


def test_every_term_is_exercised():
    exported = set()
    for body in (*_PHASES.values(), *_WAITS.values()):
        exported |= _ops(_value(pulse.compile(_kernel(body)).lower(Q[0]).export_controller()))
    # The controller's cycles come from a controller clock, tested above.
    assert exported | {"controller_cycles"} == EXPRESSION_TERMS


# ── Forms that mean the same ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "forms",
    [
        [_if("x > 0.1", "x", "0.1"), _if("0.1 < x", "x", "0.1")],
        [_if("x < 0.1", "x", "0.1"), _if("0.1 > x", "x", "0.1")],
        [["shift_phase(d.tone, x * 2.0)"], ["shift_phase(d.tone, 2.0 * x)"]],
        [_if("m", "0.3", "0.1"), _if("not m", "0.1", "0.3")],
    ],
    ids=["float max", "float min", "mul", "readout"],
)
def test_an_expression_written_in_equivalent_ways_exports_the_same(forms):
    exported = [_value(pulse.compile(_kernel(f)).lower(Q[0]).export_controller()) for f in forms]
    assert all(e == exported[0] for e in exported), exported
    assert _ops(exported[0]) & {"max", "min", "mul", "select"}


def _computing(form):
    """The program shifting by ``n * 0.5`` with ``n`` replaced by *form*, MLIR
    computing ``%form`` from ``%n`` and ``%three``."""
    lowered = pulse.compile(_kernel(["shift_phase(d.tone, n * 0.5)"])).lower(Q[0])
    text = lowered._concrete().print()
    line = next(line for line in text.splitlines() if "arith.sitofp" in line)
    result, n = re.match(r"\s*(%\S+) = arith.sitofp (%\S+) :", line).groups()
    computed = "\n".join(
        [
            "%three = arith.constant 3 : i64",
            *form.replace("%n", n).split("; "),
            f"{result} = arith.sitofp %form : i64 to f64",
        ]
    )
    module = MLIRPipeline().parse_to_module(text.replace(line, computed))
    object.__setattr__(lowered, "_module", module)
    return lowered.export_controller()


@pytest.mark.parametrize(
    ("extremum", "forms"),
    [
        (
            "max",
            [
                "%form = arith.maxsi %n, %three : i64",
                "%c = arith.cmpi sgt, %n, %three : i64; %form = arith.select %c, %n, %three : i64",
                "%c = arith.cmpi sge, %n, %three : i64; %form = arith.select %c, %n, %three : i64",
                "%c = arith.cmpi slt, %n, %three : i64; %form = arith.select %c, %three, %n : i64",
                "%c = arith.cmpi sle, %n, %three : i64; %form = arith.select %c, %three, %n : i64",
            ],
        ),
        (
            "min",
            [
                "%form = arith.minsi %n, %three : i64",
                "%c = arith.cmpi slt, %n, %three : i64; %form = arith.select %c, %n, %three : i64",
                "%c = arith.cmpi sle, %n, %three : i64; %form = arith.select %c, %n, %three : i64",
                "%c = arith.cmpi sgt, %n, %three : i64; %form = arith.select %c, %three, %n : i64",
                "%c = arith.cmpi sge, %n, %three : i64; %form = arith.select %c, %three, %n : i64",
            ],
        ),
    ],
)
def test_the_integer_max_and_min_export_the_same_however_computed(extremum, forms):
    exported = [_computing(form) for form in forms]
    answer = {"answer": _answer(exported[0]), "result": 1}
    integer = {"op": extremum, "type": "int", "of": [answer, {"const": 3}]}
    for program in exported:
        assert _value(program) == {
            "op": "mul",
            "type": "float",
            "of": [{"op": "to_float", "type": "int", "of": [integer]}, {"const": 0.5}],
        }


def test_a_float_max_keeps_the_order_of_its_operands_which_nan_tells_apart():
    def exported(form):
        program = pulse.compile(_kernel(form)).lower(Q[0]).export_controller()
        return _value(program), _answer(program)

    (one, answer), (other, _) = (
        exported(_if("x > 0.1", "x", "0.1")),
        exported(_if("x < 0.1", "0.1", "x")),
    )
    assert one["op"] == other["op"] == "max"
    assert other["of"] == one["of"][::-1]
    # As the kernels run: x > 0.1 is false for NaN, and so is x < 0.1.
    assert evaluate_expression(one, answers={answer: [math.nan, 0]}) == 0.1
    assert math.isnan(evaluate_expression(other, answers={answer: [math.nan, 0]}))


# ── No operation of the compiler in any export ───────────────────────────────

_DIALECT = re.compile(
    r"(?<![\w-])(arith|math|scf|cf|func|llvm|index|pulse|classical|affine|builtin)\.\w"
)


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _no_dialect(program):
    prefixed = [s for s in _strings(program) if _DIALECT.search(s)]
    assert prefixed == []


@pytest.mark.parametrize("version", [1, 2])
def test_no_export_of_the_corpus_names_an_operation_of_the_compiler(version):
    dev = _validation._device(MachineModel())
    for kernel in _validation._corpus().values():
        lowered = pulse.compile(kernel, device=dev).lower(*dev.qudit_handles[:2])
        _no_dialect(lowered.export_controller(version=version))
    # Format /2 is for controllers on the program clock.
    clock = ControllerLimits(clock_ghz=2.0, cycles_per_timestep=4) if version == 1 else None
    machine = MachineModel(controller=clock)
    for body in (*_PHASES.values(), *_WAITS.values(), ["wait(d, n * 4, bounds=(8, 400))"]):
        lowered = pulse.compile(_kernel(body), machine=machine).lower(Q[0])
        _no_dialect(lowered.export_controller(version=version))


@pytest.mark.parametrize("version", [1, 2])
def test_an_operation_outside_the_vocabulary_is_an_export_error(version):
    lowered = pulse.compile(_kernel(["shift_phase(d.tone, x / 3.0)"])).lower(Q[0])
    text = lowered._concrete().print()
    assert "arith.divf" in text
    object.__setattr__(
        lowered, "_module", MLIRPipeline().parse_to_module(text.replace("arith.divf", "arith.remf"))
    )
    with pytest.raises(UnsupportedError, match="'arith.remf'.*expression vocabulary"):
        lowered.export_controller(version=version)


# ── The terms' semantics ─────────────────────────────────────────────────────


def _term(op, kind, *operands, **extra):
    return {"op": op, "type": kind, "of": [{"const": v} for v in operands], **extra}


_INT_MIN, _INT_MAX = -(2**63), 2**63 - 1


@pytest.mark.parametrize(
    ("expression", "value"),
    [
        (_term("div", "int", 7, -2), -3),
        (_term("div", "int", -7, 2), -3),
        (_term("floor_div", "int", 7, -2), -4),
        (_term("floor_div", "int", -7, 2), -4),
        (_term("rem", "int", -7, 2), -1),
        (_term("rem", "int", 7, -2), 1),
        (_term("add", "int", _INT_MAX, 1), _INT_MIN),
        (_term("mul", "int", 2**62, 4), 0),
        (_term("neg", "int", _INT_MIN), _INT_MIN),
        (_term("sub", "int", _INT_MIN, 1), _INT_MAX),
        (_term("div", "float", 1, 0), math.inf),
        (_term("div", "float", -1, 0), -math.inf),
        (_term("div", "float", 1, -0.0), -math.inf),
        (_term("add", "float", 1, 2), 3.0),
        (_term("to_float", "int", 2**53 + 1), float(2**53)),
        (_term("to_int", "float", -2.7), -2),
        (_term("to_int", "float", 2.7), 2),
        (_term("max", "int", 3, 5), 5),
        (_term("min", "float", -0.0, 0.0), 0.0),
        (_term("lt", "float", math.nan, 1.0), False),
        (_term("ne", "float", math.nan, math.nan), True),
        (_term("eq", "float", math.nan, math.nan), False),
        (_term("eq", "bool", True, True), True),
        (_term("select", "int", False, 1, 2), 2),
        (_term("controller_cycles", "int", 12, num=1, den=2, timestep=4), 6),
    ],
)
def test_a_term_computes_as_the_guide_specifies(expression, value):
    got = evaluate_expression(expression)
    assert got == value and type(got) is type(value)


def test_a_float_term_of_nan_is_as_ieee_754_computes_it():
    assert math.isnan(evaluate_expression(_term("div", "float", 0, 0)))
    assert math.isnan(evaluate_expression(_term("add", "float", math.inf, -math.inf)))
    assert evaluate_expression(_term("max", "float", math.nan, 1.0)) == 1.0
    assert math.isnan(evaluate_expression(_term("max", "float", 1.0, math.nan)))
    assert math.isnan(evaluate_expression(_term("min", "float", 1.0, math.nan)))


@pytest.mark.parametrize(
    "expression",
    [
        _term("div", "int", 1, 0),
        _term("floor_div", "int", 1, 0),
        _term("rem", "int", 1, 0),
        _term("div", "int", _INT_MIN, -1),
        _term("rem", "int", _INT_MIN, -1),
        _term("to_int", "float", math.nan),
        _term("to_int", "float", math.inf),
        _term("to_int", "float", 2.0**63),
        _term("controller_cycles", "int", 10, num=1, den=2, timestep=4),
    ],
)
def test_what_has_no_value_fails_the_run(expression):
    with pytest.raises(ExecutionError, match="fails the run"):
        evaluate_expression(expression)


def test_an_expression_outside_the_vocabulary_or_its_inputs_is_refused():
    with pytest.raises(InvalidValueError, match="'arith.addf' is not a term"):
        evaluate_expression({"op": "arith.addf", "of": [{"const": 1}, {"const": 2}]})
    with pytest.raises(InvalidValueError, match="readout event 4"):
        evaluate_expression({"readout": 4})
    with pytest.raises(InvalidValueError, match="result 1 given for answer event 2"):
        evaluate_expression({"answer": 2, "result": 1}, answers={2: [0.5]})
