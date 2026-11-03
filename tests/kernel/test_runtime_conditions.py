# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Conditions built from values known only when the program runs.

Comparing such conditions with each other or with True and False, ``^``,
``&``, ``|``, ``is``, ``bool()``, ``not`` and ``in`` are tested at run time;
what cannot be (``is`` of a number, ordering them, combining a measurement result with
``^``, ``&``, ``|`` or ``==``, hashing an answer) raises at the user's line.
None of them is decided while tracing.
"""

from __future__ import annotations

import itertools
import math
import re
import warnings

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.kernel.ir_builder import CompilationError
from cudaq_pulse.targets import Qubit, TargetDescription

from tests._handles import Q

PI_AMP = math.pi / 20  # square(40, PI_AMP) is a pi pulse at 2 GHz

_DEVICE = pulse.device(
    {"target": TargetDescription(name="sim", qubits={0: Qubit(0, 5e9, -2e8, 0.0, 0.0)})}
)


def _answers(n=5, x=0.05, b=1):
    def get_n() -> int:
        return n

    def get_x() -> float:
        return x

    def get_b() -> bool:
        return bool(b)

    return {f.__name__: pulse.classical(f, max_latency_ns=0) for f in (get_n, get_x, get_b)}


def _kernel(condition, namespace=None, prelude="", after="    wait(d, 8)\n"):
    """A kernel that takes n (int), x (float) and b (bits) answers and drives
    a pi pulse when *condition* holds."""
    source = (
        "def body(q):\n"
        "    d = get_drive_line(q)\n"
        "    r = get_readout_line(q)\n"
        "    m = readout(r, square(100, 0.05))\n"
        "    n = get_n.submit().result()\n"
        "    x = get_x.submit().result()\n"
        "    b = get_b.submit().result()\n"
        "    sync(d, r)\n"
        f"{prelude}"
        f"    if {condition}:\n"
        "        drive(d, square(40, PI_AMP))\n"
        f"{after}"
    )
    scope = {**(namespace or _answers()), "PI_AMP": PI_AMP, "np": np}
    exec(compile(source, "<conditions>", "exec"), scope)  # noqa: S102 - one kernel per condition
    return pulse.kernel(scope["body"])


def _mlir(condition):
    return pulse.compile(_kernel(condition)).lower(Q[0]).mlir()


# ── Traced ────────────────────────────────────────────────────────────────────

_I1_EQ = r"arith\.cmpi eq, %\w+, %\w+ : i1"
_I1_NE = r"arith\.cmpi ne, %\w+, %\w+ : i1"
_I1_AND = r"arith\.andi %\w+, %\w+ : i1"
_I1_OR = r"arith\.ori %\w+, %\w+ : i1"


@pytest.mark.parametrize(
    ("condition", "i1_op"),
    [
        ("(n > 3) == (x > 0.1)", _I1_EQ),
        ("(n > 3) != (x > 0.1)", _I1_NE),
        ("(n > 3) ^ (x > 0.1)", _I1_NE),
        ("b == (x > 0.1)", _I1_EQ),
        ("((n > 3) == (x > 0.1)) == (n < 7)", _I1_EQ),
        ("(n > 3) == False", _I1_EQ),
        ("(not b) == (x > 0.1)", _I1_EQ),
        ("(n > 3) & (x > 0.1)", _I1_AND),
        ("(n > 3) | (x > 0.1)", _I1_OR),
        ("b & (n > 3)", _I1_AND),
        ("(n > 3) == True", None),
        ("True == (n > 3)", None),
        ("(n > 3) != False", None),
        ("bool(n > 3)", None),
        ("b == True", None),
        ("True & (n > 3)", None),
        ("(n > 3) | False", None),
        ("(n > 3) == np.True_", None),
    ],
)
def test_a_comparison_of_conditions_is_one_runtime_branch(condition, i1_op):
    text = _mlir(condition)
    assert text.count("scf.if") == 1
    found = re.findall(r"arith\.(?:cmpi \w+,|andi|ori) %\w+, %\w+ : i1", text)
    if i1_op is None:
        assert found == []
    else:
        assert any(re.fullmatch(i1_op, f) for f in found)


_SIMULATED = [
    "(n > 3) == (x > 0.1)",
    "(n > 3) != (x > 0.1)",
    "(n > 3) ^ (x > 0.1)",
    "(n > 3) == True",
    "(n > 3) == False",
    "(n > 3) != True",
    "(n > 3) == 1",
    "(n > 3) == 2",
    "bool(n > 3)",
    "bool(n)",
    "not ((n > 3) == (x > 0.1))",
    "((n > 3) == (x > 0.1)) == (n < 7)",
    "b == (x > 0.1)",
    "b != True",
    "1 < n < 7",
    "(n > 3) and (x > 0.1)",
    "(n > 3) or (x > 0.1)",
    "x == n",
    "(n > 3) & (x > 0.1)",
    "(n > 3) | (x > 0.1)",
    "b & (n > 3)",
    "b | (n > 3) & (x > 0.1)",
    "(n > 3) & 1",
    "(n > 3) ^ 1",
    "0 | (n > 3)",
    "(not (n > 3)) == (x > 0.1)",
    "(not b) != (n < 7)",
    "(n > 3) == np.True_",
    "n == None",
    "n != None",
    "n in (3, 5)",
    "n in [3, 5]",
    "n not in (0, 9)",
    "n in {3, 5}",
    "n in {5: 0.1, 9: 0.2}",
    "n in {5: 0.1}.keys()",
    "n in frozenset((5, 9))",
    "n in range(3, 7)",
    "n in range(9, 0, -4)",
    "n in range(1, 10, 4)",
    "n not in range(5, 5)",
    "n in ('five', 5)",
    "5 in [n, 3]",
    "x in (0.05, 0.25)",
    "x in {0.05}",
    "b in (True,)",
    "(n > 3) in {True}",
    "((n > 3) and (x > 0.1)) == True",
    "((n > 3) or b) != (x > 0.1)",
    "any([n > 3, x > 0.1])",
    "all([n > 3, x > 0.1])",
    "any((n, x > 0.1))",
    "all([True, b, n < 7])",
    "any([False, 0, n > 3])",
    "all([n > 3, 0])",
    "(n > 3) is True",
    "(n > 3) is False",
    "(n > 3) is not False",
    "True is (n > 3)",
    "b is True",
    "b is not True",
    "(not b) is True",
    "(n > 3) is (x > 0.1)",
    "(n > 3) is not b",
]


@pytest.mark.parametrize("condition", _SIMULATED)
def test_simulation_takes_the_branch_python_would(condition):
    for n, x, b in itertools.product((0, 5, 9), (0.05, 0.5), (0, 1)):
        kernel = pulse.compile(_kernel(condition, _answers(n, x, b)), device=_DEVICE)
        result = pulse.simulate(kernel.lower(_DEVICE.qudit_handles[0]), levels=2)
        excited = result.level_population(1)
        expected = bool(eval(condition, {"np": np}, {"n": n, "x": x, "b": bool(b)}))  # noqa: S307
        assert excited == pytest.approx(1.0 if expected else 0.0, abs=1e-3), (n, x, b)


@pytest.mark.parametrize(
    "condition", ["(n > 3) == (x > 0.1)", "(n > 3) & (x > 0.1)", "(n > 3) | (x > 0.1)"]
)
def test_the_export_tests_a_comparison_of_conditions_on_both_answers(condition):
    events = pulse.compile(_kernel(condition)).lower(Q[0]).export_controller()["events"]
    answers = [e["index"] for e in events if e["kind"] == "answer"]
    [pi] = (e for e in events if e["kind"] == "drive")
    [condition] = pi["conditions"]
    assert condition["test"] == "expression" and condition["arm"] is True
    assert sorted(condition["answers"]) == sorted(answers[:2])


@pytest.mark.parametrize("condition", ["bool(m)", "bool(not m)"])
def test_bool_of_a_measurement_is_a_runtime_branch(condition):
    text = _mlir(condition)
    assert text.count("scf.if") == 1 and "pulse.is_set" in text


def test_a_chained_comparison_with_code_after_it_is_two_branches():
    text = _mlir("1 < n < 7")
    assert text.count("scf.if") == 2


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(
    "condition",
    [
        "(x > 0.1) == (y > 0.2)",
        "(x > 0.1) != (y > 0.2)",
        "(x > 0.1) ^ (y > 0.2)",
        "(x > 0.1) == True",
        "(x > 0.1) is True",
        "(x > 0.1) < (y > 0.2)",
        "(x > 0.1) & (y > 0.2)",
        "(x > 0.1) | (y > 0.2)",
        "x in (0.05, 0.2)",
        "x in {0.05, 0.2}",
        "y not in {0.5: 1}",
    ],
)
def test_float_arguments_in_compared_conditions_specialize(condition):
    scope = {}
    exec(  # noqa: S102 - one kernel per condition
        "def body(q, x: float, y: float):\n"
        "    d = get_drive_line(q)\n"
        f"    if {condition}:\n"
        "        drive(d, square(40, 0.2))\n"
        "    wait(d, 8)\n",
        scope,
    )
    kernel = pulse.compile(pulse.kernel(scope["body"]))
    for x, y in itertools.product((0.05, 0.5), (0.05, 0.5)):
        expected = bool(eval(condition, {}, {"x": x, "y": y}))  # noqa: S307
        drives = re.search(r"pulse\.drive\b", kernel.lower(Q[0], x, y).mlir()) is not None
        assert drives == expected, (x, y)


# ── Rejected at the user's line ───────────────────────────────────────────────


@pytest.mark.parametrize(
    ("condition", "message"),
    [
        ("(n > 3) is 1", "compare values with == or !="),
        ("n is 5", "compare values with == or !="),
        ("(n > 3) < (x > 0.1)", "cannot order the conditions"),
        ("(n > 3) == x", "compares with another such condition, True or False"),
        ("n & 1", "bitwise operations on numbers"),
        ("(n > 3) & 2", "bitwise operations on numbers"),
        ("x in range(100)", "compare it with <, <= and == instead"),
        ("{n} == {5}", "it cannot be a set member or a dict key"),
        ("{n: 1}.get(5)", "it cannot be a set member or a dict key"),
        ("any([m])", "known only when the program runs"),
        ("all([m, not m])", "known only when the program runs"),
        ("any(c > 3 for c in (n,))", "give it a list, any([...])"),
        ("all(c > 3 for c in (n, 9))", "give it a list, all([...])"),
        ("[0.1, 0.2][n] > 0.1", "an index needs the value of get_n[0], the answer of a"),
        ("[1 for c in (n,) if c > 3]", "cannot filter on a condition known only"),
        ("sum(1 for c in (n,) if c > 3)", "cannot filter on a condition known only"),
        ("[c for c in (3, 5) if n > c]", "cannot filter on a condition known only"),
        ("{c for c in (3, 5) if n > c}", "cannot filter on a condition known only"),
    ],
)
def test_conditions_that_cannot_be_traced_raise_at_their_line(condition, message):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)  # `n is 5`
        kernel = _kernel(condition)
    with pytest.raises(CompilationError, match=re.escape(message)) as info:
        pulse.compile(kernel).lower(Q[0])
    assert 'File "<conditions>", line 9' in "".join(info.value.__notes__)


_MEASURED = (
    "cannot trace ({}): a measurement result is known only when the program runs, so it is "
    "not compared or combined while the kernel traces; branch on it directly with "
    "`if result:` or `if not result:`"
)
_MIXED = (
    "cannot trace ({}): it tests a measurement result and a condition on answers together, "
    "which one branch cannot; test them in nested ifs, `if result:` and in its arm an if on "
    "the condition"
)
_OF_A_PULSE_VALUE = (
    "cannot trace `{}` of a pulse value; to test a measurement result, branch on it directly "
    "with `if result:` or `if not result:`"
)


@pytest.mark.parametrize(
    ("condition", "message"),
    [
        ("m == True", _MEASURED.format("measurement == True")),
        ("m == False", _MEASURED.format("measurement == False")),
        ("m != True", _MEASURED.format("measurement != True")),
        ("True == m", _MEASURED.format("measurement == True")),
        ("m == 1", _MEASURED.format("measurement == 1")),
        ("m == 0", _MEASURED.format("measurement == 0")),
        ("1 != m", _MEASURED.format("measurement != 1")),
        ("m < 1", _MEASURED.format("measurement < 1")),
        ("m == np.True_", _MEASURED.format("measurement == np.True_")),
        ("bool(m) == True", _MEASURED.format("measurement == True")),
        ("(not m) == True", _MEASURED.format("(not measurement) == True")),
        ("m == m", _MEASURED.format("measurement == measurement")),
        ("m is True", _MEASURED.format("measurement is True")),
        ("m is not False", _MEASURED.format("measurement is False")),
        ("(not m) is False", _MEASURED.format("(not measurement) is False")),
        ("m ^ True", _MEASURED.format("measurement ^ True")),
        ("True ^ m", _MEASURED.format("measurement ^ True")),
        ("m & True", _MEASURED.format("measurement & True")),
        ("m | False", _MEASURED.format("measurement | False")),
        ("m ^ m", _MEASURED.format("measurement ^ measurement")),
        ("m in (True,)", _MEASURED.format("measurement == True")),
        ("m in {1}", _MEASURED.format("measurement == 1")),
        ("m not in (0,)", _MEASURED.format("measurement == 0")),
        ("m + 0 == 1", _MEASURED.format("measurement + 0")),
        (
            "int(m) == 1",
            "cannot trace int() of a measurement result: it is known only when the program "
            "runs; branch on it directly with `if result:` or `if not result:`",
        ),
        ("~m", _OF_A_PULSE_VALUE.format("~")),
        ("-m", _OF_A_PULSE_VALUE.format("-")),
        ("m == b", _MIXED.format("measurement == get_b[0]")),
        ("b == m", _MIXED.format("get_b[0] == measurement")),
        ("m != (n > 3)", _MIXED.format("measurement != (get_n[0] > 3)")),
        ("(not m) == (n > 3)", _MIXED.format("(not measurement) == (get_n[0] > 3)")),
        ("m & (n > 3)", _MIXED.format("measurement & (get_n[0] > 3)")),
        ("b | m", _MIXED.format("get_b[0] | measurement")),
        ("m ^ b", _MIXED.format("measurement ^ get_b[0]")),
        ("m is b", _MIXED.format("measurement is get_b[0]")),
        ("m in (b,)", _MIXED.format("get_b[0] == measurement")),
        ("(m == True) == b", _MIXED.format("(measurement == True) == get_b[0]")),
    ],
)
def test_a_measurement_result_compared_or_combined_raises_naming_the_fix(condition, message):
    """A measurement result is tested only by `if result:` and `if not result:`;
    with a condition on answers, in nested ifs."""
    with pytest.raises(CompilationError) as info:
        pulse.compile(_kernel(condition)).lower(Q[0])
    assert str(info.value).splitlines()[0] == message
    assert 'File "<conditions>", line 9' in "".join(info.value.__notes__)


@pytest.mark.parametrize(
    "subject",
    ["    match m:\n        case True:\n", "    match m:\n        case 1:\n"],
)
def test_a_match_on_a_measurement_result_raises_naming_the_fix(subject):
    kernel = _kernel("False", prelude=subject + "            wait(d, 40)\n")
    with pytest.raises(CompilationError, match=r"branch on it directly with `if result:`"):
        pulse.compile(kernel).lower(Q[0])


_CHOSEN = [
    ("    c = (n > 3) and (x > 0.1)\n", "c", "(n > 3) and (x > 0.1)"),
    ("    c = (n > 3) or (x > 0.1)\n", "c", "(n > 3) or (x > 0.1)"),
    ("    c = b and (n > 3)\n", "c", "b and (n > 3)"),
    ("    c = not b or (x > 0.1)\n", "c", "not b or (x > 0.1)"),
    ("    c = 1 < n < 7\n", "c", "1 < n < 7"),
    ("    c = (n > 3) if b else (x > 0.1)\n", "c", "(n > 3) if b else (x > 0.1)"),
    ("    c = x > 0.1 if n > 3 else False\n", "c", "x > 0.1 if n > 3 else False"),
    ("    c = (x > 0.1) == ((n > 3) and b)\n", "c", "(x > 0.1) == ((n > 3) and b)"),
    (
        "    if n > 3:\n        c = x > 0.1\n    else:\n        c = b\n",
        "c",
        "(x > 0.1) if n > 3 else b",
    ),
    # The amplitude `a` of a pulse after the if.
    ("    a = PI_AMP if n > 3 else 0.0\n", None, "n > 3"),
    ("    a = (n > 3) and PI_AMP\n", None, "n > 3"),
    ("    a = PI_AMP if (n > 3) == (x > 0.1) else 0.0\n", None, "(n > 3) == (x > 0.1)"),
]


@pytest.mark.parametrize(("prelude", "condition", "expected"), _CHOSEN)
def test_a_value_a_runtime_condition_chooses_is_the_one_python_would(prelude, condition, expected):
    """Bools are chosen as a condition on answers, numbers with a select."""
    for n, x, b in itertools.product((0, 5, 9), (0.05, 0.5), (0, 1)):
        if condition is None:
            after = "    drive(d, square(40, a))\n    wait(d, 8)\n"
            source = _kernel("False", _answers(n, x, b), prelude=prelude, after=after)
        else:
            source = _kernel(condition, _answers(n, x, b), prelude=prelude)
        kernel = pulse.compile(source, device=_DEVICE)
        result = pulse.simulate(kernel.lower(_DEVICE.qudit_handles[0]), levels=2)
        taken = bool(eval(expected, {}, {"n": n, "x": x, "b": bool(b)}))  # noqa: S307
        assert result.level_population(1) == pytest.approx(float(taken), abs=1e-3), (n, x, b)


def test_a_chosen_condition_is_one_compare_of_both_answers():
    text = _mlir_with("    c = (n > 3) and (x > 0.1)\n", "c")
    assert text.count("scf.if") == 1
    assert len(re.findall(r"arith\.andi", text)) == 1
    assert len(re.findall(r"arith\.cmp[if] ", text)) == 2


def _mlir_with(prelude, condition):
    return pulse.compile(_kernel(condition, prelude=prelude)).lower(Q[0]).mlir()


@pytest.mark.parametrize(
    ("prelude", "message"),
    [
        ("    c = m and (n > 3)\n", "both a measurement result and a condition on answers"),
        ("    c = m and b\n", "both a measurement result and a condition on answers"),
        (
            "    if m:\n        c = n > 3\n    else:\n        c = b\n",
            "'c' would depend on both a measurement result and a condition on answers",
        ),
        ("    c = (n > 3) and m\n", "the value of the conditional expression, `and` or `or`"),
        ("    c = 0.3 if n > 3 else 'a'\n", "an if chooses only between two numbers"),
    ],
)
def test_a_value_that_cannot_be_chosen_raises(prelude, message):
    kernel = _kernel("c", prelude=prelude)
    with pytest.raises(CompilationError, match=re.escape(message)):
        pulse.compile(kernel).lower(Q[0])


@pytest.mark.parametrize(
    ("subject", "cases", "expected"),
    [
        ("n > 3", ("True",), "n > 3"),
        ("b", ("False",), "not b"),
        ("x > 0.1", ("False", "True"), "not (x > 0.1)"),
    ],
)
def test_a_match_on_a_condition_with_bool_cases_takes_the_case_python_would(
    subject, cases, expected
):
    """`case True:` and `case False:` test identity, which on bools is ==;
    the first case drives a pi pulse, a second waits."""
    bodies = ("drive(d, square(40, PI_AMP))", "wait(d, 40)")
    pairs = zip(cases, bodies, strict=False)
    arms = "".join(f"        case {case}:\n            {body}\n" for case, body in pairs)
    prelude = f"    match {subject}:\n{arms}"
    for n, x, b in itertools.product((0, 5, 9), (0.05, 0.5), (0, 1)):
        source = _kernel("False", _answers(n, x, b), prelude=prelude)
        kernel = pulse.compile(source, device=_DEVICE)
        result = pulse.simulate(kernel.lower(_DEVICE.qudit_handles[0]), levels=2)
        taken = bool(eval(expected, {}, {"n": n, "x": x, "b": bool(b)}))  # noqa: S307
        assert result.level_population(1) == pytest.approx(float(taken), abs=1e-3), (n, x, b)


@pytest.mark.parametrize(
    "prelude",
    [
        "    assert n > 3\n",
        "    assert m, 'not excited'\n",
        "    if (n > 3) == (x > 0.1):\n        raise ValueError('too many')\n",
    ],
)
def test_a_raise_a_runtime_condition_decides_is_rejected_at_its_line(prelude):
    kernel = _kernel("True", prelude=prelude)
    with pytest.raises(CompilationError, match="whether it fails is decided as the program runs"):
        pulse.compile(kernel).lower(Q[0])


def test_a_while_loop_on_a_compared_condition_is_rejected():
    kernel = _kernel("True", after="    while (n > 3) == True:\n        wait(d, 8)\n")
    with pytest.raises(CompilationError, match="while loops are not supported"):
        pulse.compile(kernel).lower(Q[0])
