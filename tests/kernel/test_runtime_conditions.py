# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Conditions built from values known only when the program runs.

Comparing such conditions with each other or with True and False, ``^``,
``&``, ``|``, ``bool()`` and ``not`` are tested at run time; what cannot be
(``is``, ordering them, combining a measurement result with ``^``, ``&``,
``|`` or ``==``) raises at the user's line. None of them is decided while
tracing.
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
        ("(n > 3) is (x > 0.1)", "is known only when the program runs"),
        ("(n > 3) is True", "compare values with == or !="),
        ("(n > 3) is not False", "compare values with == or !="),
        ("m is True", "branch on a measurement result directly"),
        ("(not m) is False", "branch on a measurement result directly"),
        ("n is 5", "compare values with == or !="),
        ("(n > 3) < (x > 0.1)", "cannot order the conditions"),
        ("(n > 3) == x", "compares with another such condition, True or False"),
        ("m == (n > 3)", "a comparison of it is not traced"),
        ("(not m) == (n > 3)", "a comparison of it is not traced"),
        ("(not m) == True", "cannot compare a runtime IR value"),
        ("m == True", "cannot compare a runtime IR value"),
        ("m ^ True", "branch on a measurement result with `if result:`"),
        ("m & (n > 3)", "branch on a measurement result with `if result:`"),
        ("b | m", "branch on a measurement result with `if result:`"),
        ("n & 1", "bitwise operations on numbers"),
        ("(n > 3) & 2", "bitwise operations on numbers"),
        ("((n > 3) and (x > 0.1)) == True", "`and` or `or` would give a value"),
        ("any([m])", "known only when the program runs"),
        ("all([m, not m])", "known only when the program runs"),
        ("any([n > 3])", "known only when the program runs"),
    ],
)
def test_conditions_that_cannot_be_traced_raise_at_their_line(condition, message):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)  # `n is 5`
        kernel = _kernel(condition)
    with pytest.raises(CompilationError, match=re.escape(message)) as info:
        pulse.compile(kernel).lower(Q[0])
    assert 'File "<conditions>", line 9' in "".join(info.value.__notes__)


@pytest.mark.parametrize(
    "prelude",
    [
        "    c = (n > 3) and (x > 0.1)\n",
        "    c = (n > 3) or (x > 0.1)\n",
        "    c = m and (n > 3)\n",
        "    c = 1 < n < 7\n",
        "    c = 0.3 if n > 3 else 0.1\n",
        "    c = 0.3 if (n > 3) == (x > 0.1) else 0.1\n",
    ],
)
def test_a_value_chosen_by_a_runtime_condition_raises(prelude):
    kernel = _kernel("c", prelude=prelude)
    with pytest.raises(CompilationError, match="`and` or `or` would give a value") as info:
        pulse.compile(kernel).lower(Q[0])
    assert 'File "<conditions>", line 9' in "".join(info.value.__notes__)


def test_a_while_loop_on_a_compared_condition_is_rejected():
    kernel = _kernel("True", after="    while (n > 3) == True:\n        wait(d, 8)\n")
    with pytest.raises(CompilationError, match="while loops are not supported"):
        pulse.compile(kernel).lower(Q[0])
