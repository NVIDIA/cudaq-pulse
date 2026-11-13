# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conditions on measurement results kept in a value: `x = s0 and not s1`,
then `if x:`, compile to the ifs the condition written in the if gives."""

from __future__ import annotations

import itertools
import math
import re

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.classical import emulation
from cudaq_pulse.kernel.ir_builder import CompilationError
from cudaq_pulse.targets import DeviceDescription, PhysicalQudit

PI_AMP = math.pi / 20  # square(40, PI_AMP) is a pi pulse

_DEVICE = pulse.device(
    DeviceDescription(
        name="three",
        qudits={i: PhysicalQudit(i, 5e9 + 1e8 * i, -2e8, 0.0, 0.0) for i in range(3)},
    )
)


@pulse.classical(max_latency_ns=0)
def get_n() -> int:
    return 5


def _kernel(body: str) -> pulse.Kernel:
    """Readouts s0 and s1 of qudits a and b, *body* (which may drive the
    flag f), then a readout of the flag."""
    source = (
        "def k(a, b, c):\n"
        "    f, rf = get_drive_line(c), get_readout_line(c)\n"
        "    s0 = readout(get_readout_line(a), square(100, 0.05))\n"
        "    s1 = readout(get_readout_line(b), square(100, 0.05))\n"
        f"{body}"
        "    sync(f, rf)\n"
        "    readout(rf, square(100, 0.05))\n"
    )
    scope = {"PI_AMP": PI_AMP, "get_n": get_n}
    exec(compile(source, "<conditions>", "exec"), scope)  # noqa: S102 - one kernel per case
    return pulse.kernel(scope["k"])


def _lower(body: str) -> pulse.Lowered:
    return pulse.compile(_kernel(body), device=_DEVICE).lower(*_DEVICE.qudits[:3])


_FLAG = "        drive(f, square(40, PI_AMP))\n"

# A condition kept in x, the same condition in the if, and what Python
# makes of it on the two outcomes.
_FORMS = [
    ("s0 and not s1", lambda a, b: a and not b),
    ("s0 or s1", lambda a, b: a or b),
    ("not s0 or s1", lambda a, b: not a or b),
    ("not (s0 and s1)", lambda a, b: not (a and b)),
    ("s1 if s0 else not s1", lambda a, b: b if a else not b),
    ("s1 and s0", lambda a, b: b and a),
]


def _mlir(low: pulse.Lowered) -> str:
    text = low.mlir()
    return re.sub(r"loc\([^)]*\)", "", text[text.index("func.func") :])


@pytest.mark.parametrize(("condition", "_"), _FORMS)
def test_a_kept_condition_compiles_as_the_condition_in_the_if(condition, _):
    kept = _lower(f"    x = {condition}\n    if x:\n{_FLAG}")
    inline = _lower(f"    if {condition}:\n{_FLAG}")
    assert _mlir(kept) == _mlir(inline)
    assert "arith.select" not in kept.mlir() and "pulse.selected_values" not in kept.mlir()
    assert kept.export_controller() == inline.export_controller()
    runs = [emulation.emulate(low, shots=8, seed=4, service_time="none") for low in (kept, inline)]
    assert runs[0].outcomes == runs[1].outcomes and runs[0].ends_ns == runs[1].ends_ns


_KEPT = [
    *[(f"    x = {c}\n    if x:\n{_FLAG}", fn) for c, fn in _FORMS],
    ("    x = s0 and s1\n    if not x:\n" + _FLAG, lambda a, b: not (a and b)),
    ("    x = any([s0, s1])\n    if x:\n" + _FLAG, lambda a, b: a or b),
    ("    x = all([not s0, s1])\n    if x:\n" + _FLAG, lambda a, b: not a and b),
    ("    x = (not s0) ^ (not s1)\n    if x:\n" + _FLAG, lambda a, b: a != b),
    ("    x = (not s0) & (not s1)\n    if x:\n" + _FLAG, lambda a, b: not a and not b),
    (
        "    x = s0 and s1\n    y = x == (not s1)\n    if y:\n" + _FLAG,
        lambda a, b: (a and b) == (not b),
    ),
    (
        "    if s0:\n        x = s1\n    else:\n        x = not s1\n    if x:\n" + _FLAG,
        lambda a, b: b if a else not b,
    ),
    # Used twice: the flag goes up and back down where the condition holds.
    ("    x = s0 or s1\n    if x:\n" + _FLAG + "    if x:\n" + _FLAG, lambda a, b: False),
    # With a condition on an answer (n = 5).
    (
        "    n = get_n.submit().result()\n    x = s0 and (n > 3)\n    if x:\n" + _FLAG,
        lambda a, b: a,
    ),
    (
        "    n = get_n.submit().result()\n    x = (n < 3) or not s1\n    if x:\n" + _FLAG,
        lambda a, b: not b,
    ),
]


@pytest.mark.parametrize(("body", "expected"), _KEPT)
def test_simulation_drives_where_python_s_condition_holds(body, expected):
    plus = np.array([1, 1]) / np.sqrt(2)
    state = np.kron(np.array([1, 0]), np.kron(plus, plus))  # a, b in |+>, the flag in |0>
    low = _lower(body)
    result = pulse.simulate(low, initial_state=state, levels=2)
    # A branch's levels are in the order the readouts are scheduled.
    readouts = [e for e in low.schedule if e.kind == "readout"]
    order = [e.qudit for _, _, e in sorted((e.start_cycles, k, e) for k, e in enumerate(readouts))]
    seen = set()
    for branch in result.branches:
        s0, s1, flag = (dict(zip(order, branch.levels, strict=True))[q] for q in (0, 1, 2))
        assert flag == int(bool(expected(bool(s0), bool(s1)))), (s0, s1)
        seen.add((s0, s1))
    assert seen == set(itertools.product((0, 1), repeat=2))


_KEPT_AT = r"a condition on measurement results kept at line 5"
_REJECTED = [
    (
        "    x = s0 and not s1\n    drive(f, square(20, 0.1 * x))\n",
        rf"cannot trace `\*` of {_KEPT_AT}: a condition on measurement results is known only "
        r"when the program runs, and only an if tests it",
    ),
    ("    x = s0 or s1\n    y = x + 1\n", rf"cannot trace `\+` of {_KEPT_AT}"),
    ("    x = s0 or s1\n    drive(f, square(20, x))\n", rf"cannot trace square\(\) of {_KEPT_AT}"),
    ("    x = s0 or s1\n    wait(f, x)\n", rf"cannot trace wait\(\) of {_KEPT_AT}"),
    ("    x = s0 or s1\n    y = int(x)\n", rf"cannot trace int\(\) or float\(\) of {_KEPT_AT}"),
    (
        "    x = s0 or s1\n    if x == s1:\n" + _FLAG,
        r"cannot trace \(measurement == (True|False)\): a measurement result is known only",
    ),
    ("    if s0 & s1:\n" + _FLAG, r"cannot trace \(measurement & measurement\)"),
    ("    if s0 == s1:\n" + _FLAG, r"cannot trace \(measurement == measurement\)"),
    ("    x = s1 if s0 else 0.5\n", r"an if chooses only between two numbers"),
]


@pytest.mark.parametrize(("body", "message"), _REJECTED)
def test_a_condition_on_measurement_results_is_only_tested(body, message):
    with pytest.raises(CompilationError, match=message) as info:
        _lower(body)
    text = str(info.value)
    assert "%v" not in text and "_Choice" not in text and "_Not" not in text
