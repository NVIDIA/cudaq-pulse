# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Measurement-conditioned feedback: retained as an if region end to end."""

from __future__ import annotations

import random

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.kernel.ir_builder import CompilationError, Parameter

from tests._handles import Q
from tests._ir import python_ir
from tests._work import python_work
from tests.kernel.ir_corpus import active_reset, feedback_in_loop, negated_feedback

_FREQ = {0: 5.0e9, 1: 5.1e9}


def _compile(kernel, n_qubits=1, **kwargs):
    return pulse.compile(kernel, **kwargs).lower(*[Q[i] for i in range(n_qubits)])


def _makespan(compiled):
    return max(e.start_cycles + e.duration_cycles for e in compiled.schedule)


def test_active_reset_retains_the_conditional():
    ir = python_ir(active_reset, Q[0])
    kinds = [op.kind for op in ir.ops]
    start = kinds.index("scf.if")
    assert kinds[start:] == [
        "scf.if",
        "pulse.drive",
        "scf.yield",
        "scf.else",
        "scf.yield",
        "scf.if_end",
        "pulse.wait",
    ]
    cond = ir.ops[start].operands[0]
    assert cond.vtype == "measurement"
    end = ir.ops[kinds.index("scf.if_end")]
    assert end.attrs == {"result_names": ["d0", "d0.tone"]}
    # The wait after the if continues the line the if produced.
    assert ir.ops[-1].operands[0] is end.results[0]


def test_negation_swaps_the_arms():
    ir = python_ir(negated_feedback, Q[0], Q[1])
    kinds = [op.kind for op in ir.ops]
    then_arm = kinds[kinds.index("scf.if") + 1 : kinds.index("scf.else")]
    else_arm = kinds[kinds.index("scf.else") + 1 : kinds.index("scf.if_end")]
    assert "pulse.wait" in then_arm and "pulse.drive" not in then_arm
    assert "pulse.drive" in else_arm and "pulse.shift_phase" in else_arm


def test_compiled_feedback_verifies():
    for kernel, n in ((active_reset, 1), (negated_feedback, 2), (feedback_in_loop, 1)):
        compiled = _compile(kernel, n)
        assert "pulse.is_set" in compiled.mlir()
        compiled._module.run_passes(["pulse-verify"])


@pytest.mark.parametrize("schedule", ["asap", "alap"])
def test_schedule_allows_for_the_longer_arm(schedule):
    # readout (300) then the conditional X (40) and a wait (10)
    assert _makespan(_compile(active_reset, schedule=schedule)) == 300 + 40 + 10
    # else arm (wait 40 on d0) vs then arm (shift + 40 on d1); both start
    # after the readout, then a 20 drive on d0.
    assert _makespan(_compile(negated_feedback, 2, schedule=schedule)) == 300 + 40 + 20


def test_conditional_waits_for_the_measurement():
    events = _compile(active_reset, schedule="asap").schedule
    readout = next(e for e in events if e.kind == "readout")
    drive = next(e for e in events if e.kind == "drive")
    assert drive.start_cycles >= readout.start_cycles + readout.duration_cycles


def test_feedback_inside_a_retained_loop():
    ir = python_ir(feedback_in_loop, Q[0])
    kinds = [op.kind for op in ir.ops]
    assert kinds.count("scf.for") == 1 and kinds.count("scf.if") == 1
    compiled = _compile(feedback_in_loop, schedule="asap")
    assert compiled.mlir().count("scf.for") == 1
    # Each readout may start while the previous iteration's conditional
    # drive is still playing on the drive line.
    assert _makespan(compiled) == 5 * 300 + 40


def test_python_state_cannot_depend_on_a_measurement():

    @pulse.kernel
    def k(q0):
        d0 = get_drive_line(q0)
        r0 = get_readout_line(q0)
        bit = readout(r0, square(300, 0.05))
        n = [1]
        if bit:
            n = [2]
        wait(d0, 10 * n[0])

    # Numbers are chosen (test_measurement_choice.py); other Python values are not.
    with pytest.raises(CompilationError, match="'n' would depend on a measurement result"):
        k(Q[0])


def test_one_armed_return_is_rejected():

    @pulse.kernel(unroll_threshold=0)
    def k(q0):
        d0 = get_drive_line(q0)
        r0 = get_readout_line(q0)
        bit = readout(r0, square(300, 0.05))
        for _ in range(2):
            if bit:
                return
            wait(d0, 10)

    with pytest.raises(CompilationError, match="must continue after it"):
        k(Q[0])


def test_break_out_of_an_arm_is_rejected():

    @pulse.kernel
    def k(q0):
        d0 = get_drive_line(q0)
        r0 = get_readout_line(q0)
        for i in range(3):
            bit = readout(r0, square(300, 0.05))
            if bit:
                break
            wait(d0, 10 + i)

    with pytest.raises(CompilationError, match="must continue after it"):
        k(Q[0])


@pytest.mark.parametrize("condition", ["amp == 0.5", "amp != 0.5", "amp < 1", "0 <= amp"])
def test_symbolic_comparison_raises(condition):
    source = (
        f"def k(q0, amp):\n    d0 = get_drive_line(q0)\n    if {condition}:\n        wait(d0, 10)\n"
    )
    namespace: dict = {}
    exec(compile(source, "<symbolic>", "exec"), namespace)
    kernel = pulse.kernel(namespace["k"])
    with pytest.raises(CompilationError, match="cannot decide"):
        kernel(Q[0], Parameter("amp", 0, "f64"))


def test_comparing_a_measurement_raises():

    @pulse.kernel
    def k(q0):
        r0 = get_readout_line(q0)
        bit = readout(r0, square(10, 0.2))
        if bit == 0:
            wait(r0, 10)

    with pytest.raises(CompilationError, match=r"branch on it directly with `if result:`"):
        k(Q[0])


def test_only_measurements_can_be_conditions():

    @pulse.kernel
    def k(q0):
        d0 = get_drive_line(q0)
        if d0:
            wait(d0, 10)

    with pytest.raises(CompilationError, match="only measurement results"):
        k(Q[0])


_HELD_HEAD = """\
def k(q0, q1):
    d0 = get_drive_line(q0)
    d1 = get_drive_line(q1)
    t0 = d0.tone
    r = get_readout_line(q0)
"""
_HELD_READ = "    m = readout(r, square(100, 0.05))\n    sync(d0, d1, r)\n"
_ARM = "square(20, 0.1)"
_AFTER = "square(8, 0.1)"

# (lines before the readout, the rest of the body, the body with plain names)
_HELD = {
    "unused list": (
        "    lines = [d0, d1]\n",
        f"    if m:\n        drive(d0, {_ARM})\n        drive(d1, {_ARM})\n",
        None,
    ),
    "else arm": (
        "    lines = [d0, d1]\n",
        f"    if m:\n        pass\n    else:\n        drive(lines[0], {_ARM})\n"
        f"    drive(lines[0], {_AFTER})\n",
        f"    if m:\n        pass\n    else:\n        drive(d0, {_ARM})\n    drive(d0, {_AFTER})\n",
    ),
    "both arms": (
        "    lines = [d0, d1]\n",
        f"    if m:\n        drive(d0, {_ARM})\n    else:\n        drive(lines[1], {_ARM})\n"
        f"    drive(lines[0], {_AFTER})\n    drive(d1, {_AFTER})\n",
        f"    if m:\n        drive(d0, {_ARM})\n    else:\n        drive(d1, {_ARM})\n"
        f"    drive(d0, {_AFTER})\n    drive(d1, {_AFTER})\n",
    ),
    "one line twice": (
        "    lines = [d0, d0]\n",
        f"    if m:\n        drive(lines[0], {_ARM})\n    drive(lines[1], {_AFTER})\n",
        f"    if m:\n        drive(d0, {_ARM})\n    drive(d0, {_AFTER})\n",
    ),
    "two lists and an alias": (
        "    e = d0\n    xs = [e]\n    ys = [d1, d0]\n",
        f"    if m:\n        drive(e, {_ARM})\n    else:\n        drive(ys[1], {_ARM})\n"
        f"    drive(xs[0], {_AFTER})\n    drive(ys[0], {_AFTER})\n",
        f"    if m:\n        drive(d0, {_ARM})\n    else:\n        drive(d0, {_ARM})\n"
        f"    drive(d0, {_AFTER})\n    drive(d1, {_AFTER})\n",
    ),
    "only the list": (
        "    lines = [d0, d1]\n",
        f"    del d0\n    if m:\n        drive(lines[0], {_ARM})\n    drive(lines[0], {_AFTER})\n",
        f"    if m:\n        drive(d0, {_ARM})\n    drive(d0, {_AFTER})\n",
    ),
    "tuple": (
        "    lines = (d0, d1)\n",
        f"    if m:\n        drive(lines[0], {_ARM})\n    drive(lines[0], {_AFTER})\n",
        f"    if m:\n        drive(d0, {_ARM})\n    drive(d0, {_AFTER})\n",
    ),
    "dict": (
        "    lines = {'a': d0, 'b': d1}\n",
        f"    if m:\n        drive(lines['a'], {_ARM})\n    else:\n        drive(d1, {_ARM})\n"
        f"    drive(lines['a'], {_AFTER})\n",
        f"    if m:\n        drive(d0, {_ARM})\n    else:\n        drive(d1, {_ARM})\n"
        f"    drive(d0, {_AFTER})\n",
    ),
    "nested containers": (
        "    lines = {'a': [d0, (d1,)], 'b': [[d0]]}\n",
        f"    if m:\n        drive(lines['b'][0][0], {_ARM})\n"
        f"    else:\n        drive(lines['a'][1][0], {_ARM})\n"
        f"    drive(lines['a'][0], {_AFTER})\n    drive(lines['a'][1][0], {_AFTER})\n",
        f"    if m:\n        drive(d0, {_ARM})\n    else:\n        drive(d1, {_ARM})\n"
        f"    drive(d0, {_AFTER})\n    drive(d1, {_AFTER})\n",
    ),
    "tones": (
        "    tones = [t0]\n    lines = [d0]\n",
        f"    if m:\n        drive(lines[0], {_ARM}, tone=tones[0])\n"
        f"    drive(d0, {_AFTER}, tone=tones[0])\n",
        f"    if m:\n        drive(d0, {_ARM}, tone=t0)\n    drive(d0, {_AFTER}, tone=t0)\n",
    ),
    "nested ifs": (
        "    lines = [d0, d1]\n",
        "    m2 = readout(r, square(100, 0.05))\n"
        f"    if m:\n        drive(lines[0], {_ARM})\n"
        f"        if m2:\n            drive(lines[1], {_ARM})\n"
        f"    drive(lines[1], {_AFTER})\n",
        "    m2 = readout(r, square(100, 0.05))\n"
        f"    if m:\n        drive(d0, {_ARM})\n        if m2:\n            drive(d1, {_ARM})\n"
        f"    drive(d1, {_AFTER})\n",
    ),
}


def _get_n() -> int:
    return 5


def _held_kernel(before, rest, tag, condition="m"):
    if condition != "m":
        rest = "    n = get_n.submit().result()\n" + rest.replace(
            "    if m:", f"    if {condition}:"
        )
    source = _HELD_HEAD + before + _HELD_READ + rest
    scope = vars(pulse) | {"__name__": tag, "get_n": pulse.classical(_get_n, max_latency_ns=0)}
    namespace: dict = {}
    exec(compile(source, f"<{tag}>", "exec"), scope, namespace)
    return pulse.kernel(namespace["k"])


def _yields_once(kernel):
    for op in python_ir(kernel, Q[0], Q[1]).ops:
        if op.kind in ("scf.yield", "scf.if_end", "scf.for_end"):
            vids = [v.vid for v in (op.operands or op.results)]
            assert len(vids) == len(set(vids)), op


@pytest.mark.parametrize("condition", ["m", "n > 3"])
@pytest.mark.parametrize("case", list(_HELD))
def test_lines_held_in_containers_cross_a_runtime_branch(case, condition):
    before, rest, plain = _HELD[case]
    held = _held_kernel(before, rest, case, condition)
    if condition == "m":  # the Python IR builder makes no classical calls
        _yields_once(held)
    compiled = _compile(held, 2, schedule="asap")
    compiled._module.run_passes(["pulse-verify"])
    reference = _compile(_held_kernel("", plain or rest, case, condition), 2, schedule="asap")
    assert compiled.schedule == reference.schedule


@pytest.mark.parametrize("unroll_threshold", [64, 0])
def test_lines_held_in_a_list_cross_a_branch_in_a_loop(unroll_threshold):
    loop = (
        "    for _ in range(3):\n"
        "        m = readout(r, square(100, 0.05))\n"
        "        sync(d0, d1, r)\n"
        "        if m:\n"
        "            drive({0}, square(20, 0.1))\n"
        "        else:\n"
        "            drive({1}, square(30, 0.1))\n"
        "    drive({0}, square(8, 0.1))\n"
    )
    held = _held_kernel("    lines = [d0, d1]\n", loop.format("lines[0]", "lines[1]"), "loop")
    plain = _held_kernel("", loop.format("d0", "d1"), "loop")
    _yields_once(held)
    compiled = _compile(held, 2, schedule="asap", unroll_threshold=unroll_threshold)
    compiled._module.run_passes(["pulse-verify"])
    assert compiled.mlir().count("scf.for") == (unroll_threshold == 0)
    reference = _compile(plain, 2, schedule="asap", unroll_threshold=unroll_threshold)
    assert compiled.schedule == reference.schedule


_ROUNDS = (0, 1)


def test_nested_feedback_ending_an_unrolled_loop_body():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        m = readout(r, square(10, 0.05))
        for _ in _ROUNDS:
            if m:
                if m:
                    drive(d, square(10, 0.1))

    events = k.lower(Q[0]).schedule
    assert [e.kind for e in events] == ["readout", "drive", "drive"]


def _rounds(n):
    @pulse.kernel(unroll_threshold=10**9)
    def rounds(q):
        d = get_drive_line(q)  # noqa: F821
        r = get_readout_line(q)  # noqa: F821
        outcomes = []
        for _ in range(n):
            m = readout(r, square(40, 0.05))  # noqa: F821
            sync(d, r)  # noqa: F821
            if m:
                drive(d, square(16, 0.1))  # noqa: F821
            outcomes.append(m)

    return rounds


@pytest.mark.slow
def test_branches_beside_a_growing_list_of_outcomes_trace_in_linear_work():
    # An arm that leaves a list as it was costs nothing for its length.
    work = {n: python_work(lambda n=n: pulse.compile(_rounds(n)).lower(Q[0])) for n in (50, 200)}
    assert all(work[200][count] <= 4 * work[50][count] for count in work[50]), work


def _beside_a_list(condition, listed, change=""):
    source = (
        "def k(q):\n"
        "    d = get_drive_line(q)\n"
        "    r = get_readout_line(q)\n"
        "    n = get_n.submit().result()\n"
        "    outcomes = [0.5]\n"
        "    for _ in range(3):\n"
        "        m = readout(r, square(40, 0.05))\n"
        "        m2 = readout(r, square(40, 0.05))\n"
        "        sync(d, r)\n"
        f"        if {condition}:\n"
        f"            drive(d, square(16, 0.1)){change}\n"
        "        drive(d, square(8, 0.2 if n > 3 else 0.1))\n"
        + ("        outcomes.append(m)\n" if listed else "")
    )
    scope = vars(pulse) | {"get_n": pulse.classical(_get_n, max_latency_ns=0)}
    namespace: dict = {}
    exec(compile(source, "<beside>", "exec"), scope, namespace)
    return pulse.kernel(unroll_threshold=10**9)(namespace["k"])


_CONDITIONS = [
    "m and not m2",
    "(n > 3) & (n < 9)",
    "(n > 3) | (n < 1)",
    "n in (1, 5)",
    "(n > 3) is True",
    "any([n > 3, n < 1])",
    "all([n > 1, n < 9])",
]


@pytest.mark.parametrize("condition", _CONDITIONS)
def test_runtime_conditions_beside_a_growing_list_trace_as_without_it(condition):
    # The list is copied only if an arm changes it: these arms do not.
    listed = pulse.compile(_beside_a_list(condition, True)).lower(Q[0]).schedule
    assert listed == pulse.compile(_beside_a_list(condition, False)).lower(Q[0]).schedule


@pytest.mark.parametrize("condition", ["m and m2", "(n > 3) & (n < 9)", "any([n > 3, n < 1])"])
def test_an_arm_on_a_combined_condition_that_changes_a_list_entry_is_rejected(condition):
    kernel = _beside_a_list(condition, True, change="; outcomes[0] = 0.25")
    with pytest.raises(CompilationError, match="may only change list entries that hold lines"):
        pulse.compile(kernel).lower(Q[0])


def test_an_arm_that_changes_a_list_entry_is_rejected():
    @pulse.kernel
    def changes(q):
        d = get_drive_line(q)  # noqa: F821
        r = get_readout_line(q)  # noqa: F821
        values = [1.0, 2.0]
        if readout(r, square(40, 0.05)):  # noqa: F821
            values[0] = 3.0
            drive(d, square(16, 0.1))  # noqa: F821

    with pytest.raises(CompilationError, match="may only change list entries that hold lines"):
        pulse.compile(changes).lower(Q[0])


def test_an_arm_whose_plain_python_call_reorders_a_list_is_rejected():
    rng = random.Random(0)

    @pulse.kernel
    def shuffles(q):
        d = get_drive_line(q)  # noqa: F821
        r = get_readout_line(q)  # noqa: F821
        values = [1.0, 2.0, 3.0, 4.0]
        if readout(r, square(40, 0.05)):  # noqa: F821
            rng.shuffle(values)
            drive(d, square(16, 0.1))  # noqa: F821

    with pytest.raises(CompilationError, match="may only change list entries that hold lines"):
        pulse.compile(shuffles).lower(Q[0])
