# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Measurement-conditioned feedback: retained as an if region end to end."""

from __future__ import annotations

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.kernel.ir_builder import CompilationError, Parameter

from tests._handles import Q
from tests._ir import python_ir
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
    assert end.attrs == {"result_names": ["d0", "t0"]}
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
        d0, t0 = get_drive_line(q0)
        r0, rt = get_readout_line(q0)
        bit = readout(r0, square(300, 0.05), rt)
        n = [1]
        if bit:
            n = [2]
        wait(d0, 10 * n[0])

    # Numbers are chosen (test_measurement_choice.py); other Python values are not.
    with pytest.raises(CompilationError, match="'n' would depend on a measurement result"):
        k(Q[0])


def test_one_armed_return_is_rejected():

    @pulse.kernel
    def k(q0):
        d0, t0 = get_drive_line(q0)
        r0, rt = get_readout_line(q0)
        bit = readout(r0, square(300, 0.05), rt)
        for _ in range(2):
            if bit:
                return
            wait(d0, 10)

    with pytest.raises(CompilationError, match="must continue after it"):
        k(Q[0])


def test_break_out_of_an_arm_is_rejected():

    @pulse.kernel
    def k(q0):
        d0, t0 = get_drive_line(q0)
        r0, rt = get_readout_line(q0)
        for i in range(3):
            bit = readout(r0, square(300, 0.05), rt)
            if bit:
                break
            wait(d0, 10 + i)

    with pytest.raises(CompilationError, match="must continue after it"):
        k(Q[0])


@pytest.mark.parametrize("condition", ["amp == 0.5", "amp != 0.5", "amp < 1", "0 <= amp"])
def test_symbolic_comparison_raises(condition):
    source = (
        "def k(q0, amp):\n"
        "    d0, t0 = get_drive_line(q0)\n"
        f"    if {condition}:\n"
        "        wait(d0, 10)\n"
    )
    namespace: dict = {}
    exec(compile(source, "<symbolic>", "exec"), namespace)
    kernel = pulse.kernel(namespace["k"])
    with pytest.raises(CompilationError, match="cannot decide"):
        kernel(Q[0], Parameter("amp", 0, "f64"))


def test_comparing_a_measurement_raises():

    @pulse.kernel
    def k(q0):
        r0, t0 = get_readout_line(q0)
        bit = readout(r0, square(10, 0.2), t0)
        if bit == 0:
            wait(r0, 10)

    with pytest.raises(CompilationError, match="branch on a measurement result directly"):
        k(Q[0])


def test_only_measurements_can_be_conditions():

    @pulse.kernel
    def k(q0):
        d0, t0 = get_drive_line(q0)
        if d0:
            wait(d0, 10)

    with pytest.raises(CompilationError, match="only measurement results"):
        k(Q[0])


_ROUNDS = (0, 1)


def test_nested_feedback_ending_an_unrolled_loop_body():
    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        r, rt = get_readout_line(q)
        m = readout(r, square(10, 0.05), rt)
        for _ in _ROUNDS:
            if m:
                if m:
                    drive(d, square(10, 0.1), t)

    events = k.lower(Q[0]).schedule
    assert [e.kind for e in events] == ["readout", "drive", "drive"]
