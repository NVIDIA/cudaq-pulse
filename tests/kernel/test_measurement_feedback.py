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
from cudaq_pulse.kernel.ir_builder import CompilationError

from tests.kernel.ir_corpus import active_reset, feedback_in_loop, negated_feedback


def test_active_reset_retains_the_conditional():
    ir = active_reset(pulse.qudit_ref())
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
    ir = negated_feedback(pulse.qudit_ref(), pulse.qudit_ref())
    kinds = [op.kind for op in ir.ops]
    then_arm = kinds[kinds.index("scf.if") + 1 : kinds.index("scf.else")]
    else_arm = kinds[kinds.index("scf.else") + 1 : kinds.index("scf.if_end")]
    assert "pulse.wait" in then_arm and "pulse.drive" not in then_arm
    assert "pulse.drive" in else_arm and "pulse.shift_phase" in else_arm


def test_feedback_inside_a_retained_loop():
    ir = feedback_in_loop(pulse.qudit_ref())
    kinds = [op.kind for op in ir.ops]
    assert kinds.count("scf.for") == 1 and kinds.count("scf.if") == 1


def test_python_state_cannot_depend_on_a_measurement():

    @pulse.kernel
    def k(q0):
        d0, t0 = get_drive_line(q0)
        r0, rt = get_readout_line(q0)
        bit = readout(r0, square(300, 0.05), rt)
        n = 1
        if bit:
            n = 2
        wait(d0, 10 * n)

    with pytest.raises(CompilationError, match="'n' would depend on a measurement result"):
        k(pulse.qudit_ref())


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
        k(pulse.qudit_ref())


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
        k(pulse.qudit_ref())


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
        kernel(pulse.qudit_ref(), pulse.Parameter("amp", 0, "f64"))


def test_comparing_a_measurement_raises():

    @pulse.kernel
    def k(q0):
        r0, t0 = get_readout_line(q0)
        bit = readout(r0, square(10, 0.2), t0)
        if bit == 0:
            wait(r0, 10)

    with pytest.raises(CompilationError, match="branch on a measurement result directly"):
        k(pulse.qudit_ref())


def test_only_measurements_can_be_conditions():

    @pulse.kernel
    def k(q0):
        d0, t0 = get_drive_line(q0)
        if d0:
            wait(d0, 10)

    with pytest.raises(CompilationError, match="only measurement results"):
        k(pulse.qudit_ref())
