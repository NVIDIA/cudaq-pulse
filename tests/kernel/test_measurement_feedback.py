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
from cudaq_pulse.lower import _to_program
from cudaq_pulse.passes.canonicalize import run_canonicalize
from cudaq_pulse.passes.ir_types import Op, OpKind
from cudaq_pulse.passes.pulse_to_operator import run_pulse_to_operator
from cudaq_pulse.passes.scheduling import schedule_alap, schedule_asap
from cudaq_pulse.passes.to_pulse_mlir import program_to_pulse_mlir
from cudaq_pulse.passes.verify import check_if_structure, verify

from tests.kernel.ir_corpus import active_reset, feedback_in_loop, negated_feedback

_FREQ = {0: 5.0e9, 1: 5.1e9}


def _program(kernel, n_qubits=1):
    ir = kernel(*[pulse.qudit_ref() for _ in range(n_qubits)])
    return _to_program(ir, clock_ghz=2.0, qubit_freq_hz=_FREQ)


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


def test_lowered_program_verifies():
    program = _program(active_reset)
    kinds = [op.kind for op in program.ops]
    assert kinds.count(OpKind.IF) == kinds.count(OpKind.ELSE) == kinds.count(OpKind.END_IF) == 1
    _, metrics = schedule_alap(program)
    errors = [e for e in verify(program) if e.severity == "error"]
    assert errors == []


def test_schedule_allows_for_the_longer_arm():
    # readout (300) then the conditional X (40) and a wait (10)
    _, asap = schedule_asap(_program(active_reset))
    _, alap = schedule_alap(_program(active_reset))
    assert asap.total_length_vtu == alap.total_length_vtu == 300 + 40 + 10

    # else arm (wait 40 on d0) vs then arm (shift + 40 on d1); both start
    # after the readout, then a 20 drive on d0.
    _, asap = schedule_asap(_program(negated_feedback, 2))
    assert asap.total_length_vtu == 300 + 40 + 20


def test_conditional_waits_for_the_measurement():
    events, _ = schedule_asap(_program(active_reset))
    readout = next(e for e in events if e.kind == OpKind.READOUT)
    drive = next(e for e in events if e.kind == OpKind.DRIVE)
    assert drive.start_vtu >= readout.end_vtu


def test_feedback_inside_a_retained_loop():
    ir = feedback_in_loop(pulse.qudit_ref())
    kinds = [op.kind for op in ir.ops]
    assert kinds.count("scf.for") == 1 and kinds.count("scf.if") == 1
    _, metrics = schedule_asap(_program(feedback_in_loop))
    # Each readout may start while the previous iteration's conditional
    # drive is still playing on the drive line.
    assert metrics.total_length_vtu == 5 * 300 + 40


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


@pytest.mark.parametrize(
    "stage",
    [
        lambda p: program_to_pulse_mlir(p),
        lambda p: run_pulse_to_operator(p),
        run_canonicalize,
    ],
    ids=["pulse_mlir", "operator", "canonicalize"],
)
def test_unsupported_stages_raise_clearly(stage):
    with pytest.raises(NotImplementedError, match="measurement-conditioned"):
        stage(_program(active_reset))


def test_if_structure_is_verified():
    program = _program(active_reset)
    index = next(i for i, op in enumerate(program.ops) if op.kind == OpKind.ELSE)
    del program.ops[index]
    assert any("has no ELSE" in e.message for e in check_if_structure(program))

    program = _program(active_reset)
    start = next(i for i, op in enumerate(program.ops) if op.kind == OpKind.IF)
    op = program.ops[start]
    line = program.ops[start + 1].operands[0]
    program.ops[start] = Op(op.kind, (line,), op.results, op.attrs)
    assert any("measurement result" in e.message for e in check_if_structure(program))
