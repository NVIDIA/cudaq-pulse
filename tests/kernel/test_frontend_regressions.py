# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for kernel-frontend correctness fixes."""

from __future__ import annotations

import math
import sys
import typing

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.kernel import _bytecode_normalize as bc
from cudaq_pulse.kernel import _emitter
from cudaq_pulse.kernel.bytecode_bridge import _trace_kernel_with_builder
from cudaq_pulse.kernel.ir_builder import (
    CompilationError,
    IRValue,
    Parameter,
    PythonIRBuilder,
    _SymbolicNumber,
)
from cudaq_pulse.kernel.packed_ir_builder import PackedIRBuilder

from tests._handles import Q, ladder_target
from tests._ir import python_ir

# ── Closures are read on every call ──────────────────────────────────


def _amplitude_kernel():
    amplitude = 0.1

    @pulse.kernel
    def k(q0):
        d0 = get_drive_line(q0)
        drive(d0, square(10, amplitude))

    def rebind(value):
        nonlocal amplitude
        amplitude = value

    return k, rebind


def test_closure_rebinding_takes_effect_on_the_next_call():
    k, rebind = _amplitude_kernel()

    def amp():
        ir = python_ir(k, Q[0])
        return next(op.attrs["amplitude"] for op in ir.ops if op.kind == "pulse.square")

    assert amp() == 0.1
    rebind(0.4)
    assert amp() == 0.4


def test_closure_rebinding_takes_effect_in_compile():
    k, rebind = _amplitude_kernel()
    first = pulse.compile(k, passes=()).lower(Q[0]).mlir()
    rebind(0.4)
    second = pulse.compile(k, passes=()).lower(Q[0]).mlir()
    assert first != second


def test_empty_closure_cell_raises():

    def make():
        @pulse.kernel
        def k(q0):
            d0 = get_drive_line(q0)
            drive(d0, square(10, late))  # pyright: ignore[reportUndefinedVariable]

        return k

        late = 0.1  # noqa: F841 - deliberately unreachable: the cell stays empty

    with pytest.raises(CompilationError, match="'late' is referenced before assignment"):
        make()(Q[0])


# ── Ops are recognized by identity, not by name ──────────────────────


def test_user_function_named_like_an_op_is_evaluated_as_python():

    def gaussian(duration, amplitude, sigma):
        return pulse.ops.square(duration, amplitude)

    @pulse.kernel
    def k(q0):
        d0 = get_drive_line(q0)
        drive(d0, gaussian(10, 0.2, 3.0))

    kinds = [op.kind for op in python_ir(k, Q[0]).ops]
    assert "pulse.square" in kinds
    assert "pulse.gaussian" not in kinds


def test_qualified_and_bare_op_names_trace_identically():

    @pulse.kernel
    def bare(q0):
        d0 = get_drive_line(q0)
        drive(d0, gaussian(40, 0.3, 10.0))

    @pulse.kernel
    def qualified(q0):
        d0 = pulse.ops.get_drive_line(q0)
        pulse.ops.drive(d0, pulse.ops.gaussian(40, 0.3, 10.0))

    def kinds(k):
        return [(op.kind, op.attrs) for op in python_ir(k, Q[0]).ops]

    assert kinds(bare) == kinds(qualified)


# ── No instruction is dropped ────────────────────────────────────────


def test_unknown_opcode_raises_when_executed(monkeypatch):

    def f():
        return 1

    real = bc._instructions

    def renamed(code):
        out = real(code)
        for ins in out:
            if ins.argval == 1:
                ins.opname = "FROBNICATE"
        return out

    monkeypatch.setattr(bc, "_instructions", renamed)
    prog = bc._decode(f.__code__)
    (message,) = [arg for op, arg in prog.instrs if op == bc.UNSUPPORTED]
    assert "(bytecode FROBNICATE" in message and bc.ISSUES_URL in message
    emitter = _emitter.PulseIREmitter(PythonIRBuilder())
    with pytest.raises(CompilationError, match="FROBNICATE"):
        emitter._loop(prog, 0, -1, len(prog.instrs), -1)


def test_stack_copies_are_executed():

    @pulse.kernel
    def k(q0):
        d0 = get_drive_line(q0)
        for i in range(5):
            if 1 <= i < 3:
                wait(d0, 10 * i)
        x = 0
        x += 2
        wait(d0, x)

    ops = [bc.OP_NAMES[op] for op, _ in bc.decode(k.__wrapped__.__code__).instrs]
    assert "COPY" in ops
    waits = [op.attrs["duration"] for op in python_ir(k, Q[0]).ops if op.kind == "pulse.wait"]
    assert waits == [10, 20, 2]


def test_short_stack_swap_raises():
    def f():
        pass

    prog = bc.DecodedCode(f.__code__, ((bc.SWAP, 2), (bc.RETURN, None)), (None, None))
    emitter = _emitter.PulseIREmitter(PythonIRBuilder())
    with pytest.raises(IndexError):
        emitter._loop(prog, 0, -1, 2, -1)


# ── Packed encoding limits ───────────────────────────────────────────


@pytest.mark.parametrize("n", [84, 85, 1200])
def test_a_sync_over_many_lines_takes_an_extended_record(n):
    builder = PackedIRBuilder(name="k")
    lines = tuple(IRValue(i, "drive_line") for i in range(n))
    builder.emit("pulse.sync", lines, tuple(v.vtype for v in lines))
    header = builder.get_buffer()[0]
    payload = (header >> 8) & 0xFF
    if n == 84:
        assert payload == 1 + 3 * n
    else:
        assert payload == 0xFF and builder.get_buffer()[1] == 1 + 3 * n


def test_a_sync_over_a_thousand_lines_compiles_schedules_and_exports():
    n = 1200
    dev = pulse.quantum_system(ladder_target(n))

    @pulse.kernel
    def wide(qs):
        lines = [get_drive_line(q) for q in qs]
        drive(lines[-1], square(40, 0.1))
        sync(*lines)
        for d in lines:
            drive(d, square(8, 0.1))

    lowered = pulse.compile(wide, quantum_system=dev, schedule="asap").lower(dev.qudits[0:n])
    drives = lowered.schedule
    assert len(drives) == n + 1
    assert {e.start_cycles for e in drives[1:]} == {40}
    (barrier,) = [e for e in lowered.export_controller()["events"] if e["kind"] == "sync"]
    assert barrier["lines"] == list(range(n)) and barrier["start_cycles"] == 40


def test_packed_param_names_are_recorded():

    @pulse.kernel
    def k(q0, amp: float, width: int):
        d0 = get_drive_line(q0)
        drive(d0, gaussian(40, amp, 10.0))
        wait(d0, width)

    builder = PackedIRBuilder(name="k")
    params = [Parameter("amp", 0, "f64"), Parameter("width", 1, "i64")]
    _trace_kernel_with_builder(k.__wrapped__, builder, [Q[0], *params])
    assert builder.param_names == ["amp", "width"]


def test_dead_structure_recovery_code_is_gone():
    assert not hasattr(_emitter, "_loop_var_used")
    assert not hasattr(_emitter.PulseIREmitter, "_emit_scf_if")


# ── Interpreter support ──────────────────────────────────────────────


def test_running_interpreter_is_supported():
    bc.check_interpreter()
    assert sys.version_info[1] in bc.SUPPORTED_MINORS


@pytest.mark.parametrize("version", [(3, 16, 0, "alpha", 1), (3, 11, 9, "final", 0)])
def test_unknown_interpreter_fails_at_decoration(monkeypatch, version):
    monkeypatch.setattr(sys, "version_info", version)
    with pytest.raises(bc.UnsupportedInterpreterError) as info:

        @pulse.kernel
        def k(q0):
            pass

    message = str(info.value)
    assert f"{version[0]}.{version[1]}" in message
    assert "3.12, 3.13, 3.14, 3.15" in message and bc.ISSUES_URL in message


# ── Errors ───────────────────────────────────────────────────────────


def test_symbolic_conversions_never_return():
    hints = typing.get_type_hints(_SymbolicNumber._not_concrete)
    assert hints["return"] is typing.NoReturn
    with pytest.raises(CompilationError):
        bool(Parameter("x", 0, "f64"))


def test_evaluated_call_errors_are_chained():

    @pulse.kernel
    def k(q0):
        d0 = get_drive_line(q0)
        wait(d0, int(math.sqrt(-1.0)))

    with pytest.raises(CompilationError, match="failed to evaluate sqrt") as info:
        k(Q[0])
    assert isinstance(info.value.__cause__, ValueError)


class _BoomError(Exception):
    pass


def _explode():
    raise _BoomError


def test_unexpected_exceptions_propagate_unwrapped():

    @pulse.kernel
    def k(q0):
        d0 = get_drive_line(q0)
        wait(d0, _explode())

    with pytest.raises(_BoomError) as info:
        k(Q[0])
    decoded, index = info.value.pulse_source
    assert decoded.location(index)[1] == k.__wrapped__.__code__.co_firstlineno + 3


def test_the_locals_scan_matches_values_by_identity():
    from cudaq_pulse._native._cudaq_pulse_native import identity_keys

    class Loud:
        def __eq__(self, other):
            raise AssertionError("compared with ==")

        __hash__ = object.__hash__

    a, b, c = Loud(), Loud(), Loud()
    assert identity_keys({"x": c, "y": a, "z": b, "w": a}, a, b) == ("y", "z")
    assert identity_keys({"x": b, "y": a}, a, b) == ("y", "x")
    assert identity_keys({"x": a, "y": c, "z": a}, a, a) == ("x", "z")
    assert identity_keys({"x": c}, a, b) == (None, None)


def test_a_sync_of_one_line_with_itself_is_rejected_while_tracing():

    @pulse.kernel
    def k(q0):
        d0 = get_drive_line(q0)
        sync(d0, d0)
        drive(d0, square(40, 0.1))

    with pytest.raises(CompilationError):
        k(Q[0])
