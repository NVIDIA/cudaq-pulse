# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Tests for the version-isolated bytecode decoder.

CI runs these on every supported interpreter. Each test states a property of
the canonical instruction stream that must hold on all of them.
"""

import pytest
from cudaq_pulse.kernel import _bytecode_normalize as bc
from cudaq_pulse.kernel._bytecode_normalize import _decode, decode


def _ops(fn):
    return [bc.OP_NAMES[op] for op, _ in decode(fn.__code__).instrs]


def test_decode_is_cached_per_code_object():

    def f(x):
        return x + 1

    assert decode(f.__code__) is decode(f.__code__)
    assert _decode(f.__code__) is not decode(f.__code__)


def test_attribute_call_normalizes_to_load_attr_and_call():

    class _Mod:
        @staticmethod
        def helper():
            return 1

    def attr_call():
        return _Mod.helper()

    ops = _ops(attr_call)
    assert ops[:3] == ["LOAD_DEREF", "LOAD_ATTR", "CALL"]
    assert ops[-1] == "RETURN"


def test_no_interpreter_bookkeeping_survives():

    def f(n):
        for i in range(n):
            g(i)  # noqa: F821
        return None

    ops = set(_ops(f))
    assert ops <= set(bc.OP_NAMES) - {"UNSUPPORTED"}


def test_keyword_call_is_call_kw_with_names_constant():

    def f():
        return g(1, b=2)  # noqa: F821

    instrs = decode(f.__code__).instrs
    kw = [i for i, (op, _) in enumerate(instrs) if op == bc.CALL_KW]
    assert len(kw) == 1 and instrs[kw[0]][1] == 2
    assert instrs[kw[0] - 1] == (bc.LOAD_CONST, ("b",))


def test_jump_targets_are_instruction_indices():

    def f(a, n):
        for _ in range(n):
            if a and n:
                g()  # noqa: F821

    instrs = decode(f.__code__).instrs
    for i, (op, arg) in enumerate(instrs):
        if op == bc.FOR_ITER:
            exit_idx, var_is_read = arg
            assert i < exit_idx < len(instrs)
            assert var_is_read is False
        elif op in (bc.JUMP, bc.JUMP_IF_FALSE, bc.JUMP_IF_TRUE):
            assert 0 <= arg < len(instrs)
            if arg <= i:
                assert instrs[arg][0] == bc.FOR_ITER


def test_loop_var_read_is_detected():

    def f(n):
        for i in range(n):
            g(i)  # noqa: F821

    (loop,) = [arg for op, arg in decode(f.__code__).instrs if op == bc.FOR_ITER]
    assert loop[1] is True


def test_while_loop_decodes_to_unsupported():

    def f(n):
        while n:
            n -= 1

    instrs = decode(f.__code__).instrs
    assert any(op == bc.UNSUPPORTED and "while loops" in arg for op, arg in instrs)


def test_positions_cover_every_instruction():

    def f(x):
        y = x + 1
        return y

    d = decode(f.__code__)
    assert len(d.positions) == len(d.instrs)
    filename, line, end_line, col, end_col = d.location(0)
    assert filename == __file__
    assert line == f.__code__.co_firstlineno + 1
    # The first instruction loads ``x`` in ``y = x + 1``.
    assert (end_line, col, end_col) == (line, 12, 13)
    assert d.location(len(d.instrs) - 1)[1:] == (None, None, None, None)


@pytest.mark.parametrize("name", bc.OP_NAMES)
def test_op_names_match_constants(name):
    assert bc.OP_NAMES[getattr(bc, name)] == name
