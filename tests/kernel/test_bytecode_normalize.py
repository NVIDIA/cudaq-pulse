# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the version-isolated bytecode decoder.

CI runs these on every supported interpreter. Each test states a property of
the canonical instruction stream that must hold on all of them.
"""

import inspect
import types

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
            g(i)  # pyright: ignore[reportUndefinedVariable]  # noqa: F821
        return None

    ops = set(_ops(f))
    assert ops <= set(bc.OP_NAMES) - {"UNSUPPORTED"}


def test_keyword_call_is_call_kw_with_names_constant():

    def f():
        return g(1, b=2)  # pyright: ignore[reportUndefinedVariable]  # noqa: F821

    instrs = decode(f.__code__).instrs
    kw = [i for i, (op, _) in enumerate(instrs) if op == bc.CALL_KW]
    assert len(kw) == 1 and instrs[kw[0]][1] == 2
    assert instrs[kw[0] - 1] == (bc.LOAD_CONST, ("b",))


def test_jump_targets_are_instruction_indices():

    def f(a, n):
        for _ in range(n):
            if a and n:
                g()  # pyright: ignore[reportUndefinedVariable]  # noqa: F821

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
            g(i)  # pyright: ignore[reportUndefinedVariable]  # noqa: F821

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


def _codes(*modules):
    for module in modules:
        stack = [compile(inspect.getsource(module), module.__file__, "exec", dont_inherit=True)]
        while stack:
            code = stack.pop()
            yield code
            stack.extend(k for k in code.co_consts if isinstance(k, types.CodeType))


def _same(a, b):
    if a == b:
        return True
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)) and len(a) == len(b):
        return all(_same(x, y) for x, y in zip(a, b, strict=True))
    return a is b  # a constant unequal to itself, such as nan


def test_the_instruction_reader_decodes_as_dis_does(monkeypatch):
    import ast
    import dis

    from cudaq_pulse.classical import _infer, _infer_rare
    from cudaq_pulse.kernel import _emitter

    codes = list(_codes(ast, dis, inspect, bc, _emitter, _infer, _infer_rare))
    lean = [bc._decode(c) for c in codes]
    monkeypatch.setattr(bc, "_instructions", lambda c: list(dis.get_instructions(c)))
    for code, a in zip(codes, lean, strict=True):
        b = bc._decode(code)
        assert _same(a.instrs, b.instrs), code
        assert a.positions == b.positions, code
        assert a.notes == b.notes, code


def test_notes_say_what_each_unsupported_instruction_was():

    def f(n, xs):
        while n:
            n -= 1

        def g():
            return n

        try:
            xs.append(g())
        except ValueError:
            pass
        return xs

    d = decode(f.__code__)
    notes = {i: d.notes[i] for i, (op, _) in enumerate(d.instrs) if op == bc.UNSUPPORTED}
    assert set(notes) <= set(d.notes)
    kinds = sorted(note[0] for note in notes.values())
    assert "while" in kinds and "try" in kinds and "MAKE_FUNCTION" in kinds
    ((head,),) = [note[1:] for note in notes.values() if note[0] == "while"]
    assert d.instrs[head][0] != bc.UNSUPPORTED or head in notes
    ((start, end, handler, depth, lasti),) = [h for h in d.notes["handlers"]]
    assert start < end <= handler < len(d.instrs)
    assert d.instrs[start][0] == bc.UNSUPPORTED and d.notes[start] == ("try",)
    assert isinstance(depth, int) and isinstance(lasti, bool)


def test_a_comprehension_inside_a_try_is_caught_by_the_try():

    def f(xs):
        try:
            return [x + 1 for x in xs]
        except TypeError:
            return []

    d = decode(f.__code__)
    handlers = {h[2] for h in d.notes["handlers"]}
    assert len(handlers) == 1  # the comprehension's own cleanup re-raises to it


def test_code_without_unsupported_instructions_has_no_notes():

    def f(xs, k):
        if k:
            return xs[1:]
        return xs + [k]

    assert decode(f.__code__).notes is None
