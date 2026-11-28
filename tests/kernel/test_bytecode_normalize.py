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
            exit_idx, unrolled = arg
            assert i < exit_idx < len(instrs)
            assert unrolled is False
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


def test_an_augmented_assignment_to_a_name_a_loop_carries_in_keeps_it_unrolled():

    def carried(n, t):
        for _ in range(n):
            t += 1
            g(t)  # pyright: ignore[reportUndefinedVariable]  # noqa: F821

    def bound_in_the_body(n, t):
        for _ in range(n):
            x = t
            x += 1
            g(x)  # pyright: ignore[reportUndefinedVariable]  # noqa: F821

    def skipped(n, t, on):
        for _ in range(n):
            if on:
                t += 1
            g(t)  # pyright: ignore[reportUndefinedVariable]  # noqa: F821

    def continued(n, t, on):
        for _ in range(n):
            if on:
                continue
            t += 1
            g(t)  # pyright: ignore[reportUndefinedVariable]  # noqa: F821

    def nested(n, t):
        for _ in range(n):
            for _ in range(n):
                t += 1
            g(t)  # pyright: ignore[reportUndefinedVariable]  # noqa: F821

    def outside_a_loop(t):
        t += 1
        return t

    def loop(fn):
        instrs = decode(fn.__code__).instrs
        unrolled = [arg[1] for op, arg in instrs if op == bc.FOR_ITER]
        return unrolled, any(op == bc.NOTE_INPLACE for op, _ in instrs)

    # Unrolled without a trace; a rolled trace checks the others' left operand.
    assert loop(carried) == ([True], False)
    assert loop(bound_in_the_body) == ([False], True)
    assert loop(skipped) == ([False], True)
    assert loop(continued) == ([False], True)
    assert loop(nested) == ([True, True], False)
    assert loop(outside_a_loop) == ([], False)


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


_CONDITIONAL = (bc.JUMP_IF_FALSE, bc.JUMP_IF_TRUE, bc.JUMP_IF_NONE, bc.JUMP_IF_NOT_NONE)


def _check_join_points(d, monkeypatch):
    """join_point of each conditional jump, asked in index order, then in
    reverse order with empty caches, equals _join_point_reference's; so do
    those of the ifs inside each region, and taking a region out leaves the
    instructions as they were."""
    branches = [i for i, (op, _) in enumerate(d.instrs) if op in _CONDITIONAL]
    joins = {}
    for order in (branches, branches[::-1]):
        monkeypatch.setattr(bc, "_JOINS", {})
        monkeypatch.setattr(bc, "_POSTDOMS", {})
        for b in order:
            joins[b] = bc.join_point(d, b)
            assert joins[b] == bc._join_point_reference(d, b)
    owned: dict = {}
    for b, join in joins.items():
        if join >= len(d.instrs):
            continue
        view = bc.region(d, join, owned)
        assert view.instrs[join] == (bc.REGION_END, None)
        for c in branches:
            if b <= c < join:
                # An if inside a region joins where it does in the function.
                assert bc.join_point(view, c) == joins[c] == bc._join_point_reference(view, c)
        bc.end_region(view)
        assert tuple(owned[id(d.instrs)][1]) == d.instrs
    return [joins[b] for b in branches]


def test_join_points_of_many_ifs_match_the_reference(monkeypatch):
    src = "def f(m, n, xs):\n    a = 0\n" + "    if m:\n        a += 1\n" * 60 + "    return a\n"
    ns: dict = {}
    exec(compile(src, "<ifs>", "exec"), ns)
    d = decode(ns["f"].__code__)
    joins = _check_join_points(d, monkeypatch)
    # Each if's arms meet where the next if starts, and the last one's before
    # the return.
    assert len(joins) == 60 and joins == sorted(joins) and joins[-1] < len(d.instrs)


def test_join_points_of_nested_and_looping_ifs_match_the_reference(monkeypatch):

    def f(m, n, xs):
        a = 0
        for x in xs:
            if x is None:
                continue
            if m and not n:
                a += 1
            elif n or x:
                if a > 2:
                    break
                a -= 1
            else:
                return a
        while n:
            n -= 1
            if m:
                raise ValueError(a)
        if m:
            return 1 if n else 2
        return a

    _check_join_points(decode(f.__code__), monkeypatch)


def test_a_branch_that_can_loop_forever_asks_the_reference(monkeypatch):
    # 0: if -> 3; 1-2: a loop that never ends; 3: return. Branch 4 reaches
    # the exit as well as the loop.
    instrs = (
        (bc.JUMP_IF_FALSE, 3),
        (bc.POP_TOP, None),
        (bc.JUMP, 1),
        (bc.RETURN, None),
        (bc.JUMP_IF_TRUE, 1),
        (bc.RETURN, None),
    )
    code = (lambda: None).__code__
    d = bc.DecodedCode(code, instrs, (None,) * len(instrs))
    monkeypatch.setattr(bc, "_JOINS", {})
    monkeypatch.setattr(bc, "_POSTDOMS", {})
    calls = []
    real = bc._join_point_reference

    def spy(decoded, branch):
        calls.append(branch)
        return real(decoded, branch)

    monkeypatch.setattr(bc, "_join_point_reference", spy)
    assert [bc.join_point(d, 0), bc.join_point(d, 4)] == [3, 5]
    assert calls == [0, 4]
    assert [real(d, 0), real(d, 4)] == [3, 5]
