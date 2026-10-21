# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Version-isolated bytecode decoder.

This is the ONLY module that knows how CPython lays out bytecode. It decodes
a code object once, with :func:`dis.get_instructions`, into a flat program
over a small canonical instruction set whose semantics are identical on every
supported interpreter (3.12 through 3.15):

* calls are ``CALL n`` / ``CALL_KW n``; the implicit ``NULL`` slots are
  dropped, and 3.12's ``KW_NAMES`` becomes an explicit constant
* binary operators, comparisons and unary operators carry the Python
  ``operator`` function to apply, so no names are matched while tracing
* jump targets are instruction indices, never byte offsets, so a jump to a
  skipped instruction (``EXTENDED_ARG``, ``NOP``, ``END_FOR``) lands on the
  next canonical instruction
* ``FOR_ITER`` consumes the iterator on exit; each version's loop-exit
  cleanup (``END_FOR``, ``POP_TOP``, ``POP_ITER``) is removed here
* superinstructions are split, and ``RETURN_CONST`` becomes ``LOAD_CONST`` +
  ``RETURN``

Constructs the tracer does not support decode to ``UNSUPPORTED`` and raise
only when executed, so the error is the same on every version.

Other interpreters fail fast with ``UnsupportedInterpreterError``. Adding a
CPython version means adding it to ``SUPPORTED_MINORS`` and its loop epilogue
to ``_FOR_EXIT_CLEANUP``, teaching the opname tables any new opnames, and
running ``tests/kernel/test_ir_golden.py`` on it.
"""

from __future__ import annotations

import dis
import operator as _op
import sys
from types import CodeType
from typing import Any

# ── Canonical opcodes ────────────────────────────────────────────────

(
    LOAD_FAST,
    STORE_FAST,
    DELETE_FAST,
    LOAD_CONST,
    LOAD_GLOBAL,
    LOAD_DEREF,
    STORE_DEREF,
    LOAD_ATTR,
    CALL,
    CALL_KW,
    BINARY,
    SUBSCR,
    STORE_SUBSCR,
    BUILD_SLICE,
    UNARY,
    NOT,
    COMPARE,
    IS,
    CONTAINS,
    BUILD_TUPLE,
    BUILD_LIST,
    BUILD_MAP,
    BUILD_CONST_KEY_MAP,
    BUILD_STRING,
    LIST_APPEND,
    LIST_EXTEND,
    UNPACK_SEQUENCE,
    POP_TOP,
    COPY,
    SWAP,
    JUMP,
    JUMP_IF_FALSE,
    JUMP_IF_TRUE,
    JUMP_IF_NONE,
    JUMP_IF_NOT_NONE,
    GET_ITER,
    FOR_ITER,
    RETURN,
    IMPORT_NAME,
    IMPORT_FROM,
    CONVERT,
    FORMAT,
    RAISE,
    UNSUPPORTED,
    REGION_END,
) = range(45)

OP_NAMES: tuple[str, ...] = tuple(
    (
        "LOAD_FAST STORE_FAST DELETE_FAST LOAD_CONST LOAD_GLOBAL LOAD_DEREF "
        "STORE_DEREF LOAD_ATTR CALL CALL_KW BINARY SUBSCR STORE_SUBSCR BUILD_SLICE "
        "UNARY NOT COMPARE IS CONTAINS BUILD_TUPLE BUILD_LIST BUILD_MAP "
        "BUILD_CONST_KEY_MAP BUILD_STRING LIST_APPEND LIST_EXTEND UNPACK_SEQUENCE "
        "POP_TOP COPY SWAP JUMP JUMP_IF_FALSE JUMP_IF_TRUE JUMP_IF_NONE "
        "JUMP_IF_NOT_NONE GET_ITER "
        "FOR_ITER RETURN IMPORT_NAME IMPORT_FROM CONVERT FORMAT RAISE UNSUPPORTED "
        "REGION_END"
    ).split()
)

_JUMPS = frozenset(
    {
        JUMP,
        JUMP_IF_FALSE,
        JUMP_IF_TRUE,
        JUMP_IF_NONE,
        JUMP_IF_NOT_NONE,
        FOR_ITER,
    }
)

# A canonical instruction is an ``(op, arg)`` pair. Jumps carry the target
# instruction index; FOR_ITER carries ``(exit_index, loop_var_is_read)``.
Instr = tuple[int, Any]

# ── Opname table (union over 3.12-3.15; opnames never change meaning) ─

_SKIP = frozenset(
    {
        "CACHE",
        "COPY_FREE_VARS",
        "EXTENDED_ARG",
        "MAKE_CELL",
        "NOP",
        "NOT_TAKEN",
        "PUSH_NULL",
        "RESUME",
        "TO_BOOL",
    }
)

_SIMPLE: dict[str, int] = {
    "LOAD_FAST": LOAD_FAST,
    "LOAD_FAST_CHECK": LOAD_FAST,
    "LOAD_FAST_BORROW": LOAD_FAST,
    "STORE_FAST": STORE_FAST,
    "DELETE_FAST": DELETE_FAST,
    "LOAD_CONST": LOAD_CONST,
    "LOAD_SMALL_INT": LOAD_CONST,
    "LOAD_GLOBAL": LOAD_GLOBAL,
    "LOAD_NAME": LOAD_GLOBAL,
    "LOAD_DEREF": LOAD_DEREF,
    "LOAD_CLOSURE": LOAD_DEREF,
    "STORE_DEREF": STORE_DEREF,
    "LOAD_ATTR": LOAD_ATTR,
    "CALL": CALL,
    "CALL_KW": CALL_KW,
    "BINARY_SUBSCR": SUBSCR,
    "STORE_SUBSCR": STORE_SUBSCR,
    "BUILD_SLICE": BUILD_SLICE,
    "UNARY_NOT": NOT,
    "BUILD_TUPLE": BUILD_TUPLE,
    "BUILD_LIST": BUILD_LIST,
    "BUILD_MAP": BUILD_MAP,
    "BUILD_CONST_KEY_MAP": BUILD_CONST_KEY_MAP,
    "BUILD_STRING": BUILD_STRING,
    "LIST_APPEND": LIST_APPEND,
    "LIST_EXTEND": LIST_EXTEND,
    "UNPACK_SEQUENCE": UNPACK_SEQUENCE,
    "POP_TOP": POP_TOP,
    "POP_ITER": POP_TOP,
    "COPY": COPY,
    "SWAP": SWAP,
    "GET_ITER": GET_ITER,
    "IMPORT_NAME": IMPORT_NAME,
    "IMPORT_FROM": IMPORT_FROM,
    "RAISE_VARARGS": RAISE,
    "RETURN_VALUE": RETURN,
}

_JUMP_NAMES: dict[str, int] = {
    "JUMP_FORWARD": JUMP,
    "JUMP_BACKWARD": JUMP,
    "JUMP_BACKWARD_NO_INTERRUPT": JUMP,
    "POP_JUMP_IF_FALSE": JUMP_IF_FALSE,
    "POP_JUMP_IF_TRUE": JUMP_IF_TRUE,
    "POP_JUMP_IF_NONE": JUMP_IF_NONE,
    "POP_JUMP_IF_NOT_NONE": JUMP_IF_NOT_NONE,
    "FOR_ITER": FOR_ITER,
}

_SUPERINSTRUCTIONS: dict[str, tuple[int, int]] = {
    "LOAD_FAST_LOAD_FAST": (LOAD_FAST, LOAD_FAST),
    "LOAD_FAST_BORROW_LOAD_FAST_BORROW": (LOAD_FAST, LOAD_FAST),
    "STORE_FAST_LOAD_FAST": (STORE_FAST, LOAD_FAST),
    "STORE_FAST_STORE_FAST": (STORE_FAST, STORE_FAST),
}

# BINARY_OP sub-codes, in ``_nb_ops`` order; 26 is the 3.14+ subscript.
_NB_OPS: tuple[Any, ...] = (
    _op.add,
    _op.and_,
    _op.floordiv,
    _op.lshift,
    _op.matmul,
    _op.mul,
    _op.mod,
    _op.or_,
    _op.pow,
    _op.rshift,
    _op.sub,
    _op.truediv,
    _op.xor,
    _op.iadd,
    _op.iand,
    _op.ifloordiv,
    _op.ilshift,
    _op.imatmul,
    _op.imul,
    _op.imod,
    _op.ior,
    _op.ipow,
    _op.irshift,
    _op.isub,
    _op.itruediv,
    _op.ixor,
)
_NB_SUBSCR = 26

_UNARY_NAMES: dict[str, Any] = {
    "UNARY_NEGATIVE": _op.neg,
    "UNARY_INVERT": _op.invert,
}

_COMPARE: dict[str, Any] = {
    "<": _op.lt,
    "<=": _op.le,
    "==": _op.eq,
    "!=": _op.ne,
    ">": _op.gt,
    ">=": _op.ge,
}


def _not_supported(what: str) -> str:
    return f"{what} are not supported in @cudaq_pulse.kernel"


_NESTED_MESSAGE = _not_supported("lambdas, nested functions and comprehensions")
_TRY_MESSAGE = _not_supported("try and with statements")
_UNSUPPORTED: dict[str, str] = {
    "MAKE_FUNCTION": _NESTED_MESSAGE,
    "LOAD_FAST_AND_CLEAR": _NESTED_MESSAGE,
    "BEFORE_WITH": _TRY_MESSAGE,
    "LOAD_SPECIAL": _TRY_MESSAGE,
    "RETURN_GENERATOR": _not_supported("generators"),
    "YIELD_VALUE": _not_supported("generators"),
    "CALL_FUNCTION_EX": _not_supported("star-argument calls"),
}

_WHILE_MESSAGE = (
    "unsupported control flow: while loops are not supported in @cudaq_pulse.kernel; use a for loop"
)

_PY_MINOR = sys.version_info[1]

# CPython minors whose bytecode this decoder has been verified against.
SUPPORTED_MINORS = (12, 13, 14, 15)
ISSUES_URL = "https://github.com/NVIDIA/cudaq-pulse/issues"


class UnsupportedInterpreterError(RuntimeError):
    """The running interpreter's bytecode is not known to the kernel tracer."""


def check_interpreter() -> None:
    """Raise ``UnsupportedInterpreterError`` unless kernels can be traced here."""
    impl = sys.implementation.name
    major, minor = sys.version_info[:2]
    if impl == "cpython" and major == 3 and minor in SUPPORTED_MINORS:
        return
    supported = ", ".join(f"3.{m}" for m in SUPPORTED_MINORS)
    raise UnsupportedInterpreterError(
        f"@cudaq_pulse.kernel traces CPython bytecode and supports CPython "
        f"{supported}; this is {impl} {major}.{minor}. Run kernels on a supported "
        f"interpreter, or report the version at {ISSUES_URL}."
    )


# Instructions at a FOR_ITER exit target that pop the exhausted iterator.
# Canonical FOR_ITER already consumes it.
_FOR_EXIT_CLEANUP: tuple[str, ...] = {
    12: ("END_FOR",),
    13: ("END_FOR", "POP_TOP"),
    14: ("END_FOR", "POP_ITER"),
    15: ("END_FOR", "POP_ITER"),
}.get(_PY_MINOR, ())


def _iter_slots() -> int:
    """Stack slots a for-loop iterator occupies (3.15+ adds one)."""
    op = dis.opmap["GET_ITER"]
    try:
        return 1 + dis.stack_effect(op)
    except ValueError:  # GET_ITER takes an oparg on 3.15+
        return 1 + dis.stack_effect(op, 0)


# Canonical GET_ITER pushes the iterator plus padding to this many slots, so
# ``break`` pops as many slots as the interpreter's bytecode does.
ITER_SLOTS = _iter_slots()

# COMPARE_OP keeps the comparison in its high bits; 3.13+ adds a bool flag.
_COMPARE_ARG_SHIFT = 4 if _PY_MINOR == 12 else 5


def _compare_fn(instr: dis.Instruction) -> Any:
    if isinstance(instr.argval, str):
        name = instr.argval.removeprefix("bool(").removesuffix(")")
    else:
        name = dis.cmp_op[(instr.arg or 0) >> _COMPARE_ARG_SHIFT]
    return _COMPARE[name]


# ── Decoding ─────────────────────────────────────────────────────────


# Source span of an instruction: the ``dis`` objects themselves, shared with
# the decoded instruction stream rather than copied.
Position = dis.Positions


class DecodedCode:
    """A decoded code object: flat canonical instructions.

    ``positions[i]`` is the source span of ``instrs[i]`` (None for the
    synthetic trailing return).
    """

    __slots__ = ("code", "instrs", "positions")

    def __init__(
        self,
        code: CodeType,
        instrs: tuple[Instr, ...],
        positions: tuple[Position | None, ...],
    ):
        self.code = code
        self.instrs = instrs
        self.positions = positions

    def location(self, index: int) -> tuple[str, int | None, int | None, int | None, int | None]:
        """``(filename, line, end_line, col, end_col)`` of instruction *index*."""
        pos = self.positions[index] if 0 <= index < len(self.positions) else None
        if pos is None:
            return (self.code.co_filename, None, None, None, None)
        return (self.code.co_filename, *pos)

    def dump(self) -> str:
        """Readable listing for debugging and tests."""
        lines = []
        for i, (op, arg) in enumerate(self.instrs):
            if callable(arg) and not isinstance(arg, type):
                arg = getattr(arg, "__name__", arg)
            lines.append(f"{i:4d} {OP_NAMES[op]} {arg!r}")
        return "\n".join(lines)


def _protected_starts(code: CodeType) -> list[int]:
    """Byte offsets where an exception-table range begins."""
    if not code.co_exceptiontable:
        return []
    entries = getattr(dis.Bytecode(code), "exception_entries", ())
    return sorted({e.start for e in entries})


def _decode(code: CodeType) -> DecodedCode:
    """Decode *code* into canonical instructions (uncached)."""
    raw = list(dis.get_instructions(code))

    exit_offsets = {ins.argval for ins in raw if ins.opname == "FOR_ITER"}
    protected = _protected_starts(code)
    next_protected = 0

    out: list[list[Any]] = []
    positions: list[Position | None] = []
    pos: Position | None = None
    index_at: dict[int, int] = {}
    pending_kw_names = False
    cleanup = 0  # remaining loop-exit cleanup instructions to drop

    for ins in raw:
        if len(positions) < len(out):
            positions.extend([pos] * (len(out) - len(positions)))
        pos = ins.positions

        name = ins.opname
        off = ins.offset
        iarg = ins.arg if ins.arg is not None else 0
        index_at[off] = len(out)

        if off in exit_offsets:
            cleanup = len(_FOR_EXIT_CLEANUP)
        if cleanup:
            expected = _FOR_EXIT_CLEANUP[len(_FOR_EXIT_CLEANUP) - cleanup]
            cleanup -= 1
            if name == expected:
                continue
            cleanup = 0

        while next_protected < len(protected) and protected[next_protected] <= off:
            next_protected += 1
            out.append([UNSUPPORTED, _TRY_MESSAGE])

        if name in _SKIP:
            continue

        op = _SIMPLE.get(name)
        if op is not None:
            if op == CALL and pending_kw_names:
                op = CALL_KW
                pending_kw_names = False
            out.append([op, ins.argval])
            continue

        op = _JUMP_NAMES.get(name)
        if op is not None:
            out.append([op, ins.argval])  # byte offset, resolved below
            continue

        if name == "BINARY_OP":
            if iarg == _NB_SUBSCR:
                out.append([SUBSCR, None])
            else:
                out.append([BINARY, _NB_OPS[iarg]])
        elif name == "BINARY_SLICE":
            out.append([BUILD_SLICE, 2])
            out.append([SUBSCR, None])
        elif name in _UNARY_NAMES:
            out.append([UNARY, _UNARY_NAMES[name]])
        elif name == "CALL_INTRINSIC_1" and ins.argval == "INTRINSIC_UNARY_POSITIVE":
            out.append([UNARY, _op.pos])
        elif name == "COMPARE_OP":
            out.append([COMPARE, _compare_fn(ins)])
        elif name == "IS_OP":
            out.append([IS, bool(iarg)])
        elif name == "CONTAINS_OP":
            out.append([CONTAINS, bool(iarg)])
        elif name in _SUPERINSTRUCTIONS:
            first, second = _SUPERINSTRUCTIONS[name]
            out.append([first, ins.argval[0]])
            out.append([second, ins.argval[1]])
        elif name == "RETURN_CONST":
            out.append([LOAD_CONST, ins.argval])
            out.append([RETURN, None])
        elif name == "KW_NAMES":
            out.append([LOAD_CONST, code.co_consts[iarg]])
            pending_kw_names = True
        elif name in ("LOAD_ASSERTION_ERROR", "LOAD_COMMON_CONSTANT"):
            value = AssertionError if name == "LOAD_ASSERTION_ERROR" else ins.argval
            out.append([LOAD_CONST, value])
        elif name == "FORMAT_VALUE":
            if iarg & 0x3:
                out.append([CONVERT, iarg & 0x3])
            out.append([FORMAT, bool(iarg & 0x4)])
        elif name == "CONVERT_VALUE":
            out.append([CONVERT, iarg])
        elif name in ("FORMAT_SIMPLE", "FORMAT_WITH_SPEC"):
            out.append([FORMAT, name == "FORMAT_WITH_SPEC"])
        elif name in _UNSUPPORTED:
            out.append([UNSUPPORTED, _UNSUPPORTED[name]])
        else:
            out.append([UNSUPPORTED, f"unsupported bytecode instruction: {name}"])

    positions.extend([pos] * (len(out) - len(positions)))

    # Falling off the end, or jumping past it, returns None.
    end = len(out)
    out.append([LOAD_CONST, None])
    out.append([RETURN, None])
    positions.extend((None, None))

    # CPython versions attribute jumps to different lines. A conditional jump
    # takes the span of the value it tests; a rejected while loop points at
    # its condition: the widest single-line span on the loop's first line.
    for i, entry in enumerate(out):
        op = entry[0]
        if op not in _JUMPS:
            continue
        target = index_at.get(entry[1], end)
        if op == FOR_ITER:
            entry[1] = (target, _loop_var_is_read(out, i, target))
        elif target <= i and out[target][0] != FOR_ITER:
            # Only for-loops jump backwards to their FOR_ITER header.
            out[i] = [UNSUPPORTED, _WHILE_MESSAGE]
            spans = [p for p in positions[target : i + 1] if p is not None and p[0] is not None]
            if spans:
                positions[i] = min(spans, key=_condition_rank)
        else:
            entry[1] = target
            if op != JUMP and i > 0:
                positions[i] = positions[i - 1]

    return DecodedCode(code, tuple((op, arg) for op, arg in out), tuple(positions))


def _condition_rank(p: Any) -> tuple[int, bool, int]:
    multiline = p.end_lineno is not None and p.end_lineno != p.lineno
    width = (p.end_col_offset or 0) - (p.col_offset or 0)
    return (p.lineno or 0, multiline, -width)


def _loop_var_is_read(out: list[list[Any]], for_idx: int, exit_idx: int) -> bool:
    """Whether the loop target is a single local read inside the body."""
    store = out[for_idx + 1] if for_idx + 1 < len(out) else None
    if store is None or store[0] != STORE_FAST:
        return True
    name = store[1]
    for j in range(for_idx + 2, exit_idx):
        op, arg = out[j]
        if arg == name and op in (LOAD_FAST, DELETE_FAST, LOAD_DEREF):
            return True
    return False


def _successors(instrs: tuple[Instr, ...], i: int, exit_node: int) -> tuple[int, ...]:
    op, arg = instrs[i]
    if op == JUMP:
        return (arg,)
    if op in (JUMP_IF_FALSE, JUMP_IF_TRUE, JUMP_IF_NONE, JUMP_IF_NOT_NONE):
        return (i + 1, arg)
    if op == FOR_ITER:
        return (i + 1, arg[0])
    if op in (RETURN, RAISE, UNSUPPORTED):
        return (exit_node,)
    return (i + 1,)


def join_point(decoded: DecodedCode, branch: int) -> int:
    """Immediate post-dominator of the conditional jump at *branch*.

    This is where both arms of an ``if`` meet again. ``len(instrs)`` stands
    for the function exit, reached when both arms return.
    """
    key = (id(decoded), branch)
    hit = _JOINS.get(key)
    if hit is not None and hit[0] is decoded:
        return hit[1]
    instrs = decoded.instrs
    exit_node = len(instrs)
    reachable: list[int] = []
    seen = {branch}
    work = [branch]
    while work:
        node = work.pop()
        reachable.append(node)
        if node == exit_node:
            continue
        for succ in _successors(instrs, node, exit_node):
            if succ not in seen:
                seen.add(succ)
                work.append(succ)
    nodes = sorted(reachable, reverse=True)
    postdom: dict[int, frozenset[int]] = {exit_node: frozenset((exit_node,))}
    everything = frozenset(reachable)
    for node in nodes:
        if node != exit_node:
            postdom[node] = everything
    changed = True
    while changed:
        changed = False
        for node in nodes:
            if node == exit_node:
                continue
            succs = _successors(instrs, node, exit_node)
            common = frozenset.intersection(*(postdom[s] for s in succs))
            new = common | {node}
            if new != postdom[node]:
                postdom[node] = new
                changed = True
    strict = postdom[branch] - {branch}
    join = next(p for p in strict if postdom[p] == strict)
    _JOINS[key] = (decoded, join)
    return join


def region_view(decoded: DecodedCode, join: int) -> DecodedCode:
    """*decoded* with ``REGION_END`` at *join*, so tracing an arm stops there."""
    if join >= len(decoded.instrs):
        return decoded
    key = (id(decoded), join)
    hit = _VIEWS.get(key)
    if hit is not None and hit[0] is decoded:
        return hit[1]
    instrs = list(decoded.instrs)
    instrs[join] = (REGION_END, None)
    view = DecodedCode(decoded.code, tuple(instrs), decoded.positions)
    _VIEWS[key] = (decoded, view)
    return view


_JOINS: dict[tuple[int, int], tuple[DecodedCode, int]] = {}
_VIEWS: dict[tuple[int, int], tuple[DecodedCode, DecodedCode]] = {}
_CACHE: dict[int, DecodedCode] = {}
_CACHE_LIMIT = 4096


def decode(code: CodeType) -> DecodedCode:
    """Decode *code*, caching the result per code object."""
    hit = _CACHE.get(id(code))
    if hit is not None and hit.code is code:
        return hit
    check_interpreter()
    decoded = _decode(code)
    if len(_CACHE) >= _CACHE_LIMIT:
        _CACHE.clear()
    _CACHE[id(code)] = decoded
    return decoded
