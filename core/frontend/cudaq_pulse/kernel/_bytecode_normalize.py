# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Version-isolated bytecode decoder.

This is the ONLY module that knows how CPython lays out bytecode. It decodes
a code object once, with the tables :mod:`dis` reads, into a flat program
over a small canonical instruction set whose semantics are identical on every
supported interpreter (3.12 through 3.15):

* calls are ``CALL n`` / ``CALL_KW n``; the implicit ``NULL`` slots are
  dropped, and 3.12's ``KW_NAMES`` becomes an explicit constant; a call
  with ``*args`` or ``**kwargs`` is ``CALL_EX has_kwargs`` on every version
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
only when executed, so the error is the same on every version. What each
one stood for is kept in ``DecodedCode.notes``, for readers that follow
more of Python than the tracer.

Other interpreters fail fast with ``UnsupportedInterpreterError``. Adding a
CPython version means adding it to ``SUPPORTED_MINORS`` and its loop epilogue
to ``_FOR_EXIT_CLEANUP``, teaching the opname tables any new opnames, and
running ``tests/kernel/test_ir_golden.py`` on it.
"""

from __future__ import annotations

import dis
import opcode
import operator as _op
import sys
from types import CodeType
from typing import Any

from .._errors import UnsupportedError

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
    CALL_EX,
    DICT_MERGE,
    DELETE_SUBSCR,
    LOAD_FAST_AND_CLEAR,
    STORE_FAST_MAYBE,
    BUILD_SET,
    SET_ADD,
    SET_UPDATE,
    MAP_ADD,
    UNPACK_EX,
    MAKE_GENEXPR,
    SET_FUNCATTR,
    GEN_YIELD,
) = range(58)

OP_NAMES: tuple[str, ...] = tuple(
    (
        "LOAD_FAST STORE_FAST DELETE_FAST LOAD_CONST LOAD_GLOBAL LOAD_DEREF "
        "STORE_DEREF LOAD_ATTR CALL CALL_KW BINARY SUBSCR STORE_SUBSCR BUILD_SLICE "
        "UNARY NOT COMPARE IS CONTAINS BUILD_TUPLE BUILD_LIST BUILD_MAP "
        "BUILD_CONST_KEY_MAP BUILD_STRING LIST_APPEND LIST_EXTEND UNPACK_SEQUENCE "
        "POP_TOP COPY SWAP JUMP JUMP_IF_FALSE JUMP_IF_TRUE JUMP_IF_NONE "
        "JUMP_IF_NOT_NONE GET_ITER "
        "FOR_ITER RETURN IMPORT_NAME IMPORT_FROM CONVERT FORMAT RAISE UNSUPPORTED "
        "REGION_END CALL_EX DICT_MERGE DELETE_SUBSCR LOAD_FAST_AND_CLEAR "
        "STORE_FAST_MAYBE BUILD_SET SET_ADD SET_UPDATE MAP_ADD UNPACK_EX "
        "MAKE_GENEXPR SET_FUNCATTR GEN_YIELD"
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
    "DELETE_SUBSCR": DELETE_SUBSCR,
    "BUILD_SLICE": BUILD_SLICE,
    "UNARY_NOT": NOT,
    "BUILD_TUPLE": BUILD_TUPLE,
    "BUILD_LIST": BUILD_LIST,
    "BUILD_MAP": BUILD_MAP,
    "BUILD_CONST_KEY_MAP": BUILD_CONST_KEY_MAP,
    "BUILD_STRING": BUILD_STRING,
    "LIST_APPEND": LIST_APPEND,
    "LIST_EXTEND": LIST_EXTEND,
    "LOAD_FAST_AND_CLEAR": LOAD_FAST_AND_CLEAR,
    "BUILD_SET": BUILD_SET,
    "SET_ADD": SET_ADD,
    "SET_UPDATE": SET_UPDATE,
    "MAP_ADD": MAP_ADD,
    "UNPACK_SEQUENCE": UNPACK_SEQUENCE,
    "POP_TOP": POP_TOP,
    "POP_ITER": POP_TOP,
    "COPY": COPY,
    "SWAP": SWAP,
    "GET_ITER": GET_ITER,
    "IMPORT_NAME": IMPORT_NAME,
    "IMPORT_FROM": IMPORT_FROM,
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


_MODULE_LEVEL = (
    "; define the function at module level, where helpers called from a kernel are inlined"
)
_NESTED_MESSAGES = {
    "<genexpr>": _not_supported("generator expressions")
    + "; use a list comprehension instead, as in sum([x for x in xs])",
    "<lambda>": _not_supported("lambdas") + _MODULE_LEVEL,
}
_NESTED_MESSAGE = _not_supported("nested functions") + _MODULE_LEVEL
_TRY_MESSAGE = _not_supported("try and with statements")
_GENERATORS = _not_supported("generators")
_ASYNC = _not_supported("await, async for and async with")
_MATCH = (
    _not_supported("match statements with sequence, mapping or class patterns")
    + "; match literals and | alternatives, or use if/elif with ==, len() and isinstance()"
)
_GLOBAL = (
    _not_supported("assignments to globals")
    + "; keep the value in a local, or compute it before calling the kernel"
)
# What each opcode the tracer does not run stands for, with what to write instead.
_UNSUPPORTED: dict[str, str] = {
    "BEFORE_WITH": _TRY_MESSAGE,
    "LOAD_SPECIAL": _TRY_MESSAGE,
    "WITH_EXCEPT_START": _TRY_MESSAGE,
    "PUSH_EXC_INFO": _TRY_MESSAGE,
    "POP_EXCEPT": _TRY_MESSAGE,
    "RERAISE": _TRY_MESSAGE,
    "CHECK_EXC_MATCH": _TRY_MESSAGE,
    "CHECK_EG_MATCH": _TRY_MESSAGE,
    "RETURN_GENERATOR": _GENERATORS,
    "YIELD_VALUE": _GENERATORS,
    "GET_YIELD_FROM_ITER": _GENERATORS,
    "SEND": _GENERATORS,
    "END_SEND": _GENERATORS,
    "CLEANUP_THROW": _GENERATORS,
    "GET_AWAITABLE": _ASYNC,
    "GET_AITER": _ASYNC,
    "GET_ANEXT": _ASYNC,
    "END_ASYNC_FOR": _ASYNC,
    "BEFORE_ASYNC_WITH": _ASYNC,
    "MATCH_SEQUENCE": _MATCH,
    "MATCH_MAPPING": _MATCH,
    "MATCH_KEYS": _MATCH,
    "MATCH_CLASS": _MATCH,
    "GET_LEN": _MATCH,
    "STORE_GLOBAL": _GLOBAL,
    "DELETE_GLOBAL": _GLOBAL,
    "STORE_NAME": _GLOBAL,
    "DELETE_NAME": _GLOBAL,
    "DELETE_DEREF": _not_supported("del of a closure variable") + "; rebind it to None instead",
    "STORE_ATTR": _not_supported("attribute assignments (obj.x = ...)")
    + "; build a new object, or keep the value in a local",
    "DELETE_ATTR": _not_supported("del of attributes"),
    "LOAD_BUILD_CLASS": _not_supported("class definitions") + "; define the class at module level",
    "LOAD_SUPER_ATTR": _not_supported("super() calls"),
    "BUILD_TEMPLATE": _not_supported("template strings") + "; use an f-string",
    "BUILD_INTERPOLATION": _not_supported("template strings") + "; use an f-string",
}
_INTRINSICS: dict[str, str] = {
    "INTRINSIC_IMPORT_STAR": _not_supported("star imports (from m import *)")
    + "; import the names the kernel uses at module level",
    "INTRINSIC_STOPITERATION_ERROR": _GENERATORS,
    "INTRINSIC_ASYNC_GEN_WRAP": _GENERATORS,
}

_WHILE_MESSAGE = (
    "unsupported control flow: while loops are not supported in @cudaq_pulse.kernel; use a for loop"
)

_PY_MINOR = sys.version_info[1]
# What LOAD_COMMON_CONSTANT (3.14+) loads, by its argument.
_COMMON_CONSTANTS: tuple[Any, ...] = tuple(getattr(opcode, "_common_constants", ()))

# CPython minors whose bytecode this decoder has been verified against.
SUPPORTED_MINORS = (12, 13, 14, 15)
ISSUES_URL = "https://github.com/NVIDIA/cudaq-pulse/issues"


class UnsupportedInterpreterError(UnsupportedError):
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


def _compare_fn(instr: _Instr) -> Any:
    if isinstance(instr.argval, str):
        name = instr.argval.removeprefix("bool(").removesuffix(")")
    else:
        name = dis.cmp_op[(instr.arg or 0) >> _COMPARE_ARG_SHIFT]
    return _COMPARE[name]


# ── Reading instructions ─────────────────────────────────────────────

# What ``dis.get_instructions`` gives an argument, by opcode: the tables are
# dis's own, read once, so its rules hold on every version without the text
# it formats for each instruction.
_ALL_OPNAME: list[str] = list(getattr(dis, "_all_opname", opcode.opname))
_ALL_OPNAME += [f"<{i}>" for i in range(len(_ALL_OPNAME), 256)]
_DEOPT: list[int] = [dis._deoptop(op) for op in range(256)]  # type: ignore[attr-defined]
_cache_table: Any = getattr(opcode, "_inline_cache_entries", {})
_CACHES: list[int] = [
    _cache_table[op] if isinstance(_cache_table, list) else _cache_table.get(_ALL_OPNAME[op], 0)
    for op in _DEOPT
]
_HAS_ARG = frozenset(getattr(dis, "hasarg", range(dis.HAVE_ARGUMENT, 256)))
(
    _A_PLAIN,
    _A_CONST,
    _A_NAME,
    _A_NAME_2,
    _A_NAME_4,
    _A_JUMP,
    _A_BACK,
    _A_LOCAL,
    _A_LOCAL_PAIR,
    _A_COMPARE,
) = range(10)
_KIND: list[int] = [_A_PLAIN] * 256
for _code in range(256):
    _name = _ALL_OPNAME[_code]
    if _code in dis.hasconst:
        _KIND[_code] = _A_CONST
    elif _code in dis.hasname:
        _KIND[_code] = _A_NAME_2 if _name in ("LOAD_GLOBAL", "LOAD_ATTR") else _A_NAME
        # 3.15 keeps IMPORT_NAME's lazy/eager flags in its low two bits.
        if _name == "LOAD_SUPER_ATTR" or (_name == "IMPORT_NAME" and sys.version_info >= (3, 15)):
            _KIND[_code] = _A_NAME_4
    elif _code in dis.hasjrel:
        _KIND[_code] = _A_BACK if "JUMP_BACKWARD" in _name else _A_JUMP
    elif _name in _SUPERINSTRUCTIONS:
        _KIND[_code] = _A_LOCAL_PAIR
    elif _code in dis.haslocal or _code in dis.hasfree:
        _KIND[_code] = _A_LOCAL
    elif _code in dis.hascompare:
        _KIND[_code] = _A_COMPARE
del _code, _name
_EXTENDED_ARG = dis.EXTENDED_ARG
_INTRINSIC_1 = opcode.opmap.get("CALL_INTRINSIC_1", -1)
_INTRINSIC_1_NAMES: list[str] = list(getattr(dis, "_intrinsic_1_descs", ()))


class _Instr:
    """The fields of ``dis.Instruction`` the decoder reads. ``argrepr`` is
    only filled for ``CALL_INTRINSIC_1``; see :func:`_argrepr`."""

    __slots__ = ("arg", "argrepr", "argval", "offset", "opcode", "opname", "positions")

    def __init__(
        self, opname: str, op: int, arg: int | None, argval: Any, offset: int, positions: Any
    ) -> None:
        self.opname, self.opcode, self.arg, self.argval = opname, op, arg, argval
        self.offset, self.positions, self.argrepr = offset, positions, ""


def _instructions(code: CodeType) -> list[_Instr]:
    """*code*'s instructions with the arguments ``dis.get_instructions``
    resolves (``tests/kernel/test_bytecode_normalize.py`` checks that they
    decode the same)."""
    raw = code.co_code
    positions = [dis.Positions(*p) for p in code.co_positions()]
    consts, names, local = code.co_consts, code.co_names, code._varname_from_oparg  # type: ignore[attr-defined]
    compare_shift = _COMPARE_ARG_SHIFT
    out: list[_Instr] = []
    append = out.append
    extended = 0
    i, n = 0, len(raw)
    while i < n:
        op = _DEOPT[raw[i]]
        if op in _HAS_ARG:
            arg: int | None = raw[i + 1] | extended
            extended = arg << 8 if op == _EXTENDED_ARG else 0  # type: ignore[operator]
        else:
            arg, extended = None, 0
        argval: Any = arg
        if arg is not None:
            kind = _KIND[op]
            if kind == _A_PLAIN:
                pass
            elif kind == _A_LOCAL:
                argval = local(arg)
            elif kind == _A_CONST:
                argval = consts[arg]
            elif kind == _A_NAME_2:
                argval = names[arg >> 1]
            elif kind == _A_NAME:
                argval = names[arg]
            elif kind == _A_JUMP:
                argval = i + 2 + 2 * arg + 2 * _CACHES[op]
            elif kind == _A_BACK:
                argval = i + 2 - 2 * arg + 2 * _CACHES[op]
            elif kind == _A_LOCAL_PAIR:
                argval = (local(arg >> 4), local(arg & 15))
            elif kind == _A_COMPARE:
                argval = dis.cmp_op[arg >> compare_shift]
            else:
                argval = names[arg >> 2]
        ins = _Instr(_ALL_OPNAME[op], op, arg, argval, i, positions[i >> 1])
        if op == _INTRINSIC_1 and arg is not None:
            ins.argrepr = _INTRINSIC_1_NAMES[arg]
        append(ins)
        i += 2 + 2 * _CACHES[op]
    return out


def _argrepr(code: CodeType, offset: int) -> str:
    """``dis``'s text for the instruction at *offset*, for an error message."""
    return next((i.argrepr for i in dis.get_instructions(code) if i.offset == offset), "")


# ── Decoding ─────────────────────────────────────────────────────────


# Source span of an instruction: the ``dis`` objects themselves, shared with
# the decoded instruction stream rather than copied.
Position = dis.Positions


class DecodedCode:
    """A decoded code object: flat canonical instructions.

    ``positions[i]`` is the source span of ``instrs[i]`` (None for the
    synthetic trailing return).

    ``notes`` is None, or says what each ``UNSUPPORTED`` index stood for:
    ``("while", head)`` for a while loop's jump back to *head*, ``("try",)``
    for the start of a ``try`` or ``with`` body, else ``(opname, oparg)``
    of the raw instruction. Under the key ``"handlers"`` it lists
    ``(start, end, handler, depth, lasti)`` per instruction range a
    ``try`` or ``with`` handler catches, as in ``dis``'s exception table
    but in instruction indices; a range whose handler only re-raises
    (an inlined comprehension's) is listed with the handler it reaches.
    """

    __slots__ = ("code", "instrs", "notes", "positions")

    def __init__(
        self,
        code: CodeType,
        instrs: tuple[Instr, ...],
        positions: tuple[Position | None, ...],
        notes: dict[Any, Any] | None = None,
    ):
        self.code = code
        self.instrs = instrs
        self.positions = positions
        self.notes = notes

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


def _protected(
    code: CodeType, raw: list[_Instr]
) -> tuple[list[int], list[tuple[int, int, int, int, bool]]]:
    """Byte offsets where a ``try`` or ``with`` block's exception-table range
    begins, and ``(start, end, handler, depth, lasti)`` per range such a
    handler catches.

    ``try`` and ``with`` handlers start with ``PUSH_EXC_INFO``. An inlined
    comprehension (3.12+) has an entry too, whose handler restores the
    iteration variable and re-raises: its range is listed with the handler
    that catches the re-raise, if any.
    """
    if not code.co_exceptiontable:
        return [], []
    entries = getattr(dis.Bytecode(code), "exception_entries", ())
    handlers = {ins.offset for ins in raw if ins.opname == "PUSH_EXC_INFO"}
    starts = sorted({e.start for e in entries if e.target in handlers})
    out = []
    for e in entries:
        reached = e
        for _ in range(len(entries)):
            if reached.target in handlers:
                out.append((e.start, e.end, reached.target, reached.depth, reached.lasti))
                break
            reached = next((o for o in entries if o.start <= reached.target < o.end), None)
            if reached is None:
                break
    return starts, out


def _decode(code: CodeType) -> DecodedCode:
    """Decode *code* into canonical instructions (uncached)."""
    raw = _instructions(code)

    exit_offsets = {ins.argval for ins in raw if ins.opname == "FOR_ITER"}
    protected, handled = _protected(code, raw)
    notes: dict[Any, Any] = {}
    # Variables a comprehension saves and restores; restoring stores the
    # saved value, or unbinds the name if it had none.
    cleared = {ins.argval for ins in raw if ins.opname == "LOAD_FAST_AND_CLEAR"}
    genexpr_calls = _genexpr_calls(raw)
    calls_with_iterator = set(genexpr_calls.values())
    iterator_slots = {
        raw[i].offset
        for i in range(1, len(raw) - 1)
        if raw[i].opname == "PUSH_NULL"
        and raw[i - 1].offset in calls_with_iterator
        and raw[i + 1].opname == "FOR_ITER"
    }
    # A generator expression's own code: its yields collect its values.
    in_genexpr = code.co_name == "<genexpr>" and bool(code.co_flags & _CO_GENERATOR)
    drop_pop = False
    next_protected = 0

    out: list[list[Any]] = []
    positions: list[Position | None] = []
    pos: Position | None = None
    index_at: dict[int, int] = {}
    pending_kw_names = False
    cleanup = 0  # remaining loop-exit cleanup instructions to drop
    previous: _Instr | None = None  # the raw instruction before *ins*
    asserting = False  # an AssertionError was loaded and not yet raised

    for ins in raw:
        prev, previous = previous, ins
        prev_name = "" if prev is None else prev.opname
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
            notes[len(out)] = ("try",)
            out.append([UNSUPPORTED, _TRY_MESSAGE])

        if name == "PUSH_NULL" and off in iterator_slots:
            # 3.15 runs list(genexpr) inline: a FOR_ITER over the generator,
            # whose second iterator slot this null fills.
            out.append([LOAD_CONST, None])
            continue
        if name in _SKIP:
            continue

        if drop_pop:
            drop_pop = False
            if name == "POP_TOP":
                continue
        if in_genexpr and name == "RETURN_GENERATOR":
            drop_pop = True  # the None it pushes is popped next
            continue

        op = _SIMPLE.get(name)
        if op is not None:
            if op == CALL and pending_kw_names:
                op = CALL_KW
                pending_kw_names = False
            elif op == CALL and off in calls_with_iterator:
                # A generator expression is called with CALL 0, its iterable
                # in the slot a method call's self takes.
                out.append([CALL, 1])
                continue
            elif op == STORE_FAST and ins.argval in cleared:
                op = STORE_FAST_MAYBE
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
        elif name == "STORE_SLICE":
            out.append([BUILD_SLICE, 2])
            out.append([STORE_SUBSCR, None])
        elif name in _UNARY_NAMES:
            out.append([UNARY, _UNARY_NAMES[name]])
        elif name == "CALL_INTRINSIC_1" and ins.argrepr == "INTRINSIC_UNARY_POSITIVE":
            out.append([UNARY, _op.pos])
        elif name == "CALL_INTRINSIC_1" and ins.argrepr == "INTRINSIC_LIST_TO_TUPLE":
            out.append([UNARY, tuple])
        elif name == "CALL_FUNCTION_EX":
            # 3.14+ always has a kwargs slot, a pushed NULL when there are none.
            has_kwargs = bool(iarg & 1) if _PY_MINOR < 14 else prev_name != "PUSH_NULL"
            out.append([CALL_EX, has_kwargs])
        elif name in ("DICT_MERGE", "DICT_UPDATE"):
            out.append([DICT_MERGE, (iarg, name == "DICT_MERGE")])
        elif name == "COMPARE_OP":
            out.append([COMPARE, _compare_fn(ins)])
        elif name == "IS_OP":
            out.append([IS, bool(iarg)])
        elif name == "CONTAINS_OP":
            out.append([CONTAINS, bool(iarg)])
        elif name in _SUPERINSTRUCTIONS:
            for half, var in zip(_SUPERINSTRUCTIONS[name], ins.argval):
                if half == STORE_FAST and var in cleared:
                    half = STORE_FAST_MAYBE
                out.append([half, var])
        elif name == "RETURN_CONST":
            out.append([LOAD_CONST, ins.argval])
            out.append([RETURN, None])
        elif name == "KW_NAMES":
            out.append([LOAD_CONST, code.co_consts[iarg]])
            pending_kw_names = True
        elif name in ("LOAD_ASSERTION_ERROR", "LOAD_COMMON_CONSTANT"):
            # 3.14's dis gives LOAD_COMMON_CONSTANT the table index, 3.15's the value.
            value = AssertionError if name == "LOAD_ASSERTION_ERROR" else _COMMON_CONSTANTS[iarg]
            out.append([LOAD_CONST, value])
            asserting = value is AssertionError
        elif name == "RAISE_VARARGS":
            # ``assert c, msg`` builds AssertionError(msg) with CALL 0: the
            # message sits in the slot a method call's self takes.
            if asserting and prev_name == "CALL" and out[-1] == [CALL, 0]:
                out[-1][1] = 1
            asserting = False
            out.append([RAISE, ins.argval])
        elif name == "FORMAT_VALUE":
            if iarg & 0x3:
                out.append([CONVERT, iarg & 0x3])
            out.append([FORMAT, bool(iarg & 0x4)])
        elif name == "UNPACK_EX":
            out.append([UNPACK_EX, (iarg & 0xFF, iarg >> 8)])
        elif name == "CONVERT_VALUE":
            out.append([CONVERT, iarg])
        elif name in ("FORMAT_SIMPLE", "FORMAT_WITH_SPEC"):
            out.append([FORMAT, name == "FORMAT_WITH_SPEC"])
        elif name == "MAKE_FUNCTION" and off in genexpr_calls:
            closure = _PY_MINOR == 12 and bool(iarg & 0x08)
            out.append([MAKE_GENEXPR, closure])
        elif name == "SET_FUNCTION_ATTRIBUTE":
            out.append([SET_FUNCATTR, None])
        elif in_genexpr and name == "YIELD_VALUE":
            out.append([GEN_YIELD, None])
        elif name == "MAKE_FUNCTION":
            made = prev.argval if prev is not None and prev.opname == "LOAD_CONST" else None
            message = _NESTED_MESSAGES.get(getattr(made, "co_name", ""), _NESTED_MESSAGE)
            notes[len(out)] = (name, iarg)
            out.append([UNSUPPORTED, message])
        elif name in _UNSUPPORTED:
            notes[len(out)] = (name, iarg)
            out.append([UNSUPPORTED, _UNSUPPORTED[name]])
        elif name == "CALL_INTRINSIC_1" and ins.argrepr in _INTRINSICS:
            notes[len(out)] = (name, iarg)
            out.append([UNSUPPORTED, _INTRINSICS[ins.argrepr]])
        else:
            notes[len(out)] = (name, iarg)
            argrepr = _argrepr(code, off)
            what = f"{name} {argrepr}" if argrepr else name
            out.append(
                [
                    UNSUPPORTED,
                    f"this Python construct (bytecode {what}) is not supported in "
                    f"@cudaq_pulse.kernel; please report it at {ISSUES_URL}",
                ]
            )

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
            # A yield in a rolled loop would run once: unroll a generator's loops.
            entry[1] = (target, in_genexpr or _loop_var_is_read(out, i, target))
        elif target <= i and out[target][0] != FOR_ITER:
            # Only for-loops jump backwards to their FOR_ITER header.
            out[i] = [UNSUPPORTED, _WHILE_MESSAGE]
            notes[i] = ("while", target)
            spans = [p for p in positions[target : i + 1] if p is not None and p[0] is not None]
            if spans:
                positions[i] = min(spans, key=_condition_rank)
        else:
            entry[1] = target
            if op != JUMP and i > 0:
                positions[i] = positions[i - 1]

    if handled:
        notes["handlers"] = tuple(
            (index_at[s], index_at.get(e, end), index_at[t], depth, lasti)
            for s, e, t, depth, lasti in handled
        )
    return DecodedCode(code, tuple((op, arg) for op, arg in out), tuple(positions), notes or None)


_CO_GENERATOR = 0x20
_JUMP_OPCODES = (
    frozenset(dis.hasjrel) | frozenset(dis.hasjabs) | frozenset(getattr(dis, "hasjump", ()))
)
_ENDS_FLOW = frozenset(
    {"JUMP_FORWARD", "JUMP_BACKWARD", "JUMP_BACKWARD_NO_INTERRUPT", "RETURN_VALUE", "RETURN_CONST"}
    | {"RAISE_VARARGS", "RERAISE"}
)


def _genexpr_calls(raw: list[_Instr]) -> dict[int, int]:
    """Map the offset of each MAKE_FUNCTION that makes a generator
    expression to the offset of the CALL that runs it.

    The call is the first ``CALL 0`` reached with the function and its
    iterable on the stack; stack depths come from ``dis.stack_effect``.
    A generator expression whose call is not found is left out, and is
    rejected where it is made.
    """
    found: dict[int, int] = {}
    for i, ins in enumerate(raw):
        if ins.opname != "MAKE_FUNCTION" or i == 0 or raw[i - 1].opname != "LOAD_CONST":
            continue
        if getattr(raw[i - 1].argval, "co_name", "") != "<genexpr>":
            continue
        j = i + 1
        while j < len(raw) and raw[j].opname == "SET_FUNCTION_ATTRIBUTE":
            j += 1
        depth = 1  # the function, above whatever the stack held before
        at_target: dict[int, int] = {}
        for k in range(j, len(raw)):
            cur = raw[k]
            if raw[k - 1].opname in _ENDS_FLOW:
                if cur.offset not in at_target:
                    break
                depth = at_target[cur.offset]
            if cur.opname == "CALL" and not cur.arg and depth == 2:
                found[ins.offset] = cur.offset
                break
            oparg = cur.arg if cur.opcode >= dis.HAVE_ARGUMENT else None
            try:
                if cur.opcode in _JUMP_OPCODES:
                    at_target[cur.argval] = depth + dis.stack_effect(cur.opcode, oparg, jump=True)
                depth += dis.stack_effect(cur.opcode, oparg, jump=False)
            except ValueError:
                break
            if depth < 1:
                break
    return found


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
        if arg == name and op in (LOAD_FAST, DELETE_FAST, LOAD_DEREF, LOAD_FAST_AND_CLEAR):
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
    # Tracing an arm stops at its REGION_END, as at a return.
    if op in (RETURN, RAISE, UNSUPPORTED, REGION_END):
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
    view = DecodedCode(decoded.code, tuple(instrs), decoded.positions, decoded.notes)
    _VIEWS[key] = (decoded, view)
    return view


# Each read and write is one dict operation and a hit is checked by identity,
# so threads share these without a lock: a race only decodes twice.
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
