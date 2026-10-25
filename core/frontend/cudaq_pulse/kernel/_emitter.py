# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""PulseIREmitter -- executes decoded kernel bytecode and emits pulse IR.

Kernels are traced by running their canonical instruction stream (see
``_bytecode_normalize``) on an emulated stack. Python values are computed
concretely, so compile-time conditions, short-circuit operators and loop
bounds resolve exactly as Python would; pulse operations (``drive``,
``gaussian``, ...) are emitted as IR operations instead of being called.

``get_drive_line(q)`` and ``get_readout_line(q)`` return the qudit's one
channel of that kind for the whole trace (``_Lines``), threaded linearly
through every op and inlined call that advances it.

A ``for ... in range(n)`` loop whose induction variable is never read is
emitted once as a structured ``scf.for`` when the builder supports regions.
Every other loop, and every loop whose body turns out to carry Python state
from one iteration to the next, is unrolled. Loops over any other concrete
Python iterable (lists, tuples, dicts, ``enumerate``, ``zip``, a qubit
vector, ...) run as Python runs them, one traced iteration per element.
"""

from __future__ import annotations

import builtins as _builtins_mod
import inspect
import types
from collections.abc import Iterator
from typing import Any

from . import _bytecode_normalize as bc
from ._bytecode_normalize import (
    BINARY,
    CALL,
    COMPARE,
    FOR_ITER,
    GET_ITER,
    ITER_SLOTS,
    JUMP,
    JUMP_IF_FALSE,
    JUMP_IF_TRUE,
    LOAD_ATTR,
    LOAD_CONST,
    LOAD_DEREF,
    LOAD_FAST,
    LOAD_GLOBAL,
    POP_TOP,
    STORE_FAST,
    SUBSCR,
    UNPACK_SEQUENCE,
    decode,
)
from ._unroll import resolve_unroll_threshold
from .ir_builder import (
    LINEAR_TYPES,
    OP_TABLE,
    UNDEFINED_VID,
    CompilationError,
    IRValue,
    Parameter,
    ParameterExpression,
    PythonIRBuilder,
    SymbolicValueError,
    is_symbolic,
)

_WAVEFORMS = frozenset(
    {"gaussian", "square", "drag", "cosine", "tanh_ramp", "gaussian_square", "custom"}
)
_WAVEFORM_ARITHMETIC = frozenset({"wf_add", "wf_sub", "wf_mul", "wf_scale", "wf_neg"})
_MAKES_WAVEFORM = _WAVEFORMS | _WAVEFORM_ARITHMETIC | {"custom_samples"}
_PULSE_OP_NAMES = frozenset(OP_TABLE.keys())
_OP_NAMES = {name: f"pulse.{name}" for name in OP_TABLE}

# Loops over Python iterables are unrolled; this bounds a runaway iterator.
MAX_UNROLLED_ITERATIONS = 1_000_000
# Kernel and helper calls are inlined; this bounds a runaway recursion.
MAX_INLINE_DEPTH = 64
# Methods of these containers change them unless listed in _READ_METHODS.
_CONTAINERS = frozenset({list, dict, set})
_READ_METHODS = frozenset({"copy", "count", "index", "get", "keys", "values", "items"})
# A rolled body that acquires a line would acquire one for all iterations
# instead of one per iteration, so the acquisition is hoisted or the loop
# unrolled.
_ACQUIRES = frozenset({"get_drive_line", "get_readout_line"})
_PACKED_BRANCH_MESSAGE = "measurement-conditioned branches need an IR builder that supports regions"
_REJOIN_MESSAGE = (
    "both branches of a measurement-conditioned if must continue after it; "
    "break, continue or return in only one branch is not supported"
)
_BRANCH_TYPE_MESSAGE = (
    "only measurement results can control a branch at runtime; got an IR value of type {}"
)

# The emulated stack mixes IR values with arbitrary Python constants
# (tuples, dicts, callables), so entries are dynamically typed.
StackVal = Any

_BUILTINS = vars(_builtins_mod)
_SCALARS = (int, float, complex, str, bytes, bool, type(None))

# Bad-argument failures of a Python call evaluated at trace time. Anything
# else (CompilationError, MemoryError, KeyboardInterrupt, ...) propagates.
_EVALUATION_ERRORS = (TypeError, ValueError, ArithmeticError, LookupError)


class _EmptyCell:
    """A closure variable that was never assigned in the enclosing scope."""

    __slots__ = ()


_EMPTY_CELL = _EmptyCell()


def _extract_closures(fn: Any) -> dict[str, Any]:
    """Return the values captured by a Python function, read now.

    Called on every trace, so rebinding a captured variable between calls
    takes effect; the decoded bytecode is cached separately.
    """
    closures: dict[str, Any] = {}
    cells = getattr(fn, "__closure__", None)
    if cells and fn.__code__.co_freevars:
        for name, cell in zip(fn.__code__.co_freevars, cells):
            try:
                closures[name] = cell.cell_contents
            except ValueError:
                closures[name] = _EMPTY_CELL
    return closures


_PULSE_FUNCTIONS: dict[Any, str] = {}


def _load_pulse_functions() -> None:
    """Map each DSL function object, by identity, to its op name."""
    from .. import ops

    for name in _PULSE_OP_NAMES:
        op = getattr(ops, name, None)
        if op is not None:
            _PULSE_FUNCTIONS[op] = name


def _uses_pulse_ops(fn: types.FunctionType, seen: set[int] | None = None) -> bool:
    """Whether *fn* directly or transitively calls the pulse DSL."""
    seen = seen or set()
    if id(fn) in seen:
        return False
    seen.add(id(fn))

    closures = _extract_closures(fn) if fn.__closure__ else {}
    for name in (*fn.__code__.co_names, *fn.__code__.co_freevars):
        if name in _PULSE_OP_NAMES:
            return True
        value = closures.get(name, fn.__globals__.get(name))
        if getattr(value, "__pulse_kernel_fn__", None) is not None:
            return True
        if isinstance(value, types.FunctionType) and _uses_pulse_ops(value, seen):
            return True
    return False


def _truth(value: StackVal) -> bool:
    if isinstance(value, (IRValue, _NotMeasurement)):
        raise CompilationError(
            "a measurement result is only known at runtime; use it directly as "
            "an if condition (`if result:` or `if not result:`)"
        )
    return bool(value)


class _NotMeasurement:
    """``not result`` for a measurement result, pending use as a condition."""

    __slots__ = ("value",)

    def __init__(self, value: IRValue):
        self.value = value


class _RegionDone:
    """Returned by ``_loop`` when an if-arm reaches its join point."""

    __slots__ = ()


_REGION_DONE = _RegionDone()


class _Unrolled:
    """Iterator state of a ``for`` loop over a concrete Python iterable."""

    __slots__ = ("it", "count")

    def __init__(self, it: Any):
        self.it = it
        self.count = 0


_MISSING = object()

_LINE_OPS = frozenset({"get_drive_line", "get_readout_line"})


class _Chan:
    """The current value of a channel's line (index 0) or of its tone (1)."""

    __slots__ = ("value", "shared", "held", "index")

    def __init__(self, value: IRValue, held: _Held, index: int) -> None:
        self.value = value
        value.chan = self
        # Handed out more than once: names holding old values may be stale.
        self.shared = False
        self.held = held
        self.index = index


class _Held:
    """The current (line, tone) of one qudit's drive or readout channel.

    Indexing reads and writes the line (0) and the tone (1); a value written
    records the channel in ``IRValue.chan``.
    """

    __slots__ = ("cells", "key")

    def __init__(self, values: tuple[IRValue, ...], key: tuple[int, str]) -> None:
        line, tone = values
        self.cells = (_Chan(line, self, 0), _Chan(tone, self, 1))
        self.key = key

    def __getitem__(self, index: int) -> IRValue:
        return self.cells[index].value

    def __setitem__(self, index: int, value: IRValue) -> None:
        cell = self.cells[index]
        cell.value = value
        value.chan = cell

    def __iter__(self) -> Iterator[IRValue]:
        return (cell.value for cell in self.cells)

    def __len__(self) -> int:
        return len(self.cells)

    @property
    def label(self) -> str:
        return f"{self.key[1].removeprefix('get_')}{self.key[0]}"


class _Lines:
    """The drive and readout channels acquired so far in one trace.

    ``get_drive_line(q)`` returns the one drive channel of ``q`` for the
    whole program: the first call acquires it, later calls, in the kernel or
    in a kernel it calls, return its current line and tone, so pulses on it
    play one after another. Each value a channel takes records the channel
    in ``IRValue.chan``, so an op that consumes it moves the channel to the
    op's result, and a name left holding an old value of a shared channel
    can be brought up to date.
    """

    __slots__ = ("held", "shared", "log")

    def __init__(self) -> None:
        self.held: dict[tuple[int, str], _Held] = {}
        # Whether any channel was handed out more than once.
        self.shared = False
        # Line acquisitions, in order: those a rolled loop body makes are
        # hoisted before the loop.
        self.log: list[tuple[str, IRValue]] = []

    def add(self, key: tuple[int, str], values: tuple[IRValue, ...]) -> None:
        self.held[key] = _Held(values, key)

    def share(self, held: _Held) -> None:
        for cell in held.cells:
            cell.shared = True
        self.shared = True

    @staticmethod
    def current(value: IRValue) -> IRValue | None:
        """The channel value now standing for *value*, if it is a channel's."""
        chan: _Chan | None = value.chan
        return None if chan is None else chan.value


# Host calls evaluated at trace time whose result depends only on their
# arguments; any other C-level callable (``random.random``, ``time.time``)
# makes the trace impure, so its cached compilation is never reused.
_PURE_MODULES = frozenset({"math", "cmath", "builtins", "operator", "functools", "itertools"})


def _is_pure_host_call(fn: Any) -> bool:
    if isinstance(fn, (type, types.FunctionType)):
        return True
    if isinstance(fn, types.MethodType):
        return isinstance(fn.__func__, types.FunctionType)
    owner = getattr(fn, "__self__", None)
    if owner is not None and not isinstance(owner, types.ModuleType):
        return type(owner).__module__ == "builtins"
    module = getattr(fn, "__module__", None) or ""
    if module in _PURE_MODULES:
        return True
    return module.split(".")[0] == "numpy" and not module.startswith("numpy.random")


def _parameter_names(value: Any) -> list[str]:
    """Names of the kernel parameters a symbolic value depends on."""
    if isinstance(value, Parameter):
        return [value.name]
    if isinstance(value, ParameterExpression):
        return _parameter_names(value.lhs) + _parameter_names(value.rhs)
    return []


def _range_error(args: list[Any]) -> CompilationError:
    names = sorted({n for a in args for n in _parameter_names(a)})
    if names:
        return SymbolicValueError(
            "range() bound depends on kernel parameter "
            + ", ".join(repr(n) for n in names)
            + "; a float parameter stays symbolic, so annotate it as int to "
            "specialize the kernel on its value",
            tuple(names),
        )
    kinds = ", ".join(type(a).__name__ for a in args)
    return CompilationError(f"range() bounds must be integers known at compile time; got ({kinds})")


def _iterate(value: Any) -> Any:
    """``iter(value)`` for a ``for`` loop, with a kernel-specific error."""
    if isinstance(value, IRValue):
        raise CompilationError(
            f"cannot iterate over a pulse {value.vtype} value; for loops "
            "in pulse kernels run over Python values known at compile time"
        )
    if is_symbolic(value):
        names = _parameter_names(value)
        raise SymbolicValueError(
            f"cannot iterate over kernel parameter {', '.join(map(repr, names))}", tuple(names)
        )
    try:
        return iter(value)
    except TypeError:
        raise CompilationError(
            f"cannot iterate over {type(value).__name__} in a pulse kernel"
        ) from None


class _Loop:
    """Iterator state of a ``for ... in range()`` loop on the stack."""

    __slots__ = ("rng", "it", "rolled")

    def __init__(self, rng: range):
        self.rng = rng
        self.it = iter(rng)
        self.rolled: bool | None = None  # None until the header decides


class _NotRollableError(Exception):
    """The single traced iteration of a rolled loop left the loop body."""


class PulseIREmitter:
    """Traces decoded bytecode and emits pulse IR via a builder."""

    __slots__ = (
        "b",
        "stack",
        "locals",
        "_fn_globals",
        "_fn_closures",
        "_builtins",
        "_can_roll",
        "_slots",
        "_durations",
        "_mutated",
        "_depth",
        "_acquired",
        "_lines",
        "_unroll",
    )

    _can_roll: bool
    _slots: dict[int, tuple[list[Any], int, IRValue]]
    _durations: dict[int, int]
    _mutated: dict[int, tuple[Any, Any]] | None
    _depth: int
    _acquired: list[int]
    _lines: _Lines
    _unroll: int

    def __init__(
        self,
        builder: Any,
        fn_globals: dict[str, Any] | None = None,
        fn_closures: dict[str, Any] | None = None,
        caller: PulseIREmitter | None = None,
    ):
        self.b = builder
        self.stack: list[StackVal] = []
        self.locals: dict[str, StackVal] = {}
        self._fn_globals = {} if fn_globals is None else fn_globals
        self._fn_closures = {} if fn_closures is None else fn_closures
        self._builtins = _BUILTINS
        if caller is not None:
            # An inlined call shares the trace's state with its caller.
            self._can_roll = caller._can_roll
            self._slots = caller._slots
            self._durations = caller._durations
            self._mutated = caller._mutated
            self._depth = caller._depth + 1
            self._acquired = caller._acquired
            self._lines = caller._lines
            self._unroll = caller._unroll
            return
        self._can_roll = getattr(builder, "supports_regions", False)
        self._unroll = resolve_unroll_threshold(getattr(builder, "unroll_threshold", None))
        # id(line or tone) -> (list, index, value) for values read out of a
        # list, so the op that consumes one can rebind the list slot.
        self._slots = {}
        # Waveform vid -> duration in clock cycles, when known at trace time.
        self._durations = {}
        # While a rolled loop body or a branch arm is traced: id -> (container,
        # contents before) for each list, dict or set a Python call changed.
        self._mutated = None
        # Number of enclosing inlined kernel and helper calls.
        self._depth = 0
        # Lines acquired so far, counted across inlined calls.
        self._acquired = [0]
        self._lines = _Lines()
        if not _PULSE_FUNCTIONS:
            _load_pulse_functions()

    # ── Public entry point ───────────────────────────────────────────

    def run(self, code: types.CodeType) -> StackVal:
        """Trace *code* with the current locals; return its return value.

        An exception raised while tracing gets a ``pulse_source`` attribute,
        ``(DecodedCode, instruction_index)``, naming the innermost kernel or
        helper instruction that failed; ``DecodedCode.location(index)``
        resolves it to a source span.
        """
        prog = decode(code)
        return self._loop(prog, 0, -1, len(prog.instrs), -1)

    # ── Interpreter loop ─────────────────────────────────────────────

    def _loop(
        self,
        prog: bc.DecodedCode,
        pc: int,
        lo: int,
        hi: int,
        header: int,
        # Tracing hot loop: opcodes bound as defaults are plain local reads, and
        # are small ints, which CPython caches, so ``is`` compares them exactly.
        _load_fast: int = LOAD_FAST,
        _load_const: int = LOAD_CONST,
        _load_deref: int = LOAD_DEREF,
        _call: int = CALL,
        _store_fast: int = STORE_FAST,
        _load_global: int = LOAD_GLOBAL,
        _load_attr: int = LOAD_ATTR,
        _subscr: int = SUBSCR,
        _binary: int = BINARY,
        _compare: int = COMPARE,
        _jif: int = JUMP_IF_FALSE,
        _jit: int = JUMP_IF_TRUE,
        _jump: int = JUMP,
        _pop_top: int = POP_TOP,
        _for_iter: int = FOR_ITER,
        _get_iter: int = GET_ITER,
        _unpack: int = UNPACK_SEQUENCE,
        _return: int = bc.RETURN,
        _build_tuple: int = bc.BUILD_TUPLE,
        _build_list: int = bc.BUILD_LIST,
        _iter_slots: int = ITER_SLOTS,
    ) -> Any:
        """Run from *pc*. Jumps must stay inside ``(lo, hi)``.

        With ``header >= 0`` this traces one iteration of a rolled loop:
        it returns when control jumps back to the loop header and raises
        ``_NotRollableError`` if control leaves the body any other way.
        An exception raised here gets ``pulse_source = (prog, index)`` unless
        a nested trace already attached one.
        """
        instrs = prog.instrs
        stack = self.stack
        local_vars = self.locals
        closures = self._fn_closures
        fn_globals = self._fn_globals
        push = stack.append
        pop = stack.pop

        try:
            while True:
                op, arg = instrs[pc]
                pc += 1

                if op is _load_fast:
                    try:
                        push(local_vars[arg])
                    except KeyError:
                        raise CompilationError(f"undefined local: {arg}") from None

                elif op is _load_global:
                    value = fn_globals.get(arg, _MISSING)
                    push(self._resolve_global(arg) if value is _MISSING else value)

                elif op is _call:
                    if arg:
                        call_args = stack[-arg:]
                        del stack[-arg:]
                    else:
                        call_args = []
                    self._dispatch_call(pop(), call_args, None)

                elif op is _pop_top:
                    pop()

                elif op is _load_deref:
                    value = local_vars.get(arg, _MISSING)
                    if value is _MISSING:
                        value = closures.get(arg, _MISSING)
                        if value is _MISSING:
                            value = self._resolve_global(arg)
                        elif value is _EMPTY_CELL:
                            raise CompilationError(
                                f"closure variable {arg!r} is referenced before assignment"
                            )
                    push(value)

                elif op is _load_const:
                    push(arg)

                elif op is _store_fast:
                    local_vars[arg] = pop()

                elif op is _return:
                    if header >= 0:
                        raise _NotRollableError
                    return pop()

                elif op is _load_attr:
                    obj = pop()
                    if isinstance(obj, tuple) and len(obj) == 2 and obj[0] == "__unresolved__":
                        push(("__unresolved__", arg))
                    else:
                        try:
                            push(getattr(obj, arg))
                        except (AttributeError, TypeError) as e:
                            raise CompilationError(
                                f"cannot resolve attribute {arg!r} on {type(obj).__name__}"
                            ) from e

                elif op is _subscr:
                    index = pop()
                    obj = pop()
                    try:
                        value = obj[index]
                    except (TypeError, IndexError, KeyError) as e:
                        raise CompilationError(
                            f"subscript failed: {type(obj).__name__}[{index!r}]: {e}"
                        ) from e
                    push(value)
                    if (
                        value.__class__ is IRValue
                        and obj.__class__ is list
                        and value.vtype in LINEAR_TYPES
                    ):
                        self._slots[id(value)] = (obj, index, value)

                elif op is _binary or op is _compare:
                    right = pop()
                    left = pop()
                    push(arg(left, right))

                elif op is _jif or op is _jit:
                    cond = pop()
                    if cond.__class__ is not bool:
                        if isinstance(cond, (IRValue, _NotMeasurement)):
                            pc = self._runtime_branch(prog, pc - 1, cond, op is _jit)
                            pc = self._jump(pc, lo, hi, header)
                            if pc < 0:
                                return None
                            continue
                        cond = _truth(cond)
                    if cond == (op is _jit):
                        pc = self._jump(arg, lo, hi, header)
                        if pc < 0:
                            return None

                elif op is _jump:
                    pc = self._jump(arg, lo, hi, header)
                    if pc < 0:
                        return None

                elif op is _for_iter:
                    loop = stack[-_iter_slots]
                    if type(loop) is _Unrolled:
                        value = next(loop.it, _MISSING)
                        if value is _MISSING:
                            del stack[-_iter_slots:]
                            pc = self._jump(arg[0], lo, hi, header)
                            if pc < 0:
                                return None
                            continue
                        loop.count += 1
                        if loop.count > MAX_UNROLLED_ITERATIONS:
                            raise CompilationError(
                                f"for loop did not finish after {MAX_UNROLLED_ITERATIONS} "
                                "iterations; loops over Python iterables are unrolled, "
                                "so they must be finite"
                            )
                        push(value)
                        continue
                    exit_idx, var_is_read = arg
                    if loop.rolled is None:
                        loop.rolled = (
                            self._can_roll
                            and not var_is_read
                            and not self._unrolls(loop, prog, pc - 1, exit_idx)
                            and self._roll(loop, prog, pc - 1, exit_idx)
                        )
                    value = loop if loop.rolled else next(loop.it, loop)
                    if value is loop:
                        del stack[-_iter_slots:]
                        pc = self._jump(exit_idx, lo, hi, header)
                        if pc < 0:
                            return None
                    else:
                        push(value)

                elif op is _get_iter:
                    value = pop()
                    push(_Loop(value) if type(value) is range else _Unrolled(_iterate(value)))
                    if _iter_slots > 1:
                        stack.extend([None] * (_iter_slots - 1))

                elif op is _unpack:
                    val = pop()
                    if not isinstance(val, (tuple, list)):
                        raise CompilationError(f"cannot unpack {type(val)}")
                    if len(val) != arg:
                        raise CompilationError(f"unpack mismatch: expected {arg}, got {len(val)}")
                    stack.extend(reversed(val))

                elif op is _build_tuple or op is _build_list:
                    if arg:
                        items = stack[-arg:]
                        del stack[-arg:]
                    else:
                        items = []
                    if op is _build_tuple:
                        push(tuple(items))
                    else:
                        push(items)
                        self._track_slots(items, 0)
                        self._note_fresh(items)

                elif op is bc.LIST_APPEND or op is bc.LIST_EXTEND:
                    val = pop()
                    lst = stack[-arg]
                    if not isinstance(lst, list):
                        raise CompilationError(f"list update on non-list: {type(lst)}")
                    if op is bc.LIST_APPEND:
                        lst.append(val)
                    else:
                        lst.extend(val)

                elif op is bc.UNARY:
                    push(arg(pop()))

                elif op is bc.NOT:
                    value = pop()
                    if isinstance(value, IRValue):
                        push(_NotMeasurement(value))
                    elif isinstance(value, _NotMeasurement):
                        push(value.value)
                    else:
                        push(not _truth(value))

                elif op is bc.IS:
                    right = pop()
                    left = pop()
                    push((left is right) != arg)

                elif op is bc.CONTAINS:
                    right = pop()
                    left = pop()
                    push((left in right) != arg)

                elif op is bc.CALL_KW:
                    names = pop()
                    values = stack[-arg:]
                    del stack[-arg:]
                    n_pos = arg - len(names)
                    kwargs = dict(zip(names, values[n_pos:]))
                    self._dispatch_call(pop(), values[:n_pos], kwargs)

                elif op is bc.COPY:
                    push(stack[-arg])

                elif op is bc.SWAP:
                    stack[-1], stack[-arg] = stack[-arg], stack[-1]

                elif op is bc.JUMP_IF_NONE or op is bc.JUMP_IF_NOT_NONE:
                    if (pop() is None) == (op is bc.JUMP_IF_NONE):
                        pc = self._jump(arg, lo, hi, header)
                        if pc < 0:
                            return None

                elif op is bc.STORE_DEREF:
                    local_vars[arg] = pop()

                elif op is bc.DELETE_FAST:
                    local_vars.pop(arg, None)

                elif op is bc.BUILD_SLICE:
                    items = stack[-arg:]
                    del stack[-arg:]
                    push(slice(*items))

                elif op is bc.STORE_SUBSCR:
                    index = pop()
                    obj = pop()
                    value = pop()
                    self._note_mutation(obj)
                    obj[index] = value

                elif op is bc.BUILD_CONST_KEY_MAP:
                    keys = pop()
                    vals = stack[-arg:] if arg else []
                    del stack[len(stack) - arg :]
                    push(dict(zip(keys, vals)))
                    self._note_fresh(stack[-1])

                elif op is bc.BUILD_MAP:
                    items = stack[len(stack) - 2 * arg :]
                    del stack[len(stack) - 2 * arg :]
                    push(dict(zip(items[::2], items[1::2])))
                    self._note_fresh(stack[-1])

                elif op is bc.BUILD_STRING:
                    parts = stack[-arg:] if arg else []
                    del stack[len(stack) - arg :]
                    push("".join(parts))

                elif op is bc.CONVERT:
                    push((str, repr, ascii)[arg - 1](pop()))

                elif op is bc.FORMAT:
                    spec = pop() if arg else ""
                    push(format(pop(), spec))

                elif op is bc.IMPORT_NAME:
                    fromlist = pop()
                    level = pop()
                    push(__import__(arg, self._fn_globals, None, fromlist, level))

                elif op is bc.IMPORT_FROM:
                    push(getattr(stack[-1], arg))

                elif op is bc.RAISE:
                    exc = pop() if arg else None
                    raise CompilationError(f"kernel raised {exc!r}")

                elif op is bc.UNSUPPORTED:
                    raise CompilationError(arg)

                elif op is bc.REGION_END:
                    return _REGION_DONE

                else:  # pragma: no cover - decode only produces the ops above
                    raise CompilationError(f"unknown canonical instruction {op}")
        except _NotRollableError:
            raise
        except Exception as e:
            if getattr(e, "pulse_source", None) is None:
                try:
                    e.pulse_source = (prog, pc - 1)  # type: ignore[attr-defined]
                except AttributeError:
                    pass
            raise

    # ── Measurement-conditioned branches ─────────────────────────────

    def _runtime_branch(
        self, prog: bc.DecodedCode, index: int, cond: Any, jump_if_true: bool
    ) -> int:
        """Emit the ``if`` whose conditional jump is ``prog.instrs[index]``.

        Both arms are traced from the same state into an ``scf.if`` region.
        Pulse resources rebound by either arm become results of the region;
        any other value must end up the same in both arms. Returns the
        instruction index where tracing continues.
        """
        negated = isinstance(cond, _NotMeasurement)
        value = cond.value if negated else cond
        if value.vtype != "measurement":
            raise CompilationError(_BRANCH_TYPE_MESSAGE.format(value.vtype))
        if not self._can_roll:
            raise CompilationError(_PACKED_BRANCH_MESSAGE)

        target = prog.instrs[index][1]
        then_start, else_start = (target, index + 1) if jump_if_true else (index + 1, target)
        if negated:
            then_start, else_start = else_start, then_start
        join = bc.join_point(prog, index)
        view = bc.region_view(prog, join)
        returns = join >= len(prog.instrs)

        builder = self.b
        stack = self.stack
        saved_stack = list(stack)
        saved_locals = dict(self.locals)
        containers = [
            (v, list(v) if isinstance(v, list) else dict(v))
            for v in saved_locals.values()
            if isinstance(v, (list, dict))
        ]

        # Each channel value entering the branch, as (cell, value).
        held = self._lines.held
        n_held = len(held)
        entering = _channel_values(held) if n_held else ()

        builder.emit("scf.if", (value,), ())
        then_locals, then_slots = self._trace_arm(
            view, index, join, then_start, saved_stack, containers
        )
        then_held = [cell.value for cell, _ in entering]
        stack[:] = saved_stack
        self.locals.clear()
        self.locals.update(saved_locals)
        for obj, copy in containers:
            self._restore_container(obj, copy)
        _restore_channels(held, n_held, entering)
        else_index = builder.mark()
        builder.emit("scf.else", (), ())
        else_locals, else_slots = self._trace_arm(
            view, index, join, else_start, saved_stack, containers
        )

        merged: dict[str, StackVal] = {}
        results: list[tuple[str, IRValue, IRValue]] = []
        for name in sorted(then_locals.keys() | else_locals.keys()):
            a = then_locals.get(name, _MISSING)
            b = else_locals.get(name, _MISSING)
            if a is b:
                merged[name] = a
            elif isinstance(a, IRValue) and isinstance(b, IRValue) and a.vtype == b.vtype:
                results.append((name, a, b))
            elif a is _MISSING or b is _MISSING:
                continue  # bound in one arm only: unbound after the if
            elif type(a) is type(b) and isinstance(a, _SCALARS) and a == b:
                merged[name] = a
            else:
                raise CompilationError(
                    f"{name!r} would depend on a measurement result; only pulse "
                    "resources (lines, tones, waveforms) may differ between the "
                    "arms of a measurement-conditioned if"
                )
        # List slots holding lines or tones that either arm rebound.
        slot_results: list[tuple[Any, int, IRValue, IRValue]] = []
        originals = {id(obj): copy for obj, copy in containers}
        keys: dict[tuple[int, int], list[Any]] = {}
        for obj, i, _ in then_slots + else_slots:
            keys[(id(obj), i)] = obj
        then_new = {(id(obj), i): new for obj, i, new in then_slots}
        else_new = {(id(obj), i): new for obj, i, new in else_slots}
        for key, obj in keys.items():
            old = originals[key[0]][key[1]]
            a = then_new.get(key, old)
            b = else_new.get(key, old)
            if a is not b:
                slot_results.append((obj, key[1], a, b))

        # Channels either arm advanced become results too, unless a name
        # carries the same pair of values.
        aliases: list[tuple[_Chan, str]] = []
        by_pair: dict[tuple[int, int], str] | None = None
        for (cell, before), a in zip(entering, then_held):
            b = cell.value
            if a is before and b is before:
                continue
            if by_pair is None:
                by_pair = {(id(x), id(y)): name for name, x, y in results}
            alias = by_pair.get((id(a), id(b)))
            if alias is None:
                slot_results.append((cell.held, cell.index, a, b))
            else:
                aliases.append((cell, alias))
        labels = [name for name, _, _ in results]
        labels += [self._slot_label(obj, i) for obj, i, _, _ in slot_results]
        then_vals = tuple(a for _, a, _ in results) + tuple(a for _, _, a, _ in slot_results)
        else_vals = tuple(b for _, _, b in results) + tuple(b for _, _, _, b in slot_results)
        builder.insert(else_index, "scf.yield", then_vals)
        builder.emit("scf.yield", else_vals, ())
        outs = builder.emit(
            "scf.if_end",
            (),
            tuple(v.vtype for v in then_vals),
            {"result_names": labels},
        )
        for (name, _, _), out in zip(results, outs):
            merged[name] = out
        for (obj, i, a, b), out in zip(slot_results, outs[len(results) :]):
            obj[i] = out
            self._slots.pop(id(a), None)
            self._slots.pop(id(b), None)
            if not isinstance(obj, _Held):
                self._slots[id(out)] = (obj, i, out)
        for cell, alias in aliases:
            out = merged[alias]
            cell.value = out
            out.chan = cell
        # A channel first acquired in one arm is not held after the if.
        if len(held) != n_held:
            _restore_channels(held, n_held, ())
        self.locals.clear()
        self.locals.update(merged)
        stack[:] = saved_stack
        # Both arms returned: continue at the trailing ``return None``.
        return len(prog.instrs) - 2 if returns else join

    def _trace_arm(
        self,
        view: bc.DecodedCode,
        branch: int,
        join: int,
        start: int,
        saved_stack: list[StackVal],
        containers: list[tuple[Any, Any]],
    ) -> tuple[dict[str, StackVal], list[tuple[list[Any], int, IRValue]]]:
        """Trace one arm until it reaches *join*; return its locals and rebound slots."""
        returns = join >= len(view.instrs)
        # An arm may only move forward from the branch, except that a join at
        # an enclosing loop header (both arms continue) ends the arm.
        header = join if join < branch else -1
        outer = self._mutated
        mutated: dict[int, tuple[Any, Any]] = {}
        self._mutated = mutated
        try:
            outcome = self._loop(view, start, branch, len(view.instrs), header)
        except _NotRollableError:
            raise CompilationError(_REJOIN_MESSAGE) from None
        finally:
            self._mutated = outer
            if outer is not None:
                for key, entry in mutated.items():
                    outer.setdefault(key, entry)
        if header >= 0 and outcome is None:
            outcome = _REGION_DONE
        if returns:
            if outcome is not None:
                raise CompilationError("a measurement-conditioned branch cannot return a value")
        elif outcome is not _REGION_DONE or len(self.stack) != len(saved_stack):
            raise CompilationError(_REJOIN_MESSAGE)
        slots = _slot_changes(containers)
        if slots is None or not _only_containers(mutated, containers):
            raise CompilationError(
                "a measurement-conditioned branch may only change list entries that "
                "hold lines or tones"
            )
        return dict(self.locals), slots

    @staticmethod
    def _jump(target: int, lo: int, hi: int, header: int) -> int:
        """Return the jump target, or -1 when a rolled iteration completes."""
        if target == header:
            return -1
        if not lo < target < hi:
            raise _NotRollableError
        return target

    # ── Structured loops ─────────────────────────────────────────────

    def _roll(
        self, loop: _Loop, prog: bc.DecodedCode, for_idx: int, exit_idx: int, hoisted: bool = False
    ) -> bool:
        """Emit *loop* as one ``scf.for``; return False to unroll it instead.

        The body is traced once. Rolling is kept only if that iteration
        returned to the loop header and left every Python-level value
        (non-IR locals and the lists and dicts they hold, and any container
        a called kernel or helper reaches) unchanged, so all iterations would
        trace identically. Otherwise the emitted ops are rolled back.
        """
        rng = loop.rng
        stack = self.stack
        saved_stack = list(stack)
        saved_locals = dict(self.locals)
        containers = [
            (v, list(v) if isinstance(v, list) else dict(v))
            for v in saved_locals.values()
            if isinstance(v, (list, dict))
        ]
        lines = self._lines
        # Each channel value entering the loop, as (cell, value).
        held = lines.held
        n_held = len(held)
        entering = _channel_values(held) if n_held else ()
        log = lines.log
        mark = len(log)
        checkpoint = self.b.checkpoint()

        snap = self._ir_snapshot()
        count = len(rng)
        var = prog.instrs[for_idx + 1][1]
        self.b.emit("scf.for", (), (), {"lb": 0, "ub": count, "step": 1, "var": var})

        # The iterator stays below the loop value, where ``break`` pops it.
        stack.append(rng[0] if count else 0)
        outer = self._mutated
        mutated: dict[int, tuple[Any, Any]] = {}
        self._mutated = mutated
        acquired = self._acquired[0]
        hoist = False
        try:
            self._loop(prog, for_idx + 1, for_idx, exit_idx, for_idx)
            slot_changes = _slot_changes(containers)
            ok = (
                len(stack) == len(saved_stack)
                and self._python_state_unchanged(saved_locals)
                and slot_changes is not None
                and (not mutated or _only_containers(mutated, containers))
            )
            if ok and self._acquired[0] != acquired:
                ok = False
                # Channels of qudits that exist before the loop are acquired
                # once, before it, and the body is traced again.
                first = checkpoint[1]
                hoist = (
                    not hoisted
                    and len(log) - mark == self._acquired[0] - acquired
                    and all(q.vid < first for _, q in log[mark:])
                )
        except _NotRollableError:
            ok = False
        finally:
            self._mutated = outer
            if outer is not None:
                for key, entry in mutated.items():
                    outer.setdefault(key, entry)

        if not ok:
            for obj, copy in mutated.values():
                if copy is not None:
                    self._restore_container(obj, copy)
            self.b.rollback(checkpoint)
            stack[:] = saved_stack
            self.locals.clear()
            self.locals.update(saved_locals)
            for obj, copy in containers:
                self._restore_container(obj, copy)
            _restore_channels(held, n_held, entering)
            # Acquisitions the undone body made did not happen.
            self._acquired[0] = acquired
            undone = log[mark:]
            del log[mark:]
            if hoist:
                for fname, q in dict.fromkeys(undone):
                    self._emit_pulse_call(fname, [q])
                return self._roll(loop, prog, for_idx, exit_idx, hoisted=True)
            return False

        delta = self._ir_delta(snap)
        assert slot_changes is not None
        # Channels the body advanced are carried, unless a name carries the
        # same value; a channel first acquired in the body was hoisted.
        aliases: list[tuple[_Chan, str]] = []
        originals = {id(obj): copy for obj, copy in containers}
        by_value: dict[int, str] | None = None
        for cell, before in entering:
            new = cell.value
            if new is before:
                continue
            if by_value is None:
                by_value = {id(v): n for n, v in delta.items()}
            alias = by_value.get(id(new))
            if alias is None:
                h = cell.held
                slot_changes.append((h, cell.index, new))
                originals.setdefault(id(h), [None, None])[cell.index] = before
            else:
                aliases.append((cell, alias))
        if delta or slot_changes:
            names = sorted(delta)
            vals = tuple(delta[n] for n in names) + tuple(new for _, _, new in slot_changes)
            # A name first bound in the body has no value entering the loop;
            # the body binds it before reading it.
            inits = tuple(
                saved_locals[n] if n in saved_locals else IRValue(UNDEFINED_VID, delta[n].vtype)
                for n in names
            ) + tuple(originals[id(obj)][i] for obj, i, _ in slot_changes)
            labels = names + [self._slot_label(obj, i) for obj, i, _ in slot_changes]
            self.b.emit("scf.yield", vals, ())
            # The body was traced with the values carried into the loop; its
            # end lists them so that each iteration can rebind them.
            results = self.b.emit(
                "scf.for_end", inits, tuple(v.vtype for v in vals), {"iter_args": labels}
            )
            for n, r in zip(names, results):
                self.locals[n] = r
            for (obj, i, _), r in zip(slot_changes, results[len(names) :]):
                obj[i] = r
                if not isinstance(obj, _Held):
                    self._slots[id(r)] = (obj, i, r)
        else:
            self.b.emit("scf.for_end", (), ())
        for cell, alias in aliases:
            carried = self.locals[alias]
            cell.value = carried
            carried.chan = cell

        if count:
            self.locals[var] = rng[-1]
        elif var in saved_locals:
            self.locals[var] = saved_locals[var]
        else:
            self.locals.pop(var, None)
        return True

    def _unrolls(self, loop: _Loop, prog: bc.DecodedCode, for_idx: int, exit_idx: int) -> bool:
        """Whether *loop* is small enough to unroll: its trip count times the
        calls in its body (at least one) is at most the unroll threshold."""
        threshold = self._unroll
        count = len(loop.rng)
        if threshold == 0 or count > threshold:
            return False
        calls = 0
        for op, _ in prog.instrs[for_idx + 1 : exit_idx]:
            if op is CALL or op is bc.CALL_KW:
                calls += 1
        return count * max(calls, 1) <= threshold

    def _note_mutation(self, obj: Any) -> None:
        """Remember what *obj* held before a rolled body or branch arm changed it."""
        mutated = self._mutated
        if mutated is not None and obj.__class__ in _CONTAINERS and id(obj) not in mutated:
            mutated[id(obj)] = (obj, obj.copy())

    def _note_fresh(self, obj: Any) -> None:
        """Mark a container built by the body or arm being traced as free to change."""
        if self._mutated is not None:
            self._mutated.setdefault(id(obj), (obj, None))

    def _slot_label(self, obj: Any, index: int) -> str:
        if isinstance(obj, _Held):
            return f"{obj.label}[{index}]"
        name = next((n for n, v in self.locals.items() if v is obj), "")
        return f"{name}[{index}]"

    def _python_state_unchanged(self, before: dict[str, StackVal]) -> bool:
        after = self.locals
        for name, old in before.items():
            if isinstance(old, IRValue):
                if name in after and not isinstance(after[name], IRValue):
                    return False
                continue
            if name not in after:
                return False
            new = after[name]
            if new is not old and not (
                type(new) is type(old) and isinstance(old, _SCALARS) and new == old
            ):
                return False
        return True

    # ── Global / closure resolution ──────────────────────────────────

    def _resolve_global(self, name: str) -> Any:
        """Resolve a global name to its Python value."""
        if name in self._fn_globals:
            return self._fn_globals[name]
        if name in self._fn_closures:
            value = self._fn_closures[name]
            if value is _EMPTY_CELL:
                raise CompilationError(f"closure variable {name!r} is referenced before assignment")
            return value
        if name in self._builtins:
            return self._builtins[name]
        return ("__unresolved__", name)

    # ── Call dispatch ────────────────────────────────────────────────

    def _dispatch_call(
        self,
        callable_val: Any,
        call_args: list[StackVal],
        kwargs: dict[str, StackVal] | None,
    ) -> None:
        """Route a call to pulse IR emission or Python evaluation."""

        # Unresolved sentinel (fallback from _resolve_global)
        if isinstance(callable_val, tuple) and len(callable_val) == 2:
            tag, name = callable_val
            if tag == "__unresolved__":
                if kwargs:
                    from .. import ops

                    op_fn = getattr(ops, name, None) if name in _PULSE_OP_NAMES else None
                    if op_fn is None:
                        raise CompilationError(f"unknown pulse op: {name}")
                    call_args = _bind_positional(op_fn, call_args, kwargs)
                if name in OP_TABLE:
                    self._push_pulse_results(name, call_args)
                elif name == "range":
                    self._push_range(call_args)
                else:
                    raise CompilationError(f"unknown pulse op: {name}")
                return

        # Resolved callable: a DSL op only if it is the DSL function itself,
        # so a user function that happens to be named ``gaussian`` stays Python.
        try:
            # Functions hash and compare by identity.
            op_name = _PULSE_FUNCTIONS.get(callable_val)
        except TypeError:  # unhashable callable
            op_name = None
        if op_name is not None:
            if kwargs:
                call_args = _bind_positional(callable_val, call_args, kwargs)
            self._push_pulse_results(op_name, call_args)
            return

        inner = getattr(callable_val, "__pulse_kernel_fn__", None)
        if inner is not None:
            # A kernel with its own policy is scheduled on its own, in a block.
            policy = getattr(callable_val, "schedule", None)
            if policy is not None and getattr(self.b, "schedule_blocks", False):
                self.b.begin_schedule_block(policy, inner.__name__)
                self._inline_helper(inner, call_args, kwargs)
                self.b.end_schedule_block()
            else:
                self._inline_helper(inner, call_args, kwargs)
            return

        fname = getattr(callable_val, "__name__", "")
        if callable_val is range:
            if kwargs:
                raise CompilationError("range() takes no keyword arguments")
            self._push_range(call_args)
            return

        # Inline user helpers which consume IR values or call the pulse DSL.
        # Executing these as ordinary Python would invoke pulse operations
        # outside the compiler context and lose their IR effects.
        if isinstance(callable_val, types.FunctionType) and (
            any(isinstance(arg, IRValue) for arg in call_args)
            or (kwargs and any(isinstance(v, IRValue) for v in kwargs.values()))
            or _uses_pulse_ops(callable_val)
        ):
            self._inline_helper(callable_val, call_args, kwargs)
            return

        # Regular Python callable (int, float, len, abs, math.sin, etc.)
        # int()/float() on a symbolic value create an explicit IR cast.
        if callable(callable_val):
            if not kwargs and len(call_args) == 1 and is_symbolic(call_args[0]):
                if callable_val is int:
                    self.stack.append(call_args[0].cast("i64"))
                    return
                if callable_val is float:
                    self.stack.append(call_args[0].cast("f64"))
                    return
            if any(is_symbolic(a) for a in call_args) or (
                kwargs and any(is_symbolic(v) for v in kwargs.values())
            ):
                names = [
                    n for a in [*call_args, *(kwargs or {}).values()] for n in _parameter_names(a)
                ]
                raise SymbolicValueError(
                    f"Cannot call {fname}() with symbolic parameter "
                    "expressions. Only int(), float(), arithmetic, and pulse "
                    "numeric arguments are supported.",
                    tuple(dict.fromkeys(names)),
                )
            if not _is_pure_host_call(callable_val):
                try:
                    self.b.impure = True
                except AttributeError:
                    pass
            owner = getattr(callable_val, "__self__", None)
            if owner.__class__ in _CONTAINERS and fname not in _READ_METHODS:
                self._note_mutation(owner)
            try:
                result = callable_val(*call_args, **(kwargs or {}))
            except _EVALUATION_ERRORS as e:
                raise CompilationError(f"failed to evaluate {fname}({call_args!r}): {e}") from e
            self.stack.append(result)
            if (callable_val.__class__ is type and callable_val in _CONTAINERS) or (
                owner.__class__ in _CONTAINERS and fname == "copy"
            ):
                self._note_fresh(result)
            if owner.__class__ is list and fname in ("append", "extend", "insert"):
                self._track_slots(owner, len(owner) - 1 if fname == "append" else 0)
            return

        raise CompilationError(f"cannot call: {callable_val!r}")

    def _push_range(self, call_args: list[StackVal]) -> None:
        if not 1 <= len(call_args) <= 3:
            raise CompilationError(f"range() takes 1 to 3 arguments, got {len(call_args)}")
        if not all(isinstance(a, int) for a in call_args):
            raise _range_error(call_args)
        try:
            self.stack.append(range(*call_args))
        except ValueError as exc:
            raise CompilationError(str(exc)) from exc

    def _inline_helper(
        self,
        fn: types.FunctionType,
        call_args: list[StackVal],
        kwargs: dict[str, StackVal] | None,
    ) -> None:
        """Emit a pulse-aware Python helper into the current IR builder."""
        code = fn.__code__
        if code.co_flags & (inspect.CO_VARARGS | inspect.CO_VARKEYWORDS):
            raise CompilationError(
                f"{fn.__name__}: kernels and helpers called from a kernel do not "
                "support *args or **kwargs parameters"
            )
        if self._depth >= MAX_INLINE_DEPTH:
            raise CompilationError(
                f"{fn.__name__}: kernel calls nest more than {MAX_INLINE_DEPTH} deep; "
                "a recursive kernel must stop at a condition known at compile time"
            )
        params = list(code.co_varnames[: code.co_argcount])
        if kwargs or fn.__defaults__:
            call_args = _bind_positional(fn, call_args, kwargs or {})
        if len(call_args) != len(params):
            raise CompilationError(
                f"{fn.__name__}: expected {len(params)} args, got {len(call_args)}"
            )

        helper = PulseIREmitter(
            self.b,
            fn_globals=fn.__globals__,
            fn_closures=_extract_closures(fn) if fn.__closure__ else {},
            caller=self,
        )
        helper.locals.update(zip(params, call_args))
        result = helper.run(code)
        if self._lines.shared:
            self._refresh_lines()

        # Propagate rebindings of directly passed linear values back to the
        # caller, then leave the helper's return value on the caller stack.
        for name, original in zip(params, call_args):
            updated = helper.locals.get(name)
            if not isinstance(original, IRValue) or updated is original:
                continue
            for caller_name, caller_value in self.locals.items():
                if caller_value is original:
                    self.locals[caller_name] = updated

        self.stack.append(result)

    def _push_pulse_results(
        self,
        fname: str,
        call_args: list[StackVal],
    ) -> None:
        results = self._emit_pulse_call(fname, call_args)
        if fname == "readout":
            self.stack.append(results[-1])
        elif fname in {"drive", "wait", "sync"}:
            self.stack.append(None)
        elif len(results) == 1:
            self.stack.append(results[0])
        elif len(results) > 1:
            self.stack.append(results)
        else:
            self.stack.append(None)

    # ── Pulse IR emission ────────────────────────────────────────────

    def _emit_pulse_call(
        self,
        fname: str,
        call_args: list[StackVal],
    ) -> tuple[IRValue, ...]:
        entry = OP_TABLE.get(fname)
        if entry is None:
            raise CompilationError(f"unknown pulse op: {fname}")

        n_val, attr_names, rtypes = entry
        acquires = fname in _ACQUIRES
        line_op = False
        if acquires:
            if fname in _LINE_OPS and call_args and call_args[0].__class__ is IRValue:
                lines = self._lines
                held = lines.held.get((call_args[0].vid, fname))
                if held is not None:
                    lines.share(held)
                    line, tone = held.cells
                    return line.value, tone.value
                lines.log.append((fname, call_args[0]))
                line_op = True
            self._acquired[0] += 1

        if n_val == -1:
            operands = tuple(v for v in call_args if isinstance(v, IRValue))
            attrs: dict[str, Any] = {}
            out_types = tuple(v.vtype for v in operands) if rtypes is None else rtypes
        else:
            operands = tuple(call_args[:n_val])
            attr_vals = call_args[n_val:]
            if len(attr_vals) != len(attr_names):
                raise CompilationError(
                    f"{fname}: expected {n_val + len(attr_names)} args, got {len(call_args)}"
                )
            attrs = dict(zip(attr_names, attr_vals))
            out_types = (operands[0].vtype,) if (rtypes is None and operands) else (rtypes or ())

        op_name = _OP_NAMES[fname]
        results: tuple[IRValue, ...]
        if fname in _MAKES_WAVEFORM:
            if fname in _WAVEFORMS:
                duration = attrs.get("duration")
                if duration.__class__ is not int:
                    duration = None
            else:
                duration = self._waveform_duration(fname, operands, attrs)
            results = self.b.emit(op_name, operands, out_types, attrs)
            # A rolled-back loop body can leave an entry for a reused vid.
            if duration is None:
                self._durations.pop(results[0].vid, None)
            else:
                self._durations[results[0].vid] = duration
        else:
            results = self.b.emit(op_name, operands, out_types, attrs)

        if line_op:
            self._lines.add((call_args[0].vid, fname), results)
        else:
            self._rebind_linear(fname, call_args, results)
        return results

    def _refresh_lines(self) -> None:
        """Bring names holding an old value of a shared channel up to date."""
        lines = self._lines
        local_vars = self.locals
        for name, value in local_vars.items():
            if value.__class__ is IRValue and value.vtype in LINEAR_TYPES:
                current = lines.current(value)
                if current is not None and current is not value:
                    local_vars[name] = current

    def _waveform_duration(
        self, fname: str, operands: tuple[IRValue, ...], attrs: dict[str, Any]
    ) -> int | None:
        """The duration of the waveform arithmetic or ``custom_samples`` call
        *fname* makes, if known; rejects arithmetic on waveforms of different
        durations."""
        if fname in _WAVEFORM_ARITHMETIC:
            for operand in operands:
                if not (operand.__class__ is IRValue and operand.vtype == "waveform"):
                    raise CompilationError(
                        f"{fname}() takes waveforms, got {type(operand).__name__}"
                        + ("; call it as wf_scale(waveform, scalar)" if fname == "wf_scale" else "")
                    )
            known = self._durations
            left = known.get(operands[0].vid)
            if len(operands) == 2:
                right = known.get(operands[1].vid)
                if left is not None and right is not None and left != right:
                    raise CompilationError(
                        f"{fname}() operands have durations {left} and {right}; "
                        "waveform arithmetic needs equal durations"
                    )
                return left if left is not None else right
            return left
        if fname == "custom_samples":
            try:
                return len(attrs["samples"])
            except TypeError:
                return None
        return None

    def _rebind_linear(
        self,
        fname: str,
        call_args: list[StackVal],
        results: tuple[IRValue, ...],
    ) -> None:
        """Rebind the locals and list slots holding consumed linear values.

        Each linear result (drive_line, tone, ...) replaces the first
        unclaimed operand of the same type, wherever the kernel holds it.
        """
        slots = self._slots
        if fname == "drive" and len(results) == 2:
            # drive(line, waveform, tone) -> (line, tone): the common case.
            self._rebind_one(call_args[0], results[0])
            self._rebind_one(call_args[2], results[1])
            return
        arg_info: list[tuple[str | None, IRValue | None, Any, Any]] = []
        for a in call_args:
            if a.__class__ is IRValue and a.vtype in LINEAR_TYPES:
                slot = slots.get(id(a))
                if slot is not None and (slot[2] is not a or slot[0][slot[1]] is not a):
                    slot = None
                arg_info.append((self._find_local_name(a), a, slot, a.chan))
            else:
                arg_info.append((None, None, None, None))

        claimed: set[int] = set()
        for res in results:
            if res.vtype not in LINEAR_TYPES:
                continue
            for i, (name, op_val, slot, chan) in enumerate(arg_info):
                if (
                    i in claimed
                    or op_val is None
                    or (name is None and slot is None and chan is None)
                ):
                    continue
                if op_val.vtype == res.vtype:
                    if chan is not None and chan.value is op_val:
                        chan.value = res
                        res.chan = chan
                        if chan.shared:
                            for other, value in self.locals.items():
                                if value is op_val:
                                    self.locals[other] = res
                    if name is not None:
                        self.locals[name] = res
                    if slot is not None:
                        slot[0][slot[1]] = res
                        slots.pop(id(op_val), None)
                        slots[id(res)] = (slot[0], slot[1], res)
                    claimed.add(i)
                    break

    def _restore_container(self, obj: Any, copy: Any) -> None:
        """Undo a traced arm or loop body's changes to a list or dict.

        Restored lists hold values whose slots the undone code consumed, so
        their slots are tracked again.
        """
        if isinstance(obj, _Held):
            for i, value in enumerate(copy):
                obj[i] = value
        elif isinstance(obj, list):
            obj[:] = copy
            self._track_slots(obj, 0)
        else:
            obj.clear()
            obj.update(copy)

    def _track_slots(self, items: list[Any], start: int) -> None:
        """Remember the list slots from *start* on that hold lines or tones."""
        slots = self._slots
        for i in range(start, len(items)):
            v = items[i]
            if v.__class__ is IRValue and v.vtype in LINEAR_TYPES:
                slots[id(v)] = (items, i, v)

    def _rebind_one(self, old: Any, new: IRValue) -> None:
        if old.__class__ is not IRValue or old.vtype != new.vtype:
            return
        chan = old.chan
        if chan is not None and chan.value is old:
            chan.value = new
            new.chan = chan
            if chan.shared:
                # Every name for the channel's old value moves with it.
                local_vars = self.locals
                for other, value in local_vars.items():
                    if value is old:
                        local_vars[other] = new
        name = self._find_local_name(old)
        if name is not None:
            self.locals[name] = new
        slots = self._slots
        slot = slots.get(id(old))
        if slot is not None and slot[2] is old and slot[0][slot[1]] is old:
            slot[0][slot[1]] = new
            # The consumed value is dead: dropping its entry keeps the table,
            # and the values it holds, the size of the live lists.
            del slots[id(old)]
            slots[id(new)] = (slot[0], slot[1], new)

    def _find_local_name(self, val: IRValue) -> str | None:
        """Find the local variable name bound to a given IRValue."""
        for name, v in self.locals.items():
            if v is val:
                return name
        return None

    # ── Snapshot/delta for structured control flow ───────────────────

    def _ir_snapshot(self) -> dict[str, IRValue]:
        return {k: v for k, v in self.locals.items() if isinstance(v, IRValue)}

    def _ir_delta(self, snap: dict[str, IRValue]) -> dict[str, IRValue]:
        return {
            k: v
            for k, v in self.locals.items()
            if isinstance(v, IRValue) and (k not in snap or snap[k] is not v)
        }


def _slot_changes(
    containers: list[tuple[Any, Any]],
) -> list[tuple[Any, int, IRValue]] | None:
    """List slots whose linear value was replaced, or None for any other change."""
    changes: list[tuple[Any, int, IRValue]] = []
    for obj, copy in containers:
        if len(obj) != len(copy):
            return None
        if not isinstance(obj, list):
            if not _container_unchanged(obj, copy):
                return None
            continue
        for i, (new, old) in enumerate(zip(obj, copy)):
            if new is old:
                continue
            if not (
                isinstance(new, IRValue)
                and isinstance(old, IRValue)
                and new.vtype == old.vtype
                and new.vtype in LINEAR_TYPES
            ):
                return None
            changes.append((obj, i, new))
    return changes


def _channel_values(held: dict[tuple[int, str], _Held]) -> list[tuple[_Chan, IRValue]]:
    """Each channel's line and tone now, as (cell, value)."""
    return [(cell, cell.value) for h in held.values() for cell in h.cells]


def _restore_channels(
    held: dict[tuple[int, str], _Held],
    n_held: int,
    entering: Any,
) -> None:
    """Drop the channels acquired after the first *n_held*, and give the
    others back their *entering* values."""
    if len(held) != n_held:
        for key in list(held)[n_held:]:
            del held[key]
    for cell, value in entering:
        cell.value = value
        value.chan = cell


def _only_containers(
    mutated: dict[int, tuple[Any, Any]], containers: list[tuple[Any, Any]]
) -> bool:
    """Whether every container in *mutated* was built while tracing, or is one
    of the snapshotted *containers*."""
    if not mutated:
        return True
    known = {id(obj) for obj, _ in containers}
    return all(copy is None or key in known for key, (_, copy) in mutated.items())


def _container_unchanged(obj: list[Any] | dict[Any, Any], copy: Any) -> bool:
    if len(obj) != len(copy):
        return False
    if isinstance(obj, list):
        return all(a is b for a, b in zip(obj, copy))
    return all(k in copy and obj[k] is copy[k] for k in obj)


_SIGNATURES: dict[int, tuple[Any, inspect.Signature]] = {}


def _bind_positional(fn: Any, args: list[StackVal], kwargs: dict[str, StackVal]) -> list[StackVal]:
    """Bind a keyword call to *fn*'s parameters as a positional list."""
    hit = _SIGNATURES.get(id(fn))
    if hit is None or hit[0] is not fn:
        try:
            hit = (fn, inspect.signature(fn))
        except (TypeError, ValueError) as e:
            raise CompilationError(f"cannot bind keyword arguments for {fn!r}") from e
        _SIGNATURES[id(fn)] = hit
    try:
        bound = hit[1].bind(*args, **kwargs)
    except TypeError as e:
        raise CompilationError(f"{getattr(fn, '__name__', fn)}: {e}") from e
    bound.apply_defaults()
    if bound.kwargs:
        raise CompilationError(f"{getattr(fn, '__name__', fn)}: keyword-only parameters")
    return list(bound.args)


__all__ = ["PulseIREmitter", "PythonIRBuilder"]
