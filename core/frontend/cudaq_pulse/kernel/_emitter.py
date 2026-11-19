# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PulseIREmitter -- executes decoded kernel bytecode and emits pulse IR.

Kernels are traced by running their canonical instruction stream (see
``_bytecode_normalize``) on an emulated stack. Python values are computed
concretely, so compile-time conditions, short-circuit operators and loop
bounds resolve exactly as Python would; pulse operations (``drive``,
``gaussian``, ...) are emitted as IR operations instead of being called.

``get_drive_line(q)`` and ``get_readout_line(q)`` return the line of the
qudit's one channel of that kind for the whole trace (``_Lines``), threaded
linearly through every op and inlined call that advances it; ``line.tone``
is the channel's tone, and ``line.tones[(lower, upper)]`` a transition's.

A ``for ... in range(n)`` loop whose induction variable is never read is
emitted once as a structured ``scf.for`` when the builder supports regions.
Every other loop, and every loop whose body turns out to carry Python state
from one iteration to the next, is unrolled. Loops over any other concrete
Python iterable (lists, tuples, dicts, ``enumerate``, ``zip``, a qubit
vector, ...) run as Python runs them, one traced iteration per element.
"""

from __future__ import annotations

import builtins as _builtins_mod
import collections
import dis
import functools
import inspect
import linecache
import math
import numbers
import operator
import sys
import types
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, NoReturn

from .._errors import NotFoundError, PulseError
from ..classical import ClassicalFunction
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
from ._schedule_blocks import block_budget
from ._unroll import resolve_unroll_threshold
from .ir_builder import (
    LINEAR_TYPES,
    MAX_CYCLES,
    OP_TABLE,
    UNDEFINED_VID,
    AnswerValue,
    CompilationError,
    IRValue,
    Parameter,
    ParameterExpression,
    PythonIRBuilder,
    Selection,
    SymbolicValueError,
    UndecidableComparison,
    _UndecidableComparisons,
    contains_answer,
    is_symbolic,
)
from .packed_ir_builder import MAX_CUSTOM_SAMPLES

# The module's variables' types, for type checkers: annotations at module
# level would cost each process's import.
if TYPE_CHECKING:
    from .._units import Time

    _PULSE_FUNCTIONS: dict[Any, str]
    _PLAY_FUNCTIONS: dict[Any, str]
    _GLOBAL_NAMES: dict[int, tuple[types.CodeType, frozenset[str]]]
    _PLAIN_VALUES: tuple[type, ...]
    _HASHED: frozenset[type]
    _CHOICE_OPS: dict[str, Callable[[Any, Any], Any]]
    _CHOICE_COMPARE: dict[str, Callable[[Any, Any], Any]]
    _OPERAND_TYPES: dict[str, tuple[tuple[str, ...], ...]]
    _identity_keys: Callable[[dict[str, Any], Any, Any], tuple[str | None, str | None]] | None
    _SIGNATURES: dict[int, tuple[Any, inspect.Signature]]

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
_VARIADIC = inspect.CO_VARARGS | inspect.CO_VARKEYWORDS
# Methods of these containers change them unless listed in _READ_METHODS.
_CONTAINERS = frozenset({list, dict, set})
# The containers a trace can change in place, with their subclasses: the
# arms of a branch and the iterations of a rolled loop must leave them as
# they found them. Concrete types, which isinstance tests without running
# Python (an ABC's isinstance runs its __instancecheck__).
_MUTABLE = (list, dict, set, collections.deque, bytearray)
_READ_METHODS = frozenset({"copy", "count", "index", "get", "keys", "values", "items"})
# List methods after which the list slots holding lines and tones are found again.
_MOVES_SLOTS = frozenset({"append", "extend", "insert", "pop", "remove", "reverse", "sort"})
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
_IS_NOT = operator.is_not
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
        for name, cell in zip(fn.__code__.co_freevars, cells, strict=False):
            try:
                closures[name] = cell.cell_contents
            except ValueError:
                closures[name] = _EMPTY_CELL
    return closures


_PULSE_FUNCTIONS = {}
_PLAY_FUNCTIONS = {}
_TONE_KEYWORD = ("tone",)


class Future:
    """A request to a classical function, sent from inside a kernel."""

    __slots__ = ("function", "index", "vid", "flow", "answers", "timeout_ns")

    def __init__(
        self,
        function: ClassicalFunction[..., Any],
        index: int,
        vid: int,
        flow: tuple[tuple[int, bool], ...],
    ) -> None:
        self.function = function
        self.index = index
        self.vid = vid
        self.flow = flow
        self.answers: Any = None
        self.timeout_ns: float | None = None

    def result(self, timeout: float | Time | None = None) -> Any:
        """Take the answer; only meaningful while a kernel is traced."""
        raise CompilationError(
            "a future's result() is taken inside the kernel that submitted it",
            code="unsupported_construct",
        )

    def __repr__(self) -> str:
        return f"<future of {self.function.name}>"


_SUBMIT = ClassicalFunction.submit
_RESULT = Future.result


def _is_answer_condition(value: Any) -> bool:
    """Whether *value* is a branch condition computed from an answer."""
    return isinstance(value, (AnswerValue, UndecidableComparison)) and contains_answer(value)


# The tracing of classical calls (_classical) is loaded when a kernel first
# submits a request, takes an answer or branches on one: the first of these
# three run binds all three to the module's functions.
def _load_classical() -> None:
    global _trace_submit, _trace_take, _trace_condition
    from ._classical import condition, submit, take

    _trace_submit, _trace_take, _trace_condition = submit, take, condition


def _trace_submit(*args: Any, **kwargs: Any) -> Any:
    _load_classical()
    return _trace_submit(*args, **kwargs)


def _trace_take(*args: Any, **kwargs: Any) -> Any:
    _load_classical()
    return _trace_take(*args, **kwargs)


def _trace_condition(*args: Any, **kwargs: Any) -> Any:
    _load_classical()
    return _trace_condition(*args, **kwargs)


def _load_pulse_functions() -> None:
    """Map each DSL function object, by identity, to its op name."""
    from .. import ops

    loaded = {}
    for name in _PULSE_OP_NAMES:
        op = getattr(ops, name, None)
        if op is not None:
            loaded[op] = name
    # Not an op: _emit_pulse_call computes it while tracing.
    loaded[ops.round_duration] = "round_duration"
    _PLAY_FUNCTIONS.update({op: name for op, name in loaded.items() if name in _PLAYS})
    # One update, so another thread never sees part of the map.
    _PULSE_FUNCTIONS.update(loaded)


def _uses_pulse_ops(
    fn: types.FunctionType,
    seen: set[int] | None = None,
    owner: Any = None,
    strict: bool = False,
) -> bool:
    """Whether *fn* directly or transitively calls the pulse DSL.

    *owner* is the object a method is bound to: its instance, or its class
    for a classmethod. The attributes the method reads are looked up on it,
    and on the objects those hold, so a method that calls the DSL through
    ``self.other()``, ``cls.other()`` or ``self.gates.other()`` counts too.
    Those methods, and *fn* when *strict*, count a DSL name only where they
    load it as a global: ``self.lock.wait()`` is not ``wait``.
    """
    seen = seen or set()
    if id(fn) in seen:
        return False
    seen.add(id(fn))

    closures = _extract_closures(fn) if fn.__closure__ else {}
    names = _global_names(fn.__code__) if strict else fn.__code__.co_names
    # A helper that submits to a classical function emits IR when traced.
    if "submit" in names:
        return True
    for name in (*names, *fn.__code__.co_freevars):
        if name in _PULSE_OP_NAMES:
            return True
        value = closures.get(name, fn.__globals__.get(name))
        if getattr(value, "__pulse_kernel_fn__", None) is not None:
            return True
        if isinstance(value, types.FunctionType) and _uses_pulse_ops(value, seen, strict=strict):
            return True
    if owner is None:
        return False
    owners = [owner]
    for holder in owners:
        for name in fn.__code__.co_names:
            attr = _static_attr(holder, name)
            if attr is _MISSING:
                continue
            if getattr(attr, "__pulse_kernel_fn__", None) is not None:
                return True
            if isinstance(attr, types.FunctionType):
                if _uses_pulse_ops(attr, seen, holder, strict=True):
                    return True
            elif _may_own_helpers(attr) and all(attr is not o for o in owners):
                owners.append(attr)
    return False


# Shared by threads without a lock, as the decode cache is.
_GLOBAL_NAMES = {}


def _global_names(code: types.CodeType) -> frozenset[str]:
    """The names *code* loads as globals, not those it reads as attributes."""
    hit = _GLOBAL_NAMES.get(id(code))
    if hit is not None and hit[0] is code:
        return hit[1]
    names = frozenset(
        i.argval for i in dis.get_instructions(code) if i.opname in ("LOAD_GLOBAL", "LOAD_NAME")
    )
    if len(_GLOBAL_NAMES) >= 4096:
        _GLOBAL_NAMES.clear()
    _GLOBAL_NAMES[id(code)] = (code, names)
    return names


# Values whose attributes are never pulse helpers.
_PLAIN_VALUES = (
    int,
    float,
    complex,
    str,
    bytes,
    bool,
    list,
    tuple,
    dict,
    set,
    types.ModuleType,
)


def _may_own_helpers(value: Any) -> bool:
    """Whether *value* is a class or object whose methods may be pulse
    helpers: anything but builtin values and modules."""
    if value is None or isinstance(value, _PLAIN_VALUES):
        return False
    cls = value if isinstance(value, type) else value.__class__
    return cls.__module__ != "builtins"


def _static_attr(holder: Any, name: str) -> Any:
    """``holder.name`` without running descriptors or ``__getattr__``: the
    function a method, staticmethod, classmethod or property wraps."""
    try:
        attr = inspect.getattr_static(holder, name)
    except AttributeError:
        return _MISSING
    if isinstance(attr, (staticmethod, classmethod)):
        return attr.__func__
    if isinstance(attr, property):
        return attr.fget
    return attr


def _takes_ir(args: Sequence[Any], kwargs: dict[str, Any] | None) -> bool:
    """Whether a helper called with *args* must be traced: it is passed IR
    values, or futures (whose ``result()`` only a traced call can take),
    directly or in a list or tuple."""

    def traced(value: Any) -> bool:
        if isinstance(value, (IRValue, Future)):
            return True
        return isinstance(value, (list, tuple)) and any(isinstance(v, Future) for v in value)

    return any(traced(a) for a in args) or bool(kwargs and any(traced(v) for v in kwargs.values()))


def _truth(value: StackVal) -> bool:
    if isinstance(value, (IRValue, _NotMeasurement)):
        raise CompilationError(
            "a measurement result is only known at runtime",
            code="runtime_value",
            fix="use it directly as an if condition (`if result:` or `if not result:`)",
        )
    return bool(value)


class _NotMeasurement(_UndecidableComparisons):
    """``not value`` for a measurement result, or for the ``i1`` IR value
    a condition on answers is tested as, pending use as a condition."""

    __slots__ = ("value",)

    def __init__(self, value: Any):
        self.value = value

    def __bool__(self) -> bool:
        raise CompilationError(
            f"cannot decide {self!r} at trace time: it is known only when the program runs",
            code="runtime_value",
            fix="use it directly as an if condition",
        )

    def __eq__(self, other: object) -> UndecidableComparison:  # type: ignore[override]
        from ._conditions import negated

        return UndecidableComparison(negated(self), "==", other)

    def __ne__(self, other: object) -> UndecidableComparison:  # type: ignore[override]
        from ._conditions import negated

        return UndecidableComparison(negated(self), "!=", other)

    def __repr__(self) -> str:
        return f"(not {self.value!r})"


def _runtime_bit(value: Any) -> bool:
    """Whether *value* is a bool known only when the program runs."""
    cls = value.__class__
    if cls is UndecidableComparison or cls is _NotMeasurement:
        return True
    if cls is IRValue:
        return bool(value.vtype == "measurement" or value.vtype == "i1")
    return cls is AnswerValue and value.dtype == "i1"


def _identity_unknown(left: Any, right: Any) -> bool:
    """Whether ``left is right`` depends on values known only when the
    program runs: one is such a value, the other one too or a number."""
    a = _runtime_bit(left) or is_symbolic(left)
    b = _runtime_bit(right) or is_symbolic(right)
    return (a or b) and (a or left.__class__ in _NUMBERS) and (b or right.__class__ in _NUMBERS)


_NUMBERS = (bool, int, float)


# The containers `in` looks a value up in by its hash.
_HASHED = frozenset({dict, set, frozenset, type({}.keys())})


# ── Values a measurement-conditioned if chose ───────────────────────────

# Choices compose: `amp * scale` of two choices on different measurements
# has four values. Past this many, the ops given one would be too many ifs.
_MAX_CHOICE_VALUES = 64

_CHOICE_OPS = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": operator.truediv,
    "floordiv": operator.floordiv,
    "mod": operator.mod,
    "pow": operator.pow,
}


class _ChosenOperandError(Exception):
    """A builder was given a choice: the op is emitted by ``_emit_chosen``."""


class _Choice(Selection):
    """A value a measurement-conditioned if chose: ``then`` where ``cond``
    holds and ``other`` where it does not.

    ``then`` and ``other`` are numbers, symbolic numbers, waveforms, or
    choices on other conditions. Arithmetic and calls of pure functions
    apply to each value, so a choice stays a choice between values. An op
    given one reads a select of its values, made once (see
    ``_emit_chosen``). ``scope`` is the regions open where the choice was
    made, outside which ``cond`` is not defined; ``name`` and ``line`` name
    it in errors.
    """

    __slots__ = ("name", "line", "scope", "vtype", "size")
    __hash__ = object.__hash__
    chan = None
    vid = UNDEFINED_VID

    def __init__(
        self,
        cond: IRValue,
        then: Any,
        other: Any,
        name: str,
        line: int | None,
        scope: tuple[int, ...],
    ) -> None:
        self.cond = cond
        self.then = then
        self.other = other
        self.name = name
        self.line = line
        self.scope = scope
        self.vtype = "waveform" if getattr(then, "vtype", None) == "waveform" else None
        self.size = _choice_size(then) + _choice_size(other)
        if self.size > _MAX_CHOICE_VALUES:
            raise CompilationError(
                f"{self.origin} would have more than {_MAX_CHOICE_VALUES} values, one for "
                "each combination of measurement outcomes it depends on; use it inside "
                "the arms of the ifs instead",
                code="limit_exceeded",
            )

    @property
    def origin(self) -> str:
        if not self.name:
            return f"the value the conditional expression, `and` or `or` at line {self.line} chose"
        return (
            f"the value the measurement-conditioned if at line {self.line} chose for {self.name!r}"
        )

    def _binary(self, op: str, other: Any) -> Any:
        if self.vtype == "condition":
            from ._conditions import not_a_number

            not_a_number(self, op)
        return _choice_map(_CHOICE_OPS[op], self, other)

    def _rbinary(self, op: str, other: Any) -> Any:
        if self.vtype == "condition":
            from ._conditions import not_a_number

            not_a_number(self, op)
        return _choice_map(_CHOICE_OPS[op], other, self)

    def __neg__(self) -> Any:
        if self.vtype == "condition":
            from ._conditions import not_a_number

            not_a_number(self, "neg")
        return _choice_map(operator.neg, self)

    def __pos__(self) -> Any:
        return _choice_map(operator.pos, self)

    def __abs__(self) -> Any:
        return _choice_map(abs, self)

    def __eq__(self, other: object) -> Any:
        if self.vtype == "condition":
            from ._conditions import compared

            other = compared(other)
        return _choice_map(operator.eq, self, other)

    def __ne__(self, other: object) -> Any:
        if self.vtype == "condition":
            from ._conditions import compared

            other = compared(other)
        return _choice_map(operator.ne, self, other)

    def __lt__(self, other: Any) -> Any:
        return _choice_map(operator.lt, self, other)

    def __le__(self, other: Any) -> Any:
        return _choice_map(operator.le, self, other)

    def __gt__(self, other: Any) -> Any:
        return _choice_map(operator.gt, self, other)

    def __ge__(self, other: Any) -> Any:
        return _choice_map(operator.ge, self, other)

    def cast(self, dtype: str) -> Any:
        if dtype not in ("i64", "f64"):
            raise CompilationError(f"unsupported symbolic cast to {dtype}", code="runtime_value")
        if self.vtype == "condition":
            from ._conditions import not_a_number

            not_a_number(self, "int() or float()")
        convert = int if dtype == "i64" else float
        return _choice_map(lambda v: v.cast(dtype) if is_symbolic(v) else convert(v), self)

    def _not_concrete(self, operation: str) -> NoReturn:
        raise CompilationError(
            f"{self.origin} is known only when the program runs, so it cannot be "
            f"used for {operation} while the kernel traces; use it in a pulse op or "
            "as an if condition, or do this inside each arm of the if",
            code="runtime_value",
        )

    def __repr__(self) -> str:
        if self.vtype == "condition":
            return f"a condition on measurement results kept at line {self.line}"
        return f"_Choice({self.cond!r}, {self.then!r}, {self.other!r})"


def _choice_size(value: Any) -> int:
    return value.size if value.__class__ is _Choice else 1


def _choice_leaves(choice: _Choice) -> Iterator[Any]:
    """The values *choice* may take."""
    for value in (choice.then, choice.other):
        if value.__class__ is _Choice:
            yield from _choice_leaves(value)
        else:
            yield value


def _selectable(fname: str, operands: tuple[Any, ...], attrs: dict[str, Any]) -> bool:
    """Whether every choice the op *fname* is given can be one select: of
    real or symbolic numbers, or of waveforms, and not one whose values the
    builder checks against each other (the edges of a gaussian_square, the
    bounds of a wait)."""
    if fname == "gaussian_square" and (
        _find_choice(attrs.get("duration")) is not None
        or _find_choice(attrs.get("width")) is not None
    ):
        return False
    if fname == "wait" and attrs.get("bounds") is not None:
        return False
    return all(_selects(v) for v in operands) and all(_selects(v) for v in attrs.values())


def _selects(value: Any) -> bool:
    cls = value.__class__
    if cls is _Choice:
        return all(_select_leaf(leaf) for leaf in _choice_leaves(value))
    if cls is ParameterExpression or cls is UndecidableComparison:
        return _selects(value.lhs) and _selects(value.rhs)
    if cls is list or cls is tuple:
        return _find_choice(value) is None
    return True


def _select_leaf(value: Any) -> bool:
    cls = value.__class__
    if cls is IRValue:
        return bool(value.vtype == "waveform")
    if cls is ParameterExpression:
        return _selects(value)
    return cls is Parameter or cls is AnswerValue or isinstance(value, numbers.Real)


def _find_choice(value: Any) -> _Choice | None:
    """The first choice *value* holds, directly or in an expression, list or tuple."""
    cls = value.__class__
    if cls is _Choice:
        return value  # type: ignore[no-any-return]
    if cls is ParameterExpression or cls is UndecidableComparison:
        found = _find_choice(value.lhs)
        return found if found is not None else _find_choice(value.rhs)
    if cls is list or cls is tuple:
        for item in value:
            found = _find_choice(item)
            if found is not None:
                return found
    return None


def _choices_in(value: Any) -> Iterator[_Choice]:
    """The choices *value* holds, not counting those inside other choices."""
    cls = value.__class__
    if cls is _Choice:
        yield value
    elif cls is ParameterExpression or cls is UndecidableComparison:
        yield from _choices_in(value.lhs)
        yield from _choices_in(value.rhs)
    elif cls is list or cls is tuple:
        for item in value:
            yield from _choices_in(item)


def _in_arm(value: Any, cond: IRValue, taken: bool) -> Any:
    """*value* where an if on *cond* takes its then arm (*taken*) or its else arm."""
    cls = value.__class__
    if cls is _Choice:
        if value.cond is cond:
            return _in_arm(value.then if taken else value.other, cond, taken)
        then = _in_arm(value.then, cond, taken)
        other = _in_arm(value.other, cond, taken)
        if then is value.then and other is value.other:
            return value
        return _choose(value.cond, then, other, value, value.scope)
    if cls is ParameterExpression:
        lhs = _in_arm(value.lhs, cond, taken)
        rhs = _in_arm(value.rhs, cond, taken)
        if lhs is value.lhs and rhs is value.rhs:
            return value
        return ParameterExpression(value.op, lhs, rhs, dtype=value.dtype)
    if cls is UndecidableComparison:
        lhs = _in_arm(value.lhs, cond, taken)
        rhs = _in_arm(value.rhs, cond, taken)
        if lhs is value.lhs and rhs is value.rhs:
            return value
        return _CHOICE_COMPARE[value.op](lhs, rhs)
    if cls is list or cls is tuple:
        items = [_in_arm(item, cond, taken) for item in value]
        if all(a is b for a, b in zip(items, value, strict=False)):
            return value
        return cls(items)
    return value


_CHOICE_COMPARE = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "and": operator.and_,
    "or": operator.or_,
}


def _choose(cond: IRValue, then: Any, other: Any, like: _Choice, scope: tuple[int, ...]) -> Any:
    """The choice on *cond* between *then* and *other*, named like *like*;
    their value if they are the same."""
    if then is other:
        return then
    cls = then.__class__
    if cls is other.__class__ and (cls is int or cls is float or cls is bool) and then == other:
        return then
    chosen = _Choice(cond, then, other, like.name, like.line, scope)
    if like.vtype == "condition" and all(
        v.__class__ is bool or (v.__class__ is _Choice and v.vtype == "condition")
        for v in (then, other)
    ):
        chosen.vtype = "condition"
    return chosen


def _choice_map(fn: Callable[..., Any], *args: Any) -> Any:
    """``fn(*args)`` for the values the choices among *args* take: a choice
    on the first one's condition, or ``fn(*args)`` if there are none."""
    choices = [c for c in map(_find_choice, args) if c is not None]
    if not choices:
        return fn(*args)
    scope = max((c.scope for c in choices), key=len)
    return _map_choices(fn, args, choices[0], scope)


def _map_choices(
    fn: Callable[..., Any], args: tuple[Any, ...], like: _Choice, scope: tuple[int, ...]
) -> Any:
    choice = next((c for c in map(_find_choice, args) if c is not None), None)
    if choice is None:
        return fn(*args)
    cond = choice.cond
    then = _map_choices(fn, tuple(_in_arm(a, cond, True) for a in args), like, scope)
    other = _map_choices(fn, tuple(_in_arm(a, cond, False) for a in args), like, scope)
    return _choose(cond, then, other, like, scope)


def _choice_condition(choice: _Choice) -> Any:
    """What an if on *choice* branches on: a bool when every value it may
    take has the same truth, else the condition that chose it."""

    def truth(value: Any) -> bool:
        if isinstance(value, IRValue):
            raise CompilationError(
                f"{choice.origin} is a pulse {value.vtype}, not a condition",
                code="runtime_value",
            )
        if contains_answer(value) or (
            value.__class__ is _NotMeasurement and value.value.vtype == "i1"
        ):
            raise CompilationError(
                f"an if on {choice.origin} would test both a measurement result and "
                f"{value!r}, an answer known only when the program runs; branch on them "
                "in nested ifs, or with `and` and `or` in the if condition",
                code="runtime_value",
            )
        return bool(value)

    decided = _choice_map(truth, choice)
    if decided.__class__ is bool:
        return decided
    if decided.then.__class__ is bool and decided.other.__class__ is bool:
        return decided.cond if decided.then else _NotMeasurement(decided.cond)
    if choice.vtype == "condition":
        # A condition on measurement results kept in a value.
        from ._conditions import condition_bit

        return condition_bit(choice)
    raise CompilationError(
        f"an if on {choice.origin} would depend on more than one measurement; "
        "branch on the measurements instead",
        code="runtime_value",
    )


def _watch_choices(builder: Any) -> None:
    """Make *builder* raise ``_ChosenOperandError`` for an op given a choice,
    before the op has any effect. Until a trace makes its first choice the
    builder is left as it is, so other programs pay nothing for this."""
    if getattr(builder, "_unwatched_emit", None) is not None:
        return
    emit = builder.emit

    def watched(
        kind: str,
        operands: tuple[Any, ...] = (),
        result_types: tuple[str, ...] = (),
        attrs: dict[str, Any] | None = None,
    ) -> tuple[IRValue, ...]:
        for v in operands:
            if v.__class__ is _Choice:
                raise _ChosenOperandError
        if attrs:
            for v in attrs.values():
                if v.__class__ is not float and v.__class__ is not int:
                    if _find_choice(v) is not None:
                        raise _ChosenOperandError
        return emit(kind, operands, result_types, attrs)  # type: ignore[no-any-return]

    builder._unwatched_emit = emit
    builder.emit = watched


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

    def __iter__(self) -> Any:
        # What a generator expression returns is consumed by Python calls,
        # such as sum(), as well as by loops that the tracer runs.
        return self.it


_MISSING = object()
# What a comprehension saves for an iteration variable that was unbound.
_UNBOUND = object()
# The tag of ("__unresolved__", name), a name nothing in scope defines.
_UNRESOLVED = "__unresolved__"

_LINE_OPS = frozenset({"get_drive_line", "get_readout_line"})
# Ops that play a waveform on a line, at its tone or at the keyword tone.
_PLAYS = frozenset({"drive", "readout"})
_first = operator.itemgetter(0)

_LINE_TYPES = ("drive_line", "readout_line")
_QUDIT = ("qref",)
_TONE = ("tone",)
_WAVEFORM = ("waveform",)
# The IR types each leading operand of a pulse op may have.
_OPERAND_TYPES = {
    "get_drive_line": (_QUDIT,),
    "get_readout_line": (_QUDIT,),
    "drive": (("drive_line",), _WAVEFORM, _TONE),
    "readout": (("readout_line",), _WAVEFORM, _TONE),
    "wait": (_LINE_TYPES,),
    "shift_phase": (_TONE,),
    "set_phase": (_TONE,),
    "shift_frequency": (_TONE,),
    "set_frequency": (_TONE,),
    "wf_add": (_WAVEFORM, _WAVEFORM),
    "wf_sub": (_WAVEFORM, _WAVEFORM),
    "wf_mul": (_WAVEFORM, _WAVEFORM),
    "wf_scale": (_WAVEFORM,),
    "wf_neg": (_WAVEFORM,),
}
# The ops with an optional last argument, and its attribute.
_OPTIONAL_LAST = {"wait": "bounds", "custom_samples": "sample_rate_ghz"}

_EMIT_TABLE = {
    name: (value_count, attrs, rtypes, _OPERAND_TYPES.get(name, ()))
    for name, (value_count, attrs, rtypes) in OP_TABLE.items()
}
_TYPE_WORDS = {
    "qref": "qudit",
    "drive_line": "drive line",
    "readout_line": "readout line",
    "tone": "tone",
    "waveform": "waveform",
    "measurement": "measurement result",
    "iq_point": "integrated IQ point",
}

# What readout(..., acquire=) takes; the first is the default.
_ACQUIRE_MODES = ("level", "integrated")


def _acquire_mode(value: Any) -> str | None:
    """The mode ``readout(..., acquire=value)`` emits, None for a level (the
    default, which the packed IR leaves implicit)."""
    if isinstance(value, str) and value in _ACQUIRE_MODES:
        return None if value == "level" else str(value)
    later = " (trace and counts are not supported yet)" if value in ("trace", "counts") else ""
    raise CompilationError(
        f'readout(): acquire is "level" or "integrated", a string known while the kernel '
        f"compiles; got {value!r}{later}",
        code="wrong_argument",
    )


def _note_source(e: BaseException, prog: bc.DecodedCode, index: int) -> None:
    """Record where tracing failed: ``pulse_source`` is the innermost
    instruction, ``pulse_callers`` the calls that led to it, innermost
    first."""
    try:
        source = getattr(e, "pulse_source", None)
        if source is None:
            e.pulse_source = (prog, index)  # type: ignore[attr-defined]
        elif source[0] is not prog:
            callers = getattr(e, "pulse_callers", None)
            if callers is None:
                callers = e.pulse_callers = []  # type: ignore[attr-defined]
            if not callers or callers[-1][0] is not prog:
                callers.append((prog, index))
    except AttributeError:
        pass


def tracing_frames() -> list[tuple[bc.DecodedCode, int]]:
    """The instructions being traced, innermost first, as ``pulse_source``
    and ``pulse_callers`` would name them if tracing raised here."""
    loop = PulseIREmitter._loop.__code__
    frames: list[tuple[bc.DecodedCode, int]] = []
    frame: types.FrameType | None = sys._getframe(1)
    while frame is not None:
        if frame.f_code is loop:
            prog = frame.f_locals["prog"]
            if not frames or frames[-1][0] is not prog:
                frames.append((prog, frame.f_locals["pc"] - 1))
        frame = frame.f_back
    return frames


def _unrolled_loops() -> list[tuple[Any, int | None, int, list[tuple[bc.DecodedCode, int]]]]:
    """The loops being unrolled now, outermost first, as (loop, trip count
    or None, iterations started, the frames ``pulse_source`` and
    ``pulse_callers`` name for its ``for`` line)."""
    code = PulseIREmitter._loop.__code__
    frames: list[tuple[bc.DecodedCode, int, Any]] = []
    frame: types.FrameType | None = sys._getframe(1)
    while frame is not None:
        if frame.f_code is code:
            prog = frame.f_locals["prog"]
            if not frames or frames[-1][0] is not prog:
                frames.append((prog, frame.f_locals["pc"] - 1, frame.f_locals["self"].stack))
        frame = frame.f_back
    found: list[tuple[Any, int | None, int, list[tuple[bc.DecodedCode, int]]]] = []
    for depth in range(len(frames) - 1, -1, -1):
        prog, index, stack = frames[depth]
        headers = [
            i
            for i, (op, arg) in enumerate(prog.instrs[:index])
            if op is FOR_ITER and i < index < arg[0]
        ]
        loops = [v for v in stack if type(v) is _Loop or type(v) is _Unrolled]
        if len(loops) != len(headers):
            continue
        callers = [(p, i) for p, i, _ in frames[depth + 1 :]]
        for loop, header in zip(loops, headers, strict=False):
            if isinstance(loop, _Unrolled):
                left = operator.length_hint(loop.it, -1)
                trips = None if left < 0 else loop.count + left
                found.append((loop, trips, loop.count, [(prog, header), *callers]))
            elif loop.rolled is False:
                rng = loop.rng
                past = rng.step - (1 if rng.step > 0 else -1)
                count = max(0, (rng.stop - rng.start + past) // rng.step)
                started = count - operator.length_hint(loop.it)
                found.append((loop, count, started, [(prog, header), *callers]))
    return found


def _widest(loops: list[Any]) -> Any:
    """Of ``_unrolled_loops()``, the loop with the most iterations."""
    return max(loops, key=lambda entry: entry[2] if entry[1] is None else entry[1])


def note_unrolled_progress(op_count: int, checks: dict[int, tuple[int, int]]) -> None:
    """Remember how far the widest unrolled loop is, at *op_count* ops; the
    instruction-limit error estimates the ops per iteration from it."""
    loops = _unrolled_loops()
    if loops:
        loop, _, started, _ = _widest(loops)
        checks[id(loop)] = (started, op_count)


def instruction_limit_error(
    op_count: int, limit: int, checks: dict[int, tuple[int, int]]
) -> CompilationError:
    """The error for a program grown to *op_count* ops, past the
    controller's *limit*, noted at the ``for`` line of the widest loop
    being unrolled."""
    holds = f"more than the {limit:,} the controller holds (Controller.max_instructions)"
    loops = _unrolled_loops()
    if not loops:
        return CompilationError(
            f"the program holds {op_count:,} operations, {holds}", code="limit_exceeded"
        )
    loop, trips, started, frames = _widest(loops)
    seen = checks.get(id(loop))
    if seen is not None and started > seen[0]:
        per = (op_count - seen[1]) / (started - seen[0])
    else:
        per = op_count / max(started, 1)
    each = f"{per:,.0f}" if per >= 10 or abs(per - round(per)) < 0.05 else f"{per:.1f}"
    if type(loop) is _Loop:
        what = f"for loop over {loop.rng!r}"
        why = (
            "a range() loop stays rolled, held once, only when its body does not "
            "read the loop variable or change Python state"
        )
    else:
        what = "for loop over a Python iterable"
        why = "loops over Python iterables are always unrolled"
    if trips is None:
        size = f"has unrolled {started:,} iterations of about {each} operations"
    else:
        size = (
            f"unrolls to about {round(trips * per):,} operations "
            f"({trips:,} iterations of about {each})"
        )
    error = CompilationError(f"{what} {size}, {holds}; {why}", code="limit_exceeded")
    error.pulse_source = frames[0]  # type: ignore[attr-defined]
    error.pulse_callers = frames[1:]  # type: ignore[attr-defined]
    add_source_note(error)
    return error


def add_source_note(e: BaseException, heading: str | None = None) -> None:
    """Add a note to *e* with the kernel source lines that raised it, as a
    traceback shows them: the kernel's own line first."""
    source = getattr(e, "pulse_source", None)
    if source is None or getattr(e, "pulse_noted", False):
        return
    frames = [source, *getattr(e, "pulse_callers", ())]
    lines = [heading or "while tracing the kernel:"]
    for prog, index in reversed(frames):
        filename, lineno, *_ = prog.location(index)
        lines.append(f'  File "{filename}", line {lineno}, in {prog.code.co_name}')
        text = linecache.getline(filename, lineno).strip() if lineno else ""
        if text:
            lines.append(f"    {text}")
    try:
        e.add_note("\n".join(lines))
        e.pulse_noted = True  # type: ignore[attr-defined]
        # The line that raised, innermost, for PulseError.location.
        e._traced_at = (filename, lineno, text, prog.code.co_name)  # type: ignore[attr-defined]
    except AttributeError:
        pass


def _attribute_error(obj: Any, attr: str) -> CompilationError:
    """The error for ``obj.attr`` when *obj* has no attribute *attr*."""
    from ._emit_errors import attribute_error

    return attribute_error(obj, attr)


def _describe(value: Any) -> str:
    """How an operand error names what the kernel passed."""
    if value.__class__ is IRValue:
        word = _TYPE_WORDS.get(value.vtype, value.vtype)
        return f"an {word}" if word[0] in "aeiou" else f"a {word}"
    if isinstance(value, (list, tuple)):
        return f"a {type(value).__name__} of {len(value)}"
    if value.__class__ is _Choice:
        return str(value.origin)
    if value.__class__ is _NotMeasurement:
        return f"`not` of {_describe(value.value)}"
    return f"{type(value).__name__} {value!r}"


def _sync_lines(fname: str, call_args: list[Any]) -> list[IRValue]:
    """The current values of the lines passed to *fname*, with lists and
    tuples of lines flattened."""
    lines: list[IRValue] = []
    for v in call_args:
        if v.__class__ is not IRValue or v.vtype not in _LINE_TYPES:
            break
        chan = v.chan
        lines.append(v if chan is None else chan.value)
    else:
        if len(lines) >= 2:
            return lines
    lines = []

    def add(value: Any, where: str) -> None:
        if value.__class__ is IRValue and value.vtype in _LINE_TYPES:
            lines.append(_current(value))
        elif value.__class__ in (list, tuple):
            for i, item in enumerate(value):
                add(item, f"{where}[{i}]")
        else:
            message = f"{fname}(): {where} must be a line, got {_describe(value)}"
            if value.__class__ is IRValue and value.vtype == "tone":
                message += f"; pass the lines, not their tones, as in {fname}(d, r)"
            raise CompilationError(message, code="wrong_argument")

    for i, value in enumerate(call_args):
        add(value, f"argument {i + 1}")
    if not lines:
        raise CompilationError(
            f"{fname}() got no lines; it aligns two or more lines, as in {fname}(d, r)",
            code="wrong_argument",
        )
    if len(lines) == 1:
        raise CompilationError(
            f"{fname}() got one line, which has no other line to align with; pass two "
            f"or more lines, as in {fname}(d, r), or remove the call",
            code="wrong_argument",
        )
    if len({id(v) for v in lines}) != len(lines):
        raise CompilationError(f"{fname}(): a line is passed more than once", code="wrong_argument")
    return lines


def _sample_envelope(
    duration: Any, envelope: Any, rate: tuple[int, int] | None = None
) -> list[float]:
    """The samples of ``custom(duration, envelope)``: *envelope* called at the
    middle of each clock cycle, ``t = k + 0.5`` for ``k`` in
    ``range(duration)``, as ``wf_mul`` products are sampled; with *rate*
    (n samples every d cycles, the lines' rate), at the middle of each
    sample instead, ``t = (k + 0.5) * d / n``."""
    scale = 1.0
    if isinstance(envelope, str):
        raise CompilationError(
            f"custom(): got the name {envelope!r}; pass a callable f(t) that returns "
            "the envelope at t clock cycles, or the samples to custom_samples()",
            code="wrong_argument",
        )
    if not callable(envelope):
        raise CompilationError(
            f"custom(): argument 2 must be a callable f(t), got {_describe(envelope)}",
            code="wrong_argument",
        )
    if is_symbolic(duration):
        raise CompilationError(
            "custom(): the duration must be known while tracing, since the envelope "
            "is sampled then; pass it as an int argument or a constant",
            code="invalid_duration",
        )
    try:
        count = operator.index(duration)
    except TypeError:
        raise CompilationError(
            f"custom(): the duration must be an int number of clock cycles, got {duration!r}",
            code="invalid_duration",
        ) from None
    if rate is not None:
        from ._durations import at_rate

        count, scale, envelope = at_rate(count, rate, envelope)
    if count > MAX_CUSTOM_SAMPLES:
        raise CompilationError(
            f"custom(): a duration of {count} clock cycles is over the limit of "
            f"{MAX_CUSTOM_SAMPLES}, one sample per cycle; split the envelope into "
            "consecutive drives of at most that length",
            code="limit_exceeded",
        )
    samples = []
    for k in range(count):
        t = k + 0.5
        value = envelope(t)
        if is_symbolic(value):
            raise CompilationError(
                "custom(): the envelope returned a value that depends on a kernel "
                "parameter; scale the waveform by it instead, as "
                "wf_scale(custom(duration, f), amplitude)",
                code="invalid_waveform",
            )
        try:
            number: complex = value.__float__()
        except (AttributeError, TypeError, ValueError):
            number = _envelope_number(value, t * scale)
        if number.imag != 0.0:
            raise CompilationError(
                f"custom(): the envelope returned {number} at t={t * scale}; complex envelopes "
                "are not supported yet, so return real values",
                code="invalid_waveform",
            )
        if not math.isfinite(number.real):
            raise CompilationError(
                f"custom(): the envelope returned {number.real} at t={t * scale}",
                code="invalid_waveform",
            )
        samples.append(number.real)
    return samples


def _envelope_number(value: Any, t: float) -> complex:
    """*value*, what a ``custom()`` envelope returned at *t* without
    ``__float__``, as complex() reads it; a string, which complex() would
    parse, is an error."""
    if not isinstance(value, (str, bytes, bytearray)):
        try:
            return complex(value)
        except (TypeError, ValueError):
            pass
    raise CompilationError(
        f"custom(): the envelope returned {_describe(value)} at t={t}, not a number",
        code="invalid_waveform",
    )


def _unpack_error(value: Any, count: int) -> str:
    """The error for unpacking *value*, which is not a tuple or list, into
    *count* names."""
    if value is None:
        return (
            f"cannot unpack None into {count} names: a call that returns nothing, such as "
            "drive() or wait(), gives None; call it on its own line, not as "
            "`a, b = drive(...)` (its arguments stay usable after it)"
        )
    if value.__class__ is IRValue and value.vtype in _LINE_TYPES:
        return (
            f"cannot unpack {_describe(value)} into {count} names: a line is one value, "
            f"not a pair; write `line = get_{value.vtype}(q)` and use line.tone for its tone"
        )
    return (
        f"cannot unpack {_describe(value)} into {count} names; "
        "only a tuple or list known while tracing can be unpacked"
    )


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
    """The current (line, tone) of one qudit's drive or readout channel, or
    the current (tone,) of one of its transitions.

    Indexing reads and writes the line (0) and the tone (1); a value written
    records the channel in ``IRValue.chan``.
    """

    __slots__ = ("cells", "key")

    def __init__(self, values: tuple[IRValue, ...], key: tuple[Any, ...]) -> None:
        try:
            line, tone = values
        except ValueError:
            self.cells: tuple[_Chan, ...] = (_Chan(values[0], self, 0),)
        else:
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
        label = f"{self.key[1].removeprefix('get_')}{self.key[0]}"
        if len(self.key) == 3:
            lower, upper = self.key[2]
            label += f"_tone_{lower}_{upper}"
        return label


class _Lines:
    """The drive and readout channels acquired so far in one trace.

    ``get_drive_line(q)`` returns the line of the one drive channel of ``q``
    for the whole program: the first call acquires it, later calls, in the
    kernel or in a kernel it calls, return its current line, so pulses on
    it play one after another. Each value a channel takes records the
    channel in ``IRValue.chan``, so an op that consumes it moves the channel
    to the op's result, and a name left holding an old value of a shared
    channel can be brought up to date. The tone of a transition other than
    the default is a channel of its own, keyed by the line's key and the
    transition.
    """

    # ``watched``, set only while a measurement-conditioned arm is traced:
    # for each arm around the trace, its record of changed containers and
    # the containers it copies when they first change. ``plain``, set by the
    # first such if: id -> (list, length) for lists whose first `length`
    # entries hold no line or tone. Both are read with getattr.
    __slots__ = ("held", "shared", "log", "watched", "plain")

    if TYPE_CHECKING:
        watched: tuple[tuple[dict[int, tuple[Any, Any]], tuple[Any, ...]], ...]
        plain: dict[int, tuple[list[Any], int]]

    def __init__(self) -> None:
        self.held: dict[tuple[Any, ...], _Held] = {}
        # Whether any channel was handed out more than once.
        self.shared = False
        # Line acquisitions, in order: those a rolled loop body makes are
        # hoisted before the loop.
        self.log: list[tuple[str, IRValue]] = []

    def add(self, key: tuple[Any, ...], values: tuple[IRValue, ...]) -> None:
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


class _Tones(Mapping[tuple[int, int], IRValue]):
    """``line.tones`` while tracing: the current tone of each transition of
    the line's qudit, by ``(lower, upper)`` level. The default transition's
    is the line's own tone; the first read of another makes a tone at that
    transition's frequency, with phase 0, which later reads return as it
    then is."""

    __slots__ = ("_emitter", "_held", "_default", "_frequencies", "_slot")

    def __init__(self, emitter: PulseIREmitter, held: _Held) -> None:
        builder = emitter.b
        line_tones = getattr(builder, "line_tones", None)
        qudit = getattr(builder, "_qref_to_qudit", {}).get(held.key[0])
        if line_tones is None or qudit is None:
            raise CompilationError(
                "line.tones needs the transitions of the system's model; this trace "
                "has no quantum system, so use line.tone"
            )
        self._emitter = emitter
        self._held = held
        self._slot = qudit < 0
        self._default, self._frequencies = line_tones(qudit, held.key[1] == "get_readout_line")

    def __getitem__(self, transition: tuple[int, int]) -> IRValue:
        held = self._held
        if transition == self._default:
            return held.cells[1].value
        frequency = self._frequencies.get(transition) if transition.__class__ is tuple else None
        if frequency is None:
            raise NotFoundError(self._missing(transition))
        lines = self._emitter._lines
        key = (*held.key, transition)
        tone = lines.held.get(key)
        if tone is not None:
            lines.share(tone)
            return tone.cells[0].value
        value: IRValue
        (value,) = self._emitter.b.emit(
            "pulse.tone", (), ("tone",), {"frequency_hz": frequency, "phase_rad": 0.0}
        )
        # Made once for the whole program, like a line: a rolled loop body
        # that makes one is unrolled.
        self._emitter._acquired[0] += 1
        lines.add(key, (value,))
        return value

    def _missing(self, transition: Any) -> str:
        if self._slot:
            return (
                f"the qudit is a runtime parameter, so its transitions' frequencies depend on "
                f"the carrier each call binds; line.tones has only the default "
                f"{self._default!r}, line.tone. Pass the frequency of {transition!r} in as a "
                "float parameter, e.g. f12_hz: float, called with "
                "qs.transition_freq[q.logical, 1, 2], and play at it with "
                "set_frequency(line.tone, f12_hz)"
            )
        if self._held.key[1] == "get_readout_line":
            return (
                f"a readout line has only its own tone, that of the default transition "
                f"{self._default!r}; got {transition!r}"
            )
        known = ", ".join(map(repr, self._frequencies))
        return (
            f"the qudit has no transition {transition!r}; line.tones has the "
            f"(lower, upper) levels {known}, the default {self._default!r} being line.tone"
        )

    def __contains__(self, transition: object) -> bool:
        return transition in self._frequencies

    def __iter__(self) -> Iterator[tuple[int, int]]:
        return iter(self._frequencies)

    def __len__(self) -> int:
        return len(self._frequencies)

    def __repr__(self) -> str:
        return f"<tones of transitions {', '.join(map(repr, self._frequencies))}>"


# Host calls evaluated at trace time whose result depends only on their
# arguments; any other C-level callable (``random.random``, ``time.time``)
# makes the trace impure, so its cached compilation is never reused.
_PURE_MODULES = frozenset({"math", "cmath", "builtins", "operator", "functools", "itertools"})

# Callables that are not objects with a Python __call__.
_NATIVE_CALLABLES = frozenset(
    {
        types.BuiltinFunctionType,
        types.MethodWrapperType,
        types.WrapperDescriptorType,
        types.MethodDescriptorType,
        types.ClassMethodDescriptorType,
        types.FunctionType,
        types.MethodType,
        functools.partial,
        type,
    }
)


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


def _iterate(value: Any) -> Any:
    """``iter(value)`` for a ``for`` loop, with a kernel-specific error."""
    if isinstance(value, IRValue):
        message = (
            f"cannot iterate over a pulse {value.vtype} value; for loops "
            "in pulse kernels run over Python values known at compile time"
        )
        if value.vtype in _LINE_TYPES:
            message += "; a line is one value, and its tone is line.tone"
        raise CompilationError(message, code="unsupported_construct")
    if is_symbolic(value):
        if value.__class__ is _Choice:
            raise CompilationError(f"cannot iterate over {value.origin}", code="runtime_value")
        names = _parameter_names(value)
        raise SymbolicValueError(
            f"cannot iterate over kernel parameter {', '.join(map(repr, names))}",
            tuple(names),
            "a for loop over it",
        )
    try:
        return iter(value)
    except TypeError:
        raise CompilationError(
            f"cannot iterate over {type(value).__name__} in a pulse kernel",
            code="wrong_argument",
        ) from None


def _current(value: IRValue) -> IRValue:
    """The current value of *value*'s channel, or *value* if it has none."""
    chan = value.chan
    return value if chan is None else chan.value


_identity_keys = None


def _load_identity_keys() -> Callable[[dict[str, Any], Any, Any], tuple[str | None, str | None]]:
    """The native identity scan, imported on first use."""
    global _identity_keys
    from .._native._cudaq_pulse_native import identity_keys

    _identity_keys = identity_keys
    return identity_keys


def _slot_holds(slot: tuple[list[Any], int, IRValue], value: IRValue) -> bool:
    """Whether the list slot recorded for *value* still holds it."""
    obj, index, held = slot
    if held is not value:
        return False
    try:
        return obj[index] is value
    except IndexError:  # the list shrank since the value was read
        return False


def _star_args(value: Any) -> list[Any]:
    """The positional arguments of ``f(*value)``, unpacked at trace time."""
    if value.__class__ is tuple or value.__class__ is list:
        return list(value)
    if isinstance(value, IRValue):
        raise CompilationError(
            f"cannot unpack a pulse {value.vtype} value with *; pass it as an argument",
            code="wrong_argument",
        )
    return list(_iterate(value))


def _merge_kwargs(target: dict[Any, Any], other: Any, strict: bool) -> None:
    """``**other`` into the keyword arguments *target* of a call (or a dict display)."""
    try:
        keys = other.keys()
    except AttributeError:
        raise CompilationError(
            f"argument after ** must be a mapping, not {type(other).__name__}",
            code="wrong_argument",
        ) from None
    for key in keys:
        if strict and key in target:
            raise CompilationError(
                f"got multiple values for keyword argument {key!r}", code="wrong_argument"
            )
        target[key] = other[key]


class _GenExpr:
    """A generator expression, run all at once when it is called."""

    __slots__ = ("code",)

    def __init__(self, code: types.CodeType):
        self.code = code


class _Loop:
    """Iterator state of a ``for ... in range()`` loop on the stack."""

    __slots__ = ("rng", "it", "rolled")

    def __init__(self, rng: range):
        self.rng = rng
        self.it = iter(rng)
        self.rolled: bool | None = None  # None until the header decides


class _NotRollableError(Exception):
    """The single traced iteration of a rolled loop left the loop body."""


class _UndefinedLocalError(CompilationError):
    """A name was read before the kernel bound it."""

    def __init__(self, name: str, message: str = "") -> None:
        super().__init__(message or f"undefined local: {name}", code="unassigned_name")
        self.name = name
        self.message = message


def _choice_kind(value: Any) -> str | None:
    """What a measurement-conditioned if can choose *value* as: "number" or
    "waveform"; None if it cannot."""
    cls = value.__class__
    if cls is _Choice:
        return "number" if value.vtype is None else "waveform"
    if cls is IRValue:
        return "waveform" if value.vtype == "waveform" else None
    if cls is AnswerValue or cls is Parameter or cls is ParameterExpression:
        return "number"
    return "number" if isinstance(value, numbers.Real) else None


def _made_in_if(value: Any, first: int, scope: tuple[int, ...]) -> bool:
    """Whether the number or waveform *value* holds a value made inside a
    measurement-conditioned if, which does not exist after it. *first* is
    the first id made inside the if, *scope* the regions open around it."""
    cls = value.__class__
    if cls is _Choice:
        return (
            value.cond.vid >= first
            or _made_in_if(value.then, first, scope)
            or _made_in_if(value.other, first, scope)
        )
    if cls is IRValue:
        return bool(value.vid >= first)
    if cls is AnswerValue:
        return bool(scope[: len(value.scope)] != value.scope)
    if cls is ParameterExpression:
        return _made_in_if(value.lhs, first, scope) or _made_in_if(value.rhs, first, scope)
    return False


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
        "_one_arm",
        "_dict_items",
        "_acquire",
    )

    # The slots' types, for type checkers: annotations in the class body
    # would cost every process's first trace.
    if TYPE_CHECKING:
        _acquire: str
        _can_roll: bool
        _dict_items: dict[tuple[int, Any], tuple[dict[Any, Any], Any, Any]] | None
        _slots: dict[int, tuple[list[Any], int, IRValue]]
        _durations: dict[int, int]
        _mutated: dict[int, tuple[Any, Any]] | None
        _depth: int
        _acquired: list[int]
        _lines: _Lines
        _unroll: int
        # Names a measurement-conditioned if bound in one arm only, with the
        # branch instruction; set by the first such if.
        _one_arm: dict[str, tuple[bc.DecodedCode, int]]

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
            self._dict_items = caller._dict_items
            self._can_roll = caller._can_roll
            self._slots = caller._slots
            self._durations = caller._durations
            self._mutated = caller._mutated
            self._depth = caller._depth + 1
            self._acquired = caller._acquired
            self._lines = caller._lines
            self._unroll = caller._unroll
            return
        # The builder's record of the dict items the trace read, for the
        # guard (``_guard.py``); None for a builder that keeps none.
        self._dict_items = getattr(builder, "dict_items", None)
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

    def trace(self, prog: bc.DecodedCode) -> StackVal:
        """Trace a whole kernel, as ``run`` traces a helper. A name read
        before it is bound raises a plain ``CompilationError``: the private
        error is how a loop being rolled learns that, and stops here."""
        try:
            return self._loop(prog, 0, -1, len(prog.instrs), -1)
        except _UndefinedLocalError as e:
            if e.message:
                error = CompilationError(e.message, code="unassigned_name")
            else:
                error = CompilationError(
                    f"{e.name!r} is used before it is assigned",
                    code="unassigned_name",
                    fix="assign it on every path that reaches this line (an if on a value "
                    "known while tracing takes only one arm)",
                )
            error.__dict__.update(e.__dict__)
            raise error from None

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
        _call_kw: int = bc.CALL_KW,
        _list_append: int = bc.LIST_APPEND,
        _list_extend: int = bc.LIST_EXTEND,
        _unary: int = bc.UNARY,
        _not: int = bc.NOT,
        _is: int = bc.IS,
        _contains: int = bc.CONTAINS,
        _hashed: frozenset[type] = _HASHED,
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
        dict_items = self._dict_items
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
                        raise self._undefined(arg) from None

                elif op is _load_global:
                    value = fn_globals.get(arg, _MISSING)
                    push(self._resolve_global(arg) if value is _MISSING else value)

                elif op is _pop_top:
                    pop()

                elif op is _call:
                    if arg:
                        call_args = stack[-arg:]
                        del stack[-arg:]
                    else:
                        call_args = []
                    fn = pop()
                    # A pulse op named in no scope, as _dispatch_call routes it.
                    if (
                        fn.__class__ is tuple
                        and len(fn) == 2
                        and fn[0] is _UNRESOLVED
                        and not (fn[1] in _PLAYS and arg > 2)
                        and fn[1] in OP_TABLE
                    ):
                        self._push_pulse_results(fn[1], call_args)
                    else:
                        self._dispatch_call(fn, call_args, None)

                elif op is _load_const:
                    push(arg)

                elif op is _store_fast:
                    local_vars[arg] = pop()

                elif op is _load_deref:
                    value = local_vars.get(arg, _MISSING)
                    if value is _MISSING:
                        value = closures.get(arg, _MISSING)
                        if value is _MISSING:
                            value = self._resolve_global(arg)
                        elif value is _EMPTY_CELL:
                            raise CompilationError(
                                f"closure variable {arg!r} is referenced before assignment",
                                code="unassigned_name",
                            )
                    push(value)

                elif op is _binary or op is _compare:
                    right = pop()
                    left = pop()
                    push(arg(left, right))

                elif op is _subscr:
                    index = pop()
                    obj = pop()
                    try:
                        value = obj[index]
                    except PulseError:
                        raise
                    except (TypeError, IndexError, KeyError) as e:
                        raise CompilationError(
                            f"subscript failed: {type(obj).__name__}[{index!r}]: {e}",
                            code="python_error",
                        ) from e
                    push(value)
                    if (
                        value.__class__ is IRValue
                        and obj.__class__ is list
                        and value.vtype in LINEAR_TYPES
                    ):
                        self._slots[id(value)] = (obj, index, value)
                    elif obj.__class__ is dict and dict_items is not None:
                        dict_items[(id(obj), index)] = (obj, index, value)

                elif op is _load_attr:
                    obj = pop()
                    if obj.__class__ is IRValue:
                        # line.tone, read in place: frame ops take it as their tone.
                        chan = obj.chan
                        if arg == "tone" and chan is not None and obj.vtype in _LINE_TYPES:
                            push(chan.held.cells[1].value)
                        else:
                            push(self._value_attribute(obj, arg))
                    elif isinstance(obj, tuple) and len(obj) == 2 and obj[0] == "__unresolved__":
                        push((_UNRESOLVED, arg))
                    else:
                        try:
                            push(getattr(obj, arg))
                        except (AttributeError, TypeError) as e:
                            raise _attribute_error(obj, arg) from e

                elif op is _call_kw:
                    names = pop()
                    play = None
                    if names == _TONE_KEYWORD and arg == 3:
                        # drive(line, waveform, tone=tone): the tone is the op's last operand.
                        fn = stack[-4]
                        if fn.__class__ is types.FunctionType:
                            if fn in _PLAY_FUNCTIONS:
                                play = _PLAY_FUNCTIONS[fn]
                        elif (
                            fn.__class__ is tuple and fn[0] == "__unresolved__" and fn[1] in _PLAYS
                        ):
                            play = fn[1]
                    if play is not None:
                        values = stack[-3:]
                        del stack[-4:]
                        self._push_pulse_results(play, values)
                    else:
                        values = stack[-arg:]
                        del stack[-arg:]
                        positional_count = arg - len(names)
                        kwargs = dict(zip(names, values[positional_count:], strict=False))
                        self._dispatch_call(pop(), values[:positional_count], kwargs)

                elif op is _jif or op is _jit:
                    cond = pop()
                    if cond.__class__ is not bool:
                        if cond.__class__ is _Choice:
                            self._check_choice(cond)
                            cond = _choice_condition(cond)
                        if _is_answer_condition(cond):
                            cond = _trace_condition(self.b, cond)
                        if isinstance(cond, (IRValue, _NotMeasurement)):
                            pc = self._runtime_branch(prog, pc - 1, cond, op is _jit)
                            pc = self._jump(pc, lo, hi, header)
                            if pc < 0:
                                return None
                            continue
                        cond = _truth(cond)
                    if cond == (op is _jit):
                        if arg == header:
                            return None
                        if not lo < arg < hi:
                            raise _NotRollableError
                        pc = arg

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
                                "so they must be finite",
                                code="limit_exceeded",
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

                elif op is _jump:
                    # self._jump(arg, lo, hi, header), inline.
                    if arg == header:
                        return None
                    if not lo < arg < hi:
                        raise _NotRollableError
                    pc = arg

                elif op is _get_iter:
                    value = pop()
                    vt = type(value)
                    if vt is range:
                        push(_Loop(value))
                    elif vt is _Unrolled or vt is _Loop:
                        # iter() of an iterator: a comprehension's on 3.13 and
                        # 3.14, a generator expression's result.
                        push(value)
                    else:
                        if vt is dict and dict_items is not None:
                            self.b.dict_whole.add(id(value))
                        push(_Unrolled(_iterate(value)))
                    if _iter_slots > 1:
                        stack.extend([None] * (_iter_slots - 1))

                elif op is _unpack:
                    val = pop()
                    if not isinstance(val, (tuple, list)):
                        raise CompilationError(_unpack_error(val, arg), code="python_error")
                    if len(val) != arg:
                        raise CompilationError(
                            f"cannot unpack {len(val)} values into {arg} names; "
                            f"the {type(val).__name__} has {len(val)}",
                            code="python_error",
                        )
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

                elif op is _return:
                    if header >= 0:
                        raise _NotRollableError
                    return pop()

                elif op is _list_append or op is _list_extend:
                    val = pop()
                    lst = stack[-arg]
                    if not isinstance(lst, list):
                        raise CompilationError(
                            f"list update on non-list: {type(lst)}", code="python_error"
                        )
                    if op is _list_append:
                        lst.append(val)
                        if val.__class__ is IRValue and val.vtype in LINEAR_TYPES:
                            self._slots[id(val)] = (lst, len(lst) - 1, val)
                    else:
                        lst.extend(val)
                        self._track_slots(lst, 0)

                elif op is _unary:
                    push(arg(pop()))

                elif op is _not:
                    value = pop()
                    if _is_answer_condition(value):
                        value = _trace_condition(self.b, value)
                    if isinstance(value, IRValue):
                        push(_NotMeasurement(value))
                    elif isinstance(value, _NotMeasurement):
                        push(value.value)
                    elif value.__class__ is _Choice:
                        push(_choice_map(operator.not_, value))
                    else:
                        push(not _truth(value))

                elif op is _is:
                    right = pop()
                    left = pop()
                    if left is not None and right is not None and _identity_unknown(left, right):
                        from ._conditions import identity

                        push(identity(left, right) != arg)
                    else:
                        push((left is right) != arg)

                elif op is _contains:
                    right = pop()
                    left = pop()
                    if right.__class__ in _hashed:
                        from ._conditions import hashed_operand

                        right = hashed_operand(self, left, right)
                    try:
                        push((left in right) != arg)
                    except CompilationError as e:
                        from ._conditions import contains

                        push(contains(left, right, e) != arg)

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
                    one_arm = getattr(self, "_one_arm", None)
                    if one_arm:
                        one_arm.pop(arg, None)

                elif op is bc.BUILD_SLICE:
                    items = stack[-arg:]
                    del stack[-arg:]
                    push(slice(*items))

                elif op is bc.STORE_SUBSCR:
                    index = pop()
                    obj = pop()
                    value = pop()
                    self._note_mutation(obj)
                    try:
                        obj[index] = value
                    except (TypeError, IndexError, KeyError, ValueError) as e:
                        raise CompilationError(
                            f"assignment failed: {type(obj).__name__}[{index!r}] = ...: {e}",
                            code="python_error",
                        ) from e
                    if index.__class__ is slice and obj.__class__ is list:
                        self._track_slots(obj, 0)

                elif op is bc.BUILD_CONST_KEY_MAP:
                    keys = pop()
                    vals = stack[-arg:] if arg else []
                    del stack[len(stack) - arg :]
                    push(dict(zip(keys, vals, strict=False)))
                    self._note_fresh(stack[-1])

                elif op is bc.BUILD_MAP:
                    items = stack[len(stack) - 2 * arg :]
                    del stack[len(stack) - 2 * arg :]
                    push(dict(zip(items[::2], items[1::2], strict=False)))
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
                    if any(not loop for _, loop in getattr(self.b, "flow", ())):
                        raise CompilationError(
                            "this assert or raise is in an arm of an if on a value known only "
                            "when the program runs, so whether it fails is decided as the "
                            "program runs, and a kernel cannot raise then; check the value in "
                            "the classical function, or handle each outcome with if/else",
                            code="runtime_value",
                        )
                    if exc is AssertionError or isinstance(exc, AssertionError):
                        detail = f": {exc}" if isinstance(exc, AssertionError) and exc.args else ""
                        raise CompilationError(
                            f"assertion failed in kernel{detail}", code="kernel_raised"
                        )
                    raise CompilationError(f"kernel raised {exc!r}", code="kernel_raised")

                elif op is bc.CALL_EX:
                    star_kwargs = pop() if arg else None
                    star = pop()
                    self._dispatch_call(pop(), _star_args(star), star_kwargs or None)

                elif op is bc.DICT_MERGE:
                    index, strict = arg
                    other = pop()
                    _merge_kwargs(stack[-index], other, strict)

                elif op is bc.DELETE_SUBSCR:
                    index = pop()
                    obj = pop()
                    self._note_mutation(obj)
                    try:
                        del obj[index]
                    except (TypeError, IndexError, KeyError) as e:
                        raise CompilationError(
                            f"del failed: {type(obj).__name__}[{index!r}]: {e}",
                            code="python_error",
                        ) from e
                    if obj.__class__ is list:
                        self._track_slots(obj, 0)

                elif op is bc.LOAD_FAST_AND_CLEAR:
                    push(local_vars.pop(arg, _UNBOUND))

                elif op is bc.STORE_FAST_MAYBE:
                    value = pop()
                    if value is _UNBOUND:
                        local_vars.pop(arg, None)
                    else:
                        local_vars[arg] = value

                elif op is bc.BUILD_SET:
                    items = stack[len(stack) - arg :]
                    del stack[len(stack) - arg :]
                    push(set(items))
                    self._note_fresh(stack[-1])

                elif op is bc.SET_ADD:
                    val = pop()
                    stack[-arg].add(val)

                elif op is bc.SET_UPDATE:
                    val = pop()
                    stack[-arg].update(val)

                elif op is bc.MAP_ADD:
                    val = pop()
                    key = pop()
                    stack[-arg][key] = val

                elif op is bc.UNPACK_EX:
                    before, after = arg
                    val = pop()
                    if val.__class__ is IRValue:
                        raise CompilationError(
                            f"cannot unpack a pulse {val.vtype} value",
                            code="wrong_argument",
                        )
                    try:
                        items = list(val)
                    except TypeError:
                        raise CompilationError(
                            f"cannot unpack {type(val).__name__}", code="python_error"
                        ) from None
                    if len(items) < before + after:
                        raise CompilationError(
                            f"not enough values to unpack: expected at least {before + after}, "
                            f"got {len(items)}",
                            code="python_error",
                        )
                    rest = items[before : len(items) - after]
                    self._track_slots(rest, 0)
                    self._note_fresh(rest)
                    stack.extend(reversed([*items[:before], rest, *items[len(items) - after :]]))

                elif op is bc.MAKE_GENEXPR:
                    code = pop()
                    if arg:  # 3.12 passes the closure cells to MAKE_FUNCTION
                        pop()
                    push(_GenExpr(code))

                elif op is bc.SET_FUNCATTR:
                    fn = pop()
                    pop()
                    push(fn)

                elif op is bc.GEN_YIELD:
                    local_vars[".yielded"].append(pop())
                    push(None)

                elif op is bc.UNSUPPORTED:
                    what, _, fix = arg.partition("; ")
                    raise CompilationError(what, code="unsupported_construct", fix=fix)

                elif op is bc.REGION_END:
                    return _REGION_DONE

                elif op is bc.NOTE_INPLACE:
                    # The augmented assignment next may change its left
                    # operand, under the right one, in place.
                    self._note_mutation(stack[-2])

                else:  # pragma: no cover - decode only produces the ops above
                    raise CompilationError(f"unknown canonical instruction {op}")
        except _NotRollableError:
            raise
        except Exception as e:
            if e.__class__ is TypeError and (op is _unary or op is _binary):
                from ._conditions import operator_error

                error = operator_error(e, arg, (left, right) if op is _binary else None)
                if error is not None:
                    _note_source(error, prog, pc - 1)
                    raise error from e
            _note_source(e, prog, pc - 1)
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
        if value.vtype not in ("measurement", "i1"):
            if value.vtype == "iq_point":
                from ._emit_errors import iq_point_error

                raise iq_point_error("decide a branch")
            raise CompilationError(_BRANCH_TYPE_MESSAGE.format(value.vtype), code="runtime_value")
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
        # A container is copied when an arm first changes it
        # (_note_mutation), with its length: a growing list of outcomes is
        # not copied at every branch. A list holding lines or tones, whose
        # slots the trace rebinds, is copied now, and its slots tracked again
        # after the first arm.
        containers: list[Any] = [v for v in saved_locals.values() if isinstance(v, _MUTABLE)]
        lazy: list[tuple[Any, int]] = []
        if containers:
            found = containers
            containers = []
            rebound = {id(slot[0]) for slot in self._slots.values()}
            # id -> (list, n): the list's first n entries hold no line or
            # tone, looked at once each while they stay in it, unless a call
            # the trace does not follow ran. A line or tone a traced change
            # puts in a list has its slot tracked (rebound).
            plain = getattr(self._lines, "plain", None)
            if plain is None:
                plain = self._lines.plain = {}
            # What a generator yields is added to without a note.
            yielded: Any = saved_locals.get(".yielded")
            for v in found:
                if v is yielded:
                    containers.append((v, list(v)))
                    continue
                if v.__class__ is dict or v.__class__ is not list and not isinstance(v, list):
                    lazy.append((v, len(v)))
                    continue
                if id(v) not in rebound:
                    seen = plain.get(id(v))
                    i = 0 if seen is None or seen[0] is not v else seen[1]
                    n = len(v)
                    if i > n:
                        i = 0
                    while i < n:
                        x = v[i]
                        if x.__class__ is IRValue and x.vtype in LINEAR_TYPES:
                            break
                        i += 1
                    else:
                        plain[id(v)] = (v, n)
                        lazy.append((v, n))
                        continue
                    plain.pop(id(v), None)
                containers.append((v, list(v)))

        # Each channel value entering the branch, as (cell, value).
        held = self._lines.held
        held_count = len(held)
        entering = _channel_values(held) if held_count else ()

        first = builder._next_id  # the first id made in the if
        builder.emit("scf.if", (value,), ())
        then_locals, then_slots, then_copies = self._trace_arm(
            view, index, join, then_start, (containers, lazy)
        )
        then_stack = list(stack)
        then_held = [cell.value for cell, _ in entering]
        stack[:] = saved_stack
        self.locals.clear()
        self.locals.update(saved_locals)
        for obj, copy in containers:
            self._restore_container(obj, copy)
        if then_copies:
            for obj, copy in then_copies:
                self._restore_container(obj, copy)
        _restore_channels(held, held_count, entering)
        else_index = builder.mark()
        builder.emit("scf.else", (), ())
        else_locals, else_slots, else_copies = self._trace_arm(
            view, index, join, else_start, (containers, lazy)
        )
        # Names, and stack slots (ints), the arms leave different numbers,
        # waveforms or conditions on.
        chosen: list[tuple[str | int, Any, Any]] = []
        # Python joins the arms at one stack depth.
        if stack and not returns and any(map(_IS_NOT, stack, then_stack)):
            from ._conditions import stack_choices

            stack_choices(
                self, value, then_stack, saved_stack, jump_if_true, prog, index, first, chosen
            )

        merged: dict[str, StackVal] = {}
        results: list[tuple[str, IRValue, IRValue]] = []
        for name in sorted(then_locals.keys() | else_locals.keys()):
            a = then_locals.get(name, _MISSING)
            b = else_locals.get(name, _MISSING)
            if a is b:
                merged[name] = a
            elif (
                isinstance(a, IRValue)
                and isinstance(b, IRValue)
                and a.vtype == b.vtype
                and a.vtype != "qref"
            ):
                results.append((name, a, b))
            elif a is _MISSING or b is _MISSING:
                # Bound in one arm only: unbound after the if.
                try:
                    self._one_arm[name] = (prog, index)
                except AttributeError:
                    self._one_arm = {name: (prog, index)}
            elif type(a) is type(b) and isinstance(a, _SCALARS) and a == b:
                merged[name] = a
            else:
                from ._conditions import bits_chosen

                if not bits_chosen(value, a, b):
                    self._check_choosable(name, a, b, prog, index, first, value)
                chosen.append((name, a, b))
        slot_results: list[tuple[Any, int, IRValue, IRValue]] = []
        # Channels either arm advanced become results, unless a name
        # carries the same pair of values.
        aliases: list[tuple[_Chan, str]] = []
        by_pair: dict[tuple[int, int], str] | None = None
        for (cell, before), a in zip(entering, then_held, strict=False):
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
        # List slots holding lines or tones that either arm rebound. One that
        # holds the values a name or channel carries takes its result, so
        # that each arm yields a value once.
        listed: list[tuple[Any, int, IRValue, IRValue, IRValue | None]] = []
        if then_slots or else_slots:
            originals = {id(obj): copy for obj, copy in containers}
            if lazy:
                for obj, copy in then_copies + else_copies:
                    originals[id(obj)] = copy
            keys: dict[tuple[int, int], list[Any]] = {}
            for obj, i, _ in then_slots + else_slots:
                keys[(id(obj), i)] = obj
            then_new = {(id(obj), i): new for obj, i, new in then_slots}
            else_new = {(id(obj), i): new for obj, i, new in else_slots}
            carrying = {(id(a), id(b)) for _, a, b in results}
            carrying.update((id(a), id(b)) for _, _, a, b in slot_results)
            for key, obj in keys.items():
                old = originals[key[0]][key[1]]
                a = then_new.get(key, old)
                b = else_new.get(key, old)
                if a is b:
                    continue
                if (id(a), id(b)) in carrying:
                    self._slots.pop(id(a), None)
                    self._slots.pop(id(b), None)
                    listed.append((obj, key[1], a, b, None))
                else:
                    slot_results.append((obj, key[1], a, b))
        if slot_results:
            pairs = {name: (a, b) for name, a, b in results}
            order = self._carry_order(
                [name for name, _, _ in results],
                {n: ab[1] for n, ab in pairs.items()},
                slot_results,
            )
            labels = []
            then_list = []
            else_list = []
            for entry in order:
                if type(entry) is str:
                    a, b = pairs[entry]
                    labels.append(entry)
                else:
                    _, _, a, b = slot_results[entry[0]]
                    labels.append(entry[1])
                then_list.append(a)
                else_list.append(b)
            then_vals = tuple(then_list)
            else_vals = tuple(else_list)
        else:
            order = [name for name, _, _ in results]
            labels = order
            then_vals = tuple(a for _, a, _ in results)
            else_vals = tuple(b for _, _, b in results)
        builder.insert(else_index, "scf.yield", then_vals)
        builder.emit("scf.yield", else_vals, ())
        outs = builder.emit(
            "scf.if_end",
            (),
            tuple(v.vtype for v in then_vals),
            {"result_names": labels},
        )
        carried: dict[tuple[int, int], IRValue] = {}
        for entry, out in zip(order, outs, strict=False):
            if type(entry) is str:
                merged[entry] = out
                continue
            obj, i, a, b = slot_results[entry[0]]
            self._slots.pop(id(a), None)
            self._slots.pop(id(b), None)
            if isinstance(obj, _Held):
                carried[(id(a), id(b))] = out
                obj[i] = out
            else:
                listed.append((obj, i, a, b, out))
        if listed:
            for name, a, b in results:
                carried.setdefault((id(a), id(b)), merged[name])
        # A list slot that holds its channel's values takes the channel's result.
        for obj, i, a, b, out in listed:
            out = carried.get((id(a), id(b)), out)
            assert out is not None
            obj[i] = out
            self._slots[id(out)] = (obj, i, out)
        if chosen:
            from ._conditions import chosen_value

            line = prog.location(index)[1]
            scope = getattr(builder, "scope", ())
            for chose, a, b in chosen:
                if isinstance(chose, int):
                    stack[chose] = chosen_value(self, value, a, b, "", line, scope)
                else:
                    merged[chose] = chosen_value(self, value, a, b, chose, line, scope)
        for cell, alias in aliases:
            out = merged[alias]
            cell.value = out
            out.chan = cell
        # A channel first acquired in one arm is not held after the if.
        if len(held) != held_count:
            _restore_channels(held, held_count, ())
        self.locals.clear()
        self.locals.update(merged)
        if returns:
            stack[:] = saved_stack
            # Both arms returned: continue at the trailing ``return None``.
            return len(prog.instrs) - 2
        return join

    def _check_choosable(
        self,
        name: str,
        a: Any,
        b: Any,
        prog: bc.DecodedCode,
        index: int,
        first: int,
        cond: IRValue,
    ) -> None:
        """Reject the values *a* and *b* the arms of the if on *cond* at
        ``prog.instrs[index]`` leave on *name* (on the stack, for ``""``:
        the value of a conditional expression, ``and`` or ``or``), unless
        the if can choose between them: two numbers or two waveforms that
        exist after it."""
        line = prog.location(index)[1]
        measurement = cond.vtype == "measurement"
        on = "a measurement result" if measurement else "a runtime condition"
        if name:
            subject = f"{name!r}"
            where = (
                f"the {'measurement' if measurement else 'runtime'}-conditioned if at line {line}"
            )
            inside, outside = f"use {name!r} inside that arm", "use it inside each arm"
        else:
            subject = "the value of the conditional expression, `and` or `or`"
            where = f"that expression at line {line}"
            inside = outside = "use an if statement with the op in each arm"
        from ._conditions import is_condition, truth_choice

        if truth_choice(cond, a, b):
            # A condition on measurement results, chosen as one.
            return
        if measurement and (is_condition(a) or is_condition(b)):
            raise CompilationError(
                f"{subject} would depend on both a measurement result and a condition on "
                f"answers at line {line}, which one test cannot combine; branch on them in "
                "nested ifs, or with `and` and `or` in the if condition",
                code="runtime_value",
            )
        kind = _choice_kind(a)
        if kind is not None and kind == _choice_kind(b):
            scope = getattr(self.b, "scope", ())[:-1]  # around the if: this is its else arm
            if not (_made_in_if(a, first, scope) or _made_in_if(b, first, scope)):
                return
            raise CompilationError(
                f"{subject} would depend on {on} through a value made inside an arm of "
                f"{where}, which does not exist after it; {inside}, or make the value "
                "before it",
                code="runtime_value",
            )
        raise CompilationError(
            f"{subject} would depend on {on}: the arms of {where} leave {_describe(a)} and "
            f"{_describe(b)} on it, and an if chooses only between two numbers or two "
            "waveforms, two lines or tones of one kind, or two conditions on answers; "
            f"compute it before the if, or {outside}",
            code="runtime_value",
        )

    def _undefined(self, name: str) -> _UndefinedLocalError:
        """The error for reading *name*, which is not bound."""
        one_arm = getattr(self, "_one_arm", None)
        where = one_arm.get(name) if one_arm else None
        if where is None:
            return _UndefinedLocalError(name)
        prog, index = where
        line = prog.location(index)[1]
        return _UndefinedLocalError(
            name,
            f"{name!r} is assigned in only one arm of the measurement-conditioned if "
            f"at line {line} and has no value before it, so it has no value after the "
            "if when the other arm runs; assign it in both arms, or before the if",
        )

    def _trace_arm(
        self,
        view: bc.DecodedCode,
        branch: int,
        join: int,
        start: int,
        snapshot: tuple[list[tuple[Any, Any]], list[tuple[Any, int]]],
    ) -> tuple[dict[str, StackVal], list[tuple[list[Any], int, IRValue]], list[tuple[Any, Any]]]:
        """Trace one arm until it reaches *join*; return its locals, rebound
        slots and the contents before it of the containers it changed. Of the
        *snapshot* of containers the arm starts with, the first are copied and
        the others (each with its length) are copied when first changed."""
        containers, lazy = snapshot
        returns = join >= len(view.instrs)
        # An arm may only move forward from the branch, except that a join at
        # an enclosing loop header (both arms continue) ends the arm.
        if join < branch:
            header = join
            # There a comprehension's if adds to what it builds, on the stack.
            if not _CONTAINERS.isdisjoint(map(type, self.stack)):
                from ._conditions import add_built

                add_built(self.stack, containers)
        else:
            header = -1
        outer = self._mutated
        mutated: dict[int, tuple[Any, Any]] = {}
        self._mutated = mutated
        if lazy:
            lines = self._lines
            around = getattr(lines, "watched", ())
            lines.watched = ((mutated, tuple(obj for obj, _ in lazy)), *around)
        try:
            outcome = self._loop(view, start, branch, len(view.instrs), header)
        except _NotRollableError:
            from ._conditions import rejoin_error

            raise rejoin_error(view, branch) from None
        finally:
            self._mutated = outer
            if lazy:
                lines.watched = around
            if outer is not None:
                for key, entry in mutated.items():
                    outer.setdefault(key, entry)
        if header >= 0 and outcome is None:
            outcome = _REGION_DONE
        if returns:
            if outcome is not None:
                raise CompilationError(
                    "a measurement-conditioned branch cannot return a value",
                    code="runtime_value",
                )
        elif outcome is not _REGION_DONE:
            from ._conditions import rejoin_error

            raise rejoin_error(view, branch)
        copies: list[tuple[Any, Any]] = []
        slots: list[tuple[list[Any], int, IRValue]] | None
        if lazy:
            slots = None
            for obj, length in lazy:
                noted = mutated.get(id(obj))
                if noted is not None and noted[1] is not None:
                    copies.append(noted)
                elif len(obj) != length:
                    break
            else:
                slots = _slot_changes(containers + copies)
        elif containers:
            slots = _slot_changes(containers)
        else:
            slots = []
        if slots is not None and mutated:
            # Any other container the arm changed was made in it; one a call
            # the trace does not follow may have changed must be as it was.
            known = {id(obj) for obj, _ in containers}
            known.update(id(obj) for obj, _ in lazy)
            for key, (obj, copy) in mutated.items():
                if (
                    copy is not None
                    and key not in known
                    and not (
                        obj == copy if obj.__class__ is set else _container_unchanged(obj, copy)
                    )
                ):
                    slots = None
                    break
        if slots is None:
            from ._conditions import filtered

            if header >= 0 and filtered(self, containers):
                raise CompilationError(
                    "a comprehension or generator expression cannot filter on a condition "
                    "known only when the program runs: which items it has would be decided "
                    "as the program runs; loop with a for statement and branch with an if "
                    "statement in its body instead",
                    code="runtime_value",
                )
            raise CompilationError(
                "a measurement-conditioned branch may only change list entries that "
                "hold lines or tones",
                code="runtime_value",
            )
        return dict(self.locals), slots, copies

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
        self,
        loop: _Loop,
        prog: bc.DecodedCode,
        for_idx: int,
        exit_idx: int,
        hoisted: bool = False,
        unbound: dict[str, StackVal] | None = None,
    ) -> bool:
        """Emit *loop* as one ``scf.for``; return False to unroll it instead.

        The body is traced once. Rolling is kept only if that iteration
        returned to the loop header and left every Python-level value
        (non-IR locals and the lists and dicts they hold, and any container
        a called kernel or helper reaches) unchanged, so all iterations would
        trace identically. Otherwise the emitted ops are rolled back.
        The names in *unbound* are left unbound while the body is traced.
        """
        rng = loop.rng
        stack = self.stack
        saved_stack = list(stack)
        if unbound:
            for n in unbound:
                del self.locals[n]
        saved_locals = dict(self.locals)
        containers = [
            (v, list(v) if isinstance(v, list) else dict(v))
            for v in saved_locals.values()
            if isinstance(v, (list, dict))
        ]
        lines = self._lines
        # Each channel value entering the loop, as (cell, value).
        held = lines.held
        held_count = len(held)
        entering = _channel_values(held) if held_count else ()
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
        stale: list[tuple[str, _Chan]] = []
        crossing: list[str] = []
        fixed: list[str] = []
        try:
            self._loop(prog, for_idx + 1, for_idx, exit_idx, for_idx)
            slot_changes = _slot_changes(containers)
            rebound = self._rebound(saved_locals, var)
            ok = (
                len(stack) == len(saved_stack)
                and rebound is not None
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
            if ok:
                delta = self._ir_delta(snap)
                stale, crossing = _stale_names(delta, saved_locals, count)
                fixed, moved = _qudit_names(delta, saved_locals, count)
                assert rebound is not None
                crossing += moved + rebound
                if crossing:
                    # Traced again with them unbound, to learn whether the
                    # body reads them before rebinding them.
                    ok = False
        except _NotRollableError:
            ok = False
        except _UndefinedLocalError as e:
            # A name that enters on one channel and leaves on another is read
            # before it is rebound: iterations after the first differ.
            if unbound is None or e.name not in unbound:
                raise
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
            if unbound:
                self.locals.update(unbound)
            for obj, copy in containers:
                self._restore_container(obj, copy)
            _restore_channels(held, held_count, entering)
            # Acquisitions the undone body made did not happen.
            self._acquired[0] = acquired
            undone = log[mark:]
            del log[mark:]
            if hoist:
                for fname, q in dict.fromkeys(undone):
                    self._emit_pulse_call(fname, [q])
                return self._roll(loop, prog, for_idx, exit_idx, True, unbound)
            if crossing:
                unbound = {**(unbound or {}), **{n: saved_locals[n] for n in crossing}}
                return self._roll(loop, prog, for_idx, exit_idx, hoisted, unbound)
            return False

        assert slot_changes is not None
        lists_carried = True if slot_changes else False
        for n, _ in stale:
            del delta[n]
        qudits = [(n, delta.pop(n)) for n in fixed]
        # Channels the body advanced are carried, unless a name carries the
        # same value; a channel first acquired in the body was hoisted.
        aliases: list[tuple[_Chan, str]] = []
        originals = {id(obj): copy for obj, copy in containers}
        # A list slot the body left on an old value of the channel it entered
        # on (an op rebound a copy of it, as ``sync(*a, *b)`` does) names the
        # channel's current value, as a name does (``_stale_names``).
        for k, (obj, i, new) in enumerate(slot_changes):
            chan = new.chan
            if chan is not None and chan.value is not new and originals[id(obj)][i].chan is chan:
                slot_changes[k] = (obj, i, chan.value)
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
            order: list[Any] = (
                self._carry_order(names, delta, slot_changes) if slot_changes else names
            )
            # A name first bound in the body has no value entering the loop;
            # the body binds it before reading it, unless it binds it to a
            # channel's value read from a container: then it enters with
            # the channel's value.
            before_of: dict[int, IRValue] | None = None
            vals_list = []
            entering_vals = []
            labels = []
            kept: list[Any] = []
            # A slot that enters and leaves with the values an earlier entry
            # carries takes that entry's result, not a loop value of its own;
            # only a list slot can repeat another entry.
            owner: dict[tuple[int, int], int] | None = {} if lists_carried else None
            shared: list[tuple[Any, int]] = []
            for n in order:
                if type(n) is not str:
                    obj, i, new = slot_changes[n[0]]
                    before = originals[id(obj)][i]
                    if owner is not None:
                        at = owner.setdefault((id(new), id(before)), len(kept))
                        if at != len(kept):
                            shared.append((n, at))
                            continue
                    label = n[1]
                else:
                    new = delta[n]
                    if n in saved_locals:
                        before = saved_locals[n]
                    else:
                        if before_of is None:
                            before_of = {id(cell): before for cell, before in entering}
                        before = self._entering(new, before_of)
                    if owner is not None:
                        owner.setdefault((id(new), id(before)), len(kept))
                    label = n
                kept.append(n)
                vals_list.append(new)
                entering_vals.append(before)
                labels.append(label)
            vals = tuple(vals_list)
            inits = tuple(entering_vals)
            self.b.emit("scf.yield", vals, ())
            # The body was traced with the values carried into the loop; its
            # end lists them so that each iteration can rebind them.
            results = self.b.emit(
                "scf.for_end", inits, tuple(v.vtype for v in vals), {"iter_args": labels}
            )
            listed = []
            carried: dict[int, tuple[IRValue, IRValue]] = {}
            for n, r in (
                zip(kept, results, strict=False)
                if not shared
                else [*zip(kept, results, strict=False), *((n, results[at]) for n, at in shared)]
            ):
                if type(n) is str:
                    self.locals[n] = r
                    continue
                obj, i, new = slot_changes[n[0]]
                if isinstance(obj, _Held):
                    carried[id(obj.cells[i])] = (new, r)
                    obj[i] = r
                else:
                    listed.append((obj, i, new, r))
            if listed:
                for cell, alias in aliases:
                    carried[id(cell)] = (cell.value, self.locals[alias])
            # A list slot that yields its channel's value takes the channel's
            # result, so the line read from it after the loop has its tone.
            for obj, i, new, r in listed:
                change = carried.get(id(new.chan))
                if change is not None and change[0] is new:
                    r = change[1]
                obj[i] = r
                self._slots[id(r)] = (obj, i, r)
        else:
            self.b.emit("scf.for_end", (), ())
        for cell, alias in aliases:
            result = self.locals[alias]
            cell.value = result
            result.chan = cell
        for n, chan in stale:
            self.locals[n] = chan.value
        for n, qudit in qudits:
            if count:
                self.locals[n] = qudit
            elif n in saved_locals:
                self.locals[n] = saved_locals[n]
            else:
                del self.locals[n]
        if not count and unbound:
            # A loop that never runs rebinds no Python value.
            for n, value in unbound.items():
                if not isinstance(value, IRValue):
                    self.locals[n] = value

        if count:
            self.locals[var] = rng[-1]
        elif var in saved_locals:
            self.locals[var] = saved_locals[var]
        else:
            self.locals.pop(var, None)
        return True

    @staticmethod
    def _entering(value: IRValue, before_of: dict[int, IRValue]) -> IRValue:
        """The value entering a rolled loop of a name the body first binds to *value*."""
        chan = value.chan
        before = None if chan is None else before_of.get(id(chan))
        return IRValue(UNDEFINED_VID, value.vtype) if before is None else before

    def _unrolls(self, loop: _Loop, prog: bc.DecodedCode, for_idx: int, exit_idx: int) -> bool:
        """Whether *loop* is small enough to unroll: its trip count times the
        calls in its body (at least one) is at most the unroll threshold."""
        threshold = self._unroll
        try:
            count = len(loop.rng)
        except OverflowError:
            raise CompilationError(
                f"for loop over {loop.rng!r} runs more than 2**63 - 1 times, "
                "the most a loop can run",
                code="limit_exceeded",
            ) from None
        if threshold == 0 or count > threshold:
            return False
        calls = 0
        for op, _ in prog.instrs[for_idx + 1 : exit_idx]:
            if op is CALL or op is bc.CALL_KW or op is bc.CALL_EX:
                calls += 1
        return count * max(calls, 1) <= threshold

    def _note_mutation(self, obj: Any) -> None:
        """Remember what *obj* held before a rolled body or branch arm changed it."""
        mutated = self._mutated
        if mutated is not None and id(obj) not in mutated:
            if obj.__class__ in _CONTAINERS:
                mutated[id(obj)] = (obj, obj.copy())
            elif isinstance(obj, _MUTABLE):
                mutated[id(obj)] = (obj, _contents(obj))

    def _note_fresh(self, obj: Any) -> None:
        """Mark a container built by the body or arm being traced as free to change."""
        if self._mutated is not None:
            self._mutated.setdefault(id(obj), (obj, None))

    def _carry_order(
        self, names: list[str], values: dict[str, Any], slots: list[tuple[Any, ...]]
    ) -> list[Any]:
        """The carried *names* and (slot index, label) entries, in their order.

        A line's own tone follows the name the line is carried by, as
        ``name.tone``; the other slots come last.
        """
        by_value = {id(v): n for n, v in values.items()}
        keyed: list[tuple[str, Any]] = [(n, n) for n in names]
        rest: list[Any] = []
        unsorted = False
        for k, slot in enumerate(slots):
            obj, i = slot[0], slot[1]
            if i == 1 and type(obj) is _Held:
                line = obj.cells[0].value
                n = by_value.get(id(line))
                if n is None:
                    n = next((m for m, v in self.locals.items() if v is line), None)
                if n is not None:
                    label = n + ".tone"
                    if keyed and label < keyed[-1][0]:
                        unsorted = True
                    keyed.append((label, (k, label)))
                    continue
            rest.append((k, self._slot_label(obj, i)))
        if unsorted:
            keyed.sort(key=_first)
        return [entry for _, entry in keyed] + rest

    def _slot_label(self, obj: Any, index: int) -> str:
        if isinstance(obj, _Held):
            return f"{obj.label}[{index}]"
        name = next((n for n, v in self.locals.items() if v is obj), "")
        return f"{name}[{index}]"

    def _rebound(self, before: dict[str, StackVal], var: str) -> list[str] | None:
        """The locals *before* a traced loop iteration that it rebound to
        other Python values or IR values of another type (an inner loop's
        variable, say), or ``None`` when it deleted one. The loop variable
        *var* is exempt: the header binds it and the body never reads it, so
        what it held before the loop (an earlier loop's last value) is not
        state the iterations depend on; nor is a rebound name if the body
        binds it before reading it, which tracing the body again with it
        unbound tells."""
        after = self.locals
        rebound = []
        for name, old in before.items():
            if name == var:
                continue
            if isinstance(old, IRValue):
                new = after.get(name, old)
                if not isinstance(new, IRValue) or new.vtype != old.vtype:
                    rebound.append(name)
                continue
            if name not in after:
                return None
            new = after[name]
            if new is not old and not (
                type(new) is type(old) and isinstance(old, _SCALARS) and new == old
            ):
                rebound.append(name)
        return rebound

    # ── Global / closure resolution ──────────────────────────────────

    def _resolve_global(self, name: str) -> Any:
        """Resolve a global name to its Python value."""
        if name in self._fn_globals:
            return self._fn_globals[name]
        if name in self._fn_closures:
            value = self._fn_closures[name]
            if value is _EMPTY_CELL:
                raise CompilationError(
                    f"closure variable {name!r} is referenced before assignment",
                    code="unassigned_name",
                )
            return value
        if name in self._builtins:
            return self._builtins[name]
        return (_UNRESOLVED, name)

    def _unknown_name(self, name: str) -> CompilationError:
        """The error for calling *name*, which nothing in scope defines."""
        known = [*OP_TABLE, *self.locals, *self._fn_globals, *self._fn_closures, *self._builtins]
        from ._emit_errors import unknown

        return unknown(f"unknown pulse op: {name}", "unknown_name", name, known, call=True)

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
                if kwargs and name == "readout" and "acquire" in kwargs:
                    kwargs = dict(kwargs)
                    acquire = _acquire_mode(kwargs.pop("acquire"))
                    if acquire is not None:
                        self._acquired_readout(callable_val, call_args, kwargs, acquire)
                        return
                if (
                    kwargs
                    and name in _PLAYS
                    and "tone" in kwargs
                    and call_args[1:]
                    and not call_args[2:]
                    # Equal by identity: the tone's own __eq__ is not called.
                    and kwargs == {"tone": kwargs["tone"]}
                ):
                    call_args = [*call_args, kwargs["tone"]]
                elif kwargs:
                    from .. import ops

                    op_fn = getattr(ops, name, None) if name in _PULSE_OP_NAMES else None
                    if op_fn is None:
                        raise self._unknown_name(name)
                    call_args = _bind_positional(op_fn, call_args, kwargs)
                elif name in _PLAYS and len(call_args) > 2:
                    from ._emit_errors import play_arity_error

                    raise play_arity_error(name, len(call_args))
                if name in OP_TABLE:
                    self._push_pulse_results(name, call_args)
                elif name == "range":
                    self._push_range(call_args)
                else:
                    raise self._unknown_name(name)
                return

        # Resolved callable: a DSL op only if it is the DSL function itself,
        # so a user function that happens to be named ``gaussian`` stays Python.
        try:
            # Functions hash and compare by identity.
            op_name = _PULSE_FUNCTIONS.get(callable_val)
        except TypeError:  # unhashable callable
            op_name = None
        if op_name is not None:
            if kwargs and op_name == "readout" and "acquire" in kwargs:
                kwargs = dict(kwargs)
                acquire = _acquire_mode(kwargs.pop("acquire"))
                if acquire is not None:
                    self._acquired_readout(callable_val, call_args, kwargs, acquire)
                    return
            if (
                kwargs
                and op_name in _PLAYS
                and "tone" in kwargs
                and call_args[1:]
                and not call_args[2:]
                # Equal by identity: the tone's own __eq__ is not called.
                and kwargs == {"tone": kwargs["tone"]}
            ):
                call_args = [*call_args, kwargs["tone"]]
            elif kwargs:
                call_args = _bind_positional(callable_val, call_args, kwargs)
            elif op_name in _PLAYS and call_args[2:]:
                from ._emit_errors import play_arity_error

                raise play_arity_error(op_name, len(call_args))
            self._push_pulse_results(op_name, call_args)
            return

        # Calls to classical functions: fn.submit(...) and future.result().
        if callable_val.__class__ is types.MethodType:
            owner = callable_val.__self__
            if owner.__class__ is ClassicalFunction and callable_val.__func__ is _SUBMIT:
                self.stack.append(_trace_submit(self.b, owner, call_args, kwargs))
                return
            if owner.__class__ is Future and callable_val.__func__ is _RESULT:
                self.stack.append(_trace_take(self.b, owner, call_args, kwargs))
                return
        elif callable_val.__class__ is ClassicalFunction:
            raise CompilationError(
                f"in a kernel, call {callable_val.name}.submit(...) and take the answer "
                f"with .result(); calling {callable_val.name}(...) runs it on the host",
                code="unsupported_construct",
            )

        inner = getattr(callable_val, "__pulse_kernel_fn__", None)
        if inner is not None:
            # A kernel with its own policy is scheduled on its own, in a block.
            policy = getattr(callable_val, "schedule", None)
            budget = block_budget(callable_val, self.b)
            if policy is not None and getattr(self.b, "schedule_blocks", False):
                self.b.begin_schedule_block(policy, inner.__name__, budget)
                self._inline_helper(inner, call_args, kwargs)
                self.b.end_schedule_block()
            else:
                self._inline_helper(inner, call_args, kwargs)
            return

        fname = getattr(callable_val, "__name__", "")
        if callable_val is range:
            if kwargs:
                raise CompilationError("range() takes no keyword arguments", code="wrong_argument")
            self._push_range(call_args)
            return

        # Inline user helpers which consume IR values or call the pulse DSL.
        # Executing these as ordinary Python would invoke pulse operations
        # outside the compiler context and lose their IR effects.
        if isinstance(callable_val, types.FunctionType) and (
            _takes_ir(call_args, kwargs) or _uses_pulse_ops(callable_val)
        ):
            self._inline_helper(callable_val, call_args, kwargs)
            return
        # A method of a helper object is inlined the same way, with its object
        # (its class for a classmethod) as the first argument.
        if (
            callable_val.__class__ is types.MethodType
            and callable_val.__func__.__class__ is types.FunctionType
            and (
                _takes_ir(call_args, kwargs)
                or _uses_pulse_ops(callable_val.__func__, owner=callable_val.__self__)
            )
        ):
            self._inline_helper(callable_val.__func__, [callable_val.__self__, *call_args], kwargs)
            return
        if (
            callable_val.__class__ is functools.partial
            and callable_val.func.__class__ is types.FunctionType
            and _uses_pulse_ops(callable_val.func)
        ):
            merged = {**callable_val.keywords, **(kwargs or {})}
            self._inline_helper(callable_val.func, [*callable_val.args, *call_args], merged or None)
            return

        if callable_val.__class__ is _GenExpr:
            self._run_genexpr(callable_val.code, call_args)
            return
        # An object whose class defines __call__ in Python is a method call.
        if callable_val.__class__ not in _NATIVE_CALLABLES:
            call = getattr(callable_val.__class__, "__call__", None)  # noqa: B004
            if (
                call.__class__ is types.FunctionType
                and not isinstance(callable_val, type)
                and (
                    _takes_ir(call_args, kwargs)
                    or _uses_pulse_ops(call, owner=callable_val, strict=True)
                )
            ):
                self._inline_helper(call, [callable_val, *call_args], kwargs)
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
                if callable_val is bool and call_args[0].__class__ is AnswerValue:
                    self.stack.append(UndecidableComparison(call_args[0], "!=", 0))
                    return
            if any(is_symbolic(a) for a in call_args) or (
                kwargs and any(is_symbolic(v) for v in kwargs.values())
            ):
                if _find_choice([*call_args, *(kwargs or {}).values()]) is not None:
                    if self._call_on_choice(callable_val, fname, call_args, kwargs):
                        return
                else:
                    names = [
                        n
                        for a in [*call_args, *(kwargs or {}).values()]
                        for n in _parameter_names(a)
                    ]
                    raise SymbolicValueError(
                        f"Cannot call {fname}() with symbolic parameter "
                        "expressions. Only int(), float(), arithmetic, and pulse "
                        "numeric arguments are supported.",
                        tuple(dict.fromkeys(names)),
                        f"{fname}()",
                    )
            if not _is_pure_host_call(callable_val):
                try:
                    self.b.impure = True
                except AttributeError:
                    pass
            owner = getattr(callable_val, "__self__", None)
            if self._mutated is not None:
                if owner.__class__ in _CONTAINERS:
                    if fname not in _READ_METHODS:
                        self._note_mutation(owner)
                elif isinstance(owner, _MUTABLE):
                    # A method of another container changes that container only.
                    if fname not in _READ_METHODS:
                        self._note_mutation(owner)
                else:
                    # A call the trace does not follow may change any
                    # container an arm around it copies only when it first
                    # changes, and the lists known to hold no line or tone.
                    lines = self._lines
                    for mutated, watched in getattr(lines, "watched", ()):
                        for obj in watched:
                            if id(obj) not in mutated:
                                mutated[id(obj)] = (obj, obj.copy())
                    plain = getattr(lines, "plain", None)
                    if plain:
                        plain.clear()
            try:
                result = callable_val(*call_args, **(kwargs or {}))
            except _EVALUATION_ERRORS as e:
                if any(a.__class__ is IRValue and a.vtype == "measurement" for a in call_args):
                    from ._conditions import measurement_operand_error

                    raise measurement_operand_error(f"{fname}()") from e
                raise CompilationError(
                    f"failed to evaluate {fname}({call_args!r}): {e}", code="python_error"
                ) from e
            except NameError as e:
                if e.name not in _PULSE_OP_NAMES:
                    raise
                raise CompilationError(
                    f"{fname}() calls the pulse op {e.name}() but ran as plain Python: "
                    "only functions, methods and kernels are inlined into a kernel; "
                    f"call {e.name}() from a module-level function instead",
                    code="unsupported_construct",
                ) from e
            except CompilationError as e:
                # bool() of a condition known only at run time is that condition.
                if callable_val is bool and not kwargs and _runtime_bit(call_args[0]):
                    self.stack.append(call_args[0])
                    return
                if (callable_val is any or callable_val is all) and len(call_args) == 1:
                    from ._conditions import any_or_all

                    items = call_args[0]
                    generated = items.__class__ is _Unrolled
                    self.stack.append(any_or_all(callable_val is any, items, e, generated))
                    return
                raise
            self.stack.append(result)
            if (callable_val.__class__ is type and callable_val in _CONTAINERS) or (
                owner.__class__ in _CONTAINERS and fname == "copy"
            ):
                self._note_fresh(result)
            if owner.__class__ is list and fname in _MOVES_SLOTS:
                self._track_slots(owner, len(owner) - 1 if fname == "append" else 0)
            return

        raise CompilationError(
            f"cannot call an object of type {type(callable_val).__name__!r} in a kernel; "
            "a kernel calls pulse ops, other kernels, Python functions and methods, "
            "and classical functions' .submit()",
            code="unsupported_construct",
        )

    def _call_on_choice(
        self,
        fn: Any,
        fname: str,
        call_args: list[StackVal],
        kwargs: dict[str, StackVal] | None,
    ) -> bool:
        """Call *fn* with arguments holding a choice: inline a Python helper,
        or apply a pure function to each value the choice may take. False
        for any other callable, which is called with the choice itself."""
        if fn.__class__ is types.FunctionType:
            self._inline_helper(fn, call_args, kwargs)
            return True
        if fn.__class__ is types.MethodType and fn.__func__.__class__ is types.FunctionType:
            self._inline_helper(fn.__func__, [fn.__self__, *call_args], kwargs)
            return True
        if isinstance(fn, type):
            if fn not in (int, float, bool, complex):
                return False
        else:
            owner = getattr(fn, "__self__", None)
            if owner is not None and not isinstance(owner, types.ModuleType):
                return False
            if not _is_pure_host_call(fn):
                return False
        keys = list(kwargs or {})
        positional_count = len(call_args)

        def call(*values: Any) -> Any:
            return fn(
                *values[:positional_count],
                **dict(zip(keys, values[positional_count:], strict=False)),
            )

        try:
            result = _choice_map(call, *call_args, *(kwargs or {}).values())
        except _EVALUATION_ERRORS as e:
            raise CompilationError(
                f"failed to evaluate {fname}({call_args!r}): {e}", code="python_error"
            ) from e
        self.stack.append(result)
        return True

    def _check_choice(self, choice: _Choice) -> None:
        """Reject *choice* outside the regions where its if is."""
        if choice.vtype == "condition":
            from ._conditions import note_builder

            note_builder(self.b)
        scope = choice.scope
        if scope and getattr(self.b, "scope", ())[: len(scope)] != scope:
            raise CompilationError(
                f"{choice.origin} is used outside the loop or branch that if is in; use it there",
                code="runtime_value",
            )

    def _emit_chosen(
        self,
        fname: str,
        op_name: str,
        operands: tuple[Any, ...],
        out_types: tuple[str, ...],
        attrs: dict[str, Any],
        nested: bool = False,
    ) -> tuple[Any, ...]:
        """Emit the op *fname* given a choice, once: each number or waveform
        it chooses is one select on the choice's condition, which every op
        given that choice reads. Lowering keeps a select where the target
        takes the value at run time, and otherwise emits each op that reads
        it in both arms of an if on the condition. Choices a select cannot
        hold (inside a list, of complex numbers, or that a check needs the
        value of) are emitted per arm instead. A *nested* choice, one an arm
        of another gives, was checked with it."""
        given = [*operands, *attrs.values()]
        if not nested:
            for root in _choices_in(given):
                if root.vtype == "condition":
                    from ._conditions import not_a_number

                    not_a_number(root, f"{fname}()")
                self._check_choice(root)
        b = self.b
        select = getattr(b, "select", None)
        if select is None or not _selectable(fname, operands, attrs):
            return self._emit_per_arm(fname, op_name, operands, out_types, attrs)
        operands = tuple(
            self._select_waveform(v) if v.__class__ is _Choice else v for v in operands
        )
        if getattr(self, "_sampled", None) is not None and fname in ("drive", "readout"):
            from ._durations import for_line

            operands = for_line(self, fname, operands)
        results: tuple[Any, ...] = b._unwatched_emit(op_name, operands, out_types, attrs)
        if fname in _MAKES_WAVEFORM:
            if fname in _WAVEFORMS:
                duration = attrs.get("duration")
            else:
                duration = self._waveform_duration(fname, operands, attrs)
            if duration.__class__ is int:
                self._durations[results[0].vid] = duration
            else:
                self._durations.pop(results[0].vid, None)
            if getattr(self, "_sampled", None) is not None:
                from ._durations import track_samples

                track_samples(self, fname, None, operands, out_types, attrs, results[0].vid)
        return results

    def _select_waveform(self, choice: _Choice) -> IRValue:
        """The select of the waveforms *choice* holds, with their duration if
        they all have the same one."""
        value = IRValue(self.b.select(choice, "waveform"), "waveform")
        known = self._durations
        durations = {known.get(leaf.vid) for leaf in _choice_leaves(choice)}
        duration = durations.pop() if len(durations) == 1 else None
        if duration is None:
            known.pop(value.vid, None)
        else:
            known[value.vid] = duration
        if getattr(self, "_sampled", None) is not None:
            from ._durations import track_samples

            track_samples(self, "select", None, tuple(_choice_leaves(choice)), (), {}, value.vid)
        return value

    def _emit_per_arm(
        self,
        fname: str,
        op_name: str,
        operands: tuple[Any, ...],
        out_types: tuple[str, ...],
        attrs: dict[str, Any],
    ) -> tuple[Any, ...]:
        """Emit the op *fname* given a choice no select holds: a waveform op
        once for each value the choice may take, making a choice of
        waveforms; any other op in each arm of an if on the choice's
        condition, as if the kernel had written it in both arms."""
        choice = _find_choice([*operands, *attrs.values()])
        assert choice is not None
        cond = choice.cond
        arms = [
            (
                tuple(_in_arm(v, cond, taken) for v in operands),
                {k: _in_arm(v, cond, taken) for k, v in attrs.items()},
            )
            for taken in (True, False)
        ]
        if fname in _MAKES_WAVEFORM:
            then = self._emit_resolved(fname, op_name, *arms[0], out_types)
            other = self._emit_resolved(fname, op_name, *arms[1], out_types)
            return (_choose(cond, then[0], other[0], choice, choice.scope),)
        b = self.b
        b.emit("scf.if", (cond,), ())
        then = self._emit_resolved(fname, op_name, *arms[0], out_types)
        b.emit("scf.yield", then, ())
        b.emit("scf.else", (), ())
        other = self._emit_resolved(fname, op_name, *arms[1], out_types)
        b.emit("scf.yield", other, ())
        types_ = tuple(v.vtype for v in then)
        return b.emit("scf.if_end", (), types_, {"result_names": list(types_)})  # type: ignore[no-any-return]

    def _emit_resolved(
        self,
        fname: str,
        op_name: str,
        operands: tuple[Any, ...],
        attrs: dict[str, Any],
        out_types: tuple[str, ...],
    ) -> tuple[Any, ...]:
        """Emit the op *fname* with the values an arm of a choice gives it."""
        if getattr(self, "_sampled", None) is not None and fname in ("drive", "readout"):
            from ._durations import for_line

            operands = for_line(self, fname, operands)
        try:
            results: tuple[Any, ...] = self.b.emit(op_name, operands, out_types, attrs)
        except _ChosenOperandError:
            return self._emit_chosen(fname, op_name, operands, out_types, attrs, nested=True)
        if fname in _MAKES_WAVEFORM:
            if fname in _WAVEFORMS:
                duration = attrs.get("duration")
            else:
                duration = self._waveform_duration(fname, operands, attrs)
            if duration.__class__ is int:
                self._durations[results[0].vid] = duration
            if getattr(self, "_sampled", None) is not None:
                from ._durations import track_samples

                track_samples(self, fname, op_name, operands, out_types, attrs, results[0].vid)
        return results

    def _run_genexpr(self, code: types.CodeType, call_args: list[StackVal]) -> None:
        """Run a generator expression to the end, and push an iterator over
        what it yields: it is consumed where it is made, so it is evaluated
        there, with its pulse ops in order."""
        closures = {}
        for name in code.co_freevars:
            value = self.locals.get(name, _MISSING)
            closures[name] = (
                self._fn_closures.get(name, _EMPTY_CELL) if value is _MISSING else value
            )
        helper = self.__class__(
            self.b, fn_globals=self._fn_globals, fn_closures=closures, caller=self
        )
        yielded: list[Any] = []
        helper.locals[".0"] = call_args[0]
        helper.locals[".yielded"] = yielded
        helper.run(code)
        self._refresh_lines()
        self.stack.append(_Unrolled(iter(yielded)))

    def _push_range(self, call_args: list[StackVal]) -> None:
        if not 1 <= len(call_args) <= 3:
            raise CompilationError(
                f"range() takes 1 to 3 arguments, got {len(call_args)}",
                code="wrong_argument",
            )
        if not all(isinstance(a, int) for a in call_args):
            from ._emit_errors import range_error

            raise range_error(call_args)
        try:
            self.stack.append(range(*call_args))
        except ValueError as exc:
            raise CompilationError(str(exc), code="wrong_argument") from exc

    def _inline_helper(
        self,
        fn: types.FunctionType,
        call_args: list[StackVal],
        kwargs: dict[str, StackVal] | None,
    ) -> None:
        """Emit a pulse-aware Python helper into the current IR builder."""
        code = fn.__code__
        if self._depth >= MAX_INLINE_DEPTH:
            raise CompilationError(
                f"{fn.__name__}: kernel calls nest more than {MAX_INLINE_DEPTH} deep; "
                "a recursive kernel must stop at a condition known at compile time",
                code="limit_exceeded",
            )
        general = code.co_kwonlyargcount or code.co_flags & _VARIADIC
        if general:
            # *args, **kwargs and keyword-only parameters bind as Python binds them.
            bound = _bind_arguments(fn, call_args, kwargs or {})
        else:
            params = list(code.co_varnames[: code.co_argcount])
            if kwargs or fn.__defaults__:
                call_args = _bind_positional(fn, call_args, kwargs or {})
            if len(call_args) != len(params):
                raise CompilationError(
                    f"{fn.__name__}: expected {len(params)} args, got {len(call_args)}",
                    code="wrong_argument",
                )

        helper = self.__class__(
            self.b,
            fn_globals=fn.__globals__,
            fn_closures=_extract_closures(fn) if fn.__closure__ else {},
            caller=self,
        )
        if general:
            helper.locals.update(bound)
        else:
            helper.locals.update(zip(params, call_args, strict=False))
        result = helper.run(code)
        if self._lines.shared:
            self._refresh_lines()

        # Propagate rebindings of directly passed linear values back to the
        # caller, then leave the helper's return value on the caller stack.
        if general:
            # Lines and tones may arrive packed in *args or **kwargs; their
            # channels know their current values.
            for original in (*call_args, *(kwargs or {}).values()):
                if original.__class__ is not IRValue:
                    continue
                updated = _Lines.current(original)
                if updated is None or updated is original:
                    continue
                for caller_name, caller_value in self.locals.items():
                    if caller_value is original:
                        self.locals[caller_name] = updated
        else:
            for name, original in zip(params, call_args, strict=False):
                updated = helper.locals.get(name)
                if not isinstance(original, IRValue) or updated is original:
                    continue
                for caller_name, caller_value in self.locals.items():
                    if caller_value is original:
                        self.locals[caller_name] = updated

        self.stack.append(result)

    def _acquired_readout(
        self,
        callable_val: Any,
        call_args: list[StackVal],
        kwargs: dict[str, StackVal],
        acquire: str,
    ) -> None:
        """``readout(..., acquire=acquire)`` for a mode other than a level:
        the call without it, whose readout takes the mode."""
        self._acquire = acquire
        try:
            self._dispatch_call(callable_val, call_args, kwargs or None)
        finally:
            del self._acquire

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
        elif fname in _LINE_OPS:
            self.stack.append(results[0])
        elif len(results) == 1:
            self.stack.append(results[0])
        elif len(results) > 1:
            self.stack.append(results)
        else:
            self.stack.append(None)

    def _value_attribute(self, value: IRValue, name: str) -> Any:
        """``value.name`` for an IR value: a line's ``tones``; ``line.tone``
        the loop reads in place."""
        chan = value.chan
        if value.vtype in _LINE_TYPES and chan is not None:
            if name == "tones":
                return _Tones(self, chan.held)
            if name == "tone":
                return chan.held.cells[1].value
            if name == "get_calibration":
                from ._line_calibration import LineCalibrations

                return LineCalibrations(self, chan.held).get_calibration
        if value.vtype == "qref":
            from ._emit_errors import qudit_attribute_error

            raise qudit_attribute_error(self.locals, value, name)
        word = _TYPE_WORDS.get(value.vtype, value.vtype)
        article = "an" if word[0] in "aeiou" else "a"
        hint = "its attributes are" if value.vtype in _LINE_TYPES else "only a line has attributes,"
        close = None
        if value.vtype in _LINE_TYPES:
            from .._errors import _closest

            close = _closest(name, ("tone", "tones", "get_calibration"))
        raise CompilationError(
            f"{article} {word} has no attribute {name!r}",
            code="unknown_attribute",
            fix=f"{hint} tone, tones and get_calibration",
            did_you_mean=close,
            misspelled=None if close is None else name,
        )

    # ── Pulse IR emission ────────────────────────────────────────────

    def _emit_pulse_call(
        self,
        fname: str,
        call_args: list[StackVal],
    ) -> tuple[StackVal, ...]:
        entry = _EMIT_TABLE.get(fname)
        if entry is None:
            if fname == "round_duration":
                from ._durations import traced_round_duration

                return (traced_round_duration(self.b, call_args, None),)
            raise CompilationError(f"unknown pulse op: {fname}", code="unknown_name")

        value_count, attr_names, rtypes, operand_types = entry
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

        if value_count == -1:
            call_args = _sync_lines(fname, call_args)
            operands = tuple(call_args)
            attrs: dict[str, Any] = {}
            out_types = tuple(v.vtype for v in operands) if rtypes is None else rtypes
        else:
            optional: tuple[str, StackVal] | None = None
            if len(call_args) != value_count + len(attr_names):
                if value_count == 3:
                    # drive(line, waveform) and readout(line, waveform): the
                    # line's own tone; the operand checks name a non-line.
                    if len(call_args) != 2:
                        from ._emit_errors import play_arity_error

                        raise play_arity_error(fname, len(call_args))
                    line, waveform = call_args
                    try:
                        line_tone = line.chan.held.cells[1].value
                    except (AttributeError, IndexError):
                        line_tone = None
                    call_args = [line, waveform, line_tone]
                else:
                    last = _OPTIONAL_LAST.get(fname)
                    if last is None or len(call_args) != value_count + len(attr_names) + 1:
                        raise CompilationError(
                            f"{fname}: expected {value_count + len(attr_names)} args, "
                            f"got {len(call_args)}",
                            code="wrong_argument",
                        )
                    optional = (last, call_args[-1])
                    call_args = call_args[:-1]
            # Operands must have their op's types. A line or tone read from a
            # container the tracer does not rebind (a tuple in a list, a dict)
            # may be an old value of its channel; the op consumes the channel's
            # current value. Unrolled by arity: this runs for every op.
            if value_count == 3:
                a = call_args[0]
                b = call_args[1]
                c = call_args[2]
                try:
                    bad = (
                        a.vtype not in operand_types[0]
                        or b.vtype not in operand_types[1]
                        or c.vtype not in operand_types[2]
                    )
                except AttributeError:
                    bad = True
                if bad:
                    from ._line_calibration import play_calibrated

                    return play_calibrated(self, fname, call_args)
                ca = a.chan
                cc = c.chan
                if (ca is not None and ca.value is not a) or (cc is not None and cc.value is not c):
                    call_args[0] = _current(a)
                    call_args[2] = _current(c)
            elif value_count == 1:
                a = call_args[0]
                try:
                    bad = a.vtype not in operand_types[0]
                except AttributeError:
                    bad = True
                if bad:
                    from ._emit_errors import operand_error

                    raise operand_error(fname, call_args)
                ca = a.chan
                if ca is not None and ca.value is not a:
                    call_args[0] = ca.value
            elif value_count:
                for i in range(value_count):
                    v = call_args[i]
                    if v.__class__ is not IRValue or v.vtype not in operand_types[i]:
                        if v.__class__ is not _Choice or v.vtype not in operand_types[i]:
                            from ._emit_errors import operand_error

                            raise operand_error(fname, call_args)
                    call_args[i] = _current(v)
            if attr_names:
                operands = tuple(call_args[:value_count])
                attrs = dict(zip(attr_names, call_args[value_count:], strict=False))
            else:
                operands = tuple(call_args)
                attrs = {}
            if optional is not None:
                attrs[optional[0]] = optional[1]
            out_types = (operands[0].vtype,) if (rtypes is None and operands) else (rtypes or ())
            if fname == "readout":
                # Set only while readout(..., acquire=) of a point is emitted.
                acquire = getattr(self, "_acquire", None)
                if acquire is not None:
                    attrs["acquire"] = acquire
                    out_types = ("readout_line", "tone", "iq_point")

        op_name = _OP_NAMES[fname]
        results: tuple[IRValue, ...]
        if fname in _MAKES_WAVEFORM:
            if fname in _WAVEFORMS:
                duration = attrs.get("duration")
                if fname == "custom":
                    attrs, duration = self._custom_samples(duration, attrs)
                    op_name = _OP_NAMES["custom_samples"]
                elif duration.__class__ is not int:
                    from ._durations import traced_duration

                    duration = traced_duration(self.b, fname, attrs)
            else:
                duration = self._waveform_duration(fname, operands, attrs)
            try:
                results = self.b.emit(op_name, operands, out_types, attrs)
            except _ChosenOperandError:
                results = self._emit_chosen(fname, op_name, operands, out_types, attrs)
            else:
                # A rolled-back loop body can leave an entry for a reused vid.
                if duration is None:
                    self._durations.pop(results[0].vid, None)
                else:
                    self._durations[results[0].vid] = duration
        else:
            try:
                results = self.b.emit(op_name, operands, out_types, attrs)
            except _ChosenOperandError:
                results = self._emit_chosen(fname, op_name, operands, out_types, attrs)

        if line_op:
            self._lines.add((call_args[0].vid, fname), results)
        elif fname == "drive" and len(results) == 2:
            # drive(line, waveform, tone) -> (line, tone): the common case.
            self._rebind_pair(call_args[0], results[0], call_args[2], results[1])
        else:
            self._rebind_linear(fname, call_args, results)
        return results

    def _custom_samples(self, duration: Any, attrs: dict[str, Any]) -> tuple[dict[str, Any], int]:
        if duration.__class__ is int and 1 <= duration <= MAX_CYCLES and self.b.sample_rate is None:
            samples = _sample_envelope(duration, attrs.get("name"))
            return {"samples": samples}, len(samples)
        from ._durations import custom_samples

        return custom_samples(self.b, duration, attrs)

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
            known = self._durations
            left = known.get(operands[0].vid)
            if len(operands) == 2:
                right = known.get(operands[1].vid)
                if left is not None and right is not None and left != right:
                    raise CompilationError(
                        f"{fname}() operands have durations {left} and {right}; "
                        "waveform arithmetic needs equal durations",
                        code="invalid_waveform",
                    )
                return left if left is not None else right
            return left
        if fname == "custom_samples":
            if "sample_rate_ghz" in attrs or self.b.sample_rate is not None:
                from ._durations import custom_samples_duration

                return custom_samples_duration(self.b, attrs)
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
        names: dict[int, str | None] | None = None
        if fname == "sync" and len(call_args) == 2 and call_args[0] is not call_args[1]:
            # Two distinct lines: their names are found in one pass.
            first, second = call_args
            if first.chan is not None and second.chan is not None and len(results) == 2:
                # Both are claimable, so each claims its own result.
                self._rebind_pair(first, results[0], second, results[1])
                return
            found = self._local_names(first, second)
            names = {id(first): found[0], id(second): found[1]}
        # The operands each result may replace, by type, the first in operand
        # order last: a sync over n lines rebinds in O(n), not O(n^2).
        unclaimed: dict[str, list[tuple[str | None, IRValue, Any, Any]]] = {}
        for a in reversed(call_args):
            if a.__class__ is IRValue and a.vtype in LINEAR_TYPES:
                slot = slots.get(id(a))
                if slot is not None and not _slot_holds(slot, a):
                    slot = None
                name = self._find_local_name(a) if names is None else names[id(a)]
                if name is not None or slot is not None or a.chan is not None:
                    if a.vtype in unclaimed:
                        unclaimed[a.vtype].append((name, a, slot, a.chan))
                    else:
                        unclaimed[a.vtype] = [(name, a, slot, a.chan)]

        for res in results:
            if res.vtype not in unclaimed or not unclaimed[res.vtype]:
                continue
            name, op_val, slot, chan = unclaimed[res.vtype].pop()
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

    def _restore_container(self, obj: Any, copy: Any) -> None:
        """Undo a traced arm or loop body's changes to a container.

        Restored lists hold values whose slots the undone code consumed, so
        their slots are tracked again.
        """
        if isinstance(obj, _Held):
            for i, value in enumerate(copy):
                obj[i] = value
        elif isinstance(obj, list):
            obj[:] = copy
            self._track_slots(obj, 0)
        elif isinstance(obj, (dict, set)):
            obj.clear()
            obj.update(copy)
        else:
            obj.clear()
            obj.extend(copy)

    def _track_slots(self, items: list[Any], start: int) -> None:
        """Remember the list slots from *start* on that hold lines or tones."""
        slots = self._slots
        for i in range(start, len(items)):
            v = items[i]
            if v.__class__ is IRValue and v.vtype in LINEAR_TYPES:
                slots[id(v)] = (items, i, v)

    def _rebind_one(self, old: IRValue, new: IRValue) -> None:
        """Rebind *old* to *new*, of its type; ``_emit_pulse_call`` checked
        *old* and made it its channel's current value."""
        chan = old.chan
        if chan is not None:
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
        self._rebind_slot(old, new)

    def _rebind_pair(self, old: IRValue, new: IRValue, old2: IRValue, new2: IRValue) -> None:
        """``_rebind_one(old, new)``, then ``_rebind_one(old2, new2)``: a
        drive's line and tone. Unless a channel is shared, the names holding
        them are found in one pass over the locals."""
        chan = old.chan
        chan2 = old2.chan
        if (chan is not None and chan.shared) or (chan2 is not None and chan2.shared):
            self._rebind_one(old, new)
            self._rebind_one(old2, new2)
            return
        name, name2 = (_identity_keys or _load_identity_keys())(self.locals, old, old2)
        if chan is not None:
            chan.value = new
            new.chan = chan
        if name is not None:
            self.locals[name] = new
        self._rebind_slot(old, new)
        if chan2 is not None:
            chan2.value = new2
            new2.chan = chan2
        if name2 is not None:
            self.locals[name2] = new2
        self._rebind_slot(old2, new2)

    def _local_names(self, val: IRValue, val2: IRValue) -> tuple[str | None, str | None]:
        """``_find_local_name`` of two distinct values, in one pass."""
        return (_identity_keys or _load_identity_keys())(self.locals, val, val2)

    def _rebind_slot(self, old: IRValue, new: IRValue) -> None:
        """Put *new* in the list slot *old* was read from, if it still holds it."""
        slots = self._slots
        slot = slots.get(id(old))
        if slot is not None and slot[2] is old:
            try:
                holds = slot[0][slot[1]] is old
            except IndexError:  # the list shrank since the value was read
                holds = False
            if not holds:
                return
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
        for i, (new, old) in enumerate(zip(obj, copy, strict=False)):
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


def _stale_names(
    delta: dict[str, IRValue], before: dict[str, StackVal], count: int
) -> tuple[list[tuple[str, _Chan]], list[str]]:
    """The names a rolled loop body left on an old value of a channel, and
    the names it moved from one channel to another.

    A name on an old value was read from a tuple or dict the tracer does
    not rebind; each use of it reads its channel's current value. It is not
    carried when it entered on the same channel or the body bound it first,
    so every iteration traces alike; after the loop it names the channel's
    value, given as (name, channel). A name that entered on another channel
    is returned apart: the loop rolls only if the body does not read it
    before rebinding it.
    """
    stale: list[tuple[str, _Chan]] = []
    crossing: list[str] = []
    for n, v in delta.items():
        chan = v.chan
        if chan is None:
            continue
        old = before.get(n, v)
        if old is v or (old.__class__ is IRValue and old.chan is chan):
            if chan.value is not v:
                stale.append((n, chan))
        elif old.__class__ is IRValue and old.chan is not None:
            if count:
                crossing.append(n)
            else:
                stale.append((n, old.chan))
    return stale, crossing


def _qudit_names(
    delta: dict[str, IRValue], before: dict[str, StackVal], count: int
) -> tuple[list[str], list[str]]:
    """The names a rolled loop body left on a qudit, and the names it moved
    from one qudit to another.

    A qudit is allocated before any loop and is not carried: every
    iteration leaves the same qudit on a name, which names it after the
    loop. A name that entered on another qudit is returned apart: the loop
    rolls only if the body does not read it before rebinding it.
    """
    fixed: list[str] = []
    moved: list[str] = []
    for n, v in delta.items():
        if v.vtype == "qref":
            (moved if count and n in before else fixed).append(n)
    return fixed, moved


def _channel_values(held: dict[tuple[int, str], _Held]) -> list[tuple[_Chan, IRValue]]:
    """Each channel's line and tone now, as (cell, value)."""
    return [(cell, cell.value) for h in held.values() for cell in h.cells]


def _restore_channels(
    held: dict[tuple[int, str], _Held],
    held_count: int,
    entering: Any,
) -> None:
    """Drop the channels acquired after the first *held_count*, and give the
    others back their *entering* values."""
    if len(held) != held_count:
        for key in list(held)[held_count:]:
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


def _contents(obj: Any) -> Any:
    """A copy of what the mutable container *obj* holds, in its order."""
    if isinstance(obj, dict):
        return dict(obj)
    if isinstance(obj, set):
        return set(obj)
    return list(obj)


def _container_unchanged(obj: Any, copy: Any) -> bool:
    """Whether the container *obj* holds what its *copy* does: the same
    objects, in the same order but for a set; the numbers a bytearray makes
    on each read compare equal."""
    if len(obj) != len(copy):
        return False
    if isinstance(obj, list):
        return all(a is b for a, b in zip(obj, copy, strict=False))
    if isinstance(obj, set):
        return all(k in copy for k in obj)
    if isinstance(obj, dict):
        return all(k is j and obj[k] is copy[j] for k, j in zip(obj, copy, strict=False))
    return all(
        a is b or (a.__class__ in _NUMBERS and a.__class__ is b.__class__ and a == b)
        for a, b in zip(obj, copy, strict=False)
    )


_SIGNATURES = {}


def _bound(fn: Any, args: list[StackVal], kwargs: dict[str, StackVal]) -> inspect.BoundArguments:
    hit = _SIGNATURES.get(id(fn))
    if hit is None or hit[0] is not fn:
        try:
            hit = (fn, inspect.signature(fn, follow_wrapped=False))
        except (TypeError, ValueError) as e:
            raise CompilationError(
                f"cannot bind keyword arguments for {fn!r}", code="wrong_argument"
            ) from e
        _SIGNATURES[id(fn)] = hit
    try:
        bound = hit[1].bind(*args, **kwargs)
    except TypeError as e:
        raise CompilationError(f"{getattr(fn, '__name__', fn)}: {e}", code="wrong_argument") from e
    bound.apply_defaults()
    return bound


def _bind_positional(fn: Any, args: list[StackVal], kwargs: dict[str, StackVal]) -> list[StackVal]:
    """Bind a keyword call to *fn*'s parameters as a positional list; the
    keyword-only ``tone`` of drive and readout comes last, when given."""
    bound = _bound(fn, args, kwargs)
    if bound.kwargs:
        keywords = dict(bound.kwargs)
        tone = keywords.pop("tone", None)
        if keywords:
            raise CompilationError(
                f"{getattr(fn, '__name__', fn)}: keyword-only parameters",
                code="wrong_argument",
            )
        return list(bound.args) if tone is None else [*bound.args, tone]
    return list(bound.args)


def _bind_arguments(
    fn: types.FunctionType, args: list[StackVal], kwargs: dict[str, StackVal]
) -> dict[str, StackVal]:
    """Bind a call to *fn* as Python does: each parameter name to its value,
    ``*args`` to a tuple and ``**kwargs`` to a dict."""
    return dict(_bound(fn, args, kwargs).arguments)


__all__ = ["PulseIREmitter", "PythonIRBuilder"]
