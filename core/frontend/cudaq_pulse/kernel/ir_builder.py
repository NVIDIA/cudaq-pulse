# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Shared IR types and builder for the pulse kernel compiler.

This module contains the core data structures used by the bytecode
compiler: Op, IRValue, Parameter, PythonIRBuilder, OP_TABLE, LINEAR_TYPES,
and CompilationError.
"""

from __future__ import annotations

import numbers
from collections import namedtuple
from typing import Any, NoReturn

from .._errors import CompilationError as CompilationError
from .._errors import InternalError

Op = namedtuple("Op", ["kind", "operands", "results", "attrs"])


class SymbolicValueError(CompilationError):
    """Python code needed the value of a runtime ``float`` argument.

    ``names`` lists the arguments involved, and ``operation`` what needed
    the value. A kernel call specializes on the value of such an argument
    instead of keeping it symbolic.
    """

    def __init__(self, message: str, names: tuple[str, ...] = (), operation: str = ""):
        super().__init__(message)
        self.names = names
        self.operation = operation


def parameter_names(value: Any) -> tuple[str, ...]:
    """Names of the kernel parameters a symbolic value depends on."""
    if isinstance(value, Parameter):
        return (value.name,)
    if isinstance(value, ParameterExpression):
        return parameter_names(value.lhs) + parameter_names(value.rhs)
    if isinstance(value, UndecidableComparison):
        return parameter_names(value.lhs) + parameter_names(value.rhs)
    return ()


class _UndecidableComparisons:
    """Comparison operators that defer to ``UndecidableComparison``."""

    __slots__ = ()
    __hash__ = object.__hash__

    def __eq__(self, other: object) -> UndecidableComparison:  # type: ignore[override]
        return UndecidableComparison(self, "==", other)

    def __ne__(self, other: object) -> UndecidableComparison:  # type: ignore[override]
        return UndecidableComparison(self, "!=", other)

    def __lt__(self, other: Any) -> UndecidableComparison:
        return UndecidableComparison(self, "<", other)

    def __le__(self, other: Any) -> UndecidableComparison:
        return UndecidableComparison(self, "<=", other)

    def __gt__(self, other: Any) -> UndecidableComparison:
        return UndecidableComparison(self, ">", other)

    def __ge__(self, other: Any) -> UndecidableComparison:
        return UndecidableComparison(self, ">=", other)


class UndecidableComparison(_UndecidableComparisons):
    """A comparison whose outcome is only known when the program runs.

    Comparing a symbolic parameter or an IR value builds one of these
    instead of falling back to object identity, and so does comparing one
    of these; using it as a condition raises, so a kernel never silently
    takes one branch.
    """

    __slots__ = ("lhs", "op", "rhs")

    def __init__(self, lhs: Any, op: str, rhs: Any):
        self.lhs = lhs
        self.op = op
        self.rhs = rhs

    def __bool__(self) -> bool:
        operands = (self.lhs, self.rhs)
        if any(isinstance(v, IRValue) for v in operands):
            raise CompilationError(
                f"cannot compare a runtime IR value at trace time ({self.lhs!r} {self.op} "
                f"{self.rhs!r}); branch on a measurement result directly with "
                "`if result:` or `if not result:`"
            )
        names = parameter_names(self)
        if not names or contains_answer(self):
            raise CompilationError(
                f"cannot decide {self!r} at trace time: it is known only when the program "
                "runs; use it as an if condition, alone or with and, or, not"
            )
        raise SymbolicValueError(
            f"cannot decide {self.lhs!r} {self.op} {self.rhs!r} at trace time: a float "
            "argument has no value while the kernel compiles",
            names,
            f"the comparison {self.op}",
        )

    # On bools, ``^`` is ``!=``.
    def __xor__(self, other: Any) -> UndecidableComparison:
        return UndecidableComparison(self, "!=", other)

    __rxor__ = __xor__

    def _both(self, op: str, other: Any) -> NoReturn:
        word = "and" if op == "&" else "or"
        raise CompilationError(
            f"cannot trace {self!r} {op} {other!r}: it is known only when the program runs; "
            f"combine conditions with `{word}` in an if condition instead"
        )

    def __and__(self, other: Any) -> NoReturn:
        self._both("&", other)

    def __rand__(self, other: Any) -> NoReturn:
        self._both("&", other)

    def __or__(self, other: Any) -> NoReturn:
        self._both("|", other)

    def __ror__(self, other: Any) -> NoReturn:
        self._both("|", other)

    def __repr__(self) -> str:
        return f"({self.lhs!r} {self.op} {self.rhs!r})"


class _SymbolicNumber(_UndecidableComparisons):
    """Arithmetic shared by symbolic kernel parameters and expressions."""

    __slots__ = ()

    def _binary(self, op: str, other: Any) -> ParameterExpression:
        return ParameterExpression(op, self, other)

    def _rbinary(self, op: str, other: Any) -> ParameterExpression:
        return ParameterExpression(op, other, self)

    def __add__(self, other: Any) -> ParameterExpression:
        return self._binary("add", other)

    def __radd__(self, other: Any) -> ParameterExpression:
        return self._rbinary("add", other)

    def __sub__(self, other: Any) -> ParameterExpression:
        return self._binary("sub", other)

    def __rsub__(self, other: Any) -> ParameterExpression:
        return self._rbinary("sub", other)

    def __mul__(self, other: Any) -> ParameterExpression:
        return self._binary("mul", other)

    def __rmul__(self, other: Any) -> ParameterExpression:
        return self._rbinary("mul", other)

    def __truediv__(self, other: Any) -> ParameterExpression:
        return self._binary("div", other)

    def __rtruediv__(self, other: Any) -> ParameterExpression:
        return self._rbinary("div", other)

    def __floordiv__(self, other: Any) -> ParameterExpression:
        return self._binary("floordiv", other)

    def __rfloordiv__(self, other: Any) -> ParameterExpression:
        return self._rbinary("floordiv", other)

    def __mod__(self, other: Any) -> ParameterExpression:
        return self._binary("mod", other)

    def __rmod__(self, other: Any) -> ParameterExpression:
        return self._rbinary("mod", other)

    def __pow__(self, other: Any) -> ParameterExpression:
        return self._binary("pow", other)

    def __rpow__(self, other: Any) -> ParameterExpression:
        return self._rbinary("pow", other)

    def __neg__(self) -> ParameterExpression:
        return ParameterExpression("neg", self)

    def cast(self, dtype: str) -> ParameterExpression:
        if dtype not in ("i64", "f64"):
            raise InternalError(f"unsupported symbolic cast to {dtype}")
        return ParameterExpression("cast", self, dtype=dtype)

    def _not_concrete(self, operation: str) -> NoReturn:
        names = parameter_names(self)
        quoted = ", ".join(map(repr, dict.fromkeys(names)))
        arguments = "argument" if len(set(names)) == 1 else "arguments"
        raise SymbolicValueError(
            f"{operation} needs the value of float {arguments} {quoted}, which is only known "
            "when the kernel runs; use it only as a pulse op argument (amplitude, phase, "
            "frequency, wait length), or annotate it as int to compile one kernel per value",
            names,
            operation,
        )

    def __bool__(self) -> bool:
        self._not_concrete("bool()")

    def __float__(self) -> float:
        self._not_concrete("float()")

    def __int__(self) -> int:
        self._not_concrete("int()")

    def __complex__(self) -> complex:
        self._not_concrete("complex()")

    def __index__(self) -> int:
        self._not_concrete("an index")


class Parameter(_SymbolicNumber):
    """Sentinel for a symbolic kernel parameter (compile-once, evaluate-many).

    Instances track an index and type so the packed IR builder can emit
    a PARAM opcode instead of a concrete value.
    """

    __slots__ = ("name", "index", "dtype")

    def __init__(self, name: str, index: int, dtype: str = "unknown"):
        if dtype not in ("unknown", "i64", "f64"):
            raise InternalError(f"unsupported parameter dtype {dtype!r}")
        self.name = name
        self.index = index
        self.dtype = dtype

    def __repr__(self) -> str:
        return f"Parameter({self.name!r}, idx={self.index}, {self.dtype})"


class ParameterExpression(_SymbolicNumber):
    """A symbolic arithmetic expression rooted in kernel parameters."""

    __slots__ = ("op", "lhs", "rhs", "dtype")

    def __init__(self, op: str, lhs: Any, rhs: Any = None, *, dtype: str = "unknown"):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs
        self.dtype = dtype

    def __repr__(self) -> str:
        if self.op in ("neg", "cast"):
            return f"ParameterExpression({self.op}, {self.lhs!r}, {self.dtype})"
        return f"ParameterExpression({self.op}, {self.lhs!r}, {self.rhs!r})"


class AnswerValue(_SymbolicNumber):
    """A value a classical function returns, known only when the program runs.

    It takes part in arithmetic like a float kernel parameter, and can set
    amplitudes, phases and frequencies or decide a branch, but not a
    duration. ``dtype`` is ``"i1"``, ``"i64"`` or ``"f64"``; ``scope`` is the
    regions open where it was taken, outside which it is not defined.
    """

    __slots__ = ("vid", "dtype", "scope", "label")

    def __init__(self, vid: int, dtype: str, scope: tuple[int, ...], label: str = "") -> None:
        self.vid = vid
        self.dtype = dtype
        self.scope = scope
        self.label = label

    @property
    def vtype(self) -> str:
        return self.dtype

    def __repr__(self) -> str:
        return f"AnswerValue({self.label or self.vid}, {self.dtype})"


class Selection(_SymbolicNumber):
    """A value a condition selects while the program runs: ``then`` where
    ``cond`` (a measurement, or a boolean IR value) holds, ``other`` where it
    does not. ``then`` and ``other`` are numbers, symbolic numbers, waveforms
    or selections; the builder emits one select of them."""

    __slots__ = ("cond", "then", "other")
    __hash__ = object.__hash__

    def __init__(self, cond: IRValue, then: Any, other: Any) -> None:
        self.cond = cond
        self.then = then
        self.other = other


def contains_answer(value: Any) -> bool:
    """Whether a symbolic value is computed from an answer of a classical call."""
    if isinstance(value, AnswerValue):
        return True
    if isinstance(value, (ParameterExpression, UndecidableComparison)):
        return contains_answer(value.lhs) or contains_answer(value.rhs)
    return False


def is_symbolic(value: Any) -> bool:
    """Return whether *value* is a symbolic parameter or expression."""
    return isinstance(value, _SymbolicNumber)


_ARITHMETIC = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


def is_arithmetic(value: Any) -> bool:
    """Whether *value* is ``+ - * /`` and negation of parameters and numbers:
    what ``arithmetic_source`` can compute."""
    cls = value.__class__
    if cls is Parameter or cls is float or cls is int:
        return True
    if cls is ParameterExpression:
        op = value.op
        if op in _ARITHMETIC:
            return is_arithmetic(value.lhs) and is_arithmetic(value.rhs)
        if op == "neg" or (op == "cast" and value.dtype == "f64"):
            return is_arithmetic(value.lhs)
        return False
    return isinstance(value, numbers.Real)


def arithmetic_source(value: Any) -> str | None:
    """Python source that computes *value* from ``a``, the float arguments
    in parameter order, with the f64 operations the module computes it with;
    None unless it is ``+ - * /`` and negation of parameters and numbers."""
    if isinstance(value, Parameter):
        return f"a[{value.index}]"
    if isinstance(value, ParameterExpression):
        if value.op == "neg":
            inner = arithmetic_source(value.lhs)
            return None if inner is None else f"(-{inner})"
        if value.op == "cast":
            return arithmetic_source(value.lhs) if value.dtype == "f64" else None
        symbol = _ARITHMETIC.get(value.op)
        if symbol is None:
            return None
        lhs, rhs = arithmetic_source(value.lhs), arithmetic_source(value.rhs)
        if lhs is None or rhs is None:
            return None
        return f"({lhs} {symbol} {rhs})"
    if isinstance(value, numbers.Real):
        return repr(float(value))
    return None


def without_answers(value: Any) -> Any:
    """*value* with every answer of a classical call taken as 0: the part of
    a wait length that its float parameters give."""
    if isinstance(value, AnswerValue):
        return 0.0
    if isinstance(value, ParameterExpression):
        return ParameterExpression(
            value.op, without_answers(value.lhs), without_answers(value.rhs), dtype=value.dtype
        )
    return value


def arithmetic_text(value: Any) -> str:
    """*value* as the user wrote it, roughly: ``tau / 2``, ``2.0 * tau + 40.0``."""
    if isinstance(value, Parameter):
        return value.name
    if isinstance(value, AnswerValue):
        return value.label or "answer"
    if isinstance(value, ParameterExpression):
        if value.op == "neg":
            return f"-{_operand_text(value.lhs, 3)}"
        if value.op == "cast":
            return arithmetic_text(value.lhs)
        symbol = _ARITHMETIC.get(value.op, value.op)
        level = _PRECEDENCE.get(value.op, 3)
        lhs = _operand_text(value.lhs, level)
        rhs = _operand_text(value.rhs, level + (value.op in ("sub", "div")))
        return f"{lhs} {symbol} {rhs}"
    return repr(value)


_PRECEDENCE = {"add": 1, "sub": 1, "mul": 2, "div": 2}


def _operand_text(value: Any, level: int) -> str:
    text = arithmetic_text(value)
    if isinstance(value, ParameterExpression) and _PRECEDENCE.get(value.op, 3) < level:
        return f"({text})"
    return text


class IRValue(_UndecidableComparisons):
    __slots__ = ("vid", "vtype", "name", "chan")

    def __init__(self, vid: int, vtype: str, name: str = ""):
        self.vid = vid
        self.vtype = vtype
        self.name = name
        # The qudit channel cell this line or tone value is, or was, a
        # version of; set by the tracer.
        self.chan: Any = None

    def __bool__(self) -> bool:
        raise CompilationError(
            f"cannot decide {self!r} at trace time: it is known only when the program runs; "
            "use a measurement result directly as an if condition (`if result:` or "
            "`if not result:`)"
        )

    def __repr__(self) -> str:
        return f"{self.name or f'%v{self.vid}'}:{self.vtype}"


LINEAR_TYPES = frozenset({"drive_line", "readout_line", "tone"})

# The vid of a loop-carried value that has no value entering the loop.
UNDEFINED_VID = -1

# The longest duration, in cycles: a time a float holds to the cycle.
MAX_CYCLES = 2**53

# (n_value_args, attr_names, result_types)
#   n_value_args: leading args that are IR values; -1 = variadic all-values
#   result_types: None = mirror operand types
OP_TABLE: dict[str, tuple[int, tuple[str, ...], tuple[str, ...] | None]] = {
    "get_drive_line": (1, (), ("drive_line", "tone")),
    "get_readout_line": (1, (), ("readout_line", "tone")),
    "gaussian": (0, ("duration", "amplitude", "sigma"), ("waveform",)),
    "square": (0, ("duration", "amplitude"), ("waveform",)),
    "drag": (0, ("duration", "amplitude", "sigma", "beta"), ("waveform",)),
    "cosine": (0, ("duration", "amplitude"), ("waveform",)),
    "tanh_ramp": (0, ("duration", "amplitude", "sigma"), ("waveform",)),
    "gaussian_square": (0, ("duration", "amplitude", "sigma", "width"), ("waveform",)),
    "custom": (0, ("duration", "name"), ("waveform",)),
    "custom_samples": (0, ("samples",), ("waveform",)),
    "drive": (3, (), ("drive_line", "tone")),
    "readout": (3, (), ("readout_line", "tone", "measurement")),
    "wait": (1, ("duration",), None),
    "sync": (-1, (), None),
    "shift_phase": (1, ("phase_rad",), ("tone",)),
    "set_phase": (1, ("phase_rad",), ("tone",)),
    "shift_frequency": (1, ("freq_hz",), ("tone",)),
    "set_frequency": (1, ("freq_hz",), ("tone",)),
    "wf_add": (2, (), ("waveform",)),
    "wf_sub": (2, (), ("waveform",)),
    "wf_mul": (2, (), ("waveform",)),
    "wf_scale": (1, ("scale",), ("waveform",)),
    "wf_neg": (1, (), ("waveform",)),
}


# The ops with times (pulse.custom is sampled before it is emitted).
_TIMED_OPS = frozenset(
    {"gaussian", "square", "drag", "cosine", "tanh_ramp", "gaussian_square", "wait"}
)
_UNCOUNTED_OPS = frozenset({"pulse.qudit_arg", "pulse.qudit_alloc", "scf.yield"})


class PythonIRBuilder:
    """An in-memory IR builder that keeps each op as Python data, so tests can
    inspect what a kernel traces to; compilation uses ``PackedIRBuilder``."""

    # Accepts scf.for regions, so the tracer may emit a loop once.
    supports_regions = True
    # The samples every line plays per program cycle; see PackedIRBuilder.
    sample_rate: tuple[int, int] | None = None
    # Each line's sample rate, when they differ; see PackedIRBuilder.
    sample_rate_of: Any = None

    def __init__(self, name: str = "main"):
        self.name = name
        self.ops: list[Op] = []
        self._next_id = 0
        # The trace's loop-unroll threshold; None for the default.
        self.unroll_threshold: int | None = None
        # What the traced kernel returned.
        self.result: Any = None

    def emit(
        self,
        kind: str,
        operands: tuple[IRValue, ...] = (),
        result_types: tuple[str, ...] = (),
        attrs: dict[str, Any] | None = None,
    ) -> tuple[IRValue, ...]:
        vid = self._next_id
        count = len(result_types)
        results: tuple[IRValue, ...]
        if count == 1:
            results = (IRValue(vid, result_types[0]),)
        elif count == 2:
            results = (IRValue(vid, result_types[0]), IRValue(vid + 1, result_types[1]))
        else:
            results = tuple([IRValue(vid + i, rt) for i, rt in enumerate(result_types)])
        self._next_id = vid + count
        if attrs and kind.removeprefix("pulse.") in _TIMED_OPS:
            from ._durations import in_program_cycles

            in_program_cycles(kind.removeprefix("pulse."), attrs, getattr(self, "clock_ghz", 2.0))
        self.ops.append(Op(kind, operands, results, attrs or {}))
        return results

    def mark(self) -> int:
        """A position that ``insert`` can later insert at."""
        return len(self.ops)

    def insert(self, index: int, kind: str, operands: tuple[IRValue, ...]) -> None:
        """Insert a result-less op before ``ops[index]``."""
        self.ops.insert(index, Op(kind, operands, (), {}))

    def op_count(self) -> int:
        """Pulse-program ops, not counting qudit binding or region yields."""
        return sum(op.kind not in _UNCOUNTED_OPS for op in self.ops)

    def checkpoint(self) -> tuple[int, int]:
        return len(self.ops), self._next_id

    def rollback(self, checkpoint: tuple[int, int]) -> None:
        n_ops, self._next_id = checkpoint
        del self.ops[n_ops:]
