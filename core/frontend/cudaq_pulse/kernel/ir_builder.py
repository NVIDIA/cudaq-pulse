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

from collections import namedtuple
from typing import Any, NoReturn

Op = namedtuple("Op", ["kind", "operands", "results", "attrs"])


class CompilationError(Exception):
    pass


class SymbolicValueError(CompilationError):
    """Python code needed the value of a runtime ``float`` argument.

    ``names`` lists the arguments involved. A kernel call specializes on
    the value of such an argument instead of keeping it symbolic.
    """

    def __init__(self, message: str, names: tuple[str, ...]):
        super().__init__(message)
        self.names = names


def parameter_names(value: Any) -> tuple[str, ...]:
    """Names of the kernel parameters a symbolic value depends on."""
    if isinstance(value, Parameter):
        return (value.name,)
    if isinstance(value, ParameterExpression):
        return parameter_names(value.lhs) + parameter_names(value.rhs)
    if isinstance(value, UndecidableComparison):
        return parameter_names(value.lhs) + parameter_names(value.rhs)
    return ()


class UndecidableComparison:
    """A comparison whose outcome is only known when the program runs.

    Comparing a symbolic parameter or an IR value builds one of these
    instead of falling back to object identity; using it as a condition
    raises, so a kernel never silently takes one branch.
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
        raise SymbolicValueError(
            f"cannot decide {self.lhs!r} {self.op} {self.rhs!r} at trace time: a float "
            "argument has no value while the kernel compiles",
            parameter_names(self),
        )

    def __repr__(self) -> str:
        return f"({self.lhs!r} {self.op} {self.rhs!r})"


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
            raise CompilationError(f"unsupported symbolic cast to {dtype}")
        return ParameterExpression("cast", self, dtype=dtype)

    def _not_concrete(self, operation: str) -> NoReturn:
        raise SymbolicValueError(
            f"Cannot evaluate symbolic parameter expression with {operation}. "
            "Use it in a supported pulse numeric argument.",
            parameter_names(self),
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
            raise ValueError(f"unsupported parameter dtype {dtype!r}")
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


def is_symbolic(value: Any) -> bool:
    """Return whether *value* is a symbolic parameter or expression."""
    return isinstance(value, _SymbolicNumber)


class IRValue(_UndecidableComparisons):
    __slots__ = ("vid", "vtype", "name")

    def __init__(self, vid: int, vtype: str, name: str = ""):
        self.vid = vid
        self.vtype = vtype
        self.name = name

    def __repr__(self) -> str:
        return f"{self.name or f'%v{self.vid}'}:{self.vtype}"


LINEAR_TYPES = frozenset({"drive_line", "readout_line", "tone"})

# The vid of a loop-carried value that has no value entering the loop.
UNDEFINED_VID = -1

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
    "qudit_ref": (0, (), ("qref",)),
    "qvec_ref": (0, ("size",), ("qref",)),
}


_UNCOUNTED_OPS = frozenset({"pulse.qudit_arg", "pulse.qudit_alloc", "scf.yield"})


class PythonIRBuilder:
    """Lightweight in-memory IR builder (swap for real MLIR bindings)."""

    # Accepts scf.for regions, so the tracer may emit a loop once.
    supports_regions = True

    def __init__(self, name: str = "main"):
        self.name = name
        self.ops: list[Op] = []
        self._next_id = 0
        # What the traced kernel returned.
        self.result: Any = None

    def _mk(self, vtype: str, name: str = "") -> IRValue:
        v = IRValue(self._next_id, vtype, name)
        self._next_id += 1
        return v

    def emit(
        self,
        kind: str,
        operands: tuple[IRValue, ...] = (),
        result_types: tuple[str, ...] = (),
        attrs: dict[str, Any] | None = None,
    ) -> tuple[IRValue, ...]:
        results = tuple(self._mk(rt) for rt in result_types)
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

    def pretty(self) -> str:
        lines = [f"func.func @{self.name}() {{"]
        for op in self.ops:
            res = ", ".join(repr(r) for r in op.results)
            ops_s = ", ".join(repr(o) for o in op.operands)
            att = ", ".join(f"{k}={v!r}" for k, v in op.attrs.items())
            parts = [s for s in (ops_s, att) if s]
            lines.append(f"  {res} = {op.kind}({', '.join(parts)})")
        lines.append("}")
        return "\n".join(lines)
