# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Durations while tracing: cycles, or times with units converted to cycles.

A duration (a waveform's length or flat top, a wait and its bounds, a
``round_duration`` length) is an int number of cycles or a ``pulse.Time``; a
plain float there is ambiguous and an error. A waveform's shape times
(``sigma``, DRAG ``beta``) are cycles, fractional ones too, or times.

A time converts exactly to cycles of the program clock; a duration that is
not a whole number of them is an error. In an expression known only when
the program runs, ``f.result() * pulse.ns``, each time becomes its number of
cycles: one multiply by a constant the compile fixes.
"""

from __future__ import annotations

import numbers
import operator
from typing import Any

from .._units import Time, UnitError, in_cycles
from .ir_builder import (
    MAX_CYCLES,
    CompilationError,
    ParameterExpression,
    arithmetic_text,
    is_symbolic,
)

# The times of each op: (attribute, whole cycles, the call with it as {}).
TIMES: dict[str, tuple[tuple[str, bool, str], ...]] = {
    "gaussian": (("duration", True, "gaussian({}, ...)"), ("sigma", False, "")),
    "square": (("duration", True, "square({}, ...)"),),
    "drag": (
        ("duration", True, "drag({}, ...)"),
        ("sigma", False, ""),
        ("beta", False, ""),
    ),
    "cosine": (("duration", True, "cosine({}, ...)"),),
    "tanh_ramp": (("duration", True, "tanh_ramp({}, ...)"), ("sigma", False, "")),
    "gaussian_square": (
        ("duration", True, "gaussian_square({}, ...)"),
        ("sigma", False, ""),
        ("width", True, "gaussian_square(duration, amplitude, sigma, {})"),
    ),
    "custom": (("duration", True, "custom({}, ...)"),),
    "wait": (
        ("duration", True, "wait(line, {})"),
        ("bounds", True, "wait(line, duration, bounds=(0, {}))"),
    ),
}


# The fewest cycles of each whole-cycle time: a waveform plays for at least
# one cycle; a wait, a flat top and a bound may be 0.
LEAST: dict[tuple[str, str], int] = {
    (fname, name): int(name == "duration" and fname != "wait")
    for fname, spec in TIMES.items()
    for name, whole, _ in spec
    if whole
}


class Cycles(float):
    """The value of a ``float`` parameter the kernel is compiled for: a
    number of cycles, as the parameter is when the program computes with it,
    so a duration takes it, and what is computed from it, like a run-time
    length."""

    __slots__ = ()

    def _keep(self, value: Any) -> Any:
        return Cycles(value) if value.__class__ is float else value

    def _with(self, other: Any) -> bool:
        if other.__class__ is Time:
            raise UnitError(
                f"{float(self)!r} {other!r}: a float parameter is a number of cycles, "
                "and mixes with a time only multiplied by one (tau * pulse.ns)"
            )
        return isinstance(other, (int, float)) and other.__class__ is not bool

    def __add__(self, other: Any) -> Any:
        return self._keep(float.__add__(self, other)) if self._with(other) else NotImplemented

    def __radd__(self, other: Any) -> Any:
        return self._keep(float.__radd__(self, other)) if self._with(other) else NotImplemented

    def __sub__(self, other: Any) -> Any:
        return self._keep(float.__sub__(self, other)) if self._with(other) else NotImplemented

    def __rsub__(self, other: Any) -> Any:
        return self._keep(float.__rsub__(self, other)) if self._with(other) else NotImplemented

    def __mul__(self, other: Any) -> Any:
        if other.__class__ is Time:
            return other * float(self)
        return self._keep(float.__mul__(self, other)) if self._with(other) else NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> Any:
        return self._keep(float.__truediv__(self, other)) if self._with(other) else NotImplemented

    def __rtruediv__(self, other: Any) -> Any:
        return self._keep(float.__rtruediv__(self, other)) if self._with(other) else NotImplemented

    def __floordiv__(self, other: Any) -> Any:
        return self._keep(float.__floordiv__(self, other)) if self._with(other) else NotImplemented

    def __rfloordiv__(self, other: Any) -> Any:
        return self._keep(float.__rfloordiv__(self, other)) if self._with(other) else NotImplemented

    def __mod__(self, other: Any) -> Any:
        return self._keep(float.__mod__(self, other)) if self._with(other) else NotImplemented

    def __neg__(self) -> Cycles:
        return Cycles(-float(self))

    def __pos__(self) -> Cycles:
        return self

    def __abs__(self) -> Cycles:
        return Cycles(abs(float(self)))


# Errors name a value's type; this is the float the user passed.
Cycles.__name__ = Cycles.__qualname__ = "float"


def in_program_cycles(fname: str, attrs: dict[str, Any], clock_ghz: float) -> None:
    """Replace the times in *attrs*, the attributes of a *fname* call, by
    cycles of the *clock_ghz* program clock; reject plain floats in its
    durations."""
    for name, whole, call in TIMES[fname]:
        value = attrs.get(name)
        if value is None:
            continue
        if name == "bounds":
            if isinstance(value, (tuple, list)) and not _whole(value):
                attrs[name] = type(value)(
                    v if v is None else cycles(fname, name, v, True, call, clock_ghz) for v in value
                )
            continue
        n = value
        if n.__class__ is not int:
            n = attrs[name] = cycles(fname, name, value, whole, call, clock_ghz)
            if not whole:
                continue
            if n.__class__ is Cycles:
                # A float argument the kernel is compiled for, as a duration.
                if not LEAST[fname, name] <= n <= MAX_CYCLES:
                    raise _out_of_range(fname, name, value, n)
                continue
            if n.__class__ is bool or not isinstance(n, numbers.Integral):
                continue
            # An integer of another type, such as numpy's: its int value.
            n = attrs[name] = operator.index(n)
        if whole and not LEAST[fname, name] <= n <= MAX_CYCLES:
            raise _out_of_range(fname, name, value, n)


def _out_of_range(fname: str, what: str, value: Any, n: float) -> CompilationError:
    """The error for *value*, a duration of *n* cycles outside LEAST to MAX_CYCLES."""
    shown = f"{n:g}" if n.__class__ is Cycles else f"{n}"
    given = (
        f"{what} {value} is {shown} cycles"
        if value.__class__ is Time
        else f"{what} is {shown} cycles"
    )
    if n > MAX_CYCLES:
        rule = "a duration is at most 2**53 cycles"
    elif LEAST[fname, what]:
        rule = "a waveform lasts at least 1 cycle"
    elif fname == "wait":
        rule = "a wait cannot be negative"
    else:
        rule = "a duration cannot be negative"
    return CompilationError(f"{fname}(): {given}; {rule}")


def _whole(bounds: tuple[Any, ...] | list[Any]) -> bool:
    for v in bounds:
        if v is not None and v.__class__ is not int:
            return False
    return True


def cycles(fname: str, what: str, value: Any, whole: bool, call: str, clock_ghz: float) -> Any:
    """*value*, the *what* of a *fname* call, in program cycles."""
    cls = value.__class__
    if cls is Time:
        exact = in_cycles(value, clock_ghz)
        if not whole:
            return float(exact)
        if exact.denominator != 1:
            raise CompilationError(
                f"{fname}(): {what} {value} is {float(exact):g} cycles of the "
                f"{clock_ghz:g} GHz program clock, and durations are whole cycles; "
                f"round it with pulse.round_duration(line, {_written(value)})"
            )
        return int(exact)
    if whole and isinstance(value, float) and cls is not Cycles:
        raise float_duration(fname, what, value, call)
    if is_symbolic(value) and _has_time(value):
        if unit(value, fname, what) != 1:
            raise CompilationError(
                f"{fname}(): {what} {arithmetic_text(value)} is a number, not a time: "
                "a duration is a number of cycles or a time"
            )
        return _replaced(value, clock_ghz)
    return value


def float_duration(fname: str, what: str, value: float, call: str) -> CompilationError:
    """The error for the plain float *value* in a duration."""
    shown = f"{value:g}"
    whole = str(round(value)) if value == value and abs(value) != float("inf") else shown
    in_cycles_call = call.format(whole)
    in_ns_call = call.format(f"{shown} * pulse.ns")
    width = max(len(in_cycles_call), len(in_ns_call)) + 4
    plural = "" if whole == "1" else "s"
    return CompilationError(
        f"{fname}(): {what} is a float ({value!r}). Durations are an int number of "
        "cycles, or a time with units:\n"
        f"    {in_cycles_call.ljust(width)}# {whole} cycle{plural}\n"
        f"    {in_ns_call.ljust(width)}# {shown} nanoseconds"
    )


def _written(value: Time) -> str:
    text = repr(value)
    number, unit_name = text.split()
    return f"{number} * pulse.{unit_name}"


def _has_time(value: Any) -> bool:
    if value.__class__ is Time:
        return True
    if value.__class__ is ParameterExpression:
        return _has_time(value.lhs) or _has_time(value.rhs)
    return False


def unit(value: Any, fname: str, what: str) -> int:
    """1 if *value* is a time, 0 if a number; raises for arithmetic that
    mixes them."""
    if value.__class__ is Time:
        return 1
    if value.__class__ is not ParameterExpression:
        return 0
    op = value.op
    left = unit(value.lhs, fname, what)
    if op in ("neg", "cast"):
        return left
    right = unit(value.rhs, fname, what)
    if op in ("add", "sub"):
        if left != right:
            raise CompilationError(
                f"{fname}(): {what} {arithmetic_text(value)} mixes a time with a number "
                "of cycles; give both with units (n * pulse.ns) or both in cycles"
            )
        return left
    if op == "mul":
        if left and right:
            raise CompilationError(
                f"{fname}(): {what} {arithmetic_text(value)} is a time times a time"
            )
        return left + right
    if op == "div":
        if right and not left:
            raise CompilationError(
                f"{fname}(): {what} {arithmetic_text(value)} is a number over a time"
            )
        return left - right
    if left or right:
        raise CompilationError(
            f"{fname}(): {what} {arithmetic_text(value)} takes {op} of a time; use + - * / on times"
        )
    return 0


def _replaced(value: Any, clock_ghz: float) -> Any:
    if value.__class__ is Time:
        return float(in_cycles(value, clock_ghz))
    if value.__class__ is ParameterExpression:
        return ParameterExpression(
            value.op,
            _replaced(value.lhs, clock_ghz),
            _replaced(value.rhs, clock_ghz),
            dtype=value.dtype,
        )
    return value
