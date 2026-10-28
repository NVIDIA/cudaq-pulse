# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Wall-clock times: ``20 * pulse.ns``, ``4 * pulse.us``.

A kernel counts time in cycles of the device's program clock; a bare int in
a duration is a number of cycles. A time with units is converted to cycles
when the kernel compiles, for the device it compiles for, so one kernel
written in nanoseconds runs on devices with different clocks.

A time is a float of nanoseconds that also holds its exact value, a
fraction: ``0.1 * pulse.us`` is exactly 100 ns, and converts to whole
cycles exactly or not at all. Arithmetic keeps the unit where it means a
time (time times a number, time plus time, time over a number, minus a
time) and rejects the rest (time times time, time plus a number of cycles).
"""

from __future__ import annotations

import numbers
from fractions import Fraction
from typing import Any

__all__ = ["Time", "ns", "us", "ms", "s"]


def _exact(value: Any) -> Fraction:
    """*value*, a real number, as the fraction its decimal digits give:
    ``0.1`` is 1/10."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, numbers.Rational):
        return Fraction(int(value.numerator), int(value.denominator))
    return Fraction(repr(float(value)))


def _is_number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, (bool, Time))


class UnitError(TypeError):
    """Arithmetic that mixes a time with a number of cycles, or makes
    something other than a time."""


class Time(float):
    """A wall-clock time: a float of nanoseconds holding its exact value.

    Made by multiplying a unit, ``20 * pulse.ns``; durations, waits and
    ``pulse.cycles_on`` take it, and convert it to cycles of the clock the
    kernel compiles for.
    """

    __slots__ = ("exact_ns",)
    exact_ns: Fraction

    def __new__(cls, nanoseconds: Fraction | int | float) -> Time:
        exact = _exact(nanoseconds)
        self = super().__new__(cls, float(exact))
        self.exact_ns = exact
        return self

    def __reduce__(self) -> tuple[Any, ...]:
        return (_time, (self.exact_ns.numerator, self.exact_ns.denominator))

    def __repr__(self) -> str:
        value = self.exact_ns
        for unit, scale in (("s", 10**9), ("ms", 10**6), ("us", 10**3)):
            if abs(value) >= scale:
                return f"{float(value / scale):g} {unit}"
        return f"{float(value):g} ns"

    __str__ = __repr__

    def __format__(self, spec: str) -> str:
        return repr(self) if not spec else float.__format__(self, spec)

    # ── Arithmetic that keeps the unit ──────────────────────────────────

    def __mul__(self, other: Any) -> Time:
        if other.__class__ is Time:
            raise UnitError(f"{self} * {other} is a time times a time, which is not a time")
        if _is_number(other):
            return Time(self.exact_ns * _exact(other))
        return NotImplemented

    # Not an alias of __mul__: editors would type 100 * pulse.ns as float.
    def __rmul__(self, other: Any) -> Time:
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> Any:
        if other.__class__ is Time:
            return float(self.exact_ns / other.exact_ns)
        if _is_number(other):
            return Time(self.exact_ns / _exact(other))
        return NotImplemented

    def __rtruediv__(self, other: Any) -> Any:
        if _is_number(other):
            raise UnitError(f"{other!r} / {self} is a number over a time, which is not a time")
        return NotImplemented

    def __add__(self, other: Any) -> Time:
        if other.__class__ is Time:
            return Time(self.exact_ns + other.exact_ns)
        if other == 0 and other.__class__ is int:
            return self
        if _is_number(other):
            raise _mixed(self, "+", other)
        return NotImplemented

    def __radd__(self, other: Any) -> Time:
        if other == 0 and other.__class__ is int:
            return self
        if _is_number(other):
            raise _mixed(other, "+", self)
        return NotImplemented

    def __sub__(self, other: Any) -> Time:
        if other.__class__ is Time:
            return Time(self.exact_ns - other.exact_ns)
        if _is_number(other):
            raise _mixed(self, "-", other)
        return NotImplemented

    def __rsub__(self, other: Any) -> Any:
        if _is_number(other):
            raise _mixed(other, "-", self)
        return NotImplemented

    def __neg__(self) -> Time:
        return Time(-self.exact_ns)

    def __pos__(self) -> Time:
        return self

    def __abs__(self) -> Time:
        return Time(abs(self.exact_ns))

    def __floordiv__(self, other: Any) -> Any:
        if other.__class__ is Time:
            return int(self.exact_ns // other.exact_ns)
        raise UnitError(f"{self} // {other!r}: divide a time by a time, or use /")

    def __rfloordiv__(self, other: Any) -> Any:
        raise UnitError(f"{other!r} // {self} is a number over a time, which is not a time")

    def __mod__(self, other: Any) -> Time:
        if other.__class__ is Time:
            return Time(self.exact_ns % other.exact_ns)
        raise UnitError(f"{self} % {other!r}: the remainder of a time is by a time")

    def __rmod__(self, other: Any) -> Any:
        raise UnitError(f"{other!r} % {self}: the remainder of a time is by a time")

    def __pow__(self, other: Any, modulo: Any = None) -> Any:
        raise UnitError(f"{self} ** {other!r} is not a time")

    def __rpow__(self, other: Any, modulo: Any = None) -> Any:
        raise UnitError(f"{other!r} ** {self} raises to a time")

    # ── Comparisons: with times, or with zero ───────────────────────────

    def _other(self, other: Any) -> Fraction | None:
        if other.__class__ is Time:
            return other.exact_ns
        if _is_number(other):
            if other == 0:
                return Fraction(0)
            raise UnitError(
                f"comparing {self} with {other!r}: give the number with units "
                f"({other!r} * pulse.ns) to compare it with a time"
            )
        return None

    def __eq__(self, other: object) -> bool:
        if other.__class__ is Time:
            return self.exact_ns == other.exact_ns
        return other == 0 and self.exact_ns == 0 if _is_number(other) else False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    __hash__ = float.__hash__

    def __lt__(self, other: Any) -> bool:
        exact = self._other(other)
        return NotImplemented if exact is None else self.exact_ns < exact

    def __le__(self, other: Any) -> bool:
        exact = self._other(other)
        return NotImplemented if exact is None else self.exact_ns <= exact

    def __gt__(self, other: Any) -> bool:
        exact = self._other(other)
        return NotImplemented if exact is None else self.exact_ns > exact

    def __ge__(self, other: Any) -> bool:
        exact = self._other(other)
        return NotImplemented if exact is None else self.exact_ns >= exact


def _time(numerator: int, denominator: int) -> Time:
    return Time(Fraction(numerator, denominator))


def _mixed(left: Any, op: str, right: Any) -> UnitError:
    cycles = right if left.__class__ is Time else left
    return UnitError(
        f"{left!r} {op} {right!r} mixes a time with {cycles!r} cycles; give both with "
        f"units ({cycles!r} * pulse.ns) or both in cycles"
    )


ns = Time(Fraction(1))
us = Time(Fraction(10**3))
ms = Time(Fraction(10**6))
s = Time(Fraction(10**9))


_CYCLES_PER_NS: dict[float, Fraction] = {}


def cycles_per_ns(clock_ghz: float) -> Fraction:
    """Cycles of a *clock_ghz* clock in one nanosecond, exactly as written."""
    found = _CYCLES_PER_NS.get(clock_ghz)
    if found is None:
        found = _CYCLES_PER_NS[clock_ghz] = _exact(clock_ghz)
    return found


def in_cycles(time: Time, clock_ghz: float) -> Fraction:
    """*time* in cycles of a *clock_ghz* clock, exactly."""
    return time.exact_ns * cycles_per_ns(clock_ghz)
