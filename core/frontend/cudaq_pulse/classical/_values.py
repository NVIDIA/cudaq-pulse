# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The wire types of classical function arguments and results.

A function's parameter and return annotations become these when it is
declared (``pulse.Bits`` of a known length is ``bits:n``, ``float`` is
``f64``, ``list[int]`` of measurements is ``levels:n``, and so on). They are
what the IR, the wire schema and the simulator read; users never name them.
"""

from __future__ import annotations

from .._errors import InvalidValueError

__all__ = ["ValueType"]


class ValueType:
    """The type of one argument or result of a classical function.

    ``descriptor`` is its spelling in the IR, ``count`` how many values it
    holds, ``source`` what a kernel passes for it (``"measurement"``,
    ``"float"`` or ``"integer"``), and ``answer`` the type of the values a
    kernel gets back for it (``"i1"``, ``"i64"`` or ``"f64"``).
    """

    __slots__ = ("count", "descriptor", "source", "answer", "answer_count", "wire_bytes")

    def __init__(
        self,
        descriptor: str,
        count: int,
        source: str,
        answer: str,
        answer_count: int,
        wire_bytes: int,
    ) -> None:
        self.descriptor = descriptor
        self.count = count
        self.source = source
        self.answer = answer
        self.answer_count = answer_count
        self.wire_bytes = wire_bytes

    @property
    def is_scalar(self) -> bool:
        return ":" not in self.descriptor and "[" not in self.descriptor

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ValueType) and other.descriptor == self.descriptor

    def __hash__(self) -> int:
        return hash(self.descriptor)

    def __repr__(self) -> str:
        return f"<{self.descriptor}>"


def _count(n: int, what: str) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise InvalidValueError(f"{what} needs a positive whole number of values, got {n!r}")
    return n


class _Bits(ValueType):
    """``n`` measurement results, sent as ``outcome != 0``, bit-packed; as a
    result, ``n`` booleans."""

    __slots__ = ()

    def __init__(self, n: int) -> None:
        n = _count(n, "a pulse.Bits value")
        super().__init__(f"bits:{n}", n, "measurement", "i1", n, (n + 7) // 8)


class _OneBit(_Bits):
    """``bits:1`` answered as one ``bool`` (a ``-> bool`` result)."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(1)

    @property
    def is_scalar(self) -> bool:
        return True


class _Levels(ValueType):
    """``n`` measured levels, one byte each; as a result, ``n`` integers."""

    __slots__ = ()

    def __init__(self, n: int) -> None:
        n = _count(n, "a list[int] of measurements")
        super().__init__(f"levels:{n}", n, "measurement", "i64", n, n)


class _IQ(ValueType):
    """The IQ points of ``n`` readouts, as ``2n`` 32-bit floats (I, then Q)."""

    __slots__ = ()

    def __init__(self, n: int) -> None:
        n = _count(n, "a list[complex]")
        super().__init__(f"iq:{n}", n, "measurement", "f64", 2 * n, 8 * n)


class _OnePoint(_IQ):
    """``iq:1`` taken as one ``complex`` (a ``complex`` parameter)."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(1)

    @property
    def is_scalar(self) -> bool:
        return True


def _width(bits: int) -> int:
    if bits not in (64, 32):
        raise InvalidValueError(f"a number has 64 or 32 bits, got {bits!r}")
    return bits


class _Float(ValueType):
    """One floating-point number, of 64 bits (what ``float`` is) or 32."""

    __slots__ = ()

    def __init__(self, bits: int = 64) -> None:
        bits = _width(bits)
        super().__init__(f"f{bits}", 1, "float", "f64", 1, bits // 8)


class _Integer(ValueType):
    """One signed integer, of 64 bits (the default) or 32."""

    __slots__ = ()

    def __init__(self, bits: int = 64) -> None:
        bits = _width(bits)
        super().__init__(f"i{bits}", 1, "integer", "i64", 1, bits // 8)


class _Byte(ValueType):
    """One unsigned byte."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("u8", 1, "integer", "i64", 1, 1)


class _Floats(ValueType):
    """``n`` floating-point numbers, of 64 bits (``list[float]``) or 32 each."""

    __slots__ = ()

    def __init__(self, n: int, bits: int = 64) -> None:
        n = _count(n, "a list[float]")
        bits = _width(bits)
        super().__init__(f"f{bits}[{n}]", n, "float", "f64", n, n * bits // 8)


class _Integers(ValueType):
    """``n`` signed 32-bit integers."""

    __slots__ = ()

    def __init__(self, n: int) -> None:
        n = _count(n, "a list[int]")
        super().__init__(f"i32[{n}]", n, "integer", "i64", n, 4 * n)


class _Bytes(ValueType):
    """``n`` unsigned bytes."""

    __slots__ = ()

    def __init__(self, n: int) -> None:
        n = _count(n, "a byte array")
        super().__init__(f"u8[{n}]", n, "integer", "i64", n, n)
