# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The arithmetic a controller computes a classical function's body in
(``number_format``), for ``controller.evaluate_function``.

``"float64"`` is the run-time expressions' own arithmetic. In ``"float32"``
every float an argument, a constant or a term gives is rounded to IEEE 754
binary32, to nearest even, an overflow to ±inf; a term is computed in
binary64 first, which for ``+ - * /`` and ``sqrt`` is the binary32 result
exactly. In ``"fixed:<i>.<f>"`` a float is a signed two's complement
number of ``i + f`` bits, ``f`` of them fractional: ``add``, ``sub``,
``mul``, ``div``, ``neg`` and ``abs`` are exact, then rounded to a multiple
of ``2 ** -f`` (nearest, ties to even) and saturated to the format's range;
the other terms are computed in binary64 and rounded the same way, an
infinity saturating and a NaN becoming 0, and a division by zero saturates
(0 / 0 is 0). Integers and bools keep their 64-bit arithmetic.
"""

from __future__ import annotations

import math
import re
import struct
from collections.abc import Mapping, Sequence
from typing import Any

from ._errors import InvalidValueError

_FIXED = re.compile(r"fixed:(\d+)\.(\d+)")
# A fixed-point number is a float exactly: its bits fit a binary64 significand.
_MOST_BITS = 53
_EXACT_TERMS = frozenset({"add", "sub", "mul", "div", "neg", "abs"})
_BOOLEAN = frozenset({"eq", "ne", "lt", "le", "gt", "ge"})


class Format:
    """A parsed ``number_format``: float32, or fixed point of *integer* and
    *fraction* bits."""

    __slots__ = ("fraction", "integer", "name")

    def __init__(self, name: str, integer: int = 0, fraction: int = 0) -> None:
        self.name, self.integer, self.fraction = name, integer, fraction

    def round(self, x: float) -> float:
        """*x* in this format."""
        if self.name == "float32":
            try:
                rounded: float = struct.unpack("<f", struct.pack("<f", x))[0]
            except OverflowError:
                return math.copysign(math.inf, x)
            return rounded
        if x != x:
            return 0.0
        scale = 2.0**self.fraction
        scaled = x * scale
        if math.isinf(scaled):
            return (self.most if x > 0 else self.least) / scale
        return self.saturated(round(scaled)) / scale

    @property
    def most(self) -> int:
        return (1 << (self.integer + self.fraction - 1)) - 1

    @property
    def least(self) -> int:
        return -(1 << (self.integer + self.fraction - 1))

    def saturated(self, n: int) -> int:
        return min(max(n, self.least), self.most)

    def exact(self, op: str, values: Sequence[float]) -> float:
        """The fixed-point *op* of *values*: exact, then rounded and saturated."""
        f = self.fraction
        units = [int(v * 2.0**f) for v in values]
        if op == "neg":
            n = -units[0]
        elif op == "abs":
            n = abs(units[0])
        else:
            a, b = units
            if op == "add":
                n = a + b
            elif op == "sub":
                n = a - b
            elif op == "mul":
                n = _divided(a * b, 1 << f)
            elif b == 0:
                n = self.most if a > 0 else self.least if a < 0 else 0
            else:
                n = _divided(a << f, b)
        return self.saturated(n) / 2.0**f


def _divided(a: int, b: int) -> int:
    """``a / b`` rounded to the nearest integer, ties to even."""
    q, r = divmod(a, b)
    twice = 2 * r
    if b < 0:
        twice, b = -twice, -b
    if twice > b or (twice == b and q % 2):
        q += 1
    return q


def parse(text: Any, name: str = "") -> Format | None:
    """The format *text* names; None for ``"float64"``. Raises
    ``InvalidValueError`` for any other text, naming the function *name*."""
    if text == "float64":
        return None
    if text == "float32":
        return Format("float32")
    found = _FIXED.fullmatch(text) if isinstance(text, str) else None
    if found is not None:
        integer, fraction = int(found[1]), int(found[2])
        if integer >= 1 and integer + fraction <= _MOST_BITS:
            return Format(str(text), integer, fraction)
    where = f"{name}: " if name else ""
    raise InvalidValueError(
        f"{where}number_format must be 'float64', 'float32' or 'fixed:<i>.<f>' with i >= 1 "
        f"integer bits (the sign among them) and i + f <= {_MOST_BITS}, got {text!r}"
    )


def evaluate(
    number: Format, body: Mapping[str, Any], arguments: Sequence[Any], descriptors: Sequence[str]
) -> tuple[float | int | bool, ...]:
    """``evaluate_function`` in *number*'s arithmetic."""
    from .controller import _evaluate

    floats = [d.split(":")[0].split("[")[0] in ("f64", "f32") for d in descriptors]
    args = [
        (
            ([number.round(float(v)) for v in a] if isinstance(a, list) else number.round(float(a)))
            if is_float
            else a
        )
        for a, is_float in zip(arguments, floats, strict=True)
    ]
    local: dict[str, Any] = {}

    def value(e: Any) -> Any:
        if "const" in e:
            return e["const"]
        if "local" in e:
            if e["local"] not in local:
                raise InvalidValueError(f"the local {e['local']!r} is read before it is bound")
            return local[e["local"]]
        if "argument" in e:
            got = args[e["argument"]]
            return got[e["index"]] if "index" in e else got
        op, kind = e["op"], e["type"]
        of = [value(x) for x in e["of"]]
        if kind == "float":
            at = 1 if op == "select" else 0
            of[at:] = [number.round(float(v)) for v in of[at:]]
            if number.name != "float32" and op in _EXACT_TERMS:
                return number.exact(op, of)
        term = {k: v for k, v in e.items() if k != "of"}
        got = _evaluate({**term, "of": [{"const": v} for v in of]}, {}, {})
        floating = op == "to_float" or (kind == "float" and op not in _BOOLEAN and op != "to_int")
        return number.round(float(got)) if floating else got

    for binding in body["let"]:
        local[binding["name"]] = value(binding["value"])
    return tuple(value(result) for result in body["results"])
