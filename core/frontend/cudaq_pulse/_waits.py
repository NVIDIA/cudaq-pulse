# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The checks of a module's variable waits, which bind runs on every call's
values. Only a kernel with a wait whose length a float sets loads them."""

from __future__ import annotations

import marshal
import math
from collections.abc import Sequence
from types import FunctionType
from typing import Any

from ._errors import InvalidValueError
from .kernel.ir_builder import MAX_CYCLES

# Longest wait a call may ask for: lengths stay exact in a float.
_MAX_WAIT_CYCLES = MAX_CYCLES


class _WaitLength:
    """A wait whose length the module computes from its float arguments.

    The call's values give its length, rounded to the nearest cycle with ties
    up as the module rounds it, and checked against its bounds before the
    call runs. For a length that also takes answers of classical calls this
    is its float part, the length with its answers at 0, which *text* names:
    it has no lowest length (*lowest* is None) and must be finite and at most
    the highest, since the run clamps the whole length to the bounds.
    """

    __slots__ = ("source", "length", "lowest", "highest", "text", "kernel")

    def __init__(
        self,
        expression: Any,
        lowest: int | None,
        highest: int,
        kernel: str,
        text: str | None = None,
    ) -> None:
        from .kernel.ir_builder import arithmetic_source, arithmetic_text

        self.source = arithmetic_source(expression)
        self.length = eval(f"lambda a: {self.source}", {"__builtins__": {}})
        self.lowest = lowest
        self.highest = None if highest < 0 else highest
        self.text = arithmetic_text(expression) if text is None else text
        self.kernel = kernel

    def __reduce__(self) -> tuple[Any, ...]:
        return (_rebuilt_wait, (self.source, self.lowest, self.highest, self.text, self.kernel))

    def check(self, values: Sequence[float]) -> None:
        """Raise ``ValueError`` saying why the length is not allowed."""
        try:
            x = self.length(values)
        except ZeroDivisionError:
            x = math.nan
        if not math.isfinite(x):
            raise ValueError(
                f"{self.kernel}(): the wait length {self.text} is {x!r}; it must be a "
                "finite number of clock cycles"
            )
        cycles = math.floor(x + 0.5)
        if self.lowest is None:
            if self.highest is not None and cycles > self.highest:
                raise ValueError(
                    f"{self.kernel}(): the wait length {self.text} = {x!r} rounds to "
                    f"{cycles} clock cycles, above its highest length {self.highest}"
                )
            return
        if cycles < 0 and self.lowest == 0:
            raise ValueError(
                f"{self.kernel}(): the wait length {self.text} = {x!r} rounds to "
                f"{cycles} clock cycles; a wait cannot be negative"
            )
        highest = _MAX_WAIT_CYCLES if self.highest is None else self.highest
        if not self.lowest <= cycles <= highest:
            bounds = (self.lowest, self.highest)
            raise ValueError(
                f"{self.kernel}(): the wait length {self.text} = {x!r} rounds to "
                f"{cycles} clock cycles, outside its bounds {bounds!r}"
            )


class _WaitLengths:
    """The variable waits of a module, checked together on every call."""

    __slots__ = ("ok", "waits")

    # By wait, the steps of the lengths a controller with a coarser clock
    # than the program's can time (none: every length).
    grids: tuple[int, ...] = ()

    def __init__(
        self, waits: Sequence[tuple[Any, ...]], kernel: str, grids: Sequence[int] = ()
    ) -> None:
        self.waits = tuple(
            _WaitLength(wait[0], wait[1], wait[2], kernel, wait[3] if len(wait) > 3 else None)
            for wait in waits
        )
        # floor(x + 0.5) is in [lowest, highest] exactly when
        # lowest <= x + 0.5 < highest + 1, which NaN fails; without a lowest,
        # -1e400 (minus infinity) < x + 0.5 also fails for minus infinity.
        tests = " and ".join(
            f"{'-1e400 <' if w.lowest is None else f'{w.lowest} <='} {w.source} + 0.5 < "
            f"{(_MAX_WAIT_CYCLES if w.highest is None else w.highest) + 1}"
            for w in self.waits
        )
        if grids:
            tests += "".join(
                f" and ({w.source} + 0.5) // 1 % {grid} == 0"
                for w, grid in zip(self.waits, grids, strict=True)
                if w.lowest is not None and grid > 1
            )
        self.ok = eval(f"lambda a: {tests}", {"__builtins__": {}})

    def __reduce__(self) -> tuple[Any, ...]:
        # The check as compiled: the stamp of the pickle names the Python
        # whose marshal format wrote it.
        check = marshal.dumps(self.ok.__code__)
        return (_rebuilt_wait_lengths, (type(self), self.waits, self.grids, check))

    def check(self, values: Sequence[float]) -> None:
        try:
            if self.ok(values):
                return
        except ZeroDivisionError:
            pass
        for wait in self.waits:
            wait.check(values)
        for wait, grid in zip(self.waits, self.grids):
            cycles = math.floor(wait.length(values) + 0.5)
            if wait.lowest is not None and cycles % grid:
                lower = cycles // grid * grid
                raise InvalidValueError(
                    f"{wait.kernel}(): the wait length {wait.text} rounds to {cycles} clock "
                    f"cycles; its line moves in timesteps of {grid} cycles, and "
                    f"{cycles} cycles isn't a whole number of timesteps: the nearest "
                    f"lengths are {lower} and {lower + grid}"
                )


class _GridWaitLengths(_WaitLengths):
    """Variable waits on lines that move in timesteps of more than one
    cycle (*grids*, by wait). Its own class keeps the grids off the waits of
    every module on one clock."""

    __slots__ = ("grids",)

    def __init__(self, waits: Sequence[tuple[Any, ...]], kernel: str, grids: Sequence[int]) -> None:
        self.grids = tuple(grids)
        super().__init__(waits, kernel, self.grids)


def wait_lengths(
    waits: Sequence[tuple[Any, ...]], kernel: str, grid: int, grids: Sequence[int] | None = None
) -> _WaitLengths:
    """The checks of *waits*, each on *grid*, or on its own of *grids*."""
    if grids is not None:
        return _GridWaitLengths(waits, kernel, grids)
    if grid == 1:
        return _WaitLengths(waits, kernel)
    return _GridWaitLengths(waits, kernel, [grid] * len(waits))


def _rebuilt_wait(
    source: str, lowest: int | None, highest: int | None, text: str, kernel: str
) -> _WaitLength:
    wait = _WaitLength.__new__(_WaitLength)
    wait.source = source
    wait.length = eval(f"lambda a: {source}", {"__builtins__": {}})
    wait.lowest = lowest
    wait.highest = highest
    wait.text = text
    wait.kernel = kernel
    return wait


def _rebuilt_wait_lengths(
    cls: type[_WaitLengths], waits: tuple[_WaitLength, ...], grids: tuple[int, ...], check: bytes
) -> _WaitLengths:
    lengths = cls.__new__(cls)
    lengths.waits = waits
    if isinstance(lengths, _GridWaitLengths):
        lengths.grids = grids
    lengths.ok = FunctionType(marshal.loads(check), {"__builtins__": {}})
    return lengths
