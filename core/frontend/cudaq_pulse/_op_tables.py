# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Values by class of operation: what an operation of each class costs on a
processor (``OpCycles``) and how many of each a function runs
(``OpCounts``). Their fields are the classes of ``OpClass``, by value."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING, Any, ClassVar

from ._errors import InvalidValueError, NotFoundError

if TYPE_CHECKING:
    from typing import Self

# The classes a function body's operations are counted in; a processor also
# costs "loop", a rolled loop's counter and jump per iteration.
_COUNTED = (
    "add",
    "mul",
    "div",
    "compare",
    "branch",
    "load",
    "convert",
    "popcount",
    "sqrt",
    "exp",
    "log",
    "trig",
)
_PROCESSOR_OPS = (*_COUNTED, "loop")


class _OpTable:
    """A read-only value per class of operation: ``table.div``, or
    ``table[pulse.controller.OpClass.DIV]``."""

    __slots__ = _COUNTED
    _names: ClassVar[tuple[str, ...]] = _COUNTED
    _default: ClassVar[int]

    add: int
    mul: int
    div: int
    compare: int
    branch: int
    load: int
    convert: int
    popcount: int
    sqrt: int
    exp: int
    log: int
    trig: int

    def _set(self, values: tuple[int, ...]) -> None:
        cls = type(self).__name__
        for name, value in zip(self._names, values, strict=True):
            if value.__class__ is not int or value < 0:
                raise InvalidValueError(f"{cls}.{name} is a whole number >= 0, got {value!r}")
            object.__setattr__(self, name, value)

    def __getitem__(self, op: str) -> int:
        if op in self._names:
            value: int = getattr(self, op)
            return value
        raise NotFoundError(
            f"{type(self).__name__} has no class {op!r}; its classes are {', '.join(self._names)}"
        )

    # A value, not a mapping or a sequence: read it by class.
    __iter__ = None

    def _values(self) -> tuple[int, ...]:
        return tuple(getattr(self, name) for name in self._names)

    def _given(self) -> dict[str, int]:
        """The classes whose value is not the default, as a file holds them."""
        default = self._default
        return {n: v for n in self._names if (v := getattr(self, n)) != default}

    def __setattr__(self, name: str, value: Any) -> None:
        raise FrozenInstanceError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> None:
        raise FrozenInstanceError(f"cannot delete field {name!r}")

    def __eq__(self, other: object) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self._values() == other._values()

    def __hash__(self) -> int:
        return hash((type(self).__name__, self._values()))

    def __repr__(self) -> str:
        given = ", ".join(f"{n}={v!r}" for n, v in self._given().items())
        return f"{type(self).__name__}({given})"

    def __reduce__(self) -> tuple[Any, ...]:
        from ._pickling import rebuilt

        return (rebuilt, (type(self), self._given()))

    def replace(self, **changes: int) -> Self:
        """A copy with the classes *changes* names set to their values."""
        values = {n: getattr(self, n) for n in self._names}
        values.update(changes)
        return type(self)(**values)

    __replace__ = replace


class OpCounts(_OpTable):
    """How many operations of each class something runs: a function's
    longest path (``FunctionCost.ops``), or what a ``controller_cost`` hook
    says a call of a built-in takes. Every class not given is 0; an unknown
    one is a ``TypeError``, as for any keyword."""

    __slots__ = ()
    _default = 0

    def __init__(
        self,
        *,
        add: int = 0,
        mul: int = 0,
        div: int = 0,
        compare: int = 0,
        branch: int = 0,
        load: int = 0,
        convert: int = 0,
        popcount: int = 0,
        sqrt: int = 0,
        exp: int = 0,
        log: int = 0,
        trig: int = 0,
    ) -> None:
        self._set((add, mul, div, compare, branch, load, convert, popcount, sqrt, exp, log, trig))

    @property
    def total(self) -> int:
        """The operations of every class."""
        return sum(self._values())


class OpCycles(_OpTable):
    """The clock cycles one operation of each class takes on a processor
    (``Processor.cycles_per_op``), and ``loop``, a rolled loop's counter and
    jump per iteration. Every class not given takes 1 cycle; an unknown one
    is a ``TypeError``, as for any keyword."""

    __slots__ = ("loop",)
    _names = _PROCESSOR_OPS
    _default = 1

    loop: int

    def __init__(
        self,
        *,
        add: int = 1,
        mul: int = 1,
        div: int = 1,
        compare: int = 1,
        branch: int = 1,
        load: int = 1,
        convert: int = 1,
        popcount: int = 1,
        sqrt: int = 1,
        exp: int = 1,
        log: int = 1,
        trig: int = 1,
        loop: int = 1,
    ) -> None:
        self._set(
            (add, mul, div, compare, branch, load, convert, popcount, sqrt, exp, log, trig, loop)
        )


# A processor whose cycles_per_op is not given.
_ONE_CYCLE = OpCycles()
