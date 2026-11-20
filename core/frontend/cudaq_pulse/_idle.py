# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Idle replacement: the rule that ``pulse.compile(k, idle_replacement=)``
takes, the options of a compilation that has one, and that compilation.
Only a compilation that lists ``Pass.IDLE_REPLACEMENT`` loads this.

An idle window is a stretch of the schedule, at least the rule's threshold
long, in which a qudit does nothing; the rule's kernel, the fill, replaces
it. The windows are known only once the program is scheduled, so the pass
is not one of the rewrites that ``passes=`` runs before the schedule:
``CompileOptions.make`` takes it out of that list and keeps the rule.
"""

from __future__ import annotations

import dataclasses
import re
from typing import TYPE_CHECKING, Any

from ._errors import (
    CompilationError,
    InvalidTypeError,
    InvalidValueError,
    UnsupportedError,
    _closest,
)
from .compiler import CompileOptions

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ._device import Qudit
    from ._units import Time
    from .kernel._kernel import Kernel, _Entry

__all__ = ["IdleReplacement"]

_ALIGNS = ("start", "center", "end")
# The float parameters a wait's length reads, as its source names them.
_ARGUMENT = re.compile(r"a\[(\d+)\]")


def _threshold(value: Any) -> int | Time:
    """*value* as a threshold: an int of clock cycles or a time."""
    from ._units import Time

    if isinstance(value, Time):
        if float(value) < 0:  # a time compares only with a time; its float is nanoseconds
            raise InvalidValueError(
                f"IdleReplacement(): threshold is {value!r}; it cannot be negative"
            )
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidTypeError(
            "IdleReplacement(): threshold is an int number of clock cycles or a time with "
            f"units, as 100 * pulse.ns, not {value!r}"
        )
    if isinstance(value, float):
        shown = f"{value:g}"
        whole = str(round(value)) if value == value and abs(value) != float("inf") else shown
        raise InvalidTypeError(
            f"IdleReplacement(): threshold is a float ({value!r}). It is an int number of "
            "clock cycles, or a time with units:\n"
            f"    threshold={whole}\n"
            f"    threshold={shown} * pulse.ns"
        )
    if value < 0:
        raise InvalidValueError(f"IdleReplacement(): threshold is {value}; it cannot be negative")
    return value


def _fill(fill: Any) -> Kernel[...] | None:
    """*fill* as the kernel that replaces a window, or None."""
    if fill is None:
        return None
    from .kernel._kernel import Kernel

    if not isinstance(fill, Kernel):
        raise InvalidTypeError(
            "IdleReplacement(): fill is a @pulse.kernel that takes (qudit, window), or None "
            f"to list the idle windows only, not {fill!r}"
        )
    parameters = list(fill.__signature__.parameters.values())
    positional = [p for p in parameters if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    plain = len(positional) == len(parameters) == 2 and not fill.runtime_parameters
    if not plain:
        raise InvalidTypeError(
            f"IdleReplacement(): fill {fill.__name__}{fill.__signature__} does not take "
            "(qudit, window): a fill takes the idle qudit and the window's length in clock "
            "cycles, neither annotated float or Qudit"
        )
    return fill


def _qudits(qudits: Iterable[Qudit] | None) -> tuple[Qudit, ...] | None:
    if qudits is None:
        return None
    from ._device import Qudit

    if isinstance(qudits, Qudit):
        qudits = (qudits,)
    try:
        chosen = tuple(qudits)
    except TypeError:
        chosen = (qudits,)  # type: ignore[assignment]
    for qudit in chosen:
        if not isinstance(qudit, Qudit):
            raise InvalidTypeError(
                "IdleReplacement(): qudits are qudit handles of a quantum system, as "
                f"system.qudits[0:4], not {qudit!r}"
            )
    if not chosen:
        raise InvalidValueError(
            "IdleReplacement(): qudits is empty, so no window could be replaced; leave it "
            "out for every qudit"
        )
    return chosen


class IdleReplacement:
    """What ``Pass.IDLE_REPLACEMENT`` replaces idle time with.

    *fill* is a ``@pulse.kernel`` that takes ``(qudit, window)``: the idle
    qudit and the length of its idle window in clock cycles. It is traced
    once for each window, where the window is, as a kernel called there;
    ``None`` replaces nothing and only lists the windows. *threshold* is the
    shortest idle that is a window, an ``int`` of clock cycles or a time
    (``100 * pulse.ns``); an idle of exactly that length is one. *qudits*
    keeps the rule to those qudits. The idle before a qudit's first
    operation and after its last is left alone unless *leading* or
    *trailing* is true. *align* places a fill shorter than its window in
    it: ``"center"``, ``"start"`` or ``"end"``.

    A rule is immutable; two rules are equal when they hold the same kernel
    and settings.
    """

    __slots__ = ("fill", "threshold", "qudits", "leading", "trailing", "align")

    fill: Kernel[...] | None
    threshold: int | Time
    qudits: tuple[Qudit, ...] | None
    leading: bool
    trailing: bool
    align: str

    def __init__(
        self,
        fill: Kernel[...] | None,
        /,
        *,
        threshold: int | Time,
        qudits: Iterable[Qudit] | None = None,
        leading: bool = False,
        trailing: bool = False,
        align: str = "center",
    ) -> None:
        if align not in _ALIGNS:
            close = _closest(str(align), _ALIGNS)
            hint = "" if close is None else f" Did you mean {close!r}?"
            raise InvalidValueError(
                f"IdleReplacement(): unknown align {align!r}.{hint} Choose from: "
                f"{', '.join(_ALIGNS)}"
            )
        fields = (
            _fill(fill),
            _threshold(threshold),
            _qudits(qudits),
            bool(leading),
            bool(trailing),
            align,
        )
        for name, value in zip(self.__slots__, fields, strict=True):
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("an IdleReplacement is immutable; make another one")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("an IdleReplacement is immutable; make another one")

    def _key(self) -> tuple[Any, ...]:
        # A time and a count of cycles are different thresholds even when
        # one clock makes them the same length.
        threshold = self.threshold
        return (
            id(self.fill),
            threshold.__class__,
            threshold,
            self.qudits,
            self.leading,
            self.trailing,
            self.align,
        )

    def __eq__(self, other: object) -> bool:
        return other.__class__ is IdleReplacement and other._key() == self._key()  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        fill = self.fill
        name = "None" if fill is None else f"{fill.__module__}.{fill.__qualname__}"
        threshold = self.threshold
        parts = [name, f"threshold={threshold!r}"]
        if self.qudits is not None:
            parts.append(f"qudits=({', '.join(str(q.index) for q in self.qudits)})")
        if self.leading:
            parts.append("leading=True")
        if self.trailing:
            parts.append("trailing=True")
        if self.align != "center":
            parts.append(f"align={self.align!r}")
        return f"IdleReplacement({', '.join(parts)})"


@dataclasses.dataclass(frozen=True, repr=False)
class _IdleOptions(CompileOptions):
    """The options of a compilation that replaces idle windows: those of the
    compilation without it, and the rule. Equal only to others with an equal
    rule, so its programs are traced and kept apart."""

    # Fields here: every way of copying options (dataclasses.replace, the
    # field-by-field copy of the unit warnings) keeps them.
    idle_replacement: IdleReplacement | None = None  # type: ignore[assignment]
    unit_warnings: bool = False  # type: ignore[assignment]

    def without_rule(self) -> CompileOptions:
        """The options of the same compilation with no idle replacement."""
        plain = CompileOptions(*(getattr(self, f.name) for f in dataclasses.fields(CompileOptions)))
        if self.unit_warnings:
            from .kernel._unit_warnings import with_unit_warnings

            return with_unit_warnings(plain)
        return plain

    def __repr__(self) -> str:
        return f"{self.without_rule()!r} + {self.idle_replacement!r}"


def with_idle_replacement(
    options: CompileOptions, rule: IdleReplacement | None, listed: bool
) -> CompileOptions:
    """*options* for a compilation whose ``passes=`` named
    ``Pass.IDLE_REPLACEMENT`` (*listed*) and whose ``idle_replacement=`` is
    *rule*: both, or an error that names the missing one."""
    if rule is None:
        raise InvalidValueError(
            "passes= lists Pass.IDLE_REPLACEMENT, which needs "
            "idle_replacement=pulse.IdleReplacement(fill, threshold=...)"
        )
    if not isinstance(rule, IdleReplacement):
        raise InvalidTypeError(
            f"compile(idle_replacement=) takes a pulse.IdleReplacement, got {rule!r}"
        )
    if not listed:
        raise InvalidValueError(
            "idle_replacement= is given but passes= does not list Pass.IDLE_REPLACEMENT; "
            "add it, as passes=(*pulse.compiler.DEFAULT_PASSES, pulse.Pass.IDLE_REPLACEMENT)"
        )
    return _IdleOptions(
        *(getattr(options, f.name) for f in dataclasses.fields(CompileOptions)),
        idle_replacement=rule,
        unit_warnings=options.unit_warnings,
    )


def _float_wait(kernel: Kernel[...], entry: _Entry) -> CompilationError | None:
    """The error for a kernel one of whose waits has a length that its
    float parameters set, or None: such a length, and with it every idle
    window near it, exists only once a call gives the values."""
    waits = entry.lowered.waits
    if waits is None:
        return None
    names: list[str] = []
    for wait in waits.waits:
        if wait.lowest is None:
            continue  # as long as an answer: a run-time length, not a float's
        for index in _ARGUMENT.findall(wait.source):
            name = kernel._names[entry.runtime[int(index)]]
            if name not in names:
                names.append(name)
    if not names:
        return None
    one = len(names) == 1
    which = f"parameter {names[0]!r}" if one else "parameters " + ", ".join(map(repr, names))
    return CompilationError(
        f"idle replacement: the length of a wait in {kernel.__name__}() is computed from the "
        f"float {which}, so its idle windows differ at every call",
        fix=(
            f"annotate {names[0] if one else 'them'} as int to compile for each value, or "
            "compile without idle replacement"
        ),
        code="symbolic_parameter",
    )


def specialize(
    kernel: Kernel[...], values: tuple[Any, ...], target: Any, options: CompileOptions
) -> _Entry:
    """Compile *kernel* for *values* on *target* with the idle windows of
    its schedule replaced as the rule of *options* says.

    The program without the rule is compiled first, as any program: its
    schedule is where the windows are read from.
    """
    assert isinstance(options, _IdleOptions)
    if kernel._relocatable:
        raise UnsupportedError(
            f"idle replacement: {kernel.__name__}() takes qudits or lists that each call "
            "binds (parameters annotated Qudit, Register or list), which idle replacement "
            "does not support yet; leave the qudit parameters unannotated"
        )
    base = kernel._specialize(values, target, options.without_rule())
    refused = _float_wait(kernel, base)
    if refused is not None:
        raise refused
    raise UnsupportedError(
        "idle replacement: reading the idle windows of a schedule and replacing them is not "
        "built yet; this version has the option, its checks and the refusals"
    )
