# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Idle replacement: the rule ``compiled.replace_idle(rule)`` takes, the
options of the kernel it returns, and how that kernel compiles. Only a
kernel whose idle windows are replaced loads this.

An idle window is a stretch of the schedule, at least the rule's threshold
long, in which a qudit does nothing; the rule's kernel, the fill, replaces
it. The windows are known only once the program is scheduled, so the
replacement is not one of the rewrites that ``passes=`` runs before the
schedule: it applies to a compiled kernel, whose compilations it runs.
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
from .compiler import _DEFAULT_MACHINE, CompileOptions
from .kernel._kernel import CompiledKernel

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ._device import Qudit
    from ._idle_windows import IdleWindows, Plan, Sites
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
    """What ``compiled.replace_idle(rule)`` replaces idle time with.

    *fill* is a ``@pulse.kernel`` that takes ``(qudit, window)``: the idle
    qudit and the length of its idle window in clock cycles. It is traced
    once for each window, where the window is, as a kernel called there;
    ``None`` replaces nothing and only lists the windows
    (``ScheduledKernel.idle_windows``). In this version a fill drives and
    waits on its own qudit's drive line, at that line's tone, and may shift
    the tone's phase by constants that add up to zero. *threshold* is the
    shortest idle that is a window, an ``int`` of clock cycles or a time
    (``100 * pulse.ns``); an idle of exactly that length is one. *qudits*
    keeps the rule to those qudits. The idle before a qudit's first
    operation and after its last is left alone unless *leading* or
    *trailing* is true. *align* places a fill shorter than its window in
    it, completed with waits: ``"center"``, ``"start"`` or ``"end"``; a fill
    longer than its window is a ``CompilationError``.

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
        return other.__class__ is IdleReplacement and other._key() == self._key()

    def __reduce__(self) -> tuple[Any, ...]:
        # With its fill, which pickles as a kernel does.
        return (_rebuilt, tuple(getattr(self, name) for name in self.__slots__))

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


def _rebuilt(
    fill: Any, threshold: Any, qudits: Any, leading: bool, trailing: bool, align: str
) -> IdleReplacement:
    return IdleReplacement(
        fill, threshold=threshold, qudits=qudits, leading=leading, trailing=trailing, align=align
    )


@dataclasses.dataclass(frozen=True, repr=False)
class _IdleOptions(CompileOptions):
    """The options of a compiled kernel whose idle windows are replaced:
    those it was compiled with, and the rule. Equal only to others with an
    equal rule, so its programs are traced and kept apart."""

    # Fields here: every way of copying options (dataclasses.replace, the
    # field-by-field copy of the unit warnings) keeps them.
    idle_replacement: IdleReplacement | None = None
    units: bool = False

    def without_rule(self) -> CompileOptions:
        """The options of the same compilation with no idle replacement."""
        plain = CompileOptions(*(getattr(self, f.name) for f in dataclasses.fields(CompileOptions)))
        if self.units:
            from .kernel._unit_warnings import with_unit_warnings

            return with_unit_warnings(plain)
        return plain

    def traced(self, sites: Sites) -> CompileOptions:
        """The options of one of the compilations an idle replacement makes
        of the program: no rule, and the record *sites* its trace keeps."""
        return _SiteOptions(
            **{f.name: getattr(self, f.name) for f in dataclasses.fields(CompileOptions)},
            units=self.units,
            _sites=sites,
        )

    def __repr__(self) -> str:
        return f"{self.without_rule()!r} + {self.idle_replacement!r}"


@dataclasses.dataclass(frozen=True)
class _SiteOptions(CompileOptions):
    """Options of one of the compilations an idle replacement makes of a
    program: those of the program without the rule, with the record its
    trace keeps of the operations it emits. A record equals only itself,
    so no compilation is taken for another. Its ``unit_warnings`` is set,
    so that ``build`` traces with ``tracer``: a compilation without idle
    replacement runs none of this."""

    units: bool = False
    _sites: Sites | None = None

    unit_warnings = True

    def tracer(self, builder: Any, target: Any) -> Any:
        checked = CompileOptions.tracer(self, builder, target) if self.units else None
        assert self._sites is not None
        return self._sites.attach(builder, checked)


class _IdleKernel:
    """*kernel*, as a dispatcher of a kernel whose idle windows are
    replaced sees it: each specialization is ``specialize``'s."""

    __slots__ = ("_kernel",)

    def __init__(self, kernel: Kernel[...]) -> None:
        object.__setattr__(self, "_kernel", kernel)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._kernel, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._kernel, name, value)

    def _specialize(self, values: tuple[Any, ...], target: Any, options: CompileOptions) -> _Entry:
        return specialize(self._kernel, values, target, options)


class IdledKernel(CompiledKernel):  # type: ignore[type-arg]
    """``compiled.replace_idle(rule)``: *compiled*, whose programs have each
    idle window replaced as *rule* says. It lowers, calls, simulates and
    pickles as a compiled kernel; *compiled* is unchanged."""

    def __init__(self, compiled: CompiledKernel[...], rule: IdleReplacement) -> None:
        if not isinstance(rule, IdleReplacement):
            raise InvalidTypeError(f"replace_idle() takes a pulse.IdleReplacement, got {rule!r}")
        if isinstance(compiled, IdledKernel):
            raise InvalidValueError(
                "replace_idle() is given a kernel whose idle windows are replaced already; "
                "call it on the compiled kernel"
            )
        base = compiled._dispatch.options
        options = _IdleOptions(
            **{f.name: getattr(base, f.name) for f in dataclasses.fields(CompileOptions)},
            idle_replacement=rule,
            units=bool(base.unit_warnings),
        )
        kernel = compiled.kernel
        self.kernel = kernel
        self.quantum_system = compiled.quantum_system
        self._compiled = compiled
        self._rule = rule
        self._dispatch = kernel._dispatcher_type(_IdleKernel(kernel), options, None)  # type: ignore[arg-type]
        self._dispatch.pin = compiled.quantum_system

    def __reduce__(self) -> tuple[Any, ...]:
        return (IdledKernel, (self._compiled, self._rule))

    def __repr__(self) -> str:
        return f"{self._compiled!r}.replace_idle({self._rule!r})"


class _LaneGapError(Exception):
    """Under rcp or alap_rcp, the fills moved an operation of the kernel:
    the policy's drive lanes, not the fills, decide where pulses go."""


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
        for index in _ARGUMENT.findall(wait.source or ""):
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


def _cycles(threshold: int | Time, clock_ghz: float) -> int:
    """*threshold* in clock cycles: the fewest that last at least a time."""
    if threshold.__class__ is int:
        return threshold
    from ._units import in_cycles

    numerator, denominator = in_cycles(threshold, clock_ghz)  # type: ignore[arg-type]
    return -(-numerator // denominator)


def _chosen(rule: IdleReplacement) -> frozenset[int] | None:
    """The indices of the qudits *rule* is restricted to, or None."""
    return None if rule.qudits is None else frozenset(q.index for q in rule.qudits)


def _traced(
    kernel: Kernel[...], values: tuple[Any, ...], target: Any, options: _IdleOptions, sites: Sites
) -> _Entry:
    """*kernel* compiled for *values* without the rule, its trace recorded
    in *sites*. An error is the one the program raises without the rule:
    the numbers the traced operations carry show in an error's text."""
    try:
        return kernel._specialize(values, target, options.traced(sites))
    except Exception as error:
        failed = error
    kernel._specialize(values, target, options.without_rule())
    raise failed


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
    if target.control_system is not None and options.machine is _DEFAULT_MACHINE:
        # Compiled for no control system: the one of the qudits' quantum
        # system, as any compilation takes it (Kernel._specialize).
        options = dataclasses.replace(options, machine=target.control_system._machine)
    if options.machine._line_delays:
        raise UnsupportedError(
            "idle replacement: the control system delays some lines (delay_ns), which idle "
            "replacement does not support yet"
        )
    from ._idle_windows import Sites, gaps, plan, read, report

    rule = options.idle_replacement
    assert rule is not None
    for qudit in rule.qudits or ():
        if qudit.quantum_system.fingerprint != target.fingerprint:
            raise InvalidValueError(
                f"idle replacement: qudits= names {qudit!r}, a qudit of another quantum "
                f"system than the one {kernel.__name__}() is compiled for ({target.name})"
            )
    sites = Sites()
    base = _traced(kernel, values, target, options, sites)
    refused = _float_wait(kernel, base)
    if refused is not None:
        raise refused
    lowered = base.lowered
    if not lowered.scheduled:
        raise UnsupportedError(
            f"idle replacement: {kernel.__name__}() is scheduled only once a call gives "
            "its arguments' values, so its idle windows differ at every call"
        )
    module = lowered._module
    lines, items = read(module)
    found = gaps(lines, items)
    threshold, chosen = _cycles(rule.threshold, target.clock_ghz), _chosen(rule)
    forward = options.schedule in ("asap", "rcp")
    if rule.fill is None:
        windows = report(found, rule, sites.log, threshold, chosen, forward)
    else:
        planned, windows = plan(found, rule, sites.log, threshold, chosen, forward, _PADS)
        if planned.tiles and base.guard.impure:
            raise UnsupportedError(
                f"idle replacement: {kernel.__name__}() calls a function whose result can "
                "differ each time the kernel is traced (such as random.random()), and "
                "replacing its idle windows traces it twice"
            )
        if planned.tiles:
            try:
                return _filled(
                    kernel, values, target, options, planned, windows, lines, items, sites, lowered
                )
            except _LaneGapError:
                pass
            # Under rcp or alap_rcp: each window's fill kept if, with those
            # kept before it, no operation of the kernel moves; the others
            # are left alone and say why.
            ready = [(w.qudit, w.start, w.end) for w in windows if w.filled]
            kept: list[tuple[int, int, int]] = []
            for window in ready:
                trial = frozenset(w for w in ready if w != window and w not in kept)
                attempt, _ = plan(found, rule, sites.log, threshold, chosen, forward, _PADS, trial)
                try:
                    _filled(
                        kernel,
                        values,
                        target,
                        options,
                        attempt,
                        windows,
                        lines,
                        items,
                        sites,
                        lowered,
                    )
                except _LaneGapError:
                    continue
                kept.append(window)
            lanes = frozenset(w for w in ready if w not in kept)
            planned, windows = plan(
                found, rule, sites.log, threshold, chosen, forward, _PADS, lanes
            )
            if planned.tiles:
                return _filled(
                    kernel, values, target, options, planned, windows, lines, items, sites, lowered
                )
    # Nothing is replaced: the program is the one compiled without the rule.
    module.strip_sites()
    lowered.options = options.without_rule()
    lowered.metrics.idle_windows = windows  # type: ignore[attr-defined]
    return base


# For tests: write each fill as a wait of its window's length.
_PADS = False


def _filled(
    kernel: Kernel[...],
    values: tuple[Any, ...],
    target: Any,
    options: _IdleOptions,
    planned: Plan,
    windows: IdleWindows,
    lines: list[tuple[str, int]],
    items: list[Any],
    sites: Sites,
    plain: Any,
) -> _Entry:
    """*kernel* compiled again for *values* with *planned* written into
    it, checked against *plain*, the program without: *lines* and *items*
    are what was read of that one, *sites* its trace's record."""
    from ._idle_windows import Sites, prove
    from .kernel._guard import Guard, build_guard

    rule = options.idle_replacement
    assert rule is not None and rule.fill is not None
    from .controller import ControllerLimitError

    try:
        entry = kernel._specialize(values, target, options.traced(Sites(planned, rule)))
    except ControllerLimitError as error:
        # The program without the fills fits: the fills' pulses do not, as
        # they count against the control system like any pulse. The error
        # keeps its type and what it lists, and says so.
        fix = (
            "fill fewer qudits (qudits=), use a fill with fewer pulses, or raise the "
            "control system's limit"
        )
        error.args = (
            f"idle replacement: with the idle windows of {kernel.__name__}() filled by "
            f"{rule.fill.__name__}(), {error.args[0]}; {fix}",
            *error.args[1:],
        )
        error.fix = fix
        raise
    lowered = entry.lowered
    module = lowered._module
    try:
        prove(lines, items, module, sites.log, kernel.__name__, options.schedule in ("asap", "rcp"))
    except CompilationError as error:
        if error.code == "timing" and options.schedule in ("rcp", "alap_rcp"):
            # Not an error (decision 280): specialize() leaves the windows
            # whose fills move another operation unfilled.
            raise _LaneGapError() from None
        raise
    if lowered.duration != plain.duration:
        raise CompilationError(
            f"idle replacement: with the idle windows of {kernel.__name__}() filled, the "
            f"program lasts {lowered.duration} instead of {plain.duration}",
            code="timing",
        )
    module.strip_sites()
    lowered.options = options.without_rule()
    lowered.metrics.idle_windows = windows  # type: ignore[attr-defined]
    # The program is valid while the kernel's reads and the fill's hold.
    host, fill = entry.guard, build_guard(rule.fill.__pulse_kernel_fn__)
    entry.guard = Guard(
        lambda: host.check() and fill.check(), host.reads + fill.reads, host.impure, (), 0
    )
    return entry
