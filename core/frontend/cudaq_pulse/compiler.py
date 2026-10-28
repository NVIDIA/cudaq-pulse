# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Lowering a traced kernel to a scheduled MLIR module, and its inspection.

``cudaq_pulse.compile(k)`` returns a callable ``CompiledKernel`` (see
``kernel/_kernel.py``); this module holds what it is built from: the options
that pick passes and a schedule, the native build of one specialization, and
``Lowered``, the read-only view of one compiled module.
"""

from __future__ import annotations

import difflib
import math
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from ._errors import InvalidValueError
from ._native import PulseModuleBuilder
from .classical import Link
from .kernel._bytecode_normalize import UnsupportedInterpreterError
from .kernel._schedule_blocks import check_root_schedule, root_schedule
from .kernel._unroll import resolve_unroll_threshold
from .kernel.ir_builder import CompilationError, SymbolicValueError

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ._device import Device
    from .controller import ControllerLimits
    from .kernel._kernel import CompiledKernel, Kernel

__all__ = [
    "DEFAULT_PASSES",
    "CompileMetrics",
    "Lowered",
    "MachineModel",
    "Pass",
    "ScheduledEvent",
    "UnsupportedInterpreterError",
    "compile",
]


_VALID_SCHEDULES = frozenset({"asap", "alap", "rcp", "alap_rcp"})


class Pass(StrEnum):
    """An optimization pass of ``pulse.compile(k, passes=)``.

    Each member is a string, and the plain string (``"fold_phases"``) is
    accepted wherever a member is. Passes run in the order given, any subset
    in any order; the pattern passes (all but ``VERIFY`` and
    ``HOIST_LOOP_INVARIANTS``) that follow one another run as one rewrite of
    the program, and each of them also removes the operations its rewrites
    leave unused.
    """

    #: Check waveform durations and that each line is used once at each step.
    VERIFY = "verify"
    #: Compute constant arithmetic at compile time and merge equal constants.
    FOLD_CONSTANTS = "fold_constants"
    #: Remove operations whose results nothing uses, unreachable blocks and
    #: unused block arguments.
    REMOVE_UNUSED = "remove_unused"
    #: Join constant waits in a row on one line into one wait.
    MERGE_WAITS = "merge_waits"
    #: Remove syncs that order nothing, inline schedule blocks that take no
    #: time, branch on ``is_set``, and inline or remove loops that run once
    #: or never.
    SIMPLIFY_CONTROL_FLOW = "simplify_control_flow"
    #: Fold constant frame changes (``shift_phase``) into the next drive.
    FOLD_PHASES = "fold_phases"
    #: Merge back-to-back square-pulse drives of the same amplitude into one.
    FUSE_PULSES = "fuse_pulses"
    #: Move operations that compute the same value in every iteration out of
    #: loops.
    HOIST_LOOP_INVARIANTS = "hoist_loop_invariants"


# The cudaq-pulse-opt pass each member runs.
_PASS_MAP = {
    Pass.VERIFY: "pulse-verify",
    Pass.FOLD_CONSTANTS: "pulse-fold-constants",
    Pass.REMOVE_UNUSED: "pulse-remove-unused",
    Pass.MERGE_WAITS: "pulse-merge-waits",
    Pass.SIMPLIFY_CONTROL_FLOW: "pulse-simplify-control-flow",
    Pass.FOLD_PHASES: "pulse-fold-phases",
    Pass.FUSE_PULSES: "pulse-fuse-pulses",
    Pass.HOIST_LOOP_INVARIANTS: "pulse-hoist-loop-invariants",
}

DEFAULT_PASSES: tuple[Pass, ...] = (
    Pass.VERIFY,
    Pass.FOLD_CONSTANTS,
    Pass.REMOVE_UNUSED,
    Pass.FOLD_PHASES,
    Pass.FUSE_PULSES,
)


def _passes(passes: Sequence[str]) -> tuple[Pass, ...]:
    """*passes* as members; an InvalidValueError names any that is not one."""
    if isinstance(passes, str):
        raise InvalidValueError(
            f"passes must be a sequence of pass names, not the string {passes!r}"
        )
    members = []
    names = [p.value for p in Pass]
    for name in passes:
        try:
            members.append(Pass(name))
        except ValueError:
            close = difflib.get_close_matches(str(name), names, n=1)
            hint = f" Did you mean {close[0]!r}?" if close else ""
            raise InvalidValueError(
                f"Unknown pass {name!r}.{hint} The passes are: {', '.join(names)}"
            ) from None
    return tuple(members)


_STAGES = ("pulse", "llvm")


class _Links(dict[int, Link]):
    """The links of a machine model: a read-only, hashable mapping by id,
    which keeps the names the model gave them (``names``), so that a model
    made from another's links (``dataclasses.replace``) keeps them too."""

    names: Mapping[str, int] = MappingProxyType({})

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(tuple(sorted(self.items(), key=lambda item: item[0])))

    def _readonly(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("MachineModel.links is read-only")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _readonly  # type: ignore[assignment]


@dataclass(frozen=True)
class MachineModel:
    """Hardware resource limits.

    The drive and readout limits, readout latency and switch penalty apply
    to the ``rcp`` and ``alap_rcp`` schedules. ``links`` describes the links
    to classical devices (see ``cudaq_pulse.classical.Link``), keyed by name
    (``"decoder"``, ``"controller"``) or by link id; a named link gets the
    lowest id no other link has, in the order given, so the first named
    link of a model without ids is link 0, the default. A classical
    function's ``link=`` names one or gives its id. Their slots and
    latencies apply to every schedule, because they are hardware. A link id
    a program uses that is not listed gets ``Link()``; an unknown name is an
    error. ``controller`` gives the limits of the controller the program is
    for (``cudaq_pulse.controller.ControllerLimits``); every scheduled
    program is checked against them.
    """

    max_concurrent_drives: int = 4
    max_concurrent_readouts: int = 2
    readout_latency_cycles: int = 0
    line_switch_penalty_cycles: int = 0
    links: Mapping[int | str, Link] = field(default_factory=dict)
    controller: ControllerLimits | None = None
    link_names: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.controller is not None:
            from .controller import ControllerLimits

            if not isinstance(self.controller, ControllerLimits):
                raise TypeError(
                    f"MachineModel.controller must be a ControllerLimits, got {self.controller!r}"
                )
        if isinstance(self.links, _Links):
            # Another model's links, as dataclasses.replace passes them: by
            # id, with their names beside them.
            named = set(self.links.names.values())
            given: dict[int | str, Link] = {
                key: value for key, value in self.links.items() if key not in named
            }
            for name, link_id in sorted(self.links.names.items(), key=lambda item: item[1]):
                given[name] = self.links[link_id]
        else:
            given = dict(self.links)
        links = _Links()
        names: list[str] = []
        for key, value in given.items():
            if isinstance(key, str):
                if not key.isidentifier():
                    raise ValueError(f"MachineModel.links names are identifiers, got {key!r}")
                names.append(key)
            elif isinstance(key, bool) or not isinstance(key, int) or key < 0:
                raise ValueError(f"MachineModel.links keys are names or link ids >= 0, got {key!r}")
            if not isinstance(value, Link):
                raise TypeError(f"MachineModel.links[{key!r}] must be a Link, got {value!r}")
            if isinstance(key, int):
                dict.__setitem__(links, key, value)
        by_name: dict[str, int] = {}
        next_id = 0
        for name in names:
            while next_id in links:
                next_id += 1
            dict.__setitem__(links, next_id, given[name])
            by_name[name] = next_id
        links.names = MappingProxyType(by_name)
        object.__setattr__(self, "links", links)
        object.__setattr__(self, "link_names", links.names)
        object.__setattr__(
            self,
            "_hash",
            hash(
                (
                    self.max_concurrent_drives,
                    self.max_concurrent_readouts,
                    self.readout_latency_cycles,
                    self.line_switch_penalty_cycles,
                    self.links,
                    tuple(sorted(by_name.items())),
                    self.controller,
                )
            ),
        )

    def __hash__(self) -> int:
        return self._hash  # type: ignore[no-any-return,attr-defined]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MachineModel):
            return NotImplemented
        return (
            self.max_concurrent_drives,
            self.max_concurrent_readouts,
            self.readout_latency_cycles,
            self.line_switch_penalty_cycles,
            dict(self.links),
            dict(self.link_names),
            self.controller,
        ) == (
            other.max_concurrent_drives,
            other.max_concurrent_readouts,
            other.readout_latency_cycles,
            other.line_switch_penalty_cycles,
            dict(other.links),
            dict(other.link_names),
            other.controller,
        )

    def link_id(self, link: int | str) -> int:
        """The id of *link*, a link name of this model or a link id."""
        if isinstance(link, str):
            found = self.link_names.get(link)
            if found is None:
                known = ", ".join(repr(n) for n in self.link_names) or "none"
                raise CompilationError(
                    f"no link named {link!r} in the machine model (its named links: {known}); "
                    "add it to MachineModel(links={...})"
                )
            return found
        return link

    def link_name(self, link_id: int) -> str | None:
        """The name of the link with id *link_id*, if it has one."""
        for name, found in self.link_names.items():
            if found == link_id:
                return name
        return None

    def link(self, link: int | str) -> Link:
        """The ``Link`` of *link* (a name or an id; ``Link()`` if unlisted)."""
        return self.links.get(self.link_id(link), Link())

    def links_attribute(self, used: Sequence[int]) -> str:
        """The module attribute ``classical.links`` for the links in *used*."""
        entries = ", ".join(
            f'"{link}" = {self.links.get(link, Link()).attribute()}' for link in sorted(set(used))
        )
        return f"{{classical.links = {{{entries}}}}}"


@dataclass(frozen=True)
class ScheduledEvent:
    """One timed operation of a compiled kernel.

    ``line`` numbers the kernel's drive and readout lines in allocation order.
    ``start_cycles`` and ``duration_cycles`` are ``None`` until the kernel is
    scheduled (a kernel with float parameters is scheduled when it is called
    with their values). ``start`` and ``end`` are the scheduled times in
    clock cycles, and raise ``ValueError`` before then.

    A call to a classical function adds a ``"submit"`` event, while the
    request holds its link, and a zero-length ``"answer"`` event, when the
    answer is taken. For those, ``line`` is the link id and ``qubit`` is -1.
    """

    kind: str
    line: int
    qubit: int
    start_cycles: int | None
    duration_cycles: int | None

    @property
    def start(self) -> int:
        if self.start_cycles is None:
            raise ValueError(f"{self.kind} on line {self.line} is not scheduled")
        return self.start_cycles

    @property
    def end(self) -> int:
        if self.duration_cycles is None:
            raise ValueError(f"{self.kind} on line {self.line} is not scheduled")
        return self.start + self.duration_cycles

    @property
    def link(self) -> int | None:
        """The link of a classical call's event, else ``None``."""
        return self.line if self.kind in ("submit", "answer") else None


@dataclass(frozen=True)
class Epoch:
    """One epoch of a compiled kernel, from ``Lowered.epochs()``.

    A variable wait (its length computed from a classical answer) or a
    blocking classical wait is a barrier for the lines that depend on it:
    those it waits on or whose pulses use its answer, and every line tied to
    them by a ``sync``, a shared tone, a branch or a kernel with its own
    ``schedule=``. They sync before it, and what follows on them shifts with
    its run-time length; other lines keep their time. Epoch 0 starts the
    program; epoch k starts when the k-th barrier, in program order, ends.
    ``start_cycles`` is the epoch's start in the static schedule, which places
    every barrier at its lowest length. At run time the epoch starts
    ``shift_cycles`` later, with ``min_shift_cycles <= shift_cycles <=
    max_shift_cycles`` (``None``: unbounded); the bound adds up the barriers
    before it on the same lines. ``opened_by`` is ``"start"``, ``"wait"`` (a
    variable wait on line ``line``) or ``"answer"`` (a blocking wait on link
    ``line``).
    """

    index: int
    opened_by: str
    line: int
    start_cycles: int
    min_shift_cycles: int
    max_shift_cycles: int | None


@dataclass(frozen=True)
class ClassicalCall:
    """One execution of a call to a classical function, from ``classical_report()``.

    Times are in clock cycles. The request is sent from
    ``submit_cycles`` for ``submit_duration_cycles``; its answer is due at
    ``due_cycles`` (the budgeted latency after that) and taken at ``taken_cycles``
    (``None`` if the kernel never takes it). ``first_use_cycles`` is the
    earliest start of a pulse whose amplitude, tone or branch comes from the
    answer, and ``deadline_cycles`` the latest due time the function's
    ``deadline_ns`` allows. ``epoch`` is the epoch it is submitted in.
    ``link_name`` is the link's name in the machine model, if it has one.

    ``exposed_cycles`` is how long the first use waits for the answer beyond
    when it could start without the call: after its line is free and after
    the readout resonator of the last readout before the call has depleted
    (``Qubit.depletion_ns``), when pulses must wait anyway. The rest of the
    budget, ``hidden_cycles``, overlaps that work. A latency within the
    depletion time is hidden entirely.
    """

    function: str
    link: int
    iteration: tuple[int, ...]
    submit_cycles: int
    submit_duration_cycles: int
    due_cycles: int
    taken_cycles: int | None
    first_use_cycles: int | None
    deadline_cycles: int | None
    epoch: int
    link_name: str | None = None
    exposed_cycles: int | None = None

    @property
    def hidden_cycles(self) -> int | None:
        """The part of the budget that overlaps other work."""
        if self.exposed_cycles is None:
            return None
        return max(0, self.due_cycles - self.submit_cycles - self.exposed_cycles)

    @property
    def slack_cycles(self) -> int | None:
        """How long the answer could arrive later without delaying a pulse."""
        return None if self.first_use_cycles is None else self.first_use_cycles - self.due_cycles

    @property
    def deadline_slack_cycles(self) -> int | None:
        """How far the due time is inside the deadline (negative: missed)."""
        return None if self.deadline_cycles is None else self.deadline_cycles - self.due_cycles


@dataclass()
class CompileMetrics:
    """Per-stage timing of one compilation (milliseconds) and its size."""

    trace_ms: float = 0.0
    ffi_ms: float = 0.0
    passes_ms: float = 0.0
    schedule_ms: float = 0.0
    total_ms: float = 0.0
    op_count: int = 0
    target: str = ""


@dataclass(frozen=True)
class CompileOptions:
    """Passes, schedule policy, machine model and unroll threshold of a
    compilation. An ``unroll_threshold`` of None is resolved when a call is
    specialized: from ``CUDAQ_PULSE_UNROLL_THRESHOLD``, else the default."""

    schedule: str = "alap"
    passes: tuple[Pass, ...] = DEFAULT_PASSES
    machine: MachineModel = MachineModel()
    flatten_schedules: bool = False
    unroll_threshold: int | None = None

    @staticmethod
    def make(
        schedule: str = "alap",
        passes: Sequence[str] | None = None,
        machine: MachineModel | None = None,
        flatten_schedules: bool = False,
        unroll_threshold: int | None = None,
    ) -> CompileOptions:
        if schedule not in _VALID_SCHEDULES:
            raise ValueError(
                f"Unknown schedule policy {schedule!r}. "
                f"Choose from: {', '.join(sorted(_VALID_SCHEDULES))}"
            )
        passes = DEFAULT_PASSES if passes is None else _passes(passes)
        machine = machine or MachineModel()
        if machine.max_concurrent_drives <= 0:
            raise ValueError("machine.max_concurrent_drives must be positive")
        if machine.max_concurrent_readouts <= 0:
            raise ValueError("machine.max_concurrent_readouts must be positive")
        if machine.readout_latency_cycles < 0:
            raise ValueError("machine.readout_latency_cycles cannot be negative")
        if machine.line_switch_penalty_cycles < 0:
            raise ValueError("machine.line_switch_penalty_cycles cannot be negative")
        if unroll_threshold is not None:
            unroll_threshold = resolve_unroll_threshold(unroll_threshold)
        return CompileOptions(schedule, passes, machine, bool(flatten_schedules), unroll_threshold)

    def schedule_args(self) -> tuple[str, int, int, int, int]:
        m = self.machine
        return (
            self.schedule,
            m.max_concurrent_drives,
            m.max_concurrent_readouts,
            int(m.readout_latency_cycles),
            int(m.line_switch_penalty_cycles),
        )


# Longest wait a call may ask for: lengths stay exact in a float.
_MAX_WAIT_CYCLES = 2**53


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

    def __init__(self, waits: Sequence[tuple[Any, ...]], kernel: str) -> None:
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
        self.ok = eval(f"lambda a: {tests}", {"__builtins__": {}})

    def check(self, values: Sequence[float]) -> None:
        try:
            if self.ok(values):
                return
        except ZeroDivisionError:
            pass
        for wait in self.waits:
            wait.check(values)


class Lowered:
    """One compiled module of a kernel: read-only text, schedule and metrics.

    ``kernel.lower(*args)`` and ``compiled.lower(*args)`` return the module
    of one call. Float parameters stay function arguments of the module, and
    the call's values are kept beside it in ``arguments``, so a call with new
    values shares the module; ``generic`` is the module without values.
    ``mlir()`` of a call shows its values substituted.

    A wait whose length is computed from float parameters keeps the module
    generic too: without values it is scheduled at its lowest length, and a
    call's schedule, text and simulation come from the module with the call's
    values bound and scheduled again.

    ``qudits`` holds the device qudits the program drives or measures (gets
    a line on), in increasing order: the qudits ``simulate()`` models.
    """

    __slots__ = (
        "_module",
        "_text",
        "_lock",
        "_concrete_module",
        "target",
        "metrics",
        "parameters",
        "arguments",
        "qudits",
        "options",
        "scheduled",
        "generic",
        "waits",
        "classical",
    )

    def __init__(
        self,
        module: Any,
        *,
        target: Device,
        metrics: CompileMetrics,
        parameters: tuple[str, ...],
        qudits: tuple[int, ...],
        options: CompileOptions,
        scheduled: bool = True,
        waits: _WaitLengths | None = None,
    ) -> None:
        self._module = module
        self._text: str | None = None
        self._lock = threading.Lock()
        self._concrete_module: Any = None
        self.target = target
        self.metrics = metrics
        self.parameters = parameters
        self.arguments: tuple[float, ...] = ()
        self.qudits = qudits
        self.options = options
        self.scheduled = scheduled
        self.generic: Lowered | None = None
        self.waits = waits
        # The classical functions the program calls, in declaration order.
        self.classical: tuple[Any, ...] = ()

    def bind(self, values: Sequence[float]) -> Lowered:
        """This module with values for its float parameters, sharing it.

        Raises ``ValueError`` if a wait length computed from them is not a
        finite, non-negative number of cycles within the wait's bounds."""
        waits = self.waits
        if waits is not None:
            waits.check(values)
        view = Lowered.__new__(Lowered)
        view._module = self._module
        view._text = None
        view._lock = self._lock
        view._concrete_module = None
        view.target = self.target
        view.metrics = self.metrics
        view.parameters = ()
        view.arguments = tuple(values)
        view.qudits = self.qudits
        view.options = self.options
        view.scheduled = self.scheduled
        view.generic = self
        view.waits = waits
        view.classical = self.classical
        return view

    @property
    def _retimed(self) -> bool:
        """Whether the call's values move ops: its times come from the module
        with them bound."""
        return not self.scheduled or (self.waits is not None and bool(self.arguments))

    def _concrete(self) -> Any:
        """The module with the values substituted (a private copy), or the
        module itself when there are no values."""
        if not self.arguments:
            return self._module
        if self._concrete_module is None:
            with self._lock:
                module = self._module.specialize(
                    list(self.arguments), [], *self.options.schedule_args()
                )
            self._concrete_module = module
        return self._concrete_module

    def mlir(self, stage: str = "pulse") -> str:
        """The module as MLIR text: ``"pulse"`` (scheduled pulse dialect) or
        ``"llvm"`` (after the full pulse -> qop -> cudm -> LLVM lowering)."""
        if stage == "pulse":
            if self._text is None:
                module = self._concrete()
                with self._lock:
                    self._text = module.print()
            return self._text
        if stage == "llvm":
            if self.parameters:
                raise ValueError(
                    "the module has float parameters "
                    f"({', '.join(self.parameters)}); lower it with their values first"
                )
            module = self._concrete()
            with self._lock:
                text: str = module.run_full_lowering()
            return text
        raise ValueError(f"unknown stage {stage!r}; choose from {', '.join(_STAGES)}")

    @property
    def schedule(self) -> list[ScheduledEvent]:
        """The drives, readouts and waits with their timing.

        Ops in a loop are listed once per iteration; both arms of a
        measurement-conditioned branch are listed. Without values, a wait
        whose length a float parameter sets is listed at its lowest length.
        """
        module = self._concrete() if self._retimed else self._module
        with self._lock:
            events = module.events()
        return [ScheduledEvent(*event) for event in events]

    def epochs(self) -> list[Epoch]:
        """The epochs of the program, separated by variable and blocking waits.

        A program without such waits has one epoch.
        """
        return self._epochs()[0]

    def _epochs(self) -> tuple[list[Epoch], list[int | None]]:
        """The epochs, and the epoch each one follows (``None`` for the
        first): the one its barrier ends, on the lines the barrier holds
        back. Its shift adds to that epoch's."""
        module = self._module if self.scheduled else self._concrete()
        with self._lock:
            boundaries = module.epoch_boundaries()
        epochs = [Epoch(0, "start", -1, 0, 0, 0)]
        after: list[int | None] = [None]
        for k, (kind, line, start, lowest, highest, before) in enumerate(boundaries, start=1):
            most = epochs[before].max_shift_cycles
            extra = None if most is None or highest < 0 else most + highest - lowest
            epochs.append(Epoch(k, kind, line, start + lowest, 0, extra))
            after.append(before)
        return epochs, after

    def _event_epochs(self, syncs: bool) -> list[int]:
        """The epoch of each event of ``schedule`` (with *syncs*, of the
        controller export), in the same order."""
        if not self.classical:
            return []
        module = self._concrete() if syncs or self._retimed else self._module
        with self._lock:
            return list(module.event_epochs(syncs))

    def classical_report(self) -> list[ClassicalCall]:
        """Every call to a classical function, with its timing and slack.

        Calls in a loop are listed once per iteration, in program order.
        """
        module = self._module if self.scheduled else self._concrete()
        with self._lock:
            calls = module.classical_calls()
            events = module.events() if calls else []
        names = self.options.machine.link_name
        exposed = _Exposure(self, events) if calls else None
        return [
            ClassicalCall(
                name,
                link,
                tuple(iteration),
                start,
                length,
                due,
                taken,
                use,
                limit,
                epoch,
                names(link),
                None if exposed is None or use is None else exposed(start, use),
            )
            for name, link, iteration, start, length, due, taken, use, limit, epoch in calls
        ]

    def export_emulation(self) -> dict[str, Any]:
        """The program for a controller emulator: its schedule by epoch, its
        calls and the manifest of its classical functions (see
        ``cudaq_pulse.classical.emulation``)."""
        from .classical.emulation import export

        return export(self)

    def export_controller(self, sample_rate_ghz: float | None = None) -> dict[str, Any]:
        """The program for an out-of-tree controller backend: its lines and
        every timed operation with its waveform, tone, frame and branches
        (see ``cudaq_pulse.controller``); with *sample_rate_ghz*, each
        envelope's samples too."""
        from .controller import export

        return export(self, sample_rate_ghz)

    def timeline(self, **kwargs: Any) -> Figure:
        """Plot the schedule, one row per line (needs matplotlib)."""
        from .viz.timeline import plot_schedule

        return plot_schedule(self, **kwargs)

    def __repr__(self) -> str:
        params = f", parameters={list(self.parameters)}" if self.parameters else ""
        return f"<Lowered {self.metrics.op_count} ops on {self.target.name}{params}>"


class _Exposure:
    """How long a call's first use waits beyond when it could start without
    the call (see ``ClassicalCall.exposed_cycles``), in O(log n) per call."""

    def __init__(self, lowered: Lowered, events: Sequence[Any]) -> None:
        import bisect

        self._bisect = bisect.bisect_right
        clock = lowered.target.clock_ghz
        qubits = lowered.target.target.qubits
        ends: dict[int, list[int]] = {}
        starting: dict[int, list[int]] = {}
        readouts: list[tuple[int, int]] = []
        holds: dict[int, int] = {}
        for kind, line, qubit, start, duration in events:
            if (kind != "drive" and kind != "readout") or start is None or duration is None:
                continue
            end = start + duration
            ends.setdefault(line, []).append(end)
            starting.setdefault(start, []).append(line)
            if kind == "readout":
                hold = holds.get(qubit)
                if hold is None:
                    q = qubits.get(qubit)
                    hold = math.ceil(q.depletion_ns * clock) if q is not None else 0
                    holds[qubit] = hold
                readouts.append((end, end + hold))
        for values in ends.values():
            values.sort()
        readouts.sort()
        self._ends = ends
        self._starting = starting
        self._readout_ends = [end for end, _ in readouts]
        self._holds = [hold for _, hold in readouts]

    def __call__(self, submit_cycles: int, use_cycles: int) -> int | None:
        lines = self._starting.get(use_cycles)
        if not lines:
            return None
        free = 0
        for line in lines:
            ends = self._ends[line]
            k = self._bisect(ends, use_cycles)
            if k:
                free = max(free, ends[k - 1])
        k = self._bisect(self._readout_ends, submit_cycles)
        if k:
            free = max(free, self._holds[k - 1])
        return max(0, use_cycles - max(free, submit_cycles))


def build(
    fn: Any,
    trace_args: Sequence[Any],
    parameters: tuple[str, ...],
    target: Device,
    options: CompileOptions,
) -> tuple[Lowered, Any]:
    """Trace *fn* and build its module on *target*.

    *trace_args* holds a ``Parameter`` sentinel for every float parameter
    named in *parameters*; those stay function arguments of the module.
    Returns the module and the builder, whose record of readouts and host
    calls the caller needs. Raises ``SymbolicValueError`` naming the float
    parameters an integer of the program (a waveform duration, a cast to
    int) was computed from: the module needs their values.
    """
    from .kernel.bytecode_bridge import _trace_kernel_with_builder
    from .kernel.packed_ir_builder import PackedIRBuilder

    metrics = CompileMetrics(target=target.name)
    name = getattr(fn, "__name__", "kernel")
    t0 = time.perf_counter()
    builder = PackedIRBuilder(
        name=name,
        clock_ghz=target.clock_ghz,
        qubit_freq_hz=dict(target.frequencies),
        flatten_schedules=options.flatten_schedules,
    )
    builder.unroll_threshold = options.unroll_threshold
    builder.link_id = options.machine.link_id
    builder.returned = _trace_kernel_with_builder(fn, builder, trace_args)
    needed = tuple(sorted(builder.structural & set(parameters)))
    if needed:
        raise SymbolicValueError(
            f"{name}(): {', '.join(needed)} must be known while the kernel compiles",
            needed,
            "a value that must be an integer, such as a waveform duration",
        )
    check_root_schedule(options.schedule, builder)
    qudits = builder.qubits
    missing = [q for q in qudits if q not in target.frequencies]
    if missing:
        defined = sorted(target.frequencies)
        span = f"{defined[0]}..{defined[-1]}" if defined else "none"
        raise CompilationError(
            f"{name}() uses qudit {', '.join(map(str, missing))}, which target "
            f"{target.name!r} does not define (it defines qudits {span})"
        )
    buf: Any = builder.get_buffer()
    freq: Any = builder.get_freq_array()
    links = [options.machine.link_id(function.link) for function in builder.classical]
    metrics.trace_ms = (time.perf_counter() - t0) * 1000
    metrics.op_count = builder.op_count

    t0 = time.perf_counter()
    if parameters:
        module = PulseModuleBuilder().build_from_packed(
            buf,
            target.clock_ghz,
            builder.n_qubits,
            freq,
            list(parameters),
            ["f64"] * len(parameters),
            name=name,
        )
    else:
        module = PulseModuleBuilder().build_from_packed(
            buf, target.clock_ghz, builder.n_qubits, freq, name=name
        )
    if links:
        module.set_attributes(options.machine.links_attribute(links))
    metrics.ffi_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    cpp_passes = [_PASS_MAP[p] for p in options.passes]
    if cpp_passes:
        module.run_passes(cpp_passes)
    metrics.passes_ms = (time.perf_counter() - t0) * 1000

    # Waveform durations never depend on a float parameter (a float that
    # reaches one is specialized on); a wait whose length does is scheduled at
    # its lowest length, and again with each call's values.
    t0 = time.perf_counter()
    scheduled = True
    try:
        module.schedule(*options.schedule_args())
    except Exception:
        if not parameters:
            raise
        scheduled = False
    metrics.schedule_ms = (time.perf_counter() - t0) * 1000
    metrics.total_ms = metrics.trace_ms + metrics.ffi_ms + metrics.passes_ms + metrics.schedule_ms

    lowered = Lowered(
        module,
        target=target,
        metrics=metrics,
        parameters=parameters,
        qudits=builder.used_qubits or qudits,
        options=options,
        scheduled=scheduled,
        waits=_WaitLengths(builder.variable_waits, name) if builder.variable_waits else None,
    )
    lowered.classical = tuple(builder.classical)
    if options.machine.controller is not None and scheduled:
        from .controller import ControllerLimitError, check

        violations = check(lowered)
        if violations:
            raise ControllerLimitError(violations)
    return lowered, builder


def compile[**P](
    kernel: Kernel[P],
    /,
    *,
    schedule: str | None = None,
    passes: Sequence[str] | None = None,
    machine: MachineModel | None = None,
    device: Device | None = None,
    flatten_schedules: bool = False,
    unroll_threshold: int | None = None,
) -> CompiledKernel[P]:
    """Compile a ``@cudaq_pulse.kernel`` without its arguments.

    Returns a ``CompiledKernel``, a family of specializations. Calling it
    with the kernel's arguments runs the kernel on the device of the qudit
    handles passed and returns what the kernel returns. Float parameters
    stay symbolic: the module takes them as function arguments, so calls
    with new values never recompile. Parameters of any other type (qudits,
    ints) shape the program; each distinct value, on each device, is
    compiled on first use and kept for the life of the compiled object.

    Parameters
    ----------
    kernel:
        A ``@cudaq_pulse.kernel``.
    schedule:
        Scheduling policy of *kernel* itself: ``"asap"``, ``"alap"``,
        ``"rcp"``, ``"alap_rcp"``; by default the kernel's own
        ``schedule=``, else ``"alap"``. Kernels it calls keep their own.
    passes:
        Optimization passes (``Pass`` members or their names), run in the
        order given; ``DEFAULT_PASSES`` by default, ``()`` for none.
    machine:
        Machine model for resource-constrained scheduling.
    flatten_schedules:
        Ignore the ``schedule=`` of every called kernel and schedule the whole
        program with *schedule*.
    unroll_threshold:
        A ``for ... in range(n)`` loop whose variable the body never reads is
        unrolled while tracing when n times the number of calls in its body
        is at most this, and emitted once as a rolled loop otherwise. By
        default the kernel's own ``unroll_threshold=``, else the
        ``CUDAQ_PULSE_UNROLL_THRESHOLD`` environment variable, else
        ``DEFAULT_UNROLL_THRESHOLD``; 0 rolls every such loop. Either way the
        program, its schedule and its simulation are the same.
    device:
        Pin the compiled kernel to one device: a call with handles of
        another raises ``DeviceMismatchError``, and a call without handles
        runs on it. A pinned kernel without static parameters is built now.
    """
    from .kernel._kernel import CompiledKernel, Kernel

    if not isinstance(kernel, Kernel):
        raise TypeError(f"compile() expects a @cudaq_pulse.kernel, got {type(kernel).__name__}")
    options = CompileOptions.make(
        root_schedule(kernel.schedule, schedule),
        passes,
        machine,
        flatten_schedules,
        kernel.unroll_threshold if unroll_threshold is None else unroll_threshold,
    )
    return CompiledKernel(kernel, options, device)
