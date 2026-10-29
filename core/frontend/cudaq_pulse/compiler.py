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
import marshal
import math
import struct
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction
from types import FunctionType, MappingProxyType
from typing import TYPE_CHECKING, Any

from ._errors import InvalidTypeError, InvalidValueError
from ._native import PackedValueError, PulseModuleBuilder
from ._units import Time, as_nanoseconds
from .classical import Link
from .kernel._bytecode_normalize import UnsupportedInterpreterError
from .kernel._schedule_blocks import check_root_schedule, root_schedule
from .kernel._unroll import DEFAULT_UNROLL_THRESHOLD, resolve_unroll_threshold
from .kernel.ir_builder import MAX_CYCLES, CompilationError, SymbolicValueError

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from . import _clocks
    from ._device import Device
    from ._system import ControlSystem
    from .controller import ControllerLimits, UsageReport
    from .kernel._kernel import CompiledKernel, Kernel

__all__ = [
    "DEFAULT_PASSES",
    "CompileMetrics",
    "IRLevel",
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


class IRLevel(StrEnum):
    """A level of the program's IR, for ``mlir(level)``: each lowers the one
    before it. The plain string (``"qop"``) is accepted wherever a member is.
    """

    #: The scheduled Pulse dialect.
    PULSE = "pulse"
    #: Quantum operators: the program's Hamiltonian and dissipators.
    QOP = "qop"
    #: cuDensityMat operators and evolution.
    CUDM = "cudm"
    #: LLVM dialect, ready to compile.
    LLVM = "llvm"


_LEVELS = tuple(IRLevel)

# The parameters a measurement can select while the program runs, in the
# order lowering asks about them, and those any target takes that way.
_RUNTIME_KINDS = ("amplitude", "phase", "frequency")
_SELECTED_AT_RUNTIME: frozenset[str] = frozenset()


def _can_select_at_runtime(target: Device, parameter_kind: str) -> bool:
    """Whether a program compiled for *target* can take a value of
    *parameter_kind* that a measurement selects while it runs.

    *target* is the device the program is built for. *parameter_kind* is
    ``"amplitude"`` (of a waveform, or the factor of ``wf_scale``),
    ``"phase"`` (of ``shift_phase`` and ``set_phase``) or ``"frequency"``
    (of ``shift_frequency`` and ``set_frequency``). Durations, waveform
    shapes and anything else that moves ops in time are never asked about.

    Where this holds, the measurement selects the value once, every op that
    uses it reads that run-time value, and the schedule starts those ops
    once the measurement is in. Everywhere else, each op that uses the value
    runs in both arms of an if on the measurement, with that arm's value.
    Simulation gives the same result either way. No target takes any kind
    at run time: ``export_controller`` gives a kept value as ``null`` with
    its expression beside it (the select on the measurement), and
    ``export_emulation`` carries no parameters.
    """
    return parameter_kind in _SELECTED_AT_RUNTIME


class _Links(dict[int, Link]):
    """The links of a machine model: a read-only, hashable mapping by id,
    which keeps the names the model gave them (``names``), so that a model
    made from another's links (``dataclasses.replace``) keeps them too."""

    names: Mapping[str, int] = MappingProxyType({})

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(tuple(sorted(self.items(), key=lambda item: item[0])))

    def __reduce__(self) -> tuple[Any, ...]:
        from ._pickling import rebuilt_links

        return (rebuilt_links, (_Links, dict(self), dict(self.names)))

    def _readonly(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("MachineModel.links is read-only")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _readonly  # type: ignore[assignment]


_NO_DELAYS: Mapping[tuple[str, int], float] = MappingProxyType({})


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
    program is checked against them. ``reference_clock_ghz`` is a clock the
    controllers share, if they do: each controller's clock must be a whole
    multiple of it, and without it each controller's start is known only to
    within one of its own cycles. ``line_delays_ns`` gives calibrated
    delays of lines, as times or numbers of ns, keyed by ``("drive", qudit)``
    or ``("readout", qudit)``: every operation on the line starts that much
    later than scheduled. Whole steps of the line's grid move its
    operations, which are then checked like any others; the rest is the
    line's ``fine_delay_ns`` in the controller's program, which its
    controller must be able to add (``ControllerLimits.max_fine_delay_ns``)
    or tolerate
    (``ControllerLimits.alignment_tolerance_cycles``).
    """

    max_concurrent_drives: int = 4
    max_concurrent_readouts: int = 2
    readout_latency_cycles: int = 0
    line_switch_penalty_cycles: int = 0
    links: Mapping[int | str, Link] = field(default_factory=dict)
    controller: ControllerLimits | None = None
    reference_clock_ghz: float | None = None
    line_delays_ns: Mapping[tuple[str, int], float | Time] = field(default_factory=dict)
    link_names: Mapping[str, int] = field(init=False, repr=False, compare=False)
    # The ControlSystem this model was made from, whose wiring and
    # capabilities every program is checked against.
    _system: ControlSystem | None = field(default=None, repr=False)
    _plain: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.controller is not None:
            from .controller import ControllerLimits

            if not isinstance(self.controller, ControllerLimits):
                raise TypeError(
                    f"MachineModel.controller must be a ControllerLimits, got {self.controller!r}"
                )
        if self.reference_clock_ghz is not None and not self.reference_clock_ghz > 0:
            raise InvalidValueError(
                f"MachineModel.reference_clock_ghz must be positive, got {self.reference_clock_ghz}"
            )
        delays: dict[tuple[str, int], float] = {}
        for line, ns in dict(self.line_delays_ns).items() if self.line_delays_ns else ():
            if (
                not isinstance(line, tuple)
                or len(line) != 2
                or line[0] not in ("drive", "readout")
                or isinstance(line[1], bool)
                or not isinstance(line[1], int)
                or line[1] < 0
            ):
                raise InvalidValueError(
                    "MachineModel.line_delays_ns keys are ('drive', qudit) or "
                    f"('readout', qudit), got {line!r}"
                )
            ns = as_nanoseconds(ns)
            if isinstance(ns, bool) or not isinstance(ns, (int, float)) or not math.isfinite(ns):
                raise InvalidValueError(
                    f"MachineModel.line_delays_ns[{line!r}] must be a number, got {ns!r}"
                )
            if ns < 0:
                raise InvalidValueError(
                    f"MachineModel.line_delays_ns[{line!r}] is {ns} ns; delays cannot be "
                    "negative: to play a line earlier, delay every other line instead"
                )
            delays[line] = float(ns)
        object.__setattr__(
            self,
            "line_delays_ns",
            MappingProxyType(dict(sorted(delays.items()))) if delays else _NO_DELAYS,
        )
        # Nothing of clocks to do: no controller, control system, reference
        # clock or delays.
        object.__setattr__(
            self,
            "_plain",
            self.controller is None
            and self._system is None
            and self.reference_clock_ghz is None
            and not delays,
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
                    self.reference_clock_ghz,
                    tuple(self.line_delays_ns.items()),
                    self._system,
                )
            ),
        )

    def __hash__(self) -> int:
        return self._hash  # type: ignore[no-any-return,attr-defined]

    def __reduce__(self) -> tuple[Any, ...]:
        from ._pickling import frozen_reduce

        return frozen_reduce(self)

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
            self.reference_clock_ghz,
            dict(self.line_delays_ns),
            self._system,
        ) == (
            other.max_concurrent_drives,
            other.max_concurrent_readouts,
            other.readout_latency_cycles,
            other.line_switch_penalty_cycles,
            dict(other.links),
            dict(other.link_names),
            other.controller,
            other.reference_clock_ghz,
            dict(other.line_delays_ns),
            other._system,
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


_DEFAULT_MACHINE = MachineModel()


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
    compilation. ``unroll_threshold`` is a non-negative ``int``, 8 by
    default; 0 never unrolls (see ``cudaq_pulse.compile``)."""

    schedule: str = "alap"
    passes: tuple[Pass, ...] = DEFAULT_PASSES
    machine: MachineModel = _DEFAULT_MACHINE
    flatten_schedules: bool = False
    unroll_threshold: int = DEFAULT_UNROLL_THRESHOLD

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
        machine = machine or _DEFAULT_MACHINE
        if machine.max_concurrent_drives <= 0:
            raise ValueError("machine.max_concurrent_drives must be positive")
        if machine.max_concurrent_readouts <= 0:
            raise ValueError("machine.max_concurrent_readouts must be positive")
        if machine.readout_latency_cycles < 0:
            raise ValueError("machine.readout_latency_cycles cannot be negative")
        if machine.line_switch_penalty_cycles < 0:
            raise ValueError("machine.line_switch_penalty_cycles cannot be negative")
        return CompileOptions(
            schedule,
            passes,
            machine,
            bool(flatten_schedules),
            resolve_unroll_threshold(unroll_threshold),
        )

    def schedule_args(self) -> tuple[str, int, int, int, int]:
        m = self.machine
        return (
            self.schedule,
            m.max_concurrent_drives,
            m.max_concurrent_readouts,
            int(m.readout_latency_cycles),
            int(m.line_switch_penalty_cycles),
        )


# Placements whose qudits a role remembers.
_PLACEMENTS = 4096
# Placed modules a role keeps for simulations: each is a copy of the module.
_PLACED_MODULES = 16

# Longest wait a call may ask for: lengths stay exact in a float.
_MAX_WAIT_CYCLES = MAX_CYCLES

# Values whose controller limits were checked, kept per module.
_ACCEPTED_CALLS = 4096
_PACKERS: dict[int, Callable[..., bytes]] = {}


def _bits(values: tuple[float, ...]) -> bytes:
    """*values* as a key, by their bits, so that -0.0 and 0.0, or two NaNs,
    are told apart."""
    try:
        return _PACKERS[len(values)](*values)
    except KeyError:
        pack = _PACKERS[len(values)] = struct.Struct(f"{len(values)}d").pack
        return pack(*values)


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

    # Lengths a controller with a coarser clock than the program's can time.
    grid = 1

    def __init__(self, waits: Sequence[tuple[Any, ...]], kernel: str, grid: int = 1) -> None:
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
        if grid > 1:
            tests += "".join(
                f" and ({w.source} + 0.5) // 1 % {grid} == 0"
                for w in self.waits
                if w.lowest is not None
            )
        self.ok = eval(f"lambda a: {tests}", {"__builtins__": {}})

    def __reduce__(self) -> tuple[Any, ...]:
        # The check as compiled: the stamp of the pickle names the Python
        # whose marshal format wrote it.
        check = marshal.dumps(self.ok.__code__)
        return (_rebuilt_wait_lengths, (type(self), self.waits, self.grid, check))

    def check(self, values: Sequence[float]) -> None:
        try:
            if self.ok(values):
                return
        except ZeroDivisionError:
            pass
        for wait in self.waits:
            wait.check(values)
        for wait in self.waits if self.grid > 1 else ():
            cycles = math.floor(wait.length(values) + 0.5)
            if wait.lowest is not None and cycles % self.grid:
                lower = cycles // self.grid * self.grid
                raise InvalidValueError(
                    f"{wait.kernel}(): the wait length {wait.text} rounds to {cycles} clock "
                    f"cycles; its line moves in timesteps of {self.grid} cycles, and "
                    f"{cycles} cycles isn't a whole number of timesteps: the nearest "
                    f"lengths are {lower} and {lower + self.grid}"
                )


class _GridWaitLengths(_WaitLengths):
    """Variable waits on lines that move in timesteps of *grid* cycles. Its
    own class keeps the grid off the waits of every module on one clock."""

    __slots__ = ("grid",)

    def __init__(self, waits: Sequence[tuple[Any, ...]], kernel: str, grid: int) -> None:
        self.grid = grid
        super().__init__(waits, kernel, grid)


def _wait_lengths(waits: Sequence[tuple[Any, ...]], kernel: str, grid: int) -> _WaitLengths:
    return _WaitLengths(waits, kernel) if grid == 1 else _GridWaitLengths(waits, kernel, grid)


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
    cls: type[_WaitLengths], waits: tuple[_WaitLength, ...], grid: int, check: bytes
) -> _WaitLengths:
    lengths = cls.__new__(cls)
    lengths.waits = waits
    if isinstance(lengths, _GridWaitLengths):
        lengths.grid = grid
    lengths.ok = FunctionType(marshal.loads(check), {"__builtins__": {}})
    return lengths


def _bind_count_error(parameters: tuple[str, ...], values: Any) -> InvalidTypeError:
    """The error for ``bind(values)`` with the wrong number of *values*."""
    try:
        count = len(values)
    except TypeError:
        return InvalidTypeError(
            f"bind() takes a list or tuple of values, one per float parameter, "
            f"got {type(values).__name__}"
        )
    got = f"{count} value" if count == 1 else f"{count} values"
    if not parameters:
        return InvalidTypeError(
            f"bind() got {got} for a module with no float parameters; a module of one "
            "call has its values already"
        )
    names = ", ".join(parameters)
    if count < len(parameters):
        missing = ", ".join(repr(p) for p in parameters[count:])
        detail = f"missing the value of {missing}"
    else:
        detail = f"{count - len(parameters)} more than it has parameters"
    return InvalidTypeError(
        f"bind() got {got} for {len(parameters)} float parameters ({names}); {detail}"
    )


def _bind_checked(parameters: tuple[str, ...], values: Any) -> list[float]:
    """*values* as floats, one per parameter, or the error for their count or
    naming the first parameter whose value is not a real number."""
    try:
        fits = len(values) == len(parameters)
    except TypeError:
        fits = False
    if not fits:
        raise _bind_count_error(parameters, values)
    out = []
    for name, v in zip(parameters, values):
        if isinstance(v, (bool, str, bytes)) or not hasattr(v, "__float__"):
            raise InvalidTypeError(
                f"bind(): the value of {name!r} must be a real number, got {type(v).__name__} {v!r}"
            )
        if isinstance(v, complex):
            raise InvalidTypeError(
                f"bind(): the value of {name!r} must be a real number, got complex {v!r}"
            )
        out.append(float(v))
    return out


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

    A module pickles as MLIR bytecode with its device and options, and runs
    unpickled without compiling again, in any process of the build that
    compiled it; another build raises ``UnsupportedError``. A call's view
    pickles its values with its generic module.

    Qudit parameters annotated ``Qudit`` or ``Register`` are runtime qudits:
    the module takes them as slots, numbered in parameter order, and a call
    binds each slot to its handle's carrier (``carriers``), on the call's
    device. Calls with other handles of the same shape share the module.

    The module with a call's values substituted is built once per distinct
    values and shared by every view with them, so exporting or lowering the
    same call again does not build it again. The latest few are kept per
    module, fewer for a large one.
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
        "_generic",
        "waits",
        "classical",
        "runtime_selects",
        "_arity",
    )

    # Its controllers' clocks, when one is not the program clock or they
    # share a reference clock; a call's view reads its generic module's.
    clocks: _clocks.ClockPlan | None = None
    # Each line's fine delay in ns, when the machine has line delays; a
    # call's view reads its generic module's.
    fine_delays: Mapping[tuple[str, int], float] | None = None
    # How many values bind takes: set by a generic module's first bind.
    _arity: int
    # A module with list or qudit parameters is a _Placed: what its calls
    # share, and the carrier of each runtime qudit of a call, in slot order.
    _role: _Role | None = None
    carriers: tuple[int, ...] = ()

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
        # The module a call's view shares; None for that module itself.
        self._generic: Lowered | None = None
        self.waits = waits
        # The classical functions the program calls, in declaration order.
        self.classical: tuple[Any, ...] = ()
        # Whether ops read values measurements select while the program runs.
        self.runtime_selects = False

    def bind(self, values: Sequence[float]) -> Lowered:
        """This module with values for its float parameters, sharing it.

        *values* holds one real number per entry of ``parameters``, in that
        order. Too few or too many values are an ``InvalidTypeError`` naming
        the parameters, and a wait length computed from the values that is not
        a non-negative number of cycles within the wait's bounds a
        ``ValueError``. A value that is not a real number (a bool or a str) is
        an ``InvalidTypeError``, and a NaN or infinity an
        ``InvalidValueError``, each naming the parameter, where the module is
        used: run, simulated, exported or printed."""
        try:
            arity = self._arity
            arguments = tuple(values)
        except (TypeError, AttributeError):
            # How many values a generic module takes is kept by its first
            # bind; a call's view keeps none and takes none.
            if self._generic is None:
                self._arity = len(self.parameters)
            return self._with_values(_bind_checked(self.parameters, values))
        if len(arguments) != arity:
            raise _bind_count_error(self.parameters, arguments)
        # The view is made here as in _with_values: a call to it would be a
        # third of a bind.
        waits = self.waits
        if waits is not None:
            try:
                waits.check(arguments)
            except TypeError:
                _bind_checked(self.parameters, arguments)
                raise
        view = _View()
        view._module = self._module
        view._text = None
        view._lock = self._lock
        view._concrete_module = None
        view.target = self.target
        view.metrics = self.metrics
        view.parameters = ()
        view.arguments = arguments
        view.qudits = self.qudits
        view.options = self.options
        view.scheduled = self.scheduled
        view._generic = self
        view.waits = waits
        view.classical = self.classical
        view.runtime_selects = self.runtime_selects
        return view

    def _with_values(self, values: Sequence[float]) -> Lowered:
        """``bind`` for *values* known to be one real number per parameter,
        such as a kernel call's. Whether each is a finite float is checked
        where the module uses them (``_concrete`` and simulation)."""
        waits = self.waits
        if waits is not None:
            waits.check(values)
        view = _View()
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
        view._generic = self
        view.waits = waits
        view.classical = self.classical
        view.runtime_selects = self.runtime_selects
        return view

    @property
    def generic(self) -> Lowered:
        """The module without values: the one a call's view shares, or this
        module when it holds no values. ``generic.bind(values)`` makes a call
        of it with new values."""
        generic = self._generic
        return self if generic is None else generic

    @property
    def _retimed(self) -> bool:
        """Whether the call's values move ops: its times come from the module
        with them bound."""
        return not self.scheduled or (self.waits is not None and bool(self.arguments))

    def _unbound(self) -> bool:
        """Whether the module has runtime qudits and no carriers for them."""
        return self._role is not None and self._role.slots > 0 and not self.carriers

    def _concrete(self) -> Any:
        """The module with the values and carriers substituted, shared by
        every view with the same ones, or the module itself without them."""
        if not self.arguments and not self.carriers:
            return self._module
        if self._concrete_module is None:
            arguments: Sequence[float] = self.arguments
            for v in arguments:
                if v.__class__ is not float and v.__class__ is not int or v - v != 0.0:
                    arguments = self._checked_arguments()
                    break
            with self._lock:
                if self.carriers:
                    module = self._module.specialize_placed(
                        list(arguments),
                        [],
                        *self.options.schedule_args(),
                        list(self.carriers),
                        self.target._frequency_table(),
                        self.target.fingerprint,
                    )
                else:
                    module = self._module.specialize(
                        list(arguments), [], *self.options.schedule_args()
                    )
            self._concrete_module = module
        return self._concrete_module

    def _checked_arguments(self) -> list[float]:
        """The call's values as floats, or the error naming the first that is
        not a real number or not finite."""
        generic = self._generic
        names = generic.parameters if generic is not None else ()
        values = _bind_checked(names, self.arguments)
        for name, v in zip(names, values):
            if v - v != 0.0:
                raise InvalidValueError(
                    f"argument {name!r} is {v!r}; a float argument must be a finite number"
                )
        return values

    def mlir(self, level: IRLevel | str = IRLevel.PULSE) -> str:
        """The module as MLIR text at *level* (an ``IRLevel`` or its name):
        ``"pulse"``, the scheduled Pulse dialect; ``"qop"``, ``"cudm"`` and
        ``"llvm"``, after lowering it that far (pulse -> qop -> cudm ->
        LLVM). Every level but ``"pulse"`` needs the float parameters' values,
        and a program that measures lowers only along a measurement-outcome
        path, as ``simulate`` does."""
        if level == "pulse":
            if self._text is None:
                module = self._concrete()
                with self._lock:
                    self._text = module.print()
            return self._text
        if level not in _LEVELS:
            raise InvalidValueError(f"unknown IR level {level!r}; choose from {', '.join(_LEVELS)}")
        if self.parameters:
            raise ValueError(
                "the module has float parameters "
                f"({', '.join(self.parameters)}); lower it with their values first"
            )
        if self._unbound():
            raise InvalidValueError("the module has runtime qudits; lower it with handles first")
        if level == "llvm":
            module = self._concrete()
            with self._lock:
                text: str = module.run_full_lowering()
            return text
        from ._native._cudaq_pulse_native import MLIRPipeline

        pipeline = MLIRPipeline()
        text = pipeline.run_pulse_to_qop(self.mlir())
        return text if level == "qop" else pipeline.run_qop_to_cudm(text)

    @property
    def duration(self) -> Time:
        """How long the program runs: the latest end of its operations, over
        both arms of every branch, as a time (``float(duration)`` is in ns).

        A wait a float parameter sets counts once the module has values."""
        clock = Fraction(self.target.clock_ghz)
        end = 0
        for event in self.schedule:
            if event.start_cycles is None or event.duration_cycles is None:
                raise ValueError(
                    "the module is scheduled when it has the values of its float "
                    f"parameters ({', '.join(self.parameters)}); lower it with them first"
                )
            end = max(end, event.start_cycles + event.duration_cycles)
        return Time(end / clock)

    @property
    def schedule(self) -> list[ScheduledEvent]:
        """The drives, readouts and waits with their timing.

        Ops in a loop are listed once per iteration; both arms of a
        measurement-conditioned branch are listed. Without values, a wait
        whose length a float parameter sets is listed at its lowest length.
        """
        module = self._concrete() if self._retimed or self.carriers else self._module
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

    @property
    def controller_usage(self) -> UsageReport:
        """What the classical functions use of the processors of the
        device's control system (``cudaq_pulse.controller.UsageReport``),
        built when read; ``print`` it for a table."""
        from .controller import usage

        return usage(self)

    def classical_report(self) -> list[ClassicalCall]:
        """Every call to a classical function, with its timing and slack.

        Calls in a loop are listed once per iteration, in program order.
        """
        module = self._module if self.scheduled and not self.carriers else self._concrete()
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

    # Read-only: a copy is the module itself.
    def __copy__(self) -> Lowered:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Lowered:
        return self

    def __reduce__(self) -> tuple[Any, ...]:
        from ._pickling import program_reduce

        return program_reduce(self)

    def __repr__(self) -> str:
        params = f", parameters={list(self.parameters)}" if self.parameters else ""
        return f"<Lowered {self.metrics.op_count} ops on {self.target.name}{params}>"


class _View(Lowered):
    """A call's view of a module, made by ``bind``: ``_View()`` runs no
    ``__init__`` and is the cheapest way to make a ``Lowered``."""

    __slots__ = ()
    if TYPE_CHECKING:

        def __init__(self) -> None: ...

    else:
        __init__ = object.__init__


class _ClockedLowered(Lowered):
    """A module with a clock plan or line delays. Its own class keeps them off
    every other module and every call's view, which stay the size they are
    on one clock."""

    __slots__ = ("clocks", "fine_delays")


def _lowered(
    module: Any,
    clocks: _clocks.ClockPlan | None,
    fine_delays: Mapping[tuple[str, int], float] | None,
    role: _Role | None = None,
    **fields: Any,
) -> Lowered:
    lowered: Lowered
    if role is not None:
        lowered = _Placed(module, role=role, **fields)
    elif clocks is None and fine_delays is None:
        return Lowered(module, **fields)
    else:
        lowered = _ClockedLowered(module, **fields)
    lowered.clocks = clocks
    lowered.fine_delays = fine_delays
    return lowered


class _Role:
    """What the calls of a module with list or qudit parameters share: its
    symbols, which slots and static qudits get a line, and a content hash of
    the module without any call's values (``program_id``).

    ``symbols`` holds ``(name, kind, parameter, position)`` for each float
    argument of the module, in order (kind ``"float"``, or ``"vector"`` for a
    list element), then for each slot (kind ``"slot"``); *position* is the
    element's or qudit's index in its parameter, or ``None``.
    """

    __slots__ = (
        "symbols",
        "slots",
        "lines",
        "used",
        "every",
        "program_id",
        "dependencies",
        "placed",
        "views",
        "modules",
        "export_plan",
    )

    def __init__(self, symbols: tuple[tuple[str, str, str, int | None], ...]) -> None:
        self.symbols = symbols
        self.slots = 0
        self.lines: tuple[int, ...] = ()
        self.used: tuple[int, ...] = ()
        self.every: tuple[int, ...] = ()
        self.program_id = ""
        # The symbols each exported event's fields depend on, once exported.
        self.dependencies: list[dict[str, list[str]]] | None = None
        # The qudits a call uses, by its carriers (at most _PLACEMENTS).
        self.placed: dict[tuple[int, ...], tuple[int, ...]] = {}
        # A call without float values is its carriers and device: its view,
        # shared by the calls that repeat them (at most _PLACEMENTS).
        self.views: dict[tuple[tuple[int, ...], Device], Lowered] = {}
        # The module with a placement's carriers bound and its float
        # arguments kept, for simulations (the latest _PLACED_MODULES).
        self.modules: dict[tuple[tuple[int, ...], Device], Any] = {}
        # What every export shares, built by the first (controller._export_plan).
        self.export_plan: Any = None

    @staticmethod
    def identify(
        buffer: Any, name: str, parameters: Sequence[str], target: Device, options: CompileOptions
    ) -> str:
        import hashlib

        digest = hashlib.blake2b(digest_size=16)
        digest.update(memoryview(buffer).cast("B"))
        digest.update(repr((name, tuple(parameters), target._structure, options)).encode())
        return digest.hexdigest()


class _Placed(Lowered):
    """A module with list or qudit parameters, or a call's view of one."""

    __slots__ = ("_role", "carriers", "clocks", "fine_delays")

    def __init__(self, *args: Any, role: _Role, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._role = role
        self.carriers = ()
        self.clocks = None
        self.fine_delays = None

    def bind(
        self,
        values: Sequence[float],
        carriers: tuple[int, ...] = (),
        target: Device | None = None,
    ) -> Lowered:
        """This module with values for its float parameters and, with
        *carriers*, its runtime qudits bound on *target*, sharing it."""
        return _place(self, values, carriers, target)


def _placement(view: Lowered) -> Any:
    """*view*'s module with its carriers bound and its float arguments kept,
    shared by the views of one placement."""
    role = view._role
    assert role is not None and view.carriers
    key = (view.carriers, view.target)
    module = role.modules.get(key)
    if module is None:
        with view._lock:
            module = view._module.place_carriers(
                list(view.carriers), view.target._frequency_table()
            )
        if len(role.modules) >= _PLACED_MODULES:
            role.modules.clear()
        role.modules[key] = module
    return module


def _place(
    lowered: Lowered, values: Sequence[float], carriers: tuple[int, ...], target: Device | None
) -> _Placed:
    """``Lowered.bind`` of a module with list or qudit parameters."""
    waits = lowered.waits
    if waits is not None:
        waits.check(values)
    role = lowered._role
    assert role is not None
    view = _Placed.__new__(_Placed)
    view._module = lowered._module
    view._text = None
    view._lock = lowered._lock
    view._concrete_module = None
    view.target = lowered.target
    view.metrics = lowered.metrics
    view.parameters = ()
    view.arguments = tuple(values)
    view.qudits = lowered.qudits
    view.options = lowered.options
    view.scheduled = lowered.scheduled
    view._generic = lowered
    view.waits = waits
    view.classical = lowered.classical
    view.runtime_selects = lowered.runtime_selects
    view._role = role
    view.carriers = carriers
    view.clocks = lowered.clocks
    view.fine_delays = lowered.fine_delays
    if carriers:
        assert target is not None
        view.target = target
        placed = role.placed.get(carriers)
        if placed is None:
            qudits = {*role.used, *(carriers[k] for k in role.lines)}
            placed = tuple(sorted(qudits or {*role.every, *carriers}))
            if len(role.placed) < _PLACEMENTS:
                role.placed[carriers] = placed
        view.qudits = placed
    return view


class _CheckedPerCall(Lowered):
    """A module whose controller limits some float values exceed and others
    do not: each call's values are checked when they are bound, once per
    distinct values, so that no limit fails first at export."""

    __slots__ = ("_accepted", "clocks", "fine_delays")

    def __init__(self, lowered: Lowered) -> None:
        super().__init__(
            lowered._module,
            target=lowered.target,
            metrics=lowered.metrics,
            parameters=lowered.parameters,
            qudits=lowered.qudits,
            options=lowered.options,
            scheduled=lowered.scheduled,
            waits=lowered.waits,
        )
        self.classical = lowered.classical
        self.runtime_selects = lowered.runtime_selects
        self.clocks = lowered.clocks
        self.fine_delays = lowered.fine_delays
        self._accepted: set[bytes] = set()

    def bind(self, values: Sequence[float]) -> Lowered:
        view = super().bind(values)
        return self._accept(view, _bits(view.arguments))

    def _accept(self, view: Lowered, key: bytes) -> Lowered:
        """*view*, once its values are within the limits (checked once per
        distinct *key*)."""
        if key not in self._accepted:
            from .controller import _call_violations, _limit_error

            violations = _call_violations(view)
            if violations:
                raise _limit_error(view, violations)
            if len(self._accepted) >= _ACCEPTED_CALLS:
                self._accepted.clear()
            self._accepted.add(key)
        return view


class _CheckedPlaced(_CheckedPerCall):
    """A ``_CheckedPerCall`` module with list or qudit parameters."""

    __slots__ = ("_role", "carriers")

    def __init__(self, lowered: Lowered) -> None:
        super().__init__(lowered)
        self._role = lowered._role
        self.carriers = ()

    def bind(
        self,
        values: Sequence[float],
        carriers: tuple[int, ...] = (),
        target: Device | None = None,
    ) -> Lowered:
        view = _place(self, values, carriers, target)
        key = _bits(view.arguments)
        if carriers:
            key += _bits(tuple(map(float, carriers))) + view.target.fingerprint.encode()
        return self._accept(view, key)


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


def _rejected_literal(
    fn: Any, trace_args: Sequence[Any], traced: Any, error: PackedValueError
) -> CompilationError:
    """The error for a literal the extension rejected, with the kernel line
    that wrote it. The kernel is traced again to find that line; when the
    trace differs (the kernel read state that changed), the line is left out
    rather than guessed."""
    from .kernel._emitter import add_source_note
    from .kernel.bytecode_bridge import _trace_kernel_with_builder

    message, word = error.args
    located = CompilationError(message)
    finder = traced.finder(word)
    try:
        _trace_kernel_with_builder(fn, finder, trace_args)
    except Exception:
        return located
    if finder.frames and finder.get_buffer() == traced.get_buffer():
        located.pulse_source = finder.frames[0]  # type: ignore[attr-defined]
        located.pulse_callers = finder.frames[1:]  # type: ignore[attr-defined]
        add_source_note(located)
    return located


def build(
    fn: Any,
    trace_args: Sequence[Any],
    parameters: tuple[str, ...],
    target: Device,
    options: CompileOptions,
    slots: int = 0,
    role: _Role | None = None,
) -> tuple[Lowered, Any]:
    """Trace *fn* and build its module on *target*.

    *trace_args* holds a ``Parameter`` sentinel for every float parameter
    named in *parameters*; those stay function arguments of the module, as
    do the *slots* runtime qudits (``Slot`` sentinels). A *role* names the
    symbols of a module with list or qudit parameters, and gets the rest of
    what its calls share.
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
    machine = options.machine
    builder = PackedIRBuilder(
        name=name,
        clock_ghz=target.clock_ghz,
        qubit_freq_hz=dict(target.frequencies),
        flatten_schedules=options.flatten_schedules,
    )
    builder.unroll_threshold = options.unroll_threshold
    builder.line_tones = target._line_tones
    builder.link_id = machine.link_id
    builder.locate_structural = frozenset(parameters)
    plain = machine._plain
    clocks = None if plain else _clocked_builder(builder, machine, target)
    builder.returned = _trace_kernel_with_builder(fn, builder, trace_args)
    needed = tuple(sorted(builder.structural & set(parameters)))
    if needed:
        error = SymbolicValueError(
            f"{name}(): {', '.join(needed)} must be known while the kernel compiles",
            needed,
            "a value that must be an integer, such as a waveform duration",
        )
        frames = getattr(builder, "structural_at", {}).get(needed[0])
        if frames:
            error.pulse_source = frames[0]  # type: ignore[attr-defined]
            error.pulse_callers = frames[1:]  # type: ignore[attr-defined]
        raise error
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
    links = [machine.link_id(function.link) for function in builder.classical]
    metrics.trace_ms = (time.perf_counter() - t0) * 1000
    metrics.op_count = builder.op_count

    t0 = time.perf_counter()
    try:
        if slots:
            module = PulseModuleBuilder().build_from_packed(
                buf,
                target.clock_ghz,
                builder.n_qubits,
                freq,
                list(parameters),
                ["f64"] * len(parameters),
                name=name,
                n_slots=slots,
            )
        elif parameters:
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
    except PackedValueError as error:
        raise _rejected_literal(fn, trace_args, builder, error) from None
    if links:
        module.set_attributes(machine.links_attribute(links))
    fine_delays = None if plain else _clocked_module(module, machine, target, clocks, builder)
    metrics.ffi_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    runtime_selects = False
    if builder.selects:
        runtime_selects = module.lower_selects(
            [kind for kind in _RUNTIME_KINDS if _can_select_at_runtime(target, kind)]
        )
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

    if plain and role is None:
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
    else:
        lowered = _lowered(
            module,
            clocks,
            fine_delays,
            role,
            target=target,
            metrics=metrics,
            parameters=parameters,
            qudits=builder.used_qubits or qudits,
            options=options,
            scheduled=scheduled,
            waits=(
                _wait_lengths(builder.variable_waits, name, 1 if clocks is None else clocks.lcm)
                if builder.variable_waits
                else None
            ),
        )
    lowered.classical = tuple(builder.classical)
    lowered.runtime_selects = runtime_selects
    if role is not None:
        role.slots = slots
        role.lines = builder.used_slots
        role.used = builder.used_qubits
        role.every = qudits
        role.program_id = role.identify(buf, name, parameters, target, options)
    if not plain:
        lowered = _check_controller(lowered)
    return lowered, builder


def _clocked_builder(
    builder: Any, machine: MachineModel, target: Device
) -> _clocks.ClockPlan | None:
    """Set *builder* up for *machine*'s controllers; their clock plan."""
    from . import _clocks

    clocks = (
        None
        if machine.controller is None and machine.reference_clock_ghz is None
        else _clocks.plan(machine.controller, target.clock_ghz, machine.reference_clock_ghz)
    )
    if clocks is not None:
        builder.line_grid = clocks.grid_of
        builder.wait_grid = clocks.lcm
    if machine.controller is not None:
        builder.sample_rate = _clocks.sample_rate(machine.controller, target.clock_ghz)
    return clocks


def _clocked_module(
    module: Any,
    machine: MachineModel,
    target: Device,
    clocks: _clocks.ClockPlan | None,
    builder: Any,
) -> Mapping[tuple[str, int], float] | None:
    """Set the grid and line delay attributes of *module*; the fine delays."""
    if clocks is not None:
        grids = clocks.attributes(builder.n_qubits)
        if grids is not None:
            module.set_attributes(grids)
    fine_delays = None
    if machine.line_delays_ns:
        shifts, fine_delays = _line_delays(machine, target, clocks, builder.n_qubits)
        if shifts is not None:
            module.set_attributes(f"{{pulse.line_delays = {shifts}}}")
    return fine_delays


def _line_delays(
    machine: MachineModel, target: Device, clocks: _clocks.ClockPlan | None, n_qudits: int
) -> tuple[str | None, dict[tuple[str, int], float]]:
    from . import _clocks

    try:
        return _clocks.line_delays(
            machine.line_delays_ns, machine.controller, clocks, target.clock_ghz, n_qudits
        )
    except ValueError as e:
        raise CompilationError(str(e)) from None


def _check_controller(lowered: Lowered) -> Lowered:
    """*lowered*, checked against its controller limits: those it exceeds
    whatever its float values fail now, and when some values may exceed
    others, each call's values are checked when they are bound."""
    machine = lowered.options.machine
    limits = machine.controller
    system = machine._system
    if limits is None and system is None:
        return lowered
    from .controller import _compile_check, _limit_error

    exact = False
    # A float wait length places the ops after it per call: the schedule
    # without values is not one any call plays.
    if lowered.scheduled and lowered.waits is None:
        violations, exact = _compile_check(lowered, limits, system)
        if violations:
            raise _limit_error(lowered, violations)
    if exact:
        return lowered
    return (_CheckedPerCall if lowered._role is None else _CheckedPlaced)(lowered)


def with_line_delays(lowered: Lowered, options: CompileOptions, n_qudits: int) -> Lowered | None:
    """*lowered*, a scheduled module of *n_qudits* qudits, compiled again for
    *options*, which differ from its own only in ``machine.line_delays_ns``:
    its module before scheduling is kept, and only scheduling, the checks and
    the exports are done again. ``None`` when scheduling it again fails, for
    a full compile to report, and for a module with list or qudit
    parameters, whose calls share what its own schedule gives them."""
    if lowered._role is not None:
        return None
    target = lowered.target
    clocks = lowered.clocks
    fine_delays = None
    shifts = None
    if options.machine.line_delays_ns:
        shifts, fine_delays = _line_delays(options.machine, target, clocks, n_qudits)
    t0 = time.perf_counter()
    try:
        with lowered._lock:
            module = lowered._module.with_line_delays(shifts or "", *options.schedule_args())
    except Exception:
        return None
    old = lowered.metrics
    metrics = CompileMetrics(target=old.target)
    metrics.op_count = old.op_count
    metrics.schedule_ms = (time.perf_counter() - t0) * 1000
    metrics.total_ms = metrics.schedule_ms
    result = _lowered(
        module,
        clocks,
        fine_delays,
        target=target,
        metrics=metrics,
        parameters=lowered.parameters,
        qudits=lowered.qudits,
        options=options,
        waits=lowered.waits,
    )
    result.classical = lowered.classical
    result.runtime_selects = lowered.runtime_selects
    return _check_controller(result)


def compile[**P](
    kernel: Kernel[P],
    /,
    *,
    schedule: str | None = None,
    passes: Sequence[str] | None = None,
    machine: MachineModel | ControlSystem | None = None,
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
        Machine model for resource-constrained scheduling, or a
        ``ControlSystem``: every program is then checked against its
        controllers' wiring, capabilities and limits. By default the
        control system of *device*, if it has one.
    flatten_schedules:
        Ignore the ``schedule=`` of every called kernel and schedule the whole
        program with *schedule*.
    unroll_threshold:
        A ``for ... in range(n)`` loop whose variable the body never reads is
        unrolled while tracing when n times the number of calls in its body
        is at most this, and emitted once as a rolled loop otherwise. By
        default the kernel's own ``unroll_threshold=``, else 8. A
        non-negative ``int``; 0 never unrolls, so every such loop stays
        rolled, even one of no or one iteration. Either way the
        program, its schedule and its simulation are the same.
    device:
        Pin the compiled kernel to one device: a call with handles of
        another raises ``DeviceMismatchError``, and a call without handles
        runs on it. A pinned kernel without static parameters is built now.
    """
    from .kernel._kernel import CompiledKernel, Kernel

    if not isinstance(kernel, Kernel):
        raise TypeError(f"compile() expects a @cudaq_pulse.kernel, got {type(kernel).__name__}")
    if machine is None and device is not None:
        machine = device.control_system
    if machine is not None and not isinstance(machine, MachineModel):
        from ._system import ControlSystem

        if not isinstance(machine, ControlSystem):
            raise InvalidTypeError(
                f"compile(machine=) takes a MachineModel or a ControlSystem, got {machine!r}"
            )
        machine = machine._machine
    options = CompileOptions.make(
        root_schedule(kernel.schedule, schedule),
        passes,
        machine,
        flatten_schedules,
        kernel.unroll_threshold if unroll_threshold is None else unroll_threshold,
    )
    return CompiledKernel(kernel, options, device)
