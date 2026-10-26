# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Lowering a traced kernel to a scheduled MLIR module, and its inspection.

``cudaq_pulse.compile(k)`` returns a callable ``CompiledKernel`` (see
``kernel/kernel.py``); this module holds what it is built from: the options
that pick passes and a schedule, the native build of one specialization, and
``Lowered``, the read-only view of one compiled module.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

try:
    from ._native._cudaq_pulse_native import PulseModuleBuilder
except ImportError as _e:
    raise RuntimeError(
        "cudaq-pulse native bindings required. Install the cudaq-pulse wheel, "
        "or build from source with `pip install .`; see CONTRIBUTING.md."
    ) from _e

from .classical import Link
from .kernel._schedule_blocks import check_root_schedule, root_schedule
from .kernel._unroll import resolve_unroll_threshold
from .kernel.ir_builder import CompilationError

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ._device import Device
    from .kernel._kernel import CompiledKernel, Kernel

_VALID_SCHEDULES = frozenset({"asap", "alap", "rcp", "alap_rcp"})

# C++ pass names for the full pipeline
_PASS_MAP = {
    "verify": "pulse-verify",
    "canonicalize": "pulse-canonicalize",
    "virtual_z": "pulse-virtual-z",
    "fusion": "pulse-fusion",
    "licm": "loop-invariant-code-motion",
}

DEFAULT_PASSES: tuple[str, ...] = (
    "verify",
    "virtual_z",
    "fusion",
)

_STAGES = ("pulse", "llvm")


class _Links(dict[int, Link]):
    """The links of a machine model: a read-only, hashable mapping."""

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
    error.
    """

    max_concurrent_drives: int = 4
    max_concurrent_readouts: int = 2
    readout_latency_vtu: int = 0
    line_switch_penalty_vtu: int = 0
    links: Mapping[int | str, Link] = field(default_factory=dict)
    link_names: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
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
        object.__setattr__(self, "links", links)
        object.__setattr__(self, "link_names", MappingProxyType(by_name))
        object.__setattr__(
            self,
            "_hash",
            hash(
                (
                    self.max_concurrent_drives,
                    self.max_concurrent_readouts,
                    self.readout_latency_vtu,
                    self.line_switch_penalty_vtu,
                    self.links,
                    tuple(sorted(by_name.items())),
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
            self.readout_latency_vtu,
            self.line_switch_penalty_vtu,
            dict(self.links),
            dict(self.link_names),
        ) == (
            other.max_concurrent_drives,
            other.max_concurrent_readouts,
            other.readout_latency_vtu,
            other.line_switch_penalty_vtu,
            dict(other.links),
            dict(other.link_names),
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
    ``start_vtu`` and ``duration_vtu`` are ``None`` until the kernel is
    scheduled (a kernel with float parameters is scheduled when it is called
    with their values).

    A call to a classical function adds a ``"submit"`` event, while the
    request holds its link, and a zero-length ``"answer"`` event, when the
    answer is taken. For those, ``line`` is the link id and ``qubit`` is -1.
    """

    kind: str
    line: int
    qubit: int
    start_vtu: int | None
    duration_vtu: int | None

    @property
    def link(self) -> int | None:
        """The link of a classical call's event, else ``None``."""
        return self.line if self.kind in ("submit", "answer") else None


@dataclass(frozen=True)
class Epoch:
    """One epoch of a compiled kernel, from ``Lowered.epochs()``.

    A variable wait (its length computed from a classical answer) or a
    blocking classical wait is a barrier: every line syncs before it, and
    everything after it shifts with its run-time length. Epoch 0 starts the
    program; epoch k starts when the k-th barrier ends. ``start_vtu`` is the
    epoch's start in the static schedule, which places every barrier at its
    lowest length. At run time the epoch starts ``shift_vtu`` later, with
    ``min_shift_vtu <= shift_vtu <= max_shift_vtu`` (``None``: unbounded).
    ``opened_by`` is ``"start"``, ``"wait"`` (a variable wait on line
    ``line``) or ``"answer"`` (a blocking wait on link ``line``).
    """

    index: int
    opened_by: str
    line: int
    start_vtu: int
    min_shift_vtu: int
    max_shift_vtu: int | None


@dataclass(frozen=True)
class ClassicalCall:
    """One execution of a call to a classical function, from ``classical_report()``.

    Times are in virtual time units (clock cycles). The request is sent from
    ``submit_vtu`` for ``submit_duration_vtu``; its answer is due at
    ``due_vtu`` (the budgeted latency after that) and taken at ``taken_vtu``
    (``None`` if the kernel never takes it). ``first_use_vtu`` is the
    earliest start of a pulse whose amplitude, tone or branch comes from the
    answer, and ``deadline_vtu`` the latest due time the function's
    ``deadline_ns`` allows. ``epoch`` counts the blocking waits before it.
    ``link_name`` is the link's name in the machine model, if it has one.
    """

    function: str
    link: int
    iteration: tuple[int, ...]
    submit_vtu: int
    submit_duration_vtu: int
    due_vtu: int
    taken_vtu: int | None
    first_use_vtu: int | None
    deadline_vtu: int | None
    epoch: int
    link_name: str | None = None

    @property
    def slack_vtu(self) -> int | None:
        """How long the answer could arrive later without delaying a pulse."""
        return None if self.first_use_vtu is None else self.first_use_vtu - self.due_vtu

    @property
    def deadline_slack_vtu(self) -> int | None:
        """How far the due time is inside the deadline (negative: missed)."""
        return None if self.deadline_vtu is None else self.deadline_vtu - self.due_vtu


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
    passes: tuple[str, ...] = DEFAULT_PASSES
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
        passes = DEFAULT_PASSES if passes is None else tuple(passes)
        unknown = sorted(set(passes) - set(_PASS_MAP))
        if unknown:
            raise ValueError("Unknown pulse passes: " + ", ".join(unknown))
        machine = machine or MachineModel()
        if machine.max_concurrent_drives <= 0:
            raise ValueError("machine.max_concurrent_drives must be positive")
        if machine.max_concurrent_readouts <= 0:
            raise ValueError("machine.max_concurrent_readouts must be positive")
        if machine.readout_latency_vtu < 0:
            raise ValueError("machine.readout_latency_vtu cannot be negative")
        if machine.line_switch_penalty_vtu < 0:
            raise ValueError("machine.line_switch_penalty_vtu cannot be negative")
        if unroll_threshold is not None:
            unroll_threshold = resolve_unroll_threshold(unroll_threshold)
        return CompileOptions(schedule, passes, machine, bool(flatten_schedules), unroll_threshold)

    def schedule_args(self) -> tuple[str, int, int, int, int]:
        m = self.machine
        return (
            self.schedule,
            m.max_concurrent_drives,
            m.max_concurrent_readouts,
            int(m.readout_latency_vtu),
            int(m.line_switch_penalty_vtu),
        )


class Lowered:
    """One compiled module of a kernel: read-only text, schedule and metrics.

    ``kernel.lower(*args)`` and ``compiled.lower(*args)`` return the module
    of one call. Float parameters stay function arguments of the module, and
    the call's values are kept beside it in ``arguments``, so a call with new
    values shares the module; ``generic`` is the module without values.
    ``mlir()`` of a call shows its values substituted.

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
        # The classical functions the program calls, in declaration order.
        self.classical: tuple[Any, ...] = ()

    def bind(self, values: Sequence[float]) -> Lowered:
        """This module with values for its float parameters, sharing it."""
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
        view.classical = self.classical
        return view

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
        measurement-conditioned branch are listed.
        """
        module = self._module if self.scheduled else self._concrete()
        with self._lock:
            events = module.events()
        return [ScheduledEvent(*event) for event in events]

    def epochs(self) -> list[Epoch]:
        """The epochs of the program, separated by variable and blocking waits.

        A program without such waits has one epoch.
        """
        module = self._module if self.scheduled else self._concrete()
        with self._lock:
            boundaries = module.epoch_boundaries()
        epochs = [Epoch(0, "start", -1, 0, 0, 0)]
        unbounded = False
        extra = 0
        for k, (kind, line, start, lowest, highest) in enumerate(boundaries, start=1):
            if highest < 0:
                unbounded = True
            else:
                extra += highest - lowest
            epochs.append(Epoch(k, kind, line, start + lowest, 0, None if unbounded else extra))
        return epochs

    def classical_report(self) -> list[ClassicalCall]:
        """Every call to a classical function, with its timing and slack.

        Calls in a loop are listed once per iteration, in program order.
        """
        module = self._module if self.scheduled else self._concrete()
        with self._lock:
            calls = module.classical_calls()
        names = self.options.machine.link_name
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
            )
            for name, link, iteration, start, length, due, taken, use, limit, epoch in calls
        ]

    def export_emulation(self) -> dict[str, Any]:
        """The program for a controller emulator: its schedule by epoch, its
        calls and the manifest of its classical functions (see
        ``cudaq_pulse.classical.emulation``)."""
        from .classical.emulation import export

        return export(self)

    def timeline(self, **kwargs: Any) -> Figure:
        """Plot the schedule, one row per line (needs matplotlib)."""
        from .viz.timeline import plot_schedule

        return plot_schedule(self, **kwargs)

    def __repr__(self) -> str:
        params = f", parameters={list(self.parameters)}" if self.parameters else ""
        return f"<Lowered {self.metrics.op_count} ops on {self.target.name}{params}>"


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
    calls the caller needs.
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

    # Durations never depend on a float parameter (a float that reaches one
    # is specialized on), so the module schedules with them symbolic.
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
    )
    lowered.classical = tuple(builder.classical)
    return lowered, builder


def compile(
    kernel: Kernel,
    /,
    *,
    schedule: str | None = None,
    passes: Sequence[str] | None = None,
    machine: MachineModel | None = None,
    device: Device | None = None,
    flatten_schedules: bool = False,
    unroll_threshold: int | None = None,
) -> CompiledKernel:
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
        Optimization passes, ``DEFAULT_PASSES`` by default; ``()`` for none.
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
