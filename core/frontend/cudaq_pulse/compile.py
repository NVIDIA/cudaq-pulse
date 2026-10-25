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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

try:
    from ._native._cudaq_pulse_native import PulseModuleBuilder
except ImportError as _e:
    raise RuntimeError(
        "cudaq-pulse native bindings required. Install the cudaq-pulse wheel, "
        "or build from source with `pip install .`; see CONTRIBUTING.md."
    ) from _e

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


@dataclass(frozen=True)
class MachineModel:
    """Hardware resource limits for the ``rcp`` and ``alap_rcp`` schedules."""

    max_concurrent_drives: int = 4
    max_concurrent_readouts: int = 2
    readout_latency_vtu: int = 0
    line_switch_penalty_vtu: int = 0


@dataclass(frozen=True)
class ScheduledEvent:
    """One timed operation of a compiled kernel.

    ``line`` numbers the kernel's drive and readout lines in allocation order.
    ``start_vtu`` and ``duration_vtu`` are ``None`` until the kernel is
    scheduled (a kernel with float parameters is scheduled when it is called
    with their values).
    """

    kind: str
    line: int
    qubit: int
    start_vtu: int | None
    duration_vtu: int | None


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
        "n_qudits",
        "qudits",
        "options",
        "scheduled",
        "generic",
    )

    def __init__(
        self,
        module: Any,
        *,
        target: Device,
        metrics: CompileMetrics,
        parameters: tuple[str, ...],
        n_qudits: int,
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
        self.n_qudits = n_qudits
        self.qudits = qudits
        self.options = options
        self.scheduled = scheduled
        self.generic: Lowered | None = None

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
        view.n_qudits = self.n_qudits
        view.qudits = self.qudits
        view.options = self.options
        view.scheduled = self.scheduled
        view.generic = self
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

    def timeline(self, **kwargs: Any) -> Figure:
        """Plot the schedule, one row per line (needs matplotlib)."""
        from .viz.timeline import plot_schedule

        return plot_schedule(self.schedule, **kwargs)

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
        n_qudits=builder.n_qubits,
        qudits=qudits,
        options=options,
        scheduled=scheduled,
    )
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
