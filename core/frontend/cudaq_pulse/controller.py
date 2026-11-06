# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""A compiled kernel for a controller, and what a controller can play.

``Lowered.export_controller()`` (``export`` here) writes a scheduled kernel in
the form an out-of-tree controller backend lowers to its own instructions,
format ``FORMAT`` (``cudaq-pulse.controller/1``): the lines and the qudits
they belong to, and every drive, readout, wait, sync and classical call with
its exact start and end, its waveform (the shape and its parameters, and
optionally its samples), its tone's frequency and frame phase, the
measurement-conditioned branches it lies in, and the epochs and calls of
``Lowered.export_emulation()``. The schedule is final: a backend places each
event at its ``start_cycles`` and does not schedule again.

Times count the controller's own clock cycles, at ``clock_ghz``. When the
controller's clock is not the program clock, the export also gives
``program_clock_ghz``, every time is converted to its cycles (exactly for
operations on its lines, which are on its grid, and rounded up for the
others) and every waveform's time parameters are scaled with them. With a
shared reference clock (``MachineModel.reference_clock_ghz``) it also gives
``reference_clock_ghz``.

The number after the slash is the format's major version. Within one version,
new keys and new event kinds may be added, so a backend ignores keys it does
not know and refuses event kinds it does not know; a change that removes or
changes the meaning of a key gets a new version. ``export(version=2)`` gives
the structured format ``cudaq-pulse.controller/2``: loops whose iterations
repeat stay loops, each distinct waveform is listed once, and what follows
from the rest is left out; ``unroll`` turns it into format /1.

``ControllerLimits`` states what a controller can play. Given to the
machine model, ``MachineModel(controller=ControllerLimits(...))``, it makes
``compile()`` check every scheduled program against it and raise
``ControllerLimitError`` naming each violated limit, so that a program the
controller cannot play fails at compile time. ``check`` returns the
violations without raising.

The limits also state what the controller can compute, for the classical
functions it runs itself (those on its ``compute_links``): the cycles each
class of operation takes, how many operations a function may hold, how many
bytes of state, and how many calls and bytes of requests and answers one
shot may use. ``resources`` reports, for every classical function of a
program, an estimate of its operations by class (read from its body), its
state in bytes, its compute time against its ``max_latency_ns`` and its
calls, slots and buffer bytes per shot, and whether its body is in
*controller form*: a fixed number of scalars of state, loops of a trip count
known when the program compiles, no ``while``, recursion, allocation or
exceptions. ``analyze`` does the same for one function on its own.
"""

from __future__ import annotations

import copy
import math
import os
import sys
import types
from collections.abc import Mapping, Sequence
from dataclasses import field, fields
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

from ._errors import ExecutionError, InvalidTypeError, InvalidValueError, UnsupportedError
from ._units import Time, as_nanoseconds
from .kernel.ir_builder import CompilationError

try:
    # hashlib's own blake2b, without hashlib's import of OpenSSL (a millisecond).
    from _blake2 import blake2b as _blake2b
except ImportError:  # pragma: no cover - CPython builds without the _blake2 module
    from hashlib import blake2b as _blake2b

if TYPE_CHECKING:
    from dataclasses import dataclass as _dataclass_when_used
    from fractions import Fraction

    import numpy as np

    from . import _clocks
    from ._controller_cost import ControllerUsage as ControllerUsage
    from ._controller_cost import FunctionCost as FunctionCost
    from ._controller_cost import FunctionResources as FunctionResources
    from ._controller_cost import FunctionUsage as FunctionUsage
    from ._controller_cost import LimitExceeded as LimitExceeded
    from ._controller_cost import LinkUsage as LinkUsage
    from ._controller_cost import ProcessorUsage as ProcessorUsage
    from ._controller_cost import ResourceReport as ResourceReport
    from ._controller_cost import UsageReport as UsageReport
    from ._controller_cost import analyze as analyze
    from ._controller_cost import format_report as format_report
    from ._controller_cost import resources as resources
    from ._controller_cost import usage as usage
    from ._system import Controller, ControlSystem
    from .compiler import Lowered

    # The real and the imaginary parts of a waveform's samples.
    _Samples = tuple[list[float], list[float]]
else:
    from ._errors import _dataclass_when_used

__all__ = [
    "EXPRESSION_TERMS",
    "FORMAT",
    "OP_CLASSES",
    "ControllerLimitError",
    "ControllerLimits",
    "ControllerUsage",
    "Event",
    "FunctionCost",
    "FunctionResources",
    "FunctionUsage",
    "LimitExceeded",
    "LinkUsage",
    "ProcessorUsage",
    "ResourceReport",
    "UsageReport",
    "WaveformMemoryScope",
    "analyze",
    "check",
    "evaluate_expression",
    "export",
    "resources",
    "sample_waveform",
    "shape_key",
    "usage",
    "unroll",
]

FORMAT = "cudaq-pulse.controller/1"
# The structured format, ``export_controller(version=2)``.
_STRUCTURED = "cudaq-pulse.controller/2"
# Every format this cudaq-pulse writes.
_FORMATS = (FORMAT, _STRUCTURED)
# The names format /2 gives a controller's and a link's fields where they
# differ from the /1 names; unroll() turns them back.
_CONTROLLER_KEYS: Mapping[str, str] = MappingProxyType(
    {
        "issue_cycles": "instruction_cycles",
        "min_wait_cycles": "min_idle_cycles",
        "acquirers": "detectors",
        "acquirer_hold_cycles": "detector_recovery_cycles",
        "waveform_samples": "waveform_memory_samples",
        "tones_per_line": "max_tones_per_line",
        "max_events": "max_pulses",
        "answer_timeouts": "supports_timeouts",
    }
)
_LINK_KEYS: Mapping[str, str] = MappingProxyType(
    {
        "fixed_ns": "send_ns",
        "bytes_per_ns": "bandwidth_bytes_per_ns",
        "feedback_ns": "answer_delay_ns",
        "max_in_flight": "slots",
    }
)
# What a controller computes is one processor in format /2 (none when no
# compute limit is set): each /2 field, in order, with the /1 field it
# comes from (None: one /1 does not have).
_PROCESSOR_KEYS: tuple[tuple[str, str | None], ...] = (
    ("links", "compute_links"),
    ("clock_ghz", "compute_clock_ghz"),
    ("cycles_per_op", "op_cycles"),
    ("unsupported", None),
    ("max_concurrent_calls", None),
    ("fixed_cost_only", "controller_form"),
    ("max_ops", "max_ops"),
    ("max_state_bytes", "state_bytes"),
    ("max_calls_per_shot", "max_calls_per_shot"),
    ("max_bytes_per_shot", "buffer_bytes"),
)
_COMPUTE_KEYS = frozenset(old for _, old in _PROCESSOR_KEYS if old is not None)
# The /2 processor fields ControllerLimits cannot set: the calls it computes
# at once, and the operations it cannot compute (named, not costed).
_PROCESSOR_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {"unsupported": (), "max_concurrent_calls": 1}
)
# The /2 name of the /1 key naming the quantum system.
_SYSTEM_KEYS: Mapping[str, str] = MappingProxyType({"target": "quantum_system"})


class Event(StrEnum):
    """The kind of an event of a format /2 program: what the controller
    does, named alike for every modality.

    ``PLAY`` plays a waveform on a line and ``ACQUIRE`` plays one and
    acquires a measurement result; ``WAIT`` idles a line, ``SYNC`` aligns
    lines and ``LOOP`` repeats its body. ``CALL`` sends a request to a
    classical function over a link and ``ANSWER`` takes its answer. An event
    in a branch keeps its kind and lists its ``conditions``; a frame's
    phase and frequency are carried by the events that play on it, and
    ``FRAME_UPDATE`` is a phase or frequency change that no later event
    plays with on some path, at its place on the timeline. The export holds
    the plain strings, so ``Event.PLAY == "play"``.
    """

    PLAY = "play"
    ACQUIRE = "acquire"
    WAIT = "wait"
    SYNC = "sync"
    LOOP = "loop"
    CALL = "call"
    ANSWER = "answer"
    FRAME_UPDATE = "frame_update"


_EVENT_KINDS = frozenset(Event)
# The /2 kinds of /1 events whose /1 names differ.
_KINDS: Mapping[str, str] = MappingProxyType(
    {"drive": "play", "readout": "acquire", "submit": "call"}
)
_FLAT_KINDS: Mapping[str, str] = MappingProxyType({v: k for k, v in _KINDS.items()})

# The classes operations are counted in.
OP_CLASSES: tuple[str, ...] = (
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

# The parameter names of each waveform op, in operand order.
_PARAMETERS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "square": ("duration", "amplitude", "amplitude_imag"),
        "gaussian": ("duration", "amplitude", "sigma"),
        "gaussian_square": ("duration", "amplitude", "sigma", "risefall"),
        "drag": ("duration", "amplitude", "sigma", "beta"),
        "cosine": ("duration", "amplitude"),
        "tanh_ramp": ("duration", "amplitude", "sigma"),
        "custom": ("duration",),
        "scale": ("factor",),
    }
)
# The operands an overall amplitude multiplies, per waveform op.
_AMPLITUDES: Mapping[str, tuple[int, ...]] = MappingProxyType(
    {
        "square": (1, 2),
        "gaussian": (1,),
        "gaussian_square": (1,),
        "drag": (1,),
        "cosine": (1,),
        "tanh_ramp": (1,),
    }
)
# The names of those operands.
_AMPLITUDE_NAMES: Mapping[str, frozenset[str]] = MappingProxyType(
    {shape: frozenset(_PARAMETERS[shape][i] for i in at) for shape, at in _AMPLITUDES.items()}
)
_PLAYED = frozenset({"drive", "readout", "play", "acquire"})
# The keys a format /2 waveform has beside its expression.
_IDS = ("id", "shape_id")


class ControllerLimitError(CompilationError):
    """A scheduled program exceeds a limit of the controller it is for.

    ``violations`` has every limit exceeded: a tuple, or past 10,000 of
    them in rolled loops a sequence that builds each one when read, with an
    exact ``len()``. For a control system with processors, ``usage`` is the
    program's ``UsageReport``: what each processor holds against its limits,
    and each limit exceeded with by how much (``usage.exceeded``); the
    message ends with its table.
    """

    def __init__(
        self,
        violations: Sequence[str],
        *,
        system: bool = False,
        usage: UsageReport | None = None,
    ) -> None:
        self.violations: Sequence[str] = tuple(violations) if _listed(violations) else violations
        self.usage = usage
        shown = "\n  ".join(self.violations[:10])
        more = len(self.violations) - 10
        tail = f"\n  ... and {more} more" if more > 0 else ""
        what = "does not fit the control system" if system else "exceeds the controller's limits"
        table = ""
        if usage is not None:
            from ._controller_cost import _format_usage

            table = "\n" + _format_usage(usage, exceeds=False)
        super().__init__(f"the program {what}:\n  {shown}{tail}{table}")


def _listed(violations: Sequence[str]) -> bool:
    """Whether *violations* are not the lazy ones of a summarized loop."""
    if isinstance(violations, (list, tuple)):
        return True
    from ._loops import Violations

    return not isinstance(violations, Violations)


def _limit_error(lowered: Lowered, violations: Sequence[str]) -> ControllerLimitError:
    """The error for the *violations* of *lowered*, with its usage report
    when its control system has processors and they were checked: always
    for one call, and at compile time without float or qudit parameters."""
    system = lowered.options.machine._system
    report = None
    if system is not None and system._placement:
        placed = lowered._role is not None and lowered._role.slots > 0
        if lowered._generic is not None or not (lowered.parameters or placed):
            from ._controller_cost import usage

            report = usage(lowered)
    return ControllerLimitError(violations, system=system is not None, usage=report)


class WaveformMemoryScope(StrEnum):
    """Which lines one waveform memory of a controller serves
    (``ControllerLimits.waveform_memory``); the strings ``"per_line"`` and
    ``"shared"`` mean the same."""

    #: Each line has a memory of its own.
    PER_LINE = "per_line"
    #: One memory serves every line, and holds a waveform they share once.
    SHARED = "shared"


@_dataclass_when_used(frozen=True)
class ControllerLimits:
    """What a controller can play; every limit is in its own clock cycles.

    ``clock_ghz``: the controller's clock (``None``: the program clock,
    ``Device.clock_ghz``). It must be a whole-cycle ratio of the program
    clock; its program counts its own cycles.
    ``cycles_per_timestep``: the controller starts and times operations only
    every this many of its own cycles; every operation on its lines must
    start and last whole timesteps (see "Clocks" in the controller backends
    guide; ``pulse.round_duration`` rounds a length to them).
    ``issue_cycles`` and ``min_wait_cycles``: two operations on one line start
    exactly ``issue_cycles`` apart, or at least ``issue_cycles + min_wait_cycles``
    apart, because the controller issues one instruction per ``issue_cycles``
    and idles no less than ``min_wait_cycles``.
    ``sample_rate_ghz``: the rate its lines play samples at (``None``: one
    per cycle of its clock). Its ratio to the clock may be a fraction, and
    every operation on its lines must start and last a whole number of
    samples, which the timestep includes: at 2.4 GHz on a 1 GHz clock, whole
    multiples of 5 cycles. A length off them is a compile error, never
    rounded.
    ``waveform_samples``: a waveform memory holds this many samples, at
    ``sample_rate_ghz``; waveforms of one shape at different amplitudes
    share it (``shape_key``).
    ``waveform_memory``: ``"per_line"`` (each line has a memory of
    ``waveform_samples``) or ``"shared"`` (one memory serves every line);
    see ``WaveformMemoryScope``.
    ``tones_per_line``: distinct tone frequencies one line plays.
    ``max_amplitude``: the largest magnitude of a sample its lines play
    (``None``: no limit). The compiler sets no amplitude limit of its own;
    a drive or readout whose envelope peaks above this one is a violation
    naming its line. Exports list it among the limits only when it is set.
    ``acquirers`` and ``acquirer_hold_cycles``: the acquirer reading each
    qudit; readouts on one acquirer are serialized, each holding it for its
    duration plus ``acquirer_hold_cycles``.
    ``feedback_latency_cycles``: the least time from the end of a readout to
    the start of an operation in a branch on its outcome.
    ``answer_timeouts``: whether the controller can stop waiting for an
    answer after a timeout (``result(timeout=...)``, ``latency_mode=
    "timeout"``); without, only waits that block or are budgeted are
    allowed.
    ``max_events``: the drives and readouts of one program.
    ``max_instructions``: the operations one program may hold, as the
    kernel traces them (``None``: no limit). An unrolled loop holds its
    body once per iteration, a rolled one once; tracing stops when the
    program grows past the limit, and the error names the loop that
    unrolled it. Exports list it among the limits only when it is set.
    ``max_program_duration``: the longest program the controller runs, a
    time or a number of ns (``None``: no limit); the compiler sums the
    program op after op, every iteration of its loops, and a program longer
    than that is a compile error at the op or loop that passes it (a wait
    whose length floats set, at the call). Without it, a program may last
    up to 2**60 cycles. Exports list it only when it is set.
    ``max_fine_delay_ns``: the longest delay, below one grid step, each line
    can add to its output (0: none), a time or a number of ns; it plays the
    part of a calibrated line delay (``MachineModel.line_delays_ns``) that
    whole grid steps cannot.
    ``alignment_tolerance_cycles``: without fine delay, how far a line's
    delay may be from a whole number of grid steps; the compiler rounds it
    to the nearest one.
    ``persistent_frames``: whether the frame of each tone, its frequency and
    phase, persists from one shot to the next. Without, every shot starts
    from the reset frame, and the export gives a frame change no later
    operation uses on some path as a frame update of its own; with, it
    exports instead the frame each tone ends the shot in (``frame_state``).

    What it computes, for the classical functions on ``compute_links`` (link
    names or ids; by default the link named ``"controller"``):
    ``op_cycles``: clock cycles per operation of each class of
    ``OP_CLASSES`` (``add``, ``mul``, ``div``, ``compare``, ``branch``,
    ``load``, ``convert``, ``popcount``, ``sqrt``, ``exp``, ``log``,
    ``trig``; a class not
    given takes one cycle), at ``compute_clock_ghz`` (``None``: the program's
    clock); with them, a function's longest path must fit its
    ``max_latency_ns``. ``max_ops``: the operations one function may hold.
    ``state_bytes``: the bytes of arguments, results and locals one function
    may hold. ``max_calls_per_shot`` and ``buffer_bytes``: the calls, and
    the bytes of requests and answers, one shot may make.
    ``controller_form``: every such function's body must be in controller
    form. Without any of these, compiling does not read function bodies.
    """

    clock_ghz: float | None = None
    cycles_per_timestep: int = 1
    issue_cycles: int = 1
    min_wait_cycles: int = 0
    sample_rate_ghz: float | None = None
    waveform_samples: int | None = None
    tones_per_line: int | None = None
    max_amplitude: float | None = None
    acquirers: Mapping[int, int] = field(default_factory=dict)
    acquirer_hold_cycles: int = 0
    feedback_latency_cycles: int = 0
    max_events: int | None = None
    max_fine_delay_ns: float | Time = 0.0
    alignment_tolerance_cycles: float = 0.0
    answer_timeouts: bool = True
    compute_links: tuple[int | str, ...] = ("controller",)
    op_cycles: Mapping[str, int] = field(default_factory=dict)
    compute_clock_ghz: float | None = None
    max_ops: int | None = None
    state_bytes: int | None = None
    max_calls_per_shot: int | None = None
    buffer_bytes: int | None = None
    controller_form: bool = False
    max_instructions: int | None = None
    max_program_duration: float | Time | None = None
    waveform_memory: WaveformMemoryScope | str = WaveformMemoryScope.PER_LINE
    persistent_frames: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.persistent_frames, bool):
            raise InvalidValueError(
                f"ControllerLimits.persistent_frames must be a bool, got {self.persistent_frames!r}"
            )
        if self.clock_ghz is not None and not self.clock_ghz > 0:
            raise ValueError(f"ControllerLimits.clock_ghz must be positive, got {self.clock_ghz}")
        try:
            scope = WaveformMemoryScope(self.waveform_memory)
        except ValueError:
            raise InvalidValueError(
                "ControllerLimits.waveform_memory must be 'per_line' or 'shared', "
                f"got {self.waveform_memory!r}"
            ) from None
        object.__setattr__(self, "waveform_memory", scope)
        steps = self.cycles_per_timestep
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise InvalidValueError(
                f"ControllerLimits.cycles_per_timestep must be an int of at least 1, got {steps!r}"
            )
        if self.issue_cycles < 1:
            raise ValueError("ControllerLimits.issue_cycles must be at least 1")
        for name in ("min_wait_cycles", "acquirer_hold_cycles", "feedback_latency_cycles"):
            if getattr(self, name) < 0:
                raise ValueError(f"ControllerLimits.{name} cannot be negative")
        object.__setattr__(self, "max_fine_delay_ns", as_nanoseconds(self.max_fine_delay_ns))
        if self.max_program_duration is not None:
            object.__setattr__(
                self, "max_program_duration", _longest_program(self.max_program_duration)
            )
        for name in ("max_fine_delay_ns", "alignment_tolerance_cycles"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise InvalidValueError(f"ControllerLimits.{name} must be a finite number >= 0")
        peak = self.max_amplitude
        if peak is not None and (
            isinstance(peak, bool)
            or not isinstance(peak, (int, float))
            or not math.isfinite(peak)
            or not peak > 0
        ):
            raise InvalidValueError(
                f"ControllerLimits.max_amplitude must be a positive number or None, got {peak!r}"
            )
        rate = self.sample_rate_ghz
        if rate is not None and (
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not math.isfinite(rate)
            or not rate > 0
        ):
            raise InvalidValueError(
                f"ControllerLimits.sample_rate_ghz must be a positive number or None, got {rate!r}"
            )
        for name in (
            "waveform_samples",
            "tones_per_line",
            "max_events",
            "max_ops",
            "state_bytes",
            "max_calls_per_shot",
            "buffer_bytes",
        ):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"ControllerLimits.{name} must be positive or None")
        size = self.max_instructions
        if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 1):
            raise InvalidValueError(
                f"ControllerLimits.max_instructions must be a positive int or None, got {size!r}"
            )
        if self.compute_clock_ghz is not None and not self.compute_clock_ghz > 0:
            raise ValueError(
                f"ControllerLimits.compute_clock_ghz must be positive, got {self.compute_clock_ghz}"
            )
        unknown = sorted(set(self.op_cycles) - set(OP_CLASSES))
        if unknown:
            raise ValueError(
                f"ControllerLimits.op_cycles has unknown classes {unknown}; "
                f"the classes are {', '.join(OP_CLASSES)}"
            )
        if any(int(v) < 0 for v in self.op_cycles.values()):
            raise ValueError("ControllerLimits.op_cycles cannot be negative")
        object.__setattr__(self, "acquirers", MappingProxyType(dict(self.acquirers)))
        object.__setattr__(
            self, "op_cycles", MappingProxyType({k: int(v) for k, v in self.op_cycles.items()})
        )
        object.__setattr__(self, "compute_links", tuple(self.compute_links))

    def __hash__(self) -> int:
        return hash(
            tuple(
                tuple(sorted(value.items())) if isinstance(value, Mapping) else value
                for value in (getattr(self, f.name) for f in fields(self))
            )
        )

    def __reduce__(self) -> tuple[Any, ...]:
        from ._pickling import frozen_reduce

        return frozen_reduce(self)

    @property
    def _runs_loops(self) -> bool:
        """Whether a rolled loop's body counts once against ``max_events``."""
        return False

    @property
    def computes(self) -> bool:
        """Whether any limit on what the controller computes is set."""
        return bool(
            self.op_cycles
            or self.controller_form
            or self.max_ops is not None
            or self.state_bytes is not None
            or self.max_calls_per_shot is not None
            or self.buffer_bytes is not None
        )

    def as_dict(self) -> dict[str, Any]:
        """The limits as JSON-serializable data."""
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data["acquirers"] = {str(k): v for k, v in sorted(self.acquirers.items())}
        data["op_cycles"] = dict(sorted(self.op_cycles.items()))
        data["compute_links"] = list(self.compute_links)
        for name in ("max_amplitude", "max_instructions", "max_program_duration"):
            if data[name] is None:
                del data[name]
        # Memory per line exports as it did before the key existed.
        if self.waveform_memory is WaveformMemoryScope.PER_LINE:
            del data["waveform_memory"]
        else:
            data["waveform_memory"] = str(self.waveform_memory)
        if not self.persistent_frames:
            del data["persistent_frames"]
        return data


def _longest_program(value: Any) -> float:
    """``ControllerLimits.max_program_duration`` in ns."""
    if not isinstance(value, bool) and isinstance(value, (int, float)):
        ns = float(as_nanoseconds(value))
        if math.isfinite(ns) and ns > 0:
            return ns
    raise InvalidValueError(
        "ControllerLimits.max_program_duration must be a positive time or number of "
        f"nanoseconds, or None; got {value!r}"
    )


# ── Waveforms ────────────────────────────────────────────────────────────────


def _number(value: float) -> float | None:
    return None if math.isnan(value) else float(value)


def _tree(nodes: Sequence[tuple[str, Sequence[float], Sequence[int]]]) -> dict[str, Any] | None:
    """The waveform expression of an event as nested dictionaries."""
    if not nodes:
        return None

    def build(index: int) -> dict[str, Any]:
        name, numbers, children = nodes[index]
        shape, _, custom = name.partition(":")
        node: dict[str, Any] = {"shape": shape}
        if custom:
            node["name"] = custom
        names = _PARAMETERS.get(shape)
        if shape == "custom_samples":
            node["samples"] = [float(v) for v in numbers]
        elif names is not None and len(names) == len(numbers):
            node["params"] = {k: _number(v) for k, v in zip(names, numbers, strict=True)}
        elif numbers:
            node["values"] = [_number(v) for v in numbers]
        if children:
            node["of"] = [build(child) for child in children]
        return node

    return build(len(nodes) - 1)


def _amplitude(node: Mapping[str, Any] | None) -> complex | None:
    """The overall complex amplitude of a waveform, through scale and neg."""
    if node is None:
        return None
    shape = node["shape"]
    if shape == "scale":
        inner = _amplitude(node["of"][0])
        factor = node.get("params", {}).get("factor")
        return None if inner is None or factor is None else inner * factor
    if shape == "neg":
        inner = _amplitude(node["of"][0])
        return None if inner is None else -inner
    if shape == "custom_samples":
        return complex(max(node["samples"], key=abs)) if node.get("samples") else None
    if shape not in _AMPLITUDES:
        return None
    params = node.get("params")
    if params is None:
        return None
    real = params.get("amplitude")
    imag = params.get("amplitude_imag", 0.0)
    if real is None or imag is None:
        return None
    return complex(real, imag)


def shape_key(node: Mapping[str, Any] | None) -> tuple[Any, ...] | None:
    """A key equal for waveforms that differ only in overall amplitude.

    Scale and negation drop out, and so does the amplitude of a built-in
    shape, known or not; custom samples are normalized to their largest
    magnitude. ``None`` when a parameter other than the amplitude is not
    known while the program compiles.
    """
    if node is None:
        return None
    shape = node["shape"]
    if shape in ("scale", "neg"):
        return shape_key(node["of"][0])
    if shape == "custom_samples":
        samples = node.get("samples", [])
        peak = max((abs(v) for v in samples), default=0.0) or 1.0
        return (shape, tuple(round(v / peak, 12) for v in samples))
    params = node.get("params")
    if params is None:
        children = tuple(shape_key(child) for child in node.get("of", ()))
        if any(child is None for child in children):
            return None
        return (shape, tuple(node.get("values", ())), children)
    skip = _AMPLITUDE_NAMES.get(shape, frozenset())
    kept = tuple((k, v) for k, v in params.items() if k not in skip)
    if any(v is None or isinstance(v, dict) for _, v in kept):
        return None
    return (shape, node.get("name"), kept)


def _exact(value: float) -> Fraction:
    """*value* as the decimal it is written as, so that rates such as 2.4
    samples per cycle divide exactly."""
    from fractions import Fraction

    return Fraction(repr(float(value)))


# The terms of the expressions of run-time values; see evaluate_expression.
_ARITHMETIC = frozenset({"add", "sub", "mul", "div", "neg", "max", "min"})
_COMPARISONS: Mapping[str, Any] = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
}
_INTEGER = frozenset({"floor_div", "rem"})
_CONVERSIONS = frozenset({"to_float", "to_int", "controller_cycles"})
#: Every ``op`` an exported expression of a run-time value may have.
EXPRESSION_TERMS = frozenset({*_ARITHMETIC, *_COMPARISONS, *_INTEGER, *_CONVERSIONS, "select"})
_TYPES = {"int": int, "float": float, "bool": bool}
_INT_MIN = -(1 << 63)


def evaluate_expression(
    expression: Any,
    readouts: Mapping[int, int] | None = None,
    answers: Mapping[int, Sequence[float | int | bool]] | None = None,
) -> float | int | bool:
    """The value an exported run-time *expression* takes, as a controller
    must compute it (see the controller backends guide for each term).

    *readouts* gives the level each readout event the expression reads
    observed, and *answers* the results of each answer event, both by the
    event index the expression names. A frame change's value may be a
    number, which is its value. Raises ``ExecutionError`` where the
    controller must fail the run (a division by zero, say), and
    ``InvalidValueError`` for an input not given or a term not in
    ``EXPRESSION_TERMS``.
    """
    value: float | int | bool = _evaluate(expression, readouts or {}, answers or {})
    return value


def _evaluate(
    expression: Any,
    readouts: Mapping[int, int],
    answers: Mapping[int, Sequence[float | int | bool]],
) -> Any:
    if not isinstance(expression, Mapping):
        if isinstance(expression, (int, float)):
            return expression
        raise InvalidValueError(f"not an expression: {expression!r}")
    if "const" in expression:
        return expression["const"]
    if "readout" in expression:
        index = expression["readout"]
        if index not in readouts:
            raise InvalidValueError(f"no level given for readout event {index}")
        return int(readouts[index])
    if "answer" in expression:
        index, result = expression["answer"], expression["result"]
        if index not in answers or not 0 <= result < len(answers[index]):
            raise InvalidValueError(f"no result {result} given for answer event {index}")
        return answers[index][result]
    op = expression.get("op")
    if op not in EXPRESSION_TERMS:
        raise InvalidValueError(f"{op!r} is not a term of the expression vocabulary")
    kind = _TYPES[expression["type"]]
    values = [_evaluate(x, readouts, answers) for x in expression["of"]]
    if op == "select":
        condition, then, other = values
        return kind(then if condition else other)
    if op == "to_float":
        return float(int(values[0]))
    if op == "to_int":
        x = float(values[0])
        if not math.isfinite(x) or not _INT_MIN <= math.trunc(x) < -_INT_MIN:
            raise _failed(f"to_int of {x!r}, which no 64-bit integer holds")
        return math.trunc(x)
    if op == "controller_cycles":
        cycles = int(values[0])
        if cycles % expression["timestep"]:
            raise _failed(
                f"{cycles} program cycles, not a whole number of the line's timesteps of "
                f"{expression['timestep']}"
            )
        return cycles * expression["num"] // expression["den"]
    values = [kind(v) for v in values]
    if op in _COMPARISONS:
        return _COMPARISONS[op](*values)
    if op == "neg":
        return _integer(-values[0]) if kind is int else -values[0]
    a, b = values
    if op == "max":
        return a if a > b else b
    if op == "min":
        return a if a < b else b
    if kind is float:
        if op == "div":
            if b == 0.0:
                if a != a or a == 0.0:
                    return math.nan
                return math.copysign(math.inf, a) * math.copysign(1.0, b)
            return a / b
        return a + b if op == "add" else a - b if op == "sub" else a * b
    if op in ("add", "sub", "mul"):
        return _integer(a + b if op == "add" else a - b if op == "sub" else a * b)
    if b == 0 or (a == _INT_MIN and b == -1):
        raise _failed(f"{op} of {a} by {b}")
    quotient = abs(a) // abs(b) * (1 if (a < 0) == (b < 0) else -1)
    if op == "div":
        return quotient
    if op == "floor_div":
        return a // b
    return a - b * quotient


def _integer(value: int) -> int:
    """*value* in 64-bit two's complement, as integer arithmetic wraps."""
    value &= (1 << 64) - 1
    return value - (1 << 64) if value >= 1 << 63 else value


def _failed(what: str) -> ExecutionError:
    return ExecutionError(f"the controller fails the run: {what}")


def sample_waveform(
    node: Mapping[str, Any], duration_cycles: int, samples_per_cycle: float = 1
) -> np.ndarray:
    """The complex envelope of a waveform, ``samples_per_cycle`` per cycle.

    Sample ``k`` is the envelope at ``k / samples_per_cycle`` cycles from the
    start, with the definitions of the waveform ops (times in cycles), the
    same the simulator plays. ``samples_per_cycle`` may be fractional, but
    the waveform must then last a whole number of samples. Raises
    ``ValueError`` when it does not, or for a waveform the export cannot
    evaluate: a ``pulse.custom`` op of hand-written IR, whose samples the
    runtime computes, or one whose parameters are not numbers.
    """
    import numpy as np

    real, imag = _sample(node, duration_cycles, _exact(samples_per_cycle))
    samples = np.empty(len(real), complex)
    samples.real = real
    samples.imag = imag
    return samples


def _sample(node: Mapping[str, Any], duration_cycles: int, ratio: Fraction | int) -> _Samples:
    """``sample_waveform``'s samples at *ratio* samples per cycle, as the
    lists of floats the export writes: the simulator's own envelopes,
    sampled natively, without NumPy."""
    # Absolute, so that the import costs no relative name's resolution.
    from cudaq_pulse._native._cudaq_pulse_native import sample_envelope

    # The kinds of the simulator's waveform callbacks (WaveformCallback.h).
    kinds = {
        "gaussian": 2,
        "drag": 3,
        "cosine": 4,
        "tanh_ramp": 5,
        "gaussian_square": 6,
        "custom_samples": 7,
    }
    per, over = ratio.numerator, ratio.denominator
    if int(duration_cycles) * per % over:
        raise InvalidValueError(
            f"{duration_cycles} cycles at {per / over:g} samples per cycle is "
            f"{int(duration_cycles) * per / over:g} samples, not a whole number"
        )
    count = int(duration_cycles) * per // over
    rate = per if over == 1 else per / over

    def evaluate(node: Mapping[str, Any]) -> _Samples:
        shape = node["shape"]
        children = [evaluate(child) for child in node.get("of", ())]
        params = node.get("params", {})
        if shape == "scale" and isinstance(params["factor"], (int, float)):
            (re, im), factor = children[0], params["factor"]
            return [x * factor for x in re], [y * factor for y in im]
        if shape == "neg":
            re, im = children[0]
            return [-x for x in re], [-y for y in im]
        if shape in ("add", "sub"):
            (are, aim), (bre, bim) = children
            sign = 1.0 if shape == "add" else -1.0
            return [x + sign * y for x, y in zip(are, bre, strict=True)], [
                x + sign * y for x, y in zip(aim, bim, strict=True)
            ]
        if shape == "mul":
            (are, aim), (bre, bim) = children
            return (
                [ar * br - ai * bi for ar, ai, br, bi in zip(are, aim, bre, bim, strict=True)],
                [ar * bi + ai * br for ar, ai, br, bi in zip(are, aim, bre, bim, strict=True)],
            )
        # The callback's parameters: start, duration, amplitude (real and
        # imaginary), sigma, beta, risefall, then the phase, quadrature and
        # rotation an envelope leaves at 0, then any samples.
        if shape == "custom_samples":
            held = (0.0, max(duration_cycles, 1), *[0.0] * 8, *node["samples"])
            return sample_envelope(kinds[shape], held, count, rate)
        if shape not in _AMPLITUDES or any(
            not isinstance(v, (int, float)) for v in params.values()
        ):
            raise ValueError(f"cannot sample a {shape!r} waveform without numeric parameters")
        if shape == "square":
            return [float(params["amplitude"])] * count, [float(params["amplitude_imag"])] * count
        get = params.get
        shaped = (get("sigma", 0.0), get("beta", 0.0), get("risefall", 0.0))
        envelope = (0.0, get("duration"), get("amplitude"), 0.0, *shaped, 0.0, 0.0, 0.0)
        return sample_envelope(kinds[shape], envelope, count, rate)

    return evaluate(node)


# ── Export ───────────────────────────────────────────────────────────────────


class _Waveforms:
    """The distinct waveforms of a format /2 program, in first use order."""

    def __init__(self) -> None:
        self.table: list[dict[str, Any] | None] = []
        self._index: dict[str, int] = {}

    def __call__(
        self, nodes: Sequence[tuple[str, Sequence[float], Sequence[int]]]
    ) -> tuple[int, None]:
        tree = _tree(nodes)
        key = repr(tree)
        index = self._index.get(key)
        if index is None:
            index = self._index[key] = len(self.table)
            self.table.append(tree)
        return index, None

    def inline(self, events: Sequence[dict[str, Any]]) -> None:
        """Give each event of *events* with ``runtime`` a waveform with
        those operands in place, keeping only the waveforms events play."""
        table: list[dict[str, Any] | None] = []
        index: dict[str, int] = {}
        for event in _played(events):
            tree = self.table[event["waveform"]]
            runtime = event.pop("runtime", None)
            if runtime is not None and tree is not None:
                tree = _with_expressions(tree, runtime)
            key = repr(tree)
            if key not in index:
                index[key] = len(table)
                table.append(tree)
            event["waveform"] = index[key]
        self.table, self._index = table, index

    def identified(self) -> list[dict[str, Any] | None]:
        """The table, each waveform with its ``id`` and ``shape_id``."""
        out: list[dict[str, Any] | None] = []
        for key, tree in zip(self._index, self.table, strict=True):
            if tree is None:
                out.append(None)
                continue
            entry = {"id": _digest(key), **tree}
            shape = shape_key(tree)
            if shape is not None:
                entry["shape_id"] = _digest(repr(shape))
            out.append(entry)
        return out


def _digest(text: str) -> str:
    """A content id: 16 hex digits of a hash of *text*."""
    return _blake2b(text.encode(), digest_size=8).hexdigest()


def _walk(
    lowered: Lowered,
    values: bool = True,
    own: _clocks.ControllerClock | None = None,
    waveforms: bool = True,
    slots: list[int] | None = None,
    summary: tuple[list[tuple[int, ...]], list[tuple[int, int, int]] | None] | None = None,
) -> tuple[
    list[Any],
    list[dict[str, Any]],
    dict[str, Any],
    Sequence[tuple[int, int, int]],
    list[str],
]:
    """The lines, events, and frames of *lowered* (``_with_frames``), at
    the values of its call; without *values*, a module with float parameters
    has ``None`` where they reach, and ``{"unknown": ...}`` for a value it
    computes at run time that the export has no term for.

    The lines are the dictionaries of format /1. With *summary*, (repeats,
    periods), for a check, the events are those the check walk gives: in a
    module that computes nothing at run time, a long rolled loop whose
    iterations repeat has its events once, and the list *repeats* gets its
    copies (``_loops``); with a list *periods* too, a loop whose lines move
    by different periods as well, and *periods* gets their (repeat, line,
    period). With *own* too, only loops whose periods are whole timesteps of
    its grid, their periods in its cycles.

    With *own*, the events' times are in that controller's cycles (their
    waveforms' too, with *waveforms*), what they compute at run time too
    (``_clocks.own_run_time``), and the line events off its grid are
    returned as (index, start, duration) in program cycles, with the
    violations of what they compute at run time. *slots* gets the
    runtime-qudit slot of each line (-1 for none)."""
    if values:
        if lowered.parameters:
            raise ValueError(
                "the module has float parameters "
                f"({', '.join(lowered.parameters)}); lower it with their values first"
            )
        if lowered._role is not None and lowered._unbound():
            raise InvalidValueError("the module has runtime qudits; lower it with handles first")
    # Waveforms and frames depend on the call's values, which only the
    # concrete module holds.
    module = lowered._concrete()
    off_grid: Sequence[tuple[int, int, int]] = ()
    problems: list[str] = []
    # What the summarizing walk computes at run time is in MLIR's ops, not
    # the export's vocabulary a controller's own cycles are converted in.
    if summary is not None and (own is None or not _computes_at_run_time(lowered)):
        repeats, periods = summary
        epochs = lowered._event_epochs(syncs=True)
        with lowered._lock:
            if own is None:
                lines, events = module.controller_events(
                    _PARAMETERS, _AMPLITUDES, epochs, slots, repeats, periods
                )
            else:
                ratio = own.ratio
                lines, events, off_grid, ends = module.controller_events_on(
                    _PARAMETERS,
                    _AMPLITUDES,
                    epochs,
                    ratio.numerator,
                    ratio.denominator,
                    own.grid,
                    slots,
                    repeats,
                    periods,
                )
                for index, end in ends:
                    events[index]["end_cycles"] = end
        frames: dict[str, Any] = {}
    else:
        # A check (without *values*) needs no value the export cannot name.
        with lowered._lock:
            if own is None:
                lines, events, frames = module.controller_program(
                    _PARAMETERS,
                    _AMPLITUDES,
                    lowered.classical != (),
                    slots=slots,
                    exportable=values,
                )
            else:
                ratio = own.ratio
                lines, events, frames, off_grid = module.controller_program(
                    _PARAMETERS,
                    _AMPLITUDES,
                    lowered.classical != (),
                    own=(ratio.numerator, ratio.denominator, own.grid),
                    slots=slots,
                    exportable=values,
                )
    if own is not None:
        from . import _clocks

        if waveforms and ratio != 1:
            # The events of one op share its waveform's parameters.
            factor = float(ratio)
            scaled: set[int] = set()
            for event in _played(events):
                tree = event["waveform"]
                if tree is not None:
                    _clocks._scale(tree, factor, scaled)
        if _computes_at_run_time(lowered):
            from . import _clocks

            for event in events:
                if event["kind"] in _clocks._GRIDDED:
                    problems.extend(
                        _clocks.own_run_time(own, lines[event["line"]], event, waveforms)
                    )
    fine = (lowered._generic or lowered).fine_delays
    if fine is not None:
        for line in lines:
            line["fine_delay_ns"] = fine.get((line["role"], line["qudit"]), 0.0)
    return lines, events, frames, off_grid, problems


def _walk_structured(
    lowered: Lowered, table: _Waveforms, slots: list[int] | None = None
) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any], tuple[()], list[str]]:
    """``_walk`` for format /2, on the program clock: the lines as (role,
    qudit), loops whose iterations repeat stay loops, and events give their
    waveform's index in *table*. *slots* gets each line's runtime-qudit
    slot (-1 for none)."""
    if lowered.parameters:
        raise InvalidValueError(
            "the module has float parameters "
            f"({', '.join(lowered.parameters)}); lower it with their values first"
        )
    if lowered._role is not None and lowered._unbound():
        raise InvalidValueError("the module has runtime qudits; lower it with handles first")
    module = lowered._concrete()
    with lowered._lock:
        lines, events, frames = module.controller_program(
            _PARAMETERS,
            _AMPLITUDES,
            lowered.classical != (),
            True,
            table,
            slots=slots,
        )
    if any("runtime" in event for event in _played(events)):
        table.inline(events)
    return lines, events, frames, (), []


def _computes_at_run_time(lowered: Lowered) -> bool:
    """Whether the program computes values at run time: from the answers of
    classical calls, or values a readout selects while it runs."""
    return bool(lowered.classical) or lowered.runtime_selects


def _native_clock(clocks: _clocks.ClockPlan | None) -> _clocks.ControllerClock | None:
    """The controller whose cycles the native walk counts the events in, when
    the program has one and it is not on the program clock or times in
    steps of more than one cycle."""
    if clocks is None or clocks.owners is not None:
        return None
    own = clocks.controllers[0]
    return None if own.ratio == 1 and own.grid == 1 else own


def _one_clock(clocks: _clocks.ClockPlan | None) -> bool:
    """Whether the program's lines all count program cycles one at a time."""
    return clocks is None or (clocks.owners is None and _native_clock(clocks) is None)


def _unknown_kinds(events: Sequence[Mapping[str, Any]]) -> list[str]:
    """The kinds of *events*, in loops too, that format /2 does not have,
    sorted."""
    found: set[str] = set()
    stack = [events]
    while stack:
        for event in stack.pop():
            if event["kind"] == "loop":
                stack.append(event["body"])
            elif event["kind"] not in _EVENT_KINDS:
                found.add(str(event["kind"]))
    return sorted(found)


def _played(events: Sequence[dict[str, Any]]) -> Any:
    """The drives and readouts of *events*, in the bodies of loops too."""
    for event in events:
        if event["kind"] == "loop":
            yield from _played(event["body"])
        elif event["kind"] in _PLAYED:
            yield event


def export(
    lowered: Lowered,
    sample_rate_ghz: float | None = None,
    controller: int | None = None,
    version: int = 1,
) -> dict[str, Any]:
    """A scheduled kernel for a controller backend, as JSON-serializable data.

    With *sample_rate_ghz*, each drive and readout also carries its
    envelope's samples at that rate (``samples``: real and imaginary parts),
    which must be a whole multiple of the clock. *version* 2 gives format
    ``cudaq-pulse.controller/2``: a loop whose iterations repeat is one
    ``loop`` event holding its body, each distinct waveform is listed once,
    and what follows from the rest is left out (``unroll`` gives format /1
    back). Raises ``ControllerLimitError`` when the machine model's
    controller limits are exceeded. On a control system of several
    controllers, the program of the one *controller* indexes (see
    ``_export_part``); without it, a program on more than one is a
    ``ValueError``.
    """
    from .classical.wire import _export as emulation

    machine = lowered.options.machine
    limits = machine.controller
    system = machine._system
    role = lowered._role
    if version != 1:
        if version != 2:
            raise InvalidValueError(f"version must be 1 or 2, got {version!r}")
        if system is not None:
            raise UnsupportedError(
                "format /2 is written for one controller; export version 1 for a control system"
            )
        if controller is not None:
            _controller_index(controller, 1)
        return _export_structured(lowered, sample_rate_ghz, emulation, role)
    plan: _ExportPlan | None = None
    slots: list[int] | None = None
    if system is not None or role is not None or controller is not None:
        if system is not None and limits is None:
            return _export_part(lowered, sample_rate_ghz, controller)
        if controller is not None:
            _controller_index(controller, 1)
        if role is not None:
            plan = role.export_plan
            if plan is None:
                slots = []
    clocks = (lowered._generic or lowered).clocks
    native = None if clocks is None else _native_clock(clocks)
    # In the controller's own cycles from here on.
    lines, events, frames, off, problems = _walk(lowered, True, native, True, slots)
    if system is not None:
        violations = system._violations(
            lowered, lines, events, clocks, off, problems, True, None, None
        )
        if violations:
            raise _limit_error(lowered, violations)
    elif limits is not None:
        _check_at_export(lowered, limits, clocks, (lines, events, frames, off, problems))
    if clocks is None:
        clock = lowered.target.clock_ghz
    else:
        from . import _clocks

        clock = clocks.controllers[0].clock_ghz
    if sample_rate_ghz is not None:
        _add_samples(events, sample_rate_ghz, clock)
    base = emulation(lowered, False)
    program = {
        "format": FORMAT,
        "target": lowered.target.name,
        "clock_ghz": clock,
        **({} if clocks is None else _clocks.clock_keys(clocks, clocks.controllers[0])),
        "schedule": lowered.options.schedule,
        "sample_rate_ghz": sample_rate_ghz,
        "qudits": list(lowered.qudits),
        "lines": lines,
        "events": events,
        "epochs": base["epochs"],
        "calls": base["calls"],
        "links": base["links"],
        "manifest": base["manifest"],
        "limits": None if limits is None else limits.as_dict(),
    }
    if role is not None:
        if plan is None or len(plan[1]) != len(events):
            if slots is None:
                slots = []
                lines = _walk(lowered, own=native, slots=slots)[0]
            plan = _export_plan(lowered, role, slots, lines, events)
        program["program_id"] = role.program_id
        program["symbols"] = _symbols(lowered, plan[0])
        for event, template in zip(events, plan[1], strict=True):
            if template is not None:
                symbols = template[0].copy()
                for key in template[1]:
                    symbols[key] = symbols[key][:]
                event["symbols"] = symbols
    if frames:
        program = _with_frames(program, frames, _slot_names(events, role))
    return program


def _with_frames(
    program: dict[str, Any], frames: Mapping[str, Any], slots: Mapping[int, str]
) -> dict[str, Any]:
    """*program* with what the native *frames* hold right after its
    ``events``: its ``frame_state``, then its ``runtime_tones``, of the tones
    on its ``lines`` (``_runtime_tones``; *slots* names a runtime qudit's)."""
    added = []
    if "frame_state" in frames:
        added.append(("frame_state", frames["frame_state"]))
    if "runtime_tones" in frames:
        table = _runtime_tones(frames["runtime_tones"], program["lines"], slots)
        if table:
            added.append(("runtime_tones", table))
    items = list(program.items())
    at = list(program).index("events") + 1
    return dict([*items[:at], *added, *items[at:]])


def _runtime_tones(
    found: Sequence[tuple[int, float | None, float | None]],
    lines: Sequence[Mapping[str, Any]],
    slots: Mapping[int, str],
) -> dict[str, dict[str, Any]]:
    """The ``runtime_tones`` of an export: each tone tracked at run time
    that *found* gives, on one of *lines* or on none, by its id, with its
    line and the frame every shot starts it in. On the line of a runtime
    qudit (its slot in *slots*, by line) the tone is the line's own, and its
    reset frequency is the symbol of the slot's native frequency."""
    roles = {line["line"]: line["role"] for line in lines}
    table = {}
    for tone, (line, frequency, phase) in enumerate(found):
        if line >= 0 and line not in roles:
            continue
        entry: dict[str, Any] = {
            "line": None if line < 0 else line,
            "reset_frequency_hz": frequency,
            "reset_phase": phase,
        }
        slot = slots.get(line)
        if slot is not None:
            native = "f01" if roles[line] == "drive" else "f_ro"
            entry["symbols"] = {"qudit": slot, "reset_frequency_hz": [f"{slot}.{native}"]}
        table[str(tone)] = entry
    return table


def _slot_names(events: Sequence[Mapping[str, Any]], role: Any) -> dict[int, str]:
    """The slot of each runtime qudit's line, as format /1 *events* name it."""
    if role is None:
        return {}
    return {
        event["line"]: event["symbols"]["qudit"]
        for event in events
        if "qudit" in event.get("symbols", ())
    }


def _check_at_export(
    lowered: Lowered,
    limits: ControllerLimits,
    clocks: Any,
    walked: tuple[Any, Any, Any, Any, Any] | None,
) -> None:
    """Raises ``ControllerLimitError`` for the limits an export exceeds.

    On one clock, compile() checked the module it scheduled, and a call
    whose values some limit depends on was checked when it bound them; on
    another, the events in the controller's cycles are checked here, from
    the format /1 *walked* (``None``: walk them).
    """
    if not _one_clock(clocks) or (not lowered.scheduled and not lowered.arguments):
        violations = _limit_violations(lowered, limits, clocks, walked)
        if violations:
            raise _limit_error(lowered, violations)


def _export_structured(
    lowered: Lowered, sample_rate_ghz: float | None, emulation: Any, role: Any = None
) -> dict[str, Any]:
    """``export`` in format ``cudaq-pulse.controller/2``; with the *role* of
    a module of list or qudit parameters, its symbols as format /1 gives
    them."""
    from . import _clocks

    waveforms = _Waveforms()
    limits = lowered.options.machine.controller
    clocks = (lowered._generic or lowered).clocks
    on_program_clock = clocks is None or (
        clocks.owners is None and clocks.controllers[0].ratio == 1
    )
    if not on_program_clock or (lowered._generic or lowered).fine_delays is not None:
        raise UnsupportedError(
            "format /2 is written for a controller on the program clock, without line "
            "delays; export version 1 for this machine"
        )
    slots: list[int] | None = None if role is None else []
    raw_lines, events, frames, _, _ = _walk_structured(lowered, waveforms, slots)
    if limits is not None:
        _check_at_export(lowered, limits, clocks, None)
    own = None if clocks is None else clocks.controllers[0]
    clock = lowered.target.clock_ghz if own is None else own.clock_ghz
    samples_of: list[dict[str, list[float]] | None] = []
    # The controller's converter rate, else sample_rate_ghz, else one sample
    # per cycle.
    own_rate = None if limits is None else limits.sample_rate_ghz
    rate = clock if own_rate is None else own_rate
    if sample_rate_ghz is not None:
        if not sample_rate_ghz > 0:
            raise InvalidValueError(f"sample_rate_ghz must be positive, got {sample_rate_ghz}")
        if own_rate is not None and sample_rate_ghz != own_rate:
            raise InvalidValueError(
                f"sample_rate_ghz={sample_rate_ghz}; the controller samples at {own_rate} GHz"
            )
        from fractions import Fraction

        ratio = Fraction(*_clocks.made_for(sample_rate_ghz, clock))
        rate = sample_rate_ghz
        samples_of = _add_samples(events, sample_rate_ghz, clock, waveforms, ratio)
    base = emulation(lowered, False)
    program: dict[str, Any] = {
        "format": _STRUCTURED,
        "target": lowered.target.name,
        "clock_ghz": clock,
    }
    if clocks is not None:
        program.update(_clocks.clock_keys(clocks, clocks.controllers[0]))
    program["sample_rate_ghz"] = rate
    program["schedule"] = lowered.options.schedule
    program["qudits"] = list(lowered.qudits)
    # A list, so that a line may later address many qudits.
    program["lines"] = [
        {"line": index, "role": role, "qudits": [qudit]}
        for index, (role, qudit) in enumerate(raw_lines)
    ]
    if clocks is not None and clocks.controllers[0].grid > 1 and _computes_at_run_time(lowered):
        _on_timesteps(clocks.controllers[0], program["lines"], events)
    program["waveforms"] = waveforms.identified()
    if sample_rate_ghz is not None:
        program["samples"] = samples_of
    generic = lowered._generic
    if generic is not None and generic.parameters:
        with lowered._lock:
            roles = generic._module.parameter_roles()
        program["parameters"] = {
            name: {"value": value, "reaches": reaches}
            for name, value, reaches in zip(
                generic.parameters, lowered.arguments, roles, strict=True
            )
        }
    program["events"] = events
    if frames:
        names = {} if slots is None else {i: f"s{k}" for i, k in enumerate(slots) if k >= 0}
        program = _with_frames(program, frames, names)
    program["epochs"] = base["epochs"]
    program["calls"] = base["calls"]
    program["links"] = base["links"]
    program["manifest"] = base["manifest"]
    program["controller"] = None if limits is None else _structured_controller(limits)
    if role is not None:
        assert slots is not None
        program["program_id"] = role.program_id
        program["symbols"] = _symbols(lowered, _symbol_plan(role, slots, program["lines"]))
        _structured_symbols(lowered, role, events, slots)
    program = {_SYSTEM_KEYS.get(k, k): v for k, v in program.items()}
    program["links"] = {
        k: {_LINK_KEYS.get(f, f): v for f, v in link.items()}
        for k, link in program["links"].items()
    }
    program["programs"] = {}
    program["bindings"] = {}
    return program


def _on_timesteps(
    own: _clocks.ControllerClock, lines: Sequence[Mapping[str, Any]], events: list[dict[str, Any]]
) -> None:
    """Give each wait of format /2 *events*, in loops too, whose length is
    computed at run time the length format /1 gives it on *own*'s
    timesteps, in place; the check at export has found any it can decide
    is off them."""
    from . import _clocks

    for event in events:
        if event["kind"] == "loop":
            _on_timesteps(own, lines, event["body"])
        elif event["kind"] == "wait" and "length" in event:
            _clocks.own_run_time(own, lines[event["line"]], event, False)


def _structured_controller(limits: ControllerLimits) -> dict[str, Any]:
    """The controller's fields under their format /2 names."""
    data = limits.as_dict()
    out = {_CONTROLLER_KEYS.get(k, k): v for k, v in data.items() if k not in _COMPUTE_KEYS}
    processor = {
        new: list(_PROCESSOR_DEFAULTS[new])
        if new == "unsupported"
        else _PROCESSOR_DEFAULTS[new]
        if old is None
        else data[old]
        for new, old in _PROCESSOR_KEYS
    }
    out["processors"] = [processor] if limits.computes else []
    return out


# The fields a controller exports only when they are set.
_OPTIONAL_KEYS = frozenset({"max_amplitude", "waveform_memory", "persistent_frames"})


def _flat_controller(data: Mapping[str, Any]) -> dict[str, Any]:
    """A format /2 controller under its /1 names, in /1 order."""
    processors = data["processors"]
    if len(processors) > 1:
        raise InvalidValueError("format /1 has at most one processor per controller")
    if processors:
        compute = {old: processors[0][new] for new, old in _PROCESSOR_KEYS if old is not None}
    else:
        compute = ControllerLimits().as_dict()
    return {
        f.name: compute[f.name]
        if f.name in _COMPUTE_KEYS
        else data[_CONTROLLER_KEYS.get(f.name, f.name)]
        for f in fields(ControllerLimits)
        if f.name in _COMPUTE_KEYS or _CONTROLLER_KEYS.get(f.name, f.name) in data
    }


def _with_expressions(
    tree: Mapping[str, Any], runtime: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """A copy of waveform *tree* with the operands known at run time
    (``path`` to a node, its ``params`` by name) as expressions in its
    parameters; one the node has no parameter for stays in its ``runtime``."""
    out = copy.deepcopy(dict(tree))
    for item in runtime:
        node = out
        for step in item["path"]:
            node = node["of"][step]
        params = node.get("params", {})
        for name, expression in item["params"].items():
            if name in params:
                params[name] = expression
            else:
                node.setdefault("runtime", {})[name] = expression
    return out


def _run_time_operands(tree: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """The operands of format /2 waveform *tree* that are expressions, as
    (``path``, ``params``) items in pre-order."""
    out: list[dict[str, Any]] = []

    def visit(node: Mapping[str, Any], path: list[int]) -> None:
        params = {k: v for k, v in node.get("params", {}).items() if isinstance(v, dict)}
        params.update(node.get("runtime", {}))
        if params:
            out.append({"path": list(path), "params": params})
        for i, child in enumerate(node.get("of", ())):
            visit(child, [*path, i])

    if tree is not None:
        visit(tree, [])
    return out


def _with_runtime(
    tree: Mapping[str, Any], runtime: Sequence[Mapping[str, Any]], resolve: Any = None
) -> dict[str, Any]:
    """A copy of waveform *tree* in format /1: each operand of *runtime*
    (paths to nodes, their operands by name) ``None`` in its node's
    parameters and, with *resolve*, resolved in the node's ``runtime``."""
    out = copy.deepcopy(dict(tree))
    for item in runtime:
        node = out
        for step in item["path"]:
            node = node["of"][step]
        node.pop("runtime", None)
        params = node.get("params", {})
        for name in item["params"]:
            if name in params:
                params[name] = None
        if resolve is not None:
            node["runtime"] = {k: resolve(v) for k, v in item["params"].items()}
    return out


def unroll(program: Mapping[str, Any]) -> dict[str, Any]:
    """A format /2 export in format /1: every event once per loop iteration.

    A program already in format /1 is returned as it is. Times, iterations
    and references are those of the /1 export of the same kernel; a phase
    is its first iteration's plus its steps, equal up to rounding; branches
    are numbered in the order they first appear.
    """
    if program["format"] == FORMAT:
        return dict(program)
    if program["format"] != _STRUCTURED:
        raise InvalidValueError(f"cannot unroll format {program['format']!r}")
    qudit_of = {line["line"]: line["qudits"][0] for line in program["lines"]}
    waveforms = [
        None if entry is None else {k: v for k, v in entry.items() if k not in _IDS}
        for entry in program["waveforms"]
    ]
    run_time = [_run_time_operands(tree) for tree in waveforms]
    waveforms = [
        _with_runtime(tree, operands) if tree is not None and operands else tree
        for tree, operands in zip(waveforms, run_time, strict=True)
    ]
    amplitudes = [_amplitude(tree) for tree in waveforms]
    samples = program.get("samples")
    # Every event of /2, loops included, by its place in pre-order: the
    # index references give.
    place: dict[int, int] = {}

    def number(items: Sequence[Mapping[str, Any]]) -> None:
        for item in items:
            place[id(item)] = len(place)
            if item["kind"] == "loop":
                number(item["body"])

    number(program["events"])
    unknown = _unknown_kinds(program["events"])
    if unknown:
        raise InvalidValueError(f"cannot unroll events of kind {', '.join(map(repr, unknown))}")
    events: list[dict[str, Any]] = []
    latest: dict[int, int] = {}
    branches: dict[tuple[Any, ...], int] = {}

    def resolve(expression: Any) -> Any:
        """A run-time expression with its /2 places as /1 indices."""
        if not isinstance(expression, dict):
            return expression
        if "readout" in expression:
            return {"readout": latest[expression["readout"]]}
        if "answer" in expression:
            return {"answer": latest[expression["answer"]], "result": expression["result"]}
        if "of" in expression:
            return {**expression, "of": [resolve(e) for e in expression["of"]]}
        return dict(expression)

    def frame_updates(
        updates: Sequence[Mapping[str, Any]], branch: Any, shift: int = 0
    ) -> list[dict[str, Any]]:
        """Frame changes with their /2 places as /1 indices, made *shift*
        cycles after their place in the loops' first iterations."""
        return [
            {
                **({"tone": update["tone"]} if "tone" in update else {}),
                "conditions": [
                    {
                        "answers": [latest[r] for r in c.get("answers", ())],
                        "arm": c["arm"],
                        "branch": branch(c),
                        **({"expression": resolve(c["expression"])} if "expression" in c else {}),
                        "readouts": [latest[r] for r in c.get("readouts", ())],
                        "test": c["test"],
                    }
                    for c in update.get("conditions", ())
                ],
                "id": update["id"],
                "op": update["op"],
                "start_cycles": (
                    None if update["start_cycles"] is None else update["start_cycles"] + shift
                ),
                "value": resolve(update["value"]),
            }
            for update in updates
        ]

    def expand(items: Sequence[Mapping[str, Any]], loops: list[tuple[Any, ...]]) -> None:
        # loops: (depth, iteration, iterations since the first, period, the
        # branches around the loop) for each rolled loop around *items*.
        for item in items:
            kind = _FLAT_KINDS.get(item["kind"], item["kind"])
            if kind == "loop":
                outer = {c["branch"] for c in item.get("conditions", ())}
                depth = len(item.get("iteration", ()))
                first = item.get("first_iteration", 0)
                for j in range(item["trips"]):
                    loop = (depth, first + j, j, item["period_cycles"], outer)
                    expand(item["body"], [*loops, loop])
                continue
            shift = sum(j * period for _, _, j, period, _ in loops)
            start, duration = item["start_cycles"], item["duration_cycles"]
            if start is not None:
                start += shift
            iteration = list(item.get("iteration", ()))
            for depth, k, _, _, _ in loops:
                iteration[depth] = k

            def branch(condition: Mapping[str, Any]) -> int:
                inner = tuple(j for _, _, j, _, outer in loops if condition["branch"] not in outer)
                return branches.setdefault((condition["branch"], inner), len(branches))

            conditions = [
                {
                    "branch": branch(condition),
                    "arm": condition["arm"],
                    "readouts": [latest[r] for r in condition.get("readouts", ())],
                    "answers": [latest[r] for r in condition.get("answers", ())],
                    "test": condition["test"],
                    **(
                        {"expression": resolve(condition["expression"])}
                        if "expression" in condition
                        else {}
                    ),
                }
                for condition in item.get("conditions", ())
            ]
            if kind == "sync":
                line, qudit = None, None
            elif kind in ("submit", "answer"):
                line, qudit = item["link"], -1
            else:
                line, qudit = item["line"], qudit_of.get(item["line"], -1)
            event: dict[str, Any] = {
                "index": len(events),
                "kind": kind,
                "line": line,
                "qudit": qudit,
                "start_cycles": start,
                "duration_cycles": duration,
                "end_cycles": None if start is None or duration is None else start + duration,
                "iteration": iteration,
                "epoch": item.get("epoch", 0),
                "conditions": conditions,
            }
            if kind in _PLAYED or kind == "frame_update":
                (tone,) = item["tones"]
                phase = tone["phase_rad"]
                if phase is not None:
                    phase += sum(
                        j * step
                        for (_, _, j, _, _), step in zip(
                            loops, tone.get("phase_step_rad", ()), strict=False
                        )
                        if j and step is not None
                    )
            if kind == "frame_update":
                event["op"] = item["op"]
                event["value"] = item["value"]
                event["frequency_hz"] = tone["frequency_hz"]
                event["phase_rad"] = phase
            elif kind in _PLAYED:
                amplitude = amplitudes[item["waveform"]]
                event["waveform"] = waveforms[item["waveform"]]
                event["amplitude"] = (
                    None if amplitude is None else {"re": amplitude.real, "im": amplitude.imag}
                )
                event["frequency_hz"] = tone["frequency_hz"]
                event["phase_rad"] = phase
                if run_time[item["waveform"]]:
                    event["waveform"] = _with_runtime(
                        event["waveform"], run_time[item["waveform"]], resolve
                    )
            elif kind == "sync":
                event["lines"] = list(item["lines"])
            elif kind in ("submit", "answer"):
                event["link"] = item["link"]
                if "controller" in item:
                    event["controller"] = item["controller"]
                event["function"] = item["function"]
                if kind == "submit":
                    event["measurements"] = [latest[r] for r in item["measurements"]]
                else:
                    event["wait"] = item["wait"]
                    event["timeout_ns"] = item.get("timeout_ns")
            if "tone" in item:
                event["tone"] = item["tone"]
                event["frame_updates"] = frame_updates(item["frame_updates"], branch, shift)
                if "phase_offset_rad" in item:
                    event["phase_offset_rad"] = item["phase_offset_rad"]
            if "length" in item:
                event["length"] = resolve(item["length"])
                event["min_cycles"] = item.get("min_cycles")
                event["max_cycles"] = item.get("max_cycles")
            if "operands" in item:
                event["operands"] = [resolve(operand) for operand in item["operands"]]
            if kind in _PLAYED and samples is not None and samples[item["waveform"]] is not None:
                event["samples"] = samples[item["waveform"]]
            if "symbols" in item:
                event["symbols"] = {
                    k: v[:] if type(v) is list else v for k, v in item["symbols"].items()
                }
            latest[place[id(item)]] = event["index"]
            events.append(event)

    expand(program["events"], [])
    frame_state = [
        {
            **frame,
            "frame_updates": frame_updates(
                frame["frame_updates"],
                lambda c: branches.setdefault((c["branch"], ()), len(branches)),
            ),
        }
        if "frame_updates" in frame
        else dict(frame)
        for frame in program.get("frame_state", ())
    ]
    flat_keys = {"controller": "limits", **{v: k for k, v in _SYSTEM_KEYS.items()}}
    from . import _clocks

    # Format /1 states the rate after the schedule, and only when sampled.
    rate = program["sample_rate_ghz"] if "samples" in program else None
    if rate is not None and _clocks.made_for(rate, program["clock_ghz"])[1] != 1:
        raise InvalidValueError(
            f"format /1 samples at a whole multiple of the clock; this program samples "
            f"at {rate} GHz on a {program['clock_ghz']} GHz clock"
        )
    flat: dict[str, Any] = {}
    for k, v in program.items():
        if k in ("programs", "bindings", "waveforms", "samples", "parameters", "sample_rate_ghz"):
            continue
        flat[flat_keys.get(k, k)] = v
        if k == "schedule":
            flat["sample_rate_ghz"] = rate
    flat["format"] = FORMAT
    if flat["limits"] is not None:
        flat["limits"] = _flat_controller(flat["limits"])
    names = {v: k for k, v in _LINK_KEYS.items()}
    flat["links"] = {
        k: {names.get(f, f): v for f, v in link.items()} for k, link in flat["links"].items()
    }
    flat["lines"] = [
        {"line": line["line"], "role": line["role"], "qudit": line["qudits"][0]}
        for line in program["lines"]
    ]
    flat["events"] = events
    if "frame_state" in flat:
        flat["frame_state"] = frame_state
    return flat


def _add_samples(
    events: Sequence[dict[str, Any]],
    sample_rate_ghz: float,
    clock: float,
    waveforms: _Waveforms | None = None,
    ratio: Fraction | int | None = None,
) -> list[dict[str, list[float]] | None]:
    """Give each drive and readout of *events* its envelope's samples at
    *sample_rate_ghz*, a whole multiple of the *clock* its times count; with
    *waveforms* (format /2), the samples of each of its waveforms instead,
    at *ratio* samples per cycle."""
    if ratio is None:
        per_cycles = round(sample_rate_ghz / clock)
        if per_cycles < 1 or abs(sample_rate_ghz / clock - per_cycles) > 1e-9:
            raise ValueError(
                f"sample_rate_ghz={sample_rate_ghz} is not a whole multiple of the "
                f"{clock} GHz clock"
            )
        ratio = per_cycles
    # The events of one op share its waveform, so each is sampled once.
    sampled: dict[tuple[int, int], dict[str, list[float]]] = {}
    samples_of: list[dict[str, list[float]] | None] = []
    if waveforms is not None:
        samples_of = [None] * len(waveforms.table)
    for event in _played(events):
        index: int = event["waveform"] if waveforms is not None else -1
        tree = event["waveform"] if waveforms is None else waveforms.table[index]
        duration = event["duration_cycles"]
        if tree is None or duration is None:
            continue
        key = (id(tree), duration)
        if key not in sampled:
            try:
                real, imag = _sample(tree, duration, ratio)
            except ValueError as e:
                where = event.get("index", "of waveform")
                raise ValueError(f"event {where} ({event['kind']}): {e}") from None
            sampled[key] = {"re": real, "im": imag}
        if waveforms is None:
            event["samples"] = sampled[key]
        else:
            samples_of[index] = sampled[key]
    return samples_of


def _controller_index(controller: Any, count: int) -> int:
    """*controller*, checked as the index of one of *count* controllers."""
    if isinstance(controller, bool) or not isinstance(controller, int):
        raise InvalidTypeError(
            f"controller= is the index of a controller of the control system, got {controller!r}"
        )
    if not 0 <= controller < count:
        raise InvalidValueError(
            f"controller={controller}, and the program's control system has "
            + ("one controller (0)" if count == 1 else f"{count} controllers (0 to {count - 1})")
        )
    return controller


def _part(
    lowered: Lowered, controller: int | None
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[int | None],
    list[list[int]],
    int,
    dict[str, Any],
]:
    """Of a program on a control system of several controllers: its lines
    and events in program cycles, each line's controller, the controllers
    taking part in each event, and the controller whose program is wanted:
    *controller*, else the only one the program uses (``ValueError`` when it
    uses more), and the program's frames (``_walk``). A controller the
    program does not use gets an empty part."""
    system = lowered.options.machine._system
    assert system is not None
    lines, events, frames, _, _ = _walk(lowered, slots=None)
    owners, _ = system._owners(lines)
    parts = system._parts(events, owners)
    count = len(system.controllers)
    if controller is not None:
        index = _controller_index(controller, count)
    else:
        used = sorted({o for o in owners if o is not None})
        if len(used) > 1:
            raise InvalidValueError(
                f"the program plays lines of {len(used)} controllers ("
                + ", ".join(f"controllers[{i}]" for i in used)
                + f"); pass controller= (0 to {count - 1}) to export one controller's program"
            )
        if not used:
            # Without lines, the program is its calls.
            used = sorted({c for part in parts for c in part})
            if len(used) > 1:
                raise InvalidValueError(
                    f"the program's calls are made on {len(used)} controllers ("
                    + ", ".join(f"controllers[{i}]" for i in used)
                    + f"); pass controller= (0 to {count - 1}) to export one controller's "
                    "program"
                )
        index = used[0] if used else 0
    return lines, events, owners, parts, index, frames


def _export_part(
    lowered: Lowered, sample_rate_ghz: float | None, controller: int | None
) -> dict[str, Any]:
    """``export`` on a control system of several controllers: the program of
    one, in its own cycles, with its lines and limits and the events it
    takes part in (see ``ControlSystem._parts``); a sync lists its lines
    alone. Line and event ids are the program's, so an event names one
    another controller plays (a readout whose outcome it is sent) by the id
    that controller's program gives it."""
    import copy

    from .classical.wire import _export as emulation

    machine = lowered.options.machine
    system = machine._system
    assert system is not None
    clocks = (lowered._generic or lowered).clocks
    lines, events, owners, parts, index, frames = _part(lowered, controller)
    # The checks count each controller's events in its cycles, in place.
    checked = events if clocks is None else copy.deepcopy(events)
    violations = system._violations(lowered, lines, checked, clocks, (), (), True, None, None)
    if violations:
        raise _limit_error(lowered, violations)
    role = lowered._role
    plan: _ExportPlan | None = None if role is None else role.export_plan
    if role is not None:
        if plan is None or len(plan[1]) != len(events):
            slots: list[int] = []
            walked = _walk(lowered, slots=slots)[0]
            plan = _export_plan(lowered, role, slots, walked, events)
        for event, template in zip(events, plan[1], strict=True):
            if template is not None:
                symbols = template[0].copy()
                for key in template[1]:
                    symbols[key] = symbols[key][:]
                event["symbols"] = symbols
    mine = []
    for event, part in zip(events, parts, strict=True):
        if index in part:
            if event["kind"] == "sync":
                event["lines"] = [line for line in event["lines"] if owners[line] == index]
            mine.append(event)
    own = None
    keys: dict[str, float] = {}
    if clocks is not None:
        from . import _clocks

        own = clocks.controllers[index]
        keys = _clocks.clock_keys(clocks, own)
        if own.ratio != 1 or own.grid > 1:
            alone = _clocks.ClockPlan(clocks.program_ghz, (own,), clocks.reference_clock_ghz)
            _clocks.off_grid(alone, lines, mine, True, True, _computes_at_run_time(lowered))
    clock = lowered.target.clock_ghz if own is None else own.clock_ghz
    if sample_rate_ghz is not None:
        _add_samples(mine, sample_rate_ghz, clock)
    functions = frozenset(e["function"] for e in mine if e["kind"] in ("submit", "answer"))
    base = emulation(lowered, False, (index, own, functions, ()))
    played = [line for line, owner in zip(lines, owners, strict=True) if owner == index]
    exported = {
        "format": FORMAT,
        "target": lowered.target.name,
        "clock_ghz": clock,
        **keys,
        "controller": index,
        "schedule": lowered.options.schedule,
        "sample_rate_ghz": sample_rate_ghz,
        "qudits": sorted({line["qudit"] for line in played}),
        "lines": played,
        "events": mine,
        "epochs": base["epochs"],
        "calls": base["calls"],
        "links": base["links"],
        "manifest": base["manifest"],
        "limits": system.controllers[index].as_dict(),
    }
    if role is not None:
        assert plan is not None
        exported["program_id"] = role.program_id
        exported["symbols"] = _symbols(lowered, plan[0])
    if "runtime_tones" in frames:
        # A controller's part has the run-time tones of its lines, and no
        # frame_state.
        tones = {"runtime_tones": frames["runtime_tones"]}
        exported = _with_frames(exported, tones, _slot_names(events, role))
    return exported


_DEPENDENT_FIELDS = ("frequency_hz", "phase_rad", "waveform")
# The kinds of event, in either format, that name their line's qudit.
_ON_LINE = _PLAYED | {"wait", "frame_update"}


# Where a symbol's value comes from in a call: its float arguments, the
# carrier bound to a slot, or that carrier's 0-1 or readout frequency.
_ARGUMENT, _CARRIER, _NATIVE, _READOUT = range(4)


def _symbols(
    lowered: Lowered, templates: list[tuple[dict[str, Any], int, int]]
) -> list[dict[str, Any]]:
    """Every symbol of a call's module, with its value in the call: its
    float arguments, its slots (the carriers bound) and the native
    frequencies of each slot that gets a line."""
    arguments = lowered.arguments
    carriers = lowered.carriers
    found = []
    for template, source, k in templates:
        entry = template.copy()
        if source == _ARGUMENT:
            entry["value"] = arguments[k]
        elif source == _CARRIER:
            entry["value"] = carriers[k]
        elif source == _NATIVE:
            entry["value"] = lowered.target.frequencies[carriers[k]]
        else:
            target = lowered.target
            carrier = carriers[k]
            entry["value"] = target._frequency_tables(carrier + 1)[2][carrier]
        found.append(entry)
    return found


# Each symbol without its value and where the value comes from; each event's
# symbols, with the keys whose values are lists.
_ExportPlan = tuple[
    list[tuple[dict[str, Any], int, int]],
    list[tuple[dict[str, Any], tuple[str, ...]] | None],
]


def _symbol_plan(
    role: Any, slots: Sequence[int], lines: Sequence[Mapping[str, Any]]
) -> list[tuple[dict[str, Any], int, int]]:
    """Each symbol of a role without its value, and where the value comes
    from (see ``_symbols``). By the runtime-qudit *slots* of *lines*, a slot
    with a drive line has its 0-1 frequency (``s0.f01``), and one with a
    readout line its readout frequency (``s0.f_ro``): the native frequency
    of each such line's tone."""
    lined = {(slot, line["role"]) for slot, line in zip(slots, lines, strict=True)}
    symbols: list[tuple[dict[str, Any], int, int]] = []
    argument = 0
    for name, kind, parameter, position in role.symbols:
        entry: dict[str, Any] = {"symbol": name, "kind": kind, "parameter": parameter}
        if position is not None:
            entry["position"] = position
        if kind == "slot":
            slot = int(name[1:])
            symbols.append((entry, _CARRIER, slot))
            if (slot, "drive") in lined:
                native = {"symbol": f"{name}.f01", "kind": "frequency", "slot": name}
                symbols.append((native, _NATIVE, slot))
            if (slot, "readout") in lined:
                native = {"symbol": f"{name}.f_ro", "kind": "frequency", "slot": name}
                symbols.append((native, _READOUT, slot))
        else:
            symbols.append((entry, _ARGUMENT, argument))
            argument += 1
    return symbols


def _named_dependencies(role: Any, found: Sequence[Sequence[Any]]) -> list[dict[str, list[str]]]:
    """The arguments of `main` each event depends on (see
    ``controller_dependencies``), as the names of their symbols by field."""
    names = [name for name, kind, _, _ in role.symbols if kind != "slot"]
    names += [f"s{k}.f01" for k in range(role.slots)]
    names += [f"s{k}.f_ro" for k in range(role.slots)]
    return [
        {
            field: [names[a] for a in arguments]
            for field, arguments in zip(_DEPENDENT_FIELDS, item, strict=True)
            if arguments
        }
        for item in found
    ]


def _structured_symbols(
    lowered: Lowered, role: Any, events: list[dict[str, Any]], slots: Sequence[int]
) -> None:
    """Gives each event of a format /2 program the symbols its fields depend
    on, as format /1 gives them; an event of a loop's body, those of any of
    its iterations. They come from the module the role compiled; a loop
    that rolls only at this call's values is ``UnsupportedError``."""
    structured = role.structured
    if structured is None:
        with lowered._lock:
            found = lowered._module.controller_rolled_dependencies()
        named = _named_dependencies(role, [item for _, _, item in found])
        structured = role.structured = [
            (_KINDS.get(kind, kind), trips, symbols)
            for (kind, trips, _), symbols in zip(found, named, strict=True)
        ]
    place = iter(structured)

    def give(items: list[dict[str, Any]]) -> bool:
        for event in items:
            entry = next(place, None)
            kind = event["kind"]
            if entry is None or entry[0] != kind or entry[1] != event.get("trips", -1):
                return False
            symbols: dict[str, Any] = {}
            line = event.get("line")
            if kind in _ON_LINE and line is not None:
                if 0 <= line < len(slots) and slots[line] >= 0:
                    symbols["qudit"] = f"s{slots[line]}"
            for key, names in entry[2].items():
                symbols[key] = names[:]
            if symbols:
                event["symbols"] = symbols
            if kind == "loop" and not give(event["body"]):
                return False
        return True

    if not give(events) or next(place, None) is not None:
        raise UnsupportedError(
            "format /2 cannot name the symbols of this call: a loop repeats only at "
            "its values; export version 1 for it"
        )


def _export_plan(
    lowered: Lowered,
    role: Any,
    slots: Sequence[int],
    lines: Sequence[Mapping[str, Any]],
    events: list[dict[str, Any]],
) -> _ExportPlan:
    """What a role's exports share, built once per role from a call's
    *events*, its *lines* and their runtime-qudit *slots*."""
    symbols = _symbol_plan(role, slots, lines)
    dependencies = role.dependencies
    if dependencies is None:
        with lowered._lock:
            found = lowered._module.controller_dependencies()
        dependencies = role.dependencies = _named_dependencies(role, found)
    aligned = len(dependencies) == len(events)
    templates: list[tuple[dict[str, Any], tuple[str, ...]] | None] = []
    for index, event in enumerate(events):
        template: dict[str, Any] = {}
        line = event["line"]
        if event["kind"] in _ON_LINE and line is not None:
            if 0 <= line < len(slots) and slots[line] >= 0:
                template["qudit"] = f"s{slots[line]}"
        if aligned:
            template.update(dependencies[index])
        lists = tuple(k for k, v in template.items() if type(v) is list)
        templates.append((template, lists) if template else None)
    plan: _ExportPlan = (symbols, templates)
    role.export_plan = plan
    return plan


# ── Checks ───────────────────────────────────────────────────────────────────


def _memory(limits: ControllerLimits) -> tuple[int, int, bool, bool, list[str], bool]:
    """The waveform memory of *limits* as the native check takes it:
    (samples, granule, shared, scaling, synthesized shapes, hold)."""
    samples = -1 if limits.waveform_samples is None else limits.waveform_samples
    shared = limits.waveform_memory is WaveformMemoryScope.SHARED
    return samples, 1, shared, True, [], False


def _checks_natively(limits: ControllerLimits, clocks: _clocks.ClockPlan | None) -> bool:
    """Whether the native check counts every limit of *limits* on *clocks*."""
    return _one_clock(clocks) and not (limits.max_events is not None and limits._runs_loops)


def _native_check(
    lowered: Lowered, limits: ControllerLimits, unresolved: list[str] | None = None
) -> list[str]:
    """The limits *lowered* exceeds as the native check finds them: a loop
    whose iterations repeat is checked once, its first iterations and its
    last standing for all of them, so a violation inside it is reported
    once. With *unresolved*, a tone or waveform whose value is not known yet
    counts only if the limit is exceeded whatever its value; a limit that
    only some values exceed is appended to *unresolved*."""
    num, den = _samples_per_own_cycle(limits, lowered.target.clock_ghz)
    module = lowered._concrete()
    found: list[str]
    with lowered._lock:
        found, maybe = module.controller_check(
            limits.issue_cycles,
            limits.issue_cycles + limits.min_wait_cycles,
            num,
            den,
            _memory(limits),
            -1 if limits.tones_per_line is None else limits.tones_per_line,
            -1 if limits.max_events is None else limits.max_events,
            dict(limits.acquirers),
            limits.acquirer_hold_cycles,
            limits.feedback_latency_cycles,
            limits.answer_timeouts,
            [fn.name for fn in lowered.classical if fn.latency_mode == "timeout"],
            unresolved is not None,
        )
    if maybe and unresolved is not None:
        unresolved.append("tone frequencies or waveform memory")
    # Any found sends the check to its walk, which counts the amplitudes too.
    if limits.max_amplitude is not None and not found:
        found.extend(_amplitude_violations(lowered, module, limits.max_amplitude, unresolved))
    return found


def _found_natively(
    lowered: Lowered, limits: ControllerLimits, unresolved: list[str] | None = None
) -> bool:
    """Whether *lowered* exceeds a limit of *limits* (``_checks_natively``):
    a check then lists its violations, one per execution, from a walk."""
    if _native_check(lowered, limits, unresolved):
        return True
    if limits.computes:
        from ._controller_cost import resources

        if resources(lowered, limits).violations:
            return True
    return (
        limits is not lowered.options.machine.controller
        and _sample_rates(lowered, limits) is not None
    )


def _limit_violations(
    lowered: Lowered,
    limits: ControllerLimits,
    clocks: _clocks.ClockPlan | None,
    walked: tuple[Any, Any, Any, Any, Any] | None = None,
    unresolved: list[str] | None = None,
) -> Sequence[str]:
    """The limits a call's schedule exceeds, as ``_violations`` finds them,
    one per execution. On one clock, without limits the native check does
    not count, the native check says whether there are any. They are listed
    from the events of *walked* (``_walk``'s format /1, in the controller's
    cycles), else of a walk made here that has a long rolled loop whose
    iterations repeat once (``_loops``).

    With *unresolved*, a tone or waveform whose value is not known yet
    counts only if the limit is exceeded whatever its value; a limit that
    only some values exceed is appended to *unresolved*."""
    if _checks_natively(limits, clocks):
        maybe: list[str] | None = None if unresolved is None else []
        if not _found_natively(lowered, limits, maybe):
            if unresolved is not None and maybe:
                unresolved.extend(maybe)
            return []
    if walked is not None:
        lines, events, _, off, problems = walked
        return _violations(lowered, events, limits, lines, clocks, off, problems, True, unresolved)
    repeats: list[tuple[int, ...]] = []
    for unsynced in (True, False):
        periods: list[tuple[int, int, int]] | None = [] if unsynced else None
        lines, events, _, off, problems = _walk(
            lowered,
            unresolved is None,
            _native_clock(clocks),
            False,
            summary=(repeats, periods),
        )
        if not repeats:
            break
        from ._loops import UnsyncedError, violations

        assert not problems
        own = _native_clock(clocks)
        try:
            return violations(
                lowered, events, repeats, limits, lines, unresolved, None, periods or (), own, off
            )
        except UnsyncedError:
            repeats = []
    return _violations(lowered, events, limits, lines, clocks, off, problems, False, unresolved)


def _exclusive(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Whether two events lie in different arms of one branch execution."""
    if not a["conditions"] or not b["conditions"]:
        return False
    arms = {(c["branch"], tuple(a["iteration"])): c["arm"] for c in a["conditions"]}
    for c in b["conditions"]:
        arm = arms.get((c["branch"], tuple(b["iteration"])))
        if arm is not None and arm != c["arm"]:
            return True
    return False


def _pairs(
    events: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Each event with the latest earlier event it can follow on one path."""
    ordered = sorted(events, key=lambda e: (e["start_cycles"], e["index"]))
    pairs = []
    for k in range(1, len(ordered)):
        event = ordered[k]
        for j in range(k - 1, -1, -1):
            if not _exclusive(ordered[j], event):
                pairs.append((ordered[j], event))
                break
    return pairs


def _violations(
    lowered: Lowered,
    events: Sequence[dict[str, Any]],
    limits: ControllerLimits,
    lines: Sequence[Mapping[str, Any]],
    clocks: _clocks.ClockPlan | None,
    off: Sequence[tuple[int, int, int]] = (),
    run_time: Sequence[str] = (),
    waveforms: bool = True,
    unresolved: list[str] | None = None,
    natives: Sequence[int] | None = None,
) -> list[str]:
    """The limits *events* exceed. With *clocks*, the events are in the
    controller's own cycles: the native walk's, with *off* its line events
    off the grid and *run_time* the violations of what they compute at run
    time, or else (lines on several controllers) converted here in place,
    their waveforms too unless *waveforms* is false (the limits count
    waveforms alike either way). With *unresolved*, a tone or waveform
    whose value is not known yet counts only if the limit is exceeded
    whatever its value; a limit that only some values exceed is appended to
    *unresolved* instead, to be checked with them. *natives* has, by event
    index, the qudit argument whose native frequency the event plays, or -1."""
    if clocks is None:
        found = []
    else:
        from . import _clocks

        if clocks.owners is None:
            found = _clocks.off_grid_messages(clocks.controllers[0], lines, events, off)
            found.extend(run_time)
        else:
            found = _clocks.off_grid(
                clocks, lines, events, True, waveforms, _computes_at_run_time(lowered)
            )
    timed: list[Mapping[str, Any]] = [
        e for e in events if e["kind"] in _PLAYED and e["end_cycles"] is not None
    ]
    if limits.max_events is not None and len(timed) > limits.max_events:
        found.extend(_event_count(timed, limits.max_events, limits._runs_loops))
    by_line: dict[int, list[Mapping[str, Any]]] = {}
    for event in timed:
        by_line.setdefault(event["line"], []).append(event)
    issue, gap = limits.issue_cycles, limits.issue_cycles + limits.min_wait_cycles
    num, den = _samples_per_own_cycle(limits, lowered.target.clock_ghz)
    shared = limits.waveform_memory is WaveformMemoryScope.SHARED
    stored: dict[Any, int] = {}
    unknown_stored = 0
    for line, items in sorted(by_line.items()):
        for previous, event in _pairs(items):
            distance = event["start_cycles"] - previous["start_cycles"]
            if distance != issue and distance < gap:
                found.append(
                    f"line {line}: {event['kind']} at {event['start_cycles']} starts "
                    f"{distance} cycles after the {previous['kind']} at "
                    f"{previous['start_cycles']}; the controller "
                    f"needs exactly {issue} or at least {gap}"
                )
        if limits.tones_per_line is not None:
            tones = {e["frequency_hz"] for e in items if e.get("frequency_hz") is not None}
            if len(tones) > limits.tones_per_line:
                found.append(
                    f"line {line}: {len(tones)} tone frequencies; the controller holds "
                    f"{limits.tones_per_line} per line"
                )
            elif unresolved is not None:
                unknown = [e for e in items if e.get("frequency_hz") is None]
                if natives is None:
                    count = len(unknown)
                else:
                    # A runtime qudit's own frequency is one tone however
                    # often it plays: an upper bound for every placement.
                    keys = [natives[e["index"]] for e in unknown]
                    count = len({k for k in keys if k >= 0}) + keys.count(-1)
                if len(tones) + count > limits.tones_per_line:
                    unresolved.append(f"line {line}: tone frequencies")
        if limits.max_amplitude is not None:
            found.extend(_over_amplitude(lines[line], items, limits.max_amplitude, unresolved))
        if limits.waveform_samples is not None:
            shapes = stored if shared else {}
            unknown_samples = 0
            for e in items:
                key = shape_key(e.get("waveform"))
                samples = -(-e["duration_cycles"] * num // den)
                if key is not None:
                    shapes[(key, e["duration_cycles"])] = samples
                elif unresolved is not None:
                    unknown_samples += samples
                else:
                    shapes[("event", e["index"])] = samples
            used = sum(shapes.values())
            if not shared and used > limits.waveform_samples:
                found.append(
                    f"line {line}: its waveforms take {used} samples; the controller's "
                    f"waveform memory holds {limits.waveform_samples}"
                )
            elif not shared and unresolved is not None:
                if used + unknown_samples > limits.waveform_samples:
                    unresolved.append(f"line {line}: waveform memory")
            unknown_stored += unknown_samples
    if shared and limits.waveform_samples is not None:
        used = sum(stored.values())
        if used > limits.waveform_samples:
            found.append(
                f"its waveforms take {used} samples; the controller's shared waveform "
                f"memory holds {limits.waveform_samples}"
            )
        elif unresolved is not None and used + unknown_stored > limits.waveform_samples:
            unresolved.append("waveform memory")
    if limits.acquirers:
        by_acquirer: dict[int, list[Mapping[str, Any]]] = {}
        for event in timed:
            if event["kind"] == "readout" and event["qudit"] in limits.acquirers:
                by_acquirer.setdefault(limits.acquirers[event["qudit"]], []).append(event)
        for acquirer, items in sorted(by_acquirer.items()):
            for previous, event in _pairs(items):
                free = previous["end_cycles"] + limits.acquirer_hold_cycles
                if event["start_cycles"] < free:
                    found.append(
                        f"acquirer {acquirer}: the readout of qudit {event['qudit']} at "
                        f"{event['start_cycles']} starts before the acquirer is free at {free} "
                        f"(the readout of qudit {previous['qudit']} at {previous['start_cycles']})"
                    )
    if limits.feedback_latency_cycles:
        for event in timed:
            for condition in event["conditions"]:
                for ref in condition["readouts"]:
                    readout = events[ref]
                    if readout["end_cycles"] is None:
                        continue
                    ready = readout["end_cycles"] + limits.feedback_latency_cycles
                    if event["start_cycles"] < ready:
                        found.append(
                            f"line {event['line']}: the {event['kind']} at {event['start_cycles']} "
                            f"depends on the readout ending at {readout['end_cycles']}; the "
                            f"controller needs {limits.feedback_latency_cycles} cycles of feedback "
                            f"latency (start at {ready} or later; schedule with rcp and "
                            "readout_latency_cycles >= feedback_latency_cycles)"
                        )
    if not limits.answer_timeouts:
        functions = {fn.name: fn for fn in lowered.classical}
        for event in events:
            if event["kind"] != "answer":
                continue
            fn = functions.get(event["function"])
            if event["wait"] == "timeout" or (fn is not None and fn.latency_mode == "timeout"):
                found.append(
                    f"the wait for {event['function']} at {event['start_cycles']} has a timeout; "
                    "the controller cannot stop waiting for an answer, so take it with "
                    "result() (no timeout)"
                )
    if limits.computes:
        from ._controller_cost import resources

        found.extend(resources(lowered, limits).violations)
    return found


def _over_amplitude(
    line: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    most: float,
    unresolved: list[str] | None = None,
) -> list[str]:
    """The events of *line* whose envelope peaks above *most*. With
    *unresolved*, an envelope whose peak needs the call's values is noted
    there, to be checked with them."""
    found = []
    for event in events:
        tree = event.get("waveform")
        peak = _peak_amplitude(tree)
        if peak is None and tree is not None and unresolved is not None:
            unresolved.append(f"line {line['line']}: amplitude")
            unresolved = None
        if peak is not None and peak > most:
            found.append(
                f"line {line['line']} ({line['role']} of qudit {line['qudit']}): the "
                f"{event['kind']} at {event['start_cycles']} peaks at amplitude {peak:.6g}; "
                f"the controller plays at most {most:g} (ControllerLimits.max_amplitude)"
            )
    return found


def _amplitude_violations(
    lowered: Lowered, module: Any, most: float, unresolved: list[str] | None
) -> list[str]:
    """``_over_amplitude`` on one clock, on the format /2 walk of *module*:
    each distinct waveform is sampled once, and a loop whose iterations
    repeat is checked once. With *unresolved*, a waveform a float parameter
    sets is noted there instead, to be checked with each call's values."""
    table = _Waveforms()
    with lowered._lock:
        lines, events, _ = module.controller_program(
            _PARAMETERS, _AMPLITUDES, lowered.classical != (), True, table, exportable=False
        )
    peaks: dict[int, float | None] = {}
    by_line: dict[int, list[str]] = {}
    for event in _played(events):
        index = event["waveform"]
        if index not in peaks:
            peaks[index] = _peak_amplitude(table.table[index])
            if peaks[index] is None and unresolved is not None and _unset(table.table[index]):
                unresolved.append(f"line {event['line']}: amplitude")
        peak = peaks[index]
        if peak is not None and peak > most:
            role, qudit = lines[event["line"]]
            by_line.setdefault(event["line"], []).append(
                f"line {event['line']} ({role} of qudit {qudit}): the "
                f"{_FLAT_KINDS.get(event['kind'], event['kind'])} at {event['start_cycles']} "
                f"peaks at amplitude {peak:.6g}; the controller plays at most {most:g} "
                "(ControllerLimits.max_amplitude)"
            )
    return [message for _, messages in sorted(by_line.items()) for message in messages]


def _unset(node: Mapping[str, Any]) -> bool:
    """Whether a value of waveform *node* waits for a float argument."""
    return None in node.get("params", {}).values() or any(
        _unset(child) for child in node.get("of", ())
    )


# Samples per cycle the peak of an envelope is found at.
_PEAK_SAMPLES_PER_CYCLE = 8


def _peak_amplitude(node: Mapping[str, Any] | None) -> float | None:
    """The largest magnitude of the envelope of *node*, over its own
    duration; ``None`` when it is not built from numbers alone."""
    if node is None:
        return None
    duration = _span(node)
    if duration is None:
        return None
    try:
        samples = sample_waveform(node, math.ceil(duration), _PEAK_SAMPLES_PER_CYCLE)
    except (ValueError, TypeError, KeyError):
        return None
    return float(abs(samples).max()) if len(samples) else 0.0


def _span(node: Mapping[str, Any]) -> float | None:
    """The duration, in cycles, a waveform tree states for itself."""
    duration = node.get("params", {}).get("duration")
    if duration is not None:
        return float(duration)
    if node["shape"] == "custom_samples":
        return float(len(node.get("samples", ())) or 1)
    for child in node.get("of", ()):
        found = _span(child)
        if found is not None:
            return found
    return None


def _samples_per_own_cycle(limits: ControllerLimits, program_ghz: float) -> tuple[int, int]:
    """The controller's samples per cycle of its clock, as (num, den)."""
    if limits.sample_rate_ghz is None:
        return 1, 1
    from . import _clocks

    return _clocks.made_for(limits.sample_rate_ghz, limits.clock_ghz or program_ghz)


def _event_count(timed: Sequence[Mapping[str, Any]], most: int, runs_loops: bool) -> list[str]:
    """The violation of ``max_events`` (*most*) by *timed*, which exceeds it
    unrolled; a controller that *runs_loops* holds a rolled body once."""
    if runs_loops:
        held = sum(1 for e in timed if not any(e["iteration"]))
        if held <= most:
            return []
        return [
            f"{held} drives and readouts, a rolled loop's body counted once; the "
            f"controller holds at most {most}"
        ]
    text = f"{len(timed)} drives and readouts; the controller plays at most {most}"
    loops = _rolled_loops(timed)
    if loops:
        grown = "; ".join(
            f"the loop of {trips} iterations from cycle {start} plays {extra} more"
            for trips, start, extra in loops
        )
        text += f". It does not support 'loop', so rolled loops are unrolled for it: {grown}"
    return [text]


def _rolled_loops(timed: Sequence[Mapping[str, Any]]) -> list[tuple[int, int, int]]:
    """Each outermost rolled loop of *timed* (in program order): its trip
    count, its first start cycle and the events past its first iteration."""
    loops: list[list[int]] = []
    previous = -1
    for e in timed:
        iteration = e["iteration"]
        if not iteration:
            previous = -1
            continue
        if previous < 0 or iteration[0] < previous:
            loops.append([0, e["start_cycles"], 0])
        loop = loops[-1]
        loop[0] = max(loop[0], iteration[0] + 1)
        loop[2] += any(iteration)
        previous = iteration[0]
    return [(trips, start, extra) for trips, start, extra in loops if extra]


def check(lowered: Lowered, limits: ControllerLimits | None = None) -> Sequence[str]:
    """The limits *lowered* exceeds, of *limits* or else its machine model's:
    a list, or past 10,000 violations in rolled loops a sequence that builds
    each one when read (``ControllerLimitError.violations``).

    Of a module with float parameters (``lowered.generic``), the limits it
    exceeds whatever their values; a call's values are checked when it is
    made, so a limit never fails first at export.
    """
    limits = limits if limits is not None else lowered.options.machine.controller
    if limits is None:
        return []
    return _compile_check(lowered, limits)[0]


def _compile_check(
    lowered: Lowered, limits: ControllerLimits | None, system: ControlSystem | None = None
) -> tuple[Sequence[str], bool]:
    """The limits *lowered* exceeds whatever its float values, and whether
    that is exact: ``False`` when some values may exceed others. With
    *system*, of its controllers (*limits* is then its only one, or ``None``)."""
    machine = lowered.options.machine
    if limits is machine.controller:
        clocks = (lowered._generic or lowered).clocks
    else:
        from . import _clocks

        clocks = _clocks.plan(limits, lowered.target.clock_ghz, machine.reference_clock_ghz)
    slots = lowered._role is not None and lowered._role.slots > 0
    if system is None and not slots:
        assert limits is not None
        maybe: list[str] = []
        if _checks_natively(limits, clocks) and not _found_natively(
            lowered, limits, maybe if lowered.parameters else None
        ):
            return [], not maybe
    repeats: list[tuple[int, ...]] | None = []
    # Loops of lines of different periods summarized too, then not, then none.
    passes: tuple[bool | None, ...] = (True, False, None)
    if system is not None:
        passes = (False, None)
        if not system._lines_up or slots and not system._plays_everything:
            repeats = None
        elif clocks is not None and clocks.owners is not None:
            # A controller counting cycles of its own is walked in full.
            if any(c.ratio != 1 for c in clocks.controllers):
                repeats = None
    for unsynced in passes:
        periods: list[tuple[int, int, int]] | None = [] if unsynced else None
        lines, events, _, off, problems = _walk(
            lowered,
            values=False,
            own=_native_clock(clocks),
            waveforms=False,
            summary=None if repeats is None else (repeats, periods),
        )
        if not repeats:
            break
        assert not problems
        from ._loops import summarized_check

        checked = summarized_check(
            lowered, limits, system, clocks, slots, lines, events, repeats, periods, off
        )
        if checked is not None:
            return checked
        repeats = [] if unsynced else None
    if system is not None and slots and not system._plays_everything:
        # Which controller plays a line depends on the qudits a call binds.
        return [], False
    unresolved: list[str] | None = [] if lowered.parameters or slots else None
    natives = None
    if slots and limits is not None and limits.tones_per_line is not None:
        with lowered._lock:
            natives = lowered._module.controller_native_tones()
    if system is None:
        assert limits is not None
        found = _violations(
            lowered, events, limits, lines, clocks, off, problems, False, unresolved, natives
        )
    else:
        found = system._violations(
            lowered, lines, events, clocks, off, problems, False, unresolved, natives
        )
    if slots and limits is not None and limits.acquirers:
        assert unresolved is not None
        # Acquirers are given per carrier, which each call binds.
        unresolved.append("acquirers")
    if limits is not None and limits is not machine.controller:
        found.extend(_sample_rate_violations(lowered, limits, clocks, events))
    return found, not unresolved


def _call_violations(lowered: Lowered) -> Sequence[str]:
    """The limits one call's values exceed, as its export checks them."""
    machine = lowered.options.machine
    clocks = (lowered._generic or lowered).clocks
    system = machine._system
    if system is not None:
        repeats: list[tuple[int, ...]] | None = []
        if not system._lines_up:
            repeats = None
        elif clocks is not None and clocks.owners is not None:
            # A controller counting cycles of its own is walked in full.
            if any(c.ratio != 1 for c in clocks.controllers):
                repeats = None
        # Summarized while the system's controllers line up, else every iteration.
        while True:
            lines, events, _, off, problems = _walk(
                lowered,
                own=_native_clock(clocks),
                summary=None if repeats is None else (repeats, None),
            )
            if not repeats:
                break
            from ._loops import finished

            assert not problems
            parts = system._summarized(
                lowered, lines, events, repeats, clocks, off, None, None, None
            )
            if parts is not None:
                return finished(parts)
            repeats = None
        return system._violations(lowered, lines, events, clocks, off, problems, True, None, None)
    assert machine.controller is not None
    return _limit_violations(lowered, machine.controller, clocks)


def _sample_rate_violations(
    lowered: Lowered,
    limits: ControllerLimits,
    clocks: _clocks.ClockPlan | None,
    events: Sequence[Mapping[str, Any]] | None = None,
) -> list[str]:
    """The sample arrays that *limits*' lines would play at another rate than
    the one they were made for: the rate of the lines *lowered* was compiled
    for. *events* are its events, else walked here."""
    rates = _sample_rates(lowered, limits)
    if rates is None:
        return []
    made_ghz, plays_ghz = rates
    if events is None:
        events = _walk(lowered, values=False, own=_native_clock(clocks), waveforms=False)[1]
    return [
        f"line {e['line']}: the {e['kind']} at {e['start_cycles']} plays samples made for "
        f"{made_ghz:g} GHz, and the controller's lines play {plays_ghz:g} GHz"
        for e in events
        if e["kind"] in _PLAYED and _has_samples(e.get("waveform"))
    ]


def _sample_rates(lowered: Lowered, limits: ControllerLimits) -> tuple[float, float] | None:
    """The rate, in GHz, the samples of *lowered* were made for and the one
    *limits*' lines play, when they differ."""
    from . import _clocks

    machine = lowered.options.machine
    made = None
    program = lowered.target.clock_ghz
    if machine.controller is not None:
        made = _clocks.sample_rate(machine.controller, program)
    plays = _clocks.sample_rate(limits, program)
    if made == plays:
        return None
    made_ghz = program if made is None else program * made[0] / made[1]
    plays_ghz = program if plays is None else program * plays[0] / plays[1]
    return made_ghz, plays_ghz


def _has_samples(node: Mapping[str, Any] | None) -> bool:
    if node is None:
        return False
    return node["shape"] == "custom_samples" or any(_has_samples(c) for c in node.get("of", ()))


# ── pulse.controller(...) ────────────────────────────────────────────────────


if TYPE_CHECKING:
    _Limits = ControllerLimits
    _Report = ResourceReport
    _Resources = FunctionResources
    _Usage = UsageReport


class ControllerNames(Protocol):
    """The names this module exports, as attributes: what a module
    satisfies of ``ControllerModule``, which it can't be called as."""

    EXPRESSION_TERMS: frozenset[str]
    FORMAT: str
    OP_CLASSES: tuple[str, ...]
    ControllerLimitError: type[ControllerLimitError]
    ControllerLimits: type[ControllerLimits]
    Event: type[Event]
    FunctionCost: type[FunctionCost]
    FunctionResources: type[FunctionResources]
    LimitExceeded: type[LimitExceeded]
    LinkUsage: type[LinkUsage]
    ResourceReport: type[ResourceReport]
    ControllerUsage: type[ControllerUsage]
    FunctionUsage: type[FunctionUsage]
    ProcessorUsage: type[ProcessorUsage]
    UsageReport: type[UsageReport]
    WaveformMemoryScope: type[WaveformMemoryScope]

    def analyze(self, function: Any, limits: _Limits | None = None) -> _Resources: ...
    def check(self, lowered: Lowered, limits: _Limits | None = None) -> Sequence[str]: ...
    def evaluate_expression(
        self,
        expression: Any,
        readouts: Mapping[int, int] | None = None,
        answers: Mapping[int, Sequence[float | int | bool]] | None = None,
    ) -> float | int | bool: ...
    def export(
        self,
        lowered: Lowered,
        sample_rate_ghz: float | None = None,
        controller: int | None = None,
        version: int = 1,
    ) -> dict[str, Any]: ...
    def resources(self, lowered: Lowered, limits: _Limits | None = None) -> _Report: ...
    def sample_waveform(
        self, node: Mapping[str, Any], duration_cycles: int, samples_per_cycle: float = 1
    ) -> np.ndarray: ...
    def shape_key(self, node: Mapping[str, Any] | None) -> tuple[Any, ...] | None: ...
    def unroll(self, program: Mapping[str, Any]) -> dict[str, Any]: ...
    def usage(self, lowered: Lowered) -> _Usage: ...


class ControllerModule(ControllerNames, Protocol):
    """What type checkers take ``pulse.controller`` to be: the loader of
    controller descriptions, with this module's names as attributes.

    ``pulse.controller("vendor.model", drive_lines=..., readout_lines=...)``
    is the ``Controller`` an installed description (or a TOML file, by path)
    states, with the installation's wiring; each file is parsed once. A
    module can't be declared callable, so ``cudaq_pulse`` declares its
    ``controller`` attribute with this type for checkers only; at run time
    it is this module.
    """

    def __call__(self, description: str | os.PathLike[str], /, **wiring: Any) -> Controller: ...


class _Loaded(str):
    """The name of a ``_controller_cost`` attribute, imported on first use and
    then kept in this module's namespace, where later reads find it as a plain
    attribute. Being its own name, it is built without a call into Python."""

    def __get__(self, instance: types.ModuleType | None, owner: type) -> Any:
        if instance is None:
            return self
        import cudaq_pulse._controller_cost as reports

        name = str(self)
        value = instance.__dict__[name] = getattr(reports, name)
        return value


class _ControllerModule(types.ModuleType):
    """This module, callable as ``pulse.controller(description, **wiring)``."""

    def __call__(self, description: str | os.PathLike[str], /, **wiring: Any) -> Controller:
        from ._system import load_controller

        return load_controller(description, **wiring)

    def __dir__(self) -> list[str]:
        loaded = [
            name for name, value in vars(_ControllerModule).items() if isinstance(value, _Loaded)
        ]
        return sorted({*super().__dir__(), *loaded})

    # An export builds no report and analyzes no function: building their
    # dataclasses and the analyzer costs more than the export itself.
    if TYPE_CHECKING:
        # ControllerModule's attributes are settable, and a descriptor with
        # no __set__ is not, to a checker.
        FunctionCost: Any
        ControllerUsage: Any
        FunctionResources: Any
        FunctionUsage: Any
        LimitExceeded: Any
        LinkUsage: Any
        ProcessorUsage: Any
        ResourceReport: Any
        UsageReport: Any
        analyze: Any
        format_report: Any
        resources: Any
        usage: Any

    FunctionCost = _Loaded("FunctionCost")
    ControllerUsage = _Loaded("ControllerUsage")
    FunctionResources = _Loaded("FunctionResources")
    FunctionUsage = _Loaded("FunctionUsage")
    LimitExceeded = _Loaded("LimitExceeded")
    LinkUsage = _Loaded("LinkUsage")
    ProcessorUsage = _Loaded("ProcessorUsage")
    ResourceReport = _Loaded("ResourceReport")
    UsageReport = _Loaded("UsageReport")
    analyze = _Loaded("analyze")
    format_report = _Loaded("format_report")
    resources = _Loaded("resources")
    usage = _Loaded("usage")


sys.modules[__name__].__class__ = _ControllerModule
