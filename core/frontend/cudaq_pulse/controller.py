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
changes the meaning of a key gets a new version.

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

import math
import os
import sys
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

from . import _controller_cost
from ._controller_cost import OP_CLASSES, FunctionCost
from ._errors import InvalidValueError
from ._units import Time, as_nanoseconds
from .kernel.ir_builder import CompilationError

if TYPE_CHECKING:
    import numpy as np

    from . import _clocks
    from ._system import Controller, ControlSystem
    from .compiler import Lowered

__all__ = [
    "FORMAT",
    "OP_CLASSES",
    "ControllerLimitError",
    "ControllerLimits",
    "FunctionCost",
    "FunctionResources",
    "LinkUsage",
    "ResourceReport",
    "analyze",
    "check",
    "export",
    "resources",
    "sample_waveform",
    "shape_key",
]

FORMAT = "cudaq-pulse.controller/1"

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
_PLAYED = frozenset({"drive", "readout"})


class ControllerLimitError(CompilationError):
    """A scheduled program exceeds a limit of the controller it is for."""

    def __init__(self, violations: Sequence[str], *, system: bool = False) -> None:
        self.violations = tuple(violations)
        shown = "\n  ".join(self.violations[:10])
        more = len(self.violations) - 10
        tail = f"\n  ... and {more} more" if more > 0 else ""
        what = "does not fit the control system" if system else "exceeds the controller's limits"
        super().__init__(f"the program {what}:\n  {shown}{tail}")


@dataclass(frozen=True)
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
    ``waveform_samples``: the waveform memory of a line holds this many
    samples, at ``sample_rate_ghz``; waveforms of one shape at different
    amplitudes share it (``shape_key``).
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
    ``max_fine_delay_ns``: the longest delay, below one grid step, each line
    can add to its output (0: none), a time or a number of ns; it plays the
    part of a calibrated line delay (``MachineModel.line_delays_ns``) that
    whole grid steps cannot.
    ``alignment_tolerance_cycles``: without fine delay, how far a line's
    delay may be from a whole number of grid steps; the compiler rounds it
    to the nearest one.

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

    def __post_init__(self) -> None:
        if self.clock_ghz is not None and not self.clock_ghz > 0:
            raise ValueError(f"ControllerLimits.clock_ghz must be positive, got {self.clock_ghz}")
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
        if self.max_amplitude is None:
            del data["max_amplitude"]
        return data


# ── Waveforms ────────────────────────────────────────────────────────────────


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
    skip = {_PARAMETERS[shape][i] for i in _AMPLITUDES.get(shape, ())}
    kept = tuple((k, v) for k, v in params.items() if k not in skip)
    if any(v is None for _, v in kept):
        return None
    return (shape, node.get("name"), kept)


def sample_waveform(
    node: Mapping[str, Any], duration_cycles: int, samples_per_cycle: int = 1
) -> np.ndarray:
    """The complex envelope of a waveform, ``samples_per_cycle`` per cycle.

    Sample ``k`` is the envelope at ``k / samples_per_cycle`` cycles from the
    start, with the definitions of the waveform ops (times in cycles), the
    same the simulator plays. Raises ``ValueError`` for a waveform the
    export cannot evaluate: a ``pulse.custom`` op of hand-written IR, whose
    samples the runtime computes, or one whose parameters are not numbers.
    """
    import numpy as np

    count = int(duration_cycles) * int(samples_per_cycle)
    t = np.arange(count, dtype=float) / samples_per_cycle

    def evaluate(node: Mapping[str, Any]) -> Any:
        shape = node["shape"]
        children = [evaluate(child) for child in node.get("of", ())]
        params = node.get("params", {})
        if shape == "scale":
            return children[0] * params["factor"]
        if shape == "neg":
            return -children[0]
        if shape == "add":
            return children[0] + children[1]
        if shape == "sub":
            return children[0] - children[1]
        if shape == "mul":
            return children[0] * children[1]
        if shape == "custom_samples":
            values = np.asarray(node["samples"], dtype=float)
            if not len(values):
                return np.zeros(count, complex)
            index = (t * len(values) / max(duration_cycles, 1)).astype(int)
            return values[np.minimum(index, len(values) - 1)].astype(complex)
        if shape not in _AMPLITUDES or any(v is None for v in params.values()):
            raise ValueError(f"cannot sample a {shape!r} waveform without numeric parameters")
        d = params["duration"]
        a = params["amplitude"]
        centered = t - d / 2.0
        if shape == "square":
            return np.full(count, complex(a, params["amplitude_imag"]))
        if shape == "gaussian":
            return (a * np.exp(-0.5 * centered**2 / params["sigma"] ** 2)).astype(complex)
        if shape == "drag":
            sigma = params["sigma"]
            g = a * np.exp(-0.5 * centered**2 / sigma**2)
            return g - 1j * params["beta"] * centered * g / sigma**2
        if shape == "cosine":
            return (a * 0.5 * (1.0 - np.cos(2.0 * math.pi * t / d))).astype(complex)
        if shape == "tanh_ramp":
            return (a * 0.5 * (1.0 + np.tanh((t - d / 2.0) / params["sigma"]))).astype(complex)
        sigma, edge = params["sigma"], params["risefall"]
        out = np.full(count, complex(a))
        rise = t < edge
        fall = t >= d - edge
        out[rise] = a * np.exp(-0.5 * (t[rise] - edge) ** 2 / sigma**2)
        out[fall] = a * np.exp(-0.5 * (t[fall] - (d - edge)) ** 2 / sigma**2)
        return out

    return np.asarray(evaluate(node), dtype=complex)


# ── Export ───────────────────────────────────────────────────────────────────


def _walk(
    lowered: Lowered,
    values: bool = True,
    own: _clocks.ControllerClock | None = None,
    waveforms: bool = True,
    slots: list[int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Sequence[tuple[int, int, int]], list[str]]:
    """The lines and events of *lowered*, at the values of its call; without
    *values*, a module with float parameters has ``None`` where they reach.

    With *own*, the events' times are in that controller's cycles (their
    waveforms' too, with *waveforms*), what they compute at run time too
    (``_clocks.own_run_time``), and the line events off its grid are
    returned as (index, start, duration) in program cycles, with the
    violations of what they compute at run time. *slots* gets the
    runtime-qudit slot of each line (-1 for none)."""
    if values and lowered.parameters:
        raise ValueError(
            "the module has float parameters "
            f"({', '.join(lowered.parameters)}); lower it with their values first"
        )
    if values and lowered._role is not None and lowered._unbound():
        raise InvalidValueError("the module has runtime qudits; lower it with handles first")
    # Waveforms and frames depend on the call's values, which only the
    # concrete module holds.
    module = lowered._concrete()
    epochs = lowered._event_epochs(syncs=True)
    off_grid: Sequence[tuple[int, int, int]] = ()
    problems: list[str] = []
    with lowered._lock:
        if own is None:
            lines, events = module.controller_events(_PARAMETERS, _AMPLITUDES, epochs, slots)
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
            )
    if own is not None:
        from . import _clocks

        for index, end in ends:
            events[index]["end_cycles"] = end
        if waveforms and ratio != 1:
            factor = float(ratio)
            for event in events:
                tree = event.get("waveform")
                if tree is not None:
                    _clocks._scale(tree, factor)
        if _computes_at_run_time(lowered):
            for event in events:
                if event["kind"] in _clocks._GRIDDED:
                    problems.extend(
                        _clocks.own_run_time(own, lines[event["line"]], event, waveforms)
                    )
    fine = (lowered._generic or lowered).fine_delays
    if fine is not None:
        for line in lines:
            line["fine_delay_ns"] = fine.get((line["role"], line["qudit"]), 0.0)
    return lines, events, off_grid, problems


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


def export(lowered: Lowered, sample_rate_ghz: float | None = None) -> dict[str, Any]:
    """A scheduled kernel for a controller backend, as JSON-serializable data.

    With *sample_rate_ghz*, each drive and readout also carries its
    envelope's samples at that rate (``samples``: real and imaginary parts),
    which must be a whole multiple of the clock. Raises
    ``ControllerLimitError`` when the machine model's controller limits are
    exceeded.
    """
    from .classical.emulation import export as emulation

    role = lowered._role
    plan: _ExportPlan | None = None if role is None else role.export_plan
    slots: list[int] | None = [] if role is not None and plan is None else None
    machine = lowered.options.machine
    limits = machine.controller
    system = machine._system
    clocks = (lowered._generic or lowered).clocks
    native = None if clocks is None else _native_clock(clocks)
    # In the controller's own cycles from here on.
    lines, events, off, problems = _walk(lowered, own=native, slots=slots)
    if limits is not None or system is not None:
        violations = _all_violations(lowered, lines, events, limits, system, clocks, off, problems)
        if violations:
            raise ControllerLimitError(violations, system=system is not None)
    own = None
    if clocks is not None:
        from . import _clocks

        own = clocks.controllers[0]
    clock = lowered.target.clock_ghz if own is None else own.clock_ghz
    if sample_rate_ghz is not None:
        ratio = sample_rate_ghz / clock
        per_cycles = round(ratio)
        if per_cycles < 1 or abs(ratio - per_cycles) > 1e-9:
            raise ValueError(
                f"sample_rate_ghz={sample_rate_ghz} is not a whole multiple of the "
                f"{clock} GHz clock"
            )
        for event in events:
            if event.get("waveform") is None or event["duration_cycles"] is None:
                continue
            try:
                samples = sample_waveform(event["waveform"], event["duration_cycles"], per_cycles)
            except ValueError as e:
                raise ValueError(f"event {event['index']} ({event['kind']}): {e}") from None
            event["samples"] = {"re": samples.real.tolist(), "im": samples.imag.tolist()}
    base = emulation(lowered, False)
    exported = {
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
                _walk(lowered, own=native, slots=slots)
            plan = _export_plan(lowered, role, slots, events)
        exported["program_id"] = role.program_id
        exported["symbols"] = _symbols(lowered, plan[0])
        for event, template in zip(events, plan[1], strict=True):
            if template is not None:
                symbols = template[0].copy()
                for key in template[1]:
                    symbols[key] = symbols[key][:]
                event["symbols"] = symbols
    return exported


_DEPENDENT_FIELDS = ("frequency_hz", "phase_rad", "waveform")


# Where a symbol's value comes from in a call: its float arguments, the
# carrier bound to a slot, or that carrier's native frequency.
_ARGUMENT, _CARRIER, _NATIVE = range(3)


def _symbols(
    lowered: Lowered, templates: list[tuple[dict[str, Any], int, int]]
) -> list[dict[str, Any]]:
    """Every symbol of a call's module, with its value in the call: its
    float arguments, its slots (the carriers bound) and the native frequency
    of each slot that gets a line."""
    arguments = lowered.arguments
    carriers = lowered.carriers
    found = []
    for template, source, k in templates:
        entry = template.copy()
        if source == _ARGUMENT:
            entry["value"] = arguments[k]
        elif source == _CARRIER:
            entry["value"] = carriers[k]
        else:
            entry["value"] = lowered.target.frequencies[carriers[k]]
        found.append(entry)
    return found


# Each symbol without its value and where the value comes from; each event's
# symbols, with the keys whose values are lists.
_ExportPlan = tuple[
    list[tuple[dict[str, Any], int, int]],
    list[tuple[dict[str, Any], tuple[str, ...]] | None],
]


def _export_plan(
    lowered: Lowered, role: Any, slots: Sequence[int], events: list[dict[str, Any]]
) -> _ExportPlan:
    """What a role's exports share, built once per role from a call's
    *events* and the runtime-qudit *slots* of its lines."""
    lined = set(slots)
    symbols: list[tuple[dict[str, Any], int, int]] = []
    argument = 0
    for name, kind, parameter, position in role.symbols:
        entry: dict[str, Any] = {"symbol": name, "kind": kind, "parameter": parameter}
        if position is not None:
            entry["position"] = position
        if kind == "slot":
            slot = int(name[1:])
            symbols.append((entry, _CARRIER, slot))
            if slot in lined:
                native = {"symbol": f"{name}.f01", "kind": "frequency", "slot": name}
                symbols.append((native, _NATIVE, slot))
        else:
            symbols.append((entry, _ARGUMENT, argument))
            argument += 1

    dependencies = role.dependencies
    if dependencies is None:
        with lowered._lock:
            found = lowered._module.controller_dependencies()
        names = [name for name, kind, _, _ in role.symbols if kind != "slot"]
        names += [f"s{k}.f01" for k in range(role.slots)]
        dependencies = role.dependencies = [
            {
                field: [names[a] for a in arguments]
                for field, arguments in zip(_DEPENDENT_FIELDS, item, strict=True)
                if arguments
            }
            for item in found
        ]
    aligned = len(dependencies) == len(events)
    templates: list[tuple[dict[str, Any], tuple[str, ...]] | None] = []
    for index, event in enumerate(events):
        template: dict[str, Any] = {}
        line = event["line"]
        if (event["kind"] in _PLAYED or event["kind"] == "wait") and line is not None:
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
        found.append(
            f"{len(timed)} drives and readouts; the controller plays at most {limits.max_events}"
        )
    by_line: dict[int, list[Mapping[str, Any]]] = {}
    for event in timed:
        by_line.setdefault(event["line"], []).append(event)
    issue, gap = limits.issue_cycles, limits.issue_cycles + limits.min_wait_cycles
    num, den = _samples_per_own_cycle(limits, lowered.target.clock_ghz)
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
            shapes: dict[Any, int] = {}
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
            if used > limits.waveform_samples:
                found.append(
                    f"line {line}: its waveforms take {used} samples; the controller's "
                    f"waveform memory holds {limits.waveform_samples}"
                )
            elif unresolved is not None and used + unknown_samples > limits.waveform_samples:
                unresolved.append(f"line {line}: waveform memory")
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


def check(lowered: Lowered, limits: ControllerLimits | None = None) -> list[str]:
    """The limits *lowered* exceeds, of *limits* or else its machine model's.

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
) -> tuple[list[str], bool]:
    """The limits *lowered* exceeds whatever its float values, and whether
    that is exact: ``False`` when some values may exceed others. With
    *system*, of its controllers (*limits* is then its only one, or ``None``)."""
    machine = lowered.options.machine
    if limits is machine.controller:
        clocks = (lowered._generic or lowered).clocks
    else:
        from . import _clocks

        clocks = _clocks.plan(limits, lowered.target.clock_ghz, machine.reference_clock_ghz)
    lines, events, off, problems = _walk(
        lowered, values=False, own=_native_clock(clocks), waveforms=False
    )
    slots = lowered._role is not None and lowered._role.slots > 0
    if system is not None and slots and not system._plays_everything:
        # Which controller plays a line depends on the qudits a call binds.
        return [], False
    unresolved: list[str] | None = [] if lowered.parameters or slots else None
    natives = None
    if slots and limits is not None and limits.tones_per_line is not None:
        with lowered._lock:
            natives = lowered._module.controller_native_tones()
    found = _all_violations(
        lowered,
        lines,
        events,
        limits,
        system,
        clocks,
        off,
        problems,
        waveforms=False,
        unresolved=unresolved,
        natives=natives,
    )
    if slots and limits is not None and limits.acquirers:
        assert unresolved is not None
        # Acquirers are given per carrier, which each call binds.
        unresolved.append("acquirers")
    if limits is not None and limits is not machine.controller:
        found.extend(_sample_rate_violations(lowered, limits, clocks, events))
    return found, not unresolved


def _all_violations(
    lowered: Lowered,
    lines: Sequence[Mapping[str, Any]],
    events: Sequence[dict[str, Any]],
    limits: ControllerLimits | None,
    system: ControlSystem | None,
    clocks: _clocks.ClockPlan | None,
    off: Sequence[tuple[int, int, int]] = (),
    run_time: Sequence[str] = (),
    waveforms: bool = True,
    unresolved: list[str] | None = None,
    natives: Sequence[int] | None = None,
) -> list[str]:
    if system is not None:
        return system._violations(
            lowered, lines, events, clocks, off, run_time, waveforms, unresolved, natives
        )
    assert limits is not None
    return _violations(
        lowered, events, limits, lines, clocks, off, run_time, waveforms, unresolved, natives
    )


def _call_violations(lowered: Lowered) -> list[str]:
    """The limits one call's values exceed, as its export checks them."""
    machine = lowered.options.machine
    clocks = (lowered._generic or lowered).clocks
    lines, events, off, problems = _walk(lowered, own=_native_clock(clocks))
    return _all_violations(
        lowered, lines, events, machine.controller, machine._system, clocks, off, problems
    )


def _sample_rate_violations(
    lowered: Lowered,
    limits: ControllerLimits,
    clocks: _clocks.ClockPlan | None,
    events: Sequence[Mapping[str, Any]],
) -> list[str]:
    """The sample arrays that *limits*' lines would play at another rate than
    the one they were made for: the rate of the lines *lowered* was compiled
    for."""
    from . import _clocks

    machine = lowered.options.machine
    made = None
    program = lowered.target.clock_ghz
    if machine.controller is not None:
        made = _clocks.sample_rate(machine.controller, program)
    plays = _clocks.sample_rate(limits, program)
    if made == plays:
        return []
    made_ghz = program if made is None else program * made[0] / made[1]
    plays_ghz = program if plays is None else program * plays[0] / plays[1]
    return [
        f"line {e['line']}: the {e['kind']} at {e['start_cycles']} plays samples made for "
        f"{made_ghz:g} GHz, and the controller's lines play {plays_ghz:g} GHz"
        for e in events
        if e["kind"] in _PLAYED and _has_samples(e.get("waveform"))
    ]


def _has_samples(node: Mapping[str, Any] | None) -> bool:
    if node is None:
        return False
    return node["shape"] == "custom_samples" or any(_has_samples(c) for c in node.get("of", ()))


# ── What the classical functions cost ───────────────────────────────────────

# A request and an answer carry a 24-byte header (classical.wire).
_HEADER_BYTES = 24


@dataclass(frozen=True)
class FunctionResources:
    """What one classical function costs, from ``resources`` or ``analyze``.

    ``cost`` is its body's static cost (operations by class on the longest
    path, size, locals, whether it is in controller form). ``state_bytes``
    adds its arguments, results and locals; ``request_bytes`` and
    ``answer_bytes`` are its messages with their headers. ``compute_ns`` is
    the longest path's time at the limits' ``op_cycles`` and clock (``None``
    without them), against ``max_latency_ns``. ``calls_per_shot`` counts its
    executions in one shot of the program (0 from ``analyze``).
    ``on_controller`` says whether it runs on the controller (its link is
    one of the limits' ``compute_links``).
    """

    function: str
    link: int | str
    on_controller: bool
    cost: FunctionCost
    state_bytes: int
    request_bytes: int
    answer_bytes: int
    max_latency_ns: float
    compute_cycles: int | None
    compute_ns: float | None
    calls_per_shot: int = 0

    @property
    def margin_ns(self) -> float | None:
        """How much of ``max_latency_ns`` the compute time leaves."""
        return None if self.compute_ns is None else self.max_latency_ns - self.compute_ns


@dataclass(frozen=True)
class LinkUsage:
    """One link's use in one shot: its calls, the most requests it holds
    at once against its slots, its largest message against its slot size,
    and the bytes of all its requests and answers."""

    link: int | str
    calls_per_shot: int
    peak_in_flight: int
    max_in_flight: int
    largest_message_bytes: int
    slot_bytes: int
    bytes_per_shot: int


@dataclass(frozen=True)
class ResourceReport:
    """What a program's classical functions cost the controller, per
    function and per shot, and the limits it exceeds (``violations``).
    ``str(report)`` is a table."""

    functions: tuple[FunctionResources, ...]
    links: tuple[LinkUsage, ...]
    calls_per_shot: int
    buffer_bytes: int
    limits: ControllerLimits | None
    violations: tuple[str, ...]

    def __str__(self) -> str:
        return format_report(self)


def _payload_bytes(types: Sequence[Any]) -> int:
    return sum(int(t.wire_bytes) for t in types)


def _value_lengths(function: Any) -> dict[str, int]:
    """The lengths of a function's sequence arguments, by parameter name."""
    import inspect

    implementation = function.implementation
    try:
        names = list(inspect.signature(implementation).parameters)
    except (TypeError, ValueError):
        return {}
    lengths = {}
    for name, spec in zip(names, function.args):
        if not spec.is_scalar:
            lengths[name] = int(spec.count)
    return lengths


def _link_key(function: Any) -> int | str:
    link: int | str = function.link
    return link


def _on_controller(function: Any, limits: ControllerLimits | None, machine: Any) -> bool:
    links = limits.compute_links if limits is not None else ("controller",)
    if function.link in links:
        return True
    if machine is not None:
        try:
            link_id = machine.link_id(function.link)
        except (KeyError, ValueError):
            return False
        return any(
            (isinstance(link, int) and link == link_id)
            or (
                isinstance(link, str)
                and link in machine.link_names
                and machine.link_id(link) == link_id
            )
            for link in links
        )
    return False


def _function_resources(
    function: Any,
    limits: ControllerLimits | None,
    clock_ghz: float | None,
    machine: Any = None,
    calls: int = 0,
) -> FunctionResources:
    if function.implementation is None:
        cost = FunctionCost(
            function.name, {}, 0, 0, 0, (), (f"{function.name}: external, with no body to read",)
        )
    elif function.uniform is not None:
        cost = FunctionCost(function.name, {"load": 1}, 1, 0, 0, (), ())
    else:
        cost = _controller_cost.analyze_callable(function.implementation, _value_lengths(function))
    args_bytes = _payload_bytes(function.args)
    results_bytes = _payload_bytes(function.results)
    state = args_bytes + results_bytes + _controller_cost.register_bytes(cost.locals)
    cycles = compute_ns = None
    if limits is not None and limits.op_cycles:
        cycles = cost.cycles(limits.op_cycles)
        rate = limits.compute_clock_ghz or limits.clock_ghz or clock_ghz
        if rate:
            compute_ns = cycles / rate
    return FunctionResources(
        function=function.name,
        link=_link_key(function),
        on_controller=_on_controller(function, limits, machine),
        cost=cost,
        state_bytes=state,
        request_bytes=_HEADER_BYTES + args_bytes,
        answer_bytes=_HEADER_BYTES + results_bytes,
        max_latency_ns=function.max_latency_ns,
        compute_cycles=cycles,
        compute_ns=compute_ns,
        calls_per_shot=calls,
    )


def analyze(function: Any, limits: ControllerLimits | None = None) -> FunctionResources:
    """What one classical function costs, on its own (see ``resources``)."""
    return _function_resources(function, limits, None)


def _function_violations(row: FunctionResources, limits: ControllerLimits) -> list[str]:
    found = []
    name = row.function
    if limits.controller_form and row.cost.violations:
        found.append(
            f"{name} is not in controller form: "
            + "; ".join(row.cost.violations[:4])
            + (
                f" (and {len(row.cost.violations) - 4} more)"
                if len(row.cost.violations) > 4
                else ""
            )
        )
    if limits.max_ops is not None and row.cost.size > limits.max_ops:
        found.append(
            f"{name} holds {row.cost.size} operations; the controller holds {limits.max_ops} "
            "per function"
        )
    if limits.state_bytes is not None and row.state_bytes > limits.state_bytes:
        found.append(
            f"{name} needs {row.state_bytes} bytes of state (arguments, results and "
            f"{row.cost.locals} locals); the controller holds {limits.state_bytes} per function"
        )
    if row.compute_ns is not None and row.compute_ns > row.max_latency_ns:
        found.append(
            f"{name} computes for about {row.compute_ns:.0f} ns ({row.compute_cycles} cycles on "
            f"its longest path), beyond its max_latency_ns={row.max_latency_ns:g}"
        )
    return found


def resources(lowered: Lowered, limits: ControllerLimits | None = None) -> ResourceReport:
    """What the classical functions of *lowered* cost the controller.

    *limits*, else the machine model's, give the cycle costs and the limits
    checked; every function is reported, and the limits apply to those on
    the controller's ``compute_links``. Calls per shot, slots in flight and
    buffer bytes come from the scheduled program (``classical_report``).
    """
    limits = limits if limits is not None else lowered.options.machine.controller
    machine = lowered.options.machine
    calls = lowered.classical_report() if lowered.classical and lowered.scheduled else []
    per_function: dict[str, int] = {}
    for call in calls:
        per_function[call.function] = per_function.get(call.function, 0) + 1
    rows = tuple(
        _function_resources(
            fn, limits, lowered.target.clock_ghz, machine, per_function.get(fn.name, 0)
        )
        for fn in lowered.classical
    )
    by_name = {row.function: row for row in rows}
    usage: dict[int, dict[str, Any]] = {}
    for call in calls:
        row = by_name.get(call.function)
        entry = usage.setdefault(
            call.link, {"calls": 0, "bytes": 0, "largest": 0, "spans": [], "name": call.link_name}
        )
        entry["calls"] += 1
        if row is not None:
            entry["bytes"] += row.request_bytes + row.answer_bytes
            entry["largest"] = max(entry["largest"], row.request_bytes, row.answer_bytes)
        end = call.taken_cycles if call.taken_cycles is not None else call.due_cycles
        entry["spans"].append((call.submit_cycles, max(end, call.submit_cycles + 1)))
    links = []
    for link_id, entry in sorted(usage.items()):
        edges = sorted([(a, 1) for a, _ in entry["spans"]] + [(b, -1) for _, b in entry["spans"]])
        peak = held = 0
        for _, step in edges:
            held += step
            peak = max(peak, held)
        link = machine.link(link_id)
        links.append(
            LinkUsage(
                link=entry["name"] if entry["name"] is not None else link_id,
                calls_per_shot=entry["calls"],
                peak_in_flight=peak,
                max_in_flight=link.max_in_flight,
                largest_message_bytes=entry["largest"],
                slot_bytes=link.slot_bytes,
                bytes_per_shot=entry["bytes"],
            )
        )
    found: list[str] = []
    controller_rows = [row for row in rows if row.on_controller]
    controller_calls = sum(row.calls_per_shot for row in controller_rows)
    controller_bytes = sum(
        row.calls_per_shot * (row.request_bytes + row.answer_bytes) for row in controller_rows
    )
    if limits is not None:
        for row in controller_rows:
            found.extend(_function_violations(row, limits))
        if limits.max_calls_per_shot is not None and controller_calls > limits.max_calls_per_shot:
            found.append(
                f"one shot makes {controller_calls} calls to the controller; it takes "
                f"{limits.max_calls_per_shot}"
            )
        if limits.buffer_bytes is not None and controller_bytes > limits.buffer_bytes:
            found.append(
                f"one shot's requests and answers to the controller take {controller_bytes} "
                f"bytes; its buffer holds {limits.buffer_bytes}"
            )
    for usage_row in links:
        if usage_row.peak_in_flight > usage_row.max_in_flight:
            found.append(
                f"link {usage_row.link!r} holds {usage_row.peak_in_flight} requests at once; it "
                f"has {usage_row.max_in_flight} slots"
            )
        if usage_row.largest_message_bytes > usage_row.slot_bytes:
            found.append(
                f"link {usage_row.link!r} carries a {usage_row.largest_message_bytes}-byte "
                f"message; its slots hold {usage_row.slot_bytes}"
            )
    return ResourceReport(
        functions=rows,
        links=tuple(links),
        calls_per_shot=len(calls),
        buffer_bytes=sum(entry["bytes"] for entry in usage.values()),
        limits=limits,
        violations=tuple(found),
    )


def format_report(report: ResourceReport) -> str:
    """The report as a table, one row per function and per link."""
    lines = ["Classical functions"]
    header = (
        f"  {'function':<24} {'link':<11} {'ops':>5} {'size':>5} {'state B':>7} "
        f"{'compute':>9} {'budget':>8} {'calls':>5}  form"
    )
    lines.append(header)
    for row in report.functions:
        compute = "-" if row.compute_ns is None else f"{row.compute_ns:.0f} ns"
        form = "ok" if row.cost.in_controller_form else "no"
        lines.append(
            f"  {row.function:<24} {str(row.link):<11} {row.cost.total:>5} {row.cost.size:>5} "
            f"{row.state_bytes:>7} {compute:>9} {row.max_latency_ns:>5.0f} ns "
            f"{row.calls_per_shot:>5}  {form}"
        )
        ops = ", ".join(f"{k} {n}" for k, n in row.cost.ops.items() if n)
        if ops:
            lines.append(f"      {ops}")
    if report.links:
        lines.append("Links, per shot")
        lines.append(f"  {'link':<11} {'calls':>5} {'in flight':>9} {'largest B':>9} {'bytes':>7}")
        for usage in report.links:
            lines.append(
                f"  {str(usage.link):<11} {usage.calls_per_shot:>5} "
                f"{usage.peak_in_flight:>4}/{usage.max_in_flight:<4} "
                f"{usage.largest_message_bytes:>4}/{usage.slot_bytes:<4} {usage.bytes_per_shot:>7}"
            )
    lines.append(f"Per shot: {report.calls_per_shot} calls, {report.buffer_bytes} bytes")
    if report.violations:
        lines.append("Exceeds:")
        lines.extend(f"  {v}" for v in report.violations)
    return "\n".join(lines)


_Limits = ControllerLimits
_Report = ResourceReport
_Resources = FunctionResources


# ── pulse.controller(...) ────────────────────────────────────────────────────


class ControllerModule(Protocol):
    """What type checkers take ``pulse.controller`` to be: the loader of
    controller descriptions, with this module's names as attributes.

    ``pulse.controller("vendor.model", drive_lines=..., readout_lines=...)``
    is the ``Controller`` an installed description (or a TOML file, by path)
    states, with the installation's wiring; each file is parsed once. A
    module can't be declared callable, so ``cudaq_pulse`` declares its
    ``controller`` attribute with this type for checkers only; at run time
    it is this module.
    """

    FORMAT: str
    OP_CLASSES: tuple[str, ...]
    ControllerLimitError: type[ControllerLimitError]
    ControllerLimits: type[ControllerLimits]
    FunctionCost: type[FunctionCost]
    FunctionResources: type[FunctionResources]
    LinkUsage: type[LinkUsage]
    ResourceReport: type[ResourceReport]

    def __call__(self, description: str | os.PathLike[str], /, **wiring: Any) -> Controller: ...
    def analyze(self, function: Any, limits: _Limits | None = None) -> _Resources: ...
    def check(self, lowered: Lowered, limits: _Limits | None = None) -> list[str]: ...
    def export(self, lowered: Lowered, sample_rate_ghz: float | None = None) -> dict[str, Any]: ...
    def format_report(self, report: _Report) -> str: ...
    def resources(self, lowered: Lowered, limits: _Limits | None = None) -> _Report: ...
    def sample_waveform(
        self, node: Mapping[str, Any], duration_cycles: int, samples_per_cycles: int = 1
    ) -> np.ndarray: ...
    def shape_key(self, node: Mapping[str, Any] | None) -> tuple[Any, ...] | None: ...


class _ControllerModule(types.ModuleType):
    """This module, callable as ``pulse.controller(description, **wiring)``."""

    def __call__(self, description: str | os.PathLike[str], /, **wiring: Any) -> Controller:
        from ._system import load_controller

        return load_controller(description, **wiring)


sys.modules[__name__].__class__ = _ControllerModule
