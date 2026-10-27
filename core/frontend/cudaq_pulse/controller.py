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
format ``cudaq-pulse.controller/1``: the lines and the qudits they belong to,
and every drive, readout, wait and classical call with its exact start and
duration, its waveform (the shape and its parameters, and optionally its
samples), its tone's frequency and frame phase, the measurement-conditioned
branches it lies in, and the epochs and calls of
``Lowered.export_emulation()``. The schedule is final: a backend places each
event at its ``start_vtu`` and does not schedule again.

``ControllerLimits`` states what a controller can play. Given to the
machine model, ``MachineModel(controller=ControllerLimits(...))``, it makes
``compile()`` check every scheduled program against it and raise
``ControllerLimitError`` naming each violated limit, so that a program the
controller cannot play fails at compile time. ``check`` returns the
violations without raising.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from .kernel.ir_builder import CompilationError

if TYPE_CHECKING:
    import numpy as np

    from .compiler import Lowered

__all__ = [
    "FORMAT",
    "ControllerLimitError",
    "ControllerLimits",
    "check",
    "export",
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


class ControllerLimitError(CompilationError):
    """A scheduled program exceeds a limit of the controller it is for."""

    def __init__(self, violations: Sequence[str]) -> None:
        self.violations = tuple(violations)
        shown = "\n  ".join(self.violations[:10])
        more = len(self.violations) - 10
        tail = f"\n  ... and {more} more" if more > 0 else ""
        super().__init__(f"the program exceeds the controller's limits:\n  {shown}{tail}")


@dataclass(frozen=True)
class ControllerLimits:
    """What a controller can play; every limit is in clock cycles (VTU).

    ``clock_ghz``: the only clock the controller runs at (``None``: any).
    ``issue_vtu`` and ``min_wait_vtu``: two operations on one line start
    exactly ``issue_vtu`` apart, or at least ``issue_vtu + min_wait_vtu``
    apart, because the controller issues one instruction per ``issue_vtu``
    and idles no less than ``min_wait_vtu``.
    ``samples_per_vtu`` and ``waveform_samples``: the waveform memory of a
    line holds this many samples, at this many samples per cycle; waveforms
    of one shape at different amplitudes share it (``shape_key``).
    ``tones_per_line``: distinct tone frequencies one line plays.
    ``acquirers`` and ``acquirer_hold_vtu``: the acquirer reading each
    qudit; readouts on one acquirer are serialized, each holding it for its
    duration plus ``acquirer_hold_vtu``.
    ``feedback_latency_vtu``: the least time from the end of a readout to
    the start of an operation in a branch on its outcome.
    ``answer_timeouts``: whether the controller can stop waiting for an
    answer after a timeout (``result(timeout_ns=...)``, ``latency_mode=
    "timeout"``); without, only waits that block or are budgeted are
    allowed.
    ``max_events``: the timed operations of one program.
    """

    clock_ghz: float | None = None
    issue_vtu: int = 1
    min_wait_vtu: int = 0
    samples_per_vtu: int | None = None
    waveform_samples: int | None = None
    tones_per_line: int | None = None
    acquirers: Mapping[int, int] = field(default_factory=dict)
    acquirer_hold_vtu: int = 0
    feedback_latency_vtu: int = 0
    answer_timeouts: bool = True
    max_events: int | None = None

    def __post_init__(self) -> None:
        if self.clock_ghz is not None and not self.clock_ghz > 0:
            raise ValueError(f"ControllerLimits.clock_ghz must be positive, got {self.clock_ghz}")
        for name in ("issue_vtu",):
            if getattr(self, name) < 1:
                raise ValueError(f"ControllerLimits.{name} must be at least 1")
        for name in ("min_wait_vtu", "acquirer_hold_vtu", "feedback_latency_vtu"):
            if getattr(self, name) < 0:
                raise ValueError(f"ControllerLimits.{name} cannot be negative")
        for name in ("samples_per_vtu", "waveform_samples", "tones_per_line", "max_events"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"ControllerLimits.{name} must be positive or None")
        object.__setattr__(self, "acquirers", MappingProxyType(dict(self.acquirers)))

    def __hash__(self) -> int:
        return hash(
            (
                self.clock_ghz,
                self.issue_vtu,
                self.min_wait_vtu,
                self.samples_per_vtu,
                self.waveform_samples,
                self.tones_per_line,
                tuple(sorted(self.acquirers.items())),
                self.acquirer_hold_vtu,
                self.feedback_latency_vtu,
                self.answer_timeouts,
                self.max_events,
            )
        )

    def as_dict(self) -> dict[str, Any]:
        """The limits as JSON-serializable data."""
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data["acquirers"] = {str(k): v for k, v in sorted(self.acquirers.items())}
        return data


# ── Waveforms ────────────────────────────────────────────────────────────────


def _tree(nodes: Sequence[tuple[str, Sequence[float], Sequence[int]]]) -> dict[str, Any] | None:
    """The waveform expression of an event as nested dictionaries."""
    if not nodes:
        return None

    def number(value: float) -> float | None:
        return None if math.isnan(value) else float(value)

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
            node["params"] = {k: number(v) for k, v in zip(names, numbers, strict=True)}
        elif numbers:
            node["values"] = [number(v) for v in numbers]
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
    params = node.get("params")
    if params is None:
        if shape == "custom_samples" and node.get("samples"):
            return complex(max(node["samples"], key=abs))
        return None
    real = params.get("amplitude")
    imag = params.get("amplitude_imag", 0.0)
    if real is None or imag is None:
        return None
    return complex(real, imag)


def shape_key(node: Mapping[str, Any] | None) -> tuple[Any, ...] | None:
    """A key equal for waveforms that differ only in overall amplitude.

    Scale and negation drop out, and so does the amplitude of a built-in
    shape; custom samples are normalized to their largest magnitude.
    ``None`` when the waveform is not built from numbers alone.
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
        values = tuple(node.get("values", ()))
        return (shape, values, children)
    if any(v is None for v in params.values()):
        return None
    skip = {_PARAMETERS[shape][i] for i in _AMPLITUDES.get(shape, ())}
    return (shape, tuple((k, v) for k, v in params.items() if k not in skip))


def sample_waveform(
    node: Mapping[str, Any], duration_vtu: int, samples_per_vtu: int = 1
) -> np.ndarray:
    """The complex envelope of a waveform, ``samples_per_vtu`` per cycle.

    Sample ``k`` is the envelope at ``k / samples_per_vtu`` cycles from the
    start, with the definitions of the waveform ops (times in cycles), the
    same the simulator plays.
    """
    import numpy as np

    count = int(duration_vtu) * int(samples_per_vtu)
    t = np.arange(count, dtype=float) / samples_per_vtu

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
            index = (t * len(values) / max(duration_vtu, 1)).astype(int)
            return values[np.minimum(index, len(values) - 1)].astype(complex)
        if shape not in _PARAMETERS or any(v is None for v in params.values()):
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
        if shape == "gaussian_square":
            sigma, edge = params["sigma"], params["risefall"]
            out = np.full(count, complex(a))
            rise = t < edge
            fall = t >= d - edge
            out[rise] = a * np.exp(-0.5 * (t[rise] - edge) ** 2 / sigma**2)
            out[fall] = a * np.exp(-0.5 * (t[fall] - (d - edge)) ** 2 / sigma**2)
            return out
        raise ValueError(f"cannot sample a {shape!r} waveform")

    return np.asarray(evaluate(node), dtype=complex)


# ── Export ───────────────────────────────────────────────────────────────────


def _module(lowered: Lowered) -> Any:
    # Waveforms and frames depend on the call's values, which only the
    # concrete module holds.
    if lowered.scheduled and not lowered.arguments:
        return lowered._module
    return lowered._concrete()


def _events(lowered: Lowered) -> list[dict[str, Any]]:
    module = _module(lowered)
    with lowered._lock:
        raw = module.controller_events()
    starts = [e.start_vtu for e in lowered.epochs()]
    events = []
    for index, item in enumerate(raw):
        (
            kind,
            line,
            qubit,
            start,
            duration,
            iteration,
            nodes,
            frequency,
            phase,
            conditions,
            refs,
            callee,
            mode,
            timeout,
        ) = item
        epoch = 0
        for k, epoch_start in enumerate(starts):
            if start is not None and start >= epoch_start and k > 0:
                epoch = k
        event: dict[str, Any] = {
            "index": index,
            "kind": kind,
            "line": line,
            "qudit": qubit,
            "start_vtu": start,
            "duration_vtu": duration,
            "iteration": list(iteration),
            "epoch": epoch,
            "conditions": [
                {
                    "branch": branch,
                    "arm": bool(arm),
                    "readouts": list(readouts),
                    "answers": list(answers),
                    "test": test,
                }
                for branch, arm, readouts, answers, test in conditions
            ],
        }
        if kind in ("drive", "readout"):
            tree = _tree(nodes)
            amplitude = _amplitude(tree)
            event["waveform"] = tree
            event["amplitude"] = (
                None if amplitude is None else {"re": amplitude.real, "im": amplitude.imag}
            )
            event["frequency_hz"] = None if math.isnan(frequency) else frequency
            event["phase_rad"] = None if math.isnan(phase) else phase
        if kind in ("submit", "answer"):
            event["link"] = line
            event["function"] = callee
            if kind == "submit":
                event["measurements"] = list(refs)
            else:
                event["wait"] = mode
                event["timeout_ns"] = None if math.isnan(timeout) else timeout
        events.append(event)
    return events


def _lines(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lines: dict[int, dict[str, Any]] = {}
    for event in events:
        if event["kind"] in ("submit", "answer") or event["line"] < 0:
            continue
        entry = lines.setdefault(
            event["line"], {"line": event["line"], "role": "drive", "qudit": event["qudit"]}
        )
        if event["kind"] == "readout":
            entry["role"] = "readout"
    return [lines[k] for k in sorted(lines)]


def export(lowered: Lowered, sample_rate_ghz: float | None = None) -> dict[str, Any]:
    """A scheduled kernel for a controller backend, as JSON-serializable data.

    With *sample_rate_ghz*, each drive and readout also carries its
    envelope's samples at that rate (``samples``: real and imaginary parts),
    which must be a whole multiple of the clock. Raises
    ``ControllerLimitError`` when the machine model's controller limits are
    exceeded.
    """
    from .classical.emulation import export as emulation

    if lowered.parameters:
        raise ValueError(
            "the module has float parameters "
            f"({', '.join(lowered.parameters)}); lower it with their values first"
        )
    events = _events(lowered)
    limits = lowered.options.machine.controller
    if limits is not None:
        violations = _violations(lowered, events, limits)
        if violations:
            raise ControllerLimitError(violations)
    clock = lowered.target.clock_ghz
    if sample_rate_ghz is not None:
        ratio = sample_rate_ghz / clock
        per_vtu = round(ratio)
        if per_vtu < 1 or abs(ratio - per_vtu) > 1e-9:
            raise ValueError(
                f"sample_rate_ghz={sample_rate_ghz} is not a whole multiple of the "
                f"{clock} GHz clock"
            )
        for event in events:
            if event.get("waveform") is not None and event["duration_vtu"] is not None:
                samples = sample_waveform(event["waveform"], event["duration_vtu"], per_vtu)
                event["samples"] = {"re": samples.real.tolist(), "im": samples.imag.tolist()}
    base = emulation(lowered)
    return {
        "format": FORMAT,
        "target": lowered.target.name,
        "clock_ghz": clock,
        "schedule": lowered.options.schedule,
        "sample_rate_ghz": sample_rate_ghz,
        "qudits": list(lowered.qudits),
        "lines": _lines(events),
        "events": events,
        "epochs": base["epochs"],
        "calls": base["calls"],
        "links": base["links"],
        "manifest": base["manifest"],
        "limits": None if limits is None else limits.as_dict(),
    }


# ── Checks ───────────────────────────────────────────────────────────────────


def _exclusive(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Whether two events lie in different arms of one branch execution."""
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
    ordered = sorted(events, key=lambda e: (e["start_vtu"], e["index"]))
    pairs = []
    for k, event in enumerate(ordered):
        for previous in reversed(ordered[:k]):
            if not _exclusive(previous, event):
                pairs.append((previous, event))
                break
    return pairs


def _violations(
    lowered: Lowered, events: Sequence[Mapping[str, Any]], limits: ControllerLimits
) -> list[str]:
    found: list[str] = []
    clock = lowered.target.clock_ghz
    if limits.clock_ghz is not None and abs(clock - limits.clock_ghz) > 1e-12:
        found.append(
            f"the target's clock is {clock} GHz; the controller runs at {limits.clock_ghz} GHz"
        )
    timed = [
        e
        for e in events
        if e["kind"] in ("drive", "readout")
        and e["start_vtu"] is not None
        and e["duration_vtu"] is not None
    ]
    if limits.max_events is not None and len(timed) > limits.max_events:
        found.append(
            f"{len(timed)} drives and readouts; the controller plays at most {limits.max_events}"
        )
    by_line: dict[int, list[Mapping[str, Any]]] = {}
    for event in timed:
        by_line.setdefault(event["line"], []).append(event)
    issue, gap = limits.issue_vtu, limits.issue_vtu + limits.min_wait_vtu
    for line, items in sorted(by_line.items()):
        for previous, event in _pairs(items):
            distance = event["start_vtu"] - previous["start_vtu"]
            if distance != issue and distance < gap:
                found.append(
                    f"line {line}: {event['kind']} at {event['start_vtu']} starts "
                    f"{distance} cycles after the {previous['kind']} at "
                    f"{previous['start_vtu']}; the controller "
                    f"needs exactly {issue} or at least {gap}"
                )
        if limits.tones_per_line is not None:
            tones = {e["frequency_hz"] for e in items if e.get("frequency_hz") is not None}
            if len(tones) > limits.tones_per_line:
                found.append(
                    f"line {line}: {len(tones)} tone frequencies; the controller holds "
                    f"{limits.tones_per_line} per line"
                )
        if limits.waveform_samples is not None:
            per_vtu = limits.samples_per_vtu or 1
            shapes: dict[Any, int] = {}
            for e in items:
                key = shape_key(e.get("waveform"))
                shapes[(key, e["duration_vtu"]) if key is not None else ("event", e["index"])] = (
                    e["duration_vtu"] * per_vtu
                )
            used = sum(shapes.values())
            if used > limits.waveform_samples:
                found.append(
                    f"line {line}: its waveforms take {used} samples; the controller's "
                    f"waveform memory holds {limits.waveform_samples}"
                )
    if limits.acquirers:
        by_acquirer: dict[int, list[Mapping[str, Any]]] = {}
        for event in timed:
            if event["kind"] == "readout" and event["qudit"] in limits.acquirers:
                by_acquirer.setdefault(limits.acquirers[event["qudit"]], []).append(event)
        for acquirer, items in sorted(by_acquirer.items()):
            for previous, event in _pairs(items):
                free = previous["start_vtu"] + previous["duration_vtu"] + limits.acquirer_hold_vtu
                if event["start_vtu"] < free:
                    found.append(
                        f"acquirer {acquirer}: the readout of qudit {event['qudit']} at "
                        f"{event['start_vtu']} starts before the acquirer is free at {free} "
                        f"(the readout of qudit {previous['qudit']} at {previous['start_vtu']})"
                    )
    if limits.feedback_latency_vtu:
        for event in timed:
            for condition in event["conditions"]:
                for ref in condition["readouts"]:
                    readout = events[ref]
                    ready = (
                        readout["start_vtu"] + readout["duration_vtu"] + limits.feedback_latency_vtu
                    )
                    if event["start_vtu"] < ready:
                        found.append(
                            f"line {event['line']}: the {event['kind']} at {event['start_vtu']} "
                            f"depends on the readout ending at "
                            f"{readout['start_vtu'] + readout['duration_vtu']}; the controller "
                            f"needs {limits.feedback_latency_vtu} cycles of feedback latency "
                            f"(start at {ready} or later; schedule with rcp and "
                            "readout_latency_vtu >= feedback_latency_vtu)"
                        )
    if not limits.answer_timeouts:
        functions = {fn.name: fn for fn in lowered.classical}
        for event in events:
            if event["kind"] != "answer":
                continue
            fn = functions.get(event["function"])
            if event["wait"] == "timeout" or (fn is not None and fn.latency_mode == "timeout"):
                found.append(
                    f"the wait for {event['function']} at {event['start_vtu']} has a timeout; "
                    "the controller cannot stop waiting for an answer, so take it with "
                    "result() (no timeout_ns)"
                )
    return found


def check(lowered: Lowered, limits: ControllerLimits | None = None) -> list[str]:
    """The limits *lowered* exceeds, of *limits* or else its machine model's."""
    limits = limits if limits is not None else lowered.options.machine.controller
    if limits is None:
        return []
    return _violations(lowered, _events(lowered), limits)
