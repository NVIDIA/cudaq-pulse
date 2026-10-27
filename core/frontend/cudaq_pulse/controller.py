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

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from . import _controller_cost
from ._controller_cost import OP_CLASSES, FunctionCost
from .kernel.ir_builder import CompilationError

if TYPE_CHECKING:
    import numpy as np

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

    def __init__(self, violations: Sequence[str]) -> None:
        self.violations = tuple(violations)
        shown = "\n  ".join(self.violations[:10])
        more = len(self.violations) - 10
        tail = f"\n  ... and {more} more" if more > 0 else ""
        super().__init__(f"the program exceeds the controller's limits:\n  {shown}{tail}")


@dataclass(frozen=True)
class ControllerLimits:
    """What a controller can play; every limit is in clock cycles.

    ``clock_ghz``: the only clock the controller runs at (``None``: any).
    ``issue_cycles`` and ``min_wait_cycles``: two operations on one line start
    exactly ``issue_cycles`` apart, or at least ``issue_cycles + min_wait_cycles``
    apart, because the controller issues one instruction per ``issue_cycles``
    and idles no less than ``min_wait_cycles``.
    ``samples_per_cycles`` and ``waveform_samples``: the waveform memory of a
    line holds this many samples, at this many samples per cycle; waveforms
    of one shape at different amplitudes share it (``shape_key``).
    ``tones_per_line``: distinct tone frequencies one line plays.
    ``acquirers`` and ``acquirer_hold_cycles``: the acquirer reading each
    qudit; readouts on one acquirer are serialized, each holding it for its
    duration plus ``acquirer_hold_cycles``.
    ``feedback_latency_cycles``: the least time from the end of a readout to
    the start of an operation in a branch on its outcome.
    ``answer_timeouts``: whether the controller can stop waiting for an
    answer after a timeout (``result(timeout_ns=...)``, ``latency_mode=
    "timeout"``); without, only waits that block or are budgeted are
    allowed.
    ``max_events``: the drives and readouts of one program.

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
    issue_cycles: int = 1
    min_wait_cycles: int = 0
    samples_per_cycles: int | None = None
    waveform_samples: int | None = None
    tones_per_line: int | None = None
    acquirers: Mapping[int, int] = field(default_factory=dict)
    acquirer_hold_cycles: int = 0
    feedback_latency_cycles: int = 0
    max_events: int | None = None
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
        if self.issue_cycles < 1:
            raise ValueError("ControllerLimits.issue_cycles must be at least 1")
        for name in ("min_wait_cycles", "acquirer_hold_cycles", "feedback_latency_cycles"):
            if getattr(self, name) < 0:
                raise ValueError(f"ControllerLimits.{name} cannot be negative")
        for name in (
            "samples_per_cycles",
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
        return data


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


def _runtime_operands(
    tree: dict[str, Any],
    nodes: Sequence[tuple[str, Sequence[float], Sequence[int]]],
    runtime: Sequence[Mapping[str, Any]],
) -> None:
    """Adds to each node of *tree* the expression of its operands computed
    from a readout or an answer (``runtime``), by parameter name."""
    computed: dict[int, dict[int, Any]] = {}
    for item in runtime:
        computed.setdefault(item["node"], {})[item["operand"]] = item["expr"]

    def attach(index: int, node: dict[str, Any]) -> None:
        name, numbers, children = nodes[index]
        if index in computed:
            names = _PARAMETERS.get(name.partition(":")[0])
            if names is not None and len(names) != len(numbers):
                names = None
            node["runtime"] = {
                (names[k] if names is not None else str(k)): expr
                for k, expr in sorted(computed[index].items())
            }
        for child, sub in zip(children, node.get("of", ()), strict=True):
            attach(child, sub)

    attach(len(nodes) - 1, tree)


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
        return (shape, tuple(node.get("values", ())), children)
    if any(v is None for v in params.values()):
        return None
    skip = {_PARAMETERS[shape][i] for i in _AMPLITUDES.get(shape, ())}
    return (shape, node.get("name"), tuple((k, v) for k, v in params.items() if k not in skip))


def sample_waveform(
    node: Mapping[str, Any], duration_cycles: int, samples_per_cycles: int = 1
) -> np.ndarray:
    """The complex envelope of a waveform, ``samples_per_cycles`` per cycle.

    Sample ``k`` is the envelope at ``k / samples_per_cycles`` cycles from the
    start, with the definitions of the waveform ops (times in cycles), the
    same the simulator plays. Raises ``ValueError`` for a waveform the
    export cannot evaluate: a ``pulse.custom`` op of hand-written IR, whose
    samples the runtime computes, or one whose parameters are not numbers.
    """
    import numpy as np

    count = int(duration_cycles) * int(samples_per_cycles)
    t = np.arange(count, dtype=float) / samples_per_cycles

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
    lowered: Lowered, values: bool = True
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The lines and events of *lowered*, at the values of its call; without
    *values*, a module with float parameters has ``None`` where they reach."""
    if values and lowered.parameters:
        raise ValueError(
            "the module has float parameters "
            f"({', '.join(lowered.parameters)}); lower it with their values first"
        )
    # Waveforms and frames depend on the call's values, which only the
    # concrete module holds.
    module = lowered._concrete()
    with lowered._lock:
        raw_lines, raw_events, extras = module.controller_events()
    lines = [
        {"line": index, "role": role, "qudit": qudit}
        for index, (role, qudit) in enumerate(raw_lines)
    ]
    epochs = lowered._event_epochs(syncs=True)
    events = []
    for index, item in enumerate(raw_events):
        (
            kind,
            line,
            qudit,
            start,
            duration,
            iteration,
            nodes,
            frequency,
            phase,
            conditions,
            joined,
            refs,
            callee,
            mode,
            timeout,
        ) = item
        event: dict[str, Any] = {
            "index": index,
            "kind": kind,
            "line": line,
            "qudit": qudit,
            "start_cycles": start,
            "duration_cycles": duration,
            "end_cycles": None if start is None or duration is None else start + duration,
            "iteration": list(iteration),
            "epoch": epochs[index] if epochs and start is not None else 0,
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
        if kind in _PLAYED:
            tree = _tree(nodes)
            amplitude = _amplitude(tree)
            event["waveform"] = tree
            event["amplitude"] = (
                None if amplitude is None else {"re": amplitude.real, "im": amplitude.imag}
            )
            event["frequency_hz"] = _number(frequency)
            event["phase_rad"] = _number(phase)
        elif kind == "sync":
            event["line"] = None
            event["qudit"] = None
            event["lines"] = list(joined)
        elif kind in ("submit", "answer"):
            event["link"] = line
            event["function"] = callee
            if kind == "submit":
                event["measurements"] = list(refs)
            else:
                event["wait"] = mode
                event["timeout_ns"] = _number(timeout)
        events.append(event)
    # What is known only at run time, on the few events that have any.
    for index, tone, known in json.loads(extras) if extras else ():
        event = events[index]
        if "waveform" in known:
            _runtime_operands(event["waveform"], raw_events[index][6], known["waveform"])
        if tone >= 0:
            event["tone"] = tone
            event["frame_updates"] = known["frame_updates"]
            if "phase_offset_rad" in known:
                event["phase_offset_rad"] = known["phase_offset_rad"]
        if "length" in known:
            event["length"] = known["length"]
            event["min_cycles"] = known.get("min_cycles")
            event["max_cycles"] = known.get("max_cycles")
        if "operands" in known:
            event["operands"] = known["operands"]
    return lines, events


def export(lowered: Lowered, sample_rate_ghz: float | None = None) -> dict[str, Any]:
    """A scheduled kernel for a controller backend, as JSON-serializable data.

    With *sample_rate_ghz*, each drive and readout also carries its
    envelope's samples at that rate (``samples``: real and imaginary parts),
    which must be a whole multiple of the clock. Raises
    ``ControllerLimitError`` when the machine model's controller limits are
    exceeded.
    """
    from .classical.emulation import export as emulation

    lines, events = _walk(lowered)
    limits = lowered.options.machine.controller
    if limits is not None:
        violations = _violations(lowered, events, limits)
        if violations:
            raise ControllerLimitError(violations)
    clock = lowered.target.clock_ghz
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
    base = emulation(lowered)
    return {
        "format": FORMAT,
        "target": lowered.target.name,
        "clock_ghz": clock,
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
    lowered: Lowered, events: Sequence[Mapping[str, Any]], limits: ControllerLimits
) -> list[str]:
    found: list[str] = []
    clock = lowered.target.clock_ghz
    if limits.clock_ghz is not None and abs(clock - limits.clock_ghz) > 1e-12:
        found.append(
            f"the target's clock is {clock} GHz; the controller runs at {limits.clock_ghz} GHz"
        )
    timed = [e for e in events if e["kind"] in _PLAYED and e["end_cycles"] is not None]
    if limits.max_events is not None and len(timed) > limits.max_events:
        found.append(
            f"{len(timed)} drives and readouts; the controller plays at most {limits.max_events}"
        )
    by_line: dict[int, list[Mapping[str, Any]]] = {}
    for event in timed:
        by_line.setdefault(event["line"], []).append(event)
    issue, gap = limits.issue_cycles, limits.issue_cycles + limits.min_wait_cycles
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
        if limits.waveform_samples is not None:
            per_cycles = limits.samples_per_cycles or 1
            shapes: dict[Any, int] = {}
            for e in items:
                key = shape_key(e.get("waveform"))
                shapes[
                    (key, e["duration_cycles"]) if key is not None else ("event", e["index"])
                ] = e["duration_cycles"] * per_cycles
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
                    "result() (no timeout_ns)"
                )
    if limits.computes:
        found.extend(resources(lowered, limits).violations)
    return found


def check(lowered: Lowered, limits: ControllerLimits | None = None) -> list[str]:
    """The limits *lowered* exceeds, of *limits* or else its machine model's."""
    limits = limits if limits is not None else lowered.options.machine.controller
    if limits is None:
        return []
    return _violations(lowered, _walk(lowered, values=False)[1], limits)


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
