# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The control system of a QPU: its controllers, their wiring and links.

A ``Controller`` is one box: what it can play and compute (the fields of
``ControllerLimits``), the lines it drives and reads out (``drive_lines``,
``readout_lines``: carrier indices of the quantum system), its latency to
its peers, what it supports (``Capability``) and the backend that lowers its
programs. A ``ControlSystem`` is the controllers of one QPU, the links to
the classical devices beside them (by name) and the limits that apply to
all of them::

    controllers = [pulse.Controller(drive_lines=range(8), readout_lines=range(8))]
    qpu = pulse.device({
        "quantum_system": "transmon_17q",
        "control_system": pulse.ControlSystem(controllers=controllers),
    })
    compiled = pulse.compile(kernel, device=qpu)

``Device.as_dict()`` writes a QPU as one JSON-serializable mapping (format
``SYSTEM_FORMAT``), which ``pulse.device()`` reads back.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import math
import os
import re
import tomllib
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from . import _clocks
from ._errors import (
    DeviceDefinitionError,
    InvalidTypeError,
    InvalidValueError,
    NotFoundError,
    Problem,
    UpgradeNotice,
    _closest,
    _count,
    _ranges,
)
from .classical import Link
from .controller import OP_CLASSES, ControllerLimits

if TYPE_CHECKING:
    from ._device import Device
    from .compiler import Lowered, MachineModel
    from .targets.base import TargetDescription

__all__ = ["SYSTEM_FORMAT", "Capability", "ControlSystem", "Controller", "Processor"]

SYSTEM_FORMAT = "cudaq-pulse.system/1"


class Capability(enum.StrEnum):
    """What a controller can do, for ``Controller.supports``.

    ``play`` a drive; ``acquire`` a readout; ``wait`` for a length computed
    while the QPU runs; ``sync`` lines; ``call`` a classical function and
    take its ``answer``; ``branch`` on an outcome or an answer;
    ``runtime_frame_update`` a tone's frequency or phase from a value
    computed while the QPU runs; ``forward_outcome`` of one of its readouts
    to another controller, which branches on it; ``loop``: run a rolled
    loop itself, holding its body once. A controller without ``loop`` plays
    a rolled loop unrolled, so each iteration counts against its
    ``max_events``. Unrolling does not change its waveform memory: a
    waveform is stored once per shape however often it plays, so only the
    pulse count goes up.
    """

    PLAY = "play"
    ACQUIRE = "acquire"
    WAIT = "wait"
    SYNC = "sync"
    CALL = "call"
    ANSWER = "answer"
    BRANCH = "branch"
    RUNTIME_FRAME_UPDATE = "runtime_frame_update"
    FORWARD_OUTCOME = "forward_outcome"
    LOOP = "loop"


_PLAYED = frozenset({"drive", "readout"})
_NEEDS = MappingProxyType(
    {"drive": Capability.PLAY, "readout": Capability.ACQUIRE, "wait": Capability.WAIT}
)


def _lines(value: Iterable[Any], name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise InvalidTypeError(f"Controller.{name} is a sequence of carrier indices, got {value!r}")
    lines = tuple(value)
    for line in lines:
        if isinstance(line, bool) or not isinstance(line, int):
            raise InvalidTypeError(
                f"Controller.{name} names a line by the carrier index of its qudit, "
                f"got {line!r}; named and shared lines are not supported yet"
            )
        if line < 0:
            raise InvalidValueError(f"Controller.{name} has a negative carrier index {line}")
    if len(set(lines)) != len(lines):
        raise InvalidValueError(f"Controller.{name} lists a carrier twice: {list(lines)}")
    return lines


def _cycles(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidTypeError(f"Controller.{name} is a whole number of cycles, got {value!r}")
    if value < 0:
        raise InvalidValueError(f"Controller.{name} cannot be negative, got {value}")
    return value


# What a processor's cycles_per_op and unsupported name: the classes of
# operations, and "loop", a rolled loop's counter and jump per iteration.
PROCESSOR_OPS = (*OP_CLASSES, "loop")
_PROCESSOR_LIMITS = (
    "max_ops_per_function",
    "max_ops_total",
    "max_state_bytes_per_function",
    "max_state_bytes_total",
    "max_calls_per_shot",
    "max_bytes_per_shot",
)


def _op_name(given: Any, field_name: str) -> str:
    if given in PROCESSOR_OPS:
        return str(given)
    hint = _closest(str(given), PROCESSOR_OPS)
    raise InvalidValueError(
        f"Processor.{field_name} names an unknown operation {given!r}"
        + (f" (did you mean {hint!r}?)" if hint else "")
        + f"; the operations are {', '.join(PROCESSOR_OPS)}"
    )


@dataclass(frozen=True)
class Processor:
    """A classical processor on a controller, which runs the classical
    functions called on its ``links`` (link names of the control system, or
    link ids). A link belongs to at most one processor of a system; a
    function on no processor runs off the controllers, and only its link is
    checked.

    ``clock_ghz``: its clock (``None``: the controller's, else the
    program's). ``cycles_per_op``: the cycles of one operation of each class
    of ``OP_CLASSES``, and of ``"loop"``, a rolled loop's counter and jump
    per iteration; one cycle when not given. ``branch`` is an ``if`` or
    ``else``. ``unsupported``: the operations it cannot run; a function that
    uses one does not compile, except ``"loop"``: a processor without it
    runs each loop unrolled, so its code is counted unrolled.
    ``max_concurrent_calls``: the calls it computes at once, from submission
    to the time the answer is due.
    ``max_ops_per_function`` and ``max_ops_total``: the operations one
    function, and all its functions, may hold. ``max_state_bytes_per_function``
    and ``max_state_bytes_total``: the bytes of arguments, results and
    locals of one function, and of all of them. ``max_calls_per_shot`` and
    ``max_bytes_per_shot``: the calls, and the bytes of requests and
    answers, of one shot. ``None`` is no limit. ``fixed_cost_only``: every
    function's cost must be fixed when the program compiles (no ``while``,
    recursion or allocation). ``dispatch_keys``: the dispatch key of each
    function it runs, by function name, when the processor fixes them; it
    then lists every function it runs.
    """

    links: Sequence[int | str] = ("controller",)
    clock_ghz: float | None = None
    cycles_per_op: Mapping[str, int] = field(default_factory=dict)
    unsupported: Sequence[str] = ()
    max_concurrent_calls: int = 1
    max_ops_per_function: int | None = None
    max_ops_total: int | None = None
    max_state_bytes_per_function: int | None = None
    max_state_bytes_total: int | None = None
    max_calls_per_shot: int | None = None
    max_bytes_per_shot: int | None = None
    fixed_cost_only: bool = False
    dispatch_keys: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.links, (str, bytes)) or not isinstance(self.links, Iterable):
            raise InvalidTypeError(
                f"Processor.links is a list of link names or ids, got {self.links!r}"
            )
        links = tuple(self.links)
        for link in links:
            if isinstance(link, str):
                if not link.isidentifier():
                    raise InvalidValueError(f"Processor.links names a link, got {link!r}")
            elif isinstance(link, bool) or not isinstance(link, int) or link < 0:
                raise InvalidTypeError(
                    f"Processor.links holds link names or ids >= 0, got {link!r}"
                )
        if len(set(links)) != len(links):
            raise InvalidValueError(f"Processor.links lists a link twice: {list(links)}")
        object.__setattr__(self, "links", links)
        clock = self.clock_ghz
        if clock is not None and (
            isinstance(clock, bool)
            or not isinstance(clock, (int, float))
            or not math.isfinite(clock)
            or not clock > 0
        ):
            raise InvalidValueError(
                f"Processor.clock_ghz must be a positive number or None, got {clock!r}"
            )
        if not isinstance(self.cycles_per_op, Mapping):
            raise InvalidTypeError(
                f"Processor.cycles_per_op maps operations to cycles, got {self.cycles_per_op!r}"
            )
        costs = {}
        for op, cycles in self.cycles_per_op.items():
            name = _op_name(op, "cycles_per_op")
            if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles < 0:
                raise InvalidValueError(
                    f"Processor.cycles_per_op[{name!r}] is a whole number of cycles >= 0, "
                    f"got {cycles!r}"
                )
            costs[name] = cycles
        object.__setattr__(self, "cycles_per_op", MappingProxyType(costs))
        if isinstance(self.unsupported, (str, bytes)) or not isinstance(self.unsupported, Iterable):
            raise InvalidTypeError(
                f"Processor.unsupported is a list of operations, got {self.unsupported!r}"
            )
        unsupported = tuple(dict.fromkeys(_op_name(op, "unsupported") for op in self.unsupported))
        both = [op for op in unsupported if op in costs]
        if both:
            raise InvalidValueError(
                f"Processor.unsupported lists {both[0]!r}, which cycles_per_op gives a cost; "
                "an operation is supported or not"
            )
        object.__setattr__(self, "unsupported", unsupported)
        calls = self.max_concurrent_calls
        if isinstance(calls, bool) or not isinstance(calls, int) or calls < 1:
            raise InvalidValueError(
                f"Processor.max_concurrent_calls is a whole number >= 1, got {calls!r}"
            )
        for name in _PROCESSOR_LIMITS:
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise InvalidValueError(
                    f"Processor.{name} is a whole number >= 1 or None, got {value!r}"
                )
        if not isinstance(self.fixed_cost_only, bool):
            raise InvalidTypeError(
                f"Processor.fixed_cost_only is True or False, got {self.fixed_cost_only!r}"
            )
        if not isinstance(self.dispatch_keys, Mapping):
            raise InvalidTypeError(
                f"Processor.dispatch_keys maps function names to keys, got {self.dispatch_keys!r}"
            )
        owners: dict[int, str] = {}
        for name, key in self.dispatch_keys.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise InvalidValueError(
                    f"Processor.dispatch_keys is keyed by function name, got {name!r}"
                )
            if isinstance(key, bool) or not isinstance(key, int) or not 0 <= key <= 0xFFFFFFFF:
                raise InvalidValueError(
                    f"Processor.dispatch_keys[{name!r}] is {key!r}; a dispatch key is a whole "
                    "number from 0 to 0xFFFFFFFF"
                )
            if key in owners:
                raise InvalidValueError(
                    f"Processor.dispatch_keys gives {key} to both {owners[key]!r} and {name!r}; "
                    "give each function its own key"
                )
            owners[key] = name
        object.__setattr__(self, "dispatch_keys", MappingProxyType(dict(self.dispatch_keys)))

    def __hash__(self) -> int:
        return hash(
            tuple(
                tuple(sorted(value.items())) if isinstance(value, Mapping) else value
                for value in (getattr(self, f.name) for f in fields(self))
            )
        )

    @property
    def loops(self) -> bool:
        """Whether it runs a rolled loop as a loop (``"loop"`` is not
        ``unsupported``); without, its loops are unrolled."""
        return "loop" not in self.unsupported

    def as_dict(self) -> dict[str, Any]:
        """The processor as JSON-serializable data, which ``Processor(**data)``
        reads back."""
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data["links"] = list(self.links)
        data["cycles_per_op"] = dict(self.cycles_per_op)
        data["unsupported"] = list(self.unsupported)
        data["dispatch_keys"] = dict(self.dispatch_keys)
        return data


@dataclass(frozen=True)
class Controller(ControllerLimits):
    """One controller of a ``ControlSystem``: its limits and its wiring.

    The fields of ``ControllerLimits`` state what it can play and compute.
    ``drive_lines`` and ``readout_lines`` are the carriers whose drive and
    readout lines it plays (carrier indices of the quantum system); the only
    controller of a system that lists none plays every line.
    ``peer_latency_cycles`` is the time an outcome takes to reach another
    controller: one number for all, or a mapping from the other controller's
    index in the system. ``clock_skew_cycles`` bounds how far its clock is
    from the system's, and ``alignment_tolerance_cycles`` how far its pulses
    may land from where they are meant to be on other controllers.
    ``supports`` lists its capabilities (``None``: everything); a program
    that needs one it lacks fails to compile. ``backend`` is the controller
    backend that lowers its programs, by entry-point name. ``model`` and
    ``version`` identify the description it was made from. ``processors``
    are its classical processors (``Processor``); a controller with
    processors states what it computes on them, not with the compute fields
    of ``ControllerLimits``.
    """

    drive_lines: Sequence[int] = ()
    readout_lines: Sequence[int] = ()
    peer_latency_cycles: int | Mapping[int, int] = 0
    clock_skew_cycles: int = 0
    alignment_tolerance_cycles: int = 0
    supports: Sequence[Capability | str] | None = None
    backend: str | None = None
    model: str | None = None
    version: str | None = None
    processors: Sequence[Processor] = ()

    __hash__ = ControllerLimits.__hash__

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "drive_lines", _lines(self.drive_lines, "drive_lines"))
        object.__setattr__(self, "readout_lines", _lines(self.readout_lines, "readout_lines"))
        peers = self.peer_latency_cycles
        if isinstance(peers, Mapping):
            object.__setattr__(
                self,
                "peer_latency_cycles",
                MappingProxyType(
                    {
                        _cycles(k, "peer_latency_cycles keys"): _cycles(v, "peer_latency_cycles")
                        for k, v in peers.items()
                    }
                ),
            )
        else:
            _cycles(peers, "peer_latency_cycles")
        _cycles(self.clock_skew_cycles, "clock_skew_cycles")
        _cycles(self.alignment_tolerance_cycles, "alignment_tolerance_cycles")
        if self.supports is not None:
            if isinstance(self.supports, str):
                raise InvalidTypeError(
                    f"Controller.supports is a list of capabilities, got {self.supports!r}"
                )
            supported = []
            for given in self.supports:
                try:
                    supported.append(Capability(given))
                except ValueError:
                    hint = _closest(str(given), [c.value for c in Capability])
                    raise InvalidValueError(
                        f"Controller.supports has an unknown capability {given!r}"
                        + (f" (did you mean {hint!r}?)" if hint else "")
                        + f"; the capabilities are {', '.join(c.value for c in Capability)}"
                    ) from None
            object.__setattr__(self, "supports", tuple(dict.fromkeys(supported)))
        for name in ("backend", "model", "version"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise InvalidTypeError(f"Controller.{name} is a string, got {value!r}")
        if isinstance(self.processors, (str, bytes, Processor, Mapping)) or not isinstance(
            self.processors, Iterable
        ):
            raise InvalidTypeError(
                f"Controller.processors is a list of pulse.Processor, got {self.processors!r}"
            )
        processors = []
        for index, item in enumerate(self.processors):
            if isinstance(item, Mapping):
                # As a description or a system file states it.
                try:
                    item = Processor(**item)
                except TypeError as error:
                    raise InvalidValueError(f"Controller.processors[{index}]: {error}") from None
            elif not isinstance(item, Processor):
                raise InvalidTypeError(
                    f"Controller.processors[{index}] is not a pulse.Processor: {item!r}"
                )
            processors.append(item)
        if processors and (self.computes or self.compute_links != ("controller",)):
            raise InvalidValueError(
                "a Controller with processors states what it computes on them; move its "
                "compute_links, op_cycles, compute_clock_ghz, max_ops, state_bytes, "
                "max_calls_per_shot, buffer_bytes and controller_form to a pulse.Processor"
            )
        object.__setattr__(self, "processors", tuple(processors))

    def as_dict(self) -> dict[str, Any]:
        """The controller as JSON-serializable data, which ``Controller(**data)``
        reads back after ``_controller_fields``."""
        data = super().as_dict()
        data["drive_lines"] = list(self.drive_lines)
        data["readout_lines"] = list(self.readout_lines)
        peers = self.peer_latency_cycles
        if isinstance(peers, Mapping):
            data["peer_latency_cycles"] = {str(k): v for k, v in sorted(peers.items())}
        if self.supports is not None:
            data["supports"] = [str(c) for c in self.supports]
        if self.processors:
            data["processors"] = [p.as_dict() for p in self.processors]
        else:
            del data["processors"]
        return data

    def supports_capability(self, capability: Capability) -> bool:
        """Whether the controller can do *capability*."""
        return self.supports is None or capability in self.supports

    @property
    def _runs_loops(self) -> bool:
        return self.supports is None or Capability.LOOP in self.supports


def _known(owners: Iterable[int | None]) -> list[int]:
    return sorted({o for o in owners if o is not None})


def _label(index: int, controller: Controller) -> str:
    name = f"controllers[{index}]"
    return f"{name} ({controller.model})" if controller.model else name


@dataclass(frozen=True)
class ControlSystem:
    """The controllers of one QPU and what connects them; see the module
    docstring.

    ``links`` are the links to classical devices, by name (``"feedback"``,
    ``"controller"``); a classical function's ``link=`` names one, and
    ``Link.controllers`` says which controllers it reaches.
    ``max_concurrent_drives``, ``max_concurrent_readouts`` and
    ``line_switch_penalty_cycles`` apply to the ``rcp`` and ``alap_rcp``
    schedules, which keep a branch on a readout at least the
    ``feedback_latency_cycles`` of the controllers that play readouts after it.
    """

    controllers: Sequence[Controller] = ()
    links: Mapping[str, Link] = field(default_factory=dict)
    max_concurrent_drives: int = 4
    max_concurrent_readouts: int = 2
    line_switch_penalty_cycles: int = 0
    _machine: MachineModel = field(init=False, repr=False, compare=False)
    _plays_everything: bool = field(init=False, repr=False, compare=False)
    # By link id, the controller and processor that run its calls.
    _placement: Mapping[int, tuple[int, int]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        from .compiler import MachineModel

        if isinstance(self.controllers, Controller) or not isinstance(self.controllers, Iterable):
            raise InvalidTypeError(
                f"ControlSystem.controllers is a list of pulse.Controller, got {self.controllers!r}"
            )
        controllers = tuple(self.controllers)
        for index, controller in enumerate(controllers):
            if not isinstance(controller, Controller):
                fix = (
                    "; make it a pulse.Controller with the same fields"
                    if isinstance(controller, ControllerLimits)
                    else ""
                )
                raise InvalidTypeError(
                    f"ControlSystem.controllers[{index}] is not a pulse.Controller: "
                    f"{controller!r}{fix}"
                )
        object.__setattr__(self, "controllers", controllers)
        if not isinstance(self.links, Mapping):
            raise InvalidTypeError(
                f"ControlSystem.links maps link names to Link, got {self.links!r}"
            )
        for name, link in self.links.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise InvalidValueError(
                    f"ControlSystem.links are keyed by name (an identifier), got {name!r}"
                )
            if not isinstance(link, Link):
                raise InvalidTypeError(
                    f"ControlSystem.links[{name!r}] must be a Link, got {link!r}"
                )
        object.__setattr__(self, "links", MappingProxyType(dict(self.links)))
        problems = _structure_problems(self, "")
        if problems:
            raise DeviceDefinitionError(
                f"{_count(len(problems), 'problem')} in the control system",
                problems=problems,
            )
        object.__setattr__(
            self,
            "_plays_everything",
            len(controllers) == 1
            and not (controllers[0].drive_lines or controllers[0].readout_lines),
        )
        object.__setattr__(self, "_placement", MappingProxyType(_placement(self)))
        # rcp holds one readout latency: the feedback latency of the slowest
        # controller that plays readouts.
        readers = [c for c in controllers if c.readout_lines] or list(controllers)
        links: dict[int | str, Link] = {name: link for name, link in self.links.items()}
        object.__setattr__(
            self,
            "_machine",
            MachineModel(
                max_concurrent_drives=self.max_concurrent_drives,
                max_concurrent_readouts=self.max_concurrent_readouts,
                readout_latency_cycles=max((c.feedback_latency_cycles for c in readers), default=0),
                line_switch_penalty_cycles=self.line_switch_penalty_cycles,
                links=links,
                controller=controllers[0] if len(controllers) == 1 else None,
                _system=self,
            ),
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.controllers,
                tuple(sorted(self.links.items())),
                self.max_concurrent_drives,
                self.max_concurrent_readouts,
                self.line_switch_penalty_cycles,
            )
        )

    def as_dict(self) -> dict[str, Any]:
        """The control system as JSON-serializable data; links name the
        controllers they reach by index."""
        return {
            "controllers": [c.as_dict() for c in self.controllers],
            "links": {
                name: {
                    **link.cost(),
                    **(
                        {}
                        if link.controllers is None
                        else {"controllers": [self._index(c) for c in link.controllers]}
                    ),
                }
                for name, link in self.links.items()
            },
            "max_concurrent_drives": self.max_concurrent_drives,
            "max_concurrent_readouts": self.max_concurrent_readouts,
            "line_switch_penalty_cycles": self.line_switch_penalty_cycles,
        }

    def _index(self, controller: Controller) -> int:
        for index, candidate in enumerate(self.controllers):
            if candidate is controller:
                return index
        return self.controllers.index(controller)

    # ── Checks of a compiled program ────────────────────────────────────────

    def _owners(self, lines: Sequence[Mapping[str, Any]]) -> tuple[list[int | None], list[str]]:
        """The index of the controller that plays each line, and a violation
        for each line no controller plays."""
        if self._plays_everything:
            return [0] * len(lines), []
        by_role: dict[str, dict[int, int]] = {"drive": {}, "readout": {}}
        for index, controller in enumerate(self.controllers):
            for carrier in controller.drive_lines:
                by_role["drive"][carrier] = index
            for carrier in controller.readout_lines:
                by_role["readout"][carrier] = index
        owners: list[int | None] = []
        found: list[str] = []
        for line in lines:
            owner = by_role.get(line["role"], {}).get(line["qudit"])
            owners.append(owner)
            if owner is None:
                found.append(
                    f"line {line['line']} (the {line['role']} line of qudit {line['qudit']}) "
                    f"is on no controller; add {line['qudit']} to the {line['role']}_lines "
                    "of one of the control system's controllers"
                )
        return owners, found

    def _violations(
        self,
        lowered: Lowered,
        lines: Sequence[Mapping[str, Any]],
        events: Sequence[dict[str, Any]],
        clocks: _clocks.ClockPlan | None,
        off: Sequence[tuple[int, int, int]],
        run_time: Sequence[str],
        waveforms: bool,
        unresolved: list[str] | None,
        natives: Sequence[int] | None,
    ) -> list[str]:
        """What the program needs of the system that it does not have: a
        controller for each line, each controller's limits on what it plays,
        its capabilities and the reach of the links. *clocks*, *off* and
        *run_time* are as ``controller._violations`` takes them."""
        from .controller import _violations

        owners, found = self._owners(lines)
        if len(self.controllers) == 1:
            found.extend(
                _violations(
                    lowered,
                    events,
                    self.controllers[0],
                    lines,
                    clocks,
                    off,
                    run_time,
                    waveforms,
                    unresolved,
                    natives,
                )
            )
        else:
            program = lowered.target.clock_ghz
            for index, controller in enumerate(self.controllers):
                label = _label(index, controller)
                if _clocks.plan(controller, program) is not None:
                    found.append(
                        f"{label}: runs on its own clock, which a control system of "
                        "several controllers does not support"
                    )
                    continue
                # Every controller sees the others' pulses as untimed.
                mine = [
                    e
                    if e["kind"] not in _PLAYED or owners[e["line"]] == index
                    else {**e, "end_cycles": None}
                    for e in events
                ]
                found.extend(
                    f"{label}: {v}"
                    for v in _violations(
                        lowered,
                        mine,
                        controller,
                        lines,
                        clocks,
                        off,
                        run_time,
                        waveforms,
                        unresolved,
                        natives,
                    )
                )
        found.extend(self._needs(lowered, events, owners, unresolved))
        return found

    def _needs(
        self,
        lowered: Lowered,
        events: Sequence[Mapping[str, Any]],
        owners: Sequence[int | None],
        unresolved: list[str] | None = None,
    ) -> list[str]:
        """The capabilities the program needs that its controllers lack, and
        the calls whose link does not reach the controllers taking part. With
        *unresolved*, a tone or phase a call's values may set is noted there
        instead, to be checked with them."""
        checks_capabilities = any(c.supports is not None for c in self.controllers)
        checks_reach = any(link.controllers is not None for link in self.links.values())
        if not (checks_capabilities or checks_reach):
            return []
        found: list[str] = []
        used = _known(owners)
        users: dict[int, set[int]] = {}

        def owner(event: Mapping[str, Any]) -> int | None:
            found: int | None = owners[event["line"]]
            return found

        def where(event: Mapping[str, Any]) -> str:
            text = f"the {event['kind']}"
            if event["kind"] in _NEEDS:
                text += f" on line {event['line']} (qudit {event['qudit']})"
            elif event["kind"] in ("submit", "answer"):
                text += f" of {event['function']}"
            start = event["start_cycles"]
            return text if start is None else f"{text} at cycle {start}"

        def need(index: int, capability: Capability, event: Mapping[str, Any]) -> None:
            controller = self.controllers[index]
            if controller.supports_capability(capability):
                return
            found.append(
                f"{_label(index, controller)}: {where(event)} needs "
                f"{capability.value!r}, which the controller does not support; run that "
                "part of the program on a controller that supports it, or write it without "
                f"{capability.value!r}"
            )

        for event in events:
            kind = event["kind"]
            if kind in _NEEDS:
                index = owner(event)
                if index is None:
                    continue
                need(index, _NEEDS[kind], event)
                if kind in _PLAYED and (
                    event["frequency_hz"] is None or event["phase_rad"] is None
                ):
                    # Without a call's values, the tone may be a parameter's.
                    if unresolved is None:
                        need(index, Capability.RUNTIME_FRAME_UPDATE, event)
                    elif not self.controllers[index].supports_capability(
                        Capability.RUNTIME_FRAME_UPDATE
                    ):
                        unresolved.append(f"line {event['line']}: frame")
                if event["conditions"]:
                    need(index, Capability.BRANCH, event)
                for condition in event["conditions"]:
                    for ref in condition["readouts"]:
                        source = owner(events[ref])
                        if source is not None and source != index:
                            need(source, Capability.FORWARD_OUTCOME, events[ref])
                    for ref in condition["answers"]:
                        need(index, Capability.ANSWER, events[ref])
                        users.setdefault(ref, set()).add(index)
            elif kind == "sync":
                for index in _known(owners[line] for line in event["lines"]):
                    need(index, Capability.SYNC, event)
        for event in events:
            kind = event["kind"]
            if kind == "submit":
                callers = _known(owner(events[ref]) for ref in event["measurements"]) or used
                for index in callers:
                    need(index, Capability.CALL, event)
                self._reach(lowered, event, callers, "sends the measurements of", found)
            elif kind == "answer":
                takers = sorted(users.get(event["index"], ())) or used
                for index in takers:
                    need(index, Capability.ANSWER, event)
                self._reach(lowered, event, takers, "answers", found)
        return found

    def _reach(
        self,
        lowered: Lowered,
        event: Mapping[str, Any],
        indices: Sequence[int],
        verb: str,
        found: list[str],
    ) -> None:
        machine = lowered.options.machine
        link = machine.link(event["link"])
        if link.controllers is None:
            return
        reached = {self._index(c) for c in link.controllers}
        name = machine.link_name(event["link"]) or f"link {event['link']}"
        for index in indices:
            if index not in reached:
                found.append(
                    f"the {event['kind']} of {event['function']} over link {name!r} {verb} "
                    f"{_label(index, self.controllers[index])}, which the link does not "
                    f"reach; add it to the link's controllers, or play those lines on "
                    f"one of {sorted(reached)}"
                )


def _structure_problems(system: ControlSystem, prefix: str) -> list[Problem]:
    """What is wrong with *system* on its own: lines on two controllers,
    peers that are not in it, links that reach controllers it does not have."""
    problems: list[Problem] = []
    controllers = system.controllers
    for index, controller in enumerate(controllers):
        for other in range(index):
            if controllers[other] is controller:
                problems.append(
                    Problem(
                        f"is the same Controller as controllers[{other}]",
                        path=f"{prefix}controllers[{index}]",
                        fix="give each controller of the system its own Controller",
                    )
                )
    for role in ("drive_lines", "readout_lines"):
        first: dict[int, int] = {}
        for index, controller in enumerate(controllers):
            for position, carrier in enumerate(getattr(controller, role)):
                if carrier in first:
                    problems.append(
                        Problem(
                            f"qudit {carrier}'s {role[:-6]} line is also on "
                            f"controllers[{first[carrier]}]",
                            path=f"{prefix}controllers[{index}].{role}[{position}]",
                            value=carrier,
                            fix="a line is on one controller",
                        )
                    )
                else:
                    first[carrier] = index
    for index, controller in enumerate(controllers):
        peers = controller.peer_latency_cycles
        if isinstance(peers, Mapping):
            for peer in peers:
                if peer == index or peer >= len(controllers):
                    problems.append(
                        Problem(
                            f"names controller {peer}, which is "
                            + ("this controller" if peer == index else "not in the system"),
                            path=f"{prefix}controllers[{index}].peer_latency_cycles",
                            value=peer,
                            fix="key it by the index of another controller "
                            f"(0..{len(controllers) - 1})",
                        )
                    )
    for name, link in system.links.items():
        for position, controller in enumerate(link.controllers or ()):
            if controller not in controllers:
                problems.append(
                    Problem(
                        "is not a controller of the system",
                        path=f"{prefix}links.{name}.controllers[{position}]",
                        fix="list the system's own Controller objects",
                    )
                )
    names = list(system.links)
    claimed: dict[int, str] = {}
    for index, controller in enumerate(controllers):
        for number, processor in enumerate(controller.processors):
            here = f"controllers[{index}].processors[{number}]"
            for position, named in enumerate(processor.links):
                path = f"{prefix}{here}.links[{position}]"
                if isinstance(named, str) and named not in system.links:
                    hint = _closest(named, names)
                    problems.append(
                        Problem(
                            "is not a link of the control system",
                            path=path,
                            value=named,
                            fix=f"did you mean {hint!r}?"
                            if hint
                            else "add it to ControlSystem(links={...})",
                        )
                    )
                    continue
                link_id = names.index(named) if isinstance(named, str) else named
                if link_id in claimed:
                    problems.append(
                        Problem(
                            f"is also a link of {claimed[link_id]}",
                            path=path,
                            value=named,
                            fix="a link belongs to one processor",
                        )
                    )
                else:
                    claimed[link_id] = here
    return problems


def _placement(system: ControlSystem) -> dict[int, tuple[int, int]]:
    """The controller and processor each processor's link ids lead to."""
    names = list(system.links)
    found = {}
    for index, controller in enumerate(system.controllers):
        for number, processor in enumerate(controller.processors):
            for link in processor.links:
                found[names.index(link) if isinstance(link, str) else link] = (index, number)
    return found


def _wiring_problems(system: ControlSystem, target: TargetDescription) -> list[Problem]:
    """The lines and acquirers of *system* that name carriers *target* lacks."""
    carriers = set(target.qubits)
    problems: list[Problem] = []
    for index, controller in enumerate(system.controllers):
        path = f"control_system.controllers[{index}]"
        for role in ("drive_lines", "readout_lines"):
            for position, carrier in enumerate(getattr(controller, role)):
                if carrier not in carriers:
                    problems.append(
                        Problem(
                            f"qudit {carrier} is not in the quantum system",
                            path=f"{path}.{role}[{position}]",
                            value=carrier,
                            fix=f"it has qudits {_ranges(sorted(carriers))}",
                        )
                    )
        for carrier in controller.acquirers:
            if carrier not in carriers:
                problems.append(
                    Problem(
                        f"qudit {carrier} is not in the quantum system",
                        path=f"{path}.acquirers",
                        value=carrier,
                        fix=f"it has qudits {_ranges(sorted(carriers))}",
                    )
                )
    return problems


# ── Controller descriptions ──────────────────────────────────────────────────

_DESCRIPTIONS = Path(__file__).resolve().parent / "controllers"
# What the installation gives, not the description.
_WIRING = (
    "drive_lines",
    "readout_lines",
    "acquirers",
    "peer_latency_cycles",
    "clock_skew_cycles",
    "alignment_tolerance_cycles",
)
_NOTICED: set[tuple[str, str, str]] = set()


def installed_descriptions() -> dict[str, Path]:
    """The controller descriptions installed with cudaq-pulse, by name
    (``"vendor.model"``)."""
    return {f"{p.parent.name}.{p.stem}": p for p in sorted(_DESCRIPTIONS.glob("*/*.toml"))}


def _toml_line(text: str, key: str) -> int | None:
    match = re.search(rf"^\s*{re.escape(key)}\s*=", text, re.MULTILINE)
    return None if match is None else text.count("\n", 0, match.start()) + 1


_PARSED: dict[tuple[str, int], Controller] = {}


def _description(path: str, mtime_ns: int) -> Controller:
    """The controller a description file states, parsed once per version of
    the file."""
    key = (path, mtime_ns)
    parsed = _PARSED.get(key)
    if parsed is None:
        parsed = _PARSED[key] = _parse_description(path)
    return parsed


def _parse_description(path: str) -> Controller:
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        where = re.search(r"\(at line (\d+), column (\d+)\)", str(error))
        raise DeviceDefinitionError(
            "the controller description is not valid TOML",
            problems=[
                Problem(
                    str(error).split(" (at line")[0],
                    file=path,
                    line=int(where.group(1)) if where else None,
                    column=int(where.group(2)) if where else None,
                )
            ],
        ) from None
    known = [f.name for f in fields(Controller) if f.name not in _WIRING]
    problems: list[Problem] = []

    def problem(key: str, message: str, fix: str = "") -> None:
        problems.append(Problem(message, path=key, fix=fix, file=path, line=_toml_line(text, key)))

    for key in data:
        if key in _WIRING:
            problem(
                key,
                "is wiring, which the installation gives",
                f"pass it when loading: pulse.controller(name, {key}=...)",
            )
        elif key not in known and key != "owner":
            hint = _closest(key, known)
            problem(key, "is not a controller field", f"did you mean {hint!r}?" if hint else "")
    for key in ("model", "version"):
        if not isinstance(data.get(key), str):
            problems.append(
                Problem(
                    f"the description has no {key} (a string)", file=path, fix=f'add {key} = "..."'
                )
            )
    if problems:
        raise DeviceDefinitionError(
            f"{_count(len(problems), 'problem')} in the controller description",
            problems=problems,
        )
    given = {k: v for k, v in data.items() if k in known}
    if "compute_links" in given:
        given["compute_links"] = tuple(given["compute_links"])
    try:
        return Controller(**given)
    except (TypeError, ValueError) as error:
        raise DeviceDefinitionError(
            "the controller description has a bad value",
            problems=[Problem(str(error), file=path)],
        ) from None


def load_controller(description: str | os.PathLike[str], /, **wiring: Any) -> Controller:
    """The controller a description states, with the installation's wiring:
    ``pulse.controller("vendor.model", drive_lines=..., readout_lines=...)``.

    *description* is the name of an installed description or the path of a
    TOML file. Each file is parsed once. *wiring* sets any field of
    ``Controller``: its lines, acquirers and peers, and overrides of the
    description's values.
    """
    if isinstance(description, str) and not description.endswith(".toml"):
        installed = installed_descriptions()
        path = installed.get(description)
        if path is None:
            hint = _closest(description, installed)
            raise NotFoundError(
                f"no controller description named {description!r}"
                + (f" (did you mean {hint!r}?)" if hint else "")
                + f"; installed: {', '.join(installed) or 'none'}, or give the path of a .toml file"
            )
    else:
        path = Path(description)
        if not path.is_file():
            raise NotFoundError(f"no controller description file {os.fspath(description)!r}")
    path = path.resolve()
    base = _description(str(path), path.stat().st_mtime_ns)
    if not wiring:
        return base
    names = [f.name for f in fields(Controller)]
    for key in wiring:
        if key not in names:
            hint = _closest(key, names)
            raise InvalidTypeError(
                f"pulse.controller() got an unknown field {key!r}"
                + (f" (did you mean {hint!r}?)" if hint else "")
            )
    return dataclasses.replace(base, **wiring)


def _notice_version(controller: Controller, path: str) -> None:
    """Notice, once, a controller made from another version of an installed
    description."""
    if controller.model is None or controller.version is None:
        return
    installed = installed_descriptions().get(controller.model)
    if installed is None:
        return
    current = load_controller(controller.model).version
    key = (controller.model, controller.version, current or "")
    if current == controller.version or key in _NOTICED:
        return
    _NOTICED.add(key)
    warnings.warn(
        f"{path}: made from {controller.model} version {controller.version}; the installed "
        f"description is version {current}. The file's values are used; to use the installed "
        f"ones, make the controller with pulse.controller({controller.model!r}, ...)",
        UpgradeNotice,
        stacklevel=4,
    )


# ── System files ─────────────────────────────────────────────────────────────

# The keys of a quantum system's data, by the field they set.
_QUDIT_FIELDS = (
    "index",
    "frequency_hz",
    "anharmonicity_hz",
    "t1_us",
    "t2_star_us",
    "label",
    "drive_params",
    "readout_params",
    "t2_echo_us",
    "thermal_population",
)
_COUPLING_KEYS = MappingProxyType(
    {
        "a": "qubit_a",
        "b": "qubit_b",
        "strength_hz": "coupling_strength_hz",
        "gate_type": "gate_type",
        "gate_duration_ns": "gate_duration_ns",
        "gate_buffer_ns": "gate_buffer_ns",
        "gate_fidelity": "gate_fidelity",
        "gate_params": "gate_params",
    }
)
_CROSSTALK_KEYS = MappingProxyType(
    {
        "a": "qubit_a",
        "b": "qubit_b",
        "zz_coupling": "zz_coupling",
        "static_zz_hz": "static_zz_hz",
        "freq_delta_hz": "freq_delta_hz",
    }
)
_SYSTEM_KEYS = (
    "name",
    "model",
    "attribution",
    "clock_ghz",
    "qudits",
    "couplings",
    "crosstalk",
)
_CONTROL_KEYS = (
    "controllers",
    "links",
    "max_concurrent_drives",
    "max_concurrent_readouts",
    "line_switch_penalty_cycles",
)
_LINK_KEYS = (
    "fixed_ns",
    "bytes_per_ns",
    "feedback_ns",
    "max_in_flight",
    "slot_bytes",
    "controllers",
)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _model_kinds() -> dict[str, type]:
    """The simulation models a system file holds, by the kind it writes."""
    from .targets.models import SimulationModel, TransmonModel

    return {"transmon": TransmonModel, "generic": SimulationModel}


def _model_data(model: Any) -> dict[str, Any] | None:
    """A simulation model as data: its kind and the levels it gives qudits."""
    if model is None:
        return None
    kind = next((k for k, cls in _model_kinds().items() if type(model) is cls), None)
    if kind is None:
        raise InvalidValueError(
            f"a system file holds only the shipped simulation models (transmon, generic); "
            f"{type(model).__name__} is not one, so give the quantum system by registered name"
        )
    data: dict[str, Any] = {"kind": kind}
    if model._levels:
        data["levels"] = {str(q): n for q, n in model._levels.items()}
    return data


def _quantum_data(target: TargetDescription) -> dict[str, Any]:
    return {
        "name": target.name,
        "model": _model_data(target.model),
        "attribution": target.attribution,
        "clock_ghz": target.clock_ghz,
        "qudits": [
            {key: _plain(getattr(q, key)) for key in _QUDIT_FIELDS}
            for q in sorted(target.qubits.values(), key=lambda q: q.index)
        ],
        "couplings": [
            {key: _plain(getattr(c, name)) for key, name in _COUPLING_KEYS.items()}
            for c in target.couplings
        ],
        "crosstalk": [
            {key: _plain(getattr(x, name)) for key, name in _CROSSTALK_KEYS.items()}
            for x in target.crosstalk
        ],
    }


def _device_data(device: Device) -> dict[str, Any]:
    from . import __version__
    from .targets.calibration import Calibration

    data: dict[str, Any] = {"format": SYSTEM_FORMAT, "written_by": f"cudaq-pulse {__version__}"}
    spec = device._spec
    data["quantum_system"] = spec if spec is not None else _quantum_data(device._base_target)
    data["layout"] = list(device.layout)
    if device.clock_ghz != device._base_target.clock_ghz:
        data["clock_ghz"] = device.clock_ghz
    calibration = device.calibration
    if calibration is not None:
        shipped = None
        try:
            shipped = Calibration.load(calibration.name, cooldown=calibration.cooldown)
        except (OSError, ValueError):
            pass
        data["calibration"] = (
            calibration.name
            if shipped == calibration
            else json.loads(json.dumps(calibration._data))
        )
        data["cooldown"] = calibration.cooldown
    if device.control_system is not None:
        data["control_system"] = device.control_system.as_dict()
    return data


class _Reader:
    """Builds objects from a system file's data, collecting a ``Problem``
    for everything wrong instead of stopping at the first."""

    def __init__(self, file: str | None = None) -> None:
        self.problems: list[Problem] = []
        self.file = file

    def problem(self, path: str, message: str, value: Any = Problem.UNSET, fix: str = "") -> None:
        self.problems.append(Problem(message, path=path, value=value, fix=fix))

    def mapping(self, data: Any, path: str, known: Iterable[str]) -> Mapping[str, Any] | None:
        if not isinstance(data, Mapping):
            self.problem(path, f"must be a mapping, got {type(data).__name__}", data)
            return None
        known = tuple(known)
        for key in data:
            if key not in known:
                hint = _closest(str(key), known)
                self.problem(
                    f"{path}.{key}",
                    "is not a known key",
                    fix=f"did you mean {hint!r}?" if hint else f"the keys are {', '.join(known)}",
                )
        return data

    def build(self, path: str, make: Any, **kwargs: Any) -> Any:
        try:
            return make(**kwargs)
        except (TypeError, ValueError) as error:
            self.problem(path, str(error))
            return None

    def quantum_system(self, data: Any, path: str) -> TargetDescription | None:
        from .targets.base import Coupling, CrosstalkEntry, Qubit, TargetDescription

        system = self.mapping(data, path, _SYSTEM_KEYS)
        if system is None:
            return None
        qudits: dict[int, Any] = {}
        for i, item in enumerate(system.get("qudits", ())):
            entry = self.mapping(item, f"{path}.qudits[{i}]", _QUDIT_FIELDS)
            if entry is not None:
                qudit = self.build(f"{path}.qudits[{i}]", Qubit, **entry)
                if qudit is not None:
                    qudits[qudit.index] = qudit
        couplings = []
        for i, item in enumerate(system.get("couplings", ())):
            entry = self.mapping(item, f"{path}.couplings[{i}]", _COUPLING_KEYS)
            if entry is not None:
                fields_ = {_COUPLING_KEYS[k]: v for k, v in entry.items() if k in _COUPLING_KEYS}
                couplings.append(self.build(f"{path}.couplings[{i}]", Coupling, **fields_))
        crosstalk = []
        for i, item in enumerate(system.get("crosstalk", ())):
            entry = self.mapping(item, f"{path}.crosstalk[{i}]", _CROSSTALK_KEYS)
            if entry is not None:
                fields_ = {_CROSSTALK_KEYS[k]: v for k, v in entry.items() if k in _CROSSTALK_KEYS}
                crosstalk.append(self.build(f"{path}.crosstalk[{i}]", CrosstalkEntry, **fields_))
        if "name" not in system:
            self.problem(path, "has no 'name'")
            return None
        rest = {k: system[k] for k in ("attribution", "clock_ghz") if k in system}
        if "model" in system:
            rest["model"] = self.model(system["model"], f"{path}.model")
        return self.build(  # type: ignore[no-any-return]
            path,
            TargetDescription,
            name=system["name"],
            qubits=qudits,
            couplings=[c for c in couplings if c is not None],
            crosstalk=[x for x in crosstalk if x is not None],
            **rest,
        )

    def model(self, data: Any, path: str) -> Any:
        if data is None:
            return None
        entry = self.mapping(data, path, ("kind", "levels"))
        if entry is None:
            return None
        kinds = _model_kinds()
        cls = kinds.get(str(entry.get("kind")))
        if cls is None:
            self.problem(f"{path}.kind", f"must be one of {', '.join(kinds)}", entry.get("kind"))
            return None
        levels = entry.get("levels")
        if isinstance(levels, Mapping):
            try:
                levels = {int(q): n for q, n in levels.items()}
            except ValueError:
                self.problem(f"{path}.levels", "keys must be qudit indices", levels)
                return None
        return self.build(path, cls, levels=levels)

    def controller(self, data: Any, path: str) -> Controller | None:
        known = [f.name for f in fields(Controller)]
        entry = self.mapping(data, path, known)
        if entry is None:
            return None
        given = {k: v for k, v in entry.items() if k in known}
        for name in ("acquirers", "peer_latency_cycles"):
            value = given.get(name)
            if isinstance(value, Mapping):
                try:
                    given[name] = {int(k): v for k, v in value.items()}
                except ValueError:
                    self.problem(f"{path}.{name}", "is keyed by integers", value)
                    return None
        if isinstance(given.get("compute_links"), list):
            given["compute_links"] = tuple(given["compute_links"])
        controller: Controller | None = self.build(path, Controller, **given)
        if controller is not None:
            _notice_version(controller, path if self.file is None else f"{self.file}: {path}")
        return controller

    def control_system(self, data: Any, path: str) -> ControlSystem | None:
        system = self.mapping(data, path, _CONTROL_KEYS)
        if system is None:
            return None
        controllers = [
            self.controller(item, f"{path}.controllers[{i}]")
            for i, item in enumerate(system.get("controllers", ()))
        ]
        if any(c is None for c in controllers):
            return None
        links: dict[str, Link] = {}
        for name, item in dict(system.get("links", {})).items():
            entry = self.mapping(item, f"{path}.links.{name}", _LINK_KEYS)
            if entry is None:
                continue
            given = dict(entry)
            reached = given.pop("controllers", None)
            if reached is not None:
                picked = []
                for i, index in enumerate(reached):
                    if isinstance(index, int) and 0 <= index < len(controllers):
                        picked.append(controllers[index])
                    else:
                        self.problem(
                            f"{path}.links.{name}.controllers[{i}]",
                            "is not the index of a controller of the system",
                            index,
                        )
                given["controllers"] = picked
            link = self.build(f"{path}.links.{name}", Link, **given)
            if link is not None:
                links[name] = link
        rest = {k: system[k] for k in _CONTROL_KEYS[2:] if k in system}
        try:
            return ControlSystem(controllers=controllers, links=links, **rest)  # type: ignore[arg-type]
        except DeviceDefinitionError as error:
            self.problems.extend(
                Problem(p.message, path=f"{path}.{p.path}", value=p.value, fix=p.fix)
                for p in error.problems
            )
        except (TypeError, ValueError) as error:
            self.problem(path, str(error))
        return None


def _check_format(config: Mapping[str, Any]) -> list[Problem]:
    """The problem with the config's ``format``, if it is not one this
    version reads."""
    given = config.get("format")
    if given is None:
        return []
    kind, _, major = str(given).rpartition("/")
    current = SYSTEM_FORMAT.rpartition("/")
    if kind != current[0] or not major.isdigit():
        return [
            Problem(
                "is not a system file format",
                path="format",
                value=given,
                fix=f"this cudaq-pulse reads {SYSTEM_FORMAT}",
            )
        ]
    if int(major) > int(current[2]):
        return [
            Problem(
                f"is newer than this cudaq-pulse reads ({SYSTEM_FORMAT})",
                path="format",
                value=given,
                fix="upgrade cudaq-pulse to load it",
            )
        ]
    return []
