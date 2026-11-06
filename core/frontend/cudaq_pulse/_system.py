# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The control system of a QPU: its controllers, their wiring and links.

A ``Controller`` is one box: what it can play, the lines it drives and
reads out (``drive_lines``, ``readout_lines``: carrier indices of the
quantum system), its latency to its peers, what it supports
(``Capability``), what it computes (``Processor``) and the backend that
lowers its programs. A ``ControlSystem`` is the controllers of one QPU, the links to
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
import math
import os
import re
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

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
from ._units import Time, as_nanoseconds
from .classical import Link

if TYPE_CHECKING:
    from dataclasses import dataclass as _dataclass_when_used
    from fractions import Fraction
    from pathlib import Path

    from . import _clocks
    from ._capability import Capability
    from ._device import Device
    from .compiler import Lowered, _MachineModel
    from .controller import WaveformMemoryScope
    from .targets.base import TargetDescription
else:
    from ._errors import _dataclass_when_used

__all__ = ["SYSTEM_FORMAT", "ControlSystem", "Controller", "Processor"]

SYSTEM_FORMAT = "cudaq-pulse.system/1"


_PLAYED = frozenset({"drive", "readout"})
_NEEDS = MappingProxyType({"drive": "play", "readout": "acquire", "wait": "wait"})


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
    if value.__class__ is Time:
        raise InvalidTypeError(
            f"Controller.{name} is a whole number of the controller's own cycles, "
            f"got the time {value}; divide it by the controller's cycle time"
        )
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidTypeError(f"Controller.{name} is a whole number of cycles, got {value!r}")
    if value < 0:
        raise InvalidValueError(f"Controller.{name} cannot be negative, got {value}")
    return value


_PROCESSOR_LIMITS = (
    "max_ops_per_function",
    "max_ops_total",
    "max_state_bytes_per_function",
    "max_state_bytes_total",
    "max_calls_per_shot",
    "max_bytes_per_shot",
)


def _processor_ops() -> tuple[str, ...]:
    """What a processor's cycles_per_op and unsupported name: the classes of
    operations, and "loop", a rolled loop's counter and jump per iteration."""
    from .controller import OP_CLASSES

    return (*OP_CLASSES, "loop")


def _op_name(given: Any, field_name: str) -> str:
    ops = _processor_ops()
    if given in ops:
        return str(given)
    hint = _closest(str(given), ops)
    raise InvalidValueError(
        f"Processor.{field_name} names an unknown operation {given!r}"
        + (f" (did you mean {hint!r}?)" if hint else "")
        + f"; the operations are {', '.join(ops)}"
    )


def _dispatch_keys(given: Any, owner: str) -> Mapping[str, int]:
    if not isinstance(given, Mapping):
        raise InvalidTypeError(f"{owner}.dispatch_keys maps function names to keys, got {given!r}")
    owners: dict[int, str] = {}
    for name, key in given.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise InvalidValueError(
                f"{owner}.dispatch_keys is keyed by function name, got {name!r}"
            )
        if isinstance(key, bool) or not isinstance(key, int) or not 0 <= key <= 0xFFFFFFFF:
            raise InvalidValueError(
                f"{owner}.dispatch_keys[{name!r}] is {key!r}; a dispatch key is a whole "
                "number from 0 to 0xFFFFFFFF"
            )
        if key in owners:
            raise InvalidValueError(
                f"{owner}.dispatch_keys gives {key} to both {owners[key]!r} and {name!r}; "
                "give each function its own key"
            )
        owners[key] = name
    return MappingProxyType(dict(given))


@_dataclass_when_used(frozen=True)
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
        object.__setattr__(self, "dispatch_keys", _dispatch_keys(self.dispatch_keys, "Processor"))

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


@_dataclass_when_used(frozen=True)
class Controller:
    """One controller: what it can play, its wiring and its processors;
    every limit is in its own clock cycles.

    ``clock_ghz``: the controller's clock (``None``: the program clock,
    ``Device.clock_ghz``). It must be a whole-cycle ratio of the program
    clock; its program counts its own cycles.
    ``cycles_per_timestep``: the controller starts and times operations only
    every this many of its own cycles; every operation on its lines must
    start and last whole timesteps (see "Clocks" in the controller backends
    guide; ``pulse.round_duration`` rounds a length to them).
    ``instruction_cycles`` and ``min_idle_cycles``: two operations on one
    line start exactly ``instruction_cycles`` apart, or at least
    ``instruction_cycles + min_idle_cycles`` apart, because the controller
    issues one instruction per ``instruction_cycles`` and idles no less than
    ``min_idle_cycles``.
    ``sample_rate_ghz``: the rate its lines play samples at (``None``: one
    per cycle of its clock). Its ratio to the clock may be a fraction, and
    every operation on its lines must start and last a whole number of
    samples, which the timestep includes: at 2.4 GHz on a 1 GHz clock, whole
    multiples of 5 cycles. A length off them is a compile error, never
    rounded.
    ``waveform_memory_samples``: a waveform memory holds this many samples,
    at ``sample_rate_ghz``; waveforms of one shape at different amplitudes
    share it (``shape_key``).
    ``waveform_memory``: ``"per_line"`` (each line has a memory of
    ``waveform_memory_samples``) or ``"shared"`` (one memory serves every
    line); see ``WaveformMemoryScope``.
    ``max_tones_per_line``: distinct tone frequencies one line plays.
    ``max_amplitude``: the largest magnitude of a sample its lines play
    (``None``: no limit). The compiler sets no amplitude limit of its own;
    a drive or readout whose envelope peaks above this one is a violation
    naming its line. Exports list it among the limits only when it is set.
    ``detectors`` and ``detector_recovery_cycles``: the detector reading
    each qudit; readouts on one detector are serialized, each holding it
    for its duration plus ``detector_recovery_cycles``.
    ``feedback_latency_cycles``: the least time from the end of a readout to
    the start of an operation in a branch on its outcome.
    ``supports_timeouts``: whether the controller can stop waiting for an
    answer after a timeout (``result(timeout=...)``, ``latency_mode=
    "timeout"``); without, only waits that block or are budgeted are
    allowed.
    ``max_pulses``: the drives and readouts of one program; a controller
    that supports ``"loop"`` holds a rolled loop's body once.
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
    part of a calibrated line delay (``ControlSystem.line_delays_ns``) that
    whole grid steps cannot.
    ``alignment_tolerance_cycles``: how far its pulses may land from where
    they are meant to be on other controllers, and, without fine delay, how
    far a line's delay may be from a whole number of grid steps (the
    compiler rounds it to the nearest one).
    ``persistent_frames``: whether the frame of each tone, its frequency and
    phase, persists from one shot to the next. Without, every shot starts
    from the reset frame, and the export gives a frame change no later
    operation uses on some path as a frame update of its own; with, it
    exports instead the frame each tone ends the shot in (``frame_state``).
    ``blocks_on_calls``: whether the controller plays nothing while one of
    its classical calls is out: each call holds every line from its
    submission until its answer is due, after the lines before it are done.
    The scheduler places calls so, and a program played on such a
    controller is checked for it; a call in a kernel with its own
    ``schedule=`` then does not compile. Exports list it among the limits
    only when it is set. In a control system of several controllers, a
    call holds the lines of each such controller that issues it, reads a
    measurement it sends or plays a pulse that uses its answer.

    ``drive_lines`` and ``readout_lines`` are the carriers whose drive and
    readout lines it plays (carrier indices of the quantum system); the only
    controller of a system that lists none plays every line.
    ``peer_latency_cycles`` is the time an outcome this controller reads
    takes to reach another controller, in this controller's own cycles: one
    number for every receiver, or a mapping from the receiving controller's
    index in the system (a receiver it does not list: 0).
    ``clock_skew_cycles`` bounds how far its clock is from the system's.
    ``supports`` lists its capabilities (``None``: everything); a program
    that needs one it lacks fails to compile. ``backend`` is the controller
    backend that lowers its programs, by entry-point name. ``model`` and
    ``version`` identify the description it was made from. ``processors``
    are its classical processors (``Processor``): the classical functions
    it computes itself, and their limits.
    """

    clock_ghz: float | None = None
    cycles_per_timestep: int = 1
    instruction_cycles: int = 1
    min_idle_cycles: int = 0
    sample_rate_ghz: float | None = None
    waveform_memory_samples: int | None = None
    max_tones_per_line: int | None = None
    max_amplitude: float | None = None
    detectors: Mapping[int, int] = field(default_factory=dict)
    detector_recovery_cycles: int = 0
    feedback_latency_cycles: int = 0
    max_pulses: int | None = None
    max_fine_delay_ns: float | Time = 0.0
    alignment_tolerance_cycles: float = 0
    supports_timeouts: bool = True
    max_instructions: int | None = None
    max_program_duration: float | Time | None = None
    waveform_memory: WaveformMemoryScope | str = "per_line"
    persistent_frames: bool = False
    blocks_on_calls: bool = False
    drive_lines: Sequence[int] = ()
    readout_lines: Sequence[int] = ()
    peer_latency_cycles: int | Mapping[int, int] = 0
    clock_skew_cycles: int = 0
    supports: Sequence[Capability | str] | None = None
    backend: str | None = None
    model: str | None = None
    version: str | None = None
    processors: Sequence[Processor] = ()

    def __post_init__(self) -> None:
        for name in ("persistent_frames", "blocks_on_calls"):
            if not isinstance(getattr(self, name), bool):
                raise InvalidValueError(
                    f"Controller.{name} must be a bool, got {getattr(self, name)!r}"
                )
        if self.clock_ghz is not None and not self.clock_ghz > 0:
            raise InvalidValueError(f"Controller.clock_ghz must be positive, got {self.clock_ghz}")
        from .controller import WaveformMemoryScope

        try:
            scope = WaveformMemoryScope(self.waveform_memory)
        except ValueError:
            raise InvalidValueError(
                "Controller.waveform_memory must be 'per_line' or 'shared', "
                f"got {self.waveform_memory!r}"
            ) from None
        object.__setattr__(self, "waveform_memory", scope)
        steps = self.cycles_per_timestep
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise InvalidValueError(
                f"Controller.cycles_per_timestep must be an int of at least 1, got {steps!r}"
            )
        if self.instruction_cycles < 1:
            raise InvalidValueError("Controller.instruction_cycles must be at least 1")
        for name in ("min_idle_cycles", "detector_recovery_cycles", "feedback_latency_cycles"):
            if getattr(self, name) < 0:
                raise InvalidValueError(f"Controller.{name} cannot be negative")
        object.__setattr__(self, "max_fine_delay_ns", as_nanoseconds(self.max_fine_delay_ns))
        if self.max_program_duration is not None:
            object.__setattr__(
                self, "max_program_duration", _longest_program(self.max_program_duration)
            )
        for name in ("max_fine_delay_ns", "alignment_tolerance_cycles"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise InvalidValueError(f"Controller.{name} must be a finite number >= 0")
        peak = self.max_amplitude
        if peak is not None and (
            isinstance(peak, bool)
            or not isinstance(peak, (int, float))
            or not math.isfinite(peak)
            or not peak > 0
        ):
            raise InvalidValueError(
                f"Controller.max_amplitude must be a positive number or None, got {peak!r}"
            )
        rate = self.sample_rate_ghz
        if rate is not None and (
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not math.isfinite(rate)
            or not rate > 0
        ):
            raise InvalidValueError(
                f"Controller.sample_rate_ghz must be a positive number or None, got {rate!r}"
            )
        for name in ("waveform_memory_samples", "max_tones_per_line", "max_pulses"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise InvalidValueError(f"Controller.{name} must be positive or None")
        size = self.max_instructions
        if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 1):
            raise InvalidValueError(
                f"Controller.max_instructions must be a positive int or None, got {size!r}"
            )
        object.__setattr__(self, "detectors", MappingProxyType(dict(self.detectors)))
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
        if self.supports is not None:
            if isinstance(self.supports, str):
                raise InvalidTypeError(
                    f"Controller.supports is a list of capabilities, got {self.supports!r}"
                )
            from ._capability import Capability

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
        object.__setattr__(self, "processors", tuple(processors))

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

    def as_dict(self) -> dict[str, Any]:
        """The controller as JSON-serializable data, which ``Controller(**data)``
        reads back after ``_controller_fields``."""
        # Every field in order: the class has none that fields() would leave out.
        data = {name: getattr(self, name) for name in self.__dataclass_fields__}
        data["detectors"] = {str(k): v for k, v in sorted(self.detectors.items())}
        for name in ("max_amplitude", "max_instructions", "max_program_duration"):
            if data[name] is None:
                del data[name]
        # Memory per line exports as it did before the key existed.
        if self.waveform_memory == "per_line":
            del data["waveform_memory"]
        else:
            data["waveform_memory"] = str(self.waveform_memory)
        if not self.persistent_frames:
            del data["persistent_frames"]
        if not self.blocks_on_calls:
            del data["blocks_on_calls"]
        data["drive_lines"] = list(self.drive_lines)
        data["readout_lines"] = list(self.readout_lines)
        peers = self.peer_latency_cycles
        if not isinstance(peers, int):
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
    def _machine(self) -> _MachineModel:
        """What compiling for this controller alone needs, as
        ``compile(control_system=controller)`` does."""
        return _one_controller(self)._machine

    @property
    def _runs_loops(self) -> bool:
        """Whether a rolled loop's body counts once against ``max_pulses``."""
        return self.supports is None or "loop" in self.supports


def _longest_program(value: Any) -> float:
    """``Controller.max_program_duration`` in ns."""
    if not isinstance(value, bool) and isinstance(value, (int, float)):
        ns = float(as_nanoseconds(value))
        if math.isfinite(ns) and ns > 0:
            return ns
    raise InvalidValueError(
        "Controller.max_program_duration must be a positive time or number of "
        f"nanoseconds, or None; got {value!r}"
    )


# What an error about the system compile() was given as one Controller says.
_DESCRIBE = (
    "give compile() a pulse.ControlSystem that describes the whole control system: its "
    "controllers, the lines each plays and its links"
)


def _one_controller(controller: Controller) -> ControlSystem:
    """The system of *controller* alone, for ``compile(control_system=controller)``:
    what the one controller cannot take is an error that says to describe
    the whole system."""
    try:
        system = ControlSystem(controllers=[controller])
    except DeviceDefinitionError as error:
        raise DeviceDefinitionError(
            f"{_count(len(error.problems), 'problem')} in the control system of the one "
            f"Controller compile() was given; {_DESCRIBE}",
            problems=error.problems,
        ) from None
    object.__setattr__(system._machine, "_bare", True)
    return system


def _checked_delays(
    delays: Mapping[tuple[str, int], float | Time],
) -> Mapping[tuple[str, int], float]:
    from ._clocks import checked_line_delays

    return checked_line_delays(delays, "ControlSystem")


def _known(owners: Iterable[int | None]) -> list[int]:
    return sorted({o for o in owners if o is not None})


def _cycles_text(cycles: Fraction) -> str:
    return str(cycles.numerator) if cycles.denominator == 1 else f"{float(cycles):g}"


def _takers(
    events: Sequence[Mapping[str, Any]], owners: Sequence[int | None]
) -> dict[int, set[int]]:
    """The controllers that take each answer: those playing a line event
    that tests it or computes a value from it at run time."""
    takers: dict[int, set[int]] = {}
    for event in events:
        if event["kind"] not in _NEEDS:
            continue
        owner = owners[event["line"]]
        if owner is None:
            continue
        for condition in event["conditions"]:
            for ref in condition["answers"]:
                takers.setdefault(ref, set()).add(owner)
        stack = [value for key, value in event.items() if key != "conditions"]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                ref = value.get("answer")
                if type(ref) is int:
                    takers.setdefault(ref, set()).add(owner)
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
    return takers


def _label(index: int, controller: Controller) -> str:
    name = f"controllers[{index}]"
    return f"{name} ({controller.model})" if controller.model else name


@dataclass(frozen=True)
class ControlSystem:
    """The controllers of one QPU and what connects them; see the module
    docstring.

    A system without controllers states only its links, lanes, reference
    clock and line delays, and nothing checks what its controllers can
    play.
    ``links`` are the links to classical devices, by name (``"feedback"``,
    ``"controller"``); a classical function's ``link=`` names one, or gives
    its id: the links' positions, so the first is link 0, the default. A
    link id a program uses that is not listed gets ``Link()``.
    ``Link.controllers`` says which controllers a link reaches. Their slots
    and latencies apply to every schedule, because they are hardware.
    ``max_concurrent_drives``, ``max_concurrent_readouts`` and
    ``line_switch_penalty_cycles`` apply to the ``rcp`` and ``alap_rcp``
    schedules, which keep a branch on a readout at least the
    ``feedback_latency_cycles`` of the controllers that play readouts after it,
    and pulses of two controllers that follow each other on one lane their
    start uncertainty apart.
    ``reference_clock_ghz`` is a clock every controller locks to, if they
    share one: each controller's clock must be a whole multiple of it, and
    without it each controller's start is known only to within one of its
    own cycles. ``line_delays_ns`` are the calibrated delays of the lines
    the controllers play, keyed by ``("drive", qudit)`` or
    ``("readout", qudit)``, as times (``2 * pulse.ns``) or numbers of ns:
    every operation on the line starts that much later than scheduled.
    Whole steps of the line's grid move its operations, which are then
    checked like any others; the rest is the line's ``fine_delay_ns`` in
    the controller's program, which its controller must be able to add
    (``Controller.max_fine_delay_ns``) or tolerate
    (``Controller.alignment_tolerance_cycles``).
    """

    controllers: Sequence[Controller] = ()
    links: Mapping[str, Link] = field(default_factory=dict)
    max_concurrent_drives: int = 4
    max_concurrent_readouts: int = 2
    line_switch_penalty_cycles: int = 0
    reference_clock_ghz: float | None = None
    line_delays_ns: Mapping[tuple[str, int], float | Time] = field(default_factory=dict)
    # Set by __post_init__, and declared for type checkers only: as fields
    # they would slow the first use of the class.
    if TYPE_CHECKING:
        _machine: _MachineModel = field(init=False)
        _plays_everything: bool = field(init=False)
        # By link id and controller, the processor that runs its calls there.
        _placement: Mapping[int, Mapping[int, int]] = field(init=False)
        # The clocks of the controllers that run their own, by label.
        _own_clocks: Mapping[str, float] = field(init=False)
        # Whether a device of it checks its program clock against its clocks.
        _needs_clock: bool = field(init=False)
        # (role, carrier) -> the index of the controller that plays the line.
        _line_owners: Mapping[tuple[str, int], int] = field(init=False)
        # By program clock, the clocks of a program on several controllers.
        _plans: dict[float, _clocks.ClockPlan | None] = field(init=False)
        # Whether a sync lines up its controllers' pulses exactly, so a check
        # may take a rolled loop's copies for its first (``_summarized``) when
        # they count program cycles.
        _lines_up: bool = field(init=False)
        # The indices of the controllers that block on calls.
        _blocking: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        from .compiler import _NO_DELAYS, _MachineModel

        if isinstance(self.controllers, Controller) or not isinstance(self.controllers, Iterable):
            raise InvalidTypeError(
                f"ControlSystem.controllers is a list of pulse.Controller, got {self.controllers!r}"
            )
        controllers = tuple(self.controllers)
        for index, controller in enumerate(controllers):
            if not isinstance(controller, Controller):
                raise InvalidTypeError(
                    f"ControlSystem.controllers[{index}] is not a pulse.Controller: {controller!r}"
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
        ghz = self.reference_clock_ghz
        if ghz is not None and not ghz > 0:
            raise InvalidValueError(
                f"ControlSystem.reference_clock_ghz must be positive, got {ghz}"
            )
        delays = self.line_delays_ns
        object.__setattr__(
            self,
            "line_delays_ns",
            _checked_delays(delays) if delays else _NO_DELAYS,
        )
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
        object.__setattr__(
            self,
            "_own_clocks",
            MappingProxyType(
                {
                    _label(index, c): c.clock_ghz
                    for index, c in enumerate(controllers)
                    if c.clock_ghz is not None
                }
            ),
        )
        object.__setattr__(
            self,
            "_needs_clock",
            bool(self._own_clocks) or self.reference_clock_ghz is not None,
        )
        object.__setattr__(
            self,
            "_line_owners",
            MappingProxyType(
                {
                    (role, carrier): index
                    for index, controller in enumerate(controllers)
                    for role in ("drive", "readout")
                    for carrier in getattr(controller, f"{role}_lines")
                }
            ),
        )
        object.__setattr__(self, "_plans", {})
        object.__setattr__(
            self,
            "_lines_up",
            len(controllers) == 1
            or (
                self.reference_clock_ghz is not None
                and not self.line_delays_ns
                and not any(c.clock_skew_cycles for c in controllers)
            ),
        )
        object.__setattr__(
            self,
            "_blocking",
            tuple(index for index, c in enumerate(controllers) if c.blocks_on_calls),
        )
        links: dict[int | str, Link] = {name: link for name, link in self.links.items()}
        one = controllers[0] if len(controllers) == 1 else None
        # Programs are checked against the system as one controller (or,
        # without controllers, not at all) when that controller plays every
        # line, supports everything, computes nothing, and no link reaches
        # only some controllers.
        alone = not controllers or (
            one is not None
            and self._plays_everything
            and one.supports is None
            and not one.processors
            and not any(
                link.controllers is not None or link.per_controller for link in links.values()
            )
        )
        object.__setattr__(
            self,
            "_machine",
            _MachineModel(
                max_concurrent_drives=self.max_concurrent_drives,
                max_concurrent_readouts=self.max_concurrent_readouts,
                readout_latency_cycles=self._outcome_latency(None) if controllers else 0,
                line_switch_penalty_cycles=self.line_switch_penalty_cycles,
                links=links,
                controller=one,
                reference_clock_ghz=self.reference_clock_ghz,
                line_delays_ns=self.line_delays_ns,
                _system=None if alone else self,
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
                self.reference_clock_ghz,
                tuple(self.line_delays_ns.items()),
            )
        )

    def __reduce__(self) -> tuple[Any, ...]:
        from ._pickling import frozen_reduce

        return frozen_reduce(self)

    def as_dict(self) -> dict[str, Any]:
        """The control system as JSON-serializable data; links name the
        controllers they reach by index, and line delays are keyed by role,
        then qudit."""
        data = {
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
        if self.reference_clock_ghz is not None:
            data["reference_clock_ghz"] = self.reference_clock_ghz
        if self.line_delays_ns:
            delays: dict[str, dict[str, float]] = {}
            for (role, qudit), ns in self.line_delays_ns.items():
                delays.setdefault(role, {})[str(qudit)] = ns
            data["line_delays_ns"] = delays
        return data

    @property
    def _one_box(self) -> bool:
        """Whether its programs are checked as of its one controller, with
        its processors: the controller plays every line and supports
        everything, and no link reaches only some controllers."""
        return (
            self._plays_everything
            and self.controllers[0].supports is None
            and not any(
                link.controllers is not None or link.per_controller for link in self.links.values()
            )
        )

    def _clock_plan(self, program_ghz: float) -> _clocks.ClockPlan | None:
        """The clocks of a program at *program_ghz* on the controllers (of a
        system of several), with each line's controller and the readout
        latency in program cycles; ``None`` when all run at the program clock
        in steps of one cycle and no reference clock is given."""
        try:
            return self._plans[program_ghz]
        except KeyError:
            pass
        from . import _clocks

        controllers = self.controllers
        found = _clocks.controllers_plan(
            controllers,
            [_label(index, c) for index, c in enumerate(controllers)],
            self._line_owners,
            program_ghz,
            self.reference_clock_ghz,
        )
        if found is not None:
            found = dataclasses.replace(found, readout_latency_cycles=self._outcome_latency(found))
        self._plans[program_ghz] = found
        return found

    def _outcome_latency(self, clocks: _clocks.ClockPlan | None) -> int:
        """The longest a readout's outcome takes to be usable, in program
        cycles: on the controller that reads it, after its feedback latency;
        on another, after its ``peer_latency_cycles`` to it and both
        controllers' start uncertainty too. The scheduler holds one readout
        latency, so every readout waits for this one."""
        controllers = self.controllers
        if clocks is None and len(controllers) == 1:
            return controllers[0].feedback_latency_cycles
        from fractions import Fraction

        everyone = range(len(controllers))
        readers = [i for i in everyone if controllers[i].readout_lines] or everyone
        users = [
            i for i in everyone if controllers[i].drive_lines or controllers[i].readout_lines
        ] or everyone
        uncertainty = self._start_uncertainty(clocks) if len(controllers) > 1 else None
        longest = Fraction(0)
        for reader in readers:
            source = controllers[reader]
            ratio = Fraction(1) if clocks is None else clocks.controllers[reader].ratio
            longest = max(longest, source.feedback_latency_cycles / ratio)
            if uncertainty is None:
                continue
            peers = source.peer_latency_cycles
            for user in users:
                if user != reader:
                    peer = peers.get(user, 0) if isinstance(peers, Mapping) else peers
                    latency = (source.feedback_latency_cycles + peer) / ratio
                    longest = max(longest, latency + uncertainty[reader] + uncertainty[user])
        return math.ceil(longest)

    def _start_uncertainty(self, clocks: _clocks.ClockPlan | None) -> list[Fraction]:
        """How well each controller's start is known, in program cycles: its
        ``clock_skew_cycles``, plus one of its own cycles when the system
        shares no reference clock (*clocks*: the program's, ``None`` when
        every controller runs at the program clock)."""
        from fractions import Fraction

        extra = 0 if self.reference_clock_ghz is not None else 1
        return [
            (c.clock_skew_cycles + extra)
            / (Fraction(1) if clocks is None else clocks.controllers[i].ratio)
            for i, c in enumerate(self.controllers)
        ]

    def _outcome_latencies(self, clocks: _clocks.ClockPlan | None) -> list[list[int]]:
        """By reader, then user: how long after a readout ends its outcome is
        usable on each controller, in program cycles (its feedback latency on
        the reader; with its ``peer_latency_cycles`` and both controllers'
        start uncertainty on another)."""
        from fractions import Fraction

        controllers = self.controllers
        uncertainty = self._start_uncertainty(clocks)
        table = []
        for reader, source in enumerate(controllers):
            ratio = Fraction(1) if clocks is None else clocks.controllers[reader].ratio
            peers = source.peer_latency_cycles
            row = []
            for user in range(len(controllers)):
                if user == reader:
                    row.append(math.ceil(source.feedback_latency_cycles / ratio))
                    continue
                peer = peers.get(user, 0) if isinstance(peers, Mapping) else peers
                latency = (source.feedback_latency_cycles + peer) / ratio
                row.append(math.ceil(latency + uncertainty[reader] + uncertainty[user]))
            table.append(row)
        return table

    def _controllers_attribute(self, program_ghz: float, n_qudits: int) -> str:
        """The module attribute ``pulse.controllers`` of a program at
        *program_ghz* on *n_qudits* qudits: the controller of each line, the
        outcome latencies between controllers, how far apart their starts may
        be (in program cycles) and each one's start uncertainty in ns; with
        ``pulse.blocks_on_calls``, those that block on calls."""
        from fractions import Fraction

        clocks = self._clock_plan(program_ghz)
        uncertainty = self._start_uncertainty(clocks)
        count = len(self.controllers)
        owners = self._line_owners

        def array(kind: str, values: Iterable[Any]) -> str:
            text = ", ".join(map(str, values))
            return f"array<{kind}: {text}>" if text else f"array<{kind}>"

        fields = [
            f"{role} = " + array("i64", (owners.get((role, q), -1) for q in range(n_qudits)))
            for role in ("drive", "readout")
        ]
        latency = (v for row in self._outcome_latencies(clocks) for v in row)
        spread = (
            0 if a == b else math.ceil(uncertainty[a] + uncertainty[b])
            for a in range(count)
            for b in range(count)
        )
        ns = (repr(float(u / Fraction(program_ghz))) for u in uncertainty)
        fields.append(f"outcome_latency = {array('i64', latency)}")
        fields.append(f"spread = {array('i64', spread)}")
        fields.append(f"uncertainty_ns = {array('f64', ns)}")
        text = "{pulse.controllers = {" + ", ".join(fields) + "}"
        if self._blocking:
            text += f", pulse.blocks_on_calls = {array('i64', self._blocking)}"
        return text + "}"

    def _links_attribute(
        self, machine: _MachineModel, used: Sequence[int], program_ghz: float
    ) -> str:
        """``machine.links_attribute(used)`` on this system: each link that
        joins controllers carries the start uncertainty between the two of
        them known least well, in ns, which the budget of a call whose sender
        and taker the scheduler cannot tell adds. A per-controller link joins
        none (its calls stay on one) and carries its name."""
        from fractions import Fraction

        each = self._start_uncertainty(self._clock_plan(program_ghz))
        entries = []
        for link_id in sorted(set(used)):
            link = machine.link(link_id)
            if link.per_controller:
                entry = link.attribute(0.0, machine.link_name(link_id))
            else:
                reached = link.controllers
                indices = (
                    range(len(self.controllers))
                    if reached is None
                    else sorted({self._index(c) for c in reached})
                )
                worst = sorted((each[i] for i in indices), reverse=True)
                uncertainty = 0.0
                if len(worst) > 1 and worst[0] + worst[1]:
                    uncertainty = float((worst[0] + worst[1]) / Fraction(program_ghz))
                entry = link.attribute(uncertainty)
            entries.append(f'"{link_id}" = {entry}')
        return f"{{classical.links = {{{', '.join(entries)}}}}}"

    def _forwarding(
        self,
        events: Sequence[Mapping[str, Any]],
        owners: Sequence[int | None],
        clocks: _clocks.ClockPlan | None,
    ) -> list[str]:
        """The pulses that branch on, or compute with, a readout of another
        controller before its outcome can reach them: the reader's feedback
        latency, its ``peer_latency_cycles`` to the user, and both
        controllers' start uncertainty, each in its own cycles (events in
        program cycles)."""
        from fractions import Fraction

        found: list[str] = []
        uncertainty: list[Fraction] | None = None
        for event in events:
            if event["kind"] not in _PLAYED:
                continue
            user = owners[event["line"]]
            if user is None or event["start_cycles"] is None:
                continue
            uses = [
                (ref, "branches on")
                for condition in event["conditions"]
                for ref in condition["readouts"]
            ]
            stack = [value for key, value in event.items() if key != "conditions"]
            while stack:
                value = stack.pop()
                if isinstance(value, dict):
                    ref = value.get("readout")
                    if type(ref) is int:
                        uses.append((ref, "computes with"))
                    stack.extend(value.values())
                elif isinstance(value, list):
                    stack.extend(value)
            for ref, verb in dict.fromkeys(uses):
                readout = events[ref]
                reader = owners[readout["line"]]
                if reader is None or reader == user or readout["end_cycles"] is None:
                    continue
                if uncertainty is None:
                    uncertainty = self._start_uncertainty(clocks)
                source = self.controllers[reader]
                peers = source.peer_latency_cycles
                peer = peers.get(user, 0) if isinstance(peers, Mapping) else peers
                ratio = Fraction(1) if clocks is None else clocks.controllers[reader].ratio
                latency = (source.feedback_latency_cycles + peer) / ratio
                spread = uncertainty[reader] + uncertainty[user]
                total = latency + spread
                ready = readout["end_cycles"] + math.ceil(total)
                if event["start_cycles"] >= ready:
                    continue
                found.append(
                    f"{_label(user, self.controllers[user])}: the {event['kind']} on line "
                    f"{event['line']} at {event['start_cycles']} {verb} the readout "
                    f"of {_label(reader, source)} ending at {readout['end_cycles']}, whose "
                    f"outcome reaches it {_cycles_text(total)} cycles later, at {ready} "
                    f"({_cycles_text(latency)} of feedback and peer latency, and "
                    f"{_cycles_text(spread)} of the two controllers' start uncertainty"
                    + (
                        ""
                        if self.reference_clock_ghz is not None
                        else ", without a shared reference_clock_ghz"
                    )
                    + f"); start it at {ready} or later: schedule with rcp, which waits "
                    f"for it, or wait {ready - event['start_cycles']} cycles before it on "
                    "its line"
                )
        return found

    def _alignment(
        self,
        lines: Sequence[Mapping[str, Any]],
        events: Sequence[Mapping[str, Any]],
        owners: Sequence[int | None],
        clocks: _clocks.ClockPlan | None,
        program_ghz: float,
    ) -> list[str]:
        """The pulses a sync lines up across controllers that may land
        further from where they are meant to than their
        ``alignment_tolerance_cycles``: the first pulse after the sync on each
        of its lines lands within its controller's start uncertainty and what
        its line delay is rounded by (each in the controller's cycles)."""
        import bisect

        if (
            self.reference_clock_ghz is not None
            and not self.line_delays_ns
            and not any(c.clock_skew_cycles for c in self.controllers)
        ):
            return []
        by_line: dict[int, list[Mapping[str, Any]]] = {}
        for event in events:
            if event["kind"] in _PLAYED and event["start_cycles"] is not None:
                by_line.setdefault(event["line"], []).append(event)
        for items in by_line.values():
            items.sort(key=lambda e: (e["start_cycles"], e["index"]))
        starts = {line: [e["start_cycles"] for e in items] for line, items in by_line.items()}
        found: list[str] = []
        for sync in events:
            if sync["kind"] != "sync" or sync["start_cycles"] is None:
                continue
            if len(_known(owners[line] for line in sync["lines"])) < 2:
                continue
            paired: list[tuple[int, Mapping[str, Any]]] = []
            for line in sync["lines"]:
                owner = owners[line]
                if owner is None or line not in by_line:
                    continue
                after = bisect.bisect_left(starts[line], sync["start_cycles"])
                if after < len(starts[line]):
                    paired.append((owner, by_line[line][after]))
            parts = _known(owner for owner, _ in paired)
            if len(parts) < 2:
                continue
            trigger = 0 if self.reference_clock_ghz is not None else 1
            for owner in parts:
                controller = self.controllers[owner]
                mine = [e for o, e in paired if o == owner]
                skew = controller.clock_skew_cycles
                rounded = max(
                    self._delay_rounding(lines[e["line"]], owner, clocks, program_ghz) for e in mine
                )
                spread = skew + trigger + rounded
                if spread <= controller.alignment_tolerance_cycles + 1e-9:
                    continue
                terms = [f"{skew} of clock_skew_cycles"] if skew else []
                if trigger:
                    terms.append("1 without a shared reference_clock_ghz")
                if rounded:
                    terms.append(f"{rounded:g} of a rounded line delay")
                fixes = ["share a reference_clock_ghz between them"] if trigger else []
                fixes.append(f"raise its alignment_tolerance_cycles to {math.ceil(spread - 1e-9)}")
                pulses = ", ".join(f"the {e['kind']} on line {e['line']}" for e in mine)
                found.append(
                    f"{_label(owner, controller)}: the sync at {sync['start_cycles']} lines "
                    f"up {pulses} with the pulses of "
                    + ", ".join(_label(o, self.controllers[o]) for o in parts if o != owner)
                    + f", but they land within {spread:g} of its cycles of where they are "
                    f"meant to ({', '.join(terms)}), more than its "
                    f"alignment_tolerance_cycles of {controller.alignment_tolerance_cycles}; "
                    + ", or ".join(fixes)
                )
        return found

    def _delay_rounding(
        self,
        line: Mapping[str, Any],
        owner: int,
        clocks: _clocks.ClockPlan | None,
        program_ghz: float,
    ) -> float:
        """How far a line's delay is rounded to its grid, in its controller's
        cycles: 0 when the controller adds the rest as fine delay."""
        ns = self.line_delays_ns.get((line["role"], line["qudit"]))
        controller = self.controllers[owner]
        if not ns or controller.max_fine_delay_ns > 0:
            return 0.0
        own = None if clocks is None else clocks.controllers[owner]
        step_ns = (1 if own is None else own.grid) / program_ghz
        rest = abs(ns - round(ns / step_ns) * step_ns)
        return 0.0 if rest < 1e-9 else rest * (program_ghz if own is None else own.clock_ghz)

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
        played = self._line_owners
        owners: list[int | None] = []
        found: list[str] = []
        for line in lines:
            owner = played.get((line["role"], line["qudit"]))
            owners.append(owner)
            if owner is None:
                where = f"line {line['line']} (the {line['role']} line of qudit {line['qudit']})"
                found.append(
                    f"{where} is not on the one Controller compile() was given; add "
                    f"{line['qudit']} to its {line['role']}_lines, or {_DESCRIBE}"
                    if getattr(self._machine, "_bare", False)
                    else f"{where} is on no controller; add {line['qudit']} to the "
                    f"{line['role']}_lines of one of the control system's controllers"
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
        from . import _clocks
        from .controller import _computes_at_run_time, _violations

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
            found.extend(self._needs(lowered, events, owners, unresolved))
        else:
            needs = self._needs(lowered, events, owners, unresolved)
            needs.extend(self._forwarding(events, owners, clocks))
            needs.extend(self._alignment(lines, events, owners, clocks, lowered.target.clock_ghz))
            if self._blocking:
                needs.extend(self._held(lowered, events, owners))
            if clocks is not None and clocks.owners is not None:
                # Each event in its controller's cycles, once for them all.
                found.extend(
                    _clocks.off_grid(
                        clocks, lines, events, True, waveforms, _computes_at_run_time(lowered)
                    )
                )
            for index, controller in enumerate(self.controllers):
                label = _label(index, controller)
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
                        None,
                        off,
                        run_time,
                        waveforms,
                        unresolved,
                        natives,
                        held=False,
                    )
                )
            found.extend(needs)
        if self._placement:
            if unresolved is None:
                from .controller import usage

                found.extend(usage(lowered).violations)
            else:
                # When calls are due can depend on a call's values.
                unresolved.append("processors")
        return found

    def _held(
        self,
        lowered: Lowered,
        events: Sequence[Mapping[str, Any]],
        owners: Sequence[int | None],
    ) -> list[str]:
        """The pulses a controller that blocks on calls plays while a call
        that holds it is out (*events* in program cycles): a call holds each
        that issues it (``_callers``), reads a measurement it sends or plays a
        line event that uses its answer."""
        from .controller import _held_text, _played_while_out

        if not lowered.classical or not lowered.scheduled:
            return []
        blocking = set(self._blocking)
        takers = _takers(events, owners)
        answers: dict[tuple[Any, ...], list[int]] = {}
        for event in events:
            if event["kind"] == "answer":
                taken = (event["link"], event["function"], tuple(event["iteration"]))
                answers.setdefault((*taken, event["start_cycles"]), []).append(event["index"])
        submits = [e for e in events if e["kind"] == "submit"]
        calls = lowered.classical_report()
        used = _known(owners)
        holders = []
        for submit, call in zip(submits, calls, strict=True):
            held: set[int | None] = set(self._callers(submit, events, owners, used, takers))
            held.update(owners[events[ref]["line"]] for ref in submit["measurements"])
            key = (call.link, call.compiled_name, tuple(call.iteration), call.taken_cycles)
            for answer in answers.get(key, ()):
                held.update(takers.get(answer, ()))
            holders.append(held & blocking)
        return [
            f"{_label(owner, self.controllers[owner])}: {_held_text(submit, call, event)}"
            for submit, call, event in _played_while_out(
                events, calls, lambda index, event: owners[event["line"]] in holders[index]
            )
            if (owner := owners[event["line"]]) is not None
        ]

    def _summarized(
        self,
        lowered: Lowered,
        lines: Sequence[Mapping[str, Any]],
        events: Sequence[dict[str, Any]],
        repeats: Sequence[Sequence[int]],
        clocks: _clocks.ClockPlan | None,
        off: Sequence[tuple[int, int, int]],
        unresolved: list[str] | None,
        natives: Sequence[int] | None,
        rates: tuple[float, float] | None,
    ) -> list[Any] | None:
        """``_violations`` of the events of a summarizing walk of loops of
        one period, on a system that ``_lines_up``, as ``_loops.parts``
        gives them, with *rates* those of
        ``controller._sample_rate_violations``; ``None`` unless every copy of
        a loop needs and forwards what its first does (the walk summarizes
        no calls), each period is whole timesteps of every controller, and
        *rates* come last. A sync lines up the pulses of several controllers
        exactly, so ``_alignment`` finds nothing."""
        from . import _clocks, _loops

        if clocks is not None and clocks.owners is not None:
            grids = {c.grid for c in clocks.controllers}
            if any(r[3] % grid for r in repeats for grid in grids):
                return None
            if _clocks.off_grid(clocks, lines, events):
                return None
        if rates is not None and self._placement:
            return None
        owners, found = self._owners(lines)
        if self._needs(lowered, events, owners, unresolved):
            return None
        if len(self.controllers) == 1:
            from .controller import _native_clock

            found.extend(
                _loops.parts(
                    lowered,
                    events,
                    repeats,
                    self.controllers[0],
                    lines,
                    unresolved,
                    rates,
                    (),
                    _native_clock(clocks),
                    off,
                    natives,
                )
            )
        else:
            ranges = [(r[0], r[1]) for r in repeats]
            for i, event in enumerate(events):
                user = owners[event["line"]] if event["kind"] in _PLAYED else None
                if user is None:
                    continue
                for condition in event["conditions"]:
                    for ref in condition["readouts"]:
                        # A later copy's readout reaches an event after its
                        # loop later than the first copy's does.
                        if owners[events[ref]["line"]] not in (None, user) and any(
                            begin <= ref < end and not begin <= i < end for begin, end in ranges
                        ):
                            return None
            if self._forwarding(events, owners, clocks):
                return None
            for index, controller in enumerate(self.controllers):
                mine = [
                    e
                    if e["kind"] not in _PLAYED or owners[e["line"]] == index
                    else {**e, "end_cycles": None}
                    for e in events
                ]
                found.extend(
                    _loops.labeled(
                        _loops.parts(
                            lowered,
                            mine,
                            repeats,
                            controller,
                            lines,
                            unresolved,
                            None,
                            (),
                            None,
                            (),
                            natives,
                        ),
                        f"{_label(index, controller)}: ",
                    )
                )
        if self._placement:
            if unresolved is None:
                from .controller import usage

                found.extend(usage(lowered).violations)
            else:
                unresolved.append("processors")
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
        checks_reach = any(
            link.controllers is not None or link.per_controller for link in self.links.values()
        )
        if not (checks_capabilities or checks_reach):
            return []
        found: list[str] = []
        used = _known(owners)
        users = _takers(events, owners)

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

        def need(index: int, capability: str, event: Mapping[str, Any]) -> None:
            controller = self.controllers[index]
            if controller.supports is None or capability in controller.supports:
                return
            needs = f"{_label(index, controller)}: {where(event)} needs {capability!r}"
            found.append(
                f"{needs}, which the one Controller compile() was given does not support; "
                f"write that part of the program without {capability!r}, or {_DESCRIBE}"
                if getattr(self._machine, "_bare", False)
                else f"{needs}, which the controller does not support; run that part of the "
                "program on a controller that supports it, or write it without "
                f"{capability!r}"
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
                        need(index, "runtime_frame_update", event)
                    elif "runtime_frame_update" not in (self.controllers[index].supports or ()):
                        unresolved.append(f"line {event['line']}: frame")
                if event["conditions"]:
                    need(index, "branch", event)
                for condition in event["conditions"]:
                    for ref in condition["readouts"]:
                        source = owner(events[ref])
                        if source is not None and source != index:
                            need(source, "forward_outcome", events[ref])
                    for ref in condition["answers"]:
                        need(index, "answer", events[ref])
            elif kind == "sync":
                for index in _known(owners[line] for line in event["lines"]):
                    need(index, "sync", event)
        for event in events:
            kind = event["kind"]
            if kind == "submit" or kind == "answer":
                callers = self._callers(event, events, owners, used, users)
                for index in callers:
                    need(index, "call" if kind == "submit" else "answer", event)
                verb = "sends the measurements of" if kind == "submit" else "answers"
                self._reach(lowered, event, callers, verb, found)
        return found

    def _parts(
        self, events: Sequence[Mapping[str, Any]], owners: Sequence[int | None]
    ) -> list[list[int]]:
        """The controllers that take part in each event, as ``_needs`` counts
        them: a line event's, a sync's lines', and a call's (``_callers``)."""
        used = _known(owners)
        takers = _takers(events, owners)
        parts: list[list[int]] = []
        for event in events:
            kind = event["kind"]
            if kind in _NEEDS:
                owner = owners[event["line"]]
                parts.append([] if owner is None else [owner])
            elif kind == "sync":
                parts.append(_known(owners[line] for line in event["lines"]))
            elif kind == "submit" or kind == "answer":
                parts.append(self._callers(event, events, owners, used, takers))
            else:
                parts.append(used)
        return parts

    def _callers(
        self,
        event: Mapping[str, Any],
        events: Sequence[Mapping[str, Any]],
        owners: Sequence[int | None],
        used: list[int],
        takers: Mapping[int, set[int]],
    ) -> list[int]:
        """The controllers that take part in a submit or answer *event*: for
        a call over a per-controller link, the one whose instance runs it;
        else a submit's callers (those of its measurements) and an answer's
        takers (those of the events that test it or compute with it), else
        every controller *used* when the link is shared, else the one its
        link leads to (``_home``)."""
        instance = event.get("controller")
        if instance is not None:
            return [instance]
        if event["kind"] == "submit":
            found = _known(owners[events[ref]["line"]] for ref in event["measurements"])
        else:
            found = sorted(takers.get(event["index"], ()))
        if found:
            return found
        if used and not _per_controller(self, event["link"]):
            return used
        return [self._home(event["link"])]

    def _home(self, link_id: int) -> int:
        """The controller a call over *link_id* is made on when the program
        plays no line: the first with a processor that runs it, else the
        first the link reaches, else ``controllers[0]``."""
        placed = self._placement.get(link_id)
        if placed:
            return min(placed)
        reached = self._machine.link(link_id).controllers
        return min(self._index(c) for c in reached) if reached else 0

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
        if link.per_controller and len(indices) > 1:
            name = machine.link_name(event["link"]) or f"link {event['link']}"
            on = ", ".join(_label(i, self.controllers[i]) for i in indices)
            found.append(
                f"the {event['kind']} of {event['function']} over link {name!r}, which has "
                f"one instance per controller, is on {on}; a call over it runs on one "
                "controller's instance: send its measurements and use its answer on one "
                "controller, or make the link shared (drop per_controller)"
            )
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
    if any(c.drive_lines or c.readout_lines for c in controllers):
        played = {
            (role[:-6], carrier)
            for c in controllers
            for role in ("drive_lines", "readout_lines")
            for carrier in getattr(c, role)
        }
        for role, carrier in system.line_delays_ns:
            if (role, carrier) not in played:
                problems.append(
                    Problem(
                        f"delays qudit {carrier}'s {role} line, which no controller plays",
                        path=f"{prefix}line_delays_ns",
                        value=(role, carrier),
                        fix=f"add {carrier} to the {role}_lines of the controller that plays it",
                    )
                )
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
    claimed: dict[tuple[int, int], str] = {}
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
                own = _per_controller(system, link_id)
                key = (link_id, index if own else -1)
                if key in claimed:
                    problems.append(
                        Problem(
                            f"is also a link of {claimed[key]}",
                            path=path,
                            value=named,
                            fix="a link with an instance per controller belongs to one "
                            "processor of each controller"
                            if own
                            else "a link belongs to one processor, or with "
                            "Link(per_controller=True) to one of each controller",
                        )
                    )
                else:
                    claimed[key] = here
    return problems


def _per_controller(system: ControlSystem, link_id: int) -> bool:
    names = list(system.links)
    return link_id < len(names) and system.links[names[link_id]].per_controller


def _placement(system: ControlSystem) -> dict[int, dict[int, int]]:
    """For each processor's link id, the processor that runs its calls on
    each controller: one controller for a shared link, each controller with
    a processor for it for a per-controller link."""
    names = list(system.links)
    found: dict[int, dict[int, int]] = {}
    for index, controller in enumerate(system.controllers):
        for number, processor in enumerate(controller.processors):
            for link in processor.links:
                link_id = names.index(link) if isinstance(link, str) else link
                found.setdefault(link_id, {})[index] = number
    return found


def _wiring_problems(system: ControlSystem, target: TargetDescription) -> list[Problem]:
    """The lines and detectors of *system* that name carriers *target* lacks."""
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
        for carrier in controller.detectors:
            if carrier not in carriers:
                problems.append(
                    Problem(
                        f"qudit {carrier} is not in the quantum system",
                        path=f"{path}.detectors",
                        value=carrier,
                        fix=f"it has qudits {_ranges(sorted(carriers))}",
                    )
                )
    for role, carrier in system.line_delays_ns:
        if carrier not in carriers:
            problems.append(
                Problem(
                    f"delays qudit {carrier}'s {role} line; the qudit is not in the quantum system",
                    path="control_system.line_delays_ns",
                    value=(role, carrier),
                    fix=f"it has qudits {_ranges(sorted(carriers))}",
                )
            )
    return problems


# ── Controller descriptions ──────────────────────────────────────────────────

# Paths are strings here: building a control system does not need pathlib.
_DESCRIPTIONS = os.path.join(os.path.dirname(os.path.realpath(__file__)), "controllers")
# What the installation gives, not the description.
_WIRING = (
    "drive_lines",
    "readout_lines",
    "detectors",
    "peer_latency_cycles",
    "clock_skew_cycles",
    "alignment_tolerance_cycles",
)
_NOTICED: set[tuple[str, str, str]] = set()


def installed_descriptions() -> dict[str, Path]:
    """The controller descriptions installed with cudaq-pulse, by name
    (``"vendor.model"``)."""
    from pathlib import Path

    return {name: Path(path) for name, path in _installed().items()}


def _installed() -> dict[str, str]:
    """``installed_descriptions()``, by path string."""
    found = {}
    for vendor in sorted(os.listdir(_DESCRIPTIONS)):
        folder = os.path.join(_DESCRIPTIONS, vendor)
        if os.path.isdir(folder):
            for file in sorted(os.listdir(folder)):
                if file.endswith(".toml"):
                    found[f"{vendor}.{file[:-5]}"] = os.path.join(folder, file)
    return found


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
    import tomllib

    with open(path, encoding="utf-8") as f:
        text = f.read()
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
    ``Controller``: its lines, detectors and peers, and overrides of the
    description's values.
    """
    if isinstance(description, str) and not description.endswith(".toml"):
        installed = _installed()
        path = installed.get(description)
        if path is None:
            hint = _closest(description, installed)
            raise NotFoundError(
                f"no controller description named {description!r}"
                + (f" (did you mean {hint!r}?)" if hint else "")
                + f"; installed: {', '.join(installed) or 'none'}, or give the path of a .toml file"
            )
    else:
        path = os.fspath(description)
        if not os.path.isfile(path):
            raise NotFoundError(f"no controller description file {path!r}")
    path = os.path.realpath(path)
    base = _description(path, os.stat(path).st_mtime_ns)
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
    "drive",
    "readout",
    "t2_echo_us",
    "thermal_population",
)
_COUPLING_FIELDS = ("qudits", "strength_hz", "gate")
_GATE_FIELDS = ("name", "duration_ns", "buffer_ns", "interaction_frequency_hz")
_RESIDUAL_ZZ_FIELDS = ("qudits", "rate_hz")
# The keys an entry must give, with the fix for each.
_QUDITS_FIX = "give the indices of the two qudits, as [0, 1]"
_COUPLING_REQUIRED = MappingProxyType(
    {
        "qudits": _QUDITS_FIX,
        "strength_hz": "give the strength in Hz, or null if it is not measured yet",
    }
)
_GATE_REQUIRED = MappingProxyType(
    {"name": 'give the name of the gate, as "cz"', "duration_ns": "give how long it takes, in ns"}
)
_RESIDUAL_ZZ_REQUIRED = MappingProxyType(
    {"qudits": _QUDITS_FIX, "rate_hz": "give the ZZ rate in Hz"}
)
_SYSTEM_KEYS = (
    "name",
    "model",
    "attribution",
    "clock_ghz",
    "qudits",
    "couplings",
    "residual_zz",
)
_CONTROL_KEYS = (
    "controllers",
    "links",
    "max_concurrent_drives",
    "max_concurrent_readouts",
    "line_switch_penalty_cycles",
    "reference_clock_ghz",
    "line_delays_ns",
)
_LINK_KEYS = (
    "fixed_ns",
    "bytes_per_ns",
    "feedback_ns",
    "max_in_flight",
    "slot_bytes",
    "controllers",
    "per_controller",
)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _component_kinds() -> dict[str, dict[str, type[Any]]]:
    """The modality components of a qudit a system file holds, by the member
    they set and the kind it writes."""
    from .targets.base import DispersiveReadout, MicrowaveDrive

    return {"drive": {"microwave": MicrowaveDrive}, "readout": {"dispersive": DispersiveReadout}}


def _component_data(member: str, component: Any) -> dict[str, Any] | None:
    if component is None:
        return None
    kinds = _component_kinds()[member]
    kind = next((k for k, cls in kinds.items() if type(component) is cls), None)
    if kind is None:
        from ._errors import UnsupportedError

        raise UnsupportedError(
            f"a system file holds a qudit's {member} as one of {', '.join(kinds)}; "
            f"got a {type(component).__name__}"
        )
    return {"kind": kind, **dict(zip(component._fields, component._values, strict=True))}


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
            {
                key: _component_data(key, value) if key in ("drive", "readout") else _plain(value)
                for key in _QUDIT_FIELDS
                for value in (getattr(q, key),)
            }
            for q in sorted(target.qubits.values(), key=lambda q: q.index)
        ],
        "couplings": [
            {
                "qudits": list(c.qudits),
                "strength_hz": c.strength_hz,
                "gate": None
                if c.gate is None
                else {key: _plain(getattr(c.gate, key)) for key in _GATE_FIELDS},
            }
            for c in target.couplings
        ],
        "residual_zz": [
            {key: _plain(getattr(x, key)) for key in _RESIDUAL_ZZ_FIELDS}
            for x in target.residual_zz
        ],
    }


def _device_data(device: Device) -> dict[str, Any]:
    import json

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


def _config_systems(
    config: Mapping[str, Any], key: str, source: str, problems: list[Problem]
) -> tuple[TargetDescription | None, ControlSystem | None]:
    """The quantum and control systems of a device config that is a system
    file or gives a control system, each ``None`` with *problems* noted."""
    from ._device import _target

    problems.extend(_check_format(config))
    reader = _Reader(None if source == "config" else source)
    spec = config.get(key)
    if isinstance(spec, Mapping):
        target = reader.quantum_system(spec, key)
    else:
        target = _target(spec, problems, key)
    control = config.get("control_system")
    if isinstance(control, Mapping):
        control = reader.control_system(control, "control_system")
    elif control is not None and not isinstance(control, ControlSystem):
        problems.append(
            Problem(
                f"must be a ControlSystem or its data, got {type(control).__name__}",
                path="control_system",
                value=control,
            )
        )
        control = None
    problems.extend(reader.problems)
    if control is not None and target is not None:
        problems.extend(_wiring_problems(control, target))
    return target, control


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

    def component(self, data: Any, path: str, member: str) -> Any:
        """A qudit's *member* (``drive`` or ``readout``): the component of the
        kind *data* names, with its fields."""
        kinds = _component_kinds()[member]
        if not isinstance(data, Mapping):
            self.problem(path, f"must be a mapping, got {type(data).__name__}", data)
            return None
        kind = data.get("kind")
        made = kinds.get(kind) if isinstance(kind, str) else None
        if made is None:
            self.problem(
                f"{path}.kind",
                f"is not a {member} kind" if "kind" in data else "is missing",
                kind if "kind" in data else Problem.UNSET,
                fix=f"the kinds are {', '.join(kinds)}",
            )
            return None
        known = made._fields
        self.mapping(data, path, ("kind", *known))
        return self.build(path, made, **{k: data[k] for k in known if k in data})

    def build[T](self, path: str, make: Callable[..., T], **kwargs: Any) -> T | None:
        try:
            return make(**kwargs)
        except (TypeError, ValueError) as error:
            self.problem(path, str(error))
            return None

    def quantum_system(self, data: Any, path: str) -> TargetDescription | None:
        from .targets.base import Coupling, Qubit, ResidualZZ, TargetDescription

        system = self.mapping(data, path, _SYSTEM_KEYS)
        if system is None:
            return None
        qudits: dict[int, Any] = {}
        for i, item in enumerate(system.get("qudits", ())):
            entry = self.mapping(item, f"{path}.qudits[{i}]", _QUDIT_FIELDS)
            if entry is not None:
                entry = {
                    key: self.component(value, f"{path}.qudits[{i}].{key}", key)
                    if key in ("drive", "readout") and value is not None
                    else value
                    for key, value in entry.items()
                }
                qudit = self.build(f"{path}.qudits[{i}]", Qubit, **entry)
                if qudit is not None:
                    qudits[qudit.index] = qudit
        # The entries read, each with its index in the file.
        entries: dict[str, list[tuple[int, Any]]] = {"couplings": [], "residual_zz": []}
        for i, item in enumerate(system.get("couplings", ())):
            where = f"{path}.couplings[{i}]"
            entry = self.mapping(item, where, _COUPLING_FIELDS)
            if entry is None or self.missing(entry, where, _COUPLING_REQUIRED):
                continue
            gate = entry.get("gate")
            if gate is not None and (gate := self.gate(gate, f"{where}.gate")) is None:
                continue
            given = {k: entry[k] for k in ("qudits", "strength_hz")}
            coupling = self.build(where, Coupling, **given, gate=gate)
            if coupling is not None:
                entries["couplings"].append((i, coupling))
        for i, item in enumerate(system.get("residual_zz", ())):
            where = f"{path}.residual_zz[{i}]"
            entry = self.mapping(item, where, _RESIDUAL_ZZ_FIELDS)
            if entry is None or self.missing(entry, where, _RESIDUAL_ZZ_REQUIRED):
                continue
            given = {k: v for k, v in entry.items() if k in _RESIDUAL_ZZ_FIELDS}
            residual = self.build(where, ResidualZZ, **given)
            if residual is not None:
                entries["residual_zz"].append((i, residual))
        if "name" not in system:
            self.problem(path, "has no 'name'")
            return None
        rest = {k: system[k] for k in ("attribution", "clock_ghz") if k in system}
        if "model" in system:
            rest["model"] = self.model(system["model"], f"{path}.model")
        try:
            return TargetDescription(
                name=system["name"],
                qubits=qudits,
                couplings=[c for _, c in entries["couplings"]],
                residual_zz=[x for _, x in entries["residual_zz"]],
                **rest,
            )
        except DeviceDefinitionError as error:
            # The path in the file of an entry's problem, which the system
            # gives as couplings[3]... by its own index: entries the file
            # gives that could not be read are not in the system.
            def in_file(entry_path: str) -> str:
                kind, _, rest = entry_path.partition("[")
                index, _, tail = rest.partition("]")
                return f"{kind}[{entries[kind][int(index)][0]}]{tail}"

            self.problems.extend(
                Problem(
                    p.message,
                    path=f"{path}.{in_file(p.path)}",
                    value=p.value,
                    fix=p.fix,
                )
                for p in error.problems
            )
        except (TypeError, ValueError) as error:
            self.problem(path, str(error))
        return None

    def gate(self, data: Any, path: str) -> Any:
        from .targets.base import Gate

        entry = self.mapping(data, path, _GATE_FIELDS)
        if entry is None or self.missing(entry, path, _GATE_REQUIRED):
            return None
        return self.build(path, Gate, **{k: v for k, v in entry.items() if k in _GATE_FIELDS})

    def missing(self, entry: Mapping[str, Any], path: str, required: Mapping[str, str]) -> bool:
        """Whether *entry* lacks a key of *required*, each noted as a problem
        with the fix *required* gives for it."""
        absent = [key for key in required if key not in entry]
        for key in absent:
            self.problem(f"{path}.{key}", "is missing", fix=required[key])
        return bool(absent)

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
        for name in ("detectors", "peer_latency_cycles"):
            value = given.get(name)
            if isinstance(value, Mapping):
                try:
                    given[name] = {int(k): v for k, v in value.items()}
                except ValueError:
                    self.problem(f"{path}.{name}", "is keyed by integers", value)
                    return None
        controller: Controller | None = self.build(path, Controller, **given)
        if controller is not None:
            _notice_version(controller, path if self.file is None else f"{self.file}: {path}")
        return controller

    def line_delays(self, data: Any, path: str) -> dict[tuple[str, int], Any] | None:
        """System-file line delays, ``{"drive": {"5": 4.0}}``, keyed as
        ``ControlSystem.line_delays_ns`` takes them."""
        roles = self.mapping(data, path, ("drive", "readout"))
        if roles is None:
            return None
        delays: dict[tuple[str, int], Any] = {}
        for role, by_qudit in roles.items():
            if not isinstance(by_qudit, Mapping):
                self.problem(f"{path}.{role}", "maps qudits to delays in ns", by_qudit)
                return None
            for qudit, ns in by_qudit.items():
                if not str(qudit).isdigit():
                    self.problem(f"{path}.{role}", "is keyed by qudit indices", qudit)
                    return None
                delays[(role, int(qudit))] = ns
        return delays

    def control_system(self, data: Any, path: str) -> ControlSystem | None:
        system = self.mapping(data, path, _CONTROL_KEYS)
        if system is None:
            return None
        read = [
            self.controller(item, f"{path}.controllers[{i}]")
            for i, item in enumerate(system.get("controllers", ()))
        ]
        controllers = [c for c in read if c is not None]
        if len(controllers) < len(read):
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
        if "line_delays_ns" in rest:
            delays = self.line_delays(rest["line_delays_ns"], f"{path}.line_delays_ns")
            if delays is None:
                return None
            rest["line_delays_ns"] = delays
        try:
            return ControlSystem(controllers=controllers, links=links, **rest)
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
