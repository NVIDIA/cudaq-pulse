# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Devices and qudit handles.

A ``Device`` is a read-only description of one machine: its target (the
calibrated physics) and its layout, which maps the user's qudit numbering to
the target's carriers. ``dev.qudit_handles[i]`` is a ``Qudit``: a handle that
names one carrier of that device. A kernel called with handles runs on their
device, so one compiled kernel serves every device it is called with.

Devices compare by content: two devices built from the same config are equal
and share compiled kernels. The content is the target, layout and
calibration (a shipped one by name, cooldown and file, any other by its
values); ``name`` and ``source`` are labels, and equal devices may differ
in them.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, overload

from ._errors import (
    DeviceDefinitionError,
    InvalidTypeError,
    NotFoundError,
    Problem,
    PulseError,
    _closest,
    _count,
    _QuditIndexError,
    _ranges,
)
from .targets._registry import NoTargetError, UnknownTargetError
from .targets.base import _EXCHANGE, TargetDescription, _coupling_kind
from .targets.models import SimulationModel

if TYPE_CHECKING:
    from array import array

    from ._system import ControlSystem

__all__ = [
    "Device",
    "DeviceMismatchError",
    "NoDeviceError",
    "Qudit",
    "Register",
    "device",
]


class NoDeviceError(NoTargetError):
    """A kernel was called without qudit handles and was not compiled for a
    device, so it has no device to run on. The process target is never used
    for a kernel call."""


class DeviceMismatchError(PulseError, RuntimeError):
    """A kernel call mixes handles of different devices, or of an unpinned one."""


_CARRIER = "qudit"
_CONFIG_KEYS = (
    "format",
    "written_by",
    "quantum_system",
    "target",
    "layout",
    "calibration",
    "cooldown",
    "control_system",
    "clock_ghz",
)
# Reserved for a model of where carriers are and how they move.
_RESERVED_CONFIG = frozenset({"zones", "sites", "moves", "transport"})


def _read_only(array: Any) -> Any:
    array.setflags(write=False)
    return array


def _timebase_ghz(device: Device) -> float:
    """The clock a program for *device* is timed in, in GHz."""
    return device.clock_ghz


def _structure(device: Device) -> str:
    """What a kernel whose qudits are runtime parameters depends on: not the
    carriers or their calibration, which are bound per call."""
    content = repr((device.target.model, _timebase_ghz(device))).encode()
    return hashlib.blake2b(content, digest_size=16).hexdigest()


def _fingerprint(target: TargetDescription, layout: tuple[int, ...], calibration: Any) -> str:
    """Content hash of everything a compiled kernel depends on, and of the
    calibration the device holds: devices of one fingerprint are one device."""
    qubits = [
        (
            q.index,
            q.frequency_hz,
            q.anharmonicity_hz,
            q.t1_us,
            q.t2_star_us,
            q.t2_echo_us,
            q.thermal_population,
            sorted(q.drive_params.items()),
            sorted(q.readout_params.items()),
        )
        for q in sorted(target.qubits.values(), key=lambda q: q.index)
    ]
    # An exchange coupling adds no kind, so exchange-only devices keep their
    # fingerprints.
    couplings: list[tuple[Any, ...]] = [
        (c.qubit_a, c.qubit_b, c.coupling_strength_hz, c.gate_type, sorted(c.gate_params.items()))
        for c in target.couplings
    ]
    if target._exchange_view is not target.couplings:
        couplings = [
            entry + (() if (kind := _coupling_kind(c)) == _EXCHANGE else (kind,))
            for entry, c in zip(couplings, target.couplings)
        ]
    crosstalk = [(x.qubit_a, x.qubit_b, x.static_zz_hz) for x in target.crosstalk]
    content: tuple[Any, ...] = (
        target.model,
        target.clock_ghz,
        qubits,
        couplings,
        crosstalk,
        layout,
    )
    # Uncalibrated devices keep their fingerprints.
    if calibration is not None:
        content += (calibration._identity,)
    return hashlib.blake2b(repr(content).encode(), digest_size=16).hexdigest()


class Qudit:
    """A handle naming one carrier of a device.

    ``index`` is the carrier's index in the target (the layout is applied
    when the handle is created); ``logical`` is the position the user
    indexed. A handle is identity, not location.
    """

    __slots__ = ("device", "carrier", "index", "logical", "_key")

    device: Device
    carrier: str
    index: int
    logical: int

    def __init__(self, device: Device, index: int, logical: int) -> None:
        self.device = device
        self.carrier = _CARRIER
        self.index = index
        self.logical = logical
        self._key = (device.fingerprint, _CARRIER, index)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Qudit) and other._key == self._key

    def __hash__(self) -> int:
        return hash(self._key)

    # Read-only: a copy is the handle itself.
    def __copy__(self) -> Qudit:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Qudit:
        return self

    def __reduce__(self) -> tuple[Any, ...]:
        return (Qudit, (self.device, self.index, self.logical))

    def __repr__(self) -> str:
        return f"<Qudit {self.logical} -> {self.carrier} {self.index} of {self.device.name}>"


class Register(Sequence[Qudit]):
    """An ordered selection of qudit handles of one device."""

    __slots__ = ("_qudits",)

    def __init__(self, qudits: Sequence[Qudit]) -> None:
        self._qudits = tuple(qudits)

    def __len__(self) -> int:
        return len(self._qudits)

    @overload
    def __getitem__(self, index: int) -> Qudit: ...
    @overload
    def __getitem__(self, index: slice | Sequence[int]) -> Register: ...

    def __getitem__(self, index: Any) -> Qudit | Register:
        qudit: Qudit
        try:
            if index.__class__ is int:
                qudit = self._qudits[index]
                return qudit
            if isinstance(index, slice):
                return Register(self._qudits[index])
            if isinstance(index, Sequence) and not isinstance(index, str):
                return Register([self._qudits[i] for i in index])
            qudit = self._qudits[index]
        except IndexError:
            raise _out_of_range(index, len(self._qudits), self._owner()) from None
        return qudit

    def _owner(self) -> str:
        return "the register"

    def __iter__(self) -> Iterator[Qudit]:
        return iter(self._qudits)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Register) and other._qudits == self._qudits

    def __hash__(self) -> int:
        return hash(self._qudits)

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (self._qudits,))

    def __repr__(self) -> str:
        return f"Register({[q.logical for q in self._qudits]})"


class _Handles:
    """``dev.qudit_handles``: index with an int, a list of ints or a slice."""

    __slots__ = ("_all",)

    def __init__(self, qudits: tuple[Qudit, ...]) -> None:
        self._all = _DeviceRegister(qudits)

    def __len__(self) -> int:
        return len(self._all)

    def __iter__(self) -> Iterator[Qudit]:
        return iter(self._all)

    @overload
    def __getitem__(self, index: int) -> Qudit: ...
    @overload
    def __getitem__(self, index: slice | Sequence[int]) -> Register: ...

    def __getitem__(self, index: Any) -> Qudit | Register:
        handles: Qudit | Register = self._all[index]
        return handles

    def __reduce__(self) -> tuple[Any, ...]:
        return (_Handles, (self._all._qudits,))


class _DeviceRegister(Register):
    """All the qudits of a device, which its errors name."""

    __slots__ = ()

    def _owner(self) -> str:
        return f"device {self._qudits[0].device.name!r}" if self._qudits else "the device"


def _out_of_range(index: Any, n: int, owner: str) -> _QuditIndexError:
    """The error for *index*, which names a qudit past the *n* of *owner*."""
    wrong = index
    if isinstance(index, Sequence) and not isinstance(index, str):
        wrong = next((i for i in index if isinstance(i, int) and not -n <= i < n), index)
    if n == 0:
        return _QuditIndexError(f"qudit index {wrong!r} is out of range: {owner} has no qudits")
    return _QuditIndexError(
        f"qudit index {wrong!r} is out of range: {owner} has {_count(n, 'qudit')}, "
        f"indexed 0 to {n - 1} (or -{n} to -1)"
    )


class Device:
    """A read-only machine description; see the module docstring.

    ``clock_ghz`` is the program clock: kernels count time in its cycles. A
    config's ``"clock_ghz"`` sets it; else the fastest clock of the control
    system's controllers, when every other one divides it (controllers whose
    clocks do not divide evenly need it declared); else the target's. Each
    controller may run its own clock, a whole-cycle ratio of it (see
    ``cudaq_pulse.controller``).

    ``qubit_freq[q]`` is the 0-1 transition frequency of qudit ``q`` (Hz) and
    ``transition_freq[q, i, j]`` that of levels ``i`` and ``j``, over the
    levels the target's model gives each qudit (NaN past a qudit's last
    level); both are indexed by the user's qudit numbering and are read-only
    arrays.
    ``control_system`` is the ``ControlSystem`` that plays it, if given;
    ``compile(kernel, device=dev)`` checks every program against it.
    """

    # The registered target the device was made from, as ``as_dict`` writes it.
    _spec: str | None = None

    def __init__(
        self,
        target: TargetDescription,
        *,
        name: str,
        layout: Sequence[int] | None = None,
        source: str = "",
        calibration: Any = None,
        control_system: ControlSystem | None = None,
        clock_ghz: float | None = None,
    ) -> None:
        if control_system is not None:
            from ._system import ControlSystem

            if not isinstance(control_system, ControlSystem):
                raise InvalidTypeError(
                    f"Device control_system must be a ControlSystem, got {control_system!r}"
                )
        problems: list[Problem] = []
        if clock_ghz is not None or (control_system is not None and control_system._needs_clock):
            from ._system import _program_clock

            clock, problems = _program_clock(control_system, clock_ghz, target.clock_ghz)
            if clock is None:
                raise DeviceDefinitionError(problems=problems)
            if clock != target.clock_ghz:
                target = dataclasses.replace(target, clock_ghz=clock)
        layout = (
            tuple(sorted(target.qubits))
            if layout is None
            else _layout(layout, target, name, problems)
        )
        if control_system is not None:
            from ._system import _wiring_problems

            problems.extend(_wiring_problems(control_system, target))
        if layout is None or problems:
            raise DeviceDefinitionError(problems=problems)
        self.control_system = control_system
        # The target before its calibration was applied, as ``as_dict`` writes it.
        self._base_target = target
        self.target = target
        self.name = name
        self.source = source
        # The Calibration the target's fitted values came from, if any.
        self.calibration = calibration
        self.layout = layout
        self.clock_ghz = float(target.clock_ghz)
        self.frequencies: Mapping[int, float] = target.frequencies
        self.fingerprint = _fingerprint(target, layout, calibration)
        # What equality compares: the fingerprint alone without a control
        # system, so comparing such devices costs one string comparison.
        self._identity: Any = (
            self.fingerprint if control_system is None else (self.fingerprint, control_system)
        )
        self._structure = _structure(self)
        self.qudit_handles = _Handles(tuple(Qudit(self, p, i) for i, p in enumerate(layout)))
        self._arrays: tuple[Any, Any] | None = None
        self._table: array[float] | None = None

    def _frequency_arrays(self) -> tuple[Any, Any]:
        if self._arrays is None:
            import numpy as np

            target = self.target
            model = target.model if isinstance(target.model, SimulationModel) else SimulationModel()
            qubits = [target.qubits[p] for p in self.layout]
            levels = [model.levels_of(p) for p in self.layout]
            f01 = np.array([q.frequency_hz for q in qubits], dtype=float)
            energies = np.full((len(qubits), max(levels, default=model.levels)), np.nan)
            for row, qubit, n in zip(energies, qubits, levels):
                row[:n] = model.energies_hz(qubit, n)
            transitions = np.abs(energies[:, None, :] - energies[:, :, None])
            self._arrays = (_read_only(f01), _read_only(transitions))
        return self._arrays

    def _line_tones(
        self, index: int, readout: bool
    ) -> tuple[tuple[int, int], dict[tuple[int, int], float]]:
        """The default transition of carrier *index*, and the frequency of
        each transition its line has a tone for: every transition of a
        drive line, only the default of a readout line."""
        target = self.target
        model = target.model if isinstance(target.model, SimulationModel) else SimulationModel()
        default = model.default_transition
        if index < 0:
            # A runtime qudit: its own tone is bound per call, and no other.
            return default, {}
        qubit = target.qubits[index]
        if readout:
            return default, {default: qubit.frequency_hz}
        return default, model.transitions_hz(qubit, model.levels_of(index))

    @property
    def qubit_freq(self) -> Any:
        return self._frequency_arrays()[0]

    @property
    def transition_freq(self) -> Any:
        return self._frequency_arrays()[1]

    def qudit(self, label: str) -> Qudit:
        """The handle of the qudit labelled *label* (``Qubit.label``), as in
        ``dev.qudit("D1")``; a ``NotFoundError`` names the labels there are."""
        qubits = self.target.qubits
        for handle in self.qudit_handles:
            if qubits[handle.index].label == label:
                return handle
        labels = [qubits[p].label for p in self.layout if qubits[p].label]
        carrier = next((q.index for q in qubits.values() if q.label == label), None)
        if carrier is not None:
            raise NotFoundError(
                f"qudit {label!r} is carrier {carrier} of {self.target.name}, which the "
                f"layout of device {self.name!r} leaves out"
            )
        close = _closest(str(label), labels)
        hint = f"did you mean {close!r}? " if close else ""
        known = f"its labels are {', '.join(labels)}" if labels else "its qudits have no labels"
        raise NotFoundError(f"device {self.name!r} has no qudit labelled {label!r}; {hint}{known}")

    @property
    def quantum_system(self) -> TargetDescription:
        """The calibrated physics of the device (``target``)."""
        return self.target

    @property
    def qudits(self) -> _Handles:
        """The qudit handles, by the user's numbering (``qudit_handles``)."""
        return self.qudit_handles

    def as_dict(self) -> dict[str, Any]:
        """The device as JSON-serializable data, which ``pulse.device()``
        reads back: its format (``cudaq-pulse.system/1``), the version that
        wrote it, its quantum system (a registered name, or every value
        inline), layout and calibration, and its control system."""
        from ._system import _device_data

        return _device_data(self)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Device) and other._identity == self._identity

    def __hash__(self) -> int:
        return hash(self.fingerprint)

    # Read-only: a copy is the device itself.
    def __copy__(self) -> Device:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Device:
        return self

    def __reduce__(self) -> tuple[Any, ...]:
        from ._pickling import device_reduce

        # The constructor's inputs; its target already has the program clock,
        # which a control system is given again to check it against.
        inputs = {
            "name": self.name,
            "layout": self.layout,
            "source": self.source,
            "calibration": self.calibration,
        }
        if self.control_system is not None:
            inputs["clock_ghz"] = self.clock_ghz
        # What as_dict writes that the inputs do not give: the registered
        # target and the target before its calibration and program clock.
        state = (
            None
            if self._spec is None and self._base_target is self.target
            else {"_spec": self._spec, "_base_target": self._base_target}
        )
        return device_reduce(self, inputs, state)

    def __repr__(self) -> str:
        return f"<Device {self.name} ({len(self.layout)} qudits)>"


def _from_spec(
    spec: str,
    source: str,
    layout: Sequence[int] | None = None,
    calibration: Any = None,
    control_system: ControlSystem | None = None,
) -> Device:
    from .targets._registry import make_target

    return _calibrated(make_target(spec), spec, source, layout, calibration, control_system)


def _calibrated(
    target: TargetDescription,
    spec: str | None,
    source: str,
    layout: Sequence[int] | None,
    calibration: Any,
    control_system: ControlSystem | None = None,
    clock_ghz: float | None = None,
) -> Device:
    """The device of *target*, registered as *spec* (``None`` for a
    ``TargetDescription``), with *calibration* applied and *clock_ghz* (if
    given) its program clock."""
    base = target
    if calibration is not None:
        target = calibration.apply(target)
    elif spec == "transmon_17q":
        from .targets.calibration import shipped_calibration

        calibration = shipped_calibration("transmon_17q")
    name = target.name if spec is None else spec
    made = Device(
        target,
        name=name,
        layout=layout,
        source=source,
        calibration=calibration,
        control_system=control_system,
        clock_ghz=clock_ghz,
    )
    made._spec = spec
    made._base_target = base
    return made


def _target(spec: Any, problems: list[Problem], key: str = "target") -> TargetDescription | None:
    from .targets._registry import available_targets, make_target

    if isinstance(spec, TargetDescription):
        return spec
    if spec is None:
        problems.append(
            Problem(
                "is missing",
                path=key,
                fix="give a registered target name ("
                + ", ".join(available_targets())
                + ") or a TargetDescription",
            )
        )
        return None
    if not isinstance(spec, str):
        problems.append(
            Problem(
                f"must be a registered target name or a TargetDescription, "
                f"got {type(spec).__name__}",
                path=key,
                value=spec,
            )
        )
        return None
    try:
        return make_target(spec)
    except (UnknownTargetError, ValueError) as error:
        problems.append(Problem(Exception.__str__(error), path=key, value=spec))
        return None


def _layout(
    layout: Any, target: TargetDescription | None, name: str, problems: list[Problem]
) -> tuple[int, ...] | None:
    """*layout* as carrier indices, checked against *target* when it is known."""
    if isinstance(layout, (str, bytes)) or not isinstance(layout, Sequence):
        problems.append(
            Problem(
                f"must be a list of carrier indices, got {type(layout).__name__}",
                path="layout",
                value=layout,
            )
        )
        return None
    found = len(problems)
    carriers: list[int] = []
    first: dict[int, int] = {}
    known = target.qubits if target is not None else None
    for k, entry in enumerate(layout):
        path = f"layout[{k}]"
        index = getattr(type(entry), "__index__", None)
        if index is None or isinstance(entry, bool):
            problems.append(
                Problem(f"must be a carrier index, got {entry!r}", path=path, value=entry)
            )
            continue
        carrier: int = index(entry)
        if known is not None and carrier not in known:
            problems.append(
                Problem(
                    f"carrier {carrier} is not in target {name!r} (it has {_ranges(known)})",
                    path=path,
                    value=carrier,
                )
            )
        elif carrier in first:
            problems.append(
                Problem(
                    f"carrier {carrier} is already named by layout[{first[carrier]}]",
                    path=path,
                    value=carrier,
                    fix="each qudit names a different carrier",
                )
            )
        else:
            first[carrier] = k
        carriers.append(carrier)
    return tuple(carriers) if len(problems) == found else None


def _calibration(config: Mapping[str, Any], problems: list[Problem]) -> Any:
    """The config's calibration: a Calibration, or a shipped name or a path,
    for the config's ``cooldown`` (the file's default if absent)."""
    from .targets.calibration import Calibration

    given = config.get("calibration")
    cooldown = config.get("cooldown")
    if given is None:
        if cooldown is not None:
            problems.append(
                Problem(
                    f"{cooldown!r} is given without a calibration",
                    path="cooldown",
                    value=cooldown,
                    fix='add "calibration", a shipped name or a file path, or remove "cooldown"',
                )
            )
        return None
    if isinstance(given, Calibration):
        if cooldown is not None and cooldown != given.cooldown:
            problems.append(
                Problem(
                    f"asks for cooldown {cooldown!r} of a calibration loaded for "
                    f"{given.cooldown!r}",
                    path="cooldown",
                    value=cooldown,
                    fix=f"pass Calibration.load(..., cooldown={cooldown!r})",
                )
            )
            return None
        return given
    if isinstance(given, Mapping):
        try:
            return Calibration(given, name="inline", cooldown=cooldown)
        except (KeyError, TypeError, ValueError) as error:
            problems.append(
                Problem(f"is not calibration data: {Exception.__str__(error)}", path="calibration")
            )
            return None
    if not isinstance(given, (str, os.PathLike)):
        problems.append(
            Problem(
                "must be a shipped calibration name, the path of a calibration file, "
                f"calibration data or a Calibration, got {type(given).__name__}",
                path="calibration",
                value=given,
            )
        )
        return None
    try:
        return Calibration.load(given, cooldown=cooldown)
    except NotFoundError as error:
        problems.append(Problem(str(error), path="calibration", value=given))
    except json.JSONDecodeError as error:
        problems.append(
            Problem(f"the calibration file is not valid JSON: {error}", path="calibration")
        )
    except ValueError as error:
        problems.append(Problem(str(error), path="cooldown", value=cooldown))
    return None


def _definition(config: Mapping[str, Any], source: str, problems: list[Problem]) -> Device | None:
    """The device of *config*, or ``None`` after adding its problems in the
    order of the config's keys."""
    found = len(problems)
    device = _checked(config, source, problems)
    order = {str(key): i for i, key in enumerate(config)}
    problems[found:] = sorted(
        problems[found:], key=lambda p: order.get(p.path.split("[")[0].split(".")[0], -1)
    )
    return device


def _checked(config: Mapping[str, Any], source: str, problems: list[Problem]) -> Device | None:
    for key in sorted(set(config) & _RESERVED_CONFIG):
        problems.append(
            Problem(
                "is reserved for a model of where carriers are and how they move, "
                "which is not available yet",
                path=key,
                fix=f"remove {key!r}",
            )
        )
    for key in sorted(set(config) - set(_CONFIG_KEYS) - _RESERVED_CONFIG, key=str):
        close = _closest(str(key), _CONFIG_KEYS)
        problems.append(
            Problem(
                "unknown device config key",
                path=str(key),
                value=key,
                fix=f"did you mean {close!r}?"
                if close
                else "the keys are " + ", ".join(_CONFIG_KEYS),
            )
        )
    key = "quantum_system" if "quantum_system" in config else "target"
    if "quantum_system" in config and "target" in config:
        problems.append(
            Problem(
                "is the former name of 'quantum_system', which the config also gives",
                path="target",
                fix="remove 'target'",
            )
        )
    spec = config.get(key)
    control = config.get("control_system")
    # A target by name or as a TargetDescription, without a format or a
    # control system, needs none of the system file machinery.
    if "format" in config or isinstance(spec, Mapping) or control is not None:
        from ._system import _config_systems

        target, control = _config_systems(config, key, source, problems)
        if isinstance(spec, Mapping):
            spec = target
    else:
        target = _target(spec, problems, key)
    name = spec if isinstance(spec, str) else getattr(target, "name", "")
    layout = config.get("layout")
    if layout is not None:
        layout = _layout(layout, target, name, problems)
    calibration = _calibration(config, problems)
    clock_ghz = config.get("clock_ghz")
    if clock_ghz is not None or (control is not None and control._needs_clock):
        from ._system import _program_clock

        problems.extend(
            _program_clock(control, clock_ghz, 0.0 if target is None else target.clock_ghz)[1]
        )
    if problems or target is None:
        return None
    try:
        return _calibrated(
            target,
            spec if isinstance(spec, str) else None,
            source,
            layout,
            calibration,
            control,
            clock_ghz,
        )
    except (KeyError, ValueError) as error:
        problems.append(
            Problem(f"does not fit target {name!r}: {Exception.__str__(error)}", path="calibration")
        )
        return None


def _json_positions(text: str) -> dict[str, tuple[int, int]]:
    """The line and column of each value of the JSON *text*, by its path."""
    decoder = json.JSONDecoder()
    found: dict[str, int] = {}

    def skip(i: int) -> int:
        while text[i] in " \t\r\n":
            i += 1
        return i

    def value(i: int, path: str) -> int:
        i = skip(i)
        found[path] = i
        opening = text[i]
        if opening not in "{[":
            return decoder.raw_decode(text, i)[1]
        closing = "}" if opening == "{" else "]"
        i = skip(i + 1)
        n = 0
        while text[i] != closing:
            if opening == "{":
                key, i = decoder.raw_decode(text, i)
                i = skip(i) + 1
                i = value(i, f"{path}.{key}" if path else key)
            else:
                i = value(i, f"{path}[{n}]")
                n += 1
            i = skip(i)
            if text[i] == ",":
                i = skip(i + 1)
        return i + 1

    try:
        value(0, "")
    except (IndexError, ValueError):
        pass
    return {
        path: (text.count("\n", 0, i) + 1, i - text.rfind("\n", 0, i)) for path, i in found.items()
    }


def _located(problems: list[Problem], file: str, text: str) -> list[Problem]:
    positions = _json_positions(text)
    located = []
    for problem in problems:
        # A path the file has no value at (a missing key) points at the
        # nearest value that holds it.
        path = problem.path
        while path and path not in positions:
            path = path[: max(path.rfind("."), path.rfind("["), 0)]
        line, column = positions.get(path, (None, None))
        located.append(dataclasses.replace(problem, file=file, line=line, column=column))
    return located


def device(config: Mapping[str, Any] | str | os.PathLike[str] | None = None) -> Device:
    """The device of a target name, a config, or the path of a JSON config file.

    ``device("transmon_17q")`` is the device of a registered target,
    with every carrier in order (a name takes its argument after a colon, as
    in ``"transmon_generic:4"``); ``cudaq_pulse.targets.available_targets()``
    lists the names. ``device()`` is the process target, named by
    ``CUDAQ_PULSE_TARGET`` (``python -m cudaq_pulse --target NAME`` sets it);
    without one it raises ``NoTargetError``. A string that names no
    registered target is the path of a config file.

    A config gives the quantum system, by registered name or as a
    ``TargetDescription``, and may give a layout, a calibration, the
    control system and a program clock::

        {"quantum_system": "transmon_17q", "layout": [4, 5, 6],
         "calibration": "transmon_17q", "cooldown": "2022-main",
         "control_system": pulse.ControlSystem(controllers=[...]), "clock_ghz": 2.0}

    ``layout[i]`` is the carrier that qudit ``i`` names; by default every
    carrier of the target, in order. ``calibration`` is a ``Calibration``,
    the name of a shipped one or the path of a calibration file, and
    ``cooldown`` one of its cooldowns; the calibration's couplings replace
    the target's, and ``Device.calibration`` holds it. The ``transmon_17q``
    target uses the shipped ``transmon_17q`` calibration by default.
    ``control_system`` is a ``ControlSystem``, whose lines must be carriers
    of the quantum system. ``clock_ghz`` is the program clock kernels count
    cycles of (``Device.clock_ghz``); by default the fastest controller clock
    when the others divide it, else the quantum system's. Controller clocks
    that do not divide evenly into one need it, and the problem lists the
    clocks that work.
    ``"target"`` is the former name of ``"quantum_system"``. A system file,
    as ``Device.as_dict()`` writes it, gives each of these as data.

    A config with problems raises ``DeviceDefinitionError`` listing all of
    them, each with its path in the config and, for a file, its line and
    column; ``python -m cudaq_pulse.doctor --system FILE`` runs the same
    checks.
    """
    if config is None:
        from .targets._registry import current_target

        return current_target()
    if isinstance(config, str):
        from .targets._registry import _registry, make_target

        if config.partition(":")[0] in _registry():
            return _from_spec(config, f"target {config}")
        if not os.path.exists(config) and os.sep not in config and "." not in config:
            make_target(config)
    if not isinstance(config, (str, os.PathLike)):
        if not isinstance(config, Mapping):
            raise InvalidTypeError(f"device config must be a mapping, got {type(config).__name__}")
        problems: list[Problem] = []
        found = _definition(config, "config", problems)
        if found is None:
            raise DeviceDefinitionError(
                f"{_count(len(problems), 'problem')} in the device config", problems=problems
            )
        return found
    file = os.fspath(config)
    with open(config, encoding="utf-8") as f:
        text = f.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        problem = Problem(
            f"not valid JSON: {error.msg}", file=file, line=error.lineno, column=error.colno
        )
        raise DeviceDefinitionError(
            f"1 problem in device file {file}", problems=[problem]
        ) from error
    problems = []
    if isinstance(data, Mapping):
        found = _definition(data, f"file {file}", problems)
    else:
        found = None
        problems.append(
            Problem(f"the file must hold a JSON object, got {type(data).__name__}", value=data)
        )
    if found is None:
        raise DeviceDefinitionError(
            f"{_count(len(problems), 'problem')} in device file {file}",
            problems=_located(problems, file, text),
        )
    return found
