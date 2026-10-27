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
and share compiled kernels.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, overload

from ._errors import DeviceDefinitionError, Problem, PulseError, _closest, _count, _ranges
from .targets._registry import NoTargetError, UnknownTargetError
from .targets.base import TargetDescription

__all__ = [
    "Device",
    "DeviceMismatchError",
    "NoDeviceError",
    "Qudit",
    "Register",
    "Target",
    "device",
    "get_device",
]


class NoDeviceError(NoTargetError):
    """A kernel was called without qudit handles and was not compiled for a
    device, so it has no device to run on. The process target is never used
    for a kernel call."""


class DeviceMismatchError(PulseError, RuntimeError):
    """A kernel call mixes handles of different devices, or of an unpinned one."""


class Target(enum.Enum):
    """Targets shipped with cudaq-pulse, for ``get_device``."""

    TRANSMON_KRINNER_17Q = "transmon_krinner_17q"
    RYDBERG_CHAIN_8 = "rydberg_chain:8"
    RYDBERG_SQUARE_3X3 = "rydberg_square:3x3"


_CARRIER = "qudit"
_CONFIG_KEYS = ("target", "layout", "calibration", "cooldown")
# Reserved for a model of where carriers are and how they move.
_RESERVED_CONFIG = frozenset({"zones", "sites", "moves", "transport"})


def _read_only(array: Any) -> Any:
    array.setflags(write=False)
    return array


def _fingerprint(target: TargetDescription, layout: tuple[int, ...]) -> str:
    """Content hash of everything a compiled kernel depends on."""
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
    couplings = [
        (c.qubit_a, c.qubit_b, c.coupling_strength_hz, c.gate_type, sorted(c.gate_params.items()))
        for c in target.couplings
    ]
    crosstalk = [(x.qubit_a, x.qubit_b, x.static_zz_hz) for x in target.crosstalk]
    content = repr(
        (target.architecture, target.clock_ghz, qubits, couplings, crosstalk, layout)
    ).encode()
    return hashlib.blake2b(content, digest_size=16).hexdigest()


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
        if isinstance(index, slice):
            return Register(self._qudits[index])
        if isinstance(index, Sequence) and not isinstance(index, str):
            return Register([self._qudits[i] for i in index])
        qudit: Qudit = self._qudits[index]
        return qudit

    def __iter__(self) -> Iterator[Qudit]:
        return iter(self._qudits)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Register) and other._qudits == self._qudits

    def __hash__(self) -> int:
        return hash(self._qudits)

    def __repr__(self) -> str:
        return f"Register({[q.logical for q in self._qudits]})"


class _Handles:
    """``dev.qudit_handles``: index with an int, a list of ints or a slice."""

    __slots__ = ("_all",)

    def __init__(self, qudits: tuple[Qudit, ...]) -> None:
        self._all = Register(qudits)

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


class Device:
    """A read-only machine description; see the module docstring.

    ``qubit_freq[q]`` is the 0-1 transition frequency of qudit ``q`` (Hz) and
    ``transition_freq[q, i, j]`` that of levels ``i`` and ``j``; both are
    indexed by the user's qudit numbering and are read-only arrays.
    """

    def __init__(
        self,
        target: TargetDescription,
        *,
        name: str,
        layout: Sequence[int] | None = None,
        source: str = "",
        calibration: Any = None,
    ) -> None:
        problems: list[Problem] = []
        layout = (
            tuple(sorted(target.qubits))
            if layout is None
            else _layout(layout, target, name, problems)
        )
        if layout is None:
            raise DeviceDefinitionError(problems=problems)
        self.target = target
        self.name = name
        self.source = source
        # The Calibration the target's fitted values came from, if any.
        self.calibration = calibration
        self.layout = layout
        self.clock_ghz = float(target.clock_ghz)
        self.frequencies: Mapping[int, float] = target.frequencies
        self.fingerprint = _fingerprint(target, layout)
        self.qudit_handles = _Handles(tuple(Qudit(self, p, i) for i, p in enumerate(layout)))
        self._arrays: tuple[Any, Any] | None = None

    def _frequency_arrays(self) -> tuple[Any, Any]:
        if self._arrays is None:
            import numpy as np

            levels = 3 if self.target.architecture == "transmon" else 2
            n = len(self.layout)
            f01 = np.array([self.target.qubits[p].frequency_hz for p in self.layout], dtype=float)
            energies = np.zeros((n, levels))
            energies[:, 1] = f01
            if levels > 2:
                alpha = np.array([self.target.qubits[p].anharmonicity_hz for p in self.layout])
                energies[:, 2] = 2 * f01 + alpha
            transitions = np.abs(energies[:, None, :] - energies[:, :, None])
            self._arrays = (_read_only(f01), _read_only(transitions))
        return self._arrays

    @property
    def qubit_freq(self) -> Any:
        return self._frequency_arrays()[0]

    @property
    def transition_freq(self) -> Any:
        return self._frequency_arrays()[1]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Device) and other.fingerprint == self.fingerprint

    def __hash__(self) -> int:
        return hash(self.fingerprint)

    def __repr__(self) -> str:
        return f"<Device {self.name} ({len(self.layout)} qudits)>"


def _from_spec(
    spec: str, source: str, layout: Sequence[int] | None = None, calibration: Any = None
) -> Device:
    from .targets._registry import make_target

    return _calibrated(make_target(spec), spec, source, layout, calibration)


def _calibrated(
    target: TargetDescription,
    spec: str | None,
    source: str,
    layout: Sequence[int] | None,
    calibration: Any,
) -> Device:
    """The device of *target*, registered as *spec* (``None`` for a
    ``TargetDescription``), with *calibration* applied."""
    if calibration is not None:
        target = calibration.apply(target)
    elif spec == "transmon_krinner_17q":
        from .targets.calibration import shipped_calibration

        calibration = shipped_calibration("krinner17")
    name = target.name if spec is None else spec
    return Device(target, name=name, layout=layout, source=source, calibration=calibration)


def _target(spec: Any, problems: list[Problem]) -> TargetDescription | None:
    from .targets._registry import available_targets, make_target

    if isinstance(spec, TargetDescription):
        return spec
    if spec is None:
        problems.append(
            Problem(
                "is missing",
                path="target",
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
                path="target",
                value=spec,
            )
        )
        return None
    try:
        return make_target(spec)
    except (UnknownTargetError, ValueError) as error:
        problems.append(Problem(Exception.__str__(error), path="target", value=spec))
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
    if not isinstance(given, (str, os.PathLike)):
        problems.append(
            Problem(
                "must be a shipped calibration name, the path of a calibration file or a "
                f"Calibration, got {type(given).__name__}",
                path="calibration",
                value=given,
            )
        )
        return None
    try:
        return Calibration.load(given, cooldown=cooldown)
    except FileNotFoundError as error:
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
    spec = config.get("target")
    target = _target(spec, problems)
    name = spec if isinstance(spec, str) else getattr(target, "name", "")
    layout = config.get("layout")
    if layout is not None:
        layout = _layout(layout, target, name, problems)
    calibration = _calibration(config, problems)
    if problems or target is None:
        return None
    try:
        return _calibrated(
            target, spec if isinstance(spec, str) else None, source, layout, calibration
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
        line, column = positions.get(problem.path, (None, None))
        located.append(dataclasses.replace(problem, file=file, line=line, column=column))
    return located


def device(config: Mapping[str, Any] | str | os.PathLike[str]) -> Device:
    """A device from a config: a mapping, or the path of a JSON file.

    The config gives the target, by registered name or as a ``TargetDescription``
    description, and may give a layout and a calibration::

        {"target": "transmon_krinner_17q", "layout": [4, 5, 6],
         "calibration": "krinner17", "cooldown": "K22-main"}

    ``layout[i]`` is the carrier that qudit ``i`` names; by default every
    carrier of the target, in order. ``calibration`` is a ``Calibration``,
    the name of a shipped one or the path of a calibration file, and
    ``cooldown`` one of its cooldowns; the calibration's couplings replace
    the target's, and ``Device.calibration`` holds it. The Krinner target
    uses the shipped ``krinner17`` calibration by default.

    A config with problems raises ``DeviceDefinitionError`` listing all of
    them, each with its path in the config and, for a file, its line and
    column; ``python -m cudaq_pulse.doctor --system FILE`` runs the same
    checks.
    """
    if not isinstance(config, (str, os.PathLike)):
        if not isinstance(config, Mapping):
            raise TypeError(f"device config must be a mapping, got {type(config).__name__}")
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


def get_device(target: Target | None = None) -> Device:
    """A device of a shipped target, or the process target with no argument."""
    if target is None:
        from .targets._registry import current_target

        return current_target()
    if not isinstance(target, Target):
        raise TypeError(f"get_device() expects a cudaq_pulse.Target, got {type(target).__name__}")
    return _from_spec(target.value, f"shipped {target.name}")
