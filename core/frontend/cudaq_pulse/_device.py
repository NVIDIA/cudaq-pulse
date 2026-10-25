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

import enum
import hashlib
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, overload

from .targets._registry import NoTargetError
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
    """A kernel call has no device: no qudit handle, no pin, no process target."""


class DeviceMismatchError(RuntimeError):
    """A kernel call mixes handles of different devices, or of an unpinned one."""


class Target(enum.Enum):
    """Targets shipped with cudaq-pulse, for ``get_device``."""

    TRANSMON_KRINNER_17Q = "transmon_krinner_17q"
    RYDBERG_CHAIN_8 = "rydberg_chain:8"
    RYDBERG_SQUARE_3X3 = "rydberg_square:3x3"


_CARRIER = "qudit"
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
        carriers = sorted(target.qubits)
        layout = tuple(carriers if layout is None else (int(i) for i in layout))
        unknown = [i for i in layout if i not in target.qubits]
        if unknown:
            raise ValueError(f"layout names carriers {unknown} that target {name!r} does not have")
        if len(set(layout)) != len(layout):
            raise ValueError("layout maps two qudits to the same carrier")
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

    target = make_target(spec)
    if calibration is not None:
        target = calibration.apply(target)
    elif spec == "transmon_krinner_17q":
        from .targets.calibration import shipped_calibration

        calibration = shipped_calibration("krinner17")
    return Device(target, name=spec, layout=layout, source=source, calibration=calibration)


def _calibration(config: Mapping[str, Any]) -> Any:
    """The config's calibration: a Calibration, or a shipped name or a path,
    for the config's ``cooldown`` (the file's default if absent)."""
    from .targets.calibration import Calibration

    given = config.get("calibration")
    cooldown = config.get("cooldown")
    if given is None:
        if cooldown is not None:
            raise ValueError("device config gives a 'cooldown' but no 'calibration'")
        return None
    if isinstance(given, Calibration):
        if cooldown is not None and cooldown != given.cooldown:
            raise ValueError(
                f"device config asks for cooldown {cooldown!r} of a {given.cooldown!r} calibration"
            )
        return given
    return Calibration.load(given, cooldown=cooldown)


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
    """
    source = "config"
    if isinstance(config, (str, os.PathLike)):
        source = f"file {os.fspath(config)}"
        with open(config, encoding="utf-8") as f:
            config = json.load(f)
    if not isinstance(config, Mapping):
        raise TypeError(f"device config must be a mapping, got {type(config).__name__}")
    reserved = sorted(set(config) & _RESERVED_CONFIG)
    if reserved:
        raise NotImplementedError(
            f"device config keys {', '.join(reserved)} are reserved: zones, sites and "
            "moves of carriers are not modeled yet"
        )
    unknown = sorted(set(config) - {"target", "layout", "calibration", "cooldown"})
    if unknown:
        raise ValueError(f"unknown device config keys: {', '.join(unknown)}")
    spec = config.get("target")
    calibration = _calibration(config)
    if isinstance(spec, TargetDescription):
        target = spec if calibration is None else calibration.apply(spec)
        return Device(
            target,
            name=spec.name,
            layout=config.get("layout"),
            source=source,
            calibration=calibration,
        )
    if not isinstance(spec, str):
        raise ValueError(
            "device config needs 'target': a registered target name or a TargetDescription"
        )
    return _from_spec(spec, source, config.get("layout"), calibration)


def get_device(target: Target | None = None) -> Device:
    """A device of a shipped target, or the process target with no argument."""
    if target is None:
        from .targets._registry import current_target

        return current_target()
    if not isinstance(target, Target):
        raise TypeError(f"get_device() expects a cudaq_pulse.Target, got {type(target).__name__}")
    return _from_spec(target.value, f"shipped {target.name}")
