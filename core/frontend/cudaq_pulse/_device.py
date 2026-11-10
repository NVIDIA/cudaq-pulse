# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Quantum systems and qudit handles.

A ``QuantumSystem`` is a read-only description of one machine: its calibrated
physics and its layout, which maps the user's qudit numbering to its
carriers. ``qs.qudits[i]`` is a ``Qudit``: a handle that
names one carrier of that quantum system. A kernel called with handles runs on their
quantum system, so one compiled kernel serves every quantum system it is called with.

Quantum systems compare by content: two quantum systems built from the same values are equal
and share compiled kernels. The content is the physics, layout and
calibration (a shipped one by name, cooldown and file, any other by its
values); ``name`` and ``source`` are labels, and equal quantum systems may differ
in them.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, overload

from ._errors import (
    IndexOutOfRangeError,
    InvalidTypeError,
    InvalidValueError,
    NotFoundError,
    Problem,
    PulseError,
    QuantumSystemDefinitionError,
    _closest,
    _count,
    _ranges,
)
from .quantum_systems._registry import (
    NoProcessQuantumSystemError,
    UnknownQuantumSystemError,
    _registry,
    make_quantum_system,
)
from .quantum_systems.base import _TRANSMON, _Description
from .quantum_systems.models import SimulationModel

try:
    # hashlib's own blake2b, without hashlib's import of OpenSSL (a millisecond).
    from _blake2 import blake2b as _blake2b
except ImportError:  # pragma: no cover - CPython builds without the _blake2 module
    from hashlib import blake2b as _blake2b

if TYPE_CHECKING:
    from array import array

    from ._system import ControlSystem
    from .gates import Gate
    from .quantum_systems.base import Coupling, DragPulse, PhysicalQudit, ReadoutPulse, ResidualZZ
    from .quantum_systems.calibration import Calibration

    type _MaybeSystem = QuantumSystem | None

__all__ = [
    "QuantumSystem",
    "QuantumSystemMismatchError",
    "NoQuantumSystemError",
    "Qudit",
    "Register",
    "quantum_system",
]


class NoQuantumSystemError(NoProcessQuantumSystemError):
    """A kernel was called without qudit handles and was not compiled for a
    quantum system, so it has no quantum system to run on. The process quantum system is never used
    for a kernel call."""


class QuantumSystemMismatchError(PulseError, RuntimeError):
    """A kernel call mixes handles of different quantum systems, or of an unpinned one."""


_CARRIER = "qudit"
_CONFIG_KEYS = (
    "format",
    "written_by",
    "name",
    "qudits",
    "couplings",
    "residual_zz",
    "model",
    "attribution",
    "layout",
    "calibration",
    "cooldown",
    "control_system",
    "clock_ghz",
)
# The keys that build a quantum system rather than name a registered one.
_BUILT_KEYS = ("qudits", "couplings", "residual_zz", "model", "attribution")
# Reserved for a model of where carriers are and how they move.
_RESERVED_CONFIG = frozenset({"zones", "sites", "moves", "transport"})


def _read_only(array: Any) -> Any:
    array.setflags(write=False)
    return array


def _fingerprint(target: _Description, layout: tuple[int, ...], calibration: Any) -> str:
    """Content hash of everything a compiled kernel depends on, and of the
    calibration the quantum system holds: quantum systems of one fingerprint are one quantum
    system."""
    qudits = [
        (
            q.index,
            q.frequency_hz,
            q.anharmonicity_hz,
            q.t1_us,
            q.t2_star_us,
            q.t2_echo_us,
            q.thermal_population,
            None if (drive := q.drive) is None else (type(drive).__name__, drive._values),
            None if (readout := q.readout) is None else (type(readout).__name__, readout._values),
            q.calibrations._content,  # type: ignore[attr-defined]
        )
        for q in sorted(target.qudits.values(), key=lambda q: q.index)
    ]
    couplings = [
        (c.qudits, c.strength_hz, c.calibrations._content)  # type: ignore[attr-defined]
        for c in target.couplings
    ]
    residual_zz = [(x.qudits, x.rate_hz) for x in target.residual_zz]
    content: tuple[Any, ...] = (
        target.model,
        target.clock_ghz,
        qudits,
        couplings,
        residual_zz,
        layout,
    )
    # Uncalibrated quantum systems keep their fingerprints.
    if calibration is not None:
        content += (calibration._identity,)
    return _blake2b(repr(content).encode(), digest_size=16).hexdigest()


class Qudit:
    """A handle naming one carrier of a quantum system.

    ``index`` is the carrier's index in the target (the layout is applied
    when the handle is created); ``logical`` is the position the user
    indexed. A handle is identity, not location.
    """

    __slots__ = ("quantum_system", "carrier", "index", "logical", "_key")

    quantum_system: QuantumSystem
    carrier: str
    index: int
    logical: int

    def __init__(self, quantum_system: QuantumSystem, index: int, logical: int) -> None:
        self.quantum_system = quantum_system
        self.carrier = _CARRIER
        self.index = index
        self.logical = logical
        self._key = (quantum_system.fingerprint, _CARRIER, index)

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
        return (Qudit, (self.quantum_system, self.index, self.logical))

    def __repr__(self) -> str:
        return (
            f"<Qudit {self.logical} -> {self.carrier} {self.index} of {self.quantum_system.name}>"
        )

    if TYPE_CHECKING:

        @overload
        def get_calibration(
            self,
            gate: Literal[Gate.X, Gate.X90, Gate.Y, Gate.Y90],
            transition: tuple[int, int] = (0, 1),
        ) -> DragPulse: ...
        @overload
        def get_calibration(
            self, gate: Literal[Gate.READOUT], transition: tuple[int, int] = (0, 1)
        ) -> ReadoutPulse: ...
        @overload
        def get_calibration(
            self, gate: Gate, transition: tuple[int, int] = (0, 1)
        ) -> DragPulse | ReadoutPulse: ...

    def get_calibration(self, gate: Any, transition: tuple[int, int] = (0, 1)) -> Any:
        """The calibrated pulse of *gate* on this qudit of its quantum system, as
        ``get_drive_line(q).get_calibration(gate)`` (``X`` to ``Y90``) and
        ``get_readout_line(q).get_calibration(pulse.gates.READOUT)`` give it
        in a kernel; see ``PhysicalQudit.get_calibration``."""
        return self.quantum_system._description.qudits[self.index].get_calibration(gate, transition)


class Register(Sequence[Qudit]):
    """An ordered selection of qudit handles of one quantum system."""

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
    """``qs.qudits``: index with an int, a list of ints or a slice."""

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
    """All the qudits of a quantum system, which its errors name."""

    __slots__ = ()

    def _owner(self) -> str:
        return (
            f"quantum system {self._qudits[0].quantum_system.name!r}"
            if self._qudits
            else "the quantum system"
        )


def _out_of_range(index: Any, n: int, owner: str) -> IndexOutOfRangeError:
    """The error for *index*, which names a qudit past the *n* of *owner*."""
    wrong = index
    if isinstance(index, Sequence) and not isinstance(index, str):
        wrong = next((i for i in index if isinstance(i, int) and not -n <= i < n), index)
    if n == 0:
        return IndexOutOfRangeError(f"qudit index {wrong!r} is out of range: {owner} has no qudits")
    return IndexOutOfRangeError(
        f"qudit index {wrong!r} is out of range: {owner} has {_count(n, 'qudit')}, "
        f"indexed 0 to {n - 1} (or -{n} to -1)"
    )


class QuantumSystem:
    """A read-only machine description; see the module docstring.

    ``clock_ghz`` is the program clock: kernels count time in its cycles.
    ``pulse.quantum_system(..., clock_ghz=)`` (a system file's ``"clock_ghz"``) sets
    it; else the fastest clock of the control
    system's controllers, when every other one divides it (controllers whose
    clocks do not divide evenly need it declared); else the target's. Each
    controller may run its own clock, a whole-cycle ratio of it (see
    ``cudaq_pulse.controller``).

    ``qudit_freq[q]`` is the 0-1 transition frequency of qudit ``q`` (Hz) and
    ``transition_freq[q, i, j]`` that of levels ``i`` and ``j``, over the
    levels the target's model gives each qudit (NaN past a qudit's last
    level); both are indexed by the user's qudit numbering and are read-only
    arrays.
    ``control_system`` is the ``ControlSystem`` that plays it, if given;
    ``compile(kernel, quantum_system=dev)`` checks every program against it.
    """

    # The registered target the quantum system was made from, as ``as_dict`` writes it.
    _spec: str | None = None
    # What a kernel whose qudits are runtime parameters depends on, made by
    # the first such kernel (_programs._structure).
    _structure: str | None = None

    def __init__(
        self,
        target: _Description,
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
                    f"QuantumSystem control_system must be a ControlSystem, got {control_system!r}"
                )
        problems: list[Problem] = []
        if clock_ghz is not None or (control_system is not None and control_system._needs_clock):
            from ._clocks import _program_clock

            clock, problems = _program_clock(control_system, clock_ghz, target.clock_ghz)
            if clock is None:
                raise QuantumSystemDefinitionError(problems=problems)
            if clock != target.clock_ghz:
                target = dataclasses.replace(target, clock_ghz=clock)
        layout = (
            tuple(sorted(target.qudits))
            if layout is None
            else _layout(layout, target, name, problems)
        )
        if control_system is not None:
            from ._system import _wiring_problems

            problems.extend(_wiring_problems(control_system, target))
        if layout is None or problems:
            raise QuantumSystemDefinitionError(problems=problems)
        self.control_system = control_system
        # The target before its calibration was applied, as ``as_dict`` writes it.
        self._base_target = target
        self._description = target
        self.name = name
        self.source = source
        # The Calibration the target's fitted values came from, if any.
        self.calibration = calibration
        self.layout = layout
        self.clock_ghz = float(target.clock_ghz)
        self.frequencies: Mapping[int, float] = target.frequencies
        self.fingerprint = _fingerprint(target, layout, calibration)
        # What equality compares: the fingerprint alone without a control
        # system, so comparing such quantum systems costs one string comparison.
        self._identity: Any = (
            self.fingerprint if control_system is None else (self.fingerprint, control_system)
        )
        self.qudits = _Handles(tuple(Qudit(self, p, i) for i, p in enumerate(layout)))
        self._arrays: tuple[Any, Any] | None = None
        # The frequency tables a compiled module records, as
        # _frequency_tables returns them: (count, 0-1 frequencies, readout
        # tones); none before the first compile.
        self._tables: tuple[int, Any, Any] = (-1, None, None)

    def _frequency_arrays(self) -> tuple[Any, Any]:
        if self._arrays is None:
            import numpy as np

            target = self._description
            model = target.model if isinstance(target.model, SimulationModel) else SimulationModel()
            qudits = [target.qudits[p] for p in self.layout]
            levels = [model.levels_of(p) for p in self.layout]
            f01 = np.array([q.frequency_hz for q in qudits], dtype=float)
            energies = np.full((len(qudits), max(levels, default=model.levels)), np.nan)
            for row, qudit, n in zip(energies, qudits, levels, strict=False):
                row[:n] = model.energies_hz(qudit, n)
            transitions = np.abs(energies[:, None, :] - energies[:, :, None])
            self._arrays = (_read_only(f01), _read_only(transitions))
        return self._arrays

    def _line_tones(
        self, index: int, readout: bool
    ) -> tuple[tuple[int, int], dict[tuple[int, int], float]]:
        """The default transition of carrier *index*, and the frequency of
        each transition its line has a tone for: every transition of a
        drive line, only the default of a readout line."""
        target = self._description
        model = target.model if isinstance(target.model, SimulationModel) else SimulationModel()
        default = model.default_transition
        if index < 0:
            # A runtime qudit: its own tone is bound per call, and no other.
            return default, {}
        qudit = target.qudits[index]
        if readout:
            return default, {default: self._frequency_tables(index + 1)[2][index]}
        return default, model.transitions_hz(qudit, model.levels_of(index))

    def _frequency_tables(self, count: int) -> tuple[int, array[float], array[float]]:
        """The 0-1 frequency of carriers 0 to *count* - 1 (or more), and the
        tone of each one's readout line (its readout's ``frequency_hz``, else
        its 0-1 frequency), by carrier index, as a compiled module records
        them, after how many carriers from 0 on the target defines: *count*
        or more, else the first it lacks (0 Hz in the tables, which then end
        at its last carrier). Kept (``_tables``) until a caller needs more
        carriers: a compile makes them for its program's qudits only, not
        for the whole quantum system."""
        if self._tables[0] >= count:
            return self._tables
        from array import array

        frequencies, qudits = self.frequencies, self._description.qudits
        carriers: Sequence[int] = range(count)
        try:
            table = array("d", map(frequencies.__getitem__, carriers))
        except KeyError:
            size = min(count, max(frequencies, default=-1) + 1)
            made = self._tables[1]
            if made is not None and len(made) >= size:
                return self._tables
            carriers = [carrier for carrier in sorted(frequencies) if carrier < size]
            table = array("d", bytes(8 * size))
            for carrier in carriers:
                table[carrier] = frequencies[carrier]
            # The first carrier the target lacks.
            count = next((i for i, carrier in enumerate(carriers) if i != carrier), len(carriers))
        for carrier in carriers:
            if qudits[carrier].readout is not None:
                break
        else:
            # Without a readout, a readout line plays at the 0-1 frequency.
            self._tables = (count, table, table)
            return self._tables
        from operator import attrgetter

        components = list(map(attrgetter("readout"), map(qudits.__getitem__, carriers)))
        try:
            given = list(map(attrgetter("frequency_hz"), components))
        except AttributeError:
            given = [None if r is None else r.frequency_hz for r in components]
        if len(given) == len(table) and None not in given:
            readouts = array("d", given)
        else:
            readouts = array("d", table)
            for carrier, frequency in zip(carriers, given, strict=True):
                if frequency is not None:
                    readouts[carrier] = frequency
        self._tables = (count, table, readouts)
        return self._tables

    @property
    def qudit_freq(self) -> Any:
        return self._frequency_arrays()[0]

    @property
    def transition_freq(self) -> Any:
        return self._frequency_arrays()[1]

    def qudit(self, label: str) -> Qudit:
        """The handle of the qudit labelled *label* (``PhysicalQudit.label``), as in
        ``dev.qudit("D1")``; a ``NotFoundError`` names the labels there are."""
        qudits = self._description.qudits
        for handle in self.qudits:
            if qudits[handle.index].label == label:
                return handle
        labels = [qudits[p].label for p in self.layout if qudits[p].label]
        carrier = next((q.index for q in qudits.values() if q.label == label), None)
        if carrier is not None:
            raise NotFoundError(
                f"qudit {label!r} is carrier {carrier} of {self._description.name}, which the "
                f"layout of quantum system {self.name!r} leaves out"
            )
        close = _closest(str(label), labels)
        hint = f"did you mean {close!r}? " if close else ""
        known = f"its labels are {', '.join(labels)}" if labels else "its qudits have no labels"
        raise NotFoundError(
            f"quantum system {self.name!r} has no qudit labelled {label!r}; {hint}{known}"
        )

    def coupling(self, a: Qudit, b: Qudit) -> Coupling:
        """The coupling of the quantum system's qudits *a* and *b* (handles, in either
        order), as calibrated: ``dev.coupling(a, b).get_calibration(
        pulse.gates.CZ)`` is its CZ. A ``NotFoundError`` names the qudits *a*
        couples to."""
        for handle in (a, b):
            if handle.__class__ is not Qudit:
                raise InvalidTypeError(
                    f"QuantumSystem.coupling takes two qudit handles, as qs.qudits[i]; got "
                    f"{handle!r}"
                )
            if handle.quantum_system is not self and handle.quantum_system != self:
                raise QuantumSystemMismatchError(
                    f"{handle!r} is a qudit of {handle.quantum_system.name!r}, not of {self.name!r}"
                )
        return self._description.coupling(a.index, b.index)

    def _line_calibration(
        self, index: int, readout: bool, gate: Any, transition: Any
    ) -> DragPulse | ReadoutPulse:
        """``line.get_calibration(gate, transition)`` of the drive (or
        readout) line of carrier *index*, while a kernel is traced."""
        from .quantum_systems.base import _DRIVE_GATES, _gate_label, _gate_name, _qudit_calibration

        name = _gate_name(gate)
        if (name == "readout") is not readout:
            line, other = ("readout", "drive") if readout else ("drive", "readout")
            if name == "cz":
                fix = "it is a coupling's: dev.coupling(a, b).get_calibration(pulse.gates.CZ)"
            else:
                fix = (
                    f"it is the {other} line's: "
                    f"get_{other}_line(q).get_calibration({_gate_label(name)})"
                )
            gates = "pulse.gates.READOUT" if readout else ", ".join(map(_gate_label, _DRIVE_GATES))
            raise InvalidValueError(
                f"a {line} line has the calibrations of {gates}, not {_gate_label(name)}; {fix}"
            )
        pulse: DragPulse | ReadoutPulse = _qudit_calibration(
            self._description.qudits[index], name, transition
        )
        return pulse

    @staticmethod
    def load(path: str | os.PathLike[str], /) -> QuantumSystem:
        """The quantum system of a system file, as ``as_dict`` writes one: JSON with
        its format (``cudaq-pulse.system/1``), its quantum system (a
        registered name, or every value inline), layout, calibration,
        cooldown, control system and program clock, each as data. A file
        with problems raises ``QuantumSystemDefinitionError`` listing all of them,
        each with its path in the file, its line and its column; ``python -m
        cudaq_pulse.doctor --system FILE`` runs the same checks."""
        return _from_file(path)

    @staticmethod
    def from_dict(data: Mapping[str, Any], /) -> QuantumSystem:
        """The quantum system of a system file's data, as ``as_dict`` returns it or
        ``json.load`` reads it; see ``load``."""
        return _from_data(data)

    if TYPE_CHECKING:
        # Read from the quantum system's description by __getattr__.
        @property
        def physical_qudits(self) -> Mapping[int, PhysicalQudit]: ...
        @property
        def couplings(self) -> Sequence[Coupling]: ...
        @property
        def residual_zz(self) -> Sequence[ResidualZZ]: ...
        @property
        def model(self) -> SimulationModel | type[SimulationModel] | None: ...
        @property
        def attribution(self) -> str: ...
        @property
        def anharmonicities(self) -> dict[int, float]: ...
        @property
        def t1_times(self) -> dict[int, float]: ...
        @property
        def t2_times(self) -> dict[int, float]: ...
        @property
        def t2_echo_times(self) -> dict[int, float | None]: ...
        @property
        def thermal_populations(self) -> dict[int, float]: ...
        @property
        def coupling_map(self) -> list[tuple[int, ...]]: ...
        def connectivity_graph(self) -> dict[int, list[int]]: ...
        def drive_amplitude_scale(self, qudit_index: int) -> float: ...

    def __getattr__(self, name: str) -> Any:
        """The description's fields, properties and methods, read from the
        description the quantum system was built from when asked for:
        ``physical_qudits`` (its ``qudits``; ``qudits`` are the handles),
        ``couplings``, ``residual_zz``, ``model``, ``attribution``,
        ``anharmonicities``, ``t1_times``, ``t2_times``, ``t2_echo_times``,
        ``thermal_populations``, ``coupling_map``, ``connectivity_graph()`` and
        ``drive_amplitude_scale(i)``."""
        if name == "physical_qudits":
            return self.__dict__["_description"].qudits
        if name in {
            "couplings",
            "residual_zz",
            "model",
            "attribution",
            "anharmonicities",
            "t1_times",
            "t2_times",
            "t2_echo_times",
            "thermal_populations",
            "coupling_map",
            "connectivity_graph",
            "drive_amplitude_scale",
        }:
            return getattr(self.__dict__["_description"], name)
        raise AttributeError(
            f"'QuantumSystem' object has no attribute {name!r}", name=name, obj=self
        )

    def as_dict(self) -> dict[str, Any]:
        """The quantum system as JSON-serializable data, which ``QuantumSystem.from_dict``
        reads back: its format (``cudaq-pulse.system/1``), the version that
        wrote it, its quantum system (a registered name, or every value
        inline), layout and calibration, and its control system."""
        from ._system import _device_data

        return _device_data(self)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, QuantumSystem) and other._identity == self._identity

    def __hash__(self) -> int:
        return hash(self.fingerprint)

    # Read-only: a copy is the quantum system itself.
    def __copy__(self) -> QuantumSystem:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> QuantumSystem:
        return self

    def __reduce__(self) -> tuple[Any, ...]:
        from ._pickling import device_reduce

        return device_reduce(self)

    def __repr__(self) -> str:
        return f"<QuantumSystem {self.name} ({len(self.layout)} qudits)>"


def _from_spec(
    spec: str,
    source: str,
    layout: Sequence[int] | None = None,
    calibration: Any = None,
    control_system: ControlSystem | None = None,
) -> QuantumSystem:
    return _calibrated(make_quantum_system(spec), spec, source, layout, calibration, control_system)


def _calibrated(
    target: _Description,
    spec: str | None,
    source: str,
    layout: Sequence[int] | None,
    calibration: Any,
    control_system: ControlSystem | None = None,
    clock_ghz: float | None = None,
) -> QuantumSystem:
    """The quantum system of *target*, registered as *spec* (``None`` for a
    ``_Description``), with *calibration* applied and *clock_ghz* (if
    given) its program clock."""
    base = target
    if calibration is not None:
        target = calibration._apply(target)
    elif spec == "transmon_17q":
        from .quantum_systems.calibration import shipped_calibration

        calibration = shipped_calibration("transmon_17q")
    name = target.name if spec is None else spec
    made = QuantumSystem(
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


def _target(spec: Any, problems: list[Problem], key: str = "name") -> _Description | None:
    """The description of the registered quantum system *spec* names."""
    from .quantum_systems._registry import available_quantum_systems, make_quantum_system

    if spec is None:
        problems.append(
            Problem(
                "is missing",
                path=key,
                fix="give a registered name ("
                + ", ".join(available_quantum_systems())
                + '), or "qudits" to build one',
            )
        )
        return None
    if not isinstance(spec, str):
        problems.append(
            Problem(
                f"must be a registered name, got {type(spec).__name__}",
                path=key,
                value=spec,
            )
        )
        return None
    try:
        return make_quantum_system(spec)
    except (UnknownQuantumSystemError, ValueError) as error:
        problems.append(Problem(Exception.__str__(error), path=key, value=spec))
        return None


def _layout(
    layout: Any, target: _Description | None, name: str, problems: list[Problem]
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
    # The last index after the loop; it stays 0 for an empty layout, which
    # then fails the check that every entry named a new carrier.
    k = 0
    carriers: list[int] = []
    first: dict[int, int] = {}
    known = target.qudits if target is not None else None
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
                    f"carrier {carrier} is not in quantum system {name!r} (it has "
                    f"{_ranges(known)})",
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
    return tuple(carriers) if len(first) > k else _empty(layout, problems)


def _empty(layout: Sequence[Any], problems: list[Problem]) -> tuple[int, ...] | None:
    """No layout: the problem of an empty one, unless its entries had others."""
    if len(layout) == 0:
        problems.append(
            Problem(
                "must name at least one carrier, got an empty list",
                path="layout",
                value=layout,
                fix="leave layout out for every carrier",
            )
        )
    return None


def _calibration(config: Mapping[str, Any], problems: list[Problem]) -> Any:
    """The config's calibration: a Calibration, or a shipped name or a path,
    for the config's ``cooldown`` (the file's default if absent)."""
    from .quantum_systems.calibration import Calibration

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
    except ValueError as error:
        import json

        if isinstance(error, json.JSONDecodeError):
            problems.append(
                Problem(f"the calibration file is not valid JSON: {error}", path="calibration")
            )
        else:
            problems.append(Problem(str(error), path="cooldown", value=cooldown))
    return None


def _definition(config: Mapping[str, Any], source: str, problems: list[Problem]) -> _MaybeSystem:
    """The quantum system of *config*, or ``None`` after adding its problems in the
    order of the config's keys. ``pulse.quantum_system`` gives the system it built
    from its keywords as the config's ``"name"``."""
    found = len(problems)
    quantum_system = _checked(config, source, problems)
    order = {str(key): i for i, key in enumerate(config)}
    problems[found:] = sorted(
        problems[found:], key=lambda p: order.get(p.path.split("[")[0].split(".")[0], -1)
    )
    return quantum_system


def _checked(config: Mapping[str, Any], source: str, problems: list[Problem]) -> _MaybeSystem:
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
                "is not a key of a system file",
                path=str(key),
                value=key,
                fix=f"did you mean {close!r}?"
                if close
                else "the keys are " + ", ".join(_CONFIG_KEYS),
            )
        )
    spec = config.get("name")
    described = spec if spec.__class__ is _Description else None
    built = described is not None or "qudits" in config
    if not built and config.keys() & _BUILT_KEYS:
        for key in sorted(config.keys() & _BUILT_KEYS):
            problems.append(
                Problem(
                    'builds a quantum system, with "qudits"; a registered name gives its own',
                    path=key,
                    fix='give "qudits" to build one, or remove ' + repr(key),
                )
            )
    if built:
        spec = None
    control = config.get("control_system")
    # A quantum system by name or built from keywords, without a format or a
    # control system, needs none of the system file machinery.
    if described is not None and control is None:
        target: _Description | None = described
    elif "format" in config or built or control is not None:
        from ._system import _config_systems

        target, control = _config_systems(config, source, problems, described)
    else:
        target = _target(spec, problems)
    name = spec if isinstance(spec, str) else getattr(target, "name", "")
    layout = config.get("layout")
    if layout is not None:
        layout = _layout(layout, target, name, problems)
    calibration = _calibration(config, problems)
    clock_ghz = config.get("clock_ghz")
    if clock_ghz is not None or (control is not None and control._needs_clock):
        from ._clocks import _program_clock

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
            Problem(
                f"does not fit quantum system {name!r}: {Exception.__str__(error)}",
                path="calibration",
            )
        )
        return None


def _json_positions(text: str) -> dict[str, tuple[int, int]]:
    """The line and column of each value of the JSON *text*, by its path."""
    import json

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


def quantum_system(
    name: str | None = None,
    *,
    qudits: Mapping[int, PhysicalQudit] | None = None,
    couplings: Sequence[Coupling] = (),
    residual_zz: Sequence[ResidualZZ] = (),
    model: SimulationModel | type[SimulationModel] | None = _TRANSMON,
    attribution: str = "",
    layout: Sequence[int] | None = None,
    calibration: Calibration | str | os.PathLike[str] | None = None,
    cooldown: str | None = None,
    control_system: ControlSystem | None = None,
    clock_ghz: float | None = None,
) -> QuantumSystem:
    """A quantum system: a registered one by name, or one built from its qudits.

    ``quantum_system("transmon_17q")`` is a registered quantum system, with
    every carrier in order (a name takes its argument after a colon, as in
    ``"transmon_generic:4"``); ``cudaq_pulse.quantum_systems.available_quantum_systems()``
    lists the names. A string is always a name: ``QuantumSystem.load(path)`` reads
    a system file, and ``QuantumSystem.from_dict(data)`` its data. ``quantum_system()`` is
    the process quantum system, named by ``CUDAQ_PULSE_QUANTUM_SYSTEM`` (``python -m
    cudaq_pulse --quantum-system NAME`` sets it); without one it raises
    ``NoProcessQuantumSystemError``.

    ``quantum_system(name="chip", qudits={0: PhysicalQudit(...), ...})`` builds
    one: ``qudits`` are its ``PhysicalQudit`` carriers by index, ``couplings``
    and ``residual_zz`` its ``Coupling`` and ``ResidualZZ`` entries, ``model``
    how the simulator treats its qudits and how many levels each has (a
    ``SimulationModel`` class or instance, ``TransmonModel`` by default,
    ``TransmonModel(levels={0: 4})`` to give qudit 0 four levels, or ``None``
    for a system with no simulation model) and ``attribution`` where its
    values come from. Its couplings and residual ZZ entries are checked, and
    a problem is at its path (``couplings[3].qudits[1]``). These keywords
    build a system, so a registered name takes none of them.

    ``layout[i]`` is the carrier that qudit ``i`` names; by default every
    carrier of the quantum system, in order. ``calibration`` is a
    ``Calibration``, the name of a shipped one or the path of a calibration
    file, and ``cooldown`` one of its cooldowns; the calibration's couplings
    replace the system's, and ``QuantumSystem.calibration`` holds it. The
    ``transmon_17q`` quantum system uses the shipped ``transmon_17q`` calibration by
    default. ``control_system`` is a ``ControlSystem``, whose lines must be
    carriers of the quantum system. ``clock_ghz`` is the program clock
    kernels count cycles of (``QuantumSystem.clock_ghz``); by default the fastest
    controller clock when the others divide it, else the quantum system's
    (2 GHz for one built here).
    Controller clocks that do not divide evenly into one need it, and the
    problem lists the clocks that work.

    Arguments with problems raise ``QuantumSystemDefinitionError`` listing all of
    them, each at the argument it is in.
    """
    described: _Description | None = None
    if qudits is not None:
        if name.__class__ is not str:
            raise InvalidTypeError(
                "pulse.quantum_system(qudits=...) builds a quantum system and needs its name=, "
                f"got {type(name).__name__}"
            )
        described = _Description(
            name=name,
            qudits=qudits,
            couplings=couplings,
            residual_zz=residual_zz,
            model=model,
            attribution=attribution,
            # A clock that is no positive number is the program clock's problem.
            **(
                {"clock_ghz": clock_ghz}
                if isinstance(clock_ghz, (int, float))
                and not isinstance(clock_ghz, bool)
                and 0 < clock_ghz < float("inf")
                else {}
            ),
        )
    elif name.__class__ is str and name.partition(":")[0] not in _registry():
        _unknown_target(name)
    elif couplings or residual_zz or attribution or model is not _TRANSMON:
        given = "couplings" if couplings else "residual_zz" if residual_zz else "attribution"
        raise InvalidTypeError(
            f"pulse.quantum_system's {'model' if model is not _TRANSMON else given} builds a "
            "quantum system with qudits=; a registered name gives its own"
        )
    elif name.__class__ is str:
        if (
            layout is None
            and calibration is None
            and cooldown is None
            and control_system is None
            and clock_ghz is None
        ):
            return _from_spec(name, f"quantum system {name}")
    elif name is None:
        if (
            layout is None
            and calibration is None
            and cooldown is None
            and control_system is None
            and clock_ghz is None
        ):
            from .quantum_systems._registry import current_quantum_system

            return current_quantum_system()
        from .quantum_systems._registry import ENV_VAR, _no_quantum_system_message

        name = os.environ.get(ENV_VAR)
        if not name:
            raise NoProcessQuantumSystemError(_no_quantum_system_message())
        if name.partition(":")[0] not in _registry():
            _unknown_target(name)
    elif isinstance(name, str):
        if name.partition(":")[0] not in _registry():
            _unknown_target(name)
    else:
        fix = (
            "; QuantumSystem.from_dict(data) reads a system file's data"
            if isinstance(name, Mapping)
            else ""
        )
        raise InvalidTypeError(
            "pulse.quantum_system takes a registered name, or name= and qudits= to build "
            f"one, got {type(name).__name__}{fix}"
        )
    if isinstance(control_system, Mapping) or isinstance(calibration, Mapping):
        given = "control_system" if isinstance(control_system, Mapping) else "calibration"
        raise InvalidTypeError(
            f"pulse.quantum_system's {given} is a "
            + ("ControlSystem" if given == "control_system" else "Calibration, a name or a path")
            + ", not data; QuantumSystem.from_dict(data) reads a system file's data"
        )
    arguments = {
        "name": name if described is None else described,
        "layout": layout,
        "calibration": calibration,
        "cooldown": cooldown,
        "control_system": control_system,
        "clock_ghz": clock_ghz,
    }
    problems: list[Problem] = []
    found = _definition(
        {k: v for k, v in arguments.items() if v is not None}, "arguments", problems
    )
    if found is None:
        raise QuantumSystemDefinitionError(
            f"{_count(len(problems), 'problem')} in pulse.quantum_system()", problems=problems
        )
    return found


def _unknown_target(name: str) -> None:
    """Raise ``UnknownQuantumSystemError`` for *name*, which names no registered
    target, naming the way to read a file when it looks like the path of one."""
    if os.sep in name or name.endswith(".json") or os.path.exists(name):
        try:
            make_quantum_system(name)
        except UnknownQuantumSystemError as error:
            raise UnknownQuantumSystemError(
                f"{Exception.__str__(error)}; a string is a target name: to read a system "
                f"file, use pulse.QuantumSystem.load({name!r})"
            ) from None
    make_quantum_system(name)


def _from_data(data: Any) -> QuantumSystem:
    """``QuantumSystem.from_dict``: the quantum system of a system file's data."""
    if not isinstance(data, Mapping):
        raise InvalidTypeError(f"quantum system data must be a mapping, got {type(data).__name__}")
    problems: list[Problem] = []
    found = _definition(data, "data", problems)
    if found is None:
        raise QuantumSystemDefinitionError(
            f"{_count(len(problems), 'problem')} in the quantum system data", problems=problems
        )
    return found


def _from_file(path: str | os.PathLike[str]) -> QuantumSystem:
    """``QuantumSystem.load``: the quantum system of a system file."""
    import json

    file = os.fspath(path)
    with open(file, encoding="utf-8") as f:
        text = f.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        problem = Problem(
            f"not valid JSON: {error.msg}", file=file, line=error.lineno, column=error.colno
        )
        raise QuantumSystemDefinitionError(
            f"1 problem in quantum system file {file}", problems=[problem]
        ) from error
    problems: list[Problem] = []
    if isinstance(data, Mapping):
        found = _definition(data, f"file {file}", problems)
    else:
        found = None
        problems.append(
            Problem(f"the file must hold a JSON object, got {type(data).__name__}", value=data)
        )
    if found is None:
        raise QuantumSystemDefinitionError(
            f"{_count(len(problems), 'problem')} in quantum system file {file}",
            problems=_located(problems, file, text),
        )
    return found
