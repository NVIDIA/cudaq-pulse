# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Architecture-neutral target API for QPU Hamiltonians and decoherence models.

A ``TargetDescription`` fully describes a quantum device: qubit frequencies,
anharmonicities, coupling topology, decoherence parameters, and readout.
It can generate Hamiltonian and Lindblad dissipator terms for the Pulse ->
QOp lowering.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import FrozenInstanceError, dataclass, fields
from itertools import chain
from operator import attrgetter, eq, is_
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from .._errors import (
    DeviceDefinitionError,
    InvalidTypeError,
    InvalidValueError,
    NotFoundError,
    Problem,
    UnsupportedError,
    _count,
    _ranges,
)
from .models import SimulationModel, TransmonModel

if TYPE_CHECKING:
    from dataclasses import dataclass as _dataclass_when_used
    from typing import ClassVar, Self
else:
    from .._errors import _dataclass_when_used

_TRANSMON = TransmonModel()

_SECONDS_PER_NANOSECOND = 1.0e-9
_NANOSECONDS_PER_MICROSECOND = 1.0e3
# The default of the mapping fields: __post_init__ keeps a read-only copy of
# whatever a field is given, so every instance may share it.
if TYPE_CHECKING:
    _EMPTY: Mapping[Any, Any]
_EMPTY = MappingProxyType({})


class _Field(int):
    """A field of a component, read from its ``_values`` at this index."""

    __slots__ = ()

    def __get__(self, component: Any, owner: Any = None) -> Any:
        if component is None:
            return self
        return component._values[self]


class _Component:
    """A read-only value of named fields (``_fields``, the keywords of its
    ``__init__``), held in ``_values`` and read through ``_Field``
    descriptors, compared, hashed, pickled and printed by them.

    A plain slotted class rather than a dataclass: the shipped target makes
    one per qudit, and making a dataclass of this many fields costs as much
    as the rest of the import.
    """

    __slots__ = ("_values",)
    if TYPE_CHECKING:
        _fields: ClassVar[tuple[str, ...]]
        _values: tuple[Any, ...]
    _fields = ()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Component):
            return NotImplemented
        return type(other) is type(self) and other._values == self._values

    def __hash__(self) -> int:
        return hash((type(self).__name__, self._values))

    def __repr__(self) -> str:
        defaults = type(self).__init__.__kwdefaults__ or {}
        given = ", ".join(
            f"{name}={value!r}"
            for name, value in zip(self._fields, self._values, strict=True)
            if value != defaults.get(name)
        )
        return f"{type(self).__name__}({given})"

    def __reduce__(self) -> tuple[Any, ...]:
        from .._pickling import rebuilt

        return (rebuilt, (type(self), dict(zip(self._fields, self._values, strict=True))))

    def replace(self, **changes: Any) -> Self:
        """A copy with the fields *changes* names set to their values."""
        values = dict(zip(self._fields, self._values, strict=True))
        values.update(changes)
        return type(self)(**values)

    __replace__ = replace


class Readout(_Component):
    """How a qudit is read out: the base of the readout modalities, such as
    ``DispersiveReadout``. Simulation and emulation pick a readout's model by
    its type, and refuse a type they have no model for."""

    __slots__ = ()

    @property
    def frequency_hz(self) -> float | None:
        """The frequency its readout line plays at by default (``None``: the
        qudit's 0-1 frequency)."""
        return None

    def _depletion_ns(self) -> float:
        """How long after a readout the next one must wait."""
        return 0.0


class Drive(_Component):
    """How a qudit is driven: the base of the drive modalities, such as
    ``MicrowaveDrive``. Simulation picks a drive's model by its type, and
    refuses a type it has no model for."""

    __slots__ = ()


class DispersiveReadout(Readout):
    """A dispersive readout through a resonator. Every field may be left out.

    ``frequency_hz`` is the resonator's frequency, where the readout line
    plays by default. The readout model in simulation and emulation
    (``IQModel``) pulls the resonator by ``(2k - 1) dispersive_shift_hz``
    for level ``k``, with linewidth ``linewidth_hz`` (kappa / 2 pi), and
    needs both. ``error_1to0`` and ``error_0to1`` are the probabilities of
    reporting 0 for ``|1>`` and 1 for ``|0>``. The resonator empties in
    ``depletion_ns`` (else ln(100) / kappa) at measurement efficiency
    ``efficiency``.

    The readout pulse the device is tuned for: of peak ``amplitude`` it
    holds ``photon_number`` photons (with no ``amplitude``, any pulse does),
    is integrated over ``integration_ns`` (else the readout's length), has
    edges of width ``sigma_ns``, and, with relaxation during the readout,
    a window of ``duration_ns``.
    """

    __slots__ = ()
    _fields = (
        "frequency_hz",
        "dispersive_shift_hz",
        "linewidth_hz",
        "efficiency",
        "error_1to0",
        "error_0to1",
        "depletion_ns",
        "amplitude",
        "duration_ns",
        "sigma_ns",
        "integration_ns",
        "photon_number",
    )

    def __init__(
        self,
        *,
        frequency_hz: float | None = None,
        dispersive_shift_hz: float | None = None,
        linewidth_hz: float | None = None,
        efficiency: float = 0.5,
        error_1to0: float = 0.0,
        error_0to1: float = 0.0,
        depletion_ns: float | None = None,
        amplitude: float | None = None,
        duration_ns: float | None = None,
        sigma_ns: float | None = None,
        integration_ns: float | None = None,
        photon_number: float = 4.0,
    ) -> None:
        object.__setattr__(
            self,
            "_values",
            (
                frequency_hz,
                dispersive_shift_hz,
                linewidth_hz,
                efficiency,
                error_1to0,
                error_0to1,
                depletion_ns,
                amplitude,
                duration_ns,
                sigma_ns,
                integration_ns,
                photon_number,
            ),
        )

    if TYPE_CHECKING:

        @property
        def frequency_hz(self) -> float | None:
            """The resonator's frequency, where the readout line plays by default."""
            ...

        @property
        def dispersive_shift_hz(self) -> float | None:
            """The dispersive shift chi / 2 pi."""
            ...

        @property
        def linewidth_hz(self) -> float | None:
            """The resonator's effective linewidth kappa / 2 pi."""
            ...

        @property
        def efficiency(self) -> float:
            """The measurement efficiency."""
            ...

        @property
        def error_1to0(self) -> float:
            """The probability that ``|1>`` is reported 0."""
            ...

        @property
        def error_0to1(self) -> float:
            """The probability that ``|0>`` is reported 1."""
            ...

        @property
        def depletion_ns(self) -> float | None:
            """The time the resonator takes to empty after a readout."""
            ...

        @property
        def amplitude(self) -> float | None:
            """The readout pulse's peak amplitude, at which it holds ``photon_number``."""
            ...

        @property
        def duration_ns(self) -> float | None:
            """The readout window, over which the qudit may relax."""
            ...

        @property
        def sigma_ns(self) -> float | None:
            """The width of the readout pulse's edges."""
            ...

        @property
        def integration_ns(self) -> float | None:
            """The time the readout signal is integrated over."""
            ...

        @property
        def photon_number(self) -> float:
            """The photons in the resonator at ``amplitude``."""
            ...

    else:
        # In the order of _fields.
        (
            frequency_hz,
            dispersive_shift_hz,
            linewidth_hz,
            efficiency,
            error_1to0,
            error_0to1,
            depletion_ns,
            amplitude,
            duration_ns,
            sigma_ns,
            integration_ns,
            photon_number,
        ) = map(_Field, range(12))

    def _depletion_ns(self) -> float:
        depletion, linewidth = self._values[6], self._values[2]
        if depletion is not None:
            return float(depletion)
        if linewidth:
            return math.log(100.0) / (2 * math.pi * float(linewidth) * 1e-9)
        return 0.0


class MicrowaveDrive(Drive):
    """A microwave drive. Every field may be left out.

    The simulator turns a pulse's amplitude into a Rabi rate in rad/ns by
    ``amplitude_scale_rad_per_ns``; without it, from the calibrated
    Gaussian or DRAG pi pulse of peak ``x_amplitude``, length
    ``x_duration_ns`` and width ``x_sigma_ns`` (all three given), so that
    the pulse makes a pi rotation; else 1.0.
    """

    __slots__ = ()
    _fields = ("amplitude_scale_rad_per_ns", "x_amplitude", "x_duration_ns", "x_sigma_ns")

    def __init__(
        self,
        *,
        amplitude_scale_rad_per_ns: float | None = None,
        x_amplitude: float | None = None,
        x_duration_ns: float | None = None,
        x_sigma_ns: float | None = None,
    ) -> None:
        object.__setattr__(
            self,
            "_values",
            (
                amplitude_scale_rad_per_ns,
                x_amplitude,
                x_duration_ns,
                x_sigma_ns,
            ),
        )

    if TYPE_CHECKING:

        @property
        def amplitude_scale_rad_per_ns(self) -> float | None:
            """The Rabi rate in rad/ns of a pulse of amplitude 1."""
            ...

        @property
        def x_amplitude(self) -> float | None:
            """The peak amplitude of the calibrated pi pulse."""
            ...

        @property
        def x_duration_ns(self) -> float | None:
            """The length of the calibrated pi pulse."""
            ...

        @property
        def x_sigma_ns(self) -> float | None:
            """The Gaussian width of the calibrated pi pulse."""
            ...

    else:
        # In the order of _fields.
        amplitude_scale_rad_per_ns, x_amplitude, x_duration_ns, x_sigma_ns = map(_Field, range(4))


@dataclass(frozen=True)
class Qubit:
    """Single qubit in a target device.

    ``drive`` and ``readout`` say how it is driven and read out, each by
    the component of its modality (``MicrowaveDrive``,
    ``DispersiveReadout``), or ``None`` when not given.
    """

    index: int
    frequency_hz: float
    anharmonicity_hz: float
    t1_us: float
    t2_star_us: float
    label: str = ""
    drive: Drive | None = None
    readout: Readout | None = None
    # Hahn-echo T2 (us), which the simulation dephases at on request, and
    # the excited-state population at equilibrium (a fraction), from which
    # it starts and towards which T1 relaxes.
    t2_echo_us: float | None = None
    thermal_population: float = 0.0

    def __hash__(self) -> int:
        return hash((self.index, self.frequency_hz, self.label))

    def __reduce__(self) -> tuple[Any, ...]:
        from .._pickling import frozen_reduce

        return frozen_reduce(self)

    @property
    def depletion_ns(self) -> float:
        """How long after a readout the next one must wait: for a
        ``DispersiveReadout``, its ``depletion_ns`` if given, else the time
        for its photons to fall to 1% at its ``linewidth_hz``, ln(100) / kappa;
        else 0.
        """
        readout = self.readout
        return 0.0 if readout is None else readout._depletion_ns()


class Gate:
    """The gate characterized on a coupling (``Coupling.gate``).

    ``name`` names it (``"cz"``), ``duration_ns`` is how long it takes,
    ``buffer_ns`` the idle time around it and ``interaction_frequency_hz``
    the frequency at which its qudits interact, for a flux-tuned CZ; ``None``
    where not given. It acts on its coupling's qudits, in their order.
    Neither compilation nor simulation reads it; ``transmon_17q`` gives the
    published CZ values. A gate is read-only and compares by content.
    """

    __slots__ = ("name", "duration_ns", "buffer_ns", "interaction_frequency_hz")
    __match_args__ = ("name", "duration_ns", "buffer_ns", "interaction_frequency_hz")

    name: str
    duration_ns: float
    buffer_ns: float | None
    interaction_frequency_hz: float | None

    def __init__(
        self,
        name: str,
        duration_ns: float,
        buffer_ns: float | None = None,
        interaction_frequency_hz: float | None = None,
    ) -> None:
        set_ = object.__setattr__
        set_(self, "name", name)
        set_(self, "duration_ns", duration_ns)
        set_(self, "buffer_ns", buffer_ns)
        set_(self, "interaction_frequency_hz", interaction_frequency_hz)

    def __setattr__(self, name: str, value: Any) -> None:
        raise FrozenInstanceError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> None:
        raise FrozenInstanceError(f"cannot delete field {name!r}")

    def __eq__(self, other: object) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented
        return all(getattr(self, n) == getattr(other, n) for n in self.__slots__)

    def __hash__(self) -> int:
        return hash((self.name, self.duration_ns, self.buffer_ns, self.interaction_frequency_hz))

    def __repr__(self) -> str:
        fields_ = ", ".join(f"{n}={getattr(self, n)!r}" for n in self.__slots__)
        return f"{type(self).__name__}({fields_})"

    def __reduce__(self) -> tuple[Any, ...]:
        return type(self), (
            self.name,
            self.duration_ns,
            self.buffer_ns,
            self.interaction_frequency_hz,
        )


@dataclass(frozen=True)
class Coupling:
    """A coupling of the qudits ``qudits`` of a quantum system: two in this
    release, by the system's qudit index, in the order the coupling names
    them (``transmon_17q``'s are (auxiliary, data)).

    The system's ``model`` decides the operator; a ``TransmonModel``
    coupling is the exchange g (a^dagger b + a b^dagger), in the
    rotating-wave approximation, and ``strength_hz`` is g / 2 pi. A
    ``strength_hz`` of ``None`` is one not measured yet: the system
    compiles, schedules and exports, and ``simulate`` refuses it. ``gate``
    is the gate characterized on the coupling, if any.
    """

    qudits: tuple[int, ...]
    strength_hz: float | None
    gate: Gate | None = None

    def __post_init__(self) -> None:
        qudits = self.qudits
        if qudits.__class__ is not tuple and isinstance(qudits, Sequence):
            object.__setattr__(self, "qudits", tuple(qudits))

    def __hash__(self) -> int:
        return hash((self.qudits, self.strength_hz))

    def __reduce__(self) -> tuple[Any, ...]:
        from .._pickling import frozen_reduce

        return frozen_reduce(self)


@_dataclass_when_used(frozen=True)
class ResidualZZ:
    """Residual ZZ between the qudits ``qudits`` (two, by the system's qudit
    index), in addition to what the couplings produce.

    ``rate_hz`` is the ZZ rate zeta / 2 pi, zeta = E11 - E10 - E01 + E00,
    modeled as zeta n_a n_b.
    """

    qudits: tuple[int, ...]
    rate_hz: float

    def __post_init__(self) -> None:
        qudits = self.qudits
        if qudits.__class__ is not tuple and isinstance(qudits, Sequence):
            object.__setattr__(self, "qudits", tuple(qudits))


# The operator a coupling stands for, by the simulation model of its system:
# the entry of the model's class or of its nearest base class here.
_COUPLING_OPERATORS: Mapping[type[SimulationModel], str] = MappingProxyType(
    {TransmonModel: "exchange"}
)


def _coupling_operator(model: Any) -> str | None:
    """The operator of the couplings of a system with the model *model*, or
    None if the model gives them none."""
    for cls in type(model).__mro__:
        operator = _COUPLING_OPERATORS.get(cls)
        if operator is not None:
            return operator
    return None


@dataclass(frozen=True, eq=False)
class TargetDescription:
    """Full description of a quantum processing unit.

    A target is read-only once built: ``qubits`` is a read-only mapping and
    ``couplings`` and ``residual_zz`` are tuples. ``clock_ghz`` is the pulse
    clock that converts nanoseconds to clock cycles. ``model`` is how the
    simulator treats its qudits, and how many levels each has: a
    ``SimulationModel`` class or instance (``TransmonModel`` by default,
    ``TransmonModel(levels={0: 4})`` to give qudit 0 four levels), or
    ``None`` for a system with no simulation model. The model also decides
    the operator of the couplings: the exchange for ``TransmonModel``.
    Targets compare by content, so a target unpickled is equal to the one
    pickled.

    A target's couplings and residual ZZ entries are checked when it is built,
    and a ``DeviceDefinitionError`` lists every problem, each at its path in
    the target (``couplings[3].qudits[1]``).
    """

    name: str
    qubits: Mapping[int, Qubit] = _EMPTY
    couplings: Sequence[Coupling] = ()
    residual_zz: Sequence[ResidualZZ] = ()
    model: SimulationModel | type[SimulationModel] | None = _TRANSMON
    attribution: str = ""
    clock_ghz: float = 2.0

    def __post_init__(self) -> None:
        if not self.clock_ghz > 0:
            raise InvalidValueError(f"clock_ghz must be positive, got {self.clock_ghz}")
        model = self.model
        if isinstance(model, type) and issubclass(model, SimulationModel):
            object.__setattr__(self, "model", model())
        elif model is not None and not isinstance(model, SimulationModel):
            raise InvalidTypeError(
                "model must be a SimulationModel class or instance, such as TransmonModel, "
                f"or None; got {model!r}"
            )
        object.__setattr__(self, "qubits", MappingProxyType(dict(self.qubits)))
        object.__setattr__(self, "couplings", tuple(self.couplings))
        object.__setattr__(self, "residual_zz", tuple(self.residual_zz))
        # Every problem with the target's entries, each at its path in it.
        problems = _entries_problems(self)
        if problems:
            raise DeviceDefinitionError(
                f"{_count(len(problems), 'problem')} in the quantum system {self.name!r}",
                problems=problems,
            )

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return all(getattr(self, f.name) == getattr(other, f.name) for f in fields(self))

    def __hash__(self) -> int:
        return hash((self.name, self.clock_ghz, len(self.qubits), len(self.couplings)))

    def __reduce__(self) -> tuple[Any, ...]:
        from .._pickling import frozen_reduce

        return frozen_reduce(self)

    @property
    def n_qubits(self) -> int:
        return len(self.qubits)

    @property
    def qubit_indices(self) -> list[int]:
        return sorted(self.qubits.keys())

    @property
    def frequencies(self) -> dict[int, float]:
        return {q.index: q.frequency_hz for q in self.qubits.values()}

    @property
    def anharmonicities(self) -> dict[int, float]:
        return {q.index: q.anharmonicity_hz for q in self.qubits.values()}

    @property
    def t1_times(self) -> dict[int, float]:
        """T1 in microseconds, keyed by qubit index."""
        return {q.index: q.t1_us for q in self.qubits.values()}

    @property
    def t2_times(self) -> dict[int, float]:
        """T2* in microseconds, keyed by qubit index."""
        return {q.index: q.t2_star_us for q in self.qubits.values()}

    @property
    def t2_echo_times(self) -> dict[int, float | None]:
        """Hahn-echo T2 in microseconds, keyed by qubit index (None if unknown)."""
        return {q.index: q.t2_echo_us for q in self.qubits.values()}

    @property
    def thermal_populations(self) -> dict[int, float]:
        """Equilibrium excited-state population, keyed by qubit index."""
        return {q.index: q.thermal_population for q in self.qubits.values()}

    @property
    def coupling_map(self) -> list[tuple[int, ...]]:
        """The qudits of each coupling, in the order of ``couplings``."""
        return [c.qudits for c in self.couplings]

    def coupling(self, a: int, b: int) -> Coupling:
        """The coupling of qudits *a* and *b*, in either order (the target's
        qudit indices, as ``Qudit.index``); a ``NotFoundError`` names the
        qudits *a* couples to."""
        for c in self.couplings:
            if c.qudits == (a, b) or c.qudits == (b, a):
                return c
        partners = sorted(q for c in self.couplings if a in c.qudits for q in c.qudits if q != a)
        if partners:
            which = f"qudit {a} couples to {', '.join(map(str, partners))}"
        else:
            which = f"qudit {a} has no couplings"
        raise NotFoundError(f"qudits {a} and {b} of {self.name} are not coupled; {which}")

    def connectivity_graph(self) -> dict[int, list[int]]:
        """The qudits each qudit couples to, by qudit index, for every qudit."""
        g: dict[int, list[int]] = {idx: [] for idx in self.qubits}
        for c in self.couplings:
            a, b = c.qudits
            g[a].append(b)
            g[b].append(a)
        return g

    def drive_amplitude_scale(self, qubit_index: int) -> float:
        """Return the conversion from pulse amplitude to radians/ns.

        A ``MicrowaveDrive`` may give ``amplitude_scale_rad_per_ns``
        explicitly; else, with a calibrated Gaussian/DRAG pi pulse
        (``x_amplitude``, ``x_duration_ns`` and ``x_sigma_ns``), the scale is
        inferred from its truncated-Gaussian area. A qudit with neither, or
        no drive, uses 1.0: pulse amplitudes are already angular rates in
        rad/ns. Another drive modality is ``UnsupportedError``.
        """
        if qubit_index not in self.qubits:
            raise NotFoundError(f"Qubit {qubit_index} not in target {self.name!r}")
        drive = self.qubits[qubit_index].drive
        if drive is None:
            return 1.0
        if not isinstance(drive, MicrowaveDrive):
            raise UnsupportedError(
                f"no drive model for {type(drive).__name__} (qubit {qubit_index}); the "
                "simulator models a MicrowaveDrive"
            )
        explicit = drive.amplitude_scale_rad_per_ns
        if explicit is not None:
            if explicit <= 0:
                raise InvalidValueError(
                    f"amplitude_scale_rad_per_ns must be positive for qubit {qubit_index}"
                )
            return float(explicit)

        amplitude = float(drive.x_amplitude or 0.0)
        duration = float(drive.x_duration_ns or 0.0)
        sigma = float(drive.x_sigma_ns or 0.0)
        if amplitude > 0.0 and duration > 0.0 and sigma > 0.0:
            area = (
                sigma
                * math.sqrt(2.0 * math.pi)
                * math.erf(duration / (2.0 * math.sqrt(2.0) * sigma))
            )
            return math.pi / (amplitude * area)
        return 1.0

    def hamiltonian_terms(self) -> list[dict[str, Any]]:
        """Generate two-level static and coupling Hamiltonian terms.

        Returns a list of term dicts compatible with ``OperatorTerm``.
        Each dict has keys: kind, qubit_indices, coefficient, time_dependent.
        Coefficients are angular frequencies in radians per nanosecond, matching
        the pulse IR time unit.

        ``anharmonicity_hz`` is calibration metadata for leakage models and is
        intentionally not emitted into this two-level spin Hamiltonian. A
        faithful anharmonicity term requires a three-or-more-level mode.

        The terms are in the lab frame: ``static_z`` is (omega / 2) Z, a
        coupling is the operator of the system's model (``coupling_exchange``,
        g (sigma+_a sigma-_b + sigma-_a sigma+_b) for ``TransmonModel``: the
        exchange in the rotating-wave approximation, as the simulator lowers
        it), and a ``residual_zz`` entry's ``crosstalk_zz`` is zeta n_a n_b.
        A coupling whose strength
        is not measured yet (``None``) is an ``InvalidValueError`` naming it,
        and couplings of a model that gives them no operator an
        ``UnsupportedError``.
        """
        terms: list[dict[str, Any]] = []
        operator = _coupling_operator(self.model) if self.couplings else None
        if self.couplings and operator is None:
            known = ", ".join(f"{c.__name__} ({op})" for c, op in _COUPLING_OPERATORS.items())
            raise UnsupportedError(
                f"the model {self.model!r} of {self.name!r} gives its couplings no "
                f"operator, so they have no Hamiltonian terms; the models whose couplings "
                f"have one are {known}"
            )
        coupling_kind = f"coupling_{operator}"

        for q in self.qubits.values():
            omega = q.frequency_hz * 2.0 * math.pi * _SECONDS_PER_NANOSECOND
            terms.append(
                {
                    "kind": "static_z",
                    "qubit_indices": (q.index,),
                    "coefficient": complex(omega / 2.0, 0),
                    "time_dependent": False,
                }
            )
        unmeasured = []
        for i, c in enumerate(self.couplings):
            if c.strength_hz is None:
                unmeasured.append(i)
                continue
            g = c.strength_hz * 2.0 * math.pi * _SECONDS_PER_NANOSECOND
            terms.append(
                {
                    "kind": coupling_kind,
                    "qubit_indices": c.qudits,
                    "coefficient": complex(g, 0),
                    "time_dependent": False,
                }
            )
        if unmeasured:
            raise _unmeasured(
                self, unmeasured, "the Hamiltonian terms need every coupling's strength"
            )

        for xt in self.residual_zz:
            zz = xt.rate_hz * 2.0 * math.pi * _SECONDS_PER_NANOSECOND
            terms.append(
                {
                    "kind": "crosstalk_zz",
                    "qubit_indices": xt.qudits,
                    "coefficient": complex(zz, 0),
                    "time_dependent": False,
                }
            )

        return terms

    def dissipator_terms(self) -> list[dict[str, Any]]:
        """Generate T1 / T2 Lindblad dissipator terms.

        Returns a list of term dicts, each with kind, qubit_indices,
        and coefficient. Collapse-operator coefficients have units of inverse
        square-root nanoseconds, matching evolution time in nanoseconds.
        """
        terms: list[dict[str, Any]] = []

        for q in self.qubits.values():
            if q.t1_us > 0:
                # Detailed balance: down at (1 - p) / T1 and up at p / T1.
                gamma1 = 1.0 / (q.t1_us * _NANOSECONDS_PER_MICROSECOND)
                p = q.thermal_population
                terms.append(
                    {
                        "kind": "dissipator_t1",
                        "qubit_indices": (q.index,),
                        "coefficient": complex(math.sqrt((1.0 - p) * gamma1), 0),
                    }
                )
                if p > 0:
                    terms.append(
                        {
                            "kind": "dissipator_thermal",
                            "qubit_indices": (q.index,),
                            "coefficient": complex(math.sqrt(p * gamma1), 0),
                        }
                    )
            if q.t2_star_us > 0:
                gamma_phi = 1.0 / (q.t2_star_us * _NANOSECONDS_PER_MICROSECOND)
                if q.t1_us > 0:
                    gamma_phi -= 1.0 / (2.0 * q.t1_us * _NANOSECONDS_PER_MICROSECOND)
                if gamma_phi > 0:
                    terms.append(
                        {
                            "kind": "dissipator_t2",
                            "qubit_indices": (q.index,),
                            "coefficient": complex(math.sqrt(gamma_phi / 2.0), 0),
                        }
                    )

        return terms


def _unmeasured(
    target: TargetDescription,
    indices: Sequence[int],
    needs: str,
    fix: str = "give the strength in Hz",
) -> InvalidValueError:
    """The error that *needs* the strengths of *target*'s couplings at
    *indices*, which are not measured yet."""
    named = [
        f"couplings[{i}] (qudits {' and '.join(map(str, target.couplings[i].qudits))})"
        for i in indices
    ]
    listed = named[0] if len(named) == 1 else ", ".join(named[:-1]) + " and " + named[-1]
    verb = "has" if len(named) == 1 else "have"
    return InvalidValueError(
        f"{needs}; {listed} of {target.name!r} {verb} strength_hz None, not measured yet: {fix}"
    )


# The couplings and residual ZZ entries last found to have no problem, and the
# qudits of their system. A system made from that one with
# dataclasses.replace, or with only its couplings' strengths changed (as a
# calibration does), checks only what changed.
_CHECKED: tuple[Sequence[Any], Sequence[Any], frozenset[int]] = ((), (), frozenset())


def _entries_problems(target: TargetDescription) -> list[Problem]:
    """Every problem with *target*'s couplings and residual ZZ entries."""
    global _CHECKED
    known = target.qubits
    couplings, residual_zz = target.couplings, target.residual_zz
    checked_couplings, checked_residual_zz, checked_qudits = _CHECKED
    if residual_zz is checked_residual_zz and known.keys() == checked_qudits:
        if couplings is checked_couplings or _strengths_changed(couplings, checked_couplings):
            return []
    if (not couplings or _plain_entries(couplings, Coupling, known)) and (
        not residual_zz or _plain_entries(residual_zz, ResidualZZ, known)
    ):
        _CHECKED = (couplings, residual_zz, frozenset(known))
        return []
    problems: list[Problem] = []
    groups: tuple[tuple[str, Sequence[Any]], ...] = (
        ("couplings", couplings),
        ("residual_zz", residual_zz),
    )
    for kind, entries in groups:
        pairs: dict[tuple[int, int], int] = {}
        for i, entry in enumerate(entries):
            found, pair = _entry_problems(kind, i, entry, known, pairs)
            problems += found
            if pair is not None:
                pairs[pair] = i
    return problems


# What the checks read of every entry or gate at once.
_QUDITS = attrgetter("qudits")
_STRENGTH = attrgetter("strength_hz")
_GATE = attrgetter("gate")
_RATE = attrgetter("rate_hz")
_GATE_FIELDS = attrgetter("name", "duration_ns", "buffer_ns", "interaction_frequency_hz")
_NONE: type[None] = type(None)


def _finite_floats(values: Sequence[Any], optional: bool = False) -> bool:
    """Whether *values* are all finite floats (or None, if *optional*): their
    sum is finite unless one is inf or NaN, or the floats overflow, which
    then only takes the slower checks."""
    types = set(map(type, values))
    if not (types <= {float, _NONE} if optional else types <= {float}):
        return False
    total = sum(filter(None, values), 0.0)
    return total - total == 0.0


def _plain_entries(entries: Sequence[Any], cls: type, known: Mapping[int, Any]) -> bool:
    """Whether *entries*, couplings or residual ZZ entries of a system with the
    qudits *known*, have nothing to report; checked in C for every entry at
    once, by class, so any other entry takes ``_entry_problems``."""
    if set(map(type, entries)) != {cls}:
        return False
    qudits = list(map(_QUDITS, entries))
    if set(map(type, qudits)) != {tuple} or set(map(len, qudits)) != {2}:
        return False
    flat = list(chain.from_iterable(qudits))
    if set(map(type, flat)) != {int} or not known.keys() >= set(flat):
        return False
    if any(map(eq, flat[::2], flat[1::2])) or len(set(map(frozenset, qudits))) < len(qudits):
        return False
    if cls is ResidualZZ:
        return _finite_floats(list(map(_RATE, entries)))
    if not _finite_floats(list(map(_STRENGTH, entries)), optional=True):
        return False
    gates = list(filter(None, map(_GATE, entries)))
    if not gates:
        return True
    if set(map(type, gates)) != {Gate}:
        return False
    names, durations, buffers, frequencies = zip(*map(_GATE_FIELDS, gates), strict=True)
    if set(map(type, names)) != {str} or "" in names:
        return False
    given = [f for f in frequencies if f is not None]
    return (
        _finite_floats(durations)
        and min(durations) >= 0.0
        and _finite_floats(buffers, optional=True)
        and min(filter(None, buffers), default=0.0) >= 0.0
        and _finite_floats(given)
        and min(given, default=1.0) > 0.0
    )


def _strengths_changed(couplings: Sequence[Any], checked: Sequence[Any]) -> bool:
    """Whether *couplings* are the *checked* ones with only their strengths
    changed, to strengths with nothing to report."""
    return (
        len(couplings) == len(checked)
        and set(map(type, couplings)) <= {Coupling}
        and all(map(is_, map(_QUDITS, couplings), map(_QUDITS, checked)))
        and all(map(is_, map(_GATE, couplings), map(_GATE, checked)))
        and _finite_floats(list(map(_STRENGTH, couplings)), optional=True)
    )


def _pair_key(qudits: tuple[int, ...]) -> tuple[int, int]:
    a, b = qudits
    return (a, b) if a < b else (b, a)


def _finite(value: Any) -> bool:
    """Whether *value* is a finite real number, not a bool."""
    return value.__class__ is not bool and isinstance(value, numbers.Real) and math.isfinite(value)


def _entry_problems(
    kind: str, index: int, entry: Any, known: Mapping[int, Any], pairs: Mapping[Any, int]
) -> tuple[list[Problem], tuple[int, int] | None]:
    """Every problem with *entry*, entry *index* of the system's *kind*
    (``"couplings"`` or ``"residual_zz"``), each at its path in the system,
    and its pair of qudits if they are two distinct known qudits of a new
    pair. *pairs* holds the earlier entries' pairs, by entry index."""
    where = f"{kind}[{index}]"
    cls = Coupling if kind == "couplings" else ResidualZZ
    if not isinstance(entry, cls):
        problem = Problem(
            f"must be a {cls.__name__}, got {type(entry).__name__}", path=where, value=entry
        )
        return [problem], None
    noun = "coupling" if cls is Coupling else "residual ZZ entry"
    problems = _qudit_problems(entry.qudits, where, known, pairs, noun)
    pair = None if problems else _pair_key(entry.qudits)
    if isinstance(entry, Coupling):
        value = entry.strength_hz
        if value is not None and not _finite(value):
            problems.append(
                Problem(
                    f"{value!r} is not a finite number of Hz",
                    path=f"{where}.strength_hz",
                    value=value,
                    fix="give the strength in Hz, or None (null in a file) if it is not "
                    "measured yet",
                )
            )
        if entry.gate is not None:
            problems += _gate_problems(entry.gate, f"{where}.gate")
        return problems, pair
    if not _finite(entry.rate_hz):
        problems.append(
            Problem(
                f"{entry.rate_hz!r} is not a finite number of Hz",
                path=f"{where}.rate_hz",
                value=entry.rate_hz,
                fix="give the ZZ rate in Hz",
            )
        )
    return problems, pair


def _qudit_problems(
    qudits: Any, where: str, known: Mapping[int, Any], pairs: Mapping[Any, int], noun: str
) -> list[Problem]:
    """The problems with *qudits*, the qudits of the entry at *where*."""
    path = f"{where}.qudits"
    fix = "give the indices of its two qudits, as (0, 1)"
    if not isinstance(qudits, tuple):
        return [
            Problem(
                f"must be a tuple of qudit indices, got {type(qudits).__name__}",
                path=path,
                value=qudits,
                fix=fix,
            )
        ]
    if len(qudits) != 2:
        return [
            Problem(
                f"names {_count(len(qudits), 'qudit')}; a {noun} is between two qudits "
                "in this release",
                path=path,
                value=qudits,
                fix=fix,
            )
        ]
    problems = []
    for j, q in enumerate(qudits):
        if q.__class__ is bool or not isinstance(q, int):
            problems.append(
                Problem(
                    f"must be a qudit index (an int), got {type(q).__name__}",
                    path=f"{path}[{j}]",
                    value=q,
                )
            )
        elif q not in known:
            problems.append(
                Problem(
                    f"{q} is not a qudit of the quantum system",
                    path=f"{path}[{j}]",
                    value=q,
                    fix=f"its qudits are {_ranges(known)}" if known else "it has no qudits",
                )
            )
    if problems:
        return problems
    a, b = qudits
    if a == b:
        return [
            Problem(
                f"names qudit {a} twice",
                path=path,
                value=qudits,
                fix=f"a {noun} is between two distinct qudits",
            )
        ]
    earlier = pairs.get(_pair_key(qudits))
    if earlier is not None:
        kind = where.partition("[")[0]
        return [
            Problem(
                f"names the same pair of qudits as {kind}[{earlier}]",
                path=path,
                value=qudits,
                fix=f"give each pair one {noun}; ({a}, {b}) and ({b}, {a}) are the same pair",
            )
        ]
    return []


def _gate_problems(gate: Any, where: str) -> list[Problem]:
    if not isinstance(gate, Gate):
        return [
            Problem(f"must be a Gate or None, got {type(gate).__name__}", path=where, value=gate)
        ]
    problems = []
    if not isinstance(gate.name, str) or not gate.name:
        problems.append(
            Problem(
                f"must be a non-empty string, got {gate.name!r}",
                path=f"{where}.name",
                value=gate.name,
                fix='name the gate, as "cz"',
            )
        )
    for name, required in (("duration_ns", True), ("buffer_ns", False)):
        value = getattr(gate, name)
        if (value is not None or required) and not (_finite(value) and value >= 0):
            problems.append(
                Problem(
                    f"{value!r} is not a finite time of 0 ns or more",
                    path=f"{where}.{name}",
                    value=value,
                    fix="give it in ns" if required else "give it in ns, or None",
                )
            )
    frequency = gate.interaction_frequency_hz
    if frequency is not None and not (_finite(frequency) and frequency > 0):
        problems.append(
            Problem(
                f"{frequency!r} is not a finite frequency above 0 Hz",
                path=f"{where}.interaction_frequency_hz",
                value=frequency,
                fix="give it in Hz, or None",
            )
        )
    return problems
