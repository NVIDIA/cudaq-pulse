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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from .._errors import InvalidTypeError, InvalidValueError, NotFoundError, UnsupportedError
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


@dataclass(frozen=True)
class Coupling:
    """Coupling edge between two qubits.

    ``coupling_strength_hz`` is g / 2 pi of the exchange coupling
    g (a^dagger b + a b^dagger).

    ``gate_type``, ``gate_duration_ns``, ``gate_buffer_ns`` (the idle time
    around the gate) and ``gate_fidelity`` describe the edge's two-qubit gate.
    Each defaults to ``None``: the target does not specify it. Neither
    compilation nor simulation reads them; ``transmon_17q`` gives the
    published values.
    """

    qubit_a: int
    qubit_b: int
    coupling_strength_hz: float
    gate_type: str | None = None
    gate_duration_ns: float | None = None
    gate_buffer_ns: float | None = None
    gate_fidelity: float | None = None
    gate_params: Mapping[str, float] = _EMPTY

    def __post_init__(self) -> None:
        object.__setattr__(self, "gate_params", MappingProxyType(dict(self.gate_params)))

    def __hash__(self) -> int:
        return hash((self.qubit_a, self.qubit_b, self.coupling_strength_hz))

    def __reduce__(self) -> tuple[Any, ...]:
        from .._pickling import frozen_reduce

        return frozen_reduce(self)


@_dataclass_when_used(frozen=True)
class CrosstalkEntry:
    """Residual ZZ or other parasitic coupling between qubit pairs.

    ``static_zz_hz`` is the ZZ rate zeta / 2 pi, zeta = E11 - E10 - E01 + E00,
    modeled as zeta n_a n_b in addition to what the couplings produce.
    ``zz_coupling`` and ``freq_delta_hz`` are reserved for a later crosstalk
    model: they are stored, but neither compilation nor simulation reads
    them yet.
    """

    qubit_a: int
    qubit_b: int
    zz_coupling: float
    static_zz_hz: float
    freq_delta_hz: float


@dataclass(frozen=True, eq=False)
class TargetDescription:
    """Full description of a quantum processing unit.

    A target is read-only once built: ``qubits`` is a read-only mapping and
    ``couplings`` and ``crosstalk`` are tuples. ``clock_ghz`` is the pulse
    clock that converts nanoseconds to clock cycles. ``model`` is how the
    simulator treats its qudits, and how many levels each has: a
    ``SimulationModel`` class or instance (``TransmonModel`` by default,
    ``TransmonModel(levels={0: 4})`` to give qudit 0 four levels), or
    ``None`` for a system with no simulation model. Targets compare by
    content, so a target unpickled is equal to the one pickled.
    """

    name: str
    qubits: Mapping[int, Qubit] = _EMPTY
    couplings: Sequence[Coupling] = ()
    crosstalk: Sequence[CrosstalkEntry] = ()
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
        object.__setattr__(self, "crosstalk", tuple(self.crosstalk))

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
    def coupling_map(self) -> list[tuple[int, int]]:
        return [(c.qubit_a, c.qubit_b) for c in self.couplings]

    def coupling(self, a: int, b: int) -> Coupling:
        """The coupling of qudits *a* and *b*, in either order (the target's
        qudit indices, as ``Qudit.index``); a ``NotFoundError`` names the
        qudits *a* couples to."""
        for c in self.couplings:
            if (c.qubit_a == a and c.qubit_b == b) or (c.qubit_a == b and c.qubit_b == a):
                return c
        partners = sorted(
            c.qubit_b if c.qubit_a == a else c.qubit_a
            for c in self.couplings
            if a in (c.qubit_a, c.qubit_b)
        )
        if partners:
            which = f"qudit {a} couples to {', '.join(map(str, partners))}"
        else:
            which = f"qudit {a} has no couplings"
        raise NotFoundError(f"qudits {a} and {b} of {self.name} are not coupled; {which}")

    def connectivity_graph(self) -> dict[int, list[int]]:
        """Adjacency list representation of qubit connectivity."""
        g: dict[int, list[int]] = {idx: [] for idx in self.qubits}
        for c in self.couplings:
            g.setdefault(c.qubit_a, []).append(c.qubit_b)
            g.setdefault(c.qubit_b, []).append(c.qubit_a)
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

        The terms are in the lab frame: ``static_z`` is (omega / 2) Z,
        ``coupling_xx`` is g X X (the exchange coupling with its
        counter-rotating terms), and ``crosstalk_zz`` is zeta n_a n_b.
        """
        terms: list[dict[str, Any]] = []

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
        for c in self.couplings:
            g = c.coupling_strength_hz * 2.0 * math.pi * _SECONDS_PER_NANOSECOND
            terms.append(
                {
                    "kind": "coupling_xx",
                    "qubit_indices": (c.qubit_a, c.qubit_b),
                    "coefficient": complex(g, 0),
                    "time_dependent": False,
                }
            )

        for xt in self.crosstalk:
            zz = xt.static_zz_hz * 2.0 * math.pi * _SECONDS_PER_NANOSECOND
            terms.append(
                {
                    "kind": "crosstalk_zz",
                    "qubit_indices": (xt.qubit_a, xt.qubit_b),
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
