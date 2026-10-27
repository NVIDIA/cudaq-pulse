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
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

_SECONDS_PER_NANOSECOND = 1.0e-9
_NANOSECONDS_PER_MICROSECOND = 1.0e3


@dataclass(frozen=True)
class Qubit:
    """Single qubit in a target device."""

    index: int
    frequency_hz: float
    anharmonicity_hz: float
    t1_us: float
    t2_star_us: float
    label: str = ""
    drive_params: Mapping[str, float] = field(default_factory=dict)
    readout_params: Mapping[str, float] = field(default_factory=dict)
    # Hahn-echo T2 (us), which the simulation dephases at on request, and
    # the excited-state population at equilibrium (a fraction), from which
    # it starts and towards which T1 relaxes.
    t2_echo_us: float | None = None
    thermal_population: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "drive_params", MappingProxyType(dict(self.drive_params)))
        object.__setattr__(self, "readout_params", MappingProxyType(dict(self.readout_params)))

    @property
    def depletion_ns(self) -> float:
        """How long the readout resonator takes to empty after a readout.

        ``readout_params["depletion_ns"]`` if given; else the time for its
        photons to fall to 1% at the decay rate ``kappa_eff_hz``,
        ln(100) / kappa; else 0.
        """
        if "depletion_ns" in self.readout_params:
            return float(self.readout_params["depletion_ns"])
        kappa_hz = self.readout_params.get("kappa_eff_hz")
        if kappa_hz:
            return math.log(100.0) / (2 * math.pi * float(kappa_hz) * 1e-9)
        return 0.0


@dataclass(frozen=True)
class Coupling:
    """Coupling edge between two qubits.

    ``coupling_strength_hz`` is g / 2 pi of the exchange coupling
    g (a^dagger b + a b^dagger).

    ``gate_type``, ``gate_duration_ns``, ``gate_buffer_ns`` (the idle time
    around the gate) and ``gate_fidelity`` describe the edge's two-qubit gate.
    Each defaults to ``None``: the target does not specify it. Neither
    compilation nor simulation reads them; ``transmon_krinner_17q`` gives the
    published values.
    """

    qubit_a: int
    qubit_b: int
    coupling_strength_hz: float
    gate_type: str | None = None
    gate_duration_ns: float | None = None
    gate_buffer_ns: float | None = None
    gate_fidelity: float | None = None
    gate_params: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "gate_params", MappingProxyType(dict(self.gate_params)))


@dataclass(frozen=True)
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
    clock that converts nanoseconds to clock cycles.
    """

    name: str
    qubits: Mapping[int, Qubit] = field(default_factory=dict)
    couplings: Sequence[Coupling] = ()
    crosstalk: Sequence[CrosstalkEntry] = ()
    architecture: str = "transmon"
    attribution: str = ""
    clock_ghz: float = 2.0

    def __post_init__(self) -> None:
        if not self.clock_ghz > 0:
            raise ValueError(f"clock_ghz must be positive, got {self.clock_ghz}")
        object.__setattr__(self, "qubits", MappingProxyType(dict(self.qubits)))
        object.__setattr__(self, "couplings", tuple(self.couplings))
        object.__setattr__(self, "crosstalk", tuple(self.crosstalk))

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

    def connectivity_graph(self) -> dict[int, list[int]]:
        """Adjacency list representation of qubit connectivity."""
        g: dict[int, list[int]] = {idx: [] for idx in self.qubits}
        for c in self.couplings:
            g.setdefault(c.qubit_a, []).append(c.qubit_b)
            g.setdefault(c.qubit_b, []).append(c.qubit_a)
        return g

    def get_drive_params(self, qubit_index: int) -> dict[str, float]:
        """Per-qubit drive parameters (amp, sigma, beta, etc.)."""
        if qubit_index not in self.qubits:
            raise KeyError(f"Qubit {qubit_index} not in target {self.name!r}")
        return dict(self.qubits[qubit_index].drive_params)

    def drive_amplitude_scale(self, qubit_index: int) -> float:
        """Return the conversion from pulse amplitude to radians/ns.

        Targets may provide ``amplitude_scale_rad_per_ns`` explicitly. For
        transmon calibration records containing a Gaussian/DRAG pi pulse
        (``x_amp``, ``x_dur``, and ``x_sigma``), the scale is inferred from
        its truncated-Gaussian area. Targets without either representation use
        1.0, meaning pulse amplitudes are already angular rates in rad/ns.
        """
        params = self.get_drive_params(qubit_index)
        explicit = params.get("amplitude_scale_rad_per_ns")
        if explicit is not None:
            if explicit <= 0:
                raise ValueError(
                    f"amplitude_scale_rad_per_ns must be positive for qubit {qubit_index}"
                )
            return float(explicit)

        amplitude = float(params.get("x_amp", 0.0))
        duration = float(params.get("x_dur", 0.0))
        sigma = float(params.get("x_sigma", 0.0))
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
