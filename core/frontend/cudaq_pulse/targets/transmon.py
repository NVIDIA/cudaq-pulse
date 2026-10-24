# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Transmon QPU target definitions.

The default target is the 17-qubit distance-three surface-code device of
Krinner et al., Nature 605, 669 (2022), arXiv:2112.03708, built from the
values the paper publishes (``_krinner17``, with a citation for each).
"""

from __future__ import annotations

from . import _krinner17 as _k
from .base import Coupling, Qubit, Target


def transmon_krinner_17q() -> Target:
    """17-qubit surface-code device of Krinner et al., Nature 605, 669 (2022).

    Qudits 0-16 are D1-D9, X1-X4, Z1-Z4, the column order of the paper's
    Tab. I; ``Qubit.label`` carries the name. The 24 couplings are the
    auxiliary-data pairs of the stabilizers of Tab. II::

        Z1: D1 D4         X1: D2 D3
        Z2: D4 D5 D7 D8   X2: D1 D2 D4 D5
        Z3: D2 D3 D5 D6   X3: D5 D6 D8 D9
        Z4: D6 D9         X4: D7 D8

    Data qubits idle at 3.74-4.14 GHz (their lower sweet spot) and
    auxiliary qubits at 5.89-6.33 GHz (their upper one), with idle
    frequencies, anharmonicities, T1 and Ramsey T2* from Tab. I. Every
    coupling is the device-wide J / 2 pi ~ 7 MHz (Methods A). Each carries
    its CZ time step, interaction frequency (``gate_params``) and measured
    gate fidelity; the gate takes 98 ns on average, with 15 ns buffers.
    Drives are the published 40 ns DRAG pulses with sigma = 10 ns; their
    amplitudes are not published, so amplitudes are angular rates in
    rad/ns. Readout parameters are those of Tab. I and App. F1.

    No residual ZZ is listed: the paper gives no per-pair values, only
    that the calculated idle ZZ is below about 8 kHz. Simulated with three
    or more levels, the couplings produce it; ``krinner_idle_zz_hz`` gives
    its second-order value.
    """
    qubits = {}
    for index, q in enumerate(_k.QUBITS):
        qubits[index] = Qubit(
            index=index,
            frequency_hz=q.idle_frequency_hz,
            anharmonicity_hz=q.anharmonicity_hz,
            t1_us=q.t1_us,
            t2_star_us=q.t2_ramsey_us,
            t2_echo_us=q.t2_echo_us,
            thermal_population=q.thermal_population,
            label=q.label,
            drive_params={"x_dur": _k.DRIVE_DURATION_NS, "x_sigma": _k.DRIVE_SIGMA_NS},
            readout_params={
                "ro_freq": q.readout_frequency_hz,
                "ro_dur": _k.READOUT_DURATION_NS[q.label[0]],
                "ro_sigma": _k.READOUT_SIGMA_NS,
                "ro_integration": _k.READOUT_INTEGRATION_NS,
                "ro_qubit_freq": q.readout_qubit_frequency_hz,
                "chi_hz": q.dispersive_shift_hz,
                "ro_chi_hz": q.readout_dispersive_shift_hz,
                "kappa_eff_hz": q.kappa_eff_hz,
                "resonator_coupling_hz": q.resonator_coupling_hz,
            },
        )
    couplings = [
        Coupling(
            _k.INDEX[e.auxiliary],
            _k.INDEX[e.data],
            coupling_strength_hz=_k.COUPLING_HZ,
            gate_type="cz",
            gate_duration_ns=_k.CZ_DURATION_NS,
            gate_buffer_ns=_k.CZ_BUFFER_NS,
            gate_fidelity=1.0 - e.cz_error,
            gate_params={
                "cz_step": float(e.cz_step),
                "interaction_frequency_hz": e.interaction_frequency_hz,
            },
        )
        for e in _k.EDGES
    ]
    return Target(
        name="transmon_krinner_17q",
        qubits=qubits,
        couplings=couplings,
        architecture="transmon",
        attribution=_k.CITATION,
    )


def krinner_idle_zz_hz() -> dict[tuple[int, int], float]:
    """Second-order idle ZZ rate of each coupled pair of the Krinner target.

    zeta = 2 J^2 (a_a + a_d) / ((D + a_d)(D - a_a)), with D the data minus
    the auxiliary idle frequency and a_a, a_d their anharmonicities: a value
    derived from the published J, frequencies and anharmonicities, keyed by
    the coupling's (auxiliary, data) qudit indices.
    """
    zz = {}
    for e in _k.EDGES:
        a, d = _k.INDEX[e.auxiliary], _k.INDEX[e.data]
        aux, data = _k.QUBITS[a], _k.QUBITS[d]
        detuning = data.idle_frequency_hz - aux.idle_frequency_hz
        alpha_a, alpha_d = aux.anharmonicity_hz, data.anharmonicity_hz
        zz[(a, d)] = (
            2.0
            * _k.COUPLING_HZ**2
            * (alpha_a + alpha_d)
            / ((detuning + alpha_d) * (detuning - alpha_a))
        )
    return zz


def transmon_generic(
    n_qubits: int,
    *,
    base_frequency_hz: float = 5.0e9,
    frequency_spread_hz: float = 100e6,
    anharmonicity_hz: float = -330e6,
    coupling_strength_hz: float = 3.0e6,
    t1_us: float = 25.0,
    t2_star_us: float = 15.0,
    topology: str = "linear",
) -> Target:
    """Build a generic transmon target with configurable parameters.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    base_frequency_hz : float
        Frequency of qubit 0.
    frequency_spread_hz : float
        Frequency spacing between adjacent qubits.
    anharmonicity_hz : float
        Uniform anharmonicity (negative for transmon).
    coupling_strength_hz : float
        Uniform nearest-neighbor coupling.
    t1_us, t2_star_us : float
        Uniform decoherence times.
    topology : str
        ``"linear"`` (chain), ``"ring"``, or ``"grid"`` (square lattice
        with side length = ceil(sqrt(n_qubits))).
    """
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")

    import math as _math

    qubits = {}
    for i in range(n_qubits):
        qubits[i] = Qubit(
            index=i,
            frequency_hz=base_frequency_hz + i * frequency_spread_hz / max(n_qubits - 1, 1),
            anharmonicity_hz=anharmonicity_hz,
            t1_us=t1_us,
            t2_star_us=t2_star_us,
            label=f"Q{i}",
        )

    edges: list[tuple[int, int]] = []
    if topology == "linear":
        edges = [(i, i + 1) for i in range(n_qubits - 1)]
    elif topology == "ring":
        edges = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
    elif topology == "grid":
        side = _math.ceil(_math.sqrt(n_qubits))
        for i in range(n_qubits):
            r, c = divmod(i, side)
            if c + 1 < side and i + 1 < n_qubits:
                edges.append((i, i + 1))
            if r + 1 < side and i + side < n_qubits:
                edges.append((i, i + side))
    else:
        raise ValueError(f"Unknown topology {topology!r}. Use 'linear', 'ring', or 'grid'.")

    couplings = [Coupling(a, b, coupling_strength_hz=coupling_strength_hz) for a, b in edges]

    return Target(
        name=f"transmon_generic_{n_qubits}q_{topology}",
        qubits=qubits,
        couplings=couplings,
        architecture="transmon",
    )
