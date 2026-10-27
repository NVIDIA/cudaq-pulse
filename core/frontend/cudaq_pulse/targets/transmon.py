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

import math
import threading
from typing import Any

from .._errors import InvalidValueError
from . import _krinner17 as _k
from .base import Coupling, Qubit, TargetDescription


def transmon_krinner_17q(calibration: Any = "krinner17") -> TargetDescription:
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
    frequencies, anharmonicities, T1 and Ramsey T2* from Tab. I.

    The paper's device-wide J / 2 pi ~ 7 MHz (Methods A) is the
    ``|11> <-> |20>`` coupling of the CZ at each pair's interaction point (7.4
    MHz measured on X2-D4 at 4.4 GHz, Hellings et al., arXiv:2503.04610).
    *calibration* (a ``Calibration``, the name of a shipped one, or None)
    supplies the device-wide charge coupling g_c (fitted to the X2-D4
    measurement, 3.12 MHz in ``krinner17``), and each coupling is the
    exchange J_qq = g_c ``<0|n_a|1> <0|n_d|1>`` at the qubits' idle points:
    5.67-6.08 MHz with the shipped ``krinner17``. With
    ``calibration=None`` it is the published value over sqrt(2), 4.95 MHz.
    Each coupling carries its CZ time step, interaction frequency and
    ``|11> <-> |20>`` coupling (``gate_params``) and measured gate fidelity; the
    gate takes 98 ns on average, with 15 ns buffers.
    Drives are the published 40 ns DRAG pulses with sigma = 10 ns; their
    amplitudes are not published, so amplitudes are angular rates in
    rad/ns. Readout parameters are those of Tab. I and App. F1.

    No residual ZZ is listed: the paper gives no per-pair values, only
    that the calculated idle ZZ is below about 8 kHz. Simulated with three
    or more levels, the couplings produce it; ``krinner_idle_zz_hz`` gives
    the charge model's value, -4.5 to -7.3 kHz with ``krinner17``.
    """
    if isinstance(calibration, str):
        return _calibrated_krinner(calibration)
    target = _published_krinner()
    return target if calibration is None else calibration.apply(target)


_BUILT: dict[str | None, TargetDescription] = {}
# Reentrant: building a calibrated target builds the published one.
_BUILT_LOCK = threading.RLock()


def _calibrated_krinner(name: str) -> TargetDescription:
    from .calibration import shipped_calibration

    built = _BUILT.get(name)
    if built is None:
        with _BUILT_LOCK:
            built = _BUILT.get(name)
            if built is None:
                built = _BUILT[name] = shipped_calibration(name).apply(_published_krinner())
    return built


def _published_krinner() -> TargetDescription:
    built = _BUILT.get(None)
    if built is None:
        with _BUILT_LOCK:
            built = _BUILT.get(None)
            if built is None:
                built = _BUILT[None] = _build_published_krinner()
    return built


def _build_published_krinner() -> TargetDescription:
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
            coupling_strength_hz=_k.coupling_11_20_hz(e.auxiliary, e.data) / math.sqrt(2.0),
            gate_type="cz",
            gate_duration_ns=_k.CZ_DURATION_NS,
            gate_buffer_ns=_k.CZ_BUFFER_NS,
            gate_fidelity=1.0 - e.cz_error,
            gate_params={
                "cz_step": float(e.cz_step),
                "interaction_frequency_hz": e.interaction_frequency_hz,
                "coupling_11_20_hz": _k.coupling_11_20_hz(e.auxiliary, e.data),
            },
        )
        for e in _k.EDGES
    ]
    return TargetDescription(
        name="transmon_krinner_17q",
        qubits=qubits,
        couplings=couplings,
        architecture="transmon",
        attribution=_k.CITATION,
    )


def krinner_idle_zz_hz(calibration: Any = "krinner17") -> dict[tuple[int, int], float]:
    """Idle ZZ of each coupled pair of the Krinner target, keyed by the
    coupling's (auxiliary, data) qudit indices.

    zeta = E_11 - E_10 - E_01 + E_00 of the dressed levels of the
    charge-coupled pair (charge basis, five levels each, no rotating-wave
    approximation) at the idle sweet spots, with the circuits and couplings
    of *calibration*: -4.5 to -7.3 kHz with the shipped ``krinner17``, below
    the paper's "below about 8 kHz". The simulation's Duffing ladder with
    exchange J_qq reproduces it within a few percent.
    """
    from .calibration import shipped_calibration

    if isinstance(calibration, str):
        calibration = shipped_calibration(calibration)
    return {
        (_k.INDEX[e.auxiliary], _k.INDEX[e.data]): calibration.idle_zz_hz(f"{e.auxiliary}-{e.data}")
        for e in _k.EDGES
    }


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
) -> TargetDescription:
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
        raise InvalidValueError(f"n_qubits must be >= 1, got {n_qubits}")

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
        raise InvalidValueError(f"Unknown topology {topology!r}. Use 'linear', 'ring', or 'grid'.")

    couplings = [Coupling(a, b, coupling_strength_hz=coupling_strength_hz) for a, b in edges]

    return TargetDescription(
        name=f"transmon_generic_{n_qubits}q_{topology}",
        qubits=qubits,
        couplings=couplings,
        architecture="transmon",
    )
