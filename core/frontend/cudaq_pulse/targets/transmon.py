# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transmon QPU target definitions.

The default target is the 17-qubit distance-three surface-code device of
Krinner et al., Nature 605, 669 (2022), arXiv:2112.03708, built from the
values the paper publishes (``_transmon_17q``, with a citation for each).
"""

from __future__ import annotations

import math
import threading
from typing import Any

from .._errors import InvalidValueError
from . import _transmon_17q as _k
from .base import (
    Coupling,
    DeviceDescription,
    DispersiveReadout,
    DragPulse,
    FluxCZ,
    MicrowaveDrive,
    Qubit,
    ReadoutPulse,
    _Calibrations,
)

#: The CZs of one error correction cycle of ``transmon_17q``, layer by layer
#: (Fig. 3a and the edge labels of Fig. 6c): each layer is three CZs on
#: disjoint pairs, each pair the (auxiliary, data) qudit indices of a
#: coupling. Layers 1-4 measure the Z stabilizers, 5-8 the X stabilizers.
CZ_LAYERS: tuple[tuple[tuple[int, int], ...], ...] = (
    ((13, 0), (14, 4), (15, 2)),  # Z1-D1 Z2-D5 Z3-D3
    ((13, 3), (14, 7), (15, 4)),  # Z1-D4 Z2-D8 Z3-D5
    ((14, 3), (15, 1), (16, 5)),  # Z2-D4 Z3-D2 Z4-D6
    ((14, 6), (15, 5), (16, 8)),  # Z2-D7 Z3-D6 Z4-D9
    ((10, 0), (11, 4), (12, 6)),  # X2-D1 X3-D5 X4-D7
    ((10, 1), (11, 5), (12, 7)),  # X2-D2 X3-D6 X4-D8
    ((9, 2), (10, 4), (11, 8)),  # X1-D3 X2-D5 X3-D9
    ((9, 1), (10, 3), (11, 7)),  # X1-D2 X2-D4 X3-D8
)


def transmon_17q(calibration: Any = "transmon_17q") -> DeviceDescription:
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
    measurement, 3.12 MHz in ``transmon_17q``), and each coupling is the
    exchange J_qq = g_c ``<0|n_a|1> <0|n_d|1>`` at the qubits' idle points:
    5.67-6.08 MHz with the shipped ``transmon_17q``. With
    ``calibration=None`` it is the published value over sqrt(2), 4.95 MHz.
    Each coupling's ``CZ`` calibration is a ``FluxCZ``: 98 ns on average,
    with 15 ns buffers, at the pair's interaction frequency of Fig. 6c.
    ``CZ_LAYERS`` gives the time steps the CZs run in.
    Each qudit's ``X`` is the published 40 ns DRAG pulse with sigma = 10 ns;
    its amplitude is not published, so amplitudes are angular rates in
    rad/ns and ``get_calibration`` gives the pi pulse's. Readout physics is
    that of Tab. I, and the ``READOUT`` pulse that of App. F1.

    No residual ZZ is listed: the paper gives no per-pair values, only
    that the calculated idle ZZ is below about 8 kHz. Simulated with three
    or more levels, the couplings produce it; ``transmon_17q_idle_zz_hz`` gives
    the charge model's value, -4.5 to -7.3 kHz with ``transmon_17q``.
    """
    if isinstance(calibration, str):
        return _calibrated_17q(calibration)
    target = _published_17q()
    return target if calibration is None else calibration.apply(target)


_BUILT: dict[str | None, DeviceDescription] = {}
# Reentrant: building a calibrated target builds the published one.
_BUILT_LOCK = threading.RLock()


def _calibrated_17q(name: str) -> DeviceDescription:
    from .calibration import shipped_calibration

    built = _BUILT.get(name)
    if built is None:
        with _BUILT_LOCK:
            built = _BUILT.get(name)
            if built is None:
                built = _BUILT[name] = shipped_calibration(name).apply(_published_17q())
    return built


def _published_17q() -> DeviceDescription:
    built = _BUILT.get(None)
    if built is None:
        with _BUILT_LOCK:
            built = _BUILT.get(None)
            if built is None:
                built = _BUILT[None] = _build_published_17q()
    return built


def _build_published_17q() -> DeviceDescription:
    qubits = {}
    drive = MicrowaveDrive()
    x = DragPulse(duration_ns=_k.DRIVE_DURATION_NS, sigma_ns=_k.DRIVE_SIGMA_NS)
    # One calibration mapping per kind of qudit (data, X and Z auxiliaries),
    # built as checked: these values are known to be right.
    calibrations = {
        kind: _Calibrations(
            {
                "x": x,
                "readout": ReadoutPulse(
                    duration_ns=duration_ns,
                    sigma_ns=_k.READOUT_SIGMA_NS,
                    integration_ns=_k.READOUT_INTEGRATION_NS,
                ),
            }
        )
        for kind, duration_ns in _k.READOUT_DURATION_NS.items()
    }
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
            drive=drive,
            readout=DispersiveReadout(
                frequency_hz=q.readout_frequency_hz,
                dispersive_shift_hz=q.dispersive_shift_hz,
                linewidth_hz=q.kappa_eff_hz,
            ),
            calibrations=calibrations[q.label[0]],
        )
    couplings = [
        Coupling(
            (_k.INDEX[e.auxiliary], _k.INDEX[e.data]),
            strength_hz=_k.coupling_11_20_hz(e.auxiliary, e.data) / math.sqrt(2.0),
            calibrations=_Calibrations(
                {
                    "cz": FluxCZ(
                        duration_ns=_k.CZ_DURATION_NS,
                        buffer_ns=_k.CZ_BUFFER_NS,
                        interaction_frequency_hz=e.interaction_frequency_hz,
                    )
                }
            ),
        )
        for e in _k.EDGES
    ]
    return DeviceDescription(
        name="transmon_17q",
        qubits=qubits,
        couplings=couplings,
        attribution=_k.CITATION,
    )


def transmon_17q_idle_zz_hz(calibration: Any = "transmon_17q") -> dict[tuple[int, int], float]:
    """Idle ZZ of each coupled pair of the ``transmon_17q`` target, keyed by the
    coupling's (auxiliary, data) qudit indices.

    zeta = E_11 - E_10 - E_01 + E_00 of the dressed levels of the
    charge-coupled pair (charge basis, five levels each, no rotating-wave
    approximation) at the idle sweet spots, with the circuits and couplings
    of *calibration*: -4.5 to -7.3 kHz with the shipped ``transmon_17q``, below
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
    qudit_count: int,
    *,
    base_frequency_hz: float = 5.0e9,
    frequency_spread_hz: float = 100e6,
    anharmonicity_hz: float = -330e6,
    coupling_strength_hz: float = 3.0e6,
    t1_us: float = 25.0,
    t2_star_us: float = 15.0,
    topology: str = "linear",
) -> DeviceDescription:
    """Build a generic transmon target with configurable parameters.

    Parameters
    ----------
    qudit_count : int
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
        with side length = ceil(sqrt(qudit_count))).
    """
    if qudit_count < 1:
        raise InvalidValueError(f"qudit_count must be >= 1, got {qudit_count}")

    import math as _math

    qubits = {}
    for i in range(qudit_count):
        qubits[i] = Qubit(
            index=i,
            frequency_hz=base_frequency_hz + i * frequency_spread_hz / max(qudit_count - 1, 1),
            anharmonicity_hz=anharmonicity_hz,
            t1_us=t1_us,
            t2_star_us=t2_star_us,
            label=f"Q{i}",
        )

    edges: list[tuple[int, int]] = []
    if topology == "linear":
        edges = [(i, i + 1) for i in range(qudit_count - 1)]
    elif topology == "ring":
        edges = [(i, (i + 1) % qudit_count) for i in range(qudit_count)]
    elif topology == "grid":
        side = _math.ceil(_math.sqrt(qudit_count))
        for i in range(qudit_count):
            r, c = divmod(i, side)
            if c + 1 < side and i + 1 < qudit_count:
                edges.append((i, i + 1))
            if r + 1 < side and i + side < qudit_count:
                edges.append((i, i + side))
    else:
        raise InvalidValueError(f"Unknown topology {topology!r}. Use 'linear', 'ring', or 'grid'.")

    couplings = [Coupling((a, b), coupling_strength_hz) for a, b in edges]

    return DeviceDescription(
        name=f"transmon_generic_{qudit_count}q_{topology}",
        qubits=qubits,
        couplings=couplings,
    )
