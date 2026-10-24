# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Published parameters of the 17-qubit distance-three surface-code device of

    S. Krinner, N. Lacroix et al., "Realizing repeated quantum error
    correction in a distance-three surface code", Nature 605, 669 (2022),
    arXiv:2112.03708.

Every value is cited by table, figure or section, and by page of the arXiv
v1 PDF (which includes the Methods and the Supplementary Information as
Appendices A-L). Values read from figures say so: the per-pair two-qubit
gate errors are the figure's printed labels, and the interaction
frequencies are digitized from line positions (about +-10 MHz).

Only published values are here; per-pair couplings, pulse amplitudes and
anything else that has to be fitted belong in a calibration, not in this
module. Not published, and not modeled yet: the flux-tuning curves (only
one sweet spot per qubit is given), the Purcell-filter parameters, the
feed-line membership (Fig. 1b, drawing only) and the crosstalk matrices
(Figs. 10-12, color maps only).
"""

from __future__ import annotations

from dataclasses import dataclass

CITATION = (
    "S. Krinner, N. Lacroix et al., 'Realizing repeated quantum error correction "
    "in a distance-three surface code', Nature 605, 669 (2022). arXiv:2112.03708"
)

# Qudit index = position in this tuple: the column order of Tab. I (p. 10).
LABELS = (
    "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9",
    "X1", "X2", "X3", "X4",
    "Z1", "Z2", "Z3", "Z4",
)  # fmt: skip

# Tab. I (p. 10), one row per parameter, columns in LABELS order. Rows marked
# (a) were measured in a different cooldown of the same device.
_IDLE_FREQUENCY_GHZ = (
    3.885, 3.994, 3.952, 3.878, 3.895, 3.74, 4.056, 3.993, 4.143,
    6.097, 5.885, 6.022, 6.049, 6.328, 6.192, 5.956, 6.037,
)  # fmt: skip
_ANHARMONICITY_MHZ = (
    -184, -184, -183, -184, -186, -184, -181, -183, -181,
    -170, -174, -170, -170, -163, -168, -171, -170,
)  # fmt: skip
_T1_US = (
    29.1, 33.0, 65.5, 59.3, 32.2, 60.2, 36.0, 33.1, 25.9,
    12.4, 17.4, 18.5, 12.6, 17.0, 42.7, 29.7, 27.5,
)  # fmt: skip
_T2_RAMSEY_US = (
    35.0, 13.5, 33.9, 74.3, 46.1, 78.1, 74.8, 41.4, 31.7,
    5.6, 14.3, 20.7, 28.9, 29.3, 33.1, 48.0, 28.9,
)  # fmt: skip
_T2_ECHO_US = (
    46.2, 52.2, 49.3, 75.5, 56.1, 89.3, 72.5, 59.1, 36.3,
    15.8, 33.0, 16.1, 33.1, 31.5, 26.0, 53.6, 53.6,
)  # fmt: skip
_RB_ERROR_1Q_PCT = (  # (a)
    0.06, 0.07, 0.04, 0.04, 0.06, 0.06, 0.04, 0.08, 0.06,
    0.16, 0.13, 0.17, 0.14, 0.1, 0.09, 0.07, 0.16,
)  # fmt: skip
_READOUT_FREQUENCY_GHZ = (
    6.769, 6.979, 6.88, 7.12, 7.18, 7.032, 6.91, 7.075, 6.868,
    7.372, 7.554, 7.258, 7.461, 7.316, 7.502, 7.2, 7.412,
)  # fmt: skip
_READOUT_QUBIT_FREQUENCY_GHZ = (
    5.321, 4.75, 5.275, 4.25, 4.42, 5.13, 4.395, 3.993, 5.0,
    5.9, 5.885, 6.022, 6.049, 6.328, 6.191, 5.956, 5.687,
)  # fmt: skip
_DISPERSIVE_SHIFT_MHZ = (  # (a)
    -2.0, -2.2, -1.9, -1.6, -1.5, -1.6, -2.3, -2.0, -2.4,
    -2.8, -1.9, -3.2, -2.6, -4.7, -2.9, -3.2, -2.8,
)  # fmt: skip
_READOUT_DISPERSIVE_SHIFT_MHZ = (
    -7.4, -3.9, -6.0, -2.0, -2.1, -4.7, -3.0, -2.0, -4.9,
    -2.1, -1.9, -3.2, -2.6, -4.7, -2.9, -3.2, -2.8,
)  # fmt: skip
_KAPPA_EFF_MHZ = (  # (a)
    6.3, 7.2, 8.3, 8.7, 3.5, 9.0, 8.6, 8.2, 8.9,
    15.1, 20.1, 13.0, 11.0, 11.2, 12.2, 14.3, 10.0,
)  # fmt: skip
_RESONATOR_COUPLING_MHZ = (  # (a)
    244, 269, 241, 241, 238, 244, 267, 265, 260,
    167, 168, 167, 168, 171, 170, 167, 171,
)  # fmt: skip
_READOUT_ERROR_2STATE_PCT = (
    0.7, 0.5, 0.5, 0.6, 0.8, 0.5, 2.3, 1.3, 0.5,
    2.7, 0.7, 0.8, 0.8, 1.3, 0.4, 0.4, 0.5,
)  # fmt: skip
_READOUT_ERROR_3STATE_PCT = (
    4.4, 1.0, 5.9, 2.1, 1.6, 1.1, 3.3, 2.1, 1.0,
    3.9, 2.0, 1.9, 1.6, 2.2, 1.2, 0.9, 1.1,
)  # fmt: skip
_THERMAL_POPULATION_PCT = (
    2.1, 0.7, 1.5, 3.8, 2.7, 2.2, 0.0, 0.8, 1.6,
    0.2, 0.2, 0.5, 0.3, 0.6, 0.5, 0.2, 0.9,
)  # fmt: skip

# The 24 auxiliary-data couplings: the supports of the stabilizers of
# Tab. II (p. 21), with no data-data or auxiliary-auxiliary coupling
# (Methods A, p. 8). Per edge: the CZ time step and the interaction
# frequency omega_int / 2 pi of Fig. 6c (p. 11; steps from the edge labels,
# omega_int digitized from the line positions, about +-10 MHz), and the
# two-qubit gate error from interleaved randomized benchmarking, Fig. 5
# (p. 9; the printed labels). Ordered by time step: steps 1-4 measure the Z
# stabilizers, 5-8 the X stabilizers (App. C2, p. 11).
_EDGES = (
    # auxiliary, data, step, omega_int / 2 pi (GHz), error (%)
    ("Z1", "D1", 1, 4.885, 1.2),
    ("Z2", "D5", 1, 4.676, 1.0),
    ("Z3", "D3", 1, 5.293, 3.4),
    ("Z1", "D4", 2, 4.785, 1.2),
    ("Z2", "D8", 2, 5.313, 1.2),
    ("Z3", "D5", 2, 4.320, 1.9),
    ("Z2", "D4", 3, 4.792, 1.3),
    ("Z3", "D2", 3, 5.249, 1.6),
    ("Z4", "D6", 3, 4.815, 1.8),
    ("Z2", "D7", 4, 4.317, 1.0),
    ("Z3", "D6", 4, 4.979, 1.2),
    ("Z4", "D9", 4, 5.446, 0.7),
    ("X2", "D1", 5, 5.423, 0.6),
    ("X3", "D5", 5, 4.487, 1.4),
    ("X4", "D7", 5, 4.336, 0.9),
    ("X2", "D2", 6, 5.495, 0.8),
    ("X3", "D6", 6, 4.931, 2.4),
    ("X4", "D8", 6, 4.354, 1.3),
    ("X1", "D3", 7, 5.197, 5.4),
    ("X2", "D5", 7, 4.891, 1.4),
    ("X3", "D9", 7, 5.598, 1.1),
    ("X1", "D2", 8, 4.693, 2.6),
    ("X2", "D4", 8, 5.276, 0.9),
    ("X3", "D8", 8, 4.560, 1.3),
)

# Methods A (p. 8): J / 2 pi ~ 7 MHz between auxiliary and data qubits at the
# interaction point, for the device as a whole; per-pair and idle values
# are not published.
COUPLING_HZ = 7.0e6

# Single-qubit gates, App. B (p. 9): DRAG pulses with a Gaussian envelope of
# sigma = 10 ns truncated at +-2 sigma, 40 ns long.
DRIVE_DURATION_NS = 40.0
DRIVE_SIGMA_NS = 10.0

# Two-qubit gates, App. B (p. 9): net-zero flux pulses of 68 ns on average,
# with 15 ns buffers before and after, 98 ns in total on average.
CZ_FLUX_PULSE_NS = 68.0
CZ_BUFFER_NS = 15.0
CZ_DURATION_NS = 98.0

# Readout, Methods B (p. 8) and App. F1 (p. 16): Gaussian-filtered (sigma =
# 10 ns) rectangular pulses of 200 ns on auxiliary and 300 ns on data qubits,
# integrated for 400 ns. Data qubits are read out flux-tuned to the qubit
# frequency during readout of Tab. I.
READOUT_SIGMA_NS = 10.0
READOUT_DURATION_NS = {"D": 300.0, "X": 200.0, "Z": 200.0}
READOUT_INTEGRATION_NS = 400.0

# Error correction cycle, abstract and Fig. 3a (p. 1, 5).
CYCLE_NS = 1100.0


@dataclass(frozen=True)
class QubitData:
    """One column of Tab. I. Frequencies in Hz, times in us, errors and
    populations as fractions. Data qubits idle at their lower flux sweet spot
    and auxiliary qubits at their upper one (Methods A, p. 8)."""

    label: str
    idle_frequency_hz: float
    anharmonicity_hz: float
    t1_us: float
    t2_ramsey_us: float
    t2_echo_us: float
    rb_error_1q: float
    readout_frequency_hz: float
    readout_qubit_frequency_hz: float
    dispersive_shift_hz: float
    readout_dispersive_shift_hz: float
    kappa_eff_hz: float
    resonator_coupling_hz: float
    readout_error_2state: float
    readout_error_3state: float
    thermal_population: float

    @property
    def sweet_spot(self) -> str:
        return "lower" if self.label.startswith("D") else "upper"


@dataclass(frozen=True)
class EdgeData:
    """One auxiliary-data coupling: its CZ time step, interaction frequency
    (Hz) and two-qubit gate error (fraction)."""

    auxiliary: str
    data: str
    cz_step: int
    interaction_frequency_hz: float
    cz_error: float


QUBITS = tuple(
    QubitData(
        label=label,
        idle_frequency_hz=_IDLE_FREQUENCY_GHZ[i] * 1e9,
        anharmonicity_hz=_ANHARMONICITY_MHZ[i] * 1e6,
        t1_us=_T1_US[i],
        t2_ramsey_us=_T2_RAMSEY_US[i],
        t2_echo_us=_T2_ECHO_US[i],
        rb_error_1q=_RB_ERROR_1Q_PCT[i] / 100,
        readout_frequency_hz=_READOUT_FREQUENCY_GHZ[i] * 1e9,
        readout_qubit_frequency_hz=_READOUT_QUBIT_FREQUENCY_GHZ[i] * 1e9,
        dispersive_shift_hz=_DISPERSIVE_SHIFT_MHZ[i] * 1e6,
        readout_dispersive_shift_hz=_READOUT_DISPERSIVE_SHIFT_MHZ[i] * 1e6,
        kappa_eff_hz=_KAPPA_EFF_MHZ[i] * 1e6,
        resonator_coupling_hz=_RESONATOR_COUPLING_MHZ[i] * 1e6,
        readout_error_2state=_READOUT_ERROR_2STATE_PCT[i] / 100,
        readout_error_3state=_READOUT_ERROR_3STATE_PCT[i] / 100,
        thermal_population=_THERMAL_POPULATION_PCT[i] / 100,
    )
    for i, label in enumerate(LABELS)
)

EDGES = tuple(
    EdgeData(auxiliary, data, step, frequency_ghz * 1e9, error_pct / 100)
    for auxiliary, data, step, frequency_ghz, error_pct in _EDGES
)

INDEX = {label: i for i, label in enumerate(LABELS)}
