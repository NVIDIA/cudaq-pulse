# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a quantum system's answers for tunings take of each shipped
calibration, without reading its file: the key of the charge coupling used,
g_c in Hz, and each calibrated circuit (E_C, E_J,max, d, idle flux) by
label. Read only when a tuning or an interaction is first asked for
(``components``); a file of another digest is read instead."""

from __future__ import annotations

# What the tuning components of a quantum system take of each shipped
# calibration (``components._tuning_values``).
SHIPPED_TUNINGS: dict[
    str, tuple[str, float, tuple[tuple[str, tuple[float, float, float, float]], ...]]
] = {
    "transmon_surface17": (
        "calibration_A",
        3121436.6758583123,
        (
            ("D1", (165216557.47241178, 26006261835.1497, 0.4790249377953073, 0.5)),
            ("D2", (165710764.85113448, 26184818333.62023, 0.5002411799971491, 0.5)),
            ("D3", (164720159.92262083, 27043950781.09888, 0.4772585566906421, 0.5)),
            ("D4", (165183889.27103812, 26173117852.00784, 0.47441924515102624, 0.5)),
            ("D5", (166861030.3026541, 23964730281.133026, 0.5177024893456946, 0.5)),
            ("D6", (164515295.15687475, 27629761284.941208, 0.42092025748848755, 0.5)),
            ("D7", (163552760.82509235, 27820314077.236103, 0.49077143227093567, 0.5)),
            ("D8", (164901076.19609588, 27996509622.393238, 0.4697424736344745, 0.5)),
            ("D9", (163906184.02462876, 27633422831.61866, 0.5135842339464568, 0.5)),
            ("X1", (159613382.45254478, 30699109344.85372, 0.5405667172300653, 0.0)),
            ("X2", (162743779.50086534, 28136487016.09633, 0.5059320014993117, 0.0)),
            ("X3", (159487288.63239884, 29990982520.554996, 0.5753753076408374, 0.0)),
            ("X4", (159533034.05239087, 30244955094.689304, 0.5674424966142803, 0.0)),
            ("Z1", (153782127.4804761, 34191236366.09627, 0.5453118595165105, 0.0)),
            ("Z2", (158005239.71346924, 31942033390.610847, 0.5397644779659176, 0.0)),
            ("Z3", (160249989.91847154, 29222905185.244343, 0.5623087183155954, 0.0)),
            ("Z4", (159512752.0687674, 30131946022.267715, 0.4939749088486023, 0.0)),
        ),
    ),
}
