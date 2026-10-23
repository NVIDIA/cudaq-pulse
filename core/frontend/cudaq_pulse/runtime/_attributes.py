# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Module attributes that carry a target's calibration into the lowering."""

from __future__ import annotations

import math
import struct
from typing import Any

_VALID_INTEGRATORS = frozenset(
    {
        "rk1",
        "rk2",
        "rk4",
        "magnus",
        "crank_nicolson",
    }
)


def _level_attributes(
    target: Any, qubits: tuple[int, ...], n_qubits: int, levels: int | dict[int, int]
) -> tuple[tuple[int, ...], str]:
    """Levels per qudit, and the module attributes that model them."""
    dims = tuple(
        (levels.get(q, 2) if isinstance(levels, dict) else int(levels)) for q in range(n_qubits)
    )
    if any(d < 2 for d in dims):
        raise ValueError("every qudit needs at least two levels")
    attrs = [f"pulse.levels = array<i64: {', '.join(map(str, dims))}>"]
    if any(d > 2 for d in dims):
        alphas = [0.0] * n_qubits
        for q in qubits:
            alphas[q] = float(target.qubits[q].anharmonicity_hz)
        attrs.append(f"pulse.anharmonicity_hz = {_f64_array(alphas)}")
    return dims, ", ".join(attrs)


def _f64_literal(value: float) -> str:
    value = float(value)
    if math.isfinite(value):
        return format(value, ".17e")
    return f"0x{struct.unpack('<Q', struct.pack('<d', value))[0]:016X}"


def _f64(value: float) -> str:
    return f"{_f64_literal(value)} : f64"


def _f64_array(values: list[float]) -> str:
    return "array<f64: " + ", ".join(_f64_literal(v) for v in values) + ">"


def _target_attributes(
    target: Any,
    qubits: tuple[int, ...],
    n_qubits: int,
    *,
    t_start: float,
    t_end: float,
    num_steps: int,
    integrator: str,
) -> str:
    """Module attributes, in MLIR syntax, that the cuDensityMat lowering reads."""
    t1_ns = [0.0] * n_qubits
    t2_ns = [0.0] * n_qubits
    drive_scales = [1.0] * n_qubits
    for qi in qubits:
        qubit = target.qubits[qi]
        t1_ns[qi] = max(0.0, float(qubit.t1_us)) * 1.0e3
        t2_ns[qi] = max(0.0, float(qubit.t2_star_us)) * 1.0e3
        drive_scales[qi] = target.drive_amplitude_scale(qi)
    attrs = [
        f"qop.n_qubits = {n_qubits} : i64",
        f"qop.t_start = {_f64(t_start)}",
        f"qop.t_end = {_f64(t_end)}",
        f"qop.num_steps = {int(num_steps)} : i64",
        f'qop.integrator = "{integrator}"',
        "pulse.t1_times = [" + ", ".join(_f64(v) for v in t1_ns) + "]",
        "pulse.t2_times = [" + ", ".join(_f64(v) for v in t2_ns) + "]",
        f"pulse.drive_scale_rad_per_ns = {_f64_array(drive_scales)}",
    ]
    active = set(qubits)
    for kind, entries, strength in (
        ("coupling", target.couplings, "coupling_strength_hz"),
        ("crosstalk", target.crosstalk, "static_zz_hz"),
    ):
        entries = [e for e in entries if e.qubit_a in active and e.qubit_b in active]
        if entries:
            pairs = ", ".join(f"{e.qubit_a}, {e.qubit_b}" for e in entries)
            attrs.append(f"pulse.{kind}_pairs = array<i64: {pairs}>")
            attrs.append(
                f"pulse.{kind}_strength_hz = " + _f64_array([getattr(e, strength) for e in entries])
            )
    return "{" + ", ".join(attrs) + "}"
