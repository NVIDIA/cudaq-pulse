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

from .._errors import InvalidValueError, UnsupportedError
from ..targets.base import _DENSITY_DENSITY, _EXCHANGE, _coupling_kind

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
    target: Any, qudits: tuple[int, ...], levels: int | dict[int, int] | None
) -> tuple[tuple[int, ...], str]:
    """Levels of each simulated qudit, in the order of *qudits*, and the
    module attributes that model them (indexed by device qudit).

    A qudit has the levels its target's model declares for it
    (``model.levels_of``) unless *levels* sets them: an int for every
    qudit, a dict by device qudit index for those it names."""
    model = target.model
    if levels is None:
        dims = tuple(model.levels_of(q) for q in qudits)
    elif isinstance(levels, dict):
        dims = tuple(levels[q] if q in levels else model.levels_of(q) for q in qudits)
    else:
        dims = (int(levels),) * len(qudits)
    if any(d < 2 for d in dims):
        raise InvalidValueError("every qudit needs at least two levels")
    # Qudits the program does not use are not simulated, and keep the levels
    # they would have.
    extent = _extent(qudits)
    if levels is None or isinstance(levels, dict):
        by_index = [model.levels_of(q) for q in range(extent)]
    else:
        by_index = [int(levels)] * extent
    for q, d in zip(qudits, dims, strict=False):
        by_index[q] = d
    attrs = [f"pulse.levels = array<i64: {', '.join(map(str, by_index))}>"]
    if any(d > 2 for d in dims):
        alphas = [0.0] * _extent(qudits)
        for q in qudits:
            alphas[q] = float(target.qubits[q].anharmonicity_hz)
        attrs.append(f"pulse.anharmonicity_hz = {_f64_array(alphas)}")
    return dims, ", ".join(attrs)


def _extent(qudits: tuple[int, ...]) -> int:
    """The length of an attribute indexed by device qudit."""
    return max(qudits) + 1


def _f64_literal(value: float) -> str:
    value = float(value)
    if math.isfinite(value):
        return format(value, ".17e")
    return f"0x{struct.unpack('<Q', struct.pack('<d', value))[0]:016X}"


def _f64(value: float) -> str:
    return f"{_f64_literal(value)} : f64"


def _f64_array(values: list[float]) -> str:
    return "array<f64: " + ", ".join(_f64_literal(v) for v in values) + ">"


def _density_density(target: Any, active: set[int]) -> list[Any]:
    """*target*'s density-density couplings; any other kind that is neither
    that nor exchange, between *active* qudits, is refused."""
    found = []
    for c in target.couplings:
        kind = _coupling_kind(c)
        if kind == _DENSITY_DENSITY:
            found.append(c)
        elif kind != _EXCHANGE and c.qubit_a in active and c.qubit_b in active:
            raise UnsupportedError(
                f"the coupling of qudits {c.qubit_a} and {c.qubit_b} is a {kind} "
                "coupling, which the simulator does not model; it models "
                f"{_EXCHANGE} and {_DENSITY_DENSITY} couplings"
            )
    return found


def _target_attributes(
    target: Any,
    qubits: tuple[int, ...],
    *,
    t_start: float,
    t_end: float,
    num_steps: int,
    integrator: str,
    dephasing: str = "t2_star",
) -> str:
    """Module attributes, in MLIR syntax, that the cuDensityMat lowering reads.

    *qubits* are the simulated qudits, one mode each in this order; the
    others are left out, along with their couplings. Qudits dephase at their
    Ramsey T2* (*dephasing* ``"t2_star"``) or their Hahn-echo T2
    (``"t2_echo"``).
    """
    n_qubits = _extent(qubits)
    t1_ns = [0.0] * n_qubits
    t2_ns = [0.0] * n_qubits
    thermal = [0.0] * n_qubits
    drive_scales = [1.0] * n_qubits
    for qi in qubits:
        qubit = target.qubits[qi]
        t2_us = qubit.t2_star_us if dephasing == "t2_star" else qubit.t2_echo_us
        if t2_us is None:
            raise InvalidValueError(f"qudit {qi} has no Hahn-echo T2 to dephase at")
        t1_ns[qi] = max(0.0, float(qubit.t1_us)) * 1.0e3
        t2_ns[qi] = max(0.0, float(t2_us)) * 1.0e3
        thermal[qi] = float(qubit.thermal_population)
        drive_scales[qi] = target.drive_amplitude_scale(qi)
    attrs = [
        f"qop.n_qubits = {len(qubits)} : i64",
        f"pulse.modes = array<i64: {', '.join(map(str, qubits))}>",
        f"qop.t_start = {_f64(t_start)}",
        f"qop.t_end = {_f64(t_end)}",
        f"qop.num_steps = {int(num_steps)} : i64",
        f'qop.integrator = "{integrator}"',
        "pulse.t1_times = [" + ", ".join(_f64(v) for v in t1_ns) + "]",
        "pulse.t2_times = [" + ", ".join(_f64(v) for v in t2_ns) + "]",
        f"pulse.drive_scale_rad_per_ns = {_f64_array(drive_scales)}",
    ]
    if any(thermal):
        attrs.append(f"pulse.thermal_populations = {_f64_array(thermal)}")
    active = set(qubits)
    exchange = target._exchange_view
    groups: tuple[tuple[str, Any, str], ...] = (
        ("coupling", exchange, "coupling_strength_hz"),
        ("crosstalk", target.crosstalk, "static_zz_hz"),
    )
    if exchange is not target.couplings:
        groups += (("density_density", _density_density(target, active), "coupling_strength_hz"),)
    for name, entries, strength in groups:
        entries = [e for e in entries if e.qubit_a in active and e.qubit_b in active]
        if entries:
            pairs = ", ".join(f"{e.qubit_a}, {e.qubit_b}" for e in entries)
            attrs.append(f"pulse.{name}_pairs = array<i64: {pairs}>")
            attrs.append(
                f"pulse.{name}_strength_hz = " + _f64_array([getattr(e, strength) for e in entries])
            )
    return "{" + ", ".join(attrs) + "}"
