# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module attributes that carry a target's calibration into the lowering."""

from __future__ import annotations

import math
import struct
from itertools import compress
from operator import attrgetter
from typing import Any

from .._errors import InvalidValueError
from ..quantum_systems.base import _unmeasured

_QUDITS = attrgetter("qudits")

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
    module attributes that model them (indexed by quantum system qudit).

    A qudit has the levels its target's model declares for it
    (``model.levels_of``) unless *levels* sets them: an int for every
    qudit, a dict by quantum system qudit index for those it names."""
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
            alphas[q] = float(target.qudits[q].anharmonicity_hz)
        attrs.append(f"pulse.anharmonicity_hz = {_f64_array(alphas)}")
    return dims, ", ".join(attrs)


def _extent(qudits: tuple[int, ...]) -> int:
    """The length of an attribute indexed by quantum system qudit."""
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


def _target_attributes(
    target: Any,
    qudits: tuple[int, ...],
    *,
    integrator: str,
    dephasing: str = "t2_star",
) -> str:
    """Module attributes, in MLIR syntax and comma-separated, that the
    cuDensityMat lowering reads, but for the time window, which each interval
    sets.

    *qudits* are the simulated qudits, one mode each in this order; the
    others are left out, along with their couplings. Qudits dephase at their
    Ramsey T2* (*dephasing* ``"t2_star"``) or their Hahn-echo T2
    (``"t2_echo"``).
    """
    qudit_count = _extent(qudits)
    t1_ns = [0.0] * qudit_count
    t2_ns = [0.0] * qudit_count
    thermal = [0.0] * qudit_count
    drive_scales = [1.0] * qudit_count
    for qi in qudits:
        qudit = target.qudits[qi]
        t2_us = qudit.t2_star_us if dephasing == "t2_star" else qudit.t2_echo_us
        if t2_us is None:
            raise InvalidValueError(f"qudit {qi} has no Hahn-echo T2 to dephase at")
        # _f64_literal makes each a float.
        t1_ns[qi] = max(0.0, qudit.t1_us) * 1.0e3
        t2_ns[qi] = max(0.0, t2_us) * 1.0e3
        thermal[qi] = qudit.thermal_population
        drive_scales[qi] = target.drive_amplitude_scale(qi)
    attrs = [
        f"qop.qudit_count = {len(qudits)} : i64",
        f"pulse.modes = array<i64: {', '.join(map(str, qudits))}>",
        f'qop.integrator = "{integrator}"',
        "pulse.t1_times = [" + ", ".join(map(_f64, t1_ns)) + "]",
        "pulse.t2_times = [" + ", ".join(map(_f64, t2_ns)) + "]",
        f"pulse.drive_scale_rad_per_ns = {_f64_array(drive_scales)}",
    ]
    if any(thermal):
        attrs.append(f"pulse.thermal_populations = {_f64_array(thermal)}")
    active = set(qudits)
    # pulse.coupling_* is the exchange, the operator of the couplings of the
    # one model that simulates (quantum_systems.base._COUPLING_OPERATORS).
    couplings = target.couplings
    if couplings:
        couplings = list(compress(couplings, map(active.issuperset, map(_QUDITS, couplings))))
    if couplings:
        strengths = [c.strength_hz for c in couplings]
        if None in strengths:
            raise _unmeasured(
                target,
                [
                    i
                    for i, c in enumerate(target.couplings)
                    if c.strength_hz is None and active.issuperset(c.qudits)
                ],
                "simulating needs the strength of every coupling of the program's qudits",
                "give the strength in Hz, or leave those qudits out of the program",
            )
        pairs = ", ".join(f"{c.qudits[0]}, {c.qudits[1]}" for c in couplings)
        attrs.append(f"pulse.coupling_pairs = array<i64: {pairs}>")
        attrs.append("pulse.coupling_strength_hz = " + _f64_array(strengths))
    residual = target.residual_zz
    if residual:
        residual = list(compress(residual, map(active.issuperset, map(_QUDITS, residual))))
    if residual:
        terms = ", ".join(
            f"#pulse.static_term<zz, qudits = [{x.qudits[0]}, {x.qudits[1]}], "
            f"rate_hz = {_f64_literal(x.rate_hz)}>"
            for x in residual
        )
        attrs.append(f"pulse.static_terms = [{terms}]")
    return ", ".join(attrs)
