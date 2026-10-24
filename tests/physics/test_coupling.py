# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Two coupled qudits: exchange, the |11> <-> |20> chevron and static ZZ."""

import cmath
import math

import cudaq_pulse as pulse
import pytest

from tests.physics._sim import basis, cycles, device, population, run
from tests.physics.conftest import ref


def _idle(duration):
    @pulse.kernel
    def idle(q0, q1):
        d0, _ = get_drive_line(q0)
        d1, _ = get_drive_line(q1)
        wait(d0, duration)
        wait(d1, duration)

    return idle


@pytest.mark.parametrize(
    "entry_id",
    [
        "exchange_d2_detuning_0MHz",
        "exchange_d2_detuning_20MHz",
        "exchange_d3_detuning_0MHz",
        "exchange_d3_detuning_20MHz",
    ],
)
def test_exchange_from_10(entry_id):
    entry = ref(entry_id)
    p = entry["params"]
    levels = p["levels"]
    dev = device(p["f_hz"], p["alpha_hz"], [(0, 1, p["g_hz"])])
    result = run(
        dev,
        _idle(cycles(p["T_ns"])),
        levels=levels,
        initial_state=basis(levels, 1, 0),
        dt_ns=entry["tolerance"]["dt_ns"],
    )
    p01 = population(result.density_matrix, [levels, levels], q0=0, q1=1)
    assert p01 == pytest.approx(entry["expected"], abs=entry["tolerance"]["abs"])


@pytest.mark.parametrize(("side", "key"), [(1, "resonant_P20"), (-1, "mirrored_P20")])
def test_chevron_11_20_is_resonant_at_wa_plus_alpha_a(side, key):
    entry = ref("chevron_11_20")
    p = entry["params"]
    fb = p["f_a_hz"] + side * p["alpha_a_hz"]
    dev = device([p["f_a_hz"], fb], [p["alpha_a_hz"], p["alpha_b_hz"]], [(0, 1, p["g_hz"])])
    result = run(
        dev,
        _idle(cycles(p["T_ns"])),
        levels=3,
        initial_state=basis(3, 1, 1),
        dt_ns=entry["tolerance"]["dt_ns"],
    )
    p20 = population(result.density_matrix, [3, 3], q0=2, q1=0)
    assert p20 == pytest.approx(entry["expected"][key], abs=entry["tolerance"]["abs"])


def test_static_zz_of_detuned_qutrits():
    entry = ref("static_zz_detuned_qutrits")
    p, tolerance = entry["params"], entry["tolerance"]
    dev = device([p["fa_hz"], p["fb_hz"]], [p["alpha_a_hz"], p["alpha_b_hz"]], [(0, 1, p["g_hz"])])
    phases = []
    for k in (0, 1):
        psi = (basis(3, 0, k) + basis(3, 1, k)) / math.sqrt(2.0)
        rho = run(
            dev,
            _idle(cycles(p["T_ns"])),
            levels=3,
            initial_state=psi,
            dt_ns=tolerance["dt_ns"],
        ).density_matrix
        phases.append(cmath.phase(rho[3 * k, 1 + 3 * k]))
    conditional = (phases[1] - phases[0] + math.pi) % (2 * math.pi) - math.pi
    assert conditional == pytest.approx(
        entry["expected"]["conditional_phase_400ns_rad"], abs=tolerance["conditional_phase_abs_rad"]
    )
    zeta_mhz = conditional / (2 * math.pi * p["T_ns"]) * 1e3
    assert zeta_mhz == pytest.approx(entry["expected"]["zeta_PT_MHz"], rel=tolerance["zeta_PT_rel"])
