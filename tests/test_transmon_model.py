# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The transmon circuit model against golden vectors from an independent
implementation (transmon_golden.json): 137 checks."""

import json
from pathlib import Path

import pytest
from cudaq_pulse.quantum_systems.transmon_model import ChargeCoupledPair, TransmonCircuit

GOLDEN = json.loads((Path(__file__).parent / "transmon_golden.json").read_text())
TOL = GOLDEN["tolerances"]
CIRCUITS = {q: TransmonCircuit(**c) for q, c in GOLDEN["circuits"].items()}
PAIR = ChargeCoupledPair(CIRCUITS["D4"], CIRCUITS["X2"], GOLDEN["g_c_hz"])


def _checks():
    """(name, compute, expected, absolute tolerance) for every golden value."""
    out = []
    rel = TOL["energy_rel"]
    for s in GOLDEN["single"]:
        q, flux = CIRCUITS[s["label"]], s["flux"]
        tag = f"{s['label']}@{flux}"
        n = len(s["levels_hz"])
        for k, want in enumerate(s["levels_hz"]):
            out.append(
                (
                    f"{tag} E{k}",
                    lambda q=q, f=flux, k=k, n=n: q.eigensystem(f, n)[0][k],
                    want,
                    rel * max(abs(want), 1.0),
                )
            )
        out.append((f"{tag} f01", lambda q=q, f=flux: q.f01_hz(f), s["f01_hz"], rel * s["f01_hz"]))
        out.append(
            (
                f"{tag} alpha",
                lambda q=q, f=flux: q.anharmonicity_hz(f),
                s["anharmonicity_hz"],
                rel * s["f01_hz"],
            )
        )
        for key, (i, j) in (("n01", (0, 1)), ("n12", (1, 2)), ("n23", (2, 3))):
            out.append(
                (
                    f"{tag} {key}",
                    lambda q=q, f=flux, i=i, j=j: q.charge_matrix(f)[i, j],
                    s[key],
                    TOL["matrix_element_abs"],
                )
            )
    for r in GOLDEN["flux_for_f01"]:
        q = CIRCUITS[r["label"]]
        out.append(
            (
                f"flux_for_f01 {r['label']}",
                lambda q=q, f=r["f01_hz"]: q.flux_for_f01(f),
                r["flux"],
                TOL["flux_abs"],
            )
        )
    for p in GOLDEN["pairs"]:
        fa, fb = p["flux_a"], p["flux_b"]
        tag = f"D4-X2@({fa:.4f},{fb:.4f})"
        out.append(
            (
                f"{tag} zz",
                lambda fa=fa, fb=fb: PAIR.idle_zz_hz(fa, fb),
                p["zz_hz"],
                TOL["zz_abs_hz"],
            )
        )
        out.append(
            (
                f"{tag} j_qq",
                lambda fa=fa, fb=fb: PAIR.j_qq_hz(fa, fb),
                p["j_qq_hz"],
                rel * 1e3 * p["j_qq_hz"],
            )
        )
        out.append(
            (
                f"{tag} gap_11_02",
                lambda fa=fa, fb=fb: PAIR.gap_11_02_hz(fa, fb),
                p["gap_11_02_hz"],
                TOL["zz_abs_hz"],
            )
        )
    # Published values, recomputed with this model.
    x2, d4 = CIRCUITS["X2"], CIRCUITS["D4"]
    compute = {
        "X2_f01_at_0_vs_tab1_hz": lambda: x2.f01_hz(0.0),
        "X2_f01_at_half_vs_fig1c_hz": lambda: x2.f01_hz(0.5),
        "X2_f01_at_half_vs_hellings_hz": lambda: x2.f01_hz(0.5),
        "X2_flux_for_minus_738MHz_vs_hellings_quarter": lambda: x2.flux_for_f01(5.887e9 - 0.738e9),
        "D4_f01_at_half_vs_tab1_hz": lambda: d4.f01_hz(0.5),
        "D4_X2_idle_zz_below_8kHz": lambda: abs(PAIR.idle_zz_hz(0.5, 0.0)),
    }
    for name, gate in GOLDEN["physics_gates"].items():
        out.append((f"gate {name}", compute[name], gate["reference"], gate["abs_tol"]))
    return out


CHECKS = _checks()


def test_there_are_137_checks():
    assert len(CHECKS) == 137


@pytest.mark.parametrize(
    ("name", "compute", "expected", "tolerance"), CHECKS, ids=[c[0] for c in CHECKS]
)
def test_golden(name, compute, expected, tolerance):
    assert abs(compute() - expected) <= tolerance, name


def test_j2_is_the_half_splitting_at_the_anticrossing():
    """J_2 at X2-D4's 4.4 GHz is 7.4 MHz for the golden coupling (it was
    fitted so), and the minimum of |E_11 - E_02| over the auxiliary's flux."""
    j2 = PAIR.j2_hz(4.4e9)
    assert j2 == pytest.approx(7.4e6, rel=1e-4)
    flux_d = CIRCUITS["D4"].flux_for_f01(4.4e9)
    center = CIRCUITS["X2"].flux_for_f12(4.4e9)
    splits = [PAIR.splitting_11_02_hz(flux_d, center + k * 1e-4) for k in range(-50, 51)]
    assert min(splits) / 2 == pytest.approx(j2, rel=1e-4)


def test_exchange_depends_on_the_operating_point():
    idle = PAIR.exchange_hz(CIRCUITS["D4"].f01_hz(0.5), CIRCUITS["X2"].f01_hz(0.0))
    assert idle == pytest.approx(PAIR.j_qq_hz(0.5, 0.0), rel=1e-9)
    # J_qq ~ n01_a n01_b grows with both frequencies.
    tuned = PAIR.exchange_hz(4.4e9, CIRCUITS["X2"].f01_hz(0.0))
    assert tuned > idle


def test_frequencies_outside_the_tuning_range_are_rejected():
    with pytest.raises(ValueError, match="outside the tuning range"):
        CIRCUITS["D4"].flux_for_f01(7e9)
