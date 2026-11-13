# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sign conventions: drive detuning, the 1-2 transition, cross resonance and
shift_phase."""

import math

import cudaq_pulse as pulse
import pytest

from tests.physics._sim import basis, cycles, device, population, run
from tests.physics.conftest import ref


@pytest.mark.parametrize("tone", ["plus", "minus"])
def test_ef_rabi_sign(tone):
    entry = ref("ef_rabi_sign")
    p = entry["params"]
    frequency = p["f_q_hz"] + (p["alpha_hz"] if tone == "plus" else -p["alpha_hz"])
    duration = cycles(p["T_ns"])

    @pulse.kernel
    def ef(q, f: float):
        d = get_drive_line(q)
        set_frequency(d.tone, f)
        drive(d, square(duration, p["A_rad_per_ns"]))

    dev = device([p["f_q_hz"]], [p["alpha_hz"]])
    result = run(
        dev, ef, frequency, levels=3, initial_state=basis(3, 1), dt_ns=entry["tolerance"]["dt_ns"]
    )
    key = "tone_at_wq_plus_alpha_P2" if tone == "plus" else "tone_at_wq_minus_alpha_P2"
    assert result.density_matrix[2, 2].real == pytest.approx(
        entry["expected"][key], abs=entry["tolerance"]["abs"]
    )


@pytest.mark.parametrize("control", [0, 1])
def test_cr_target_excitation(control):
    entry = ref("cr_target_excitation")
    p = entry["params"]
    duration = cycles(p["T_ns"])

    @pulse.kernel
    def cr(qc, qt):
        dc = get_drive_line(qc)
        dt = get_drive_line(qt)
        set_frequency(dc.tone, p["ft_hz"])
        drive(dc, square(duration, p["drive_rad_per_ns"]))
        wait(dt, duration)

    dev = device([p["fc_hz"], p["ft_hz"]], [p["alpha_c_hz"], p["alpha_t_hz"]], [(0, 1, p["g_hz"])])
    result = run(
        dev,
        cr,
        levels=3,
        initial_state=basis(3, control, 0),
        dt_ns=entry["tolerance"]["dt_ns"],
    )
    excited = population(result.density_matrix, [3, 3], q1=1)
    assert excited == pytest.approx(
        entry["expected"][f"control{control}_P_target1"], abs=entry["tolerance"]["abs"]
    )


@pytest.mark.parametrize(("theta", "x"), [(0.5 * math.pi, -1.0), (-0.5 * math.pi, 1.0)])
@pytest.mark.parametrize(
    "passes", [(), ("verify", "fold_constants", "remove_unused", "fold_phases")]
)
def test_shift_phase_is_a_mckay_virtual_z(theta, x, passes):
    """shift_phase(theta) then an X90 is an X90 about the axis at -theta."""
    rate = math.pi / 2 / 20.0  # X90 in 20 ns

    @pulse.kernel
    def x90_after_z(q):
        d = get_drive_line(q)
        shift_phase(d.tone, theta)
        drive(d, square(cycles(20.0), rate))

    dev = device([5.0e9], [-200e6])
    lowered = pulse.compile(x90_after_z, passes=passes).lower(dev.qudits[0])
    rho = pulse.simulate(lowered, dt_ns=0.01, levels=2).density_matrix
    assert 2 * rho[0, 1].real == pytest.approx(x, abs=1e-6)
    assert abs(rho[0, 1].imag) < 1e-6
