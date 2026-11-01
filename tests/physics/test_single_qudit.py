# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""One driven qudit: resonant and detuned Rabi, Ramsey and Hahn echo."""

import cudaq_pulse as pulse
import pytest

from tests.physics._sim import cycles, device, run
from tests.physics.conftest import ref


@pytest.mark.parametrize("case", range(3))
def test_rabi_resonant(case):
    entry = ref("rabi_resonant")
    p = entry["params"]
    amplitude, duration = p["cases"][case]["A"], cycles(p["cases"][case]["T_ns"])

    @pulse.kernel
    def rabi(q):
        d = get_drive_line(q)
        drive(d, square(duration, amplitude))

    result = run(device([p["f_q_hz"]], [0.0]), rabi, dt_ns=entry["tolerance"]["dt_ns"])
    assert result.density_matrix[1, 1].real == pytest.approx(
        entry["expected"][case], abs=entry["tolerance"]["abs"]
    )


@pytest.mark.parametrize("case", range(4))
def test_rabi_detuned(case):
    entry = ref("rabi_detuned")
    p = entry["params"]
    frequency = p["f_q_hz"] + p["delta_MHz"][case] * 1e6
    duration = cycles(p["T_ns"])

    @pulse.kernel
    def rabi(q, f: float):
        d = get_drive_line(q)
        set_frequency(d.tone, f)
        drive(d, square(duration, p["A_rad_per_ns"]))

    result = run(device([p["f_q_hz"]], [0.0]), rabi, frequency, dt_ns=entry["tolerance"]["dt_ns"])
    assert result.density_matrix[1, 1].real == pytest.approx(
        entry["expected"][case], abs=entry["tolerance"]["abs"]
    )


@pytest.mark.parametrize(
    "entry_id",
    [
        "ramsey_echo_delta1.0MHz_tau100",
        "ramsey_echo_delta1.0MHz_tau250",
        "ramsey_echo_delta2.5MHz_tau100",
        "ramsey_echo_delta2.5MHz_tau250",
    ],
)
@pytest.mark.parametrize("sequence", ["ramsey", "echo"])
def test_ramsey_and_echo_keep_the_tone_phase_across_waits(entry_id, sequence):
    """X90 - tau - X90 and X90 - tau - X180 - tau - X90 with a detuned tone:
    the tone keeps advancing while the line waits."""
    entry = ref(entry_id)
    p = entry["params"]
    frequency = p["f_q_hz"] + p["delta_MHz"] * 1e6
    amplitude, tau = p["A_rad_per_ns"], cycles(p["tau_ns"])
    x90, x180 = cycles(20.0), cycles(40.0)

    @pulse.kernel
    def ramsey(q, f: float):
        d = get_drive_line(q)
        set_frequency(d.tone, f)
        drive(d, square(x90, amplitude))
        wait(d, tau)
        drive(d, square(x90, amplitude))

    @pulse.kernel
    def echo(q, f: float):
        d = get_drive_line(q)
        set_frequency(d.tone, f)
        drive(d, square(x90, amplitude))
        wait(d, tau)
        drive(d, square(x180, amplitude))
        wait(d, tau)
        drive(d, square(x90, amplitude))

    kernel = ramsey if sequence == "ramsey" else echo
    result = run(
        device([p["f_q_hz"]], [-200e6]), kernel, frequency, dt_ns=entry["tolerance"]["dt_ns"]
    )
    assert result.density_matrix[1, 1].real == pytest.approx(
        entry["expected"][f"{sequence}_P1"], abs=entry["tolerance"]["abs"]
    )
