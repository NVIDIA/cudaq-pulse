# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wait-length sweeps simulate as their static specializations.

T1, Ramsey and Hahn-echo sweeps run once with ``tau: float`` (one compiled
module) and once with ``tau`` static (one module per value); the density
matrices must agree, on the CPU and on the GPU path (its cuDensityMat run
replaced by the CPU integrator).
"""

from __future__ import annotations

import math

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.runtime import _jit as jit
from cudaq_pulse.runtime import simulate as simulate_module
from cudaq_pulse.targets import DeviceDescription, MicrowaveDrive, Qubit

from tests.runtime.test_gpu_backend_logic import _CpuStandIn

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")

# A square pulse of 40 cycles (20 ns) at 1 rad/ns per unit amplitude.
_PI = math.pi / 20.0
_HALF_PI = math.pi / 40.0
_DETUNING_HZ = 10.0e6
_T1_US = 0.2
_DT_NS = 0.25
TOLERANCE = 1e-12

_TARGET = DeviceDescription(
    name="wait-sweep",
    qubits={
        0: Qubit(
            index=0,
            frequency_hz=5.0e9,
            anharmonicity_hz=-200.0e6,
            t1_us=_T1_US,
            t2_star_us=0.1,
            t2_echo_us=0.15,
            drive=MicrowaveDrive(amplitude_scale_rad_per_ns=1.0),
        )
    },
)
Q = pulse.device(_TARGET).qudits


def t1(q, tau: float):
    d = get_drive_line(q)
    drive(d, square(40, _PI))
    wait(d, tau)


def ramsey(q, tau: float):
    # Virtual detuning: the second pulse's phase turns once per 200 cycles.
    d = get_drive_line(q)
    drive(d, square(40, _HALF_PI))
    wait(d, tau)
    shift_phase(d.tone, math.pi * tau / 100)
    drive(d, square(40, _HALF_PI))


def echo(q, tau: float):
    d = get_drive_line(q)
    set_frequency(d.tone, 5.0e9 + _DETUNING_HZ)
    drive(d, square(40, _HALF_PI))
    wait(d, tau / 2)
    drive(d, square(40, _PI))
    wait(d, tau / 2)
    drive(d, square(40, _HALF_PI))


def _static(fn):
    twin = pulse.kernel(fn)
    twin._promote(set(twin._split[1]))
    return twin


def _sweep(fn, taus, backend="cpu"):
    runtime = pulse.compile(pulse.kernel(fn))
    static = pulse.compile(_static(fn))
    got, want = [], []
    for tau in taus:
        got.append(_rho(runtime.lower(Q[0], tau), backend))
        want.append(_rho(static.lower(Q[0], tau), "cpu"))
    assert runtime.cache_info().compiled == 1
    assert static.cache_info().compiled == len(taus)
    return np.array(got), np.array(want)


def _rho(lowered, backend):
    return pulse.simulate(lowered, levels=2, dt_ns=_DT_NS, backend=backend).density_matrix


def _excited(rhos):
    return np.real(rhos[:, 1, 1])


TAUS = [0.0, 50.0, 100.0, 150.0, 250.0, 400.0]


def test_t1_decay_matches_the_specializations():
    got, want = _sweep(t1, TAUS)
    np.testing.assert_allclose(got, want, rtol=0, atol=TOLERANCE)
    # Relaxation from |1> over tau cycles of 0.5 ns.
    excited = _excited(got)
    expected = excited[0] * np.exp(-np.array(TAUS) * 0.5e-3 / _T1_US)
    np.testing.assert_allclose(excited, expected, atol=2e-3)


def test_ramsey_fringes_match_the_specializations():
    got, want = _sweep(ramsey, [0.0, 50.0, 100.0, 150.0, 200.0, 300.0])
    np.testing.assert_allclose(got, want, rtol=0, atol=TOLERANCE)
    # A fringe every 200 cycles, losing contrast at T2*.
    excited = _excited(got)
    assert excited[0] > 0.8 and excited[2] < 0.3 and excited[4] > excited[2] + 0.2, excited
    assert excited[5] < excited[4]


def test_an_echo_matches_the_specializations():
    got, want = _sweep(echo, [0.0, 50.0, 100.0, 200.0, 400.0])
    np.testing.assert_allclose(got, want, rtol=0, atol=TOLERANCE)


def test_the_gpu_path_simulates_the_bound_module(monkeypatch):
    monkeypatch.setattr(jit, "JITCompiler", lambda: None)
    monkeypatch.setattr(simulate_module, "_GpuBackend", _CpuStandIn)
    got, want = _sweep(ramsey, [0.0, 75.0, 300.0], backend="gpu")
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-10)


def test_a_call_with_an_odd_echo_rounds_each_half_up():
    @pulse.kernel
    def rounded(q):
        d = get_drive_line(q)
        set_frequency(d.tone, 5.0e9 + _DETUNING_HZ)
        drive(d, square(40, _HALF_PI))
        wait(d, 51)
        drive(d, square(40, _PI))
        wait(d, 51)
        drive(d, square(40, _HALF_PI))

    got = _rho(pulse.compile(pulse.kernel(echo)).lower(Q[0], 101.0), "cpu")
    want = _rho(rounded.lower(Q[0]), "cpu")
    np.testing.assert_allclose(got, want, rtol=0, atol=TOLERANCE)
