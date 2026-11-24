# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pulses on coupling lines, simulated, against a reference.

The REFERENCE is a small numpy model written here, independent of the
simulator and of the components' own maps: the transmon circuit of
``transmon_model.TransmonCircuit`` for a flux-tuned qudit (not
``FluxTuning``'s own spectrum), a linear interpolation of the declared table
for an exchange control, and ``expm`` of the Hamiltonian in the qudits'
frames for the evolution (by its eigenvectors).
"""

from __future__ import annotations

import math

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.quantum_systems import Coupling, PhysicalQudit
from cudaq_pulse.quantum_systems.transmon_model import TransmonCircuit
from cudaq_pulse.quantum_systems.tuning import (
    ExchangeControl,
    ExponentialSettling,
    FluxTuning,
)

_EC, _EJ, _D, _IDLE = 1.65e8, 2.6e10, 0.47, 0.5
_TUNING = FluxTuning(ec_hz=_EC, ej_max_hz=_EJ, asymmetry=_D, idle_flux=_IDLE)
_CIRCUIT = TransmonCircuit(_EC, _EJ, _D)
_F0 = _CIRCUIT.f01_hz(_IDLE)
_A0 = _CIRCUIT.anharmonicity_hz(_IDLE)
_EXCHANGE = ExchangeControl(
    unit="volt", amplitudes=(-0.2, 0.0, 0.2), exchange_hz=(1.0e6, 1.0e6, 21.0e6)
)


def _evolved(h, t):
    """exp(-i h t) of a Hermitian *h*."""
    energies, vectors = np.linalg.eigh(h)
    return (vectors * np.exp(-1j * energies * t)) @ vectors.conj().T


def _tuned(tuning=_TUNING):
    return pulse.quantum_system(
        name="tuned",
        qudits={0: PhysicalQudit(0, _F0, 0.0, 0.0, anharmonicity_hz=_A0)},
        tunings={0: tuning},
    )


def _pair(detuning_hz=0.0, control=_EXCHANGE):
    return pulse.quantum_system(
        name="pair",
        qudits={
            i: PhysicalQudit(i, f, 0.0, 0.0, anharmonicity_hz=-2.0e8)
            for i, f in enumerate((5.0e9 + detuning_hz, 5.0e9))
        },
        couplings=[Coupling((0, 1), strength_hz=1.0e6)],
        controls={(0, 1): control},
    )


@pulse.kernel
def detune(q, amplitude, length):
    get_drive_line(q)
    drive(get_coupling_line(q), square(length, amplitude))


@pulse.kernel
def detune_by(q, amplitude: float, length: int):
    get_drive_line(q)
    drive(get_coupling_line(q), square(length, amplitude))


@pulse.kernel
def ramp(q, amplitude, length):
    get_drive_line(q)
    drive(get_coupling_line(q), gaussian(length, amplitude, length / 5))


@pulse.kernel
def exchange(a, b, amplitude, length):
    get_drive_line(a)
    get_drive_line(b)
    drive(get_coupling_line(a, b), square(length, amplitude))


def _reference_shifts(amplitude):
    """The qudit's frequency and anharmonicity shifts (Hz) at a flux offset,
    from the reference circuit."""
    flux = _IDLE + amplitude
    return _CIRCUIT.f01_hz(flux) - _F0, _CIRCUIT.anharmonicity_hz(flux) - _A0


def _reference_qutrit(amplitudes, dt_ns, psi):
    """The state after the qudit's levels are shifted by each of
    *amplitudes* for *dt_ns*, in its frame at the idle frequency: there
    its second level keeps the idle anharmonicity."""
    n = np.diag([0.0, 1.0, 2.0])
    pairs = np.diag([0.0, 0.0, 1.0])  # n (n - 1) / 2
    for amplitude in amplitudes:
        shift, anharmonic = _reference_shifts(amplitude)
        h = 2.0 * math.pi * 1.0e-9 * (shift * n + (_A0 + anharmonic) * pairs)
        psi = _evolved(h, dt_ns) @ psi
    return np.outer(psi, psi.conj())


def test_a_square_pulse_on_a_tuning_line_detunes_the_qudit_by_the_circuits_shift():
    qs = _tuned()
    psi = np.array([1.0, 1.0, 1.0], dtype=complex) / math.sqrt(3.0)
    amplitude, length = -0.12, 80
    shift, _ = _reference_shifts(amplitude)
    assert 1.0e8 < abs(shift) < 2.0e9  # a real detuning, many turns in 40 ns
    duration_ns = length / qs.clock_ghz
    assert duration_ns == 40.0
    rho = pulse.simulate(
        detune, qs.qudits[0], amplitude, length, levels=3, initial_state=psi
    ).density_matrix
    reference = _reference_qutrit([amplitude], duration_ns, psi)
    np.testing.assert_allclose(rho, reference, atol=2e-6)
    # The same with the amplitude a float argument of the kernel.
    again = pulse.simulate(
        detune_by, qs.qudits[0], amplitude, length, levels=3, initial_state=psi
    ).density_matrix
    np.testing.assert_allclose(again, reference, atol=2e-6)
    # Two levels: the frequency shift alone.
    two = pulse.simulate(
        detune, qs.qudits[0], amplitude, length, levels=2, initial_state=psi[:2] * math.sqrt(1.5)
    ).density_matrix
    np.testing.assert_allclose(two, reference[:2, :2] * 1.5, atol=2e-6)


def test_a_shaped_pulse_on_a_tuning_line_follows_the_circuit_at_every_cycle():
    qs = _tuned()
    psi = np.array([1.0, 1.0, 0.0], dtype=complex) / math.sqrt(2.0)
    amplitude, length = -0.1, 120
    # The amplitude at the middle of each program cycle.
    centers = np.arange(length) + 0.5
    sigma = length / 5
    envelope = amplitude * np.exp(-0.5 * ((centers - length / 2) / sigma) ** 2)
    lowered = ramp.lower(qs.qudits[0], amplitude, length)
    rho = pulse.simulate(lowered, levels=3, initial_state=psi).density_matrix
    reference = _reference_qutrit(envelope, 1.0 / qs.clock_ghz, psi)
    np.testing.assert_allclose(rho, reference, atol=5e-5)


def test_no_pulse_and_a_zero_pulse_leave_the_state():
    qs = _tuned()
    psi = np.array([1.0, 1.0j, 0.0]) / math.sqrt(2.0)
    rho = pulse.simulate(detune, qs.qudits[0], 0.0, 40, levels=3, initial_state=psi).density_matrix
    np.testing.assert_allclose(rho, np.outer(psi, psi.conj()), atol=1e-9)


@pytest.mark.parametrize("detuning_hz", [0.0, 15.0e6])
def test_an_exchange_control_turns_the_pairs_exchange_on(detuning_hz):
    qs = _pair(detuning_hz)
    amplitude, length = 0.15, 20
    j_hz = float(np.interp(amplitude, _EXCHANGE.amplitudes, _EXCHANGE.exchange_hz))
    assert j_hz == pytest.approx(16.0e6)
    psi = np.zeros(4, dtype=complex)
    psi[1] = 1.0  # |1, 0>: qudit 0 the least significant digit
    rho = pulse.simulate(
        exchange,
        qs.qudits[0],
        qs.qudits[1],
        amplitude,
        length,
        levels=2,
        initial_state=psi,
        dt_ns=0.01,
    ).density_matrix
    # Reference: the one-excitation subspace {|1,0>, |0,1>} in the frame of
    # qudit 1, where qudit 0 is detuned by D: H = [[D, g], [g, 0]].
    g = 2.0 * math.pi * 1.0e-9 * j_hz
    d = 2.0 * math.pi * 1.0e-9 * detuning_hz
    u = _evolved(np.array([[d, g], [g, 0.0]]), length / qs.clock_ghz)
    swapped = abs(u[1, 0]) ** 2
    assert 0.05 < swapped < 0.999
    assert rho[2, 2].real == pytest.approx(swapped, abs=2e-6)
    assert rho[1, 1].real == pytest.approx(1.0 - swapped, abs=2e-6)


@pulse.kernel
def twice_in_a_loop(q, amplitude, length):
    get_drive_line(q)
    line = get_coupling_line(q)
    for _ in range(3):
        drive(line, square(length, amplitude))
        wait(line, 10)
    drive(line, square(length, -amplitude))


def test_pulses_in_a_loop_and_after_it_each_act_for_their_time():
    qs = _tuned()
    psi = np.array([1.0, 1.0, 1.0], dtype=complex) / math.sqrt(3.0)
    amplitude, length = -0.08, 24
    rho = pulse.simulate(
        twice_in_a_loop, qs.qudits[0], amplitude, length, levels=3, initial_state=psi
    ).density_matrix
    cycle = 1.0 / qs.clock_ghz
    played = ([amplitude] * length + [0.0] * 10) * 3 + [-amplitude] * length
    np.testing.assert_allclose(rho, _reference_qutrit(played, cycle, psi), atol=5e-6)


@pulse.kernel
def exchange_later(a, b, amplitude, length, idle):
    get_drive_line(a)
    get_drive_line(b)
    pair = get_coupling_line(a, b)
    wait(pair, idle)
    drive(pair, square(length, amplitude))


def test_an_exchange_pulse_that_starts_later_keeps_the_frames_phase():
    # The pair is detuned, so the exchange turns in the qudits' frames from
    # the program's start: the reference evolves both stretches in one frame.
    detuning_hz = 15.0e6
    qs = _pair(detuning_hz)
    amplitude, length, idle = 0.15, 20, 37
    psi = np.zeros(4, dtype=complex)
    psi[1] = psi[2] = 1.0 / math.sqrt(2.0)  # (|1,0> + |0,1>) / sqrt(2)
    rho = pulse.simulate(
        exchange_later,
        qs.qudits[0],
        qs.qudits[1],
        amplitude,
        length,
        idle,
        levels=2,
        initial_state=psi,
        dt_ns=0.01,
    ).density_matrix
    radians = 2.0 * math.pi * 1.0e-9
    d = radians * detuning_hz
    g0 = radians * 1.0e6
    g1 = radians * float(np.interp(amplitude, _EXCHANGE.amplitudes, _EXCHANGE.exchange_hz))
    t0, t1 = idle / qs.clock_ghz, length / qs.clock_ghz
    # In the lab-like frame of qudit 1: H = [[D, g], [g, 0]] on {|1,0>, |0,1>};
    # the simulator's frame turns |1,0> back by D t.
    state = np.array([1.0, 1.0], dtype=complex) / math.sqrt(2.0)
    state = _evolved(np.array([[d, g0], [g0, 0.0]]), t0) @ state
    state = _evolved(np.array([[d, g1], [g1, 0.0]]), t1) @ state
    state = state * np.array([np.exp(1j * d * (t0 + t1)), 1.0])
    reference = np.outer(state, state.conj())
    np.testing.assert_allclose(rho[1:3, 1:3], reference, atol=5e-6)


def test_what_is_not_simulated_yet_says_so():
    psi = np.array([1.0, 0.0, 0.0], dtype=complex)
    settling = _TUNING.replace(response=ExponentialSettling(amplitude=0.02, tau_ns=40.0))
    with pytest.raises(pulse.UnsupportedError, match="which the simulator does not apply yet"):
        pulse.simulate(detune, _tuned(settling).qudits[0], -0.1, 40, levels=3, initial_state=psi)
    # On a system that says nothing of the qudit's tuning.
    lowered = detune.lower(_tuned().qudits[0], -0.1, 40)
    bare = pulse.quantum_system(
        name="tuned", qudits={0: PhysicalQudit(0, _F0, 0.0, 0.0, anharmonicity_hz=_A0)}
    )
    with pytest.raises(pulse.NotFoundError, match="qudit 0 has no tuning in quantum system"):
        pulse.simulate(lowered, quantum_system=bare, levels=3, initial_state=psi)
    # A pair whose control the simulated system does not declare.
    lowered = exchange.lower(*_pair().qudits, 0.1, 20)
    plain = pulse.quantum_system(
        name="pair",
        qudits={i: PhysicalQudit(i, 5.0e9, 0.0, 0.0, anharmonicity_hz=-2.0e8) for i in range(2)},
        couplings=[Coupling((0, 1), strength_hz=1.0e6)],
    )
    with pytest.raises(pulse.NotFoundError, match="declares no control of the coupling"):
        pulse.simulate(lowered, quantum_system=plain, levels=2)
