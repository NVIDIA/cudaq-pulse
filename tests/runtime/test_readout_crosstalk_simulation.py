# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Readout crosstalk in simulation, against a reference that keeps the
readout resonator as a quantum mode: a damped oscillator, driven by the
readout, pulled by the qudit read out and coupled to the shifted qudit by a
cross-Kerr term."""

from __future__ import annotations

import math

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.quantum_systems import ReadoutCrosstalk
from cudaq_pulse.runtime._attributes import _read_out, _target_attributes

D5, Z2 = 4, 14
# Shifts per photon by Z2's level.
CHI = (-20e3, -26e3)
# The readout tone on the resonator as Z2's level 0 pulls it (chi < 0 pulls
# it up for level 0), so that the levels hold 4.90 and 2.58 photons.
TONE_SHIFT_HZ = 2.9e6


@pulse.kernel
def ramsey(d, z, flip: int, quarter: int):
    dl = get_drive_line(d)  # noqa: F821
    zl = get_drive_line(z)  # noqa: F821
    zr = get_readout_line(z)  # noqa: F821
    if flip == 1:
        drive(zl, zl.get_calibration(pulse.gates.X))  # noqa: F821
    sync(dl, zl, zr)  # noqa: F821
    drive(dl, dl.get_calibration(pulse.gates.X90))  # noqa: F821
    sync(dl, zr)  # noqa: F821
    shift_frequency(zr.tone, TONE_SHIFT_HZ)  # noqa: F821
    readout(zr, zr.get_calibration(pulse.gates.READOUT))  # noqa: F821
    sync(dl, zr)  # noqa: F821
    wait(dl, 600)  # noqa: F821
    for _ in range(quarter):
        shift_phase(dl.tone, math.pi / 2)  # noqa: F821
    drive(dl, dl.get_calibration(pulse.gates.X90))  # noqa: F821


def _quiet():
    base = pulse.quantum_system("transmon_surface17")
    quiet = {
        i: q.replace(t1_us=0, t2_star_us=0, t2_echo_us=0) for i, q in base.physical_qudits.items()
    }
    return base.replace(qudits=quiet)


def _phase(qs, flip):
    population = []
    for quarter in (0, 1):
        program = pulse.compile(ramsey, quantum_system=qs).lower(
            qs.qudits[D5], qs.qudits[Z2], flip, quarter
        )
        result = pulse.simulate(program, quantum_system=qs, dt_ns=0.02, levels=2)
        population.append(result.level_population(1, qudit=D5))
    return math.atan2(2 * population[1] - 1, 2 * population[0] - 1)


def _reference_phase(qs, level, chi_hz, fock=16, dt=0.025):
    # D5's coherence <0|rho|1> turns by the cross-Kerr term while Z2 sits in
    # *level*: the resonator, in the frame of the readout tone, with
    # H = -Delta a^dagger a + i epsilon (u a^dagger - u* a)
    #     + 2 pi chi_hz a^dagger a n_d
    # and the collapse operator sqrt(kappa) a.
    readout = qs.physical_qudits[Z2].readout
    chi = readout.dispersive_shift_hz
    kappa = 2 * math.pi * readout.linewidth_hz * 1e-9
    pulse_cal = qs.physical_qudits[Z2].get_calibration(pulse.gates.READOUT)
    duration, sigma = pulse_cal.duration_ns, pulse_cal.sigma_ns
    delta = 2 * math.pi * (TONE_SHIFT_HZ - (2 * level - 1) * chi) * 1e-9
    epsilon = math.sqrt(pulse_cal.photon_number) * abs(complex(kappa / 2, 2 * math.pi * chi * 1e-9))
    edge = 2 * sigma

    def envelope(t):
        # The gaussian_square the READOUT calibration plays, held over each
        # 0.5 ns cycle at its midpoint as the lowering samples it.
        if not 0 <= t < duration:
            return 0.0
        t = (math.floor(t / 0.5) + 0.5) * 0.5
        if t < edge:
            return math.exp(-0.5 * ((t - edge) / sigma) ** 2)
        if t > duration - edge:
            return math.exp(-0.5 * ((t - (duration - edge)) / sigma) ** 2)
        return 1.0

    a = np.diag(np.sqrt(np.arange(1, fock)), 1)
    n = a.T @ a
    eye_d = np.eye(2)
    number_d = np.diag([0.0, 1.0])
    big_a = np.kron(a, eye_d)
    static = np.kron(-delta * n, eye_d) + 2 * math.pi * chi_hz * 1e-9 * np.kron(n, number_d)
    drive_op = np.kron(a.T, eye_d)

    def derivative(rho, t):
        u = envelope(t)
        h = static + 1j * epsilon * u * (drive_op - drive_op.conj().T)
        out = -1j * (h @ rho - rho @ h)
        out += kappa * (big_a @ rho @ big_a.conj().T)
        out -= 0.5 * kappa * (big_a.conj().T @ big_a @ rho + rho @ big_a.conj().T @ big_a)
        return out

    plus = np.array([1.0, 1.0]) / math.sqrt(2)
    vacuum = np.zeros(fock)
    vacuum[0] = 1.0
    psi = np.kron(vacuum, plus)
    rho = np.outer(psi, psi.conj()).astype(complex)
    t = 0.0
    end = duration + 30 / kappa
    while t < end - 1e-12:
        k1 = derivative(rho, t)
        k2 = derivative(rho + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = derivative(rho + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = derivative(rho + dt * k3, t + dt)
        rho = rho + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
    reduced = rho.reshape(fock, 2, fock, 2).trace(axis1=0, axis2=2)
    return math.atan2(reduced[0, 1].imag, reduced[0, 1].real)


@pytest.mark.parametrize("level", [0, 1])
def test_the_shift_follows_the_resonator_for_each_level_of_the_qudit_read_out(level):
    plain = _quiet()
    crossed = plain.replace(
        readout_crosstalk=[ReadoutCrosstalk(resonator_of=Z2, qudit=D5, chi_hz=CHI)]
    )
    shift = math.remainder(_phase(crossed, level) - _phase(plain, level), 2 * math.pi)
    reference = _reference_phase(plain, level, CHI[level])
    # The two levels hold different photons and shift by different chi.
    assert abs(math.degrees(reference)) > 3.0
    assert math.degrees(shift) == pytest.approx(math.degrees(reference), abs=0.02)


def test_a_system_without_crosstalk_lowers_no_crosstalk_attribute():
    qs = _quiet()
    text = _target_attributes(qs._description, (D5, Z2), integrator="rk4")
    assert "crosstalk" not in text and "readout_resonators" not in text
    crossed = qs.replace(readout_crosstalk=[ReadoutCrosstalk(Z2, D5, CHI)])
    text = _target_attributes(crossed._description, (D5, Z2), integrator="rk4")
    assert "pulse.crosstalk_pairs = array<i64: 14, 4, 2>" in text
    # An entry whose qudits are not both simulated lowers nothing.
    text = _target_attributes(crossed._description, (D5,), integrator="rk4")
    assert "crosstalk" not in text


def test_the_qudit_read_out_needs_its_dispersive_readout():
    qs = _quiet()
    z = qs.physical_qudits[Z2]
    bare = qs.replace(
        qudits={**qs.physical_qudits, Z2: z.replace(readout=None)},
        readout_crosstalk=[ReadoutCrosstalk(Z2, D5, CHI)],
    )
    with pytest.raises(pulse.UnsupportedError, match="readout_crosstalk\\[0\\] shifts qudit 4"):
        _target_attributes(bare._description, (D5, Z2), integrator="rk4")


def test_a_resonator_at_offset_zero_is_not_at_the_tone():
    from cudaq_pulse.quantum_systems import DispersiveReadout, PhysicalQudit
    from cudaq_pulse.quantum_systems.base import Offset

    readout = DispersiveReadout(
        frequency_hz=Offset(0.0), dispersive_shift_hz=-2.9e6, linewidth_hz=12.2e6
    )
    qudits = {i: PhysicalQudit(i, Offset(1.5e8 * i), 50.0, 30.0, readout=readout) for i in (0, 1)}
    qs = pulse.quantum_system(name="offsets", qudits=qudits).replace(
        readout_crosstalk=[ReadoutCrosstalk(resonator_of=1, qudit=0, chi_hz=-20e3)]
    )
    text = _target_attributes(qs._description, (0, 1), integrator="rk4")
    resonators = text.split("pulse.readout_resonators = array<f64: ")[1].split(">")[0]
    # [qudit, chi, linewidth, f_r, given, photons, reference amplitude]
    values = [float(v) for v in resonators.split(", ")]
    assert values[3:5] == [0.0, 1.0]


# A classical call's window. A simulation with calls lowers the program in
# windows; a readout's ring-down must reach past a window's start as it
# does without calls.
_MACHINE = pulse.ControlSystem(
    links={"controller": pulse.classical.Link(fixed_ns=4, feedback_ns=8)}
)


@pulse.classical(max_latency_ns=200, link="controller")
def _same(bit: pulse.Bits, wait_cycles: float) -> float:
    return wait_cycles


@pulse.kernel
def _around_calls(d, z, quarter: int, shape: int):
    dl = get_drive_line(d)  # noqa: F821
    zr = get_readout_line(z)  # noqa: F821
    drive(dl, dl.get_calibration(pulse.gates.X90))  # noqa: F821
    sync(dl, zr)  # noqa: F821
    m = readout(zr, zr.get_calibration(pulse.gates.READOUT))  # noqa: F821
    if shape == 1:
        # A call right after the readout.
        w = _same.submit([m], 600.0).result()
        sync(dl, zr)  # noqa: F821
        wait(dl, w, bounds=(20, 20000))  # noqa: F821
    elif shape == 2:
        # A wait the first call bounds, right after the readout, then a
        # second call whose window starts after that wait.
        w = _same.submit([m], 20.0).result()
        wait(zr, w, bounds=(20, 2000))  # noqa: F821
        w2 = _same.submit([m], 600.0).result()
        sync(dl, zr)  # noqa: F821
        wait(dl, w2, bounds=(20, 20000))  # noqa: F821
    else:
        sync(dl, zr)  # noqa: F821
        wait(dl, 600)  # noqa: F821
    for _ in range(quarter):
        shift_phase(dl.tone, math.pi / 2)  # noqa: F821
    drive(dl, dl.get_calibration(pulse.gates.X90))  # noqa: F821


def _call_phase(qs, shape):
    population = []
    for quarter in (0, 1):
        program = pulse.compile(_around_calls, quantum_system=qs, control_system=_MACHINE).lower(
            qs.qudits[D5], qs.qudits[Z2], quarter, shape
        )
        result = pulse.simulate(program, quantum_system=qs, dt_ns=0.02, levels=2, seed=0)
        population.append(result.level_population(1, qudit=D5))
    return math.atan2(2 * population[1] - 1, 2 * population[0] - 1)


@pytest.mark.parametrize("shape", [1, 2])
def test_a_calls_window_keeps_the_readouts_ring_down(shape):
    plain = _quiet()
    crossed = plain.replace(readout_crosstalk=[ReadoutCrosstalk(Z2, D5, -20e3)])
    without = math.remainder(_call_phase(crossed, 0) - _call_phase(plain, 0), 2 * math.pi)
    across = math.remainder(_call_phase(crossed, shape) - _call_phase(plain, shape), 2 * math.pi)
    assert abs(math.degrees(without)) > 3.0
    assert math.degrees(across) == pytest.approx(math.degrees(without), abs=1e-3)


@pytest.mark.parametrize("level", [0, 1])
def test_an_unstated_resonator_sits_at_the_tone_its_line_starts_at(level):
    # Z2's readout without a frequency: its line starts at Z2's own tone,
    # and the kernel's +2.9 MHz detunes the readout from that resonator as
    # it does from a stated one.
    stated = _quiet()
    z = stated.physical_qudits[Z2]
    unstated = stated.replace(
        qudits={
            **stated.physical_qudits,
            Z2: z.replace(readout=z.readout.replace(frequency_hz=None)),
        }
    )
    entry = [ReadoutCrosstalk(resonator_of=Z2, qudit=D5, chi_hz=CHI)]
    shift = {}
    for label, qs in (("stated", stated), ("unstated", unstated)):
        crossed = qs.replace(readout_crosstalk=entry)
        shift[label] = math.remainder(_phase(crossed, level) - _phase(qs, level), 2 * math.pi)
    assert math.degrees(shift["unstated"]) == pytest.approx(math.degrees(shift["stated"]), abs=0.02)


def test_the_iq_model_puts_an_unstated_resonator_at_the_lines_starting_tone():
    from cudaq_pulse.runtime._calls import IQModel

    qs = _quiet()
    z = qs.physical_qudits[Z2]
    unstated = qs.replace(
        qudits={**qs.physical_qudits, Z2: z.replace(readout=z.readout.replace(frequency_hz=None))}
    )
    stated_model = IQModel(qs._description, (Z2,), False)
    unstated_model = IQModel(unstated._description, (Z2,), False)
    f_r = z.readout.frequency_hz
    for level in (0, 1):
        # A tone 2.9 MHz above where the line started: detuned from the
        # resonator either way.
        expected = stated_model.center(Z2, level, f_r + TONE_SHIFT_HZ, 1.0, 200.0)
        got = unstated_model.center(Z2, level, f_r + TONE_SHIFT_HZ, 1.0, 200.0, f_r)
        assert got == pytest.approx(expected)
    # Without the start, the resonator follows the tone, as before.
    assert unstated_model.center(Z2, 0, f_r + TONE_SHIFT_HZ, 1.0, 200.0) == pytest.approx(
        stated_model.center(Z2, 0, f_r, 1.0, 200.0)
    )
    # A tone or a start that is not a number leaves the detuning unknown.
    with pytest.raises(pulse.UnsupportedError, match="its resonator sits at the tone its line"):
        unstated_model.center(Z2, 0, math.nan, 1.0, 200.0, f_r)
    with pytest.raises(pulse.UnsupportedError, match="the frequency its line's tone starts"):
        unstated_model.center(Z2, 0, f_r, 1.0, 200.0, math.nan)


@pulse.kernel
def lines_only(d, z):
    # Z2's readout line, synced and waited on, never read out.
    dl = get_drive_line(d)  # noqa: F821
    zr = get_readout_line(z)  # noqa: F821
    drive(dl, dl.get_calibration(pulse.gates.X90))  # noqa: F821
    sync(dl, zr)  # noqa: F821
    wait(zr, 100)  # noqa: F821
    sync(dl, zr)  # noqa: F821
    drive(dl, dl.get_calibration(pulse.gates.X90))  # noqa: F821


@pulse.kernel
def looped_readout(d, z):
    dl = get_drive_line(d)  # noqa: F821
    zr = get_readout_line(z)  # noqa: F821
    drive(dl, dl.get_calibration(pulse.gates.X90))  # noqa: F821
    for _ in range(2):
        readout(zr, zr.get_calibration(pulse.gates.READOUT))  # noqa: F821


def test_a_qudit_without_a_readout_model_is_refused_only_when_read_out():
    qs = _quiet()
    z = qs.physical_qudits[Z2]
    bare = qs.replace(qudits={**qs.physical_qudits, Z2: z.replace(readout=None)})
    crossed = bare.replace(readout_crosstalk=[ReadoutCrosstalk(Z2, D5, CHI)])

    def run(kernel, system, *args):
        program = pulse.compile(kernel, quantum_system=system).lower(
            system.qudits[D5], system.qudits[Z2], *args
        )
        result = pulse.simulate(program, quantum_system=system, dt_ns=0.02, levels=2)
        return result.level_population(1, qudit=D5)

    # A readout line the program only syncs and waits on: the entry lowers
    # nothing.
    assert run(lines_only, crossed) == pytest.approx(run(lines_only, bare), abs=1e-12)
    # Read out, in the program or in a loop: refused, with what it lacks.
    for kernel, args in ((ramsey, (0, 0)), (looped_readout, ())):
        with pytest.raises(pulse.UnsupportedError, match="qudit 14 has no readout"):
            run(kernel, crossed, *args)


class _Tree:
    """A module whose site_tree() is *lines* and *nodes*."""

    def __init__(self, lines, nodes):
        self.tree = (lines, nodes)

    def site_tree(self):
        return self.tree


def test_the_qudits_read_out_are_found_in_loops_and_branches():
    # Qudit 3 has a drive line only: no readout reads it out.
    lines = [("drive", 3), ("readout", D5), ("readout", Z2), ("readout", 7)]

    def readout(line):
        return ("readout", None, line, line, 0, 10, (), False)

    wait = ("wait", None, 0, -1, 0, 10, (), False)
    assert _read_out(_Tree(lines, [wait, ("sync", None, [0, 2])])) == set()
    assert _read_out(_Tree(lines, [("for", 2, [wait, readout(2)])])) == {Z2}
    assert _read_out(_Tree(lines, [("if", [wait], [("for", 0, [readout(1)])])])) == {D5}
    assert _read_out(_Tree(lines, [("if", [readout(3)], [])])) == {7}
    # A readout whose line is not known may read out any readout line.
    assert _read_out(_Tree(lines, [("answer",), readout(-1)])) == {D5, Z2, 7}
