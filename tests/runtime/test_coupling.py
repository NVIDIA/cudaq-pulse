# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Coupled qudits in their own rotating frames, against analytic results.

An exchange coupling g (a^dagger b + a b^dagger) between transmons at w_a and
w_b swaps an excitation at 2g when they are resonant, barely at all when they
are detuned by many g, and shifts |11> by the dispersive ZZ rate of
second-order perturbation theory. A crosstalk entry adds exactly its ZZ rate.
"""

from __future__ import annotations

import cmath
import dataclasses
import math
import re

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse._native._cudaq_pulse_native import MLIRPipeline, evaluate_callback
from cudaq_pulse.targets import Coupling, CrosstalkEntry, Qubit, TargetDescription

from tests._handles import Q
from tests.runtime._lowering import on

_TWO_PI_GHZ = 2.0 * math.pi * 1.0e-9


def _pair(f0, f1, *, g_hz=0.0, zz_hz=0.0, alphas=(-200.0e6, -180.0e6)):
    return TargetDescription(
        name="coupled-pair",
        qubits={
            q: Qubit(q, f, a, 0.0, 0.0, drive_params={"amplitude_scale_rad_per_ns": 1.0})
            for q, (f, a) in enumerate(zip((f0, f1), alphas))
        },
        couplings=[Coupling(0, 1, coupling_strength_hz=g_hz)] if g_hz else [],
        crosstalk=[CrosstalkEntry(0, 1, 0.0, zz_hz, 0.0)] if zz_hz else [],
    )


@pulse.kernel
def _idle(q0, q1):
    d0, t0 = get_drive_line(q0)
    get_drive_line(q1)  # simulate the idle partner
    drive(d0, square(2, 0.0), t0)


def _lowered(target):
    return on(target, _idle, Q[0], Q[1])


def _basis(d, n0, n1):
    """Index of |n0, n1> with qudit 0 the least significant digit."""
    return n0 + d * n1


def _swapped(target, levels, times, dt_ns=0.05):
    """P(|0,1>) after starting in |1,0>, at each of *times* (ns)."""
    lowered = _lowered(target)
    psi = np.zeros(levels * levels)
    psi[_basis(levels, 1, 0)] = 1.0
    out = []
    for t in times:
        rho = pulse.simulate(
            lowered, levels=levels, initial_state=psi, t_end=t, dt_ns=dt_ns
        ).density_matrix
        out.append(rho[_basis(levels, 0, 1), _basis(levels, 0, 1)].real)
    return np.array(out)


def _ramsey_zz_hz(target, levels, t_ns, dt_ns):
    """ZZ rate and qudit 0's frequency shift (Hz), from Ramseys with qudit 1
    in |0> and |1>."""
    lowered = _lowered(target)
    phases = []
    for k in (0, 1):
        psi = np.zeros(levels * levels, dtype=complex)
        psi[_basis(levels, 0, k)] = psi[_basis(levels, 1, k)] = 1.0 / math.sqrt(2.0)
        rho = pulse.simulate(
            lowered, levels=levels, initial_state=psi, t_end=t_ns, dt_ns=dt_ns
        ).density_matrix
        phases.append(cmath.phase(rho[_basis(levels, 0, k), _basis(levels, 1, k)]))
    to_hz = 1.0 / (2.0 * math.pi * t_ns * 1.0e-9)
    return (phases[1] - phases[0]) * to_hz, phases[0] * to_hz


def _perturbative_zz_hz(f0, f1, g_hz, a0, a1):
    """zeta = 2 g^2 (a0 + a1) / ((D + a0)(D - a1)), D = f0 - f1."""
    detuning = f0 - f1
    return 2.0 * g_hz**2 * (a0 + a1) / ((detuning + a0) * (detuning - a1))


def _exact_zz_hz(f0, f1, g_hz, a0, a1, levels=4):
    """E11 - E10 - E01 + E00 of the exchange-coupled transmons, diagonalized."""
    a = np.diag(np.sqrt(np.arange(1, levels)), 1)
    n = a.T @ a
    eye = np.eye(levels)

    def mode(f, alpha):
        return f * n + 0.5 * alpha * n @ (n - eye)

    h = (
        np.kron(eye, mode(f0, a0))
        + np.kron(mode(f1, a1), eye)
        + g_hz * (np.kron(a, a.T) + np.kron(a.T, a))
    )
    energies, vectors = np.linalg.eigh(h)

    def dressed(n0, n1):
        return energies[np.argmax(np.abs(vectors[_basis(levels, n0, n1)]))]

    return dressed(1, 1) - dressed(1, 0) - dressed(0, 1) + dressed(0, 0)


@pytest.mark.parametrize("levels", [2, 3])
def test_resonant_pair_swaps_at_twice_the_coupling(levels):
    g_hz = 5.0e6
    g = g_hz * _TWO_PI_GHZ
    times = [12.5, 25.0, 50.0, 75.0]
    swapped = _swapped(_pair(5.0e9, 5.0e9, g_hz=g_hz), levels, times)
    # P(|0,1>) = sin^2(g t): a full swap at pi / (2 g) = 50 ns.
    np.testing.assert_allclose(swapped, np.sin(g * np.array(times)) ** 2, atol=1e-7)


@pytest.mark.parametrize("levels", [2, 3])
def test_detuned_pair_follows_the_generalized_rabi_formula(levels):
    g_hz, detuning_hz = 5.0e6, 40.0e6
    times = [5.0, 10.0, 17.0, 30.0]
    swapped = _swapped(_pair(5.0e9 + detuning_hz, 5.0e9, g_hz=g_hz), levels, times, dt_ns=0.01)
    # In the one-excitation subspace: P = 4g^2 / W^2 sin^2(W t / 2),
    # W = sqrt(D^2 + 4 g^2), at most 6 % here.
    g, detuning = g_hz * _TWO_PI_GHZ, detuning_hz * _TWO_PI_GHZ
    w = math.sqrt(detuning**2 + 4.0 * g**2)
    expected = 4.0 * g**2 / w**2 * np.sin(w * np.array(times) / 2.0) ** 2
    np.testing.assert_allclose(swapped, expected, atol=1e-7)


@pytest.mark.parametrize("levels", [2, 3])
def test_transmons_detuned_by_gigahertz_stay_put(levels):
    # A data qubit and an ancilla of a surface-code device: 2 GHz apart,
    # g = 7 MHz. At most (2 g / D)^2 = 5e-5 of the excitation moves.
    times = [50.0, 333.0, 1000.0]
    swapped = _swapped(_pair(3.9e9, 5.9e9, g_hz=7.0e6), levels, times)
    assert np.all(np.abs(swapped) < 1e-4)


def test_two_level_exchange_has_no_zz():
    zeta, _ = _ramsey_zz_hz(_pair(5.0e9, 5.5e9, g_hz=10.0e6), 2, 200.0, 0.02)
    assert abs(zeta) < 1.0


@pytest.mark.parametrize(
    "f0, f1, g_hz, t_ns, dt_ns",
    [
        (5.0e9, 5.5e9, 10.0e6, 300.0, 0.02),
        # Krinner et al. (2022): data qubit and ancilla about 2 GHz apart,
        # J = 7 MHz; the idle ZZ is under about 10 kHz.
        (3.9e9, 5.9e9, 7.0e6, 1500.0, 0.025),
    ],
)
def test_dispersive_zz_matches_perturbation_theory(f0, f1, g_hz, t_ns, dt_ns):
    a0, a1 = -200.0e6, -180.0e6
    zeta, shift = _ramsey_zz_hz(_pair(f0, f1, g_hz=g_hz, alphas=(a0, a1)), 3, t_ns, dt_ns)
    exact = _exact_zz_hz(f0, f1, g_hz, a0, a1)
    assert zeta == pytest.approx(exact, rel=1e-2)
    assert zeta == pytest.approx(_perturbative_zz_hz(f0, f1, g_hz, a0, a1), rel=1e-2)
    # With qudit 1 in |0>, qudit 0 moves by the exchange shift g^2 / D.
    assert shift == pytest.approx(g_hz**2 / (f0 - f1), rel=1e-2)


@pytest.mark.parametrize("levels", [2, 3])
def test_crosstalk_zz_is_the_conditional_frequency_shift(levels):
    zz_hz = 1.0e5
    zeta, shift = _ramsey_zz_hz(_pair(5.0e9, 5.5e9, zz_hz=zz_hz), levels, 1000.0, 0.05)
    assert zeta == pytest.approx(zz_hz, rel=1e-6)
    assert abs(shift) < 1e-3


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_systems_without_a_simulation_model_are_refused(backend):
    target = dataclasses.replace(_pair(5.0e9, 5.5e9, g_hz=10.0e6), architecture="other")
    with pytest.raises(NotImplementedError, match=r"\(other\) has no simulation model"):
        pulse.simulate(_lowered(target), backend=backend)


# The GPU path lowers the same pulse-to-qop operator to cuDensityMat terms:
# rebuild H(t) from those terms and compare it with the CPU reference's.

_CONSTANT = re.compile(r"(%\S+) = arith\.constant dense<(.*)> : tensor<(\d+)x\d+x2xf64>")
_ELEMENTARY = re.compile(r"(%\S+) = cudm\.create_elementary_op %\S+, (%\S+) ")
_PRODUCT = re.compile(r"cudm\.append_elementary_product %\S+, (%\S+), ([^{]+)\{(.*)\} :")
_APPEND = re.compile(r"cudm\.operator_append_term %\S+, %\S+, (%\S+) \{(.*)\} :")


def _attribute(attributes, name):
    found = re.search(rf"{re.escape(name)} = ([^,]+?)( : \w+)?(,|$)", attributes)
    return found.group(1) if found else None


def _cudm_hamiltonian(text, dims):
    """The Hamiltonian terms of the cuDensityMat operator: those appended
    to the ket side with coefficient -i."""
    matrices, elementary, products = {}, {}, {}
    for line in text.splitlines():
        if found := _CONSTANT.search(line):
            d = int(found.group(3))
            values = np.array(
                [float(v) for v in re.findall(r"-?[\d.]+(?:[eE][-+]?\d+)?", found.group(2))]
            )
            # Column-major: element [column][row].
            matrices[found.group(1)] = (values[0::2] + 1j * values[1::2]).reshape((d, d)).T
        elif found := _ELEMENTARY.search(line):
            elementary[found.group(1)] = matrices[found.group(2)]
        elif found := _PRODUCT.search(line):
            attributes = found.group(3)
            modes = re.search(r"modes_acted_on = array<i32: ([^>]*)>", attributes).group(1)
            names = [name.strip() for name in found.group(2).split(",") if name.strip()]
            factors = [elementary[name] for name in names]
            coefficient = float(_attribute(attributes, "coeff_real")) + 1j * float(
                _attribute(attributes, "coeff_imag")
            )
            params = re.search(r"cudm\.callback_params = array<f64: ([^>]*)>", attributes)
            kind = _attribute(attributes, "cudm.callback_kind")
            callback = (int(kind), [float(p) for p in params.group(1).split(",")]) if kind else None
            products.setdefault(found.group(1), []).append(
                (coefficient, callback, [int(m) for m in modes.split(",")], factors)
            )
        elif found := _APPEND.search(line):
            attributes = found.group(2)
            if (
                int(_attribute(attributes, "duality")) != 0
                or float(_attribute(attributes, "coeff_imag")) != -1.0
            ):
                continue
            for coefficient, callback, modes, factors in products[found.group(1)]:
                yield _full(dims, dict(zip(modes, factors))), coefficient, callback


def _full(dims, on_mode):
    full = np.ones((1, 1), dtype=complex)
    for mode in reversed(range(len(dims))):
        full = np.kron(full, on_mode.get(mode, np.eye(dims[mode])))
    return full


def _cpu_hamiltonian(program, dims):
    for re_, im, kind, params, factors in program[1]:
        on_mode = {}
        for mode, d, data in factors:
            values = np.asarray(data)
            on_mode[mode] = (values[0::2] + 1j * values[1::2]).reshape((d, d), order="F")
        yield _full(dims, on_mode), complex(re_, im), ((kind, list(params)) if kind else None)


def _at(terms, times):
    total = None
    for operator, coefficient, callback in terms:
        values = (
            np.asarray(evaluate_callback(callback[0], callback[1], times))
            if callback
            else np.ones(len(times))
        )
        part = coefficient * values[:, None, None] * operator[None]
        total = part if total is None else total + part
    return total


@pytest.mark.parametrize("levels", [2, 3])
def test_gpu_lowering_builds_the_cpu_operator(levels):
    target = _pair(3.9e9, 5.9e9, g_hz=7.0e6, zz_hz=2.0e4)

    @pulse.kernel
    def driven(q0, q1):
        d0, t0 = get_drive_line(q0)
        get_drive_line(q1)
        drive(d0, gaussian(40, 0.05, 10.0), t0)

    lowered = on(target, driven, Q[0], Q[1])
    # A CPU run leaves the module with the attributes of its segment, which
    # the GPU backend lowers the same way.
    pulse.simulate(lowered, levels=levels, t_end=20.0)
    dims = (levels, levels)
    times = [0.0, 3.3, 11.7, 19.9]
    cpu_terms = list(_cpu_hamiltonian(lowered._module.operator_program([]), dims))
    pipeline = MLIRPipeline()
    text = pipeline.run_qop_to_cudm(pipeline.run_pulse_to_qop(lowered._module.print()))
    gpu_terms = list(_cudm_hamiltonian(text, dims))
    assert len(gpu_terms) == len(cpu_terms)
    cpu = _at(cpu_terms, times)
    assert np.abs(cpu).max() > 0.04
    np.testing.assert_allclose(_at(gpu_terms, times), cpu, atol=1e-12)
    # The coupling turns at the frames' difference frequency.
    assert "coupling_0_0" in text
