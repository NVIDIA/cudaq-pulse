# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The GPU simulation backend's setup and segment logic, without a GPU.

``_GpuBackend`` runs unchanged except for ``_execute``, the one call into
cuDensityMat, which the CPU reference integrator stands in for. So these
tests cover everything the GPU path does around the device: lowering the
module before the first segment, choosing pure or mixed states, and setting
each segment's measurement path and window.
"""

from __future__ import annotations

import math

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.runtime import _jit as jit
from cudaq_pulse.runtime import simulate as simulate_module
from cudaq_pulse.targets import Qubit, TargetDescription

from tests._handles import Q
from tests.runtime._lowering import evolve, on

_PI = math.pi / 20.0
_HALF_PI = math.pi / 40.0


class _CpuStandIn(simulate_module._GpuBackend):
    """The GPU backend with the cuDensityMat run replaced by the CPU one."""

    def evolve(self, state, entries, t0, t1):
        self._segment = (entries, t0, t1)
        return super().evolve(state, entries, t0, t1)

    arguments: list[float] = []

    def _execute(self, text, state, mixed):
        header = text.split("\n", 1)[0]
        assert ("qop.mixed_state" in header) == (mixed and not self.dissipative)
        assert mixed == (state.ndim == 2)
        rho = simulate_module._CpuBackend.evolve(self, self.density(state), *self._segment)
        if mixed:
            return rho
        # Without collapse operators the state stays pure: return the vector.
        values, vectors = np.linalg.eigh(rho)
        return math.sqrt(max(values[-1], 0.0)) * vectors[:, -1]


@pytest.fixture
def gpu_on_cpu(monkeypatch):
    monkeypatch.setattr(jit, "JITCompiler", lambda: None)
    monkeypatch.setattr(simulate_module, "_GpuBackend", _CpuStandIn)


def _target(n, t1_us=0.0):
    return TargetDescription(
        name="gpu-logic",
        qubits={
            q: Qubit(
                index=q,
                frequency_hz=5.0e9 + 0.1e9 * q,
                anharmonicity_hz=-200.0e6,
                t1_us=t1_us,
                t2_star_us=0.0,
                drive_params={"amplitude_scale_rad_per_ns": 1.0},
            )
            for q in range(n)
        },
    )


def _active_reset(prepare):
    @pulse.kernel
    def reset(q0):
        d0 = get_drive_line(q0)
        r0 = get_readout_line(q0)
        drive(d0, square(40, prepare))
        sync(d0, r0)
        bit = readout(r0, square(200, 0.05))
        if bit:
            drive(d0, square(40, _PI))

    return reset


@pulse.kernel
def conditional_flip(q0, q1):
    d0 = get_drive_line(q0)
    d1 = get_drive_line(q1)
    r0 = get_readout_line(q0)
    drive(d0, square(40, _HALF_PI))
    sync(d0, r0)
    bit = readout(r0, square(200, 0.05))
    if bit:
        drive(d1, square(40, _PI))


@pulse.kernel
def measure(q0):
    r0 = get_readout_line(q0)
    readout(r0, square(200, 0.05))


@pulse.kernel
def repeated_reset(q0):
    d0 = get_drive_line(q0)
    r0 = get_readout_line(q0)
    drive(d0, square(40, _HALF_PI))
    for _ in range(3):
        sync(d0, r0)
        bit = readout(r0, square(200, 0.05))
        if bit:
            drive(d0, square(40, _PI))


_P1 = math.exp(-100.0 / 500.0)
_DIAG = np.diag

# (kernel, qubits, simulate() arguments, expected density matrix or None,
# expected branch weights or None)
CASES = {
    "active_reset_excited": (_active_reset(_PI), 1, {"shots": 200}, _DIAG([1, 0]), {(1,): 1.0}),
    "active_reset_superposition": (
        _active_reset(_HALF_PI),
        1,
        {"shots": 400},
        _DIAG([1, 0]),
        {(0,): 0.5, (1,): 0.5},
    ),
    "conditional_flip": (
        conditional_flip,
        2,
        {"shots": 100},
        _DIAG([0.5, 0, 0, 0.5]),
        {(0,): 0.5, (1,): 0.5},
    ),
    "repeated_reset_loop": (
        repeated_reset,
        1,
        {"shots": 100},
        _DIAG([1, 0]),
        {(0, 0, 0): 0.5, (1, 0, 0): 0.5},
    ),
    "t1_readout_decay": (
        measure,
        1,
        {"target": _target(1, t1_us=0.5), "initial_state": [0.0, 1.0]},
        _DIAG([1 - _P1, _P1]),
        {(0,): 1 - _P1, (1,): _P1},
    ),
    "qutrit_measure": (
        measure,
        1,
        {"levels": 3, "initial_state": np.ones(3) / math.sqrt(3), "shots": 300},
        np.eye(3) / 3,
        {(0,): 1 / 3, (1,): 1 / 3, (2,): 1 / 3},
    ),
    "qutrit_active_reset": (_active_reset(_PI), 1, {"levels": 3, "shots": 100}, None, None),
    "mixed_dims_conditional_flip": (
        conditional_flip,
        2,
        {"levels": {0: 3, 1: 2}, "shots": 100},
        None,
        None,
    ),
}


def _simulate(kernel, n, backend, kwargs):
    kwargs = dict(kwargs)
    target = kwargs.pop("target", None) or _target(n)
    kwargs.setdefault("levels", 2)
    program = (kernel, *[Q[i] for i in range(n)])
    return pulse.simulate(on(target, *program), backend=backend, seed=7, **kwargs)


def _weights(result):
    return {b.levels: b.probability for b in result.branches if b.probability > 1e-9}


# The stand-in keeps pure states as vectors, as the GPU backend does, and
# recovers them from RK4 density matrices, which are rank one only to the
# integrator's accuracy; the GPU tests compare to the reference at 1e-6 too.
_ATOL = 1e-6


@pytest.mark.parametrize("name", CASES)
def test_the_gpu_backend_simulates_a_program_that_measures(name, gpu_on_cpu):
    kernel, n, kwargs, rho, branches = CASES[name]
    gpu = _simulate(kernel, n, "gpu", kwargs)
    cpu = _simulate(kernel, n, "cpu", kwargs)
    np.testing.assert_allclose(gpu.density_matrix, cpu.density_matrix, atol=_ATOL)
    gpu_weights, cpu_weights = _weights(gpu), _weights(cpu)
    assert gpu_weights.keys() == cpu_weights.keys()
    for outcome, p in cpu_weights.items():
        assert gpu_weights[outcome] == pytest.approx(p, abs=_ATOL)
    if rho is not None:
        np.testing.assert_allclose(gpu.density_matrix, rho, atol=_ATOL)
    if branches is not None:
        assert gpu_weights.keys() == branches.keys()
        for outcome, p in branches.items():
            assert gpu_weights[outcome] == pytest.approx(p, abs=_ATOL)
    if "shots" in kwargs:
        assert gpu.records.shape == (kwargs["shots"], len(gpu.measurements))


@pytest.mark.parametrize("t_start", [0.0, 130.0, 500.0])
@pytest.mark.parametrize("t1_us", [0.0, 0.5])
def test_gpu_setup_lowers_a_program_that_measures_at_any_start(t_start, t1_us, gpu_on_cpu):
    program = (_active_reset(_PI), Q[0])
    target = _target(1, t1_us)
    result = simulate_module._simulate_lowered(
        on(target, *program),
        backend="gpu",
        t_start=t_start,
        t_end=t_start + 200.0,
        levels=2,
    )
    assert np.trace(result.density_matrix).real == pytest.approx(1.0, abs=1e-9)


def test_evolve_of_a_program_that_measures_runs_the_gpu_backend(gpu_on_cpu):
    lowered = on(_target(1), _active_reset(_HALF_PI), Q[0])
    state = evolve(lowered, t_start=0.0, t_end=140.0, num_steps=2800).final_state
    rho = np.outer(state, state.conj()) if state.ndim == 1 else state
    np.testing.assert_allclose(rho, np.diag([1.0, 0.0]), atol=1e-7)


@pulse.classical(max_latency_ns=0)
def _first_bit(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]


@pulse.classical(max_latency_ns=0)
def _same(x: float) -> float:
    return x


@pulse.kernel
def decoded_reset(q0):
    d0 = get_drive_line(q0)
    r0 = get_readout_line(q0)
    bit = readout(r0, square(200, 0.05))
    sync(d0, r0)
    if _first_bit.submit([bit]).result()[0]:
        drive(d0, square(40, _PI))


@pulse.kernel
def answered_wait(q0, tau: float):
    d0 = get_drive_line(q0)
    r0 = get_readout_line(q0)
    w = _same.submit(tau).result()
    drive(d0, square(40, _PI))
    wait(d0, w)
    sync(d0, r0)
    readout(r0, square(200, 0.05))


# A program with classical calls evolves in segments between their answers,
# each lowered with the answers so far bound.
CALL_CASES = {
    "decoded_reset": ((decoded_reset,), {"initial_state": [2**-0.5, 2**-0.5]}),
    "answered_wait": ((answered_wait, 400.0), {}),
}


@pytest.mark.parametrize("name", CALL_CASES)
def test_the_gpu_backend_simulates_a_program_with_classical_calls(name, gpu_on_cpu):
    (kernel, *args), kwargs = CALL_CASES[name]
    lowered = on(_target(1, t1_us=5.0), kernel, Q[0], *args)
    gpu = pulse.simulate(lowered, backend="gpu", levels=2, **kwargs)
    cpu = pulse.simulate(lowered, backend="cpu", levels=2, **kwargs)
    np.testing.assert_allclose(gpu.density_matrix, cpu.density_matrix, atol=_ATOL)
    assert _weights(gpu) == pytest.approx(_weights(cpu), abs=_ATOL)


@pulse.kernel
def measure_then_flip(q0):
    d0 = get_drive_line(q0)
    r0 = get_readout_line(q0)
    readout(r0, square(200, 0.05))
    sync(d0, r0)
    drive(d0, square(40, _PI))


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_a_late_start_evolves_from_the_start_not_the_measurement(backend, gpu_on_cpu):
    # The readout ends at 100 ns and the pi pulse plays over [100, 120] ns.
    # Started at 120 ns, the evolution must not replay the pulse.
    target = _target(1)
    result = simulate_module._simulate_lowered(
        on(target, measure_then_flip, Q[0]),
        backend=backend,
        t_start=120.0,
        t_end=140.0,
        levels=2,
    )
    np.testing.assert_allclose(result.density_matrix, np.diag([1.0, 0.0]), atol=1e-12)
    assert result.times_ns == (120.0, 140.0)
