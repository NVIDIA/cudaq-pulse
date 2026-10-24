# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Simulation of pulse programs that measure and branch on the outcome.

The program evolves in segments between measurements. A measurement of
qudit ``q`` at the end of its readout projects the state onto each level
``k`` of ``q``; each outcome continues along its own path, where branches on
that measurement take the arm the outcome selects. Summing the unnormalized
state of every path gives the ensemble

    rho' = sum_m U_m P_m rho P_m^dagger U_m^dagger

exactly, and with ``shots`` the branch weights also sample per-shot records.

Segments are lowered by the native pipeline (pulse -> qop) along the path,
then evolved either by a NumPy reference integrator (``backend="cpu"``) or
by the cuDensityMat runtime (``backend="gpu"``), from the projected state.
"""

from __future__ import annotations

import math
import re as _re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from .._native._cudaq_pulse_native import evaluate_callback
from ..compile import ScheduledEvent
from ._attributes import _VALID_INTEGRATORS, _f64, _level_attributes, _target_attributes

if TYPE_CHECKING:
    from ..compile import Lowered

# Branches whose weight falls below this fraction of the initial state's are
# not followed.
_NEGLIGIBLE = 1e-14


@dataclass(frozen=True)
class Measurement:
    """One execution of a readout: which one, on which qudit, and when."""

    readout: int
    iteration: tuple[int, ...]
    qudit: int
    start_ns: float
    end_ns: float


@dataclass
class SimulationResult:
    """The ensemble final state, and per-shot records if shots were taken.

    ``density_matrix`` is indexed with qudit 0 as the least significant
    digit. ``measurements`` lists every measurement any path made, in the
    order of their end times; ``branches`` gives each path's outcomes (one
    per entry of ``measurements``, -1 where that path did not measure) and
    probability. ``records`` holds one such row per shot.
    """

    density_matrix: npt.NDArray[np.complex128]
    dims: tuple[int, ...]
    measurements: list[Measurement]
    branches: list[tuple[tuple[int, ...], float]]
    records: npt.NDArray[np.int64] | None = None
    times_ns: tuple[float, float] = (0.0, 0.0)
    _states: list[Any] = field(default_factory=list, repr=False)


def simulate(
    lowered: Lowered,
    *,
    levels: int | dict[int, int] = 2,
    initial_state: npt.ArrayLike | None = None,
    shots: int | None = None,
    seed: int | None = None,
    backend: str = "cpu",
    t_end: float | None = None,
    dt_ns: float | None = None,
    integrator: str = "rk4",
) -> SimulationResult:
    """Simulate one compiled kernel call on the target it was compiled for.

    Parameters
    ----------
    lowered:
        The module of one call, from ``kernel.lower(*args)`` or
        ``compiled.lower(*args)``.
    levels:
        Levels per qudit (an int for all, or a dict by qudit index). A
        transmon with three or more levels includes its anharmonicity.
    initial_state:
        A state vector or density matrix; ``|0...0>`` by default.
    shots:
        Sample this many per-shot measurement records.
    seed:
        Seed for sampling the records.
    backend:
        ``"cpu"`` for the NumPy reference integrator (RK4), ``"gpu"`` for
        the cuDensityMat runtime.
    t_end:
        End of the simulation in ns; the end of the schedule by default.
    dt_ns:
        Integration step in ns. By default 0.05 ns, or shorter when coupled
        qudits are far detuned: a coupling turns at their detuning, and the
        step resolves it with 20 steps per period (0.024 ns at 2.1 GHz),
        a whole fraction of a clock cycle.
    integrator:
        cuDensityMat integrator for the GPU backend: ``"rk1"``, ``"rk2"``,
        ``"rk4"``, ``"magnus"`` or ``"crank_nicolson"``.
    """
    from ..compile import Lowered

    if not isinstance(lowered, Lowered):
        raise TypeError(
            "simulate() expects the module of one kernel call, from "
            f"kernel.lower(*args); got {type(lowered).__name__}"
        )
    return _simulate_lowered(
        lowered,
        levels=levels,
        initial_state=initial_state,
        shots=shots,
        seed=seed,
        backend=backend,
        t_end=t_end,
        dt_ns=dt_ns,
        integrator=integrator,
    )


def _simulate_lowered(
    lowered: Lowered,
    *,
    levels: int | dict[int, int] = 2,
    initial_state: npt.ArrayLike | None = None,
    shots: int | None = None,
    seed: int | None = None,
    backend: str = "cpu",
    t_start: float = 0.0,
    t_end: float | None = None,
    dt_ns: float | None = None,
    integrator: str = "rk4",
) -> SimulationResult:
    if backend not in ("cpu", "gpu"):
        raise ValueError(f"unknown backend {backend!r}; choose 'cpu' or 'gpu'")
    if dt_ns is None:
        dt_ns = _default_step(lowered)
    elif dt_ns <= 0:
        raise ValueError("dt_ns must be positive")
    if shots is not None and shots < 0:
        raise ValueError("shots must not be negative")
    if integrator not in _VALID_INTEGRATORS:
        raise ValueError(
            f"Unknown integrator {integrator!r}. "
            f"Choose from: {', '.join(sorted(_VALID_INTEGRATORS))}"
        )
    if lowered.parameters:
        raise ValueError(
            "the module has float parameters "
            f"({', '.join(lowered.parameters)}); lower it with their values first"
        )
    target = lowered.target.target
    if getattr(target, "architecture", None) != "transmon":
        raise NotImplementedError(
            "simulation supports transmon targets; the neutral-atom Hamiltonian "
            "(Rabi drive, detuning and C6 / r^6 n n interactions) is not lowered"
        )
    return _simulate_compiled(
        lowered,
        target,
        levels=levels,
        initial_state=initial_state,
        shots=shots,
        seed=seed,
        backend=backend,
        t_start=t_start,
        t_end=t_end,
        dt_ns=dt_ns,
        integrator=integrator,
        clock_ghz=lowered.target.clock_ghz,
    )


_DEFAULT_STEP_NS = 0.05
_STEPS_PER_COUPLING_PERIOD = 20


def _default_step(lowered: Lowered) -> float:
    """The integration step: 0.05 ns, or 20 steps per period of the fastest
    coupling between the program's qudits, which turns at their detuning,
    rounded down to a whole fraction of a clock cycle."""
    target = lowered.target.target
    active = set(lowered.qudits)
    detuning = max(
        (
            abs(target.qubits[c.qubit_a].frequency_hz - target.qubits[c.qubit_b].frequency_hz)
            for c in target.couplings
            if c.qubit_a in active and c.qubit_b in active
        ),
        default=0.0,
    )
    step = _DEFAULT_STEP_NS
    if detuning > 0:
        step = min(step, 1e9 / (_STEPS_PER_COUPLING_PERIOD * detuning))
    # A whole number of steps per clock cycle keeps pulse edges on the grid.
    cycle = 1.0 / lowered.target.clock_ghz
    return cycle / math.ceil(cycle / step - 1e-9)


def _simulate_compiled(
    compiled: Any,
    target: Any,
    *,
    levels: int | dict[int, int],
    initial_state: npt.ArrayLike | None,
    shots: int | None,
    seed: int | None,
    backend: str,
    t_start: float,
    t_end: float | None,
    dt_ns: float,
    integrator: str,
    clock_ghz: float,
) -> SimulationResult:
    # A cached specialization's module is shared by every call of it, from
    # any thread: a segment sets the module attributes it lowers with and
    # reads the lowering under the module's lock, in one step. The GPU path
    # binds the float arguments into a private copy of the module.
    lock = compiled._lock
    arguments = list(compiled.arguments)
    module = compiled._concrete() if backend == "gpu" else compiled._module
    n = compiled.n_qudits
    dims, extra = _level_attributes(target, compiled.qudits, n, levels)
    attributes = _target_attributes(
        target,
        compiled.qudits,
        n,
        t_start=t_start,
        t_end=t_start + 1.0,
        num_steps=1,
        integrator=integrator,
    )
    # Segments set their own window.
    base = _re.sub(r"qop\.(t_start|t_end|num_steps) = [^,]+, ", "", attributes[1:-1]) + ", " + extra

    if t_end is None:
        events = [ScheduledEvent(*e) for e in module.events()]
        ends = [
            e.start_vtu + e.duration_vtu
            for e in events
            if e.start_vtu is not None and e.duration_vtu is not None
        ]
        t_end = max(ends, default=0) / clock_ghz
    if t_end < t_start:
        raise ValueError(f"t_end ({t_end}) must not precede t_start ({t_start})")

    dim = math.prod(dims)
    state = _initial_state(initial_state, dim)
    engine: _Backend
    if backend == "cpu":
        engine = _CpuBackend(module, dims, dt_ns, base, lock, arguments)
    else:
        engine = _GpuBackend(module, dims, dt_ns, base, lock, integrator)
    state = engine.prepare(state)

    rng = np.random.default_rng(seed)
    seen: dict[tuple[int, tuple[int, ...]], Measurement] = {}
    leaves: list[tuple[Any, list[tuple[tuple[int, tuple[int, ...]], int]], int]] = []

    def run(
        path: list[tuple[tuple[int, tuple[int, ...]], int]], state: Any, t: float, count: int
    ) -> None:
        entries = [[index, *iteration, k] for (index, iteration), k in path]
        found = module.next_measurement(entries)
        boundary = t_end if found is None or found[4] > t_end else found[4]
        if boundary > t:
            state = engine.evolve(state, entries, t, boundary)
        if found is None or found[4] > t_end:
            leaves.append((state, path, count))
            return
        index, iteration, qudit, start, end = found
        key = (index, tuple(iteration))
        seen[key] = Measurement(index, tuple(iteration), qudit, start, end)
        projected = [engine.project(state, dims, qudit, k) for k in range(dims[qudit])]
        weights = np.array([engine.weight(s) for s in projected])
        total = weights.sum()
        counts = (
            rng.multinomial(count, weights / total)
            if count and total > 0
            else np.zeros(len(weights), dtype=np.int64)
        )
        # A measurement that ends before t_start projects the initial state;
        # evolution still starts at t_start.
        for k, (branch, w) in enumerate(zip(projected, weights)):
            if w > _NEGLIGIBLE * norm:
                run([*path, (key, k)], branch, max(end, t), int(counts[k]))

    norm = engine.weight(state)
    run([], state, t_start, shots or 0)

    measurements = sorted(seen.values(), key=lambda m: (m.end_ns, m.readout, m.iteration))
    column = {(m.readout, m.iteration): i for i, m in enumerate(measurements)}
    rho = np.zeros((dim, dim), dtype=np.complex128)
    branches = []
    rows = []
    for leaf, path, count in leaves:
        density = engine.density(leaf)
        rho += density
        outcomes = [-1] * len(measurements)
        for key, k in path:
            outcomes[column[key]] = k
        branches.append((tuple(outcomes), float(np.real(np.trace(density)))))
        rows.extend([outcomes] * count)
    records = None
    if shots is not None:
        records = np.array(rows, dtype=np.int64).reshape(len(rows), len(measurements))
        rng.shuffle(records)
    return SimulationResult(
        density_matrix=rho,
        dims=dims,
        measurements=measurements,
        branches=branches,
        records=records,
        times_ns=(t_start, t_end),
        _states=[leaf for leaf, _, _ in leaves],
    )


def _initial_state(state: npt.ArrayLike | None, dim: int) -> npt.NDArray[np.complex128]:
    if state is None:
        psi = np.zeros(dim, dtype=np.complex128)
        psi[0] = 1.0
        return psi
    array = np.asarray(state, dtype=np.complex128)
    if array.shape not in ((dim,), (dim, dim)):
        raise ValueError(f"initial_state must have shape ({dim},) or ({dim}, {dim})")
    return array


def _level_mask(dims: tuple[int, ...], qudit: int, level: int) -> npt.NDArray[np.bool_]:
    stride = math.prod(dims[:qudit])
    indices = np.arange(math.prod(dims))
    return np.asarray((indices // stride) % dims[qudit] == level)


def _segment_attributes(entries: list[list[int]], t0: float, t1: float, steps: int) -> str:
    path = ", ".join(f"array<i64: {', '.join(map(str, e))}>" for e in entries)
    return (
        f"{{pulse.path = [{path}], qop.t_start = {_f64(t0)}, qop.t_end = {_f64(t1)}, "
        f"qop.num_steps = {steps} : i64}}"
    )


class _Backend:
    """Segment evolution and projection for one state representation.

    *base* holds the module attributes of the target and levels, in MLIR
    syntax without braces; every segment sets them along with its own.
    """

    def __init__(
        self, module: Any, dims: tuple[int, ...], dt_ns: float, base: str, lock: Any
    ) -> None:
        self.module = module
        self.dims = dims
        self.dim = math.prod(dims)
        self.dt_ns = dt_ns
        self.base = base
        self.lock = lock

    def _attributes_for(self, attributes: str) -> str:
        """The module attributes of one segment (*attributes* in braces)."""
        return "{" + self.base + ", " + attributes[1:]

    def steps(self, t0: float, t1: float) -> int:
        return max(1, math.ceil((t1 - t0) / self.dt_ns - 1e-9))

    def prepare(self, state: npt.NDArray[np.complex128]) -> Any:
        return state

    def evolve(self, state: Any, entries: list[list[int]], t0: float, t1: float) -> Any:
        raise NotImplementedError

    @staticmethod
    def project(state: Any, dims: tuple[int, ...], qudit: int, level: int) -> Any:
        mask = _level_mask(dims, qudit, level)
        if state.ndim == 1:
            return state * mask
        return state * np.outer(mask, mask)

    @staticmethod
    def weight(state: Any) -> float:
        if state.ndim == 1:
            return float(np.vdot(state, state).real)
        return float(np.trace(state).real)

    @staticmethod
    def density(state: Any) -> npt.NDArray[np.complex128]:
        if state.ndim == 1:
            return np.asarray(np.outer(state, state.conj()))
        return np.asarray(state)


def _full_operator(factors: list[Any], dims: tuple[int, ...]) -> npt.NDArray[np.complex128]:
    """The product of single-mode factors on the whole register."""
    on_mode: dict[int, npt.NDArray[np.complex128]] = {}
    for mode, d, data in factors:
        values = np.asarray(data, dtype=float)
        matrix = (values[0::2] + 1j * values[1::2]).reshape((d, d), order="F")
        on_mode[mode] = matrix @ on_mode[mode] if mode in on_mode else matrix
    full: npt.NDArray[np.complex128] = np.ones((1, 1), dtype=np.complex128)
    for mode in reversed(range(len(dims))):
        full = np.asarray(np.kron(full, on_mode.get(mode, np.eye(dims[mode]))), dtype=np.complex128)
    return full


class _CpuBackend(_Backend):
    """Fixed-step RK4 on the density matrix: the reference for the GPU path.

    The lowering takes the float arguments as values, so a call with new
    values lowers no differently from the last one.
    """

    def __init__(
        self,
        module: Any,
        dims: tuple[int, ...],
        dt_ns: float,
        base: str,
        lock: Any,
        arguments: list[float],
    ) -> None:
        super().__init__(module, dims, dt_ns, base, lock)
        self.arguments = arguments

    def prepare(self, state: npt.NDArray[np.complex128]) -> Any:
        return self.density(state)

    def evolve(self, state: Any, entries: list[list[int]], t0: float, t1: float) -> Any:
        steps = self.steps(t0, t1)
        with self.lock:
            self.module.set_attributes(
                self._attributes_for(_segment_attributes(entries, t0, t1, steps))
            )
            _, hamiltonian, collapse, *_ = self.module.operator_program(self.arguments)
        static = np.zeros((self.dim, self.dim), dtype=np.complex128)
        driven = []
        for re, im, kind, params, factors in hamiltonian:
            operator = _full_operator(factors, self.dims)
            if kind:
                driven.append((kind, params, operator))
            else:
                static += (re + 1j * im) * operator
        jumps = [(re + 1j * im) * _full_operator(f, self.dims) for re, im, _, _, f in collapse]
        decay = sum((c.conj().T @ c for c in jumps), np.zeros_like(static))

        h = (t1 - t0) / steps
        starts = t0 + h * np.arange(steps)
        # Callbacks are right-continuous on half-open intervals: like the
        # cuDensityMat integrator, sample a step's end just inside the step so
        # that a pulse ending there is not lost.
        ends = np.maximum(
            starts,
            starts
            + h
            - np.maximum(h * 1e-9, 8 * np.finfo(float).eps * np.maximum(1.0, np.abs(starts + h))),
        )
        times = np.stack([starts, starts + 0.5 * h, ends], axis=1).ravel()
        coefficients = [np.asarray(evaluate_callback(k, p, times.tolist())) for k, p, _ in driven]

        def hamiltonian_at(j: int) -> npt.NDArray[np.complex128]:
            total = static.copy()
            for c, (_, _, operator) in zip(coefficients, driven):
                if c[j]:
                    total += c[j] * operator
            return total

        def lindblad(rho: Any, hm: Any) -> Any:
            out = -1j * (hm @ rho - rho @ hm)
            for c in jumps:
                out += c @ rho @ c.conj().T
            if jumps:
                out -= 0.5 * (decay @ rho + rho @ decay)
            return out

        rho = state
        for step in range(steps):
            h0, hm, h1 = (hamiltonian_at(3 * step + i) for i in range(3))
            k1 = lindblad(rho, h0)
            k2 = lindblad(rho + 0.5 * h * k1, hm)
            k3 = lindblad(rho + 0.5 * h * k2, hm)
            k4 = lindblad(rho + h * k3, h1)
            rho = rho + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return rho


class _GpuBackend(_Backend):
    """Each segment is lowered and JIT-run by the cuDensityMat runtime."""

    def __init__(
        self,
        module: Any,
        dims: tuple[int, ...],
        dt_ns: float,
        base: str,
        lock: Any,
        integrator: str,
    ) -> None:
        super().__init__(module, dims, dt_ns, base, lock)
        from .jit import JITCompiler

        self.compiler = JITCompiler()
        self.integrator = integrator
        # Collapse operators depend on the target alone. A program that
        # measures lowers only along a measurement path over a window, so ask
        # along the empty path over an empty window at 0 ns, which ends before
        # any measurement; every segment then sets its own path and window.
        with lock:
            module.set_attributes(self._attributes_for(_segment_attributes([], 0.0, 0.0, 1)))
            _, _, collapse, *_ = module.operator_program()
        self.dissipative = bool(collapse)

    def prepare(self, state: npt.NDArray[np.complex128]) -> Any:
        if self.dissipative and state.ndim == 1:
            return self.density(state)
        return state

    def evolve(self, state: Any, entries: list[list[int]], t0: float, t1: float) -> Any:
        mixed = state.ndim == 2
        attributes = _segment_attributes(entries, t0, t1, self.steps(t0, t1))
        attributes = attributes[:-1]
        if mixed and not self.dissipative:
            attributes += ", qop.mixed_state"
        with self.lock:
            self.module.set_attributes(self._attributes_for(attributes + "}"))
            text = self.module.print()
        return self._execute(text, state, mixed)

    def _execute(self, text: str, state: Any, mixed: bool) -> Any:
        """Run one segment, the module *text*, on the GPU from *state*."""
        jit_module = self.compiler.compile_pulse_mlir(text)
        # cuDensityMat stores a density matrix with the ket index fastest.
        buffer = np.ascontiguousarray(state.ravel(order="F") if mixed else state)
        library = self.compiler._runtime.library
        if library.cudm_set_initial_state(buffer.ctypes.data, buffer.nbytes) != 0:
            raise RuntimeError("could not set the cuDensityMat initial state")
        results = self.compiler.load_and_run(
            jit_module, n_qubits=len(self.dims), state_dim=self.dim
        )
        values = results[0].to_numpy().ravel()
        if mixed:
            return values.reshape((self.dim, self.dim), order="F")
        return values
