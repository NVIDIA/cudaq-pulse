# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""High-level evolve() entry point for pulse-level time evolution."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from .jit import compile_and_run_pulse

_VALID_INTEGRATORS = frozenset(
    {
        "rk1",
        "rk2",
        "rk4",
        "magnus",
        "crank_nicolson",
    }
)


@dataclass()
class EvolveResult:
    """Result of a pulse-level time evolution."""

    final_state: npt.NDArray[Any]
    times: npt.NDArray[Any]
    expectation_values: dict[str, npt.NDArray[Any]] | None = None


def evolve(
    program: Any,
    *,
    target: Any,
    t_start: float,
    t_end: float,
    num_steps: int,
    integrator: str = "rk4",
    clock_ghz: float = 2.0,
    observables: dict[str, Any] | None = None,
) -> EvolveResult:
    """Run a pulse program through the full compilation pipeline and evolve.

    The kernel call that produced *program* is compiled with
    ``cudaq_pulse.compile()`` (verify, canonicalize, virtual-Z, fusion and an
    ALAP schedule), annotated with the target's calibration, lowered
    pulse -> qop -> cudm -> LLVM, and JIT-executed on the GPU. A program
    that measures is evolved between measurements, projecting at each one,
    and ``final_state`` is the ensemble density matrix over all outcomes
    (see ``cudaq_pulse.simulate`` for shot records and the CPU reference).

    Parameters
    ----------
    program:
        The IR returned by calling a ``@cudaq_pulse.kernel`` with concrete
        arguments.
    target:
        A ``Target`` providing the Hamiltonian, decoherence, and connectivity.
    t_start, t_end:
        Time window in nanoseconds.
    num_steps:
        Number of integration time steps.
    integrator:
        Time-integration strategy for the cuDensityMat runtime. One of the
        explicit Runge-Kutta schemes ``"rk1"``, ``"rk2"``, ``"rk4"``, the
        ``"magnus"`` Taylor-series midpoint expansion, or the
        ``"crank_nicolson"`` predictor-corrector method.
    clock_ghz:
        Pulse virtual-clock rate in GHz when converting a traced kernel IR.
    observables:
        Optional dict mapping names to operator expressions.

    Returns
    -------
    EvolveResult
    """
    if target is None:
        raise ValueError(
            "target is required. Pass a Target to specify the system "
            "Hamiltonian and decoherence model."
        )
    if getattr(target, "architecture", None) != "transmon":
        raise NotImplementedError(
            "GPU evolution currently supports two-level transmon targets "
            "only; neutral-atom and multilevel target lowering is not yet "
            "implemented"
        )
    if integrator not in _VALID_INTEGRATORS:
        raise ValueError(
            f"Unknown integrator {integrator!r}. "
            f"Choose from: {', '.join(sorted(_VALID_INTEGRATORS))}"
        )
    if t_end <= t_start:
        raise ValueError(f"t_end ({t_end}) must be > t_start ({t_start})")
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    if clock_ghz <= 0:
        raise ValueError(f"clock_ghz must be positive, got {clock_ghz}")
    if observables:
        raise NotImplementedError(
            "observable evaluation is not implemented in the research "
            "preview; evolve the state and evaluate observables explicitly"
        )

    compiled = _compile_for(program, target=target, clock_ghz=clock_ghz)
    if compiled.module.next_measurement([]) is not None:
        # Evolve between measurements and sum the branches of every outcome.
        from .simulate import _simulate_compiled

        result = _simulate_compiled(
            compiled,
            target,
            levels=2,
            initial_state=None,
            shots=None,
            seed=None,
            backend="gpu",
            t_start=t_start,
            t_end=t_end,
            dt_ns=(t_end - t_start) / num_steps,
            integrator=integrator,
            clock_ghz=clock_ghz,
        )
        times = np.linspace(t_start, t_end, num_steps + 1)
        return EvolveResult(final_state=result.density_matrix, times=times)
    compiled.module.set_attributes(
        _target_attributes(
            target,
            compiled._qubits,
            compiled._n_qubits,
            t_start=t_start,
            t_end=t_end,
            num_steps=num_steps,
            integrator=integrator,
        )
    )
    pulse_mlir = compiled.module.print()
    results = compile_and_run_pulse(pulse_mlir, entry="main", n_qubits=compiled._n_qubits)
    if not results:
        raise RuntimeError("JIT execution returned no results.")

    times = np.linspace(t_start, t_end, num_steps + 1)
    return EvolveResult(final_state=results[0].to_numpy(), times=times)


_EVOLVE_PASSES = ("verify", "canonicalize", "virtual_z", "fusion")


def _compile_for(program: Any, *, target: Any, clock_ghz: float) -> Any:
    """Compile the kernel call that produced *program* for *target*."""
    from ..compile import compile as compile_kernel
    from ..kernel.ir_builder import PythonIRBuilder

    if not isinstance(program, PythonIRBuilder) or program.source is None:
        raise TypeError(
            "Expected the IR returned by calling a @cudaq_pulse.kernel, got "
            f"{type(program).__name__}. Call the kernel to build its IR first."
        )
    kernel_fn, args = program.source
    frequencies = target.frequencies
    compiled = compile_kernel(
        kernel_fn,
        list(args),
        clock_ghz=clock_ghz,
        qubit_freq_hz=frequencies,
        passes=_EVOLVE_PASSES,
    )
    missing = [q for q in compiled._qubits if q not in frequencies]
    if missing:
        raise ValueError(f"target {target.name!r} does not define active qubits {missing}")
    return compiled


def _f64_literal(value: float) -> str:
    value = float(value)
    if math.isfinite(value):
        return format(value, ".17e")
    return f"0x{struct.unpack('<Q', struct.pack('<d', value))[0]:016X}"


def _f64(value: float) -> str:
    return f"{_f64_literal(value)} : f64"


def _f64_array(values: list[float]) -> str:
    return "array<f64: " + ", ".join(_f64_literal(v) for v in values) + ">"


def _target_attributes(
    target: Any,
    qubits: tuple[int, ...],
    n_qubits: int,
    *,
    t_start: float,
    t_end: float,
    num_steps: int,
    integrator: str,
) -> str:
    """Module attributes, in MLIR syntax, that the cuDensityMat lowering reads."""
    t1_ns = [0.0] * n_qubits
    t2_ns = [0.0] * n_qubits
    drive_scales = [1.0] * n_qubits
    for qi in qubits:
        qubit = target.qubits[qi]
        t1_ns[qi] = max(0.0, float(qubit.t1_us)) * 1.0e3
        t2_ns[qi] = max(0.0, float(qubit.t2_star_us)) * 1.0e3
        drive_scales[qi] = target.drive_amplitude_scale(qi)
    attrs = [
        f"qop.n_qubits = {n_qubits} : i64",
        f"qop.t_start = {_f64(t_start)}",
        f"qop.t_end = {_f64(t_end)}",
        f"qop.num_steps = {int(num_steps)} : i64",
        f'qop.integrator = "{integrator}"',
        "pulse.t1_times = [" + ", ".join(_f64(v) for v in t1_ns) + "]",
        "pulse.t2_times = [" + ", ".join(_f64(v) for v in t2_ns) + "]",
        f"pulse.drive_scale_rad_per_ns = {_f64_array(drive_scales)}",
    ]
    active = set(qubits)
    for kind, entries, strength in (
        ("coupling", target.couplings, "coupling_strength_hz"),
        ("crosstalk", target.crosstalk, "static_zz_hz"),
    ):
        entries = [e for e in entries if e.qubit_a in active and e.qubit_b in active]
        if entries:
            pairs = ", ".join(f"{e.qubit_a}, {e.qubit_b}" for e in entries)
            attrs.append(f"pulse.{kind}_pairs = array<i64: {pairs}>")
            attrs.append(
                f"pulse.{kind}_strength_hz = " + _f64_array([getattr(e, strength) for e in entries])
            )
    return "{" + ", ".join(attrs) + "}"
