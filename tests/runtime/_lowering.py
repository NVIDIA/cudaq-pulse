# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Lowering kernel calls on a test's own target, and GPU evolution of them."""

from types import SimpleNamespace

import cudaq_pulse as pulse
import numpy as np
from cudaq_pulse.kernel.decorator import QuditRef, QvecRef
from cudaq_pulse.runtime.simulate import _simulate_lowered
from cudaq_pulse.targets import Qubit, Target

EVOLVE_PASSES = ("verify", "canonicalize", "virtual_z", "fusion")


def _handles(dev, args):
    """Map ``qudit_ref()`` arguments to the device's handles, in order."""
    out, nxt = [], 0
    for a in args:
        if isinstance(a, QuditRef):
            index = nxt if a.index is None else a.index
            nxt = max(nxt, index) + 1 if a.index is None else nxt
            out.append(dev.qudit_handles[index])
        elif isinstance(a, QvecRef):
            out.append(dev.qudit_handles[nxt : nxt + len(a)])
            nxt += len(a)
        else:
            out.append(a)
    return out


def on(target, kernel, *args, passes=EVOLVE_PASSES):
    """The module of ``kernel(*args)`` on a device of *target*."""
    dev = pulse.device({"target": target})
    return pulse.compile(kernel, passes=passes).lower(*_handles(dev, args))


def evolve(lowered, *, t_start, t_end, num_steps, integrator="rk4", levels=2):
    """GPU evolution over ``[t_start, t_end]`` in *num_steps* steps."""
    result = _simulate_lowered(
        lowered,
        levels=levels,
        backend="gpu",
        t_start=t_start,
        t_end=t_end,
        dt_ns=(t_end - t_start) / num_steps,
        integrator=integrator,
    )
    rho = result.density_matrix
    state = rho
    # A pure final state is returned as a state vector, up to global phase.
    if abs(np.trace(rho @ rho).real - 1.0) < 1e-9:
        values, vectors = np.linalg.eigh(rho)
        state = vectors[:, -1] * np.sqrt(values[-1])
    return SimpleNamespace(final_state=state, times=np.linspace(t_start, t_end, num_steps + 1))


def frequency_target(qubit_freq_hz, clock_ghz=2.0):
    return Target(
        name="frequencies",
        qubits={q: Qubit(q, f, -300e6, 50.0, 30.0) for q, f in qubit_freq_hz.items()},
        clock_ghz=clock_ghz,
    )


def pulse_mlir(ir, *, qubit_freq_hz, clock_ghz=2.0):
    """The pulse MLIR a GPU run of ``ir = (kernel, *args)`` lowers."""
    return on(frequency_target(qubit_freq_hz, clock_ghz), *ir).mlir()
