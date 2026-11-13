# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lowering kernel calls on a test's own target, and GPU evolution of them."""

from types import SimpleNamespace

import cudaq_pulse as pulse
import numpy as np
from cudaq_pulse.runtime.simulate import _simulate_lowered
from cudaq_pulse.targets import DeviceDescription, PhysicalQudit

EVOLVE_PASSES = (
    "verify",
    "fold_constants",
    "remove_unused",
    "merge_waits",
    "simplify_control_flow",
    "fold_phases",
    "fuse_pulses",
)


def _handles(dev, args):
    """Map handles of another device to *dev*'s handles of the same qudits."""
    out = []
    for a in args:
        if isinstance(a, pulse.Qudit):
            out.append(dev.qudits[a.index])
        elif isinstance(a, pulse.Register):
            out.append(dev.qudits[[q.index for q in a]])
        else:
            out.append(a)
    return out


def on(target, kernel, *args, passes=EVOLVE_PASSES):
    """The module of ``kernel(*args)`` on a device of *target*."""
    dev = pulse.quantum_system(target)
    return pulse.compile(kernel, passes=passes).lower(*_handles(dev, args))


def evolve(lowered, *, t_start, t_end, step_count, integrator="rk4", levels=2, initial_state=None):
    """GPU evolution over ``[t_start, t_end]`` in *step_count* steps."""
    result = _simulate_lowered(
        lowered,
        levels=levels,
        initial_state=initial_state,
        backend="gpu",
        t_start=t_start,
        t_end=t_end,
        dt_ns=(t_end - t_start) / step_count,
        integrator=integrator,
    )
    # The state the runtime evolved: a closed system's state vector, or the
    # density matrix of one with collapse operators. Purity cannot tell them
    # apart: damping keeps the ground state pure, and a fixed-step integrator
    # does not hold a closed system's norm exactly. A program that measures
    # ends in the mixture of its branches.
    states = result._states
    state = states[0] if len(states) == 1 else result.density_matrix
    return SimpleNamespace(final_state=state, times=np.linspace(t_start, t_end, step_count + 1))


def frequency_target(qudit_freq_hz, clock_ghz=2.0):
    return DeviceDescription(
        name="frequencies",
        qudits={q: PhysicalQudit(q, f, -300e6, 50.0, 30.0) for q, f in qudit_freq_hz.items()},
        clock_ghz=clock_ghz,
    )


def pulse_mlir(ir, *, qudit_freq_hz, clock_ghz=2.0):
    """The pulse MLIR a GPU run of ``ir = (kernel, *args)`` lowers."""
    return on(frequency_target(qudit_freq_hz, clock_ghz), *ir).mlir()
