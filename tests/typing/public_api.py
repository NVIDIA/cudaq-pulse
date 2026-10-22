# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Type-checked use of the public API (mypy --strict in CI; not run by pytest).

assert_type pins what users see; a wrong call is marked with the error code
mypy must report, and --warn-unused-ignores fails if it stops reporting it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, assert_type

import cudaq_pulse as pulse
from cudaq_pulse.ops import Waveform

qubit = pulse.qudit_ref()
assert_type(qubit, pulse.QuditRef)
assert_type(pulse.qvec_ref(2)[0], pulse.QuditRef)
assert_type(list(pulse.qvec_ref(2)), list[pulse.QuditRef])

wave = pulse.gaussian(40, 0.1, 10.0)
assert_type(wave, Waveform)
assert_type(pulse.wf_scale(wave, 0.5), Waveform)
pulse.wf_scale(0.5, wave)  # type: ignore[arg-type]
pulse.wf_scale(waveform=wave, scalar=0.5)  # type: ignore[call-arg]
assert_type(pulse.wf_add(wave, wave), Waveform)


def compile_one(kernel: Callable[..., Any]) -> None:
    compiled = pulse.compile(kernel, [qubit], qubit_freq_hz={0: 5e9})
    assert_type(compiled, pulse.CompiledKernel)
    assert_type(compiled.mlir, str)
    assert_type(compiled.metrics, pulse.CompileMetrics)
    assert_type(compiled.events(), list[pulse.ScheduledEvent])


def simulate_one(program: Any, target: Any) -> None:
    # Loaded lazily at runtime, but fully typed for checkers.
    assert_type(pulse.simulate(program, target=target), pulse.SimulationResult)
    assert_type(
        pulse.evolve(program, target=target, t_start=0.0, t_end=1.0, num_steps=1),
        pulse.EvolveResult,
    )
