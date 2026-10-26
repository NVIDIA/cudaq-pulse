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

from typing import assert_type

import cudaq_pulse as pulse
from cudaq_pulse.compiler import CompileMetrics, ScheduledEvent
from cudaq_pulse.ops import Waveform

wave = pulse.ops.gaussian(40, 0.1, 10.0)
assert_type(wave, Waveform)
assert_type(pulse.ops.wf_scale(wave, 0.5), Waveform)
pulse.ops.wf_scale(0.5, wave)  # type: ignore[arg-type]
pulse.ops.wf_scale(waveform=wave, scalar=0.5)  # type: ignore[call-arg]
assert_type(pulse.ops.wf_add(wave, wave), Waveform)

dev = pulse.get_device(pulse.Target.TRANSMON_KRINNER_17Q)
assert_type(dev, pulse.Device)
assert_type(pulse.device({"target": "transmon_krinner_17q"}), pulse.Device)
assert_type(dev.qudit_handles[0], pulse.Qudit)
assert_type(dev.qudit_handles[0:2], pulse.Register)
assert_type(dev.qudit_handles[[0, 1]], pulse.Register)
assert_type(dev.qudit_handles[0:2][0], pulse.Qudit)


@pulse.kernel
def rabi(q: pulse.Qudit, amp: float) -> None:
    pass


@pulse.kernel(cache_size=4)
def idle(q: pulse.Qudit) -> None:
    pass


# A kernel keeps the signature of its function: calls are checked against it.
kernels: list[pulse.Kernel[...]] = [rabi, idle]
assert_type(rabi.cache_info().size, int)
rabi(dev.qudit_handles[0])  # type: ignore[call-arg]
rabi(dev.qudit_handles[0], 0.5, 1)  # type: ignore[call-arg]
rabi(dev.qudit_handles[0], amp="high")  # type: ignore[arg-type]
rabi.lower(dev.qudit_handles[0], ampl=0.5)  # type: ignore[call-arg]

lowered = rabi.lower(dev.qudit_handles[0], 0.5)
assert_type(lowered, pulse.Lowered)
assert_type(lowered.mlir(), str)
assert_type(lowered.mlir("llvm"), str)
assert_type(lowered.metrics, CompileMetrics)
assert_type(lowered.schedule, list[ScheduledEvent])
assert_type(lowered.arguments, tuple[float, ...])

compiled = pulse.compile(rabi, schedule="asap", device=dev)
compiled_kernels: list[pulse.CompiledKernel[...]] = [compiled]
assert_type(compiled.lower(dev.qudit_handles[0], 0.5), pulse.Lowered)
compiled(dev.qudit_handles[0])  # type: ignore[call-arg]
assert_type(compiled.parameters, tuple[str, ...])
pulse.compile(rabi, schedule=1)  # type: ignore[arg-type]

# Loaded lazily at runtime, but fully typed for checkers.
assert_type(pulse.simulate(lowered, shots=10), pulse.SimulationResult)
