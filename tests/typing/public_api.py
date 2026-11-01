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

from collections.abc import Mapping, Sequence
from typing import Any, assert_type

import cudaq_pulse as pulse
from cudaq_pulse.classical import Bits, ClassicalFunction, Pending
from cudaq_pulse.compiler import CompileMetrics, ScheduledEvent
from cudaq_pulse.controller import ControllerLimits
from cudaq_pulse.ops import Waveform

wave = pulse.ops.gaussian(40, 0.1, 10.0)
assert_type(wave, Waveform)
assert_type(pulse.ops.wf_scale(wave, 0.5), Waveform)
pulse.ops.wf_scale(0.5, wave)  # type: ignore[arg-type]
pulse.ops.wf_scale(waveform=wave, scalar=0.5)  # type: ignore[call-arg]
assert_type(pulse.ops.wf_add(wave, wave), Waveform)

dev = pulse.device("transmon_17q")
assert_type(dev, pulse.Device)
assert_type(pulse.device({"target": "transmon_17q"}), pulse.Device)
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
# Every module has a generic one, which binds a call's new values.
assert_type(lowered.generic, pulse.Lowered)
assert_type(pulse.simulate(lowered.generic, 0.7), pulse.SimulationResult)
assert_type(lowered.generic.bind([0.7]), pulse.Lowered)
# Times take units or numbers of ns.
assert_type(pulse.simulate(lowered, t_end=40 * pulse.ns, dt_ns=0.5), pulse.SimulationResult)
assert_type(pulse.simulate(rabi, dev.qudit_handles[0], 0.5), pulse.SimulationResult)
assert_type(pulse.simulate(compiled, dev.qudit_handles[0], 0.5), pulse.SimulationResult)


# pulse.classical is a module at run time; checkers see a callable decorator,
# and a classical function keeps the signature of its body.
@pulse.classical(max_latency_ns=400, link="gpu")
def decode(syndrome: Bits) -> bool:
    return bool(syndrome[0])


@pulse.classical(max_latency_ns=10)
def _scale(x: list[float], k: float) -> tuple[list[float], float]:
    return [k * v for v in x], k


@pulse.classical.external(max_latency_ns=10)
def table(index: pulse.Length[Bits, 2]) -> float:
    return float()


@pulse.classical.external(max_latency_ns=10)
def decode_surface(
    syndrome: pulse.Length[pulse.Bits, 8],
    counts: pulse.Length[list[int], 2],
) -> tuple[float, pulse.Length[pulse.Bits, 2]]:
    return (float(), pulse.Bits())


@pulse.classical.external(max_latency_ns=10)
def _halves(syndrome: pulse.Bits) -> tuple[float, pulse.Bits]:
    return (float(), syndrome[:2])


assert_type(pulse.Bits(), list[int])
assert_type(decode_surface.submit([1] * 8, [3, 4]).result(), tuple[float, list[int]])
assert_type(_halves.submit([0, 1, 1]).result(), tuple[float, list[int]])


def _double(x: float) -> float:
    return 2 * x


decode_as: ClassicalFunction[[Bits], bool] = decode
table_as: ClassicalFunction[[Bits], float] = table
double_as: ClassicalFunction[[float], float] = pulse.classical(_double, max_latency_ns=1.0)
double_too: ClassicalFunction[[float], float] = pulse.classical.function(_double, max_latency_ns=1)
mismatched: ClassicalFunction[[float], bool] = decode  # type: ignore[assignment]
assert_type(decode.submit([1]), Pending[bool])
assert_type(decode.submit([1]).result(), bool)
assert_type(decode.submit([1]).result(timeout=350 * pulse.ns), tuple[bool, bool])
assert_type(decode.submit([1]).result(timeout=700), tuple[bool, bool])
assert_type(_scale.submit([1.0], 2.0).result(), tuple[list[float], float])
assert_type(_scale.submit([1.0], 2.0).result(timeout=9), tuple[bool, list[float], float])
scale_3: ClassicalFunction[[list[float], float], tuple[list[float], float]] = _scale.specialize(x=3)
assert_type(_scale.specialize(x=3).submit([1.0], 2.0).result(), tuple[list[float], float])
assert_type(table.submit([0, 1]).result(), float)
assert_type(pulse.classical.random_index(4), ClassicalFunction[[], int])
assert_type(pulse.classical.function_id("decode"), int)
functions: list[ClassicalFunction[..., Any]] = [decode, _scale, table]
decode.submit(0.5)  # type: ignore[arg-type]
decode.submit([1], [0])  # type: ignore[call-arg]
_scale.submit([1.0], k="x")  # type: ignore[arg-type]
pulse.classical(max_latency_ns="soon")  # type: ignore[call-overload]
pulse.classical(_double)  # type: ignore[call-overload]
pulse.classical(_double, max_latency_ns=1, lengths=3)  # type: ignore[call-overload]  # remnant-ok
pulse.classical.external(max_latency_ns=1, args=())  # type: ignore[call-arg]  # remnant-ok

# pulse.controller is a module and the loader of controller descriptions.
box = pulse.controller("reference.controller", drive_lines=[0, 1])
assert_type(box, pulse.Controller)
assert_type(pulse.controller.ControllerLimits(issue_cycles=2), ControllerLimits)
assert_type(pulse.controller.FORMAT, str)
control = pulse.ControlSystem(controllers=[box], links={"feedback": pulse.Link(controllers=[box])})
assert_type(control.controllers, Sequence[pulse.Controller])
pulse.ControlSystem(line_delays_ns={("drive", 0): 2 * pulse.ns, ("readout", 0): 1.5})
pulse.Link(fixed_ns=100 * pulse.ns, feedback_ns=20.0, per_controller=True)
assert_type(box.supports_capability(pulse.Capability.BRANCH), bool)
cores = pulse.Processor(links=("feedback",), cycles_per_op={"div": 20}, unsupported=["trig"])
assert_type(cores.max_ops_total, int | None)
assert_type(cores.dispatch_keys, Mapping[str, int])
assert_type(pulse.Controller(processors=[cores]).processors, Sequence[pulse.Processor])
qpu = pulse.device({"quantum_system": "transmon_17q", "control_system": control})
assert_type(qpu.control_system, pulse.ControlSystem | None)
assert_type(qpu.as_dict(), dict[str, Any])
pulse.compile(rabi, device=qpu, machine=control)
pulse.controller(3)  # type: ignore[arg-type]
