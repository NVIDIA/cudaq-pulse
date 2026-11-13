#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The cases of the deterministic performance gate (tools/perf-gate.py).

Each case is one operation of the compiler or the runtime on fixed inputs,
run ``reps`` times. ``tools/perf-gate.py`` runs this file in three modes:

- ``--structural``: natively, printing each case's program sizes (op count,
  packed program bytes, export bytes) as JSON, with the cases' own
  tolerances. The sizes are the same on every platform.
- ``--measure LIB``: under Callgrind with instrumentation off. Around each
  case's repetitions it calls the Callgrind client requests in LIB (built
  from ``marks.c``), which zero the counters, instrument the repetitions and
  dump the counts under the case's name.
- ``--python-work``: natively, after the Callgrind runs, printing each
  case's Python-level work per repetition (``PYTHON_WORK``) as JSON. When
  the instructions move but this work does not, the cause is below Python:
  native code, or the layout of the interpreter's caches and memory.

Caches are handled the same way in every run: ``first_compile`` is the first
compile in the process (the MLIR context, lazy imports and the interpreter's
specializations are created in it); every other case runs its operation
``warmup`` times before it is measured, so it measures the steady state.
The cyclic garbage collector is run before each case and is off while it is
measured, so where a collection falls does not move counts between cases.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cudaq_pulse as pulse  # noqa: E402
from cudaq_pulse import Controller, ControlSystem  # noqa: E402
from cudaq_pulse.kernel import packed_ir_builder  # noqa: E402
from cudaq_pulse.quantum_systems import DeviceDescription, PhysicalQudit  # noqa: E402

Op = Callable[[int], object]


def _load(relative: str) -> Any:
    path = _ROOT / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


_workloads = _load("benchmarks/compiler/compile_time.py")
_calls = _load("benchmarks/classical/classical_calls.py")
_WORKLOADS = {name: (kernel, n) for name, kernel, n, _ in _workloads.WORKLOADS}

# Qudit i at 5 GHz + i * 100 MHz, as in tests/_handles.py.
_DEVICE = pulse.quantum_system(
    DeviceDescription(
        name="perf-gate",
        qudits={i: PhysicalQudit(i, 5.0e9 + 1.0e8 * i, -300e6, 50.0, 30.0) for i in range(64)},
    )
)
Q = _DEVICE.qudits


@dataclass(frozen=True)
class Case:
    name: str
    description: str
    # Builds the inputs (not measured) and returns the operation, called
    # with the repetition's index.
    setup: Callable[[], Op]
    reps: int
    warmup: int = 2
    # Program sizes of one run of the operation (see _structural).
    sizes: Callable[[object], dict[str, int]] | None = None
    # The fraction of its baseline's instructions above which the case
    # fails, if not the gate's default (tools/perf-gate.py, REGRESSION).
    # It must stay below 0.5%, the smallest regression the gate is there to
    # catch; the baseline records it, and taking a baseline copies it there.
    tolerance: float | None = None


def _compile(kernel: Any, args: tuple[Any, ...], **options: Any) -> Callable[[], Op]:
    # A fresh compiled kernel holds no specializations, so this compiles.
    return lambda: lambda i: pulse.compile(kernel, quantum_system=_DEVICE, **options).lower(*args)


def _op_count(lowered: object) -> dict[str, int]:
    return {"op_count": lowered.metrics.op_count}  # type: ignore[attr-defined]


def _json_bytes(program: object) -> dict[str, int]:
    text = json.dumps(program, sort_keys=True, separators=(",", ":"))
    return {"export_bytes": len(text.encode())}


def _schedule_size(events: Any) -> dict[str, int]:
    end = max(e.start_cycles + e.duration_cycles for e in events)
    return {"schedule_events": len(events), "schedule_cycles": end}


def _mlir_bytes(text: object) -> dict[str, int]:
    return {"mlir_bytes": len(str(text).encode())}


def _call_size(result: object) -> dict[str, int]:
    return {
        "call_args": len(_SUBMIT.args),
        "call_results": len(_SUBMIT.results),
        "result_milli": round(1000 * float(result)),  # type: ignore[arg-type]
    }


def _simulation_size(result: Any) -> dict[str, int]:
    return {
        "time_points": len(result.times_ns),
        "density_matrix_entries": int(result.density_matrix.size),
    }


# ── Kernels the existing benchmarks and guards do not provide ──────────────


@pulse.kernel
def _wait_sweep(q, tau: float):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, square(40, 0.1))
    wait(d, tau)
    sync(d, r)
    readout(r, square(100, 0.05))


@pulse.kernel
def _amplitude(q, amp: float):
    d = get_drive_line(q)
    drive(d, gaussian(40, amp, 10.0))
    drive(d, gaussian(44, amp, 10.0))


def _tone_list(iteration_count: int) -> Any:
    # tests/kernel/ir_corpus.py: tone_list_loop.
    @pulse.kernel
    def listed(qs):
        lines = [get_drive_line(q) for q in qs]
        for d, f in zip(lines, (5.0e9, 5.1e9), strict=True):
            set_frequency(d.tone, f)
        x = square(40, 0.047)
        for _ in range(iteration_count):
            for d in lines:
                drive(d, x)

    return listed


def _surface(d: int) -> tuple[Any, int]:
    # tests/kernel/ir_corpus.py: surface_syndrome.
    data_count = d * d
    ancilla_count = (d - 1) * (d - 1) + 2 * (d - 1)
    total_count = data_count + ancilla_count

    @pulse.kernel
    def round_(qudits):
        lines = []
        tones = []
        for i in range(total_count):
            line = get_drive_line(qudits[i])
            lines.append(line)
            tones.append(line.tone)
        h = drag(40, 0.25, 10.0, 0.5)
        cr_p = gaussian(160, 0.12, 40.0)
        cr_m = gaussian(160, -0.12, 40.0)
        for a in range(ancilla_count):
            anc = data_count + a
            row = a // (d - 1) if a < (d - 1) * (d - 1) else -1
            col = a % (d - 1) if a < (d - 1) * (d - 1) else a - (d - 1) * (d - 1)
            drive(lines[anc], h, tone=tones[anc])
            for dr in range(2):
                for dc in range(2):
                    r = row + dr
                    c = col + dc
                    if 0 <= r and r < d and 0 <= c and c < d:
                        q = r * d + c
                        sync(lines[anc], lines[q])
                        drive(lines[anc], cr_p, tone=tones[q])
                        drive(lines[anc], cr_m, tone=tones[q])

    return round_, total_count


def _feedback_program(n: int) -> Any:
    # tests/test_perf_guard.py: the export scaling guard's program.
    limits = Controller(
        instruction_cycles=2,
        detectors={0: 0},
        max_tones_per_line=2,
        waveform_memory_samples=10**9,
        feedback_latency_cycles=1,
    )
    machine = ControlSystem(controllers=[limits])

    @pulse.kernel
    def program(q):
        line = get_drive_line(q)
        probe = get_readout_line(q)
        for i in range(n):
            drive(line, gaussian(16 + 0 * i, 0.1, 4.0))
            sync(line, probe)
            m = readout(probe, square(40, 0.05))
            sync(probe, line)
            if m:
                drive(line, square(8, 0.1))

    compiled = pulse.compile(
        program,
        quantum_system=_DEVICE,
        schedule="rcp",
        control_system=machine,
        unroll_threshold=100_000,
    )
    return compiled.lower(Q[0])


@pulse.classical(max_latency_ns=50)
def perf_gate_submit(
    bits: pulse.Length[pulse.Bits, 1], a: float, b: float, c: float, d: float, e: float
) -> float:
    return a + b + c + d + e + bits[0]


_SUBMIT = perf_gate_submit


# ── Setups ─────────────────────────────────────────────────────────────────


def _bell_args() -> tuple[Any, ...]:
    return (Q[0], Q[1])


def _first_compile() -> Op:
    kernel, _ = _WORKLOADS["bell"]
    return lambda i: pulse.compile(kernel, quantum_system=_DEVICE).lower(*_bell_args())


def _bind_new_value() -> Op:
    compiled = pulse.compile(_wait_sweep, quantum_system=_DEVICE)
    compiled.lower(Q[0], 1.0)
    return lambda i: compiled.lower(Q[0], float(4 * i + 5))


def _bind_cached() -> Op:
    compiled = pulse.compile(_wait_sweep, quantum_system=_DEVICE)
    compiled.lower(Q[0], 1.0)
    return lambda i: compiled.lower(Q[0], 1.0)


def _specialize_wait() -> Op:
    compiled = pulse.compile(_wait_sweep, quantum_system=_DEVICE)
    compiled.lower(Q[0], 1.0)
    # A new length retimes the module: its schedule specializes it.
    return lambda i: compiled.lower(Q[0], float(4 * i + 5)).schedule


def _specialize_amplitude() -> Op:
    compiled = pulse.compile(_amplitude, quantum_system=_DEVICE)
    compiled.lower(Q[0], 0.1)
    return lambda i: compiled.lower(Q[0], 0.01 * (i + 1)).mlir()


def _submit() -> Op:
    return lambda i: _SUBMIT.submit([1], 0.1, 0.2, 0.3, 0.4, 0.5).result()


def _export_controller() -> Op:
    lowered = _feedback_program(64)
    return lambda i: lowered.export_controller()


def _export_controller_sampled() -> Op:
    lowered = _feedback_program(16)
    rate = 2 * lowered.quantum_system.clock_ghz
    return lambda i: lowered.export_controller(rate)


def _export_emulation() -> Op:
    lowered = pulse.compile(
        _calls.many_calls(64, 4), quantum_system=_DEVICE, schedule="asap"
    ).lower(Q[0])
    return lambda i: lowered.export_emulation()


def _simulate_setup() -> Op:
    @pulse.kernel
    def pi_pulse(q):
        d = get_drive_line(q)
        drive(d, square(8, 0.1))

    lowered = pulse.compile(pi_pulse, quantum_system=_DEVICE).lower(Q[0])
    # Eight cycles at one step per cycle: the run is mostly its setup. The
    # seed keeps its generator from being seeded with the system's entropy,
    # different in every run.
    return lambda i: pulse.simulate(lowered, levels=2, dt_ns=0.5, seed=0)


def _workload(name: str) -> Callable[[], Op]:
    kernel, n = _WORKLOADS[name]
    return _compile(kernel, tuple(Q[i] for i in range(n)))


def _surface_d5() -> Op:
    kernel, n = _surface(5)
    return _compile(kernel, (Q[0:n],))()


# The order is the order of measurement: first_compile must stay first.
CASES: tuple[Case, ...] = (
    Case(
        "first_compile",
        "the first compile in the process (bell)",
        _first_compile,
        1,
        0,
        sizes=_op_count,
    ),
    *(
        Case(f"compile_{name}", f"compile {name}", _workload(name), reps, sizes=_op_count)
        for name, reps in (
            ("bell", 10),
            ("cnot_cr", 10),
            ("qaoa4", 5),
            ("syndrome", 5),
            ("dd_cpmg8", 10),
            ("vqe_hea", 5),
        )
    ),
    Case(
        "compile_tone_list",
        "compile a kernel over a list of qudits, tones kept in a list (loop of 64)",
        _compile(_tone_list(64), (Q[0:2],)),
        5,
        sizes=_op_count,
    ),
    Case(
        "compile_surface_d5",
        "compile a distance-5 surface-code syndrome round (49 qudits, large)",
        _surface_d5,
        1,
        1,
        sizes=_op_count,
    ),
    Case(
        "compile_calls_small",
        "compile active reset with a decoder call",
        _compile(_calls.decoded_reset, (Q[0],)),
        5,
        sizes=_op_count,
    ),
    Case(
        "compile_calls_large",
        "compile 256 classical calls on 4 links",
        _compile(_calls.many_calls(256, 4), (Q[0],), schedule="asap"),
        1,
        1,
        sizes=_op_count,
    ),
    Case("bind_cached", "lower() of a cached call", _bind_cached, 200, sizes=_op_count),
    Case(
        "bind_new_value",
        "lower() with a new wait length (no compile)",
        _bind_new_value,
        200,
        sizes=_op_count,
    ),
    Case(
        "specialize_wait",
        "a new wait length's schedule (specializes the module)",
        _specialize_wait,
        20,
        sizes=_schedule_size,
    ),
    Case(
        "specialize_amplitude",
        "a new amplitude's MLIR text (specializes the module)",
        _specialize_amplitude,
        20,
        sizes=_mlir_bytes,
    ),
    Case(
        "submit",
        "a classical function's submit().result() outside a kernel (1 bit + 5 floats)",
        _submit,
        500,
        sizes=_call_size,
    ),
    Case(
        "export_controller",
        "controller program of 64 readouts with feedback",
        _export_controller,
        3,
        sizes=_json_bytes,
    ),
    Case(
        "export_controller_sampled",
        "controller program of 16 readouts with feedback, with envelope samples",
        _export_controller_sampled,
        3,
        sizes=_json_bytes,
    ),
    Case(
        "export_emulation",
        "emulator program of 64 classical calls on 4 links",
        _export_emulation,
        3,
        sizes=_json_bytes,
    ),
    Case(
        "simulate_setup",
        "simulate() of an 8-cycle drive (mostly setup)",
        _simulate_setup,
        3,
        sizes=_simulation_size,
        # The one case whose identical Callgrind runs have differed: by up
        # to 0.0098% (169 instructions per repetition) on CI, where every
        # other case repeated exactly. Its limit is the default 0.4% plus
        # four times that difference.
        tolerance=0.0045,
    ),
)


def _structural() -> dict[str, dict[str, int]]:
    """Each case's program sizes: those its ``sizes`` reads from one run's
    result, and ``packed_bytes``, the packed programs the run traced."""
    traced: list[int] = []
    get_buffer = packed_ir_builder.PackedIRBuilder.get_buffer

    def counting(self: Any) -> Any:
        buffer = get_buffer(self)
        traced.append(len(buffer) * buffer.itemsize)
        return buffer

    packed_ir_builder.PackedIRBuilder.get_buffer = counting  # type: ignore[method-assign]
    try:
        out = {}
        for case in CASES:
            op = case.setup()
            traced.clear()
            result = op(0)
            sizes = dict(case.sizes(result)) if case.sizes else {}
            if traced:
                sizes["packed_bytes"] = sum(traced)
            out[case.name] = sizes
        return out
    finally:
        packed_ir_builder.PackedIRBuilder.get_buffer = get_buffer  # type: ignore[method-assign]


def _measure(library: str, only: set[str] | None) -> dict[str, int]:
    import ctypes

    marks = ctypes.CDLL(library)
    marks.perf_gate_stop.argtypes = [ctypes.c_char_p]
    reps = {}
    for case in CASES:
        if only is not None and case.name not in only and case.name != "first_compile":
            continue
        op = case.setup()
        # Indices past the measured ones: a case of new values sees new ones.
        for i in range(case.warmup):
            op(case.reps + i)
        gc.collect()
        gc.disable()
        marks.perf_gate_start()
        for i in range(case.reps):
            op(i)
        marks.perf_gate_stop(case.name.encode())
        gc.enable()
        reps[case.name] = case.reps
    return reps


# The Python-level work of a case, counted by sys.monitoring (3.12+):
# bytecodes executed, Python functions started, and calls from Python code
# to anything else (C functions and methods, types, other callables).
PYTHON_WORK = ("bytecodes", "python_calls", "c_calls")


def _python_work(only: set[str] | None) -> dict[str, dict[str, float]]:
    """Each case's Python-level work per repetition, to a thousandth, run
    as ``_measure`` runs the cases (natively: the callbacks would weigh on
    Callgrind's counts), or nothing before Python 3.12."""
    monitoring = getattr(sys, "monitoring", None)
    if monitoring is None:
        return {}
    tool = monitoring.PROFILER_ID
    events = monitoring.events
    counts = [0, 0, 0]
    python = (types.FunctionType, types.MethodType)

    def on_instruction(code: object, offset: int) -> None:
        counts[0] += 1

    def on_start(code: object, offset: int) -> None:
        counts[1] += 1

    def on_call(code: object, offset: int, callable: object, arg0: object) -> None:
        if not isinstance(callable, python):
            counts[2] += 1

    monitoring.use_tool_id(tool, "perf gate")
    monitoring.register_callback(tool, events.INSTRUCTION, on_instruction)
    monitoring.register_callback(tool, events.PY_START, on_start)
    monitoring.register_callback(tool, events.CALL, on_call)
    counted = events.INSTRUCTION | events.PY_START | events.CALL
    work = {}
    try:
        for case in CASES:
            if only is not None and case.name not in only and case.name != "first_compile":
                continue
            op = case.setup()
            for i in range(case.warmup):
                op(case.reps + i)
            gc.collect()
            gc.disable()
            counts[:] = [0, 0, 0]
            monitoring.set_events(tool, counted)
            for i in range(case.reps):
                op(i)
            monitoring.set_events(tool, 0)
            gc.enable()
            work[case.name] = {
                k: round(n / case.reps, 3) for k, n in zip(PYTHON_WORK, counts, strict=True)
            }
    finally:
        monitoring.set_events(tool, 0)
        monitoring.free_tool_id(tool)
    return work


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--structural", action="store_true", help="print program sizes and tolerances as JSON"
    )
    mode.add_argument("--measure", metavar="LIB", help="run under Callgrind with these marks")
    mode.add_argument(
        "--python-work",
        action="store_true",
        help="print each case's bytecodes, Python calls and C calls per repetition as JSON",
    )
    mode.add_argument("--list", action="store_true", help="print the cases")
    parser.add_argument("--only", nargs="+", help="measure only these cases")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"{case.name:28s} x{case.reps:<4d} {case.description}")
        return 0
    if args.structural:
        tolerances = {case.name: case.tolerance for case in CASES if case.tolerance is not None}
        out = {"sizes": _structural(), "tolerances": tolerances}
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0
    only = set(args.only) if args.only else None
    if args.python_work:
        print(json.dumps(_python_work(only)))
        return 0
    print(json.dumps(_measure(args.measure, only)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
