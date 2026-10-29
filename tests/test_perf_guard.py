# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Compile-time and simulation-memory regression guard.

Skipped unless ``--run-benchmarks`` is given. Each case is timed best-of-N
and divided by a pure-Python calibration loop timed in the same process, so
the checked-in ratios transfer between machines and CI runners. A case fails
only if it is several times slower, relative to the calibration, than the
baseline in ``benchmarks/baselines/frontend_perf.json``. Refresh that file
with ``--update-perf-baseline`` after an intentional change.

Tests marked ``scaling`` compare two sizes of one program in one process
(time or peak memory), so they hold on any machine; CI fails on them.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.compiler import MachineModel
from cudaq_pulse.controller import ControllerLimits
from cudaq_pulse.kernel.bytecode_bridge import _trace_kernel_with_builder
from cudaq_pulse.kernel.ir_builder import Parameter
from cudaq_pulse.kernel.packed_ir_builder import PackedIRBuilder

from tests._handles import Q, ladder_target
from tests._ir import python_ir
from tests.kernel.ir_corpus import (
    bicycle_syndrome,
    dd_loop,
    parametric,
    surface_syndrome,
    tone_list_loop,
)

pytestmark = pytest.mark.benchmark

_ROOT = Path(__file__).resolve().parents[1]
BASELINE = _ROOT / "benchmarks" / "baselines" / "frontend_perf.json"
TOLERANCE = 3.0
# A retained loop's compile time must not grow with its trip count.
FLAT_LOOP_LIMIT = 2.0
# A sweep of new values costs about what repeating one value does, and a
# compile per value about 100x that: the limit sits between the two, far
# enough from both that a loaded runner does not cross it. What a new value
# costs exactly is gated by tools/perf-gate.py (bind_new_value).
RECOMPILE_LIMIT = 10.0


def _best(fn: Callable[[], object], min_reps: int = 5, budget_s: float = 0.25) -> float:
    fn()
    best = float("inf")
    reps = 0
    deadline = time.perf_counter() + budget_s
    while reps < min_reps or (time.perf_counter() < deadline and reps < 2000):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
        reps += 1
    return best


def _calibration() -> None:
    table: dict[int, float] = {}
    acc = 0.0
    for i in range(2000):
        table[i % 64] = table.get(i % 64, 0.0) + i * 0.5
        acc += len(str(i)) + table[i % 64] % 7
    assert acc > 0


def _compile_time_workloads():
    spec = importlib.util.spec_from_file_location(
        "compile_time", _ROOT / "benchmarks" / "compiler" / "compile_time.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return {name: (k, n, f) for name, k, n, f in module.WORKLOADS}


def _kernel_ir(kernel, n_qubits: int) -> Callable[[], object]:
    return lambda: python_ir(kernel, *[Q[i] for i in range(n_qubits)])


# A fresh compiled kernel holds no specializations, so these compile.
def _compile(kernel, n_qubits: int) -> Callable[[], object]:
    return lambda: pulse.compile(kernel).lower(*[Q[i] for i in range(n_qubits)])


def _fused(kernel, n: int) -> Callable[[], object]:
    return lambda: pulse.compile(kernel, passes=()).lower(Q[0:n])


def _packed_trace(kernel, n: int) -> Callable[[], object]:
    freq = {i: 5e9 for i in range(n)}
    fn = kernel.__wrapped__

    def run():
        builder = PackedIRBuilder(name="k", clock_ghz=2.0, qubit_freq_hz=freq)
        _trace_kernel_with_builder(fn, builder, [Q[0:n]])

    return run


def _cases() -> dict[str, Callable[[], object]]:
    workloads = _compile_time_workloads()
    cases = {}
    for name in ("bell", "qaoa4", "dd_cpmg8", "vqe_hea"):
        kernel, n_qubits, _ = workloads[name]
        cases[f"tier1_{name}_trace"] = _kernel_ir(kernel, n_qubits)
        cases[f"tier1_{name}_compile"] = _compile(kernel, n_qubits)
    for n in (8, 512):
        cases[f"dd_loop_n{n}"] = _kernel_ir(dd_loop(n), 1)
        cases[f"dd_loop_n{n}_compile"] = _compile(dd_loop(n), 1)
        cases[f"tone_list_loop_n{n}_compile"] = _fused(tone_list_loop(n), 2)
    surface5, n5 = surface_syndrome(5)
    surface9, n9 = surface_syndrome(9)
    bicycle, nb = bicycle_syndrome(6, 6)
    cases["surface_d5_compile"] = _fused(surface5, n5)
    cases["surface_d9_trace"] = _packed_trace(surface9, n9)
    cases["bicycle_6x6_compile"] = _fused(bicycle, nb)
    cases["parametric_compile"] = lambda: pulse.compile(parametric).lower(Q[0], 0.5, 2)
    cases["parametric_trace"] = lambda: _trace_kernel_with_builder(
        parametric.__wrapped__,
        PackedIRBuilder(name="p", clock_ghz=2.0, qubit_freq_hz={0: 5e9}),
        [Q[0], Parameter("amp", 0, "f64"), Parameter("n", 1, "i64")],
    )
    return cases


@pytest.fixture(scope="module")
def ratios() -> dict[str, float]:
    calib = _best(_calibration, min_reps=20)
    return {name: _best(run) / calib for name, run in _cases().items()}


def test_compile_time_within_baseline(ratios, request):
    if request.config.getoption("--update-perf-baseline"):
        BASELINE.write_text(
            json.dumps({k: round(v, 3) for k, v in sorted(ratios.items())}, indent=2) + "\n"
        )
    baseline = json.loads(BASELINE.read_text())
    assert sorted(baseline) == sorted(ratios)
    slow = {
        name: f"{ratio:.2f} vs baseline {baseline[name]:.2f}"
        for name, ratio in ratios.items()
        if ratio > baseline[name] * TOLERANCE
    }
    assert not slow, f"slower than {TOLERANCE}x the baseline ratio: {slow}"


def test_building_the_module_costs_less_than_tracing_the_kernel():
    kernel, n_qubits, _ = _compile_time_workloads()["bell"]
    best = {"trace_ms": float("inf"), "ffi_ms": float("inf")}
    for _ in range(50):
        metrics = _compile(kernel, n_qubits)().metrics
        for stage in best:
            best[stage] = min(best[stage], getattr(metrics, stage))
    assert best["ffi_ms"] < best["trace_ms"], best


@pytest.mark.scaling
def test_retained_loop_compile_time_is_flat(ratios):
    assert ratios["dd_loop_n512"] < ratios["dd_loop_n8"] * FLAT_LOOP_LIMIT
    assert ratios["dd_loop_n512_compile"] < ratios["dd_loop_n8_compile"] * FLAT_LOOP_LIMIT
    assert (
        ratios["tone_list_loop_n512_compile"]
        < ratios["tone_list_loop_n8_compile"] * FLAT_LOOP_LIMIT
    )


def _half_unused(n_qubits: int):
    @pulse.kernel
    def kernel(qubits):
        for q in range(n_qubits):
            line = get_drive_line(qubits[q])
            if q % 2 == 0:
                drive(line, square(20, 0.1))

    return kernel


@pytest.mark.scaling
def test_verify_time_is_linear_in_unused_lines():
    wide = pulse.device({"target": ladder_target(2000)}).qudit_handles

    def verify_ms(n_qubits: int) -> float:
        kernel = _half_unused(n_qubits)
        return min(
            pulse.compile(kernel, passes=("verify",)).lower(wide[0:n_qubits]).metrics.passes_ms
            for _ in range(5)
        )

    small, large = verify_ms(500), verify_ms(2000)
    assert large < small * 4 * FLAT_LOOP_LIMIT, (small, large)


@pytest.mark.scaling
def test_the_events_of_a_long_line_are_listed_in_linear_time():
    def program(n: int):
        @pulse.kernel
        def kernel(q):
            line = get_drive_line(q)
            for i in range(n):
                drive(line, square(8 + 0 * i, 0.1))

        # Without fuse_pulses, every drive is an event of its own.
        return pulse.compile(kernel, passes=("verify",), unroll_threshold=100_000).lower(Q[0])

    # Eight times the drives: about 10x the time when linear, 64x when not.
    times = {n: _best(lambda m=program(n): m.schedule, 3, 0) for n in (2000, 16000)}
    assert times[16000] < 8 * times[2000] * TOLERANCE, times


@pytest.mark.scaling
def test_a_long_program_is_exported_and_checked_in_linear_time():
    limits = ControllerLimits(
        issue_cycles=2,
        acquirers={0: 0},
        tones_per_line=2,
        waveform_samples=10**9,
        feedback_latency_cycles=1,
    )
    machine = MachineModel(readout_latency_cycles=1, controller=limits)

    def program(n: int):
        @pulse.kernel
        def kernel(q):
            line = get_drive_line(q)
            probe = get_readout_line(q)
            for i in range(n):
                drive(line, gaussian(16 + 0 * i, 0.1, 4.0))
                sync(line, probe)
                m = readout(probe, square(40, 0.05))
                sync(probe, line)
                if m:
                    drive(line, square(8, 0.1))

        compiled = pulse.compile(kernel, schedule="rcp", machine=machine, unroll_threshold=100_000)
        return compiled.lower(Q[0])

    # Eight times the events: about 8x the time when linear, 64x when not.
    times = {n: _best(program(n).export_controller, 3, 0) for n in (500, 4000)}
    assert times[4000] < 8 * times[500] * TOLERANCE, times


@pytest.mark.scaling
def test_a_wait_length_sweep_compiles_once():
    @pulse.kernel
    def t1(q, tau: float):
        d = get_drive_line(q)
        r = get_readout_line(q)
        drive(d, square(40, 0.1))
        wait(d, tau)
        sync(d, r)
        readout(r, square(100, 0.05))

    compiled = pulse.compile(t1)
    sweep = [float(4 * i + 1) for i in range(1000)]
    # A new length binds like a cached one; a compile costs about 100x as much.
    new = _best(lambda: [compiled.lower(Q[0], tau) for tau in sweep], 5, 0)
    assert compiled.cache_info().compiled == 1
    cached = _best(lambda: [compiled.lower(Q[0], sweep[0]) for _ in sweep], 5, 0)
    assert new < cached * RECOMPILE_LIMIT, (new, cached)


@pytest.mark.scaling
def test_a_sweep_of_a_wait_from_an_answer_and_a_float_compiles_once():
    from cudaq_pulse.classical import Float

    @pulse.classical(args=(Float(),), results=(Float(),), max_latency_ns=50, name="perf_offset")
    def offset(x):
        return x

    @pulse.kernel
    def t1(q, tau: float):
        d = get_drive_line(q)
        r = get_readout_line(q)
        drive(d, square(40, 0.1))
        wait(d, offset.submit(8.0).result() + tau, bounds=(0, 8000))
        sync(d, r)
        readout(r, square(100, 0.05))

    compiled = pulse.compile(t1)
    sweep = [float(4 * i + 1) for i in range(1000)]
    new = _best(lambda: [compiled.lower(Q[0], tau) for tau in sweep], 5, 0)
    assert compiled.cache_info().compiled == 1
    cached = _best(lambda: [compiled.lower(Q[0], sweep[0]) for _ in sweep], 5, 0)
    assert new < cached * RECOMPILE_LIMIT, (new, cached)


# Peak memory a simulation of two qutrits may add: their 9 x 9 density matrix
# and operators are tiny, so this only fails if the space grows with the
# qudits' indices (3^14 states for qudits 13 and 3).
SIMULATE_MEMORY_LIMIT_MB = 64

_SIMULATE_PAIRS = """
import json, resource, sys
import cudaq_pulse as pulse

q = pulse.device("transmon_krinner_17q").qudit_handles


@pulse.kernel
def pair(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    drive(da, gaussian(40, 0.08, 10.0))
    drive(db, gaussian(40, 0.03, 10.0))


scale = 1 if sys.platform == "darwin" else 1024
growth = {}
for a, b in json.loads(sys.argv[1]):
    lowered = pair.lower(q[a], q[b])
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = pulse.simulate(lowered, levels=3, t_end=2.0)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    growth[f"{a},{b}"] = [list(result.dims), (after - before) * scale / 2**20]
print(json.dumps(growth))
"""


@pytest.mark.scaling
def test_simulating_a_high_index_pair_takes_little_memory(tmp_path):
    script = tmp_path / "simulate_pairs.py"
    script.write_text(_SIMULATE_PAIRS)
    pairs = [(13, 3), (16, 8), (9, 1)]
    out = subprocess.run(
        [sys.executable, str(script), json.dumps(pairs)],
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    growth = json.loads(out.stdout)
    assert {k: dims for k, (dims, _) in growth.items()} == {f"{a},{b}": [3, 3] for a, b in pairs}
    assert all(mb < SIMULATE_MEMORY_LIMIT_MB for _, mb in growth.values()), growth


# ── Classical calls ──────────────────────────────────────────────────────────


def _classical_benchmarks():
    spec = importlib.util.spec_from_file_location(
        "classical_calls", _ROOT / "benchmarks" / "classical" / "classical_calls.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.scaling
def test_calls_compile_schedule_and_report_in_linear_time():
    bench = _classical_benchmarks()
    q = bench._device().qudit_handles[0]
    compile_s, report_s = {}, {}
    for n in (1000, 8000):
        kernel = bench.many_calls(n, 4)
        compile_s[n] = _best(lambda k=kernel: pulse.compile(k, schedule="asap").lower(q), 3, 0)
        lowered = pulse.compile(kernel, schedule="asap").lower(q)
        report_s[n] = _best(lowered.classical_report, 3, 0)
        events_s = _best(lambda m=lowered: m.schedule, 3, 0)
        assert len(lowered.classical_report()) == n
    # 8x the calls: linear is 8x, quadratic 64x.
    assert compile_s[8000] < 16 * compile_s[1000], compile_s
    assert report_s[8000] < 16 * report_s[1000], report_s
    assert events_s < report_s[8000] * 4


def test_a_cached_kernel_that_calls_lowers_in_microseconds():
    bench = _classical_benchmarks()
    dev = bench._device()
    q = dev.qudit_handles[0]
    compiled = pulse.compile(bench.decoded_reset, device=dev)
    compiled.lower(q)
    per_call = _best(lambda: [compiled.lower(q) for _ in range(100)], 5) / 100
    assert compiled.cache_info().compiled == 1
    assert per_call < 100e-6, per_call


def test_a_call_adds_little_to_simulation_on_the_same_timeline():
    bench = _classical_benchmarks()
    dev = bench._device(t1_us=20.0)
    q = dev.qudit_handles[0]
    plain = pulse.compile(bench.waiting_reset, device=dev).lower(q)
    called = pulse.compile(bench.decoded_reset, device=dev).lower(q)
    plus = [2**-0.5, 2**-0.5]
    a = _best(lambda: pulse.simulate(plain, initial_state=plus, dt_ns=0.5), 3, 0)
    b = _best(lambda: pulse.simulate(called, initial_state=plus, dt_ns=0.5), 3, 0)
    # Both take well under a millisecond, so allow the call's own binding and
    # rescheduling (about 0.3 ms) on top.
    assert b < 1.25 * a + 2e-3, (a, b)


@pytest.mark.scaling
def test_compiling_calls_takes_little_memory_per_call():
    import tracemalloc

    bench = _classical_benchmarks()
    q = bench._device().qudit_handles[0]
    peaks = {}
    for n in (500, 4000):
        kernel = bench.many_calls(n, 1)
        pulse.compile(kernel, schedule="asap").lower(q)
        tracemalloc.start()
        pulse.compile(kernel, schedule="asap").lower(q)
        peaks[n] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    per_call = (peaks[4000] - peaks[500]) / 3500
    assert per_call < 2048, per_call


def _measured_line(n):
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        for i in range(n):
            drive(d, square(8 + 0 * i, 0.1))
            shift_phase(d.tone, 0.1)
            sync(d, r)
            readout(r, square(40, 0.05))
            sync(d, r)

    return pulse.compile(k, unroll_threshold=100_000).lower(Q[0])


def test_walking_a_long_program_along_a_path_is_linear():
    # Every drive's frame comes from its tone, threaded through every op on
    # the line before it: following that chain back from each drive was
    # quadratic in the program's length.
    walk, export = {}, {}
    for n in (500, 4000):
        lowered = _measured_line(n)
        module = lowered._concrete().simulation_module(*lowered.options.schedule_args())
        walk[n] = _best(lambda m=module: m.next_measurement([]), 3, 0)
        export[n] = _best(lowered.export_controller, 2, 0)
    # 8x the ops: linear is 8x, quadratic 64x.
    assert walk[4000] < 16 * walk[500], walk
    assert export[4000] < 16 * export[500], export


def test_a_long_idle_simulates_in_constant_time():
    @pulse.kernel
    def idle(q, wait_cycles: int):
        d = get_drive_line(q)
        drive(d, square(40, 0.157))
        wait(d, wait_cycles)

    dev = pulse.device({"target": ladder_target(1)})
    compiled = pulse.compile(idle, device=dev)
    q = dev.qudit_handles[0]
    short, long = compiled.lower(q, 2_000), compiled.lower(q, 400_000)
    times = {
        name: _best(lambda m=m: pulse.simulate(m), 3, 0)
        for name, m in (("1 us", short), ("200 us", long))
    }
    assert times["200 us"] < 3 * times["1 us"] + 0.01, times


def test_an_adaptive_loop_of_fifty_rounds_simulates_one_path_per_shot():
    from cudaq_pulse.classical import Bits, Float, Link
    from cudaq_pulse.compiler import MachineModel

    @pulse.classical(
        args=(Bits(1), Float()), results=(Float(),), max_latency_ns=200, link="controller"
    )
    def next_wait(bit, w):
        return min(max(w * (1.5 if bit[0] else 0.75), 40.0), 200_000.0)

    @pulse.kernel
    def rounds(q, n: int):
        d = get_drive_line(q)
        r = get_readout_line(q)
        w = 20_000.0
        drive(d, square(40, 0.157))
        wait(d, 20_000)
        for _ in range(n):
            sync(d, r)
            m = readout(r, square(200, 0.05))
            w = next_wait.submit([m], w).result()
            sync(d, r)
            if m:
                drive(d, square(40, 0.157))
            drive(d, square(40, 0.157))
            wait(d, w, bounds=(40, 200_000))

    machine = MachineModel(links={"controller": Link(fixed_ns=4, feedback_ns=8)})
    dev = pulse.device({"target": ladder_target(1)})
    lowered = pulse.compile(rounds, device=dev, machine=machine).lower(dev.qudit_handles[0], 50)
    seconds = _best(
        lambda: pulse.simulate(lowered, shots=1, seed=1, branches="sampled", dt_ns=0.5), 2, 0
    )
    # Following all 2^50 paths is out of reach; one path takes about 0.1 s.
    assert seconds < 2.0, seconds
