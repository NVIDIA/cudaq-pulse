# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile-time and simulation-memory regression guard.

Skipped unless ``--run-benchmarks`` is given. Each case is timed best-of-N
and divided by a pure-Python calibration loop timed in the same process, so
the checked-in ratios transfer between machines and CI runners. A case fails
only if it is several times slower, relative to the calibration, than the
baseline in ``benchmarks/baselines/frontend_perf.json``. Refresh that file
with ``--update-perf-baseline`` after an intentional change.

Tests marked ``scaling`` compare two sizes of one program in one process
(the work counted, time or peak memory), so they hold on any machine; CI
fails on them. The Python work of each size is counted, and must be the same
or in proportion to the size; the times, which also take in native code, are
the best of many and are bounded loosely enough that a loaded runner cannot
fail them, while a cost that grows with the size still does.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import Controller, ControlSystem
from cudaq_pulse.kernel._emitter import PulseIREmitter
from cudaq_pulse.kernel._unroll import DEFAULT_UNROLL_THRESHOLD
from cudaq_pulse.kernel.bytecode_bridge import _trace_kernel_with_builder
from cudaq_pulse.kernel.ir_builder import Parameter
from cudaq_pulse.kernel.packed_ir_builder import PackedIRBuilder

from tests._handles import Q, ladder_target
from tests._ir import python_ir
from tests._tools import missing
from tests._work import python_work
from tests.kernel.ir_corpus import (
    bicycle_syndrome,
    dd_loop,
    parametric,
    surface_syndrome,
    tone_list_loop,
)

_guard = importlib.import_module("cudaq_pulse.kernel._guard")

pytestmark = pytest.mark.benchmark

_ROOT = Path(__file__).resolve().parents[1]
BASELINE = _ROOT / "benchmarks" / "baselines" / "frontend_perf.json"
TOLERANCE = 3.0
# A rolled loop's cost must not grow with its trip count: loops of ROLLED_TRIPS
# trips, a hundred thousand times apart, do the same Python work and take
# times within FLAT_LOOP_LIMIT of each other. Native code that went over the
# trips at a nanosecond each would add 10 ms, over 30 times a compile.
ROLLED_TRIPS = (100, 10**7)
FLAT_LOOP_LIMIT = 10.0
# A build tree stages the package in build/<dir>/python and cudaq-pulse-opt
# in build/<dir>/bin.
_BUILD_BIN = Path(pulse.__file__).parent.parent.parent / "bin"


def _best(
    fn: Callable[[], object],
    min_reps: int = 5,
    budget_s: float = 0.25,
    clock: Callable[[], float] = time.perf_counter,
) -> float:
    fn()
    best = float("inf")
    reps = 0
    deadline = time.perf_counter() + budget_s
    while reps < min_reps or (time.perf_counter() < deadline and reps < 2000):
        t0 = clock()
        fn()
        best = min(best, clock() - t0)
        reps += 1
    return best


def _assert_flat(what: str, runs: dict[int, Callable[[], object]]) -> None:
    """The run of each size in *runs* does the same Python work, and the
    largest takes less than FLAT_LOOP_LIMIT times the smallest's time."""
    small, large = min(runs), max(runs)
    work = {n: python_work(runs[n]) for n in (small, large)}
    seconds = {n: _best(runs[n], 20) for n in (small, large)}
    assert work[large] == work[small], (what, work)
    assert seconds[large] < seconds[small] * FLAT_LOOP_LIMIT, (what, seconds)


def _assert_linear(what: str, runs: dict[int, Callable[[], object]], limit: float) -> None:
    """The run of each size in *runs* does Python work in at most proportion
    to its size, and the largest takes less than *limit* times the smallest's
    CPU time; a cost quadratic in the size takes the ratio of sizes squared.
    The runs take milliseconds, long enough for a loaded machine to stretch
    their wall-clock times unevenly."""
    small, large = min(runs), max(runs)
    work = {n: python_work(runs[n]) for n in (small, large)}
    seconds = {n: _best(runs[n], 5, clock=time.process_time) for n in (small, large)}
    for count in work[small]:
        assert work[large][count] <= work[small][count] * large / small, (what, work)
    assert seconds[large] < seconds[small] * limit, (what, seconds)


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


def _kernel_ir(kernel, qudit_count: int) -> Callable[[], object]:
    return lambda: python_ir(kernel, *[Q[i] for i in range(qudit_count)])


# A fresh compiled kernel holds no specializations, so these compile.
def _compile(kernel, qudit_count: int) -> Callable[[], object]:
    return lambda: pulse.compile(kernel).lower(*[Q[i] for i in range(qudit_count)])


def _fused(kernel, n: int) -> Callable[[], object]:
    return lambda: pulse.compile(kernel, passes=()).lower(Q[0:n])


def _packed_trace(kernel, n: int) -> Callable[[], object]:
    freq = {i: 5e9 for i in range(n)}
    fn = kernel.__wrapped__

    def run():
        builder = PackedIRBuilder(name="k", clock_ghz=2.0, qudit_freq_hz=freq)
        _trace_kernel_with_builder(fn, builder, [Q[0:n]])

    return run


def _cases() -> dict[str, Callable[[], object]]:
    workloads = _compile_time_workloads()
    cases = {}
    for name in ("bell", "qaoa4", "dd_cpmg8", "vqe_hea"):
        kernel, qudit_count, _ = workloads[name]
        cases[f"tier1_{name}_trace"] = _kernel_ir(kernel, qudit_count)
        cases[f"tier1_{name}_compile"] = _compile(kernel, qudit_count)
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
        PackedIRBuilder(name="p", clock_ghz=2.0, qudit_freq_hz={0: 5e9}),
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
    kernel, qudit_count, _ = _compile_time_workloads()["bell"]
    best = {"trace_ms": float("inf"), "ffi_ms": float("inf")}
    for _ in range(50):
        metrics = _compile(kernel, qudit_count)().metrics
        for stage in best:
            best[stage] = min(best[stage], getattr(metrics, stage))
    assert best["ffi_ms"] < best["trace_ms"], best


def _compile_work(kernel, n: int, args, **options) -> dict[str, object]:
    """What compiling *kernel* does, counted: the calls the tracer
    dispatched, the ops and packed bytes it traced, and the scheduled
    module, with the trip count *n* written as ``N``."""
    work = {"calls": 0, "packed_bytes": 0}
    dispatch, get_buffer = PulseIREmitter._dispatch_call, PackedIRBuilder.get_buffer

    def counted_dispatch(self, *a, **k):
        work["calls"] += 1
        return dispatch(self, *a, **k)

    def counted_buffer(self):
        buffer = get_buffer(self)
        work["packed_bytes"] += len(buffer) * buffer.itemsize
        return buffer

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(PulseIREmitter, "_dispatch_call", counted_dispatch)
        patch.setattr(PackedIRBuilder, "get_buffer", counted_buffer)
        lowered = pulse.compile(kernel, **options).lower(*args)
    work["traced_ops"] = lowered.metrics.op_count
    work["module"] = re.sub(rf"(?<!\d){n}(?!\d)", "N", lowered.mlir())
    return work


@pytest.mark.scaling
def test_retained_loop_compile_work_is_flat():
    # A loop over the unroll threshold is kept: compiling it does the same
    # work at any trip count, and no more than the unrolled loop at the
    # threshold does.
    for make, args, options in (
        (dd_loop, (Q[0],), {}),
        (tone_list_loop, (Q[0:2],), {"passes": ()}),
    ):
        retained = [_compile_work(make(n), n, args, **options) for n in (512, 4096)]
        assert retained[0] == retained[1], make.__name__
        n = DEFAULT_UNROLL_THRESHOLD
        unrolled = _compile_work(make(n), n, args, **options)
        for count in ("calls", "packed_bytes", "traced_ops"):
            assert retained[0][count] <= unrolled[count], (make.__name__, count)


def _half_unused(qudit_count: int):
    @pulse.kernel
    def kernel(qudits):
        for q in range(qudit_count):
            line = get_drive_line(qudits[q])
            if q % 2 == 0:
                drive(line, square(20, 0.1))

    return kernel


@pytest.mark.scaling
def test_verify_does_constant_work_per_unused_line(tmp_path):
    # Each unused line is one warning that names its qubit. A warning that
    # attached its op would print it, and printing an op numbers every value
    # of its function: the whole program per unused line. cudaq-pulse-opt
    # shows the warnings that compiles from Python drop.
    opt = shutil.which("cudaq-pulse-opt", path=str(_BUILD_BIN)) or shutil.which("cudaq-pulse-opt")
    if opt is None:
        missing("cudaq-pulse-opt", "cudaq-pulse-opt is not built or on PATH")
    wide = pulse.quantum_system(ladder_target(2000)).qudits
    for n in (500, 2000):
        module = tmp_path / f"half_unused_{n}.mlir"
        module.write_text(pulse.compile(_half_unused(n), passes=()).lower(wide[0:n]).mlir())
        stderr = subprocess.run(
            [opt, "--pulse-verify", str(module), "-o", os.devnull],
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        ).stderr
        warnings = re.findall(r"^\S+: warning: (.*)$", stderr, re.M)
        assert warnings == [
            f"drive line allocated but never used (qudit {q})" for q in range(1, n, 2)
        ]
        # The message, the source line and its caret: nothing attached.
        assert stderr.count("\n") == 3 * len(warnings), stderr[:2000]


@pytest.mark.scaling
def test_compiling_a_kernel_that_reads_a_dict_is_flat_in_its_size():
    def program(n: int):
        table = dict.fromkeys(map("entry{}".format, range(n)), [0.1, 40])
        namespace = {"table": table}
        exec(
            "def kernel(q):\n"
            "    line = get_drive_line(q)\n"
            "    entry = table['entry7']\n"
            "    drive(line, square(entry[1], entry[0]))\n",
            namespace,
        )
        kernel = pulse.kernel(namespace["kernel"])
        compiled = pulse.compile(kernel)
        # Past the call that upgrades the guard: a call measures the steady state.
        for _ in range(1 + _guard.UPGRADE_AFTER):
            compiled.lower(Q[0])
        return lambda: pulse.compile(kernel).lower(Q[0]), lambda: compiled.lower(Q[0])

    # Copying a million entries, or hashing them, takes some 10 ms.
    programs = {n: program(n) for n in (16, 10**6)}
    _assert_flat("compile", {n: runs[0] for n, runs in programs.items()})
    _assert_flat("call", {n: runs[1] for n, runs in programs.items()})


@pytest.mark.scaling
def test_a_wide_sync_traces_in_linear_time():
    wide = pulse.quantum_system(ladder_target(4000)).qudits

    def program(n: int):
        @pulse.kernel
        def kernel(qudits):
            lines = [get_drive_line(q) for q in qudits]
            sync(*lines)
            for line in lines:
                drive(line, square(8, 0.1))

        return lambda: pulse.compile(kernel).lower(wide[0:n])

    # Eight times the lines: about 8x the time when linear, 64x when not.
    runs = {n: program(n) for n in (500, 4000)}
    work = {n: python_work(run) for n, run in runs.items()}
    times = {n: min(run().metrics.trace_ms for _ in range(5)) for n, run in runs.items()}
    assert all(work[4000][count] <= 8 * work[500][count] for count in work[500]), work
    assert times[4000] < 16 * times[500], times


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
    runs = {n: lambda m=program(n): m.schedule for n in (2000, 16000)}  # noqa: B008
    _assert_linear("schedule", runs, 8 * TOLERANCE)


@pytest.mark.scaling
def test_a_long_program_is_exported_and_checked_in_linear_time():
    limits = Controller(
        instruction_cycles=2,
        detectors={0: 0},
        max_tones_per_line=2,
        waveform_memory_samples=10**9,
        feedback_latency_cycles=1,
    )
    machine = ControlSystem(controllers=[limits])

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

        compiled = pulse.compile(
            kernel, schedule="rcp", control_system=machine, unroll_threshold=100_000
        )
        return compiled.lower(Q[0])

    # Eight times the events: about 8x the time when linear, 64x when not.
    runs = {n: program(n).export_controller for n in (500, 4000)}
    _assert_linear("export", runs, 8 * TOLERANCE)


def _sweep(compiled, sweep: list[float], monkeypatch) -> None:
    """Lower *compiled* at each length of *sweep*: after the first, none may
    trace the kernel or build a native module, and each shares the first's.

    These count what a compile does rather than time it, so a loaded runner
    can't fail them. What binding a new value costs is measured by
    tools/perf-gate.py (bind_new_value)."""
    from cudaq_pulse import compiler
    from cudaq_pulse.kernel import bytecode_bridge

    first = compiled.lower(Q[0], sweep[0])
    work = {"traces": 0, "modules": 0}
    trace, builder = bytecode_bridge._trace_kernel_with_builder, compiler.PulseModuleBuilder

    def counted_trace(*args, **kwargs):
        work["traces"] += 1
        return trace(*args, **kwargs)

    def counted_builder():
        work["modules"] += 1
        return builder()

    monkeypatch.setattr(bytecode_bridge, "_trace_kernel_with_builder", counted_trace)
    monkeypatch.setattr(compiler, "PulseModuleBuilder", counted_builder)
    lowered = [compiled.lower(Q[0], tau) for tau in sweep[1:]]
    monkeypatch.undo()

    assert work == {"traces": 0, "modules": 0}
    assert compiled.cache_info().compiled == 1
    assert all(low._module is first._module for low in lowered)
    assert [low.arguments for low in lowered] == [(tau,) for tau in sweep[1:]]


@pytest.mark.scaling
def test_a_rolled_loop_is_checked_and_exported_rolled_in_constant_time():
    limits = Controller(
        instruction_cycles=2, detectors={0: 0}, max_tones_per_line=2, waveform_memory_samples=10**9
    )
    machine = ControlSystem(controllers=[limits])

    def program(n: int):
        @pulse.kernel
        def kernel(q):
            line = get_drive_line(q)
            probe = get_readout_line(q)
            for _ in range(n):
                drive(line, gaussian(16, 0.1, 4.0))
                shift_phase(line.tone, 0.1)
                sync(line, probe)
                readout(probe, square(40, 0.05))
                sync(probe, line)

        return kernel

    compiles, exports, size = {}, {}, {}
    for n in ROLLED_TRIPS:
        kernel = program(n)
        compiles[n] = lambda k=kernel: pulse.compile(k, control_system=machine).lower(Q[0])
        lowered = compiles[n]()
        exports[n] = lambda m=lowered: m.export_controller(version=2)
        size[n] = len(json.dumps(lowered.export_controller(version=2)))
    # The same cost at any trip count when the loop stays rolled.
    _assert_flat("compile", compiles)
    _assert_flat("export", exports)
    small, large = ROLLED_TRIPS
    assert size[large] < size[small] + 64, size


@pytest.mark.scaling
def test_duration_budgets_of_a_rolled_loop_are_checked_in_constant_time():
    @pulse.kernel(schedule="asap", max_duration_cycles=1000)
    def cycle(q):
        line = get_drive_line(q)
        probe = get_readout_line(q)
        drive(line, gaussian(16, 0.1, 4.0))
        sync(line, probe)
        readout(probe, square(40, 0.05))

    def program(n: int):
        @pulse.kernel(max_duration_cycles=10**9)
        def kernel(q):
            for _ in range(n):
                cycle(q)

        return kernel

    runs = {n: lambda k=program(n): pulse.compile(k).lower(Q[0]) for n in ROLLED_TRIPS}  # noqa: B008
    _assert_flat("compile", runs)


@pytest.mark.scaling
def test_a_waveform_memory_of_a_rolled_loop_is_checked_in_constant_time():
    from cudaq_pulse.controller import check

    limits = Controller(waveform_memory_samples=4096, waveform_memory="shared")

    def program(n: int):
        @pulse.kernel
        def kernel(q):
            line = get_drive_line(q)
            for _ in range(n):
                drive(line, gaussian(16, 0.1, 4.0))
                drive(line, square(8, 0.2))
                shift_phase(line.tone, 0.1)

        return kernel

    runs = {}
    for n in ROLLED_TRIPS:
        lowered = pulse.compile(program(n)).lower(Q[0])
        assert check(lowered, limits) == []
        runs[n] = lambda m=lowered: check(m, limits)
    _assert_flat("check", runs)


@pytest.mark.scaling
def test_a_wait_length_sweep_compiles_once(monkeypatch):
    @pulse.kernel
    def t1(q, tau: float):
        d = get_drive_line(q)
        r = get_readout_line(q)
        drive(d, square(40, 0.1))
        wait(d, tau)
        sync(d, r)
        readout(r, square(100, 0.05))

    _sweep(pulse.compile(t1), [float(4 * i + 1) for i in range(1000)], monkeypatch)


@pytest.mark.scaling
def test_exporting_a_call_again_does_not_specialize_again():
    @pulse.kernel
    def pulses(q, amp: float):
        d = get_drive_line(q)
        for _ in range(16):
            drive(d, gaussian(40, amp, 10.0))
            shift_phase(d.tone, 0.1)

    compiled = pulse.compile(pulses, unroll_threshold=100_000)
    values = iter(1e-3 * k for k in range(1, 10**6))
    new = _best(lambda: compiled.lower(Q[0], next(values)).export_controller(), 20)
    again = _best(lambda: compiled.lower(Q[0], 0.5).export_controller(), 20)
    assert compiled.lower(Q[0], 0.5)._concrete() is compiled.lower(Q[0], 0.5)._concrete()
    # Specializing is most of an export of new values: about 3x the time.
    assert again < new / 1.5, (again, new)


@pytest.mark.scaling
def test_a_sweep_of_a_wait_from_an_answer_and_a_float_compiles_once(monkeypatch):

    @pulse.classical(max_latency_ns=50, name="perf_offset")
    def offset(x: float) -> float:
        return x

    @pulse.kernel
    def t1(q, tau: float):
        d = get_drive_line(q)
        r = get_readout_line(q)
        drive(d, square(40, 0.1))
        wait(d, offset.submit(8.0).result() + tau, bounds=(0, 8000))
        sync(d, r)
        readout(r, square(100, 0.05))

    _sweep(pulse.compile(t1), [float(4 * i + 1) for i in range(1000)], monkeypatch)


# Peak memory a simulation of two qutrits may add: their 9 x 9 density matrix
# and operators are tiny, so this only fails if the space grows with the
# qudits' indices (3^14 states for qudits 13 and 3).
SIMULATE_MEMORY_LIMIT_MB = 64

_SIMULATE_PAIRS = """
import json, resource, sys
import cudaq_pulse as pulse

q = pulse.quantum_system("transmon_17q").qudits


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
    q = bench._device().qudits[0]
    compiles, reports = {}, {}
    for n in (1000, 8000):
        kernel = bench.many_calls(n, 4)
        compiles[n] = lambda k=kernel: pulse.compile(k, schedule="asap").lower(q)
        lowered = compiles[n]()
        reports[n] = lowered.classical_report
        assert len(lowered.classical_report()) == n
    # 8x the calls: linear is 8x, quadratic 64x.
    _assert_linear("compile", compiles, 16)
    _assert_linear("report", reports, 16)
    events_s = _best(lambda m=lowered: m.schedule, 5)
    assert events_s < _best(lowered.classical_report, 5) * 4


def test_a_cached_kernel_that_calls_lowers_in_microseconds():
    bench = _classical_benchmarks()
    dev = bench._device()
    q = dev.qudits[0]
    compiled = pulse.compile(bench.decoded_reset, quantum_system=dev)
    compiled.lower(q)
    per_call = _best(lambda: [compiled.lower(q) for _ in range(100)], 5) / 100
    assert compiled.cache_info().compiled == 1
    assert per_call < 100e-6, per_call


def test_a_call_adds_little_to_simulation_on_the_same_timeline():
    bench = _classical_benchmarks()
    dev = bench._device(t1_us=20.0)
    q = dev.qudits[0]
    plain = pulse.compile(bench.waiting_reset, quantum_system=dev).lower(q)
    called = pulse.compile(bench.decoded_reset, quantum_system=dev).lower(q)
    plus = [2**-0.5, 2**-0.5]
    a = _best(lambda: pulse.simulate(plain, initial_state=plus, levels=2, dt_ns=0.5), 3, 0)
    b = _best(lambda: pulse.simulate(called, initial_state=plus, levels=2, dt_ns=0.5), 3, 0)
    # Both take well under a millisecond, so allow the call's own binding and
    # rescheduling (about 0.3 ms) on top.
    assert b < 1.25 * a + 2e-3, (a, b)


@pytest.mark.scaling
def test_compiling_calls_takes_little_memory_per_call():
    import tracemalloc

    bench = _classical_benchmarks()
    q = bench._device().qudits[0]
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
    walks, exports = {}, {}
    for n in (500, 4000):
        lowered = _measured_line(n)
        module = lowered._concrete().simulation_module(*lowered.options.schedule_args())
        walks[n] = lambda m=module: m.next_measurement([])
        exports[n] = lowered.export_controller
    # 8x the ops: linear is 8x, quadratic 64x.
    _assert_linear("walk", walks, 16)
    _assert_linear("export", exports, 16)


def test_a_long_idle_simulates_in_constant_time():
    @pulse.kernel
    def idle(q, wait_cycles: int):
        d = get_drive_line(q)
        drive(d, square(40, 0.157))
        wait(d, wait_cycles)

    dev = pulse.quantum_system(ladder_target(1))
    compiled = pulse.compile(idle, quantum_system=dev)
    q = dev.qudits[0]
    short, long = compiled.lower(q, 2_000), compiled.lower(q, 400_000)
    times = {
        name: _best(lambda m=m: pulse.simulate(m), 3, 0)
        for name, m in (("1 us", short), ("200 us", long))
    }
    assert times["200 us"] < 3 * times["1 us"] + 0.01, times


def test_an_adaptive_loop_of_fifty_rounds_simulates_one_path_per_shot():
    from cudaq_pulse.classical import Link

    @pulse.classical(max_latency_ns=200, link="controller")
    def next_wait(bit: pulse.Bits, w: float) -> float:
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

    machine = ControlSystem(links={"controller": Link(fixed_ns=4, feedback_ns=8)})
    dev = pulse.quantum_system(ladder_target(1))
    lowered = pulse.compile(rounds, quantum_system=dev, control_system=machine).lower(
        dev.qudits[0], 50
    )
    seconds = _best(
        lambda: pulse.simulate(lowered, shots_count=1, seed=1, branches="sampled", dt_ns=0.125),
        2,
        0,
    )
    # Following all 2^50 paths is out of reach; one path takes about 0.1 s.
    assert seconds < 2.0, seconds
