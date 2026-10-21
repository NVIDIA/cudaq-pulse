# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Compile-time regression guard across the benchmark tiers.

Skipped unless ``--run-benchmarks`` is given. Each case is timed best-of-N
and divided by a pure-Python calibration loop timed in the same process, so
the checked-in ratios transfer between machines and CI runners. A case fails
only if it is several times slower, relative to the calibration, than the
baseline in ``benchmarks/baselines/frontend_perf.json``. Refresh that file
with ``--update-perf-baseline`` after an intentional change.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.kernel.bytecode_bridge import _trace_kernel_with_builder
from cudaq_pulse.kernel.ir_builder import Parameter
from cudaq_pulse.kernel.packed_ir_builder import PackedIRBuilder

from tests.kernel.ir_corpus import bicycle_syndrome, dd_loop, parametric, surface_syndrome

pytestmark = pytest.mark.benchmark

_ROOT = Path(__file__).resolve().parents[1]
BASELINE = _ROOT / "benchmarks" / "baselines" / "frontend_perf.json"
TOLERANCE = 3.0
# A retained loop's compile time must not grow with its trip count.
FLAT_LOOP_LIMIT = 2.0


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
    return lambda: kernel(*[pulse.qudit_ref() for _ in range(n_qubits)])


def _compile(kernel, n_qubits: int, freq: dict[int, float]) -> Callable[[], object]:
    return lambda: pulse.compile(
        kernel, [pulse.qudit_ref() for _ in range(n_qubits)], qubit_freq_hz=freq
    )


def _fused(kernel, n: int) -> Callable[[], object]:
    freq = {i: 5e9 + 1e5 * i for i in range(n)}
    return lambda: pulse.compile(kernel, [pulse.qvec_ref(n)], qubit_freq_hz=freq, passes=())


def _packed_trace(kernel, n: int) -> Callable[[], object]:
    freq = {i: 5e9 for i in range(n)}
    fn = kernel.__wrapped__

    def run():
        builder = PackedIRBuilder(name="k", clock_ghz=2.0, qubit_freq_hz=freq)
        _trace_kernel_with_builder(fn, builder, [pulse.qvec_ref(n)])

    return run


def _cases() -> dict[str, Callable[[], object]]:
    workloads = _compile_time_workloads()
    cases = {}
    for name in ("bell", "qaoa4", "dd_cpmg8", "vqe_hea"):
        kernel, n_qubits, freq = workloads[name]
        cases[f"tier1_{name}_trace"] = _kernel_ir(kernel, n_qubits)
        cases[f"tier1_{name}_compile"] = _compile(kernel, n_qubits, freq)
    for n in (8, 512):
        cases[f"dd_loop_n{n}"] = _kernel_ir(dd_loop(n), 1)
        cases[f"dd_loop_n{n}_compile"] = _compile(dd_loop(n), 1, {0: 5e9})
    surface5, n5 = surface_syndrome(5)
    surface9, n9 = surface_syndrome(9)
    bicycle, nb = bicycle_syndrome(6, 6)
    cases["surface_d5_compile"] = _fused(surface5, n5)
    cases["surface_d9_trace"] = _packed_trace(surface9, n9)
    cases["bicycle_6x6_compile"] = _fused(bicycle, nb)
    cases["parametric_compile"] = lambda: pulse.compile(
        parametric, [pulse.qudit_ref()], qubit_freq_hz={0: 5e9}
    )
    cases["parametric_trace"] = lambda: _trace_kernel_with_builder(
        parametric.__wrapped__,
        PackedIRBuilder(name="p", clock_ghz=2.0, qubit_freq_hz={0: 5e9}),
        [pulse.qudit_ref(), Parameter("amp", 0, "f64"), Parameter("n", 1, "i64")],
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
    kernel, n_qubits, freq = _compile_time_workloads()["bell"]
    best = {"trace_ms": float("inf"), "ffi_ms": float("inf")}
    for _ in range(50):
        metrics = _compile(kernel, n_qubits, freq)().metrics
        for stage in best:
            best[stage] = min(best[stage], getattr(metrics, stage))
    assert best["ffi_ms"] < best["trace_ms"], best


def test_retained_loop_compile_time_is_flat(ratios):
    assert ratios["dd_loop_n512"] < ratios["dd_loop_n8"] * FLAT_LOOP_LIMIT
    assert ratios["dd_loop_n512_compile"] < ratios["dd_loop_n8_compile"] * FLAT_LOOP_LIMIT
