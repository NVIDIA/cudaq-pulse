# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.lower import _to_program
from cudaq_pulse.runtime.gpu import GpuRuntimeError, require_gpu_runtime


def pytest_addoption(parser):
    parser.addoption(
        "--run-benchmarks",
        action="store_true",
        help="run the timing tests marked `benchmark` (skipped by default)",
    )
    parser.addoption(
        "--update-perf-baseline",
        action="store_true",
        help="rewrite benchmarks/baselines/frontend_perf.json from this run",
    )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    """Selecting any GPU test requires a working GPU runtime.

    Runs after ``-m`` deselection, so ``pytest -m "not gpu"`` never touches
    the GPU, while a run that selects GPU tests fails outright, with the
    specific reason, instead of skipping them.
    """
    if not config.getoption("--run-benchmarks"):
        skip = pytest.mark.skip(reason="timing test; pass --run-benchmarks")
        for item in items:
            if item.get_closest_marker("benchmark"):
                item.add_marker(skip)
    if not any(item.get_closest_marker("gpu") for item in items):
        return
    try:
        require_gpu_runtime()
    except GpuRuntimeError as error:
        raise pytest.UsageError(
            f"GPU tests were selected but cannot run: {error}\n"
            'Deselect them with `-m "not gpu"` to run the compiler-only '
            "suite."
        ) from error


@pulse.kernel
def _simple_kernel(q0):
    d0, t0 = get_drive_line(q0)
    wf = gaussian(40, 0.3, 10.0)
    drive(d0, wf, t0)


@pulse.kernel
def _two_qubit_kernel(q0, q1):
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    wf = gaussian(40, 0.3, 10.0)
    drive(d0, wf, t0)
    sync(d0, d1)
    drive(d1, wf, t1)


@pulse.kernel
def _echo_kernel(q0):
    d0, t0 = get_drive_line(q0)
    for i in range(5):
        wf_pos = gaussian(40, 0.3, 10.0)
        drive(d0, wf_pos, t0)
        wait(d0, 20)
        wf_neg = gaussian(40, -0.3, 10.0)
        drive(d0, wf_neg, t0)


@pytest.fixture
def simple_program():
    """A minimal single-qubit drive program for testing."""
    ir = _simple_kernel(pulse.qudit_ref())
    return _to_program(ir, clock_ghz=2.0, qubit_freq_hz={0: 5.0e9})


@pytest.fixture
def two_qubit_program():
    """A two-qubit program with sync for testing."""
    ir = _two_qubit_kernel(pulse.qudit_ref(), pulse.qudit_ref())
    return _to_program(ir, clock_ghz=2.0, qubit_freq_hz={0: 5.0e9, 1: 5.1e9})


@pytest.fixture
def echo_program():
    """The canonical echo program from the paper."""
    ir = _echo_kernel(pulse.qudit_ref())
    return _to_program(ir, clock_ghz=2.0, qubit_freq_hz={0: 5.0e9})
