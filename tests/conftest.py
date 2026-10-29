# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os

import pytest
from cudaq_pulse.runtime.gpu import GpuRuntimeError, require_gpu_runtime
from cudaq_pulse.targets._registry import ENV_VAR, register_target

import tests.property_settings  # noqa: F401  (loads the hypothesis profile)
from tests._handles import ladder_target

# ``device()`` without an argument returns the process target.
register_target("ladder", lambda arg: ladder_target(int(arg) if arg else 64))
os.environ[ENV_VAR] = "ladder:8192"


def pytest_addoption(parser):
    parser.addoption(
        "--run-benchmarks",
        action="store_true",
        help="run the timing tests marked `benchmark` (skipped by default)",
    )
    parser.addoption(
        "--run-nightly",
        action="store_true",
        help="run the slow physics checks marked `nightly` (skipped by default)",
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
    if not config.getoption("--run-nightly"):
        skip = pytest.mark.skip(reason="slow physics check; pass --run-nightly")
        for item in items:
            if item.get_closest_marker("nightly"):
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
