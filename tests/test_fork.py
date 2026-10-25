# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""A forked child compiles in a fresh context and refuses the parent's GPU."""

from __future__ import annotations

import os
import threading
import time
import warnings
from pathlib import Path

import cudaq_pulse as pulse
import pytest
from cudaq_pulse._native import _cudaq_pulse_native as native
from cudaq_pulse.runtime import gpu

from tests._handles import Q

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")


@pytest.fixture
def fresh_guard(monkeypatch):
    monkeypatch.setattr(gpu, "_runtime", None)
    monkeypatch.setattr(gpu, "_forked_after_cuda_init", False)
    monkeypatch.setattr(gpu, "_lock", threading.Lock())


def _parent_runtime():
    return gpu.GpuRuntime(library=None, path=Path("libcudm-runtime-cu12.so"), device_count=1)


def test_a_child_of_a_gpu_parent_refuses_the_gpu(monkeypatch, fresh_guard):
    monkeypatch.setattr(gpu, "_runtime", _parent_runtime())
    gpu._after_fork_in_child()
    assert gpu._runtime is None
    with pytest.raises(gpu.GpuDeviceUnavailableError, match="'spawn' or 'forkserver'"):
        gpu.require_gpu_runtime()


def test_a_child_forked_before_gpu_use_gets_a_usable_lock(monkeypatch, fresh_guard):
    held = threading.Lock()
    held.acquire()
    monkeypatch.setattr(gpu, "_lock", held)
    gpu._after_fork_in_child()
    assert not gpu._forked_after_cuda_init
    assert gpu._lock is not held and gpu._lock.acquire(blocking=False)
    gpu._lock.release()


@pulse.kernel
def _rabi(q0):
    d0, t0 = get_drive_line(q0)
    drive(d0, square(40, 0.1), t0)


def _child(parent_contexts: int) -> int:
    """Exit status of the forked child's checks: 0 when they all hold."""
    try:
        gpu.require_gpu_runtime()
    except gpu.GpuDeviceUnavailableError as error:
        if "forked" not in str(error):
            return 2
    else:
        return 3
    if native.shared_context_stats()[1] != 0:
        return 4
    pulse.compile(_rabi).lower(Q[0])
    if native.shared_context_stats()[0] != parent_contexts + 1:
        return 5
    return 0


def test_a_forked_child_starts_a_new_context_and_refuses_the_gpu(monkeypatch, fresh_guard):
    pulse.compile(_rabi).lower(Q[0])
    contexts = native.shared_context_stats()[0]
    monkeypatch.setattr(gpu, "_runtime", _parent_runtime())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        pid = os.fork()
    if pid == 0:
        status = 1
        try:
            status = _child(contexts)
        finally:
            os._exit(status)
    deadline = time.monotonic() + 60
    while (done := os.waitpid(pid, os.WNOHANG))[0] == 0:
        if time.monotonic() > deadline:
            os.kill(pid, 9)
            os.waitpid(pid, 0)
            pytest.fail("the forked child hung")
        time.sleep(0.05)
    assert os.waitstatus_to_exitcode(done[1]) == 0
