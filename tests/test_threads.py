# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Native calls release the GIL and are safe to make from several threads."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cudaq_pulse as pulse

from tests.kernel.ir_corpus import active_reset, dd_loop, feedback_in_loop, parametric

_FREQ = {i: 5e9 + 1e8 * i for i in range(4)}


def _many_drives(n):
    @pulse.kernel
    def many(q0):
        d, t = get_drive_line(q0)
        for i in range(n):
            drive(d, square(4 + (i % 7), 0.01 * (i % 5)), t)

    return many


def _compile(kernel, schedule="alap"):
    return pulse.compile(kernel, [pulse.qudit_ref()], qubit_freq_hz=_FREQ, schedule=schedule)


def test_concurrent_compiles_match_sequential_ones():
    jobs = [(k, s) for k in (dd_loop(8), active_reset, feedback_in_loop) for s in ("asap", "alap")]
    expected = [_compile(k, s).mlir for k, s in jobs]
    with ThreadPoolExecutor(max_workers=8) as pool:
        got = list(pool.map(lambda job: _compile(*job).mlir, jobs * 8))
    assert got == expected * 8


def test_concurrent_specializations_of_one_kernel():
    compiled = pulse.compile(parametric, [pulse.qudit_ref()], qubit_freq_hz=_FREQ)
    values = [(0.1 * i, i % 3 + 1) for i in range(32)]
    expected = [compiled(*v).mlir for v in values]
    with ThreadPoolExecutor(max_workers=8) as pool:
        got = list(pool.map(lambda v: compiled(*v).mlir, values))
    assert got == expected


def test_one_module_used_from_several_threads():
    compiled = _compile(feedback_in_loop)
    module = compiled.module
    events = module.events()
    text = module.print()

    def use(i):
        if i % 3 == 0:
            module.run_passes(["pulse-verify"])
        return module.events(), module.print()

    with ThreadPoolExecutor(max_workers=8) as pool:
        for got_events, got_text in pool.map(use, range(48)):
            assert got_events == events
            assert got_text == text


def test_lowering_runs_while_python_keeps_running():
    module = _compile(_many_drives(400)).module
    done = threading.Event()
    elapsed = []

    def lower():
        start = time.perf_counter()
        module.run_full_lowering()
        elapsed.append(time.perf_counter() - start)
        done.set()

    worker = threading.Thread(target=lower)
    # With the GIL held for the whole lowering, this thread would stall for
    # all of it (already in start(), which waits for the worker to run);
    # released, it keeps ticking.
    gaps, last = [], time.perf_counter()
    worker.start()
    while not done.is_set():
        time.sleep(0.001)
        now = time.perf_counter()
        gaps.append(now - last)
        last = now
    worker.join()
    assert max(gaps) < 0.5 * elapsed[0]
