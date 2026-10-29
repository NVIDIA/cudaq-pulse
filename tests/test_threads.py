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

from tests._handles import Q
from tests.kernel.ir_corpus import active_reset, dd_loop, feedback_in_loop, parametric

_FREQ = {i: 5e9 + 1e8 * i for i in range(4)}


def _many_drives(n):
    @pulse.kernel
    def many(q0):
        d = get_drive_line(q0)
        for i in range(n):
            drive(d, square(4 + (i % 7), 0.01 * (i % 5)))

    return many


def _compile(kernel, schedule="alap"):
    return pulse.compile(kernel, schedule=schedule, unroll_threshold=0).lower(Q[0])


def test_concurrent_compiles_match_sequential_ones():
    jobs = [(k, s) for k in (dd_loop(8), active_reset, feedback_in_loop) for s in ("asap", "alap")]
    expected = [_compile(k, s).mlir() for k, s in jobs]
    with ThreadPoolExecutor(max_workers=8) as pool:
        got = list(pool.map(lambda job: _compile(*job).mlir(), jobs * 8))
    assert got == expected * 8


def test_concurrent_specializations_of_one_kernel():
    compiled = pulse.compile(parametric)
    q = Q[0]
    values = [(0.1 * i, i % 3 + 1) for i in range(32)]
    expected = [compiled.lower(q, *v).mlir() for v in values]
    with ThreadPoolExecutor(max_workers=8) as pool:
        got = list(pool.map(lambda v: compiled.lower(q, *v).mlir(), values))
    assert got == expected


def test_concurrent_first_calls_compile_once():
    kernel = _many_drives(50)
    with ThreadPoolExecutor(max_workers=8) as pool:
        got = list(pool.map(lambda _: kernel.lower(Q[0]), range(64)))
    assert kernel.cache_info().compiled == 1
    assert all(lowered is got[0] for lowered in got)


def test_one_module_used_from_several_threads():
    compiled = _compile(feedback_in_loop)
    module = compiled._module
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
    module = _compile(_many_drives(400))._module
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


def _together(n, work):
    """Run *work(i)* on *n* threads released at once; return the results."""
    barrier = threading.Barrier(n)

    def run(i):
        barrier.wait()
        return work(i)

    with ThreadPoolExecutor(n) as pool:
        return list(pool.map(run, range(n)))


def test_concurrent_first_loads_build_each_target_and_calibration_once(monkeypatch):
    from cudaq_pulse.targets import calibration, transmon

    monkeypatch.setattr(transmon, "_BUILT", {})
    monkeypatch.setattr(calibration, "_LOADED", {})
    loads = []
    load = calibration.Calibration.load
    monkeypatch.setattr(
        calibration.Calibration,
        "load",
        staticmethod(lambda *a, **k: loads.append(a) or load(*a, **k)),
    )
    targets = _together(16, lambda i: transmon.transmon_17q())
    assert len(loads) == 1
    assert all(t is targets[0] for t in targets)


def test_registering_targets_while_others_resolve(monkeypatch):
    from cudaq_pulse.targets import _registry

    monkeypatch.setattr(_registry, "_factories", dict(_registry._registry()))
    monkeypatch.setattr(_registry, "_resolved", {})

    def work(i):
        for round in range(200):
            if i % 2:
                _registry.register_target(f"stress_{i}_{round}", lambda arg: None)
            else:
                assert "transmon_generic" in pulse.targets.available_targets()
                assert _registry.make_target("transmon_generic:2").n_qubits == 2

    _together(8, work)
    names = pulse.targets.available_targets()
    assert all(f"stress_{i}_{r}" in names for i in (1, 3, 5, 7) for r in range(200))


def test_concurrent_first_traces_see_every_pulse_operation(monkeypatch):
    from cudaq_pulse.kernel import _emitter

    monkeypatch.setattr(_emitter, "_PULSE_FUNCTIONS", {})
    play = pulse.ops.drive

    def work(i):
        @pulse.kernel
        def k(q):
            line = get_drive_line(q)
            play(line, square(8 + i, 0.1))

        return k.lower(Q[0]).mlir().count("pulse.drive %")

    assert _together(16, work) == [1] * 16
