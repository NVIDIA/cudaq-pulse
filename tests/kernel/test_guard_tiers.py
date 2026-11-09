# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A cache guard runs as closures over its steps, then, once it has held
UPGRADE_AFTER times, as generated code compiled once per source. Both forms
give one verdict (every guard the suite builds is also checked in both forms
by tests/conftest.py)."""

import importlib
import sys
import threading

_guard = importlib.import_module("cudaq_pulse.kernel._guard")

SCALE = 2.0
LIMIT = 3


class _Config:
    amp = 0.5


CFG = _Config()


def _reads():
    return SCALE * CFG.amp * not_bound_here  # noqa: F821


def _reads_limit():
    return LIMIT


def _reads_both():
    return LIMIT + SCALE


def _only_unbound():
    return drive_alike(also_unbound)  # noqa: F821


def _is_generated(check):
    return check.__code__.co_filename == "<pulse guard>"


def test_a_guard_runs_as_closures_until_it_has_held_upgrade_after_times():
    guard = _guard.build_guard(_reads)
    first = guard.check
    assert not _is_generated(first)
    for _ in range(_guard.UPGRADE_AFTER - 1):
        assert guard.check()
        assert guard.check is first
    assert guard.check()
    assert _is_generated(guard.check)
    assert guard.check()


def _forms(guard):
    return (
        guard.check(),
        _guard._closures(guard.plan, guard.registers)(),
        _guard._generated(guard.plan)(),
    )


def test_both_forms_see_each_change_before_and_after_the_upgrade(monkeypatch):
    module = sys.modules[__name__]
    for upgraded in (False, True):
        guard = _guard.build_guard(_reads)
        if upgraded:
            for _ in range(_guard.UPGRADE_AFTER):
                guard.check()
            assert _is_generated(guard.check)
        assert _forms(guard) == (True, True, True)
        with monkeypatch.context() as m:
            m.setattr(module, "SCALE", 3.0)
            assert _forms(guard) == (False, False, False)
        with monkeypatch.context() as m:
            # Equal but of another type: 2 is not 2.0.
            m.setattr(module, "SCALE", 2)
            assert _forms(guard) == (False, False, False)
        with monkeypatch.context() as m:
            m.setattr(_Config, "amp", 0.25)
            assert _forms(guard) == (False, False, False)
        with monkeypatch.context() as m:
            m.setattr(module, "not_bound_here", 1, raising=False)
            assert _forms(guard) == (False, False, False)
        assert _forms(guard) == (True, True, True)


def test_guards_of_one_function_share_their_compiled_code():
    a, b = _guard.build_guard(_reads), _guard.build_guard(_reads)
    assert _guard._generated(a.plan).__code__ is _guard._generated(b.plan).__code__


def test_the_compiled_code_cache_is_bounded(monkeypatch):
    _guard.build_guard(_reads)  # makes the cache
    monkeypatch.setattr(_guard, "_CODE", {})
    monkeypatch.setattr(_guard, "_CODE_LIMIT", 2)
    for fn in (_reads, _reads_limit, _reads_both):
        guard = _guard.build_guard(fn)
        assert _guard._generated(guard.plan)()
        assert len(_guard._CODE) <= 2


def test_a_guard_of_unbound_names_only_upgrades_too(monkeypatch):
    module = sys.modules[__name__]
    guard = _guard.build_guard(_only_unbound)
    for _ in range(_guard.UPGRADE_AFTER):
        assert guard.check()
    assert _is_generated(guard.check)
    monkeypatch.setattr(module, "also_unbound", 1, raising=False)
    assert _forms(guard) == (False, False, False)


def test_threads_that_upgrade_one_guard_at_once_all_hold():
    guard = _guard.build_guard(_reads)
    start = threading.Barrier(8)
    results = []

    def run():
        start.wait()
        results.extend(guard.check() for _ in range(_guard.UPGRADE_AFTER + 2))

    threads = [threading.Thread(target=run) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 8 * (_guard.UPGRADE_AFTER + 2) and all(results)
    assert _is_generated(guard.check)
