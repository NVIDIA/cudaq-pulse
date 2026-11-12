# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lowering a retained scf.for gives the same Hamiltonian as unrolling it."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from cudaq_pulse._native._cudaq_pulse_native import MLIRPipeline, evaluate_callback
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.property_settings import examples
from tests.test_region_scheduling import _DURATIONS, _programs

_TRAIN = 0x100


def _hamiltonian(text, line_count, policy):
    module = MLIRPipeline().parse_to_module(text)
    module.schedule(policy)
    freqs = ", ".join(f"{5.0e9 + 1.0e8 * i:e}" for i in range(line_count))
    module.set_attributes(
        "{pulse.control_system = #pulse.control_system<clock_period_fs = 500000>, "
        f"pulse.quantum_system = #pulse.quantum_system<qudit_frequency_hz = [{freqs}]>}}"
    )
    return module.operator_program()


def _coefficients(program, times):
    """Each operator's total coefficient at `times`, keyed by its factors."""
    _, terms, _, *_ = program
    total = defaultdict(lambda: np.zeros(len(times)))
    trains = 0
    for re, _im, kind, params, factors in terms:
        key = tuple((mode, dim, tuple(data)) for mode, dim, data in factors)
        if kind:
            trains += bool(kind & _TRAIN)
            total[key] += np.asarray(evaluate_callback(kind, params, times))
        else:
            total[key] += re
    return total, trains


def _ops(line_count):
    line = st.integers(0, line_count - 1)
    return st.one_of(
        st.tuples(st.just("drive"), line, st.sampled_from(_DURATIONS)),
        st.tuples(st.just("wait"), line, st.sampled_from(_DURATIONS)),
        st.tuples(st.just("phase"), line, st.sampled_from([0.25, -0.5, 1.0])),
    )


@st.composite
def _case(draw):
    n = draw(st.integers(1, 2))
    ops = _ops(n)
    nested = st.tuples(st.just("loop"), st.integers(0, 3), st.lists(ops, min_size=1, max_size=3))
    return (
        n,
        draw(st.lists(ops, max_size=3)),
        draw(st.integers(0, 5)),
        draw(st.lists(st.one_of(ops, nested), min_size=1, max_size=4)),
        draw(st.lists(ops, max_size=3)),
    )


@given(case=_case(), policy=st.sampled_from(["asap", "alap"]))
@settings(max_examples=examples(80))
def test_retained_loop_lowers_like_its_unrolling(case, policy):
    rolled, unrolled = _programs(*case)
    a = _hamiltonian(rolled, case[0], policy)
    b = _hamiltonian(unrolled, case[0], policy)
    assert a[3:] == b[3:]
    times = np.linspace(0.0, max(a[4], 1.0), 2001)
    ca, _ = _coefficients(a, times)
    cb, _ = _coefficients(b, times)
    assert ca.keys() == cb.keys()
    for key in ca:
        np.testing.assert_allclose(ca[key], cb[key], atol=1e-12)


def test_a_dd_loop_lowers_to_one_train_per_quadrature():
    rolled, _ = _programs(1, [], 512, [("drive", 0, 40), ("phase", 0, 0.5)], [])
    program = _hamiltonian(rolled, 1, "asap")
    kinds = [kind for _, _, kind, _, _ in program[1] if kind]
    assert len(kinds) == 2 and all(kind & _TRAIN for kind in kinds)


def test_abutting_square_pulses_in_a_loop_lower_to_one_square_pulse():
    rolled, _ = _programs(1, [], 512, [("drive", 0, 40)], [])
    program = _hamiltonian(rolled, 1, "asap")
    drives = [(kind, params) for _, _, kind, params, _ in program[1] if kind]
    assert [kind for kind, _ in drives] == [1, 1]
    assert all(params[1] == 512 * 20.0 for _, params in drives)
