# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Processors: the classical compute of a controller, and what a program
uses of it (``Lowered.controller_usage``)."""

from __future__ import annotations

from cudaq_pulse._controller_cost import analyze_callable


def _parity(bits):
    acc = 0
    for i in range(4):
        acc = acc ^ bits[i]
    return [acc]


def _twice(x):
    return x + x


def _scaled(x):
    # A divide: by a constant it would be a multiply.
    return _twice(x) / x


def _sign(x):
    if x > 0.0:
        return 1.0
    return -1.0


# ── What a function costs on a processor ─────────────────────────────────────


def test_a_body_records_its_loops_unrolled_size_and_where_it_uses_each_operation():
    cost = analyze_callable(_parity, {"bits": 4})
    # Rolled: the xor and the load, then the counter and the jump.
    assert cost.size == 4 and cost.iterations == 4 and cost.unrolled_size == 8
    assert cost.ops == {"add": 8, "branch": 4, "load": 4}
    line = _parity.__code__.co_firstlineno
    assert cost.uses == {"add": f"_parity line {line + 3}", "load": f"_parity line {line + 3}"} | {
        "loop": f"_parity line {line + 2}"
    }
    # An inlined call is not a branch, and a helper's uses are its own lines.
    cost = analyze_callable(_scaled, {})
    assert set(cost.uses) == {"add", "div"}
    assert cost.uses["add"].startswith("_twice line")
    assert analyze_callable(_sign, {}).uses["branch"].startswith("_sign line")


def test_a_processor_prices_a_loop_by_its_loop_op_or_runs_it_unrolled():
    cost = analyze_callable(_parity, {"bits": 4})
    # Rolled: 4 xors, 4 loads and 4 iterations.
    assert cost.processor_cycles({}, loops=True) == 12
    assert cost.processor_cycles({"loop": 3}, loops=True) == 20
    assert cost.processor_cycles({"loop": 3}, loops=False) == 8
    assert cost.processor_cycles({"add": 2}, loops=True) == 16
