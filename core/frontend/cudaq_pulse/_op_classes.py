# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The classes of classical operations and the terms of run-time expressions,
as ``pulse.controller`` gives them. Loaded on first use: the library checks
names against their values without them."""

from __future__ import annotations

import enum


class OpClass(enum.StrEnum):
    """A class of operation: those a classical function's body is counted in
    (``OpCounts``), and ``LOOP``, a rolled loop's counter and jump, which a
    processor costs per iteration (``OpCycles``) and may not run
    (``Processor.unsupported``). ``BRANCH`` is an ``if`` or ``else``."""

    ADD = "add"
    MUL = "mul"
    DIV = "div"
    COMPARE = "compare"
    BRANCH = "branch"
    LOAD = "load"
    CONVERT = "convert"
    POPCOUNT = "popcount"
    SQRT = "sqrt"
    EXP = "exp"
    LOG = "log"
    TRIG = "trig"
    LOOP = "loop"


class ExpressionTerm(enum.StrEnum):
    """An ``op`` an exported expression of a run-time value may have; see
    ``evaluate_expression`` and the controller backends guide."""

    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    NEG = "neg"
    MAX = "max"
    MIN = "min"
    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"
    FLOOR_DIV = "floor_div"
    REM = "rem"
    TO_FLOAT = "to_float"
    TO_INT = "to_int"
    CONTROLLER_CYCLES = "controller_cycles"
    SELECT = "select"
