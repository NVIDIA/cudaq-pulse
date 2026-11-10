# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The loop-unroll threshold of a compilation.

A ``for ... in range(n)`` loop whose variable the body never reads is
unrolled while tracing when n times the calls in its body (at least one) is
at most the threshold, and emitted once as a rolled ``scf.for`` when it is
larger. Either way the program, its schedule and its simulation are the
same; the threshold trades tracing, lowering and scheduling time against
the size of the module.
"""

from __future__ import annotations

from .._errors import InvalidValueError

DEFAULT_UNROLL_THRESHOLD = 8


def resolve_unroll_threshold(value: int | None = None) -> int:
    """The unroll threshold: *value* if given, else
    ``DEFAULT_UNROLL_THRESHOLD``. 0 never unrolls."""
    if value is None:
        return DEFAULT_UNROLL_THRESHOLD
    if type(value) is not int or value < 0:
        raise InvalidValueError(
            f"unroll_threshold must be a non-negative int (0 means never unroll), got {value!r}"
        )
    return value
