# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Hypothesis profiles for the property tests, chosen by HYPOTHESIS_PROFILE.

- ``ci`` (default): derandomized, so every run checks the same examples and a
  pull request never fails on an input nobody has seen before.
- ``dev``: random examples, remembered in the local ``.hypothesis`` database.
- ``nightly``: random, ten times as many examples; the scheduled workflow
  passes ``--hypothesis-seed`` and keeps the example database between runs.

Every profile prints a ``@reproduce_failure`` blob for a failing example. To
replay a nightly failure, run the failing test with the same profile and the
seed from the job summary, or paste the blob above the test.
"""

from __future__ import annotations

import os

from hypothesis import HealthCheck, settings

_EXAMPLE_SCALE = {"ci": 1, "dev": 1, "nightly": 10}

settings.register_profile("ci", derandomize=True, deadline=None, print_blob=True)
settings.register_profile("dev", deadline=None, print_blob=True)
settings.register_profile(
    "nightly",
    deadline=None,
    print_blob=True,
    suppress_health_check=[HealthCheck.too_slow],
)

PROFILE = os.environ.get("HYPOTHESIS_PROFILE", "ci")
if PROFILE not in _EXAMPLE_SCALE:
    raise ValueError(f"HYPOTHESIS_PROFILE={PROFILE!r}; choose one of {', '.join(_EXAMPLE_SCALE)}")
settings.load_profile(PROFILE)


def examples(count: int) -> int:
    """*count* examples, scaled up for the nightly profile."""
    return count * _EXAMPLE_SCALE[PROFILE]
