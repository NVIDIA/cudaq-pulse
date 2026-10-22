# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Thin wrapper around the C++ nanobind extension."""

from __future__ import annotations

try:
    from ._cudaq_pulse_native import *  # noqa: F401,F403
except ImportError as _error:
    raise ImportError(
        "The cudaq-pulse native extension (_cudaq_pulse_native) could not be "
        "imported. Install the cudaq-pulse wheel, or build from source with "
        "`pip install .` (see CONTRIBUTING.md)."
    ) from _error
