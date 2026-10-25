# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Schedule visualization (needs matplotlib, the ``[viz]`` extra)."""

from .timeline import plot_schedule, save_schedule

__all__ = ["plot_schedule", "save_schedule"]
