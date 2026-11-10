# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schedule visualization (needs matplotlib, the ``[viz]`` extra)."""

from .timeline import plot_schedule, save_schedule

__all__ = ["plot_schedule", "save_schedule"]
