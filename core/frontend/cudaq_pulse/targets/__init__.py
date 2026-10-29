# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""QPU target definitions: Hamiltonians, decoherence models, connectivity."""

from ._registry import UnknownTargetError, available_targets, register_target
from .base import Coupling, CrosstalkEntry, Qubit, TargetDescription
from .calibration import Calibration
from .models import SimulationModel, TransmonModel
from .transmon import transmon_generic, transmon_krinner_17q

__all__ = [
    "TargetDescription",
    "Qubit",
    "Coupling",
    "CrosstalkEntry",
    "SimulationModel",
    "TransmonModel",
    "Calibration",
    "transmon_krinner_17q",
    "transmon_generic",
    "register_target",
    "available_targets",
    "UnknownTargetError",
]
