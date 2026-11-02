# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""QPU target definitions: Hamiltonians, decoherence models, connectivity."""

from ._registry import UnknownTargetError, available_targets, register_target
from .base import (
    Coupling,
    DispersiveReadout,
    Drive,
    Gate,
    MicrowaveDrive,
    Qubit,
    Readout,
    ResidualZZ,
    TargetDescription,
)
from .calibration import Calibration
from .models import SimulationModel, TransmonModel
from .transmon import transmon_17q, transmon_generic

__all__ = [
    "TargetDescription",
    "Qubit",
    "Readout",
    "DispersiveReadout",
    "Drive",
    "MicrowaveDrive",
    "Coupling",
    "Gate",
    "ResidualZZ",
    "SimulationModel",
    "TransmonModel",
    "Calibration",
    "transmon_17q",
    "transmon_generic",
    "register_target",
    "available_targets",
    "UnknownTargetError",
]
