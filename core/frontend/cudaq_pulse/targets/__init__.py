# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QPU target definitions: Hamiltonians, decoherence models, connectivity."""

from ._registry import UnknownTargetError, available_targets, register_target
from .base import (
    Coupling,
    DispersiveReadout,
    DragPulse,
    Drive,
    FluxCZ,
    MicrowaveDrive,
    Qubit,
    Readout,
    ReadoutPulse,
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
    "ResidualZZ",
    "DragPulse",
    "ReadoutPulse",
    "FluxCZ",
    "SimulationModel",
    "TransmonModel",
    "Calibration",
    "transmon_17q",
    "transmon_generic",
    "register_target",
    "available_targets",
    "UnknownTargetError",
]
