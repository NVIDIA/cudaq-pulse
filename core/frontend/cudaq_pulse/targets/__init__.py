# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""QPU target definitions: Hamiltonians, decoherence models, connectivity."""

from ._registry import available_targets, register_target
from .base import Coupling, CrosstalkEntry, Qubit, Target
from .rydberg import RydbergAtom, RydbergTarget, rydberg_chain, rydberg_square
from .transmon import transmon_generic, transmon_krinner_17q

__all__ = [
    "Target",
    "Qubit",
    "Coupling",
    "CrosstalkEntry",
    "transmon_krinner_17q",
    "transmon_generic",
    "RydbergAtom",
    "RydbergTarget",
    "rydberg_chain",
    "rydberg_square",
    "register_target",
    "available_targets",
]
