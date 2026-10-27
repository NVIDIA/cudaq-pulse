# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The base class of the errors cudaq-pulse raises."""


class PulseError(Exception):
    """Base class of every error cudaq-pulse raises on purpose.

    Each subclass also keeps its built-in base, so ``except RuntimeError``
    or ``except LookupError`` still catches what it caught before. Errors
    that the native extension raises while running MLIR passes, and
    ``ValueError`` or ``TypeError`` for a malformed argument, are not
    subclasses.
    """
