# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Physics verification against analytic, first-principles and published values.

Every expected value lives in ``reference_values.json`` with its formula or
citation; a test reads its entry by id with ``ref(id)``.
"""

import json
from pathlib import Path

import pytest

_VALUES = json.loads((Path(__file__).parent / "reference_values.json").read_text())
_BY_ID = {entry["id"]: entry for entry in _VALUES["entries"]}


def ref(entry_id: str) -> dict:
    """The reference entry *entry_id*."""
    return _BY_ID[entry_id]


def pytest_collection_modifyitems(items):
    here = Path(__file__).parent
    for item in items:
        if Path(str(item.fspath)).parent == here:
            item.add_marker(pytest.mark.physics)
