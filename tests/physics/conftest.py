# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    here = Path(__file__).parent
    for item in items:
        if Path(str(item.fspath)).parent == here:
            item.add_marker(pytest.mark.physics)
