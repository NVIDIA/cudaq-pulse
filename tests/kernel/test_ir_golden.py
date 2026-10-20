# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Cross-version parity: every corpus kernel traces to the checked-in IR.

CI runs this on each supported Python version against the same golden file,
so any interpreter whose bytecode the frontend mis-decodes fails here.
Regenerate with ``CUDAQ_PULSE_UPDATE_GOLDENS=1 pytest tests/kernel/test_ir_golden.py``
and review the diff; a change must be the same on every version.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from tests.kernel.ir_corpus import CASES, trace

GOLDEN = Path(__file__).parent / "golden" / "frontend_ir.json"
# Dumps longer than this are stored as a digest only.
_MAX_INLINE_LINES = 80


def _golden_record(record: dict) -> dict:
    if "error" in record:
        return record
    text = "\n".join(record["dump"])
    out = {"ops": record["ops"], "sha256": hashlib.sha256(text.encode()).hexdigest()}
    if len(record["dump"]) <= _MAX_INLINE_LINES:
        out["dump"] = record["dump"]
    return out


def _load() -> dict:
    return json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}


if os.environ.get("CUDAQ_PULSE_UPDATE_GOLDENS") == "1":
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(
        json.dumps({c.name: _golden_record(trace(c)) for c in CASES}, indent=1) + "\n"
    )

_GOLDENS = _load()


def test_golden_covers_corpus():
    assert sorted(_GOLDENS) == sorted(c.name for c in CASES)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_ir_matches_golden(case):
    expected = _GOLDENS[case.name]
    actual = trace(case)
    if "error" in expected:
        assert actual.get("error") == expected["error"]
        if "span" in expected:
            assert actual["span"] == expected["span"]
            assert actual["cols"] == expected["cols"]
        return
    assert "error" not in actual, actual.get("error")
    got = _golden_record(actual)
    if "dump" in expected:
        assert got.get("dump") == expected["dump"]
    assert got["sha256"] == expected["sha256"]
    assert got["ops"] == expected["ops"]
