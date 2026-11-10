# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-version parity: every corpus kernel traces to the checked-in IR.

CI runs this on each supported Python version against the same golden file,
so any interpreter whose bytecode the frontend mis-decodes fails here.
Regenerate with ``CUDAQ_PULSE_UPDATE_GOLDENS=1 pytest tests/kernel/test_ir_golden.py``
and review the diff; a change must be the same on every version. A run that
changes the file fails and names the cases it changed, so the next run,
without the variable, is the one that passes. CI never rewrites the file.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests._tools import in_ci
from tests.kernel.ir_corpus import CASES, trace

GOLDEN = Path(__file__).parent / "golden" / "frontend_ir.json"
UPDATE_VAR = "CUDAQ_PULSE_UPDATE_GOLDENS"
_UPDATE = os.environ.get(UPDATE_VAR, "")
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


def _rewrite() -> list[str]:
    """Write every case's record; return the cases whose record changed."""
    before = _load()
    text = json.dumps({c.name: _golden_record(trace(c)) for c in CASES}, indent=1) + "\n"
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(text)
    after = json.loads(text)
    return sorted(
        name for name in before.keys() | after.keys() if before.get(name) != after.get(name)
    )


_REWRITTEN = _rewrite() if _UPDATE == "1" and not in_ci() else []
_GOLDENS = _load()


def test_the_goldens_were_not_rewritten():
    if _UPDATE not in ("", "0", "1"):
        pytest.fail(f"{UPDATE_VAR}={_UPDATE!r}; set it to 1 to rewrite {GOLDEN.name}")
    if _UPDATE == "1" and in_ci():
        pytest.fail(
            f"{UPDATE_VAR}=1 in CI: rewrite {GOLDEN.name} locally and commit the reviewed diff"
        )
    assert not _REWRITTEN, (
        f"rewrote {len(_REWRITTEN)} records of {GOLDEN.name} ({', '.join(_REWRITTEN)}); "
        f"review the diff, then run again without {UPDATE_VAR}"
    )


def test_ci_refuses_to_rewrite_the_goldens():
    before = GOLDEN.read_bytes()
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            f"{__file__}::test_the_goldens_were_not_rewritten",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=dict(os.environ, CI="true", **{UPDATE_VAR: "1"}),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert run.returncode == 1, run.stdout + run.stderr
    assert f"{UPDATE_VAR}=1 in CI" in run.stdout
    assert GOLDEN.read_bytes() == before


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
