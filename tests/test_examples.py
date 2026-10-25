# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Every example that needs no GPU runs to completion."""

import os
import runpy
from pathlib import Path

import cudaq_pulse
import pytest

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
_GPU_ONLY = {"krinner_evolve_gpu.py"}
_SCRIPTS = sorted(p for p in _EXAMPLES.rglob("*.py") if p.name not in _GPU_ONLY)
# A build tree stages the package in build/<dir>/python and cudaq-pulse-opt in
# build/<dir>/bin; an installation puts the tool on PATH itself.
_BUILD_BIN = Path(cudaq_pulse.__file__).parent.parent.parent / "bin"


@pytest.mark.parametrize("path", _SCRIPTS, ids=lambda p: str(p.relative_to(_EXAMPLES)))
def test_example_runs(path, monkeypatch, capsys):
    if (_BUILD_BIN / "cudaq-pulse-opt").exists():
        monkeypatch.setenv("PATH", f"{_BUILD_BIN}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    # Examples choose their devices themselves.
    monkeypatch.delenv("CUDAQ_PULSE_TARGET", raising=False)
    monkeypatch.syspath_prepend(str(path.parent))
    runpy.run_path(str(path), run_name="__main__")
    assert capsys.readouterr().out
