# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every example that needs no GPU runs to completion."""

import ast
import os
import runpy
from pathlib import Path

import cudaq_pulse
import pytest

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
_GPU_ONLY = {"transmon_surface17_evolve_gpu.py"}
_SCRIPTS = sorted(p for p in _EXAMPLES.rglob("*.py") if p.name not in _GPU_ONLY)
_SLOW = {
    "21_calibration_protocol.py",
    "24_pipelined_calibration.py",
    "25_precompiled_branches.py",
    "26_drift_following.py",
}
# Examples that call CUDA-Q QEC, which installs on Linux only.
_CUDAQ_QEC = {"27_realtime_decoding.py"}
# A build tree stages the package in build/<dir>/python and cudaq-pulse-opt in
# build/<dir>/bin; an installation puts the tool on PATH itself.
_BUILD_BIN = Path(cudaq_pulse.__file__).parent.parent.parent / "bin"


@pytest.mark.parametrize(
    "path",
    [pytest.param(p, marks=pytest.mark.slow) if p.name in _SLOW else p for p in _SCRIPTS],
    ids=lambda p: str(p.relative_to(_EXAMPLES)),
)
def test_example_runs(path, monkeypatch, capsys):
    if path.name in _CUDAQ_QEC:
        pytest.importorskip("cudaq_qec", reason="needs CUDA-Q QEC (Linux)")
    if (_BUILD_BIN / "cudaq-pulse-opt").exists():
        monkeypatch.setenv("PATH", f"{_BUILD_BIN}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    # Examples choose their quantum systems themselves.
    monkeypatch.delenv("CUDAQ_PULSE_QUANTUM_SYSTEM", raising=False)
    monkeypatch.syspath_prepend(str(path.parent))
    runpy.run_path(str(path), run_name="__main__")
    assert capsys.readouterr().out


_INDEXES = [_EXAMPLES / "README.md", _EXAMPLES.parent / "docs" / "examples.rst"]


@pytest.mark.parametrize("index", _INDEXES, ids=lambda p: p.name)
def test_the_example_indexes_list_every_example(index):
    if not index.exists():
        pytest.skip("needs the source tree")
    text = index.read_text()
    scripts = sorted(_EXAMPLES.rglob("*.py"))
    assert scripts
    assert [p.name for p in scripts if p.name not in text] == []


def test_the_readme_says_which_examples_simulate_and_need_a_gpu():
    readme = _EXAMPLES / "README.md"
    if not readme.exists():
        pytest.skip("needs the source tree")
    rows = {}
    for line in readme.read_text().splitlines():
        if line.startswith("| [`") and line.count("|") == 4:
            name = line.split("`")[1].rstrip("/")
            rows[name] = line.rsplit("|", 2)[1]
    for path in _EXAMPLES.rglob("*.py"):
        calls = {
            node.func.attr
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        row = rows[str(path.relative_to(_EXAMPLES))]
        gpu = path.name in _GPU_ONLY
        assert row.strip().startswith("GPU") == gpu, path.name
        assert ("simulates" in row) == (not gpu and "simulate" in calls), path.name
