# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The README's quick example and the getting-started Hello World run.

Both run as a reader would run them, in a fresh namespace, and print what
their own comments say they print: the README's one-shot readout is 0 or 1
and the rest is the scheduled Pulse MLIR; the Hello World simulation prints
the P(1) and the shots count its comments state.
"""

from __future__ import annotations

import os
import re
import textwrap
from pathlib import Path

import pytest

from tests.test_tutorials import expected_output

_ROOT = Path(__file__).resolve().parents[1]
_README = _ROOT / "README.md"
_GETTING_STARTED = _ROOT / "docs" / "getting_started.rst"
_BUILD_BIN = _ROOT / "build" / "dev" / "bin"
_PYTHON_FENCE = re.compile(r"```python\n(.*?)\n```", re.S)
_CODE_BLOCK = re.compile(r"\s*\.\. code-block:: python\s*$")


@pytest.fixture(autouse=True)
def _cudaq_pulse_opt_on_path(monkeypatch):
    """A build tree stages cudaq-pulse-opt in build/dev/bin; put it on PATH."""
    if (_BUILD_BIN / "cudaq-pulse-opt").exists():
        monkeypatch.setenv("PATH", f"{_BUILD_BIN}{os.pathsep}{os.environ.get('PATH', '')}")
    # Both blocks choose their own quantum system.
    monkeypatch.delenv("CUDAQ_PULSE_QUANTUM_SYSTEM", raising=False)


def _first_python_fence(text: str) -> str:
    """The first block fenced with three backticks and ``python``."""
    match = _PYTHON_FENCE.search(text)
    assert match is not None, "no ```python block found"
    return match.group(1)


def _section_lines(text: str, title: str) -> list[str]:
    """The lines of the ``.rst`` section headed *title*, up to the next section."""
    lines = text.splitlines()
    start = next(
        i
        for i, line in enumerate(lines)
        if line.strip() == title and i + 1 < len(lines) and set(lines[i + 1].strip()) <= {"-", "="}
    )
    end = len(lines)
    for i in range(start + 2, len(lines) - 1):
        if lines[i].strip() and set(lines[i + 1].strip()) in ({"-"}, {"="}):
            end = i
            break
    return lines[start + 2 : end]


def _python_blocks(lines: list[str]) -> list[str]:
    """The ``.. code-block:: python`` bodies in *lines*, in order, dedented."""
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        if _CODE_BLOCK.match(lines[i]):
            i += 1
            body: list[str] = []
            while i < len(lines) and (not lines[i].strip() or lines[i][:1] in " \t"):
                body.append(lines[i])
                i += 1
            blocks.append(textwrap.dedent("\n".join(body)).strip())
        else:
            i += 1
    return blocks


@pytest.mark.skipif(not _README.is_file(), reason="needs the source tree's README.md")
def test_the_readme_quick_example_runs(capsys):
    code = _first_python_fence(_README.read_text())
    namespace: dict = {"__name__": "readme_quick_example"}
    exec(compile(code, "README.md, quick example", "exec"), namespace)
    lines = capsys.readouterr().out.splitlines()
    assert lines, "the quick example printed nothing"
    assert lines[0] in ("0", "1"), f"first line {lines[0]!r} is not a readout outcome"
    assert "pulse.drive" in "\n".join(lines[1:]), "the rest is not the scheduled Pulse MLIR"


@pytest.mark.skipif(not _GETTING_STARTED.is_file(), reason="needs the source tree's docs/")
def test_the_hello_world_blocks_run_and_print_their_comments(capsys):
    blocks = _python_blocks(_section_lines(_GETTING_STARTED.read_text(), "Hello World"))
    assert len(blocks) == 2, f"Hello World has {len(blocks)} python blocks, expected 2"
    namespace: dict = {"__name__": "getting_started_hello_world"}
    for code in blocks:
        exec(compile(code, "getting_started.rst, Hello World", "exec"), namespace)
    out = capsys.readouterr().out
    # The numbers come from the simulate block's own comments, so the test
    # follows the page when a number is updated on purpose.
    expected = expected_output(blocks[1])
    assert len(expected) == 2, f"the simulate block states {len(expected)} outputs, expected 2"
    for line in expected:
        assert line in out, f"the Hello World output has no {line!r}"
