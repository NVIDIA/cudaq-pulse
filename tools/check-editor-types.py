# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Check that kernels type-check clean in editors with no configuration.

Usage: python tools/check-editor-types.py --python PYTHON [--pyright PYRIGHT]

PYTHON is an environment with cudaq-pulse installed from a wheel or with
``cmake --install BUILD --component CudaqPulse``, as a user has it. From an
empty project directory, pyright runs in "off" mode (the default of Pylance
and Cursor) and in "standard" mode over every file of examples/ and the
Python code blocks of each docs page, joined in order into one module, and
must report nothing. A probe with a typo and a misused operation must still
report exactly those. Exits 1 listing every unexpected diagnostic.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

_PROBE = """\
import cudaq_pulse as pulse


@pulse.kernel
def probe(qubit):
    line, tone = get_drive_line(qubit)
    drive(tone, gaussian(40, 0.1, 10.0), line)
    drve(line)
"""
# (line, rule) of what the probe must report, by mode.
_PROBE_EXPECTED = {
    "off": [(8, "reportUndefinedVariable")],
    "standard": [
        (7, "reportArgumentType"),
        (7, "reportArgumentType"),
        (8, "reportUndefinedVariable"),
    ],
}


def docs_modules(docs: Path) -> dict[str, str]:
    """Each docs page's Python code blocks, in order, as one module."""
    modules = {}
    for page in sorted(docs.rglob("*.rst")):
        lines = page.read_text().splitlines()
        blocks, i = [], 0
        while i < len(lines):
            if re.match(r"\s*\.\. code-block:: python\s*$", lines[i]):
                i += 1
                block = []
                while i < len(lines) and (not lines[i].strip() or lines[i][:1] in " \t"):
                    if not lines[i].lstrip().startswith(":"):
                        block.append(lines[i])
                    i += 1
                blocks.append(textwrap.dedent("\n".join(block)).strip())
            else:
                i += 1
        if blocks:
            name = page.relative_to(docs).with_suffix("").as_posix().replace("/", "__")
            modules[f"docs__{name}.py"] = "\n\n\n".join(blocks) + "\n"
    return modules


def run_pyright(pyright: list[str], python: str, project: Path, mode: str) -> list[dict]:
    (project / "pyrightconfig.json").write_text(json.dumps({"typeCheckingMode": mode}))
    result = subprocess.run(
        [*pyright, "--pythonpath", python, "--outputjson"],
        cwd=project,
        capture_output=True,
        text=True,
    )
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError:
        sys.exit(f"pyright did not run:\n{result.stdout}{result.stderr}")
    return [d for d in report["generalDiagnostics"] if d["severity"] in ("error", "warning")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--python", required=True, help="interpreter with cudaq-pulse installed")
    parser.add_argument("--pyright", default="pyright", help="pyright command")
    args = parser.parse_args()

    paths = subprocess.run(
        [args.python, "-c", "import json, sys; print(json.dumps(sys.path))"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if not any((Path(p) / "__builtins__.pyi").is_file() for p in json.loads(paths) if p):
        print("FAIL: no __builtins__.pyi on the sys.path of the installation")
        return 1

    problems = []
    with tempfile.TemporaryDirectory() as tmp:
        project = Path(tmp)
        examples = _ROOT / "examples"
        sources = {
            f"examples__{p.relative_to(examples).as_posix().replace('/', '__')}": p.read_text()
            for p in sorted(examples.rglob("*.py"))
        }
        sources.update(docs_modules(_ROOT / "docs"))
        sources["probe.py"] = _PROBE
        for name, text in sources.items():
            (project / name).write_text(text)
        for mode, expected in _PROBE_EXPECTED.items():
            diagnostics = run_pyright(args.pyright.split(), args.python, project, mode)
            probe = []
            for d in diagnostics:
                file = Path(d["file"]).name
                line = d["range"]["start"]["line"] + 1
                if file == "probe.py":
                    probe.append((line, d.get("rule", "")))
                else:
                    message = d["message"].splitlines()[0]
                    problems.append(f"{mode}: {file}:{line}: {d.get('rule', '')}: {message}")
            if sorted(probe) != expected:
                problems.append(f"{mode}: the probe reported {sorted(probe)}, expected {expected}")
            print(f"pyright {mode}: {len(sources) - 1} files, {len(diagnostics)} diagnostics")
    for problem in problems:
        print(f"FAIL {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
