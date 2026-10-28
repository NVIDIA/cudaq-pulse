# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""What pyright and mypy, as editors run them, report on classical calls.

``pulse.classical`` is a module at run time, which checkers can't call; the
package declares it as a callable decorator for them. A planted wrong call
must be the only error, and the classical-call examples must check clean
apart from the kernel language's bare names, which checkers don't know.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cudaq_pulse
import pytest
from cudaq_pulse import ops

_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES = [
    _ROOT / "examples" / "19_classical_calls.py",
    _ROOT / "examples" / "20_adaptive_calibration.py",
]
_PACKAGE_ROOT = Path(cudaq_pulse.__file__).resolve().parents[1]
_BARE_NAMES = frozenset(
    name for name, value in vars(ops).items() if callable(value) and not name.startswith("_")
)

_PLANTED = """\
import cudaq_pulse as pulse
from cudaq_pulse.classical import Bits, Float


@pulse.classical(args=(Bits(1),), results=(Bits(1),), max_latency_ns=400, link="gpu")
def decode(syndrome):
    return [syndrome[0]]


@pulse.classical(args=(Float(),), results=(Float(),), max_latency_ns="soon")  # planted
def scale(x):
    return [2 * x]


in_time, correction = decode.submit([1]).result(timeout=350 * pulse.ns)
table = pulse.classical.external("table", args=(Bits(2),), max_latency_ns=50)
"""


def _planted_line() -> int:
    return next(i for i, line in enumerate(_PLANTED.splitlines(), 1) if "# planted" in line)


@pytest.fixture(scope="module")
def files(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    planted = tmp_path_factory.mktemp("planted") / "planted.py"
    planted.write_text(_PLANTED)
    # A wheel's tests run without the source tree's examples/.
    return [planted, *(example for example in _EXAMPLES if example.exists())]


def _env() -> dict[str, str]:
    env = dict(os.environ)
    if _PACKAGE_ROOT.name != "site-packages":
        env["MYPYPATH"] = str(_PACKAGE_ROOT)
    return env


@pytest.fixture(scope="module")
def pyright(
    files: list[Path], tmp_path_factory: pytest.TempPathFactory
) -> list[tuple[Path, int, str, str]]:
    exe = shutil.which("pyright")
    if exe is None:
        pytest.skip("pyright is not installed")
    work = tmp_path_factory.mktemp("pyright")
    config = work / "pyrightconfig.json"
    config.write_text(
        json.dumps(
            {
                "typeCheckingMode": "standard",
                "pythonVersion": f"{sys.version_info.major}.{sys.version_info.minor}",
                "extraPaths": [str(_PACKAGE_ROOT)],
            }
        )
    )
    run = subprocess.run(
        [exe, "-p", str(config), "--pythonpath", sys.executable, "--outputjson", *map(str, files)],
        cwd=work,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=600,
    )
    report = json.loads(run.stdout)
    return [
        (
            Path(d["file"]).resolve(),
            d["range"]["start"]["line"] + 1,
            d.get("rule", ""),
            d["message"],
        )
        for d in report["generalDiagnostics"]
        if d["severity"] == "error"
    ]


@pytest.fixture(scope="module")
def mypy(
    files: list[Path], tmp_path_factory: pytest.TempPathFactory
) -> list[tuple[Path, int, str, str]]:
    if importlib.util.find_spec("mypy") is not None:
        command = [sys.executable, "-m", "mypy"]
    elif (exe := shutil.which("mypy")) is not None:
        command = [exe, "--python-executable", sys.executable]
    else:
        pytest.skip("mypy is not installed")
    work = tmp_path_factory.mktemp("mypy")
    config = work / "mypy.ini"
    config.write_text("[mypy]\ncheck_untyped_defs = True\n")
    run = subprocess.run(
        [
            *command,
            "--config-file",
            str(config),
            "--cache-dir",
            str(work / "cache"),
            "--show-error-codes",
            "--no-error-summary",
            "--no-pretty",
            *map(str, files),
        ],
        cwd=work,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=600,
    )
    errors = []
    for line in run.stdout.splitlines():
        match = re.match(r"(.+?):(\d+): error: (.*?)  \[([a-z-]+)\]$", line)
        if match:
            path, number, message, code = match.groups()
            errors.append((Path(path).resolve(), int(number), code, message))
    assert errors or run.returncode == 0, run.stdout + run.stderr
    return errors


def _bare_name(rule: str, message: str) -> bool:
    match = re.match(r'(?:Name )?"(\w+)" is not defined', message)
    return (
        rule in ("reportUndefinedVariable", "name-defined")
        and bool(match)
        and match[1] in _BARE_NAMES
    )


@pytest.mark.parametrize("checker", ["pyright", "mypy"])
def test_planted_wrong_call_is_the_only_error(checker, files, request):
    errors = request.getfixturevalue(checker)
    planted = [e for e in errors if e[0] == files[0].resolve()]
    assert [line for _, line, _, _ in planted] == [_planted_line()], planted
    assert "max_latency_ns" in planted[0][3] or "overload" in planted[0][3], planted


@pytest.mark.parametrize("checker", ["pyright", "mypy"])
@pytest.mark.parametrize("example", _EXAMPLES, ids=lambda p: p.name)
def test_example_checks_clean(checker, example, request):
    if not example.exists():
        pytest.skip("needs the source tree's examples/")
    errors = request.getfixturevalue(checker)
    unexpected = [e for e in errors if e[0] == example.resolve() and not _bare_name(e[2], e[3])]
    assert not unexpected, unexpected
