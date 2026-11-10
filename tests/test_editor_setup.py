# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``python -m cudaq_pulse.doctor --setup-editor`` and the mypy plugin."""

from __future__ import annotations

import configparser
import io
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import doctor

from tests._tools import import_module

OPS = list(pulse.ops.__all__)
PLUGIN = "cudaq_pulse.mypy_plugin"


def setup(root: Path, *args: str) -> tuple[int, str]:
    out = io.StringIO()
    code = doctor.main(["--setup-editor", "--project", str(root), *args], out=out)
    return code, out.getvalue()


def test_no_configuration_writes_nothing(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "app"\n')
    code, out = setup(tmp_path, "--yes")
    assert code == 0
    assert "mypy: no configuration" in out and "ruff: no configuration" in out
    assert (tmp_path / "pyproject.toml").read_text() == '[project]\nname = "app"\n'
    assert sorted(p.name for p in tmp_path.iterdir()) == ["pyproject.toml"]


def test_without_yes_shows_the_change_and_writes_nothing(tmp_path):
    text = "[tool.ruff]\nline-length = 100\n"
    (tmp_path / "pyproject.toml").write_text(text)
    code, out = setup(tmp_path)
    assert code == 1
    assert '+    "drive",' in out and "--yes" in out
    assert (tmp_path / "pyproject.toml").read_text() == text


def test_pyproject_with_both_tools(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "app"\n\n[tool.mypy]\nstrict = true\n\n'
        '[tool.ruff]\nline-length = 100\nbuiltins = ["mine"]\n\n[tool.ruff.lint]\nselect = ["F"]\n'
    )
    code, out = setup(tmp_path, "--yes")
    assert code == 0, out
    data = tomllib.loads((tmp_path / "pyproject.toml").read_text())
    assert data["tool"]["mypy"] == {"strict": True, "plugins": [PLUGIN]}
    assert data["tool"]["ruff"]["builtins"] == ["mine", *OPS]
    assert data["tool"]["ruff"]["line-length"] == 100
    assert data["tool"]["ruff"]["lint"] == {"select": ["F"]}
    code, out = setup(tmp_path)
    assert code == 0
    assert "mypy:" in out and out.count("is already set up") == 2


def test_mypy_ini_comes_first_and_keeps_its_plugins(tmp_path):
    (tmp_path / "mypy.ini").write_text("[mypy]\nplugins = pydantic.mypy\nstrict = True\n")
    (tmp_path / "pyproject.toml").write_text("[tool.mypy]\nstrict = true\n")
    code, _ = setup(tmp_path, "--yes")
    assert code == 0
    ini = configparser.ConfigParser()
    ini.read(tmp_path / "mypy.ini")
    assert ini["mypy"]["plugins"] == f"pydantic.mypy, {PLUGIN}"
    assert ini["mypy"]["strict"] == "True"
    assert (tmp_path / "pyproject.toml").read_text() == "[tool.mypy]\nstrict = true\n"


def test_setup_cfg_mypy_section(tmp_path):
    (tmp_path / "setup.cfg").write_text("[metadata]\nname = app\n\n[mypy]\nstrict = True")
    code, _ = setup(tmp_path, "--yes")
    assert code == 0
    ini = configparser.ConfigParser()
    ini.read(tmp_path / "setup.cfg")
    assert ini["mypy"]["plugins"] == PLUGIN
    assert ini["metadata"]["name"] == "app"


def test_ruff_toml_root_table(tmp_path):
    (tmp_path / "ruff.toml").write_text("line-length = 88\n\n[lint]\nselect = ['F']")
    code, _ = setup(tmp_path, "--yes")
    assert code == 0
    data = tomllib.loads((tmp_path / "ruff.toml").read_text())
    assert data["builtins"] == OPS
    assert data["line-length"] == 88 and data["lint"] == {"select": ["F"]}


def test_yes_and_project_need_setup_editor(tmp_path, capsys):
    with pytest.raises(SystemExit):
        doctor.main(["--yes"])


def test_importing_the_package_writes_nothing(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.ruff]\n[tool.mypy]\n")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    subprocess.run([sys.executable, "-c", "import cudaq_pulse"], cwd=tmp_path, env=env, check=True)
    assert (tmp_path / "pyproject.toml").read_text() == "[tool.ruff]\n[tool.mypy]\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["pyproject.toml"]


KERNELS = """\
import cudaq_pulse as pulse


def square(x: int) -> int:
    return x * x


@pulse.kernel
def good(qubit: pulse.Qudit, amplitude: float) -> None:
    line = get_drive_line(qubit)
    drive(line, gaussian(40, amplitude, 10.0))


@pulse.kernel
def bad(qubit: pulse.Qudit) -> None:
    line = get_drive_line(qubit)
    drive(line.tone, gaussian(40, 0.1, 10.0), tone=line)
    drve(line)
"""


def test_mypy_plugin_defines_and_types_the_kernel_language(tmp_path):
    api = import_module("mypy.api")
    (tmp_path / "app.py").write_text(KERNELS)
    (tmp_path / "plain.py").write_text("drive(1)\n")
    (tmp_path / "mypy.ini").write_text(f"[mypy]\nplugins = {PLUGIN}\n")
    for _ in range(2):  # the second run reads mypy's cache
        stdout, _, _ = api.run(
            [
                "--config-file",
                str(tmp_path / "mypy.ini"),
                "--cache-dir",
                str(tmp_path / "cache"),
                str(tmp_path / "app.py"),
                str(tmp_path / "plain.py"),
            ]
        )
        errors = sorted(
            line.split(": error: ")[0].rsplit(":", 1)[-1] + " " + line.rsplit("[", 1)[-1]
            for line in stdout.splitlines()
            if ": error: " in line
        )
        assert errors == ["1 name-defined]", "17 arg-type]", "17 arg-type]", "18 name-defined]"], (
            stdout
        )
