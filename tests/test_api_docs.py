# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every public name is in the API reference (source tree only)."""

import importlib
import re
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_API = _ROOT / "docs" / "api"
pytestmark = pytest.mark.skipif(not _API.is_dir(), reason="needs the source tree's docs/")

# The documented submodules whose __all__ is the public API (CONTRIBUTING.md,
# "Versioning and stability").
PUBLIC_MODULES = [
    "cudaq_pulse",
    "cudaq_pulse.ops",
    "cudaq_pulse.compiler",
    "cudaq_pulse.quantum_systems",
    "cudaq_pulse.quantum_systems.transmon_model",
    "cudaq_pulse.runtime",
    "cudaq_pulse.runtime.simulate",
    "cudaq_pulse.viz",
    "cudaq_pulse.controller",
    "cudaq_pulse.backends",
    "cudaq_pulse.classical",
    "cudaq_pulse.classical.wire",
    "cudaq_pulse.classical.emulation",
    "cudaq_pulse.classical.calibration",
    "cudaq_pulse.classical.loop",
    "cudaq_pulse.quantum_systems.drift",
]

_DIRECTIVE = re.compile(r"^\.\. auto(\w+):: (\S+)", re.M)
_MEMBERS = re.compile(r"^\.\. automodule:: (\S+)\n((?:   :.*\n)+)", re.M)


def _resolve(path):
    """The object *path* names, by attribute access from the package, as
    Sphinx resolves it."""
    parts = path.split(".")
    obj = importlib.import_module(parts[0])
    for i, part in enumerate(parts[1:], start=1):
        if not hasattr(obj, part):
            importlib.import_module(".".join(parts[: i + 1]))
        obj = getattr(obj, part)
    return obj


def _documented():
    objects, modules = [], set()
    for page in sorted(_API.glob("*.rst")):
        text = page.read_text()
        for kind, path in _DIRECTIVE.findall(text):
            if kind == "module":
                modules.add(path)
            else:
                objects.append(_resolve(path))
        for module, options in _MEMBERS.findall(text):
            members = re.search(r":members:(.*)", options)
            if members is None:
                continue
            names = [m.strip() for m in members.group(1).split(",") if m.strip()]
            # A bare :members: documents the module's __all__.
            names = names or list(importlib.import_module(module).__all__)
            objects += [_resolve(f"{module}.{name}") for name in names]
    return objects, modules


def test_every_public_name_is_documented():
    objects, modules = _documented()
    ids = {id(o) for o in objects}
    pages = "".join(p.read_text() for p in _API.glob("*.rst"))
    missing = []
    for module in PUBLIC_MODULES:
        mod = importlib.import_module(module)
        for name in mod.__all__:
            value = getattr(mod, name)
            if isinstance(value, types.ModuleType):
                found = value.__name__ in modules or f"{value.__name__}." in pages
            else:
                found = id(value) in ids
            if not found:
                missing.append(f"{module}.{name}")
    assert missing == []


def test_contributing_names_the_public_modules():
    text = (_ROOT / "CONTRIBUTING.md").read_text()
    assert [m for m in PUBLIC_MODULES if f"`{m}`" not in text] == []
