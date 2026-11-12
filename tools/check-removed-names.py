#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail if the docs, the examples or the README name a public name that was removed.

The API was renamed before the first release, with no aliases: a removed name
in a guide or an example is an error a reader would copy. This lists them (the
old spelling and what replaced it) and finds them in README.md and the files
under docs/ and examples/. A line that must name one carries ``rename-ok``.

Usage: python tools/check-removed-names.py [FILES...]   (default: those git ls-files lists)
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

_WORDS = {
    "TargetDescription": "pulse.quantum_system(name=..., qudits=...)",
    "DeviceDescription": "pulse.quantum_system(name=..., qudits=...)",
    "Lowered": "ScheduledKernel",
    "MachineModel": "ControlSystem",
    "ControllerLimits": "Controller",
    "ResourceReport": "UsageReport",
    "FunctionResources": "FunctionUsage",
    "NoDeviceError": "NoQuantumSystemError",
    "NoProcessDeviceError": "NoProcessQuantumSystemError",
    "DeviceDefinitionError": "QuantumSystemDefinitionError",
    "DeviceMismatchError": "QuantumSystemMismatchError",
    "qudit_handles": "qs.qudits",
    "in_controller_form": "runs_on_processor",
    "max_in_flight": "slots",
    "peak_in_flight": "peak_slots_used",
    "function_id": "dispatch_key",
    "transmon_17q": "transmon_surface17",
    "transmon_generic": "transmon_lattice",
    "krinner17": "transmon_surface17",
    "frequency_spread_hz": "frequency_range_hz",
    "current_device": "current_quantum_system",
    "CUDAQ_PULSE_DEVICE": "CUDAQ_PULSE_QUANTUM_SYSTEM",
    "CUDAQ_PULSE_TARGET": "CUDAQ_PULSE_QUANTUM_SYSTEM",
    "export_controller": "export_to_controller",
    "export_emulation": "export_to_emulator",
    "n_qubits": "qudit_count",
    "n_qudits": "qudit_count",
}
_PATTERNS = [
    (re.compile(rf"(?<![\w-]){re.escape(old)}(?![\w-])"), old, new) for old, new in _WORDS.items()
] + [
    (
        re.compile(r"\bcudaq_pulse\.(?:targets|devices)\b"),
        "cudaq_pulse.targets",
        "cudaq_pulse.quantum_systems",
    ),
    (re.compile(r"\bpulse\.device\("), "pulse.device(", "pulse.quantum_system("),
    (re.compile(r"\b(?:Event|Capability)\.ACQUIRE\b"), "ACQUIRE", "READOUT"),
    (
        re.compile(r"\bgates\.(?:SX|MEASURE)\b"),
        "gates.SX / gates.MEASURE",
        "gates.X90 / gates.READOUT",
    ),
    (re.compile(r"\.resources\("), ".resources()", "processor.usage(fn) / UsageReport"),
    (
        re.compile(r"\b(?:qs|dev|device|compiled|scheduled|low|lowered)\.description\b"),
        ".description",
        "the QuantumSystem's own fields",
    ),
    (re.compile(r"\bcompile\([^()]*\bmachine="), "compile(machine=)", "compile(control_system=)"),
    (
        re.compile(r"\b(?:simulate|emulate|validate)\((?:[^()]|\([^()]*\))*\bshots="),
        "shots=",
        "shots_count=",
    ),
    (
        re.compile(r"\bsimulate\((?:[^()]|\([^()]*\))*\bplant="),
        "simulate(plant=)",
        "simulate(quantum_system=)",
    ),
]
_SUFFIXES = {".py", ".pyi", ".md", ".rst", ".txt", ".ipynb", ".svg"}


def _text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    if path.suffix != ".ipynb":
        return raw
    try:
        cells = json.loads(raw).get("cells", [])
    except ValueError:
        return raw
    return "\n".join("".join(cell.get("source", [])) for cell in cells)


def covered(path: Path) -> bool:
    parts = path.parts
    return path.as_posix() == "README.md" or (bool(parts) and parts[0] in ("docs", "examples"))


def check(path: Path) -> list[str]:
    if path.suffix not in _SUFFIXES:
        return []
    try:
        text = _text(path)
    except (UnicodeDecodeError, IsADirectoryError, FileNotFoundError):
        return []
    lines = text.splitlines()
    found = set()
    for pattern, old, new in _PATTERNS:
        for match in pattern.finditer(text):
            number = text.count("\n", 0, match.start()) + 1
            if "rename-ok" not in lines[number - 1]:
                found.add((number, old, new))
    return [f"{path}:{number}: {old} was removed; use {new}" for number, old, new in sorted(found)]


def main() -> int:
    if sys.argv[1:]:
        paths = [Path(p) for p in sys.argv[1:]]
    else:
        listed = subprocess.run(
            ["git", "ls-files", "-z", "README.md", "docs", "examples"],
            capture_output=True,
            check=True,
            text=True,
        )
        paths = [Path(p) for p in listed.stdout.split("\0") if p]
    errors = [error for path in paths if covered(path) for error in check(path)]
    for error in errors:
        print(error)
    if errors:
        print(
            f"{len(errors)} removed public names in the docs or examples: use the name"
            " given, or mark a line that must name one with rename-ok"
        )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
