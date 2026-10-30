#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Check built wheels and sdists before they are uploaded or released.

Usage: python tools/check-wheel-contents.py DIST_FILE [DIST_FILE ...]

For a wheel: the cp312-abi3 tag, or cp314-cp314t for free-threaded CPython,
with a manylinux_2_28 (x86_64, aarch64) or macosx_11_0_arm64 platform; the
package, py.typed, the native extension built for that tag and
its stub, and __builtins__.pyi at the root re-exporting exactly
cudaq_pulse.ops.__all__; LICENSE, NOTICE and every file of LICENSES/ under
.dist-info/licenses and named in METADATA; on Linux both CUDA runtime
libraries; and no bundled CUDA or cuQuantum libraries, byte code, tests or
hand-written top-level stub. For an sdist: LICENSE, NOTICE and LICENSES/.
Exits 1 listing every problem.
"""

from __future__ import annotations

import ast
import re
import sys
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_LICENSES = [
    "LICENSE",
    "NOTICE",
    *sorted(f"LICENSES/{p.name}" for p in (_ROOT / "LICENSES").iterdir()),
]
_PLATFORMS = re.compile(r"(manylinux_2_28_(x86_64|aarch64)|macosx_11_0_arm64)")
# The extension each wheel tag carries: the abi3 one for GIL builds of 3.12
# and newer, and a per-version one for free-threaded CPython, which has no
# stable ABI before abi3t.
_EXTENSIONS = {
    ("cp312", "abi3"): re.compile(r"_cudaq_pulse_native\.abi3\.so"),
    ("cp314", "cp314t"): re.compile(r"_cudaq_pulse_native\.cpython-314t-[\w-]+\.so"),
}
_BUNDLED_CUDA = re.compile(
    r"(^|/)(libcuda|libcudart|libcudensitymat|libcutensor|libcublas|libnvJitLink|libnvrtc)[^/]*$"
)


def _check_builtins_stub(stub: str, ops_source: str) -> list[str]:
    """``__builtins__.pyi`` must re-export exactly ``cudaq_pulse.ops.__all__``."""
    exported = re.findall(r"^from cudaq_pulse\.ops import (\w+) as \1$", stub, re.MULTILINE)
    match = re.search(r"^__all__ = (\[.*?\])", ops_source, re.MULTILINE | re.DOTALL)
    if match is None:
        return ["cudaq_pulse/ops/__init__.py has no __all__"]
    expected = ast.literal_eval(match.group(1))
    if exported != expected:
        return [f"__builtins__.pyi re-exports {exported}, expected cudaq_pulse.ops.__all__"]
    return []


def check_wheel(path: Path) -> list[str]:
    problems = []
    parts = path.name[: -len(".whl")].split("-")
    if len(parts) != 5:
        return [f"{path.name}: not a wheel file name"]
    name, version, python_tag, abi_tag, platform_tag = parts
    extension = _EXTENSIONS.get((python_tag, abi_tag))
    if extension is None:
        expected = " or ".join("-".join(tags) for tags in _EXTENSIONS)
        problems.append(f"tags {python_tag}-{abi_tag}, expected {expected}")
    if not _PLATFORMS.search(platform_tag):
        problems.append(f"platform {platform_tag} is not manylinux_2_28 or macosx_11_0_arm64")
    linux = "manylinux" in platform_tag

    with zipfile.ZipFile(path) as wheel:
        names = set(wheel.namelist())
        dist_info = f"{name}-{version}.dist-info"
        metadata = Parser().parsestr(wheel.read(f"{dist_info}/METADATA").decode())
        if "__builtins__.pyi" in names and "cudaq_pulse/ops/__init__.py" in names:
            problems += _check_builtins_stub(
                wheel.read("__builtins__.pyi").decode(),
                wheel.read("cudaq_pulse/ops/__init__.py").decode(),
            )

    required = [
        "__builtins__.pyi",
        "cudaq_pulse/__init__.py",
        "cudaq_pulse/py.typed",
        "cudaq_pulse/_native/_cudaq_pulse_native.pyi",
        "cudaq_pulse/targets/calibrations/transmon_17q.json",
        *(f"{dist_info}/licenses/{license}" for license in _LICENSES),
    ]
    if linux:
        required += [f"cudaq_pulse/lib/libcudm-runtime-cu{major}.so" for major in (12, 13)]
    problems += [f"missing {item}" for item in required if item not in names]
    native = sorted(
        item.removeprefix("cudaq_pulse/_native/")
        for item in names
        if item.startswith("cudaq_pulse/_native/_cudaq_pulse_native.") and item.endswith(".so")
    )
    if extension is not None and (len(native) != 1 or not extension.fullmatch(native[0])):
        problems.append(
            f"native extensions {native}, expected one matching "
            f"cudaq_pulse/_native/{extension.pattern}"
        )

    declared = set(metadata.get_all("License-File") or [])
    problems += [f"METADATA does not name License-File {f}" for f in _LICENSES if f not in declared]
    if not metadata.get("License-Expression"):
        problems.append("METADATA has no License-Expression")

    for item in sorted(names):
        if _BUNDLED_CUDA.search(item):
            problems.append(f"bundles {item}; CUDA libraries come from the extras")
        if item.endswith(".pyc") or "/__pycache__/" in item or item.startswith("tests/"):
            problems.append(f"contains {item}")
    if "cudaq_pulse/__init__.pyi" in names:
        problems.append("contains cudaq_pulse/__init__.pyi; the package is typed from its sources")
    return [f"{path.name}: {problem}" for problem in problems]


def check_sdist(path: Path) -> list[str]:
    with tarfile.open(path) as sdist:
        members = {"/".join(m.name.split("/")[1:]) for m in sdist.getmembers()}
    return [f"{path.name}: missing {f}" for f in _LICENSES if f not in members]


def main() -> int:
    files = [Path(arg) for arg in sys.argv[1:]]
    if not files:
        print(__doc__, file=sys.stderr)
        return 2
    problems = []
    for path in files:
        if path.suffix == ".whl":
            problems += check_wheel(path)
        elif path.name.endswith(".tar.gz"):
            problems += check_sdist(path)
        else:
            problems.append(f"{path}: not a wheel or sdist")
    for problem in problems:
        print(problem)
    if not problems:
        print(f"{len(files)} distribution file(s) OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
