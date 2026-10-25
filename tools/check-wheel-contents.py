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

For a wheel: the cp312-abi3 tag with a manylinux_2_28 (x86_64, aarch64) or
macosx_11_0_arm64 platform; the package, py.typed, the native extension and
its stub; LICENSE, NOTICE and every file of LICENSES/ under
.dist-info/licenses and named in METADATA; on Linux both CUDA runtime
libraries; and no bundled CUDA or cuQuantum libraries, byte code, tests or
hand-written top-level stub. For an sdist: LICENSE, NOTICE and LICENSES/.
Exits 1 listing every problem.
"""

from __future__ import annotations

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
_BUNDLED_CUDA = re.compile(
    r"(^|/)(libcuda|libcudart|libcudensitymat|libcutensor|libcublas|libnvJitLink|libnvrtc)[^/]*$"
)


def check_wheel(path: Path) -> list[str]:
    problems = []
    parts = path.name[: -len(".whl")].split("-")
    if len(parts) != 5:
        return [f"{path.name}: not a wheel file name"]
    name, version, python_tag, abi_tag, platform_tag = parts
    if (python_tag, abi_tag) != ("cp312", "abi3"):
        problems.append(f"tags {python_tag}-{abi_tag}, expected cp312-abi3")
    if not _PLATFORMS.search(platform_tag):
        problems.append(f"platform {platform_tag} is not manylinux_2_28 or macosx_11_0_arm64")
    linux = "manylinux" in platform_tag

    with zipfile.ZipFile(path) as wheel:
        names = set(wheel.namelist())
        dist_info = f"{name}-{version}.dist-info"
        metadata = Parser().parsestr(wheel.read(f"{dist_info}/METADATA").decode())

    required = [
        "cudaq_pulse/__init__.py",
        "cudaq_pulse/py.typed",
        "cudaq_pulse/_native/_cudaq_pulse_native.abi3.so",
        "cudaq_pulse/_native/_cudaq_pulse_native.pyi",
        "cudaq_pulse/targets/calibrations/krinner17.json",
        *(f"{dist_info}/licenses/{license}" for license in _LICENSES),
    ]
    if linux:
        required += [f"cudaq_pulse/lib/libcudm-runtime-cu{major}.so" for major in (12, 13)]
    problems += [f"missing {item}" for item in required if item not in names]

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
