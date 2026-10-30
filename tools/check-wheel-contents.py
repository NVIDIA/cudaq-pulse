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
cudaq_pulse.ops.__all__; LICENSE, NOTICE, THIRD_PARTY_NOTICES and every file
of LICENSES/ (LLVM's LICENSE.TXT among them) under .dist-info/licenses and
named in METADATA; THIRD_PARTY_NOTICES as tools/third-party-notices.py
generates it, listing LLVM and every library grafted into the wheel, with
each listed license in License-Expression; on Linux both CUDA runtime
libraries; and no CUDA runtime library (those come from the CUDA wheels), no
GPL or LGPL library or license, and no byte code, tests or hand-written
top-level stub. For an sdist: LICENSE, NOTICE, THIRD_PARTY_NOTICES and
LICENSES/. Exits 1 listing every problem.
"""

from __future__ import annotations

import ast
import importlib.util
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
    "THIRD_PARTY_NOTICES",
    *sorted(f"LICENSES/{p.name}" for p in (_ROOT / "LICENSES").iterdir()),
]
_spec = importlib.util.spec_from_file_location(
    "third_party_notices", _ROOT / "tools" / "third-party-notices.py"
)
assert _spec is not None and _spec.loader is not None
notices = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(notices)
_PLATFORMS = re.compile(r"(manylinux_2_28_(x86_64|aarch64)|macosx_11_0_arm64)")
# The extension each wheel tag carries: the abi3 one for GIL builds of 3.12
# and newer, and a per-version one for free-threaded CPython, which has no
# stable ABI before abi3t.
_EXTENSIONS = {
    ("cp312", "abi3"): re.compile(r"_cudaq_pulse_native\.abi3\.so"),
    ("cp314", "cp314t"): re.compile(r"_cudaq_pulse_native\.cpython-314t-[\w-]+\.so"),
}
# The CUDA runtime, math and cuQuantum libraries, any case of a name. This
# package's own libcudm-runtime-cuNN is not one.
_BUNDLED_CUDA = re.compile(
    r"(^|/)(libcuda|libcudart|libcudensitymat|libcutensor|libcutensornet|libcustatevec"
    r"|libcublas|libcublasLt|libcusolver|libcusparse|libcurand|libcufft|libcupti|libnvjitlink"
    r"|libnvrtc|libnvToolsExt|libnvtx|libnccl|libcudnn|libnvblas|libnvperf)([._-][^/]*)?$",
    re.IGNORECASE,
)
_GPL_EXPRESSION = re.compile(r"\b(L?GPL|AGPL)-")


def _check_notices(names: set[str], text: str, expression: str) -> list[str]:
    """THIRD_PARTY_NOTICES is generated, lists LLVM and every grafted library,
    and License-Expression names each license it lists and no GPL or LGPL."""
    problems = []
    listed = notices.listed_components(text)
    unknown = sorted(listed - notices._BY_NAME.keys())
    problems += [
        f"THIRD_PARTY_NOTICES lists {name}, which is not a known component" for name in unknown
    ]
    known = listed & notices._BY_NAME.keys()
    if "LLVM" not in listed:
        problems.append("THIRD_PARTY_NOTICES does not list LLVM")
    elif not unknown and not notices.missing_license_files(known) and text != notices.render(known):
        problems.append("THIRD_PARTY_NOTICES is not what tools/third-party-notices.py generates")
    grafted, graft_problems = notices.grafted_components(sorted(names))
    # A grafted GPL or LGPL library is reported with every other bundled one.
    problems += [p for p in graft_problems if not p.endswith(notices.GPL_PROBLEM)]
    problems += [
        f"THIRD_PARTY_NOTICES does not list {name}, which the wheel grafts"
        for name in sorted(grafted - listed)
    ]
    normalized = " ".join(expression.split())
    problems += [
        f"License-Expression does not name {notices._BY_NAME[name].spdx} ({name})"
        for name in sorted(known)
        if notices._BY_NAME[name].spdx not in normalized
    ]
    if _GPL_EXPRESSION.search(normalized):
        problems.append(f"License-Expression {normalized!r} names a GPL or LGPL license")
    for component in sorted(known):
        if _GPL_EXPRESSION.search(notices._BY_NAME[component].spdx):
            problems.append(f"THIRD_PARTY_NOTICES lists {component}, under a GPL or LGPL license")
    return problems


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
        notices_file = f"{dist_info}/licenses/THIRD_PARTY_NOTICES"
        if notices_file in names:
            problems += _check_notices(
                names,
                wheel.read(notices_file).decode(),
                metadata.get("License-Expression") or "",
            )
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
        if notices.GPL_LIBRARIES.search(item):
            problems.append(f"bundles {item}, {notices.GPL_PROBLEM}")
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
