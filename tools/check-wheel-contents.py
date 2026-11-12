#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check built wheels and sdists before they are uploaded or released.

Usage: python tools/check-wheel-contents.py DIST_FILE [DIST_FILE ...]

Every wheel has a manylinux_2_28 (x86_64, aarch64) or macosx_11_0_arm64
platform, and its licenses under .dist-info/licenses and named in METADATA.

A cudaq-pulse wheel: the cp312-abi3 tag, or cp314-cp314t for free-threaded
CPython; the package, py.typed, the native extension built for that tag and
its stub, and __builtins__.pyi at the root re-exporting exactly
cudaq_pulse.ops.__all__; on Linux both CUDA runtime libraries; a
requirement of exactly cudaq-pulse-tools==<its version>, whose wheel for its
platform must be among DIST_FILEs; and no cudaq-pulse-opt, byte code, tests
or hand-written top-level stub. Its licenses: LICENSE, NOTICE,
THIRD_PARTY_NOTICES and every file of LICENSES/ (LLVM's LICENSE.TXT among
them).

A cudaq-pulse-tools wheel: the py3-none tag; the cudaq-pulse-opt executable
in its scripts and nothing else but its metadata and any library a wheel
repair grafted; the licenses of packaging/cudaq-pulse-tools (LICENSE, NOTICE,
THIRD_PARTY_NOTICES, LLVM's and the other texts that file needs); and no
requirements.

Both: LICENSE, NOTICE and the LICENSES/ texts byte for byte the
repository's; THIRD_PARTY_NOTICES as tools/third-party-notices.py generates it,
listing LLVM and every library grafted into the wheel, with each listed
license in License-Expression; and no CUDA runtime library (those come from
the CUDA wheels) and no GPL or LGPL library or license.

For an sdist: LICENSE, NOTICE, THIRD_PARTY_NOTICES and LICENSES/; the
license files of packaging/cudaq-pulse-tools as regular files, all but
THIRD_PARTY_NOTICES byte for byte the repository's; a PKG-INFO requiring
exactly cudaq-pulse-tools==<its version>; and no lock file (uv.lock,
poetry.lock, pdm.lock, Pipfile.lock, or a requirements*.txt with hashes).
Exits 1 listing every problem.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
import tarfile
import zipfile
from email.message import Message
from email.parser import Parser
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _licenses(project: Path) -> list[str]:
    return [
        "LICENSE",
        "NOTICE",
        "THIRD_PARTY_NOTICES",
        *sorted(f"LICENSES/{p.name}" for p in (project / "LICENSES").iterdir()),
    ]


_LICENSES = _licenses(_ROOT)
_TOOLS = "cudaq-pulse-tools"
_TOOLS_LICENSES = _licenses(_ROOT / "packaging" / _TOOLS)
_DRIVER = "cudaq-pulse-opt"
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
# Lock files pin the development tools and the optional CUDA wheels, and
# composition scanners read them as components the sdist ships.
_LOCK_FILE = re.compile(r"(^|/)(uv|poetry|pdm|Pipfile)\.lock$")
_REQUIREMENTS = re.compile(r"(^|/)requirements[^/]*\.txt$")


def _check_notices(
    wheel: zipfile.ZipFile, names: set[str], dist_info: str, metadata: Message
) -> list[str]:
    """THIRD_PARTY_NOTICES is generated, lists LLVM and every grafted library,
    and License-Expression names each license it lists and no GPL or LGPL;
    the wheel bundles no CUDA and no GPL or LGPL library."""
    problems = []
    for item in sorted(names):
        if _BUNDLED_CUDA.search(item):
            problems.append(f"bundles {item}; CUDA libraries come from the extras")
        if notices.GPL_LIBRARIES.search(item):
            problems.append(f"bundles {item}, {notices.GPL_PROBLEM}")
    notices_file = f"{dist_info}/licenses/THIRD_PARTY_NOTICES"
    if notices_file not in names:
        return problems
    text = wheel.read(notices_file).decode()
    expression = metadata.get("License-Expression") or ""
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


def _wheel_name(path: Path) -> list[str] | None:
    """The five fields of a wheel file name, or ``None``."""
    parts = path.name[: -len(".whl")].split("-")
    return parts if len(parts) == 5 else None


def _check_licenses(
    wheel: zipfile.ZipFile, dist_info: str, metadata: Message, licenses: list[str]
) -> list[str]:
    """Every license file is present and named in METADATA, and all but
    THIRD_PARTY_NOTICES are the repository's own LICENSE, NOTICE and
    LICENSES/ texts, byte for byte."""
    names = set(wheel.namelist())
    problems = []
    for f in licenses:
        item = f"{dist_info}/licenses/{f}"
        if item not in names:
            problems.append(f"missing {item}")
        elif f != "THIRD_PARTY_NOTICES" and wheel.read(item) != (_ROOT / f).read_bytes():
            problems.append(f"{item} differs from the repository's {f}")
    declared = set(metadata.get_all("License-File") or [])
    problems += [f"METADATA does not name License-File {f}" for f in licenses if f not in declared]
    problems += [
        f"METADATA names License-File {f}, not one of its licenses"
        for f in sorted(declared - set(licenses))
    ]
    if not metadata.get("License-Expression"):
        problems.append("METADATA has no License-Expression")
    return problems


def _check_tools_requirement(metadata: Message, version: str) -> list[str]:
    """cudaq-pulse requires exactly cudaq-pulse-tools==<its version>."""
    requirement = f"{_TOOLS}=={version}"
    tools_requirements = [
        r for r in metadata.get_all("Requires-Dist") or [] if re.match(rf"{_TOOLS}\b", r)
    ]
    if tools_requirements != [requirement]:
        return [f"requires {tools_requirements}, expected exactly ['{requirement}']"]
    return []


def check_tools_wheel(path: Path) -> list[str]:
    name, version, python_tag, abi_tag, platform_tag = _wheel_name(path) or [""] * 5
    problems = []
    if (python_tag, abi_tag) != ("py3", "none"):
        problems.append(f"tags {python_tag}-{abi_tag}, expected py3-none")
    if not _PLATFORMS.search(platform_tag):
        problems.append(f"platform {platform_tag} is not manylinux_2_28 or macosx_11_0_arm64")
    with zipfile.ZipFile(path) as wheel:
        names = set(wheel.namelist())
        dist_info = f"{name}-{version}.dist-info"
        metadata = Parser().parsestr(wheel.read(f"{dist_info}/METADATA").decode())
        executable = f"{name}-{version}.data/scripts/{_DRIVER}"
        if executable not in names:
            problems.append(f"missing {executable}")
        elif not (wheel.getinfo(executable).external_attr >> 16) & 0o111:
            problems.append(f"{executable} is not executable")
        problems += _check_notices(wheel, names, dist_info, metadata)
        problems += _check_licenses(wheel, dist_info, metadata, _TOOLS_LICENSES)
    expected = {executable, *(f"{dist_info}/{f}" for f in ("METADATA", "WHEEL", "RECORD"))}
    expected |= {f"{dist_info}/licenses/{f}" for f in _TOOLS_LICENSES}
    expected |= {item for item in names if notices.GRAFTED_DIR.search(item)}
    # auditwheel and delocate write an entry for every directory.
    problems += [f"contains {item}" for item in sorted(names - expected) if not item.endswith("/")]
    if metadata.get_all("Requires-Dist"):
        problems.append(f"requires {metadata.get_all('Requires-Dist')}; it requires nothing")
    return [f"{path.name}: {problem}" for problem in problems]


def check_wheel(path: Path) -> list[str]:
    problems = []
    parts = _wheel_name(path)
    if parts is None:
        return [f"{path.name}: not a wheel file name"]
    name, version, python_tag, abi_tag, platform_tag = parts
    if name == _TOOLS.replace("-", "_"):
        return check_tools_wheel(path)
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
        problems += _check_notices(wheel, names, dist_info, metadata)
        if "__builtins__.pyi" in names and "cudaq_pulse/ops/__init__.py" in names:
            problems += _check_builtins_stub(
                wheel.read("__builtins__.pyi").decode(),
                wheel.read("cudaq_pulse/ops/__init__.py").decode(),
            )
        problems += _check_licenses(wheel, dist_info, metadata, _LICENSES)

    required = [
        "__builtins__.pyi",
        "cudaq_pulse/__init__.py",
        "cudaq_pulse/py.typed",
        "cudaq_pulse/_native/_cudaq_pulse_native.pyi",
        "cudaq_pulse/targets/calibrations/transmon_17q.json",
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

    problems += _check_tools_requirement(metadata, version)

    for item in sorted(names):
        if item.endswith(".pyc") or "/__pycache__/" in item or item.startswith("tests/"):
            problems.append(f"contains {item}")
        if item.rsplit("/", 1)[-1] == _DRIVER or f"{dist_info[: -len('.dist-info')]}.data/" in item:
            problems.append(f"contains {item}; {_DRIVER} is in the {_TOOLS} wheel")
    if "cudaq_pulse/__init__.pyi" in names:
        problems.append("contains cudaq_pulse/__init__.pyi; the package is typed from its sources")
    return [f"{path.name}: {problem}" for problem in problems]


def check_pairs(wheels: list[Path]) -> list[str]:
    """Every cudaq-pulse wheel needs the cudaq-pulse-tools wheel it requires."""
    tools = _TOOLS.replace("-", "_")
    parsed = [(path, parts) for path in wheels if (parts := _wheel_name(path)) is not None]
    drivers = {
        (version, platform) for _, (name, version, _, _, platform) in parsed if name == tools
    }
    return [
        f"{path.name}: no {tools}-{version}-py3-none-{platform}.whl among the files checked"
        for path, (name, version, _, _, platform) in parsed
        if name != tools and (version, platform) not in drivers
    ]


def check_sdist(path: Path) -> list[str]:
    problems = []
    with tarfile.open(path) as sdist:
        members = {"/".join(m.name.split("/")[1:]): m for m in sdist.getmembers()}
        pkg_info = members.get("PKG-INFO")
        file = sdist.extractfile(pkg_info) if pkg_info is not None else None
        metadata = Parser().parsestr(file.read().decode()) if file is not None else None
        # The tools wheel is built from an unpacked sdist too; its license
        # files are links in the repository, and must be files here.
        for f in _TOOLS_LICENSES:
            item = f"packaging/{_TOOLS}/{f}"
            member = members.get(item)
            if member is None:
                problems.append(f"missing {item}")
            elif not member.isfile():
                problems.append(f"{item} is not a regular file")
            elif f != "THIRD_PARTY_NOTICES":
                content = sdist.extractfile(member)
                if content is None or content.read() != (_ROOT / f).read_bytes():
                    problems.append(f"{item} differs from the repository's {f}")
        for name, member in sorted(members.items()):
            if _LOCK_FILE.search(name):
                problems.append(f"ships {name}, a lock file")
            elif _REQUIREMENTS.search(name) and member.isfile():
                content = sdist.extractfile(member)
                if content is not None and b"--hash=" in content.read():
                    problems.append(f"ships {name}, a requirements file with hashes")
    problems += [f"missing {f}" for f in _LICENSES if f not in members]
    if metadata is None:
        problems.append("missing PKG-INFO")
    else:
        problems += _check_tools_requirement(metadata, metadata.get("Version") or "")
    return [f"{path.name}: {problem}" for problem in problems]


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
    problems += check_pairs([path for path in files if path.suffix == ".whl"])
    for problem in problems:
        print(problem)
    if not problems:
        print(f"{len(files)} distribution file(s) OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
