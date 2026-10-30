#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write or check a THIRD_PARTY_NOTICES file from what the binaries link.

Usage:
  python tools/third-party-notices.py write --notices FILE
      [--map MAP ...] [--wheel WHEEL ...]
  python tools/third-party-notices.py check --notices FILE
      [--map MAP ...] [--wheel WHEEL ...]

A MAP is a linker map (ld64 -map, GNU ld or lld -Map), which the build writes
to <build dir>/link-maps/<target>.map for every binary a wheel ships. The
third-party code a binary links is read from the static archive members the
map names; a WHEEL adds the shared libraries auditwheel or delocate grafted
into it (<package>.libs/, <package>/.dylibs/).

write  writes the notices for exactly the components found, each with its
       license text from LICENSES/.
check  fails if a component found is not listed, if a map names an archive or
       a library this tool does not know, or if the file is not what write
       produces for the components it lists. A listed component nothing here
       links is only reported: another platform's build may link it.

Shared libraries the binaries load from the system are not shipped and are
not listed. Exits 1 listing every problem.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import NamedTuple

_ROOT = Path(__file__).resolve().parents[1]
_LLVM_VERSION = json.loads((_ROOT / "tools" / "mlir-toolchain.lock.json").read_text())[
    "llvm_version"
]


class Component(NamedTuple):
    name: str
    spdx: str
    origin: str
    copyright: str
    license_file: str


COMPONENTS = [
    Component(
        "LLVM",
        "Apache-2.0 WITH LLVM-exception",
        f"https://github.com/llvm/llvm-project, release llvmorg-{_LLVM_VERSION}",
        "Copyright (c) 2003-2019 University of Illinois at Urbana-Champaign and\n"
        "the LLVM project contributors.",
        "LICENSES/LLVM-LICENSE.txt",
    ),
    Component(
        "MLIR",
        "Apache-2.0 WITH LLVM-exception",
        f"https://github.com/llvm/llvm-project, release llvmorg-{_LLVM_VERSION}",
        "",
        "LICENSES/MLIR-LICENSE.txt",
    ),
    Component(
        "xxHash (LLVM Support)",
        "BSD-2-Clause",
        "llvm/lib/Support/xxhash.cpp",
        "Copyright (C) 2012-2016, Yann Collet.",
        "LICENSES/LLVM-xxhash-LICENSE.txt",
    ),
    Component(
        "regex (LLVM Support)",
        "Spencer-94",
        "llvm/lib/Support/regcomp.c, regexec.c, regerror.c, regfree.c, regstrlcpy.c",
        "Copyright 1992, 1993, 1994 Henry Spencer.",
        "LICENSES/LLVM-regex-COPYRIGHT.txt",
    ),
    Component(
        "BLAKE3 (LLVM Support)",
        "CC0-1.0 OR Apache-2.0",
        "llvm/lib/Support/BLAKE3",
        "",
        "LICENSES/LLVM-BLAKE3-LICENSE.txt",
    ),
    Component(
        "ConvertUTF (LLVM Support)",
        "Unicode-DFS-2015",
        "llvm/lib/Support/ConvertUTF.cpp",
        "Copyright (c) 1991-2015 Unicode, Inc.",
        "LICENSES/LLVM-ConvertUTF-Unicode-LICENSE.txt",
    ),
    Component(
        "MD5 (LLVM Support)",
        "LicenseRef-LLVM-MD5",
        "llvm/lib/Support/MD5.cpp",
        "Written by Alexander Peslyak (Solar Designer), placed in the public domain.",
        "LICENSES/LLVM-MD5-LICENSE.txt",
    ),
    Component(
        "nanobind",
        "BSD-3-Clause",
        "https://github.com/wjakob/nanobind",
        "Copyright (c) 2022 Wenzel Jakob.",
        "LICENSES/nanobind-LICENSE.txt",
    ),
    Component(
        "robin-map (in nanobind)",
        "MIT",
        "https://github.com/Tessil/robin-map",
        "Copyright (c) 2017 Thibaut Goetghebuer-Planchon.",
        "LICENSES/robin-map-LICENSE.txt",
    ),
    # Libraries a wheel repair could graft. None is grafted today: the MLIR
    # toolchain is built without zlib, zstd, libxml2, libedit, libffi or
    # terminfo. Grafting one needs its license text in LICENSES/ first.
    Component("zlib", "Zlib", "https://zlib.net", "", "LICENSES/zlib-LICENSE.txt"),
    Component(
        "Zstandard",
        "BSD-3-Clause",
        "https://github.com/facebook/zstd",
        "",
        "LICENSES/zstd-LICENSE.txt",
    ),
    Component(
        "libxml2",
        "MIT",
        "https://gitlab.gnome.org/GNOME/libxml2",
        "",
        "LICENSES/libxml2-LICENSE.txt",
    ),
    Component(
        "libffi",
        "MIT",
        "https://github.com/libffi/libffi",
        "",
        "LICENSES/libffi-LICENSE.txt",
    ),
    Component(
        "ncurses",
        "X11-distribute-modifications-variant",
        "https://invisible-island.net/ncurses",
        "",
        "LICENSES/ncurses-LICENSE.txt",
    ),
]
_BY_NAME = {component.name: component for component in COMPONENTS}

# (archive, member) patterns over the map's archive members, and what each
# brings in. A member matches every row whose patterns it matches.
_ARCHIVE_COMPONENTS = [
    (r"libLLVM\w*\.a", r".*", ["LLVM"]),
    (r"libMLIR\w*\.a", r".*", ["MLIR"]),
    (r"libLLVMSupport\.a", r"xxhash\.cpp\.o", ["xxHash (LLVM Support)"]),
    (
        r"libLLVMSupport\.a",
        r"reg(comp|error|exec|free|strlcpy)\.c\.o",
        ["regex (LLVM Support)"],
    ),
    (r"libLLVMSupport\.a", r"blake3[\w.-]*\.(c|S)\.o", ["BLAKE3 (LLVM Support)"]),
    (r"libLLVMSupport\.a", r"ConvertUTF\.cpp\.o", ["ConvertUTF (LLVM Support)"]),
    (r"libLLVMSupport\.a", r"MD5\.cpp\.o", ["MD5 (LLVM Support)"]),
    (r"libnanobind[\w-]*\.a", r".*", ["nanobind", "robin-map (in nanobind)"]),
]
# LLVM and MLIR sources under third-party terms no component covers yet.
_UNCOVERED_MEMBERS = {
    r"UnicodeNameToCodepointGenerated\.cpp\.o": "the Unicode data license",
    r"MPIToLLVM\.cpp\.o": "MPICH's license",
}
# This project's archives, and the toolchain's runtime archives (the GCC
# Runtime Library Exception and Apache-2.0 WITH LLVM-exception put no notice
# requirement on the binaries they link into).
_NO_NOTICE_ARCHIVES = re.compile(
    r"libCudaqPulse\w*\.a|lib(gcc|gcc_eh|atomic|c_nonshared|stdc\+\+_nonshared|stdc\+\+)\.a"
    r"|libclang_rt[\w.-]*\.a"
)
_ARCHIVE_MEMBER = re.compile(r"([^\s\[\]()]+\.a)(?:\[\d+\])?\(([^()\s]+)\)")

GRAFTED_DIR = re.compile(r"^[^/]+\.libs/|/\.dylibs/")
_GRAFTED = [
    (r"libz[.-]", "zlib"),
    (r"libzstd[.-]", "Zstandard"),
    (r"libxml2[.-]", "libxml2"),
    (r"libffi[.-]", "libffi"),
    (r"lib(ncurses|ncursesw|tinfo|tinfow)[.-]", "ncurses"),
]
# GPL and LGPL libraries a repair could graft; none may ship.
GPL_PROBLEM = "a GPL or LGPL library"
GPL_LIBRARIES = re.compile(
    r"(^|/)lib(readline|history|gmp|mpfr|mpc|gnutls|gcrypt|gpg-error|iconv|intl|"
    r"quadmath|gfortran|gomp|gcc_s|stdc\+\+|unistring|idn2|tasn1|nettle|hogweed|"
    r"asound|pulse|sndfile|gdbm|elf|dw)[.-][^/]*$"
)


def _read_map(path: Path) -> str:
    text = path.read_text(errors="replace")
    # ld64 lists every input before "# Sections:"; the rest names inputs by index.
    head, sep, _ = text.partition("\n# Sections:")
    return head if sep else text


def linked_components(maps: list[Path]) -> tuple[set[str], list[str]]:
    """The components the maps' archive members bring in, and the problems."""
    found: set[str] = set()
    problems: list[str] = []
    for path in maps:
        if not path.is_file():
            problems.append(f"{path}: no linker map; relink the binary")
            continue
        members = {
            (Path(archive).name, member)
            for archive, member in _ARCHIVE_MEMBER.findall(_read_map(path))
        }
        for archive, member in sorted(members):
            names = [
                name
                for archive_re, member_re, components in _ARCHIVE_COMPONENTS
                if re.fullmatch(archive_re, archive) and re.fullmatch(member_re, member)
                for name in components
            ]
            for member_re, terms in _UNCOVERED_MEMBERS.items():
                if re.fullmatch(member_re, member):
                    problems.append(
                        f"{path.name}: links {archive}({member}), under {terms}; "
                        "add a component for it to tools/third-party-notices.py"
                    )
            if not names and not _NO_NOTICE_ARCHIVES.fullmatch(archive):
                problems.append(
                    f"{path.name}: links {archive}({member}), which "
                    "tools/third-party-notices.py does not know"
                )
            found.update(names)
    return found, problems


def grafted_components(names: list[str]) -> tuple[set[str], list[str]]:
    """The components of the libraries grafted among a wheel's file names."""
    found: set[str] = set()
    problems: list[str] = []
    for item in names:
        if not GRAFTED_DIR.search(item) or item.endswith("/"):
            continue
        base = item.rsplit("/", 1)[-1]
        if GPL_LIBRARIES.search(item):
            problems.append(f"grafts {item}, {GPL_PROBLEM}")
            continue
        matches = [name for pattern, name in _GRAFTED if re.match(pattern, base)]
        if not matches:
            problems.append(f"grafts {item}, which tools/third-party-notices.py does not know")
        found.update(matches)
    return found, problems


def wheel_components(wheels: list[Path]) -> tuple[set[str], list[str]]:
    found: set[str] = set()
    problems: list[str] = []
    for path in wheels:
        with zipfile.ZipFile(path) as wheel:
            components, wheel_problems = grafted_components(wheel.namelist())
        found |= components
        problems += [f"{path.name}: {problem}" for problem in wheel_problems]
    return found, problems


def render(names: set[str]) -> str:
    """The THIRD_PARTY_NOTICES text for the named components."""
    components = [component for component in COMPONENTS if component.name in names]
    width = max(len(component.name) for component in components)
    lines = [
        "Third-party software notices",
        "",
        "The binaries of this distribution statically link, or ship alongside",
        "them, the third-party software below. tools/third-party-notices.py",
        "generates this file from the binaries' linker maps and the libraries",
        "grafted into the wheels; do not edit it by hand.",
        "",
        *(f"  {c.name:<{width}}  {c.spdx}" for c in components),
    ]
    for component in components:
        lines += [
            "",
            "=" * 78,
            f"Component: {component.name}",
            f"Origin: {component.origin}",
            f"License: {component.spdx}",
        ]
        if component.copyright:
            lines += [component.copyright]
        lines += ["-" * 78, "", (_ROOT / component.license_file).read_text().rstrip("\n")]
    return "\n".join(lines) + "\n"


def listed_components(text: str) -> set[str]:
    return set(re.findall(r"^Component: (.+)$", text, re.MULTILINE))


def missing_license_files(names: set[str]) -> list[str]:
    return [
        f"{_BY_NAME[name].license_file} is missing; add {name}'s license text from its source"
        for name in sorted(names)
        if not (_ROOT / _BY_NAME[name].license_file).is_file()
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["write", "check"])
    parser.add_argument("--notices", type=Path, required=True)
    parser.add_argument("--map", type=Path, action="append", default=[])
    parser.add_argument("--wheel", type=Path, action="append", default=[])
    args = parser.parse_args(argv)

    linked, problems = linked_components(args.map)
    grafted, graft_problems = wheel_components(args.wheel)
    found = linked | grafted
    problems += graft_problems
    if args.command == "write":
        problems += missing_license_files(found)
        if not problems:
            args.notices.write_text(render(found))
            print(f"wrote {args.notices}: {', '.join(sorted(found))}")
    else:
        text = args.notices.read_text() if args.notices.is_file() else ""
        listed = listed_components(text)
        unknown = sorted(listed - _BY_NAME.keys())
        problems += [
            f"{args.notices} lists {name}, which this tool does not know" for name in unknown
        ]
        problems += [
            f"{args.notices} does not list {name}, which the binaries link or the wheels graft"
            for name in sorted(found - listed)
        ]
        known = listed & _BY_NAME.keys()
        problems += missing_license_files(known)
        if not unknown and known and not missing_license_files(known) and text != render(known):
            problems.append(f"{args.notices} is not what this tool generates for what it lists")
        for name in sorted(listed - found):
            print(f"note: {args.notices.name} lists {name}, which nothing given here links")
        if problems:
            problems.append(
                "regenerate it with: python tools/third-party-notices.py write "
                f"--notices {args.notices} --map <build dir>/link-maps/<target>.map ..."
            )
    for problem in problems:
        print(problem, file=sys.stderr)
    if not problems and args.command == "check":
        print(f"{args.notices.name} covers what the binaries link")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
