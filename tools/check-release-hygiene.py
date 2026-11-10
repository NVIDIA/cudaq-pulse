#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail if a tracked file holds something that must not be published.

Looks for absolute home-directory paths, ``file://`` URLs, NVIDIA hostnames
other than the public ones below, private network addresses, Slack links,
private-key headers and common access-token prefixes. Binary files are
skipped.

Usage: python tools/check-release-hygiene.py [FILES...]   (default: git ls-files)
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# Public NVIDIA addresses the community files link to.
_PUBLIC_HOSTS = {"www.nvidia.com", "docs.nvidia.com", "developer.nvidia.com", "pypi.nvidia.com"}
_PUBLIC_MAILBOXES = {"psirt@nvidia.com", "github_conduct@nvidia.com"}

_ADDRESS = re.compile(
    r"(?<![\w.-])(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}(?![\w.-])"
)
_VERSION = re.compile(r"\s*version\s*=")
_HOST = re.compile(r"[\w.@-]*\bnvidia\.(?:com|net)\b", re.IGNORECASE)
_PATTERNS = (
    ("an absolute home-directory path", re.compile(r"(?<![\w.])/(?:Users|home)/[A-Za-z][\w.-]*/")),
    ("a file:// URL", re.compile(r"\bfile://")),
    ("a private network address", _ADDRESS),
    ("a Slack link", re.compile(r"\b[\w-]+\.slack\.com\b", re.IGNORECASE)),
    ("a private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    (
        "an access token",
        re.compile(
            r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_\w{20,}|xox[abprs]-[\w-]{10,}"
            r"|AKIA[0-9A-Z]{16}|nvapi-[\w-]{20,})"
        ),
    ),
)

_ANY = re.compile(
    "|".join([_HOST.pattern] + [pattern.pattern for _, pattern in _PATTERNS]), re.IGNORECASE
)


def _host_allowed(match: str) -> bool:
    match = match.lower()
    if "@" in match:
        return match in _PUBLIC_MAILBOXES
    return match in _PUBLIC_HOSTS


def check(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, IsADirectoryError, FileNotFoundError):
        return []
    if not _ANY.search(text):
        return []
    errors = []
    for number, line in enumerate(text.splitlines(), start=1):
        for what, pattern in _PATTERNS:
            # Four-part package versions (uv.lock) look like addresses.
            if pattern.search(line) and not (pattern is _ADDRESS and _VERSION.match(line)):
                errors.append(f"{path}:{number}: {what}")
        for host in _HOST.findall(line):
            if not _host_allowed(host):
                errors.append(f"{path}:{number}: an NVIDIA hostname that is not public ({host})")
    return errors


def main() -> int:
    if sys.argv[1:]:
        paths = [Path(p) for p in sys.argv[1:]]
    else:
        listed = subprocess.run(
            ["git", "ls-files", "-z"], capture_output=True, check=True, text=True
        )
        paths = [Path(p) for p in listed.stdout.split("\0") if p]
    me = Path(__file__).resolve()
    errors = [error for path in paths if path.resolve() != me for error in check(path)]
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
