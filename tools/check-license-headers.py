#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Fail if any given file lacks the NVIDIA Apache-2.0 header near its top.

The header's comment style varies by language, so only its two load-bearing
lines are matched. Used as a pre-commit hook; pass the files to check.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HEAD_LINES = 12
COPYRIGHT = re.compile(r"Copyright \(c\) 20\d\d( - 20\d\d)? NVIDIA Corporation & Affiliates\.")
LICENSE = "the terms of the Apache License 2.0"


def has_header(path: Path) -> bool:
    with path.open(encoding="utf-8", errors="replace") as handle:
        head = "".join(line for _, line in zip(range(HEAD_LINES), handle, strict=False))
    return bool(COPYRIGHT.search(head)) and LICENSE in head


def main(paths: list[str]) -> int:
    missing = [p for p in paths if not has_header(Path(p))]
    for path in missing:
        print(f"{path}: missing the NVIDIA Apache-2.0 license header")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
