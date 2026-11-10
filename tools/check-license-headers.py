#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail if any given file lacks the NVIDIA SPDX license header near its top.

The header is two consecutive lines with the same comment prefix (``#``,
``//``, `` *`` or an indented reStructuredText comment):

    SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    SPDX-License-Identifier: Apache-2.0

The former prose header (the entity in mixed case, the license named in a
sentence) is refused. Used as a pre-commit hook; pass the files to check.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HEAD_LINES = 12
HEADER = re.compile(
    r"^(?P<prefix>(?:#|//| \*|  ) )SPDX-FileCopyrightText: Copyright \(c\) 20\d\d(?:-20\d\d)? "
    r"NVIDIA CORPORATION & AFFILIATES\. All rights reserved\.\n"
    r"(?P=prefix)SPDX-License-Identifier: Apache-2\.0\n",
    re.MULTILINE,
)
ENTITY = "NVIDIA CORPORATION & AFFILIATES"
ANY_ENTITY = re.compile(re.escape(ENTITY), re.IGNORECASE)
OLD_LICENSE = "the terms of the Apache License 2.0"


def problem(path: Path) -> str | None:
    with path.open(encoding="utf-8", errors="replace") as handle:
        head = "".join(line for _, line in zip(range(HEAD_LINES), handle, strict=False))
    if OLD_LICENSE in head or any(m[0] != ENTITY for m in ANY_ENTITY.finditer(head)):
        return "has the old prose license header; use the two SPDX lines"
    if HEADER.search(head) is None:
        return "missing the NVIDIA SPDX license header"
    return None


def main(paths: list[str]) -> int:
    failed = 0
    for path in paths:
        message = problem(Path(path))
        if message is not None:
            print(f"{path}: {message}")
            failed = 1
    return failed


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
