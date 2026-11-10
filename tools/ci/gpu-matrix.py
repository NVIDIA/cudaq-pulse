#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plan the rows of .github/workflows/gpu-tests.yml.

Each row needs a runner role. The GPU_RUNNERS repository variable maps roles
to runner labels, as a JSON object such as
``{"x86_64": "LABEL", "x86_64-min-driver": "LABEL", "aarch64": "LABEL"}``
(a value may also be a JSON list of labels); a bare label, or a JSON list of
labels, is the ``x86_64`` role alone. A row whose role has no runner is
skipped and listed in the job summary; a plan with no row left fails.

The rows follow what the GPU code varies on: the CUDA major of the extra,
the driver (each major's minimum, which [cu13] must refuse on an older
one), the CPU architecture and the Python version of the abi3 wheel. The
wheels run in images without a CUDA toolkit; the source build runs in the
CUDA 12.9 development image. ``quick`` rows run on every push, ``full``
adds the rest on release tags and manual runs. Free-threaded Python has no
release wheels, so it has no row.

Usage: GPU_RUNNERS=... python3 tools/ci/gpu-matrix.py --source wheels|build
           [--scope quick|full]

Prints ``matrix=<JSON>`` for $GITHUB_OUTPUT, and appends the plan to
$GITHUB_STEP_SUMMARY when that is set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import NamedTuple

UBUNTU = "ubuntu:24.04@sha256:24737cef17f4e22b1977df7d9a05e77c4e28b7da5b8911f8d8ccda136bf3573c"
PYTHON_NEWEST = (
    "python:3.15-slim@sha256:94d3426f1def7db4a106a232ac121bfd975393e5dfd40a619b955fae45fe7913"
)
CUDA_DEVEL = (
    "nvidia/cuda:12.9.1-devel-ubuntu24.04"
    "@sha256:020bc241a628776338f4d4053fed4c38f6f7f3d7eb5919fecb8de313bb8ba47c"
)

# What each image needs before the checkout: the Python and git it lacks.
IMAGES = {
    UBUNTU: ("Python 3.12", "ca-certificates git python3-venv"),
    PYTHON_NEWEST: ("Python 3.15", "git"),
    CUDA_DEVEL: ("CUDA 12.9 toolkit", "ca-certificates git gh build-essential python3"),
}

ROLES = {
    "x86_64": "x86_64, newest driver",
    "x86_64-min-driver": "x86_64, oldest driver",
    "aarch64": "aarch64, newest driver",
}


class Row(NamedTuple):
    role: str
    extra: str
    scope: str
    image: str = UBUNTU
    # On the oldest driver, [cuNN] either passes or, if the driver is older
    # than the extra's minimum, must be refused by the doctor.
    old_driver_ok: bool = False


ROWS = {
    "wheels": (
        Row("x86_64", "cu12", "quick"),
        Row("x86_64", "cu13", "quick"),
        Row("x86_64-min-driver", "cu12", "full", old_driver_ok=True),
        Row("x86_64-min-driver", "cu13", "full", old_driver_ok=True),
        Row("aarch64", "cu12", "full"),
        Row("aarch64", "cu13", "full"),
        Row("x86_64", "cu13", "full", image=PYTHON_NEWEST),
    ),
    "build": (Row("x86_64", "cu12", "quick", image=CUDA_DEVEL),),
}
TIMEOUT_MINUTES = {"wheels": 45, "build": 120}


def parse_runners(value: str) -> dict[str, str | list[str]]:
    """Return the role-to-labels map that GPU_RUNNERS holds."""
    value = value.strip()
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value
    if not isinstance(parsed, dict):
        parsed = {"x86_64": parsed}
    unknown = sorted(set(parsed) - set(ROLES))
    if unknown:
        raise ValueError(f"GPU_RUNNERS: unknown roles {unknown}; the roles are {sorted(ROLES)}")
    for role, labels in parsed.items():
        if isinstance(labels, str):
            ok = bool(labels.strip())
        else:
            ok = isinstance(labels, list) and bool(labels)
            ok = ok and all(isinstance(label, str) and label.strip() for label in labels)
        if not ok:
            raise ValueError(f"GPU_RUNNERS: {role} must be a label or a list of labels")
    return parsed


def plan(source: str, scope: str, runners: dict) -> tuple[list[dict], list[tuple[str, str]]]:
    """Return the matrix rows and the (name, reason) of each row left out."""
    rows, skipped = [], []
    for row in ROWS[source]:
        arch = "aarch64" if row.role == "aarch64" else "x86_64"
        python, packages = IMAGES[row.image]
        name = f"{source} [{row.extra}], {ROLES[row.role]}, {python}"
        if scope == "quick" and row.scope != "quick":
            continue
        if row.role not in runners:
            skipped.append((name, f"no {row.role} runner in GPU_RUNNERS"))
            continue
        rows.append(
            {
                "name": name,
                "runner": runners[row.role],
                "image": row.image,
                "packages": packages,
                "platform": f"linux-{arch}",
                "extra": row.extra,
                "flags": "--allow-old-driver" if row.old_driver_ok else "",
                "timeout": TIMEOUT_MINUTES[source],
            }
        )
    return rows, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--source", choices=sorted(ROWS), required=True)
    parser.add_argument("--scope", choices=("quick", "full"), default="quick")
    args = parser.parse_args(argv)
    try:
        runners = parse_runners(os.environ.get("GPU_RUNNERS", ""))
    except ValueError as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    rows, skipped = plan(args.source, args.scope, runners)

    lines = [f"### GPU tests ({args.source}, {args.scope})", "", "| row | runner |", "|---|---|"]
    lines += [f"| {row['name']} | `{json.dumps(row['runner'])}` |" for row in rows]
    lines += [f"| {name} | skipped: {reason} |" for name, reason in skipped]
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as handle:
            handle.write("\n".join(lines) + "\n")
    print("\n".join(lines), file=sys.stderr)

    if not rows:
        print("::error::GPU_RUNNERS names no runner for any GPU row", file=sys.stderr)
        return 1
    print("matrix=" + json.dumps({"include": rows}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
