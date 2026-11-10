#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report the native libraries' exported symbols, sizes and import time.

Usage:
    python tools/check-native-exports.py PACKAGE_ROOT [--max-exports N]

PACKAGE_ROOT contains the ``cudaq_pulse`` package (``build/dev/python`` or
``site-packages``). For the Python extension and every runtime library under
``cudaq_pulse/lib`` it prints the file size and the number of exported
(dynamic, defined) symbols, then the median time of ``import cudaq_pulse`` in
fresh interpreters. With ``--max-exports``, exits 1 if the extension exports
more symbols than that.
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
from pathlib import Path

_IMPORT = "import time; t = time.perf_counter(); import {module}; print(time.perf_counter() - t)"


def exported_symbols(library: Path) -> list[str]:
    if sys.platform == "darwin":
        command = ["nm", "-gU", str(library)]
    else:
        command = ["nm", "-D", "--defined-only", str(library)]
    output = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    return [line.split()[-1] for line in output.splitlines() if line.strip()]


def import_ms(root: Path, module: str, samples: int) -> float:
    env = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1")
    times = []
    for _ in range(samples):
        result = subprocess.run(
            [sys.executable, "-c", _IMPORT.format(module=module)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        times.append(float(result.stdout.strip().splitlines()[-1]) * 1000)
    return statistics.median(times)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--max-exports", type=int)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    package = args.root / "cudaq_pulse"
    extensions = sorted((package / "_native").glob("_cudaq_pulse_native*"))
    extensions = [p for p in extensions if p.suffix in (".so", ".pyd")]
    if len(extensions) != 1:
        print(f"expected one native extension in {package / '_native'}, found {extensions}")
        return 1
    extension = extensions[0]
    runtimes = sorted(p for p in (package / "lib").glob("*") if p.suffix in (".so", ".dylib"))

    failed = False
    print(f"{'library':48s} {'size (MB)':>10s} {'exports':>8s}")
    for library in [extension, *runtimes]:
        symbols = exported_symbols(library)
        print(f"{library.name:48s} {library.stat().st_size / 2**20:10.1f} {len(symbols):8d}")
        if library == extension:
            if len(symbols) <= 20:
                print("  " + ", ".join(symbols))
            if args.max_exports is not None and len(symbols) > args.max_exports:
                print(f"  FAIL: {len(symbols)} exports, more than {args.max_exports}")
                failed = True

    for module in ("cudaq_pulse._native._cudaq_pulse_native", "cudaq_pulse"):
        print(f"import {module}: {import_ms(args.root, module, args.samples):.1f} ms (median)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
