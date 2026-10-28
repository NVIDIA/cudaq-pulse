# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""``python -m nvqconsole [--port N] [--build] [--no-warm]``"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

from .server import WEB_DIST, serve


def main() -> None:
    parser = argparse.ArgumentParser(prog="nvqconsole", description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--build", action="store_true", help="build the web app first (needs npm)")
    parser.add_argument(
        "--no-warm", action="store_true", help="do not precompute the calibration chain"
    )
    args = parser.parse_args()
    if args.build or not WEB_DIST.exists():
        npm = shutil.which("npm")
        if npm is None:
            sys.exit("the web app is not built and npm is not on PATH")
        web = WEB_DIST.parent
        if not (web / "node_modules").exists():
            subprocess.run([npm, "install", "--no-audit", "--no-fund"], cwd=web, check=True)
        subprocess.run([npm, "run", "build"], cwd=web, check=True)
    serve(args.port, warm=not args.no_warm)


if __name__ == "__main__":
    main()
