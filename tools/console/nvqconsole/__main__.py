# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""``python -m nvqconsole [--port N] [--build] [--no-warm] [--no-open] [--runs FILE]``

``--precompute FILE`` runs everything once and saves it for ``--runs``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

from . import adapter
from .server import WEB_DIST, serve, warm_all


def main() -> None:
    parser = argparse.ArgumentParser(prog="nvqconsole", description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--build", action="store_true", help="build the web app first (needs npm)")
    parser.add_argument(
        "--no-warm", action="store_true", help="do not precompute the calibration chain"
    )
    parser.add_argument("--no-open", action="store_true", help="do not open a browser")
    parser.add_argument(
        "--runs",
        type=Path,
        default=WEB_DIST.parent.parent / "runs.json",
        help="precomputed runs to start from (default: runs.json beside the console)",
    )
    parser.add_argument(
        "--precompute", type=Path, metavar="FILE", help="run everything, save it to FILE and exit"
    )
    args = parser.parse_args()
    if args.precompute is not None:
        warm_all()
        adapter.save_runs(args.precompute)
        print(f"saved {args.precompute}", flush=True)
        return
    if args.build or not WEB_DIST.exists():
        npm = shutil.which("npm")
        if npm is None:
            sys.exit("the web app is not built and npm is not on PATH")
        web = WEB_DIST.parent
        if not (web / "node_modules").exists():
            subprocess.run([npm, "install", "--no-audit", "--no-fund"], cwd=web, check=True)
        subprocess.run([npm, "run", "build"], cwd=web, check=True)
    # Codespaces and CI forward or probe the port themselves.
    if not args.no_open and not os.environ.get("CODESPACES") and not os.environ.get("CI"):
        url = f"http://127.0.0.1:{args.port}"
        threading.Timer(1.0, webbrowser.open, (url,)).start()
    serve(args.port, warm=not args.no_warm, runs=args.runs)


if __name__ == "__main__":
    main()
