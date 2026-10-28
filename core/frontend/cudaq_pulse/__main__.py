# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Run a script on a target: ``python -m cudaq_pulse --target NAME script.py``.

Sets ``CUDAQ_PULSE_TARGET`` for the script and runs it as ``__main__``. The
script's own arguments follow its path and are passed through unparsed.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys

from .targets._registry import ENV_VAR, available_targets, make_target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cudaq_pulse",
        description="Run a Python script with a cudaq-pulse process target.",
    )
    parser.add_argument(
        "--target", required=True, help="registered target name, e.g. NAME or NAME:ARG"
    )
    parser.add_argument("--quiet", action="store_true", help="do not print the target banner")
    parser.add_argument("-m", dest="module", help="run a module instead of a script")
    parser.add_argument("script", nargs="?", help="path of the script")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="arguments of the script")
    ns = parser.parse_args(argv)
    if (ns.module is None) == (ns.script is None):
        parser.error("give either a script path or -m MODULE")
    try:
        make_target(ns.target)
    except LookupError as e:
        parser.error(Exception.__str__(e))
    except ValueError as e:
        parser.error(f"{e}\navailable targets: {', '.join(available_targets())}")
    os.environ[ENV_VAR] = ns.target
    if not ns.quiet:
        print(f"cudaq-pulse: target {ns.target} ({ENV_VAR})", file=sys.stderr)
    if ns.module is not None:
        args = [ns.script] if ns.script is not None else []
        sys.argv = [ns.module, *args, *ns.args]
        runpy.run_module(ns.module, run_name="__main__", alter_sys=True)
    else:
        sys.argv = [ns.script, *ns.args]
        runpy.run_path(ns.script, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
