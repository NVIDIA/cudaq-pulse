# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a script on a quantum system: ``python -m cudaq_pulse --quantum-system NAME script.py``.

Sets ``CUDAQ_PULSE_QUANTUM_SYSTEM`` for the script and runs it as ``__main__``. The
script's own arguments follow its path and are passed through unparsed.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys

from .quantum_systems._registry import ENV_VAR, available_quantum_systems, make_quantum_system


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cudaq_pulse",
        description="Run a Python script with a cudaq-pulse process quantum system.",
    )
    parser.add_argument(
        "--quantum-system",
        required=True,
        help="registered quantum system name, e.g. NAME or NAME:ARG",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="do not print the quantum system banner"
    )
    parser.add_argument("-m", dest="module", help="run a module instead of a script")
    parser.add_argument("script", nargs="?", help="path of the script")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="arguments of the script")
    ns = parser.parse_args(argv)
    if (ns.module is None) == (ns.script is None):
        parser.error("give either a script path or -m MODULE")
    try:
        make_quantum_system(ns.quantum_system)
    except LookupError as e:
        parser.error(Exception.__str__(e))
    except ValueError as e:
        parser.error(f"{e}\navailable quantum systems: {', '.join(available_quantum_systems())}")
    os.environ[ENV_VAR] = ns.quantum_system
    if not ns.quiet:
        print(f"cudaq-pulse: quantum system {ns.quantum_system} ({ENV_VAR})", file=sys.stderr)
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
