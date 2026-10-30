# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The first use of each step loads only what the step needs.

Each check runs a fresh interpreter and lists the heavy modules a path left
in ``sys.modules``. ``test_import.py`` checks the import and compile of a
program; these add devices with their calibration, a program's queries and
exports, classical calls and the modules behind them.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

import cudaq_pulse as pulse
import pytest

_ROOT = str(Path(pulse.__file__).parent.parent)

# Modules of a millisecond or more that compiling and exporting do not use.
_HEAVY = (
    "numpy",
    "pathlib",
    "urllib.parse",
    "ipaddress",
    "ctypes",
    "socket",
    "importlib.metadata",
    "cudaq_pulse.runtime.gpu",
)

_PROGRAM = """
import sys
import cudaq_pulse as p
from cudaq_pulse.classical import Bits, Link
from cudaq_pulse.compiler import MachineModel

@p.classical(args=(Bits(1),), results=(Bits(1),), min_latency_ns=200,
             nominal_latency_ns=300, max_latency_ns=400, link="gpu")
def decode(s):
    return [s[0]]

@p.kernel
def k(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, drag(40, 0.2, 10.0, 0.5))
    sync(d, r)
    m = readout(r, square(500, 0.05))
    ok, c = decode.submit([m]).result(timeout=350 * p.ns)
    if ok:
        if c[0]:
            drive(d, square(20, 0.1))

machine = MachineModel(links={"gpu": Link(fixed_ns=100, bytes_per_ns=10, feedback_ns=50)})
dev = p.device("transmon_17q")
low = p.compile(k, machine=machine).lower(dev.qudit_handles[0])
"""


def _loaded(code: str, tmp_path: Path, modules: tuple[str, ...] = _HEAVY) -> list[str]:
    script = tmp_path / "probe.py"
    script.write_text(f"{code}\nprint(sorted(m for m in {modules!r} if m in sys.modules))\n")
    env = dict(os.environ, PYTHONPATH=_ROOT)
    result = subprocess.run(
        [sys.executable, str(script)], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return ast.literal_eval(result.stdout.strip().splitlines()[-1])


def test_devices_and_a_classical_program_load_no_heavy_module(tmp_path):
    code = (
        _PROGRAM
        + "p.device('transmon_generic:4')\n"
        + "p.device({'target': 'transmon_17q', 'clock_ghz': 1.0})\n"
        + "low.mlir(); low.schedule; low.epochs; low.classical_report; str(low)\n"
        + "low.generic.bind([])\n"
    )
    assert _loaded(code, tmp_path) == []


@pytest.mark.xfail(
    strict=True,
    reason="classical.emulation imports numpy and socket at module level and every export "
    "imports it; the fix waits for the multi-controller-links branch, which rewrites "
    "the export (tech-debt inventory, section 4)",
)
def test_exports_load_no_heavy_module(tmp_path):
    code = _PROGRAM + "low.export_controller(); low.export_emulation()\n"
    assert _loaded(code, tmp_path) == []


def test_the_classical_call_helpers_load_neither_numpy_nor_ctypes(tmp_path):
    code = "import sys\nimport cudaq_pulse.runtime._calls\nimport cudaq_pulse.classical.wire\n"
    assert _loaded(code, tmp_path, ("numpy", "ctypes")) == []


def test_simulating_a_classical_program_loads_numpy(tmp_path):
    # The positive control of the checks above: the same probe sees numpy.
    code = _PROGRAM + "p.simulate(low)\n"
    assert _loaded(code, tmp_path, ("numpy",)) == ["numpy"]
