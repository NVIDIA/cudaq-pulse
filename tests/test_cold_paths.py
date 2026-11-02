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

_ROOT = str(Path(pulse.__file__).parent.parent)

# Modules of half a millisecond or more that compiling and exporting do not use.
_HEAVY = (
    "numpy",
    "pathlib",
    "urllib.parse",
    "ipaddress",
    "ctypes",
    "socket",
    "importlib.metadata",
    "cudaq_pulse.runtime.gpu",
    "hashlib",
    "json",
    "difflib",
    "textwrap",
)

_PROGRAM = """
import sys
import cudaq_pulse as p
from cudaq_pulse.classical import Link
from cudaq_pulse.compiler import MachineModel

@p.classical(min_latency_ns=200, nominal_latency_ns=300, max_latency_ns=400, link="gpu")
def decode(s: p.Bits) -> p.Bits:
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


def test_exports_load_no_heavy_module(tmp_path):
    code = (
        _PROGRAM
        + "from cudaq_pulse.controller import evaluate_expression, unroll\n"
        + "low.export_controller(); low.export_emulation()\n"
        + "unroll(low.export_controller(version=2))\n"
        + "evaluate_expression({'op': 'ne', 'type': 'int', 'of': [{'readout': 0}, {'const': 0}]},"
        + " readouts={0: 1})\n"
    )
    assert _loaded(code, tmp_path) == []


def test_a_device_with_a_control_system_loads_no_heavy_module(tmp_path):
    code = (
        "import sys\nimport cudaq_pulse as p\n"
        "c = p.controller('reference.controller', drive_lines=[0, 1], readout_lines=[0, 1])\n"
        "p.device({'quantum_system': 'transmon_17q', 'layout': [0, 1],\n"
        "          'control_system': p.ControlSystem(controllers=[c])})\n"
    )
    assert _loaded(code, tmp_path) == []


def test_the_classical_call_helpers_load_neither_numpy_nor_ctypes(tmp_path):
    code = "import sys\nimport cudaq_pulse.runtime._calls\nimport cudaq_pulse.classical.wire\n"
    assert _loaded(code, tmp_path, ("numpy", "ctypes")) == []


def test_simulating_a_classical_program_loads_numpy(tmp_path):
    # The positive control of the checks above: the same probe sees numpy.
    code = _PROGRAM + "p.simulate(low)\n"
    assert _loaded(code, tmp_path, ("numpy",)) == ["numpy"]


_PLAIN = """
import sys
import cudaq_pulse as p

@p.kernel
def k(q):
    d = get_drive_line(q)
    drive(d, drag(40, 0.2, 10.0, 0.5))

dev = p.device("transmon_17q")
low = p.compile(k).lower(dev.qudit_handles[0])
"""

# What exporting a program does not build: the reports and their analyzer,
# the emulator and the call records.
_REPORTS = (
    "cudaq_pulse._controller_cost",
    "cudaq_pulse._classical_report",
    "cudaq_pulse.classical.emulation",
    "cudaq_pulse.classical._link",
    "cudaq_pulse.runtime._calls",
)


def test_a_program_in_cycles_loads_no_fractions(tmp_path):
    # Importing fractions compiles two regular expressions; only a time
    # (such as a program's duration), or a clock in GHz, needs exact arithmetic.
    code = _PLAIN + "low.schedule; low.export_controller(); low.export_emulation()\n"
    assert _loaded(code, tmp_path, ("fractions", "decimal")) == []
    timed = _PLAIN + "low.duration\n"
    assert _loaded(timed, tmp_path, ("fractions",)) == ["fractions"]


def test_exports_build_no_report(tmp_path):
    code = _PLAIN + "low.export_controller(); low.export_emulation()\n"
    assert _loaded(code, tmp_path, _REPORTS) == []
    calls = _PROGRAM + "low.export_controller(); low.export_emulation()\n"
    assert _loaded(calls, tmp_path, _REPORTS) == [
        "cudaq_pulse._classical_report",
        "cudaq_pulse.classical._link",
    ]
    report = _PLAIN + "import cudaq_pulse.controller as c\nc.resources(low)\n"
    assert _loaded(report, tmp_path, _REPORTS) == ["cudaq_pulse._controller_cost"]


def test_exporting_an_emulated_program_loads_no_module(tmp_path):
    # The export lives in the wire format, which emulating loads too.
    code = (
        _PROGRAM + "from cudaq_pulse.classical import emulation\n"
        "emulation.emulate(low, shots=2, seed=1)\nbefore = set(sys.modules)\n"
        "low.export_controller(); low.export_emulation()\n"
        "new = [m for m in set(sys.modules) - before if m.startswith('cudaq_pulse.classical')]\n"
        "assert not new, new\n"
    )
    assert _loaded(code, tmp_path, ()) == []


def test_copying_a_time_keeps_it():
    # Kernel globals are snapshotted by deepcopy on each definition; a time,
    # being immutable, is its own copy.
    import copy

    t = 40 * pulse.ns
    assert copy.copy(t) is t and copy.deepcopy({"t": t})["t"] is t


def test_the_report_names_resolve_on_first_use():
    code = (
        "import pickle, sys\n"
        "import cudaq_pulse.controller as c\n"
        "from cudaq_pulse.classical import emulation\n"
        "assert 'cudaq_pulse._controller_cost' not in sys.modules\n"
        "assert set(c.__all__) <= set(dir(c)) and set(emulation.__all__) <= set(dir(emulation))\n"
        "from cudaq_pulse.controller import ResourceReport, usage\n"
        "assert c.ResourceReport is ResourceReport and 'usage' in vars(c)\n"
        "assert ResourceReport.__module__ == 'cudaq_pulse.controller'\n"
        "assert pickle.loads(pickle.dumps(ResourceReport)) is ResourceReport\n"
        "from cudaq_pulse.classical import wire\n"
        "assert emulation.export is wire._emulation_export\n"
        "assert emulation.FORMAT == wire._EMULATION_FORMAT\n"
    )
    env = dict(os.environ, PYTHONPATH=_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_simulating_without_shots_loads_no_random_generator(tmp_path):
    # numpy.random takes milliseconds to import, with the hashing it loads;
    # only drawing shots (or a program's calls) uses a generator.
    modules = ("numpy.random", "hashlib")
    assert _loaded(_PLAIN + "p.simulate(low)\n", tmp_path, modules) == []
    shots = _PLAIN + "p.simulate(low, shots=10, seed=1)\n"
    assert _loaded(shots, tmp_path, modules) != []
