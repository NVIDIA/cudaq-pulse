# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

system = p.ControlSystem(links={"gpu": Link(fixed_ns=100, bytes_per_ns=10, feedback_ns=50)})
dev = p.quantum_system("transmon_17q")
low = p.compile(k, control_system=system).lower(dev.qudits[0])
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
        + "p.quantum_system('transmon_generic:4')\n"
        + "p.quantum_system('transmon_17q', clock_ghz=1.0)\n"
        + "low.mlir(); low.schedule; low.segments; low.classical_report; str(low)\n"
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


def test_sampled_exports_load_no_heavy_module(tmp_path):
    # The export samples envelopes natively; only sample_waveform, which
    # returns an array, loads numpy.
    code = (
        _PROGRAM
        + "rate = 2 * dev.clock_ghz\n"
        + "low.export_controller(sample_rate_ghz=rate)\n"
        + "low.export_controller(sample_rate_ghz=rate, version=2)\n"
    )
    assert _loaded(code, tmp_path) == []
    sample = code + "from cudaq_pulse.controller import sample_waveform\n"
    sample += "sample_waveform({'shape': 'neg', 'of': [{'shape': 'custom_samples',"
    sample += " 'samples': [0.5]}]}, 4)\n"
    assert _loaded(sample, tmp_path, ("numpy",)) == ["numpy"]


def test_a_device_with_a_control_system_loads_no_heavy_module(tmp_path):
    code = (
        "import sys\nimport cudaq_pulse as p\n"
        "c = p.controller('reference.controller', drive_lines=[0, 1], readout_lines=[0, 1])\n"
        "p.quantum_system('transmon_17q', layout=[0, 1],\n"
        "         control_system=p.ControlSystem(controllers=[c]))\n"
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

dev = p.quantum_system("transmon_17q")
low = p.compile(k).lower(dev.qudits[0])
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
    # Importing fractions compiles two regular expressions; only a clock
    # ratio of a control system, or reading a time's exact_ns, needs it.
    code = _PLAIN + "low.schedule; low.export_controller(); low.export_emulation()\n"
    assert _loaded(code, tmp_path, ("fractions", "decimal")) == []


def test_times_load_no_fractions(tmp_path):
    # Times do their exact arithmetic on ints.
    code = (
        _PLAIN + "t = (0.1 * p.us + 3 * p.ns) / 7 - p.ns\n"
        "assert repr(t) == '13.7143 ns' and t < low.duration and t // p.ns == 13\n"
        "assert (t % p.ns) * 7 == 5 * p.ns and abs(-t) == t\n"
        "dev2 = p.quantum_system('transmon_17q', clock_ghz=1.5)\n"
        "low2 = p.compile(k).lower(dev2.qudits[0])\n"
        "assert low2.duration > 0 * p.ns\n"
    )
    assert _loaded(code, tmp_path, ("fractions", "decimal")) == []
    exact = _PLAIN + "from fractions import Fraction\n"
    exact += "assert (0.1 * p.us).exact_ns == Fraction(100)\n"
    assert _loaded(exact, tmp_path, ("fractions",)) == ["fractions"]


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
        "emulation.emulate(low, shots_count=2, seed=1)\nbefore = set(sys.modules)\n"
        "low.export_controller(); low.export_emulation()\n"
        "new = [m for m in set(sys.modules) - before if m.startswith('cudaq_pulse.classical')]\n"
        "assert not new, new\n"
    )
    assert _loaded(code, tmp_path, ()) == []


def test_a_name_with_no_close_match_loads_no_difflib(tmp_path):
    # Only names that share enough letters with the unknown one go to difflib.
    code = (
        "import sys\nimport cudaq_pulse as p\n\n@p.kernel\ndef k(q):\n"
        "    d = get_drive_line(q)\n    drive(d, zzz_not_a_waveform(40))\n\n"
        "try:\n    p.compile(k).lower(p.quantum_system('transmon_17q').qudits[0])\n"
        "except p.CompilationError as e:\n    assert 'did you mean' not in str(e), e\n"
    )
    assert _loaded(code, tmp_path, ("difflib",)) == []
    near = code.replace("zzz_not_a_waveform", "squar").replace("not in", "in")
    assert _loaded(near, tmp_path, ("difflib",)) == ["difflib"]


def test_the_doctor_loads_json_only_to_write_json(tmp_path):
    # importlib.metadata, which reads the package's version and wheel tags,
    # imports json itself on Python 3.13 and 3.14. Loaded first, with json
    # then dropped from sys.modules, any later import of json is the doctor's.
    code = (
        "import io, sys\nimport importlib.metadata\n"
        "for m in [m for m in sys.modules if m == 'json' or m.startswith('json.')]:\n"
        "    del sys.modules[m]\n"
        "import cudaq_pulse.doctor as d\nd.main([], out=io.StringIO())\n"
    )
    assert _loaded(code, tmp_path, ("json",)) == []
    report = code.replace("[]", f"['--system', {str(tmp_path / 'none.json')!r}, '--json']")
    assert _loaded(report, tmp_path, ("json",)) == ["json"]


def test_every_dataclass_has_a_docstring():
    # dataclass writes a missing docstring from inspect.signature, which costs
    # more than making the rest of the class.
    root = Path(pulse.__file__).parent
    missing = []
    for path in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.ClassDef)
                and any("dataclass" in ast.unparse(d) for d in node.decorator_list)
                and ast.get_docstring(node) is None
            ):
                missing.append(f"{path.relative_to(root)}:{node.name}")
    assert missing == []


# Describes the classes made dataclasses when first used, after touching each
# first in a different way: as made when used (argv[1] "used"), or made
# dataclasses where defined ("eager"), which must describe them the same.
_DESCRIBE = r"""
import copy, dataclasses, inspect, pickle, re, sys, types

if sys.argv[1] == "eager":
    import importlib.util

    class _Eager:
        def find_spec(self, name, path=None, target=None):
            if name != "cudaq_pulse._errors":
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            # Problem, defined beside the decorator, is made where defined.
            source = spec.loader.get_source(name)
            lazy = "_made_when_used = _dataclass_when_used"
            assert lazy in source
            source = source.replace(lazy, "_made_when_used = dataclass")

            def exec_module(module):
                exec(compile(source, spec.origin, "exec"), module.__dict__)
                module._dataclass_when_used = lambda *, frozen: dataclasses.dataclass(frozen=frozen)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Eager())

import cudaq_pulse.controller as c
from cudaq_pulse import Problem
from cudaq_pulse.quantum_systems import ResidualZZ

classes = [ResidualZZ, c.FunctionUsage, c.ProcessorUsage, c.ControllerUsage, c.LimitExceeded,
           c.UsageReport, Problem]
made = [type(vars(cls)["__dataclass_fields__"]) is dict for cls in classes]
assert made == [sys.argv[1] == "eager"] * len(classes), made
out = []


def clean(value):
    return re.sub(r" at 0x[0-9a-f]+", "", repr(value))


def outcome(make):
    try:
        return clean(make())
    except Exception as error:
        return f"{type(error).__name__}: {error}"


if sys.argv[1] == "eager":
    pickled = pickle.dumps(c.LimitExceeded("f", "max_ops", 3.0, 2.0, "ops"))
    print(pickled.hex())
else:
    pickled = bytes.fromhex(sys.argv[2])
# The first use of each: unpickling, fields(), a class attribute, a
# signature, a subclass, a call, is_dataclass.
first = pickle.loads(pickled)
out.append(outcome(lambda: (repr(first), hash(first), first == first, pickle.dumps(first))))
out.append(outcome(lambda: dataclasses.fields(ResidualZZ)))
out.append(outcome(lambda: c.LimitExceeded.over))
out.append(outcome(lambda: inspect.signature(c.FunctionUsage)))


@dataclasses.dataclass(frozen=True)
class Sub(c.ProcessorUsage):
    "A subclass."

    extra: int = 0


out.append(outcome(lambda: c.ControllerUsage(0, "m", ())))
out.append(outcome(lambda: dataclasses.is_dataclass(c.UsageReport)))


class Plain(ResidualZZ):
    pass


def describe(cls):
    rows = [cls.__qualname__, [k.__name__ for k in cls.__mro__], clean(cls.__dataclass_params__)]
    for name, value in sorted(vars(cls).items()):
        if isinstance(value, types.FunctionType):
            code = value.__code__
            rows.append((name, value.__qualname__, value.__module__, code.co_code,
                         code.co_names, code.co_varnames, clean(value.__defaults__),
                         clean(value.__kwdefaults__), clean(value.__annotations__),
                         value.__doc__, hasattr(value, "__wrapped__")))
        else:
            rows.append((name, clean(value)))
    rows.append([clean(f) for f in dataclasses.fields(cls)])
    rows.append(str(inspect.signature(cls)))
    return rows


for cls in classes + [Sub, Plain]:
    out.append(describe(cls))
    obj = cls(*range(len(dataclasses.fields(cls))))
    out.append([outcome(lambda: repr(obj)), outcome(lambda: hash(obj)),
                outcome(lambda: obj == copy.copy(obj)),
                outcome(lambda: obj == dataclasses.replace(obj)),
                outcome(lambda: dataclasses.asdict(obj)), outcome(lambda: dataclasses.astuple(obj)),
                outcome(lambda: setattr(obj, "x", 1)), outcome(lambda: delattr(obj, "x")),
                outcome(lambda: obj.__match_args__), outcome(lambda: copy.deepcopy(obj) == obj)]
               + [outcome(lambda: pickle.dumps(obj, protocol)) for protocol in (2, 4, 5)]
               + [outcome(lambda: pickle.loads(pickle.dumps(obj)) == obj)])
print(repr(out))
"""


def test_a_dataclass_made_when_used_is_the_dataclass_made_where_defined():
    env = dict(os.environ, PYTHONPATH=_ROOT, PYTHONHASHSEED="0")

    def run(*args):
        result = subprocess.run(
            [sys.executable, "-c", _DESCRIBE, *args],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip().splitlines()

    pickled, eager = run("eager")
    (used,) = run("used", pickled)
    assert used == eager


def test_a_dataclass_made_when_used_has_every_name_a_dataclass_sets():
    from dataclasses import dataclass, field

    from cudaq_pulse._errors import _DATACLASS_NAMES

    class Row:
        """A row."""

        a: int
        b: list[int] = field(default_factory=list)

    before = dict(vars(Row))
    dataclass(frozen=True)(Row)
    changed = {k for k, v in vars(Row).items() if k not in before or before[k] is not v}
    # Python 3.14 and later cache the annotations they evaluate on the class,
    # as here, where this module defers none (the package's modules all do).
    changed.discard("__annotations_cache__")
    assert changed == set(_DATACLASS_NAMES)
    assert "b" in before and "b" not in vars(Row)


def test_classes_most_programs_never_use_are_made_when_used():
    usage = [
        "ControllerUsage",
        "FunctionUsage",
        "LimitExceeded",
        "ProcessorUsage",
        "UsageReport",
    ]
    unmade = sorted(["ResidualZZ", "Problem", *usage])
    code = (
        _PLAIN + "import cudaq_pulse.controller as c\n"
        "low.schedule; low.export_controller(); low.export_emulation(); c.resources(low)\n"
        f"names = [(n, getattr(c, n)) for n in {usage!r}]\n"
        "names += [('ResidualZZ', p.quantum_systems.ResidualZZ), ('Problem', p.Problem)]\n"
        "def unmade():\n"
        "    return sorted(n for n, cls in names\n"
        "                  if type(vars(cls)['__dataclass_fields__']) is not dict)\n"
        f"assert unmade() == {unmade!r}, unmade()\n"
        "low.controller_usage\n"
        f"assert unmade() == {[n for n in unmade if n != 'UsageReport']!r}, unmade()\n"
    )
    env = dict(os.environ, PYTHONPATH=_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


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
    shots_count = _PLAIN + "p.simulate(low, shots_count=10, seed=1)\n"
    assert _loaded(shots_count, tmp_path, modules) != []


def test_simulating_loads_no_numpy_typing(tmp_path):
    # numpy.typing compiles regular expressions on import; the simulator
    # names its types for checkers only.
    assert _loaded(_PLAIN + "p.simulate(low)\n", tmp_path, ("numpy", "numpy.typing")) == ["numpy"]
