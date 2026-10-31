# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The error classes: what each is caught as, how it prints, and that it
survives pickling."""

import pickle
import warnings

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.backends import BackendNotFoundError
from cudaq_pulse.kernel._bytecode_normalize import UnsupportedInterpreterError
from cudaq_pulse.kernel.ir_builder import SymbolicValueError
from cudaq_pulse.runtime import GpuRuntimeError
from cudaq_pulse.targets import UnknownTargetError

CAUGHT_AS = [
    (pulse.InvalidValueError, ValueError),
    (pulse.InvalidTypeError, TypeError),
    (pulse.NotFoundError, KeyError),
    (pulse.NotFoundError, LookupError),
    (pulse.IndexOutOfRangeError, IndexError),
    (pulse.IndexOutOfRangeError, LookupError),
    (pulse.UnsupportedError, NotImplementedError),
    (pulse.UnsupportedError, RuntimeError),
    (pulse.CompilationError, RuntimeError),
    (pulse.TimingError, pulse.CompilationError),
    (pulse.TimingError, RuntimeError),
    (pulse.ExecutionError, RuntimeError),
    (pulse.InternalError, RuntimeError),
    (pulse.DeviceDefinitionError, ValueError),
    (UnknownTargetError, pulse.NotFoundError),
    (UnknownTargetError, LookupError),
    (BackendNotFoundError, pulse.NotFoundError),
    (UnsupportedInterpreterError, pulse.UnsupportedError),
    (UnsupportedInterpreterError, RuntimeError),
    (GpuRuntimeError, pulse.ExecutionError),
    (GpuRuntimeError, RuntimeError),
]


@pytest.mark.parametrize(("cls", "base"), CAUGHT_AS, ids=lambda c: c.__name__)
def test_each_error_is_caught_as_its_builtin_kind(cls, base):
    with pytest.raises(base):
        raise cls("message")
    assert issubclass(cls, pulse.PulseError)


def test_the_value_and_type_errors_are_separate():
    assert not issubclass(pulse.InvalidValueError, TypeError)
    assert not issubclass(pulse.InvalidTypeError, ValueError)


def test_a_not_found_error_prints_without_quotes():
    error = pulse.NotFoundError("no calibration value 'q0.freq'")
    assert str(error) == "no calibration value 'q0.freq'"
    assert str(UnknownTargetError("no registered target 'x'")) == "no registered target 'x'"


def test_problems_are_listed_after_the_message():
    error = pulse.InvalidValueError(
        "2 problems in the options",
        problems=[pulse.Problem("must be positive", path="shots", value=0), pulse.Problem("x")],
    )
    assert str(error) == "2 problems in the options:\n  shots: must be positive\n  x"
    assert error.problems[0].value == 0


def test_only_the_first_ten_problems_are_shown():
    problems = [pulse.Problem(f"problem {i}") for i in range(13)]
    error = pulse.CompilationError("13 problems", problems=problems)
    lines = str(error).splitlines()
    assert lines[1:] == [f"  problem {i}" for i in range(10)] + ["  ... and 3 more"]
    assert len(error.problems) == 13


def test_an_error_without_problems_has_an_empty_tuple_and_no_location():
    error = pulse.ExecutionError("failed")
    assert error.problems == ()
    assert error.location is None
    assert str(error) == "failed"


def test_a_problem_prints_its_location_path_and_fix():
    problem = pulse.Problem(
        "must be positive", path="shots", fix="pass shots=1000", file="run.json", line=3, column=7
    )
    assert str(problem) == "run.json:3:7: shots: must be positive; pass shots=1000"
    assert str(pulse.Problem("bad", file="f.json")) == "f.json: bad"
    assert pulse.Problem("bad").value is pulse.Problem.UNSET


ALL = [
    pulse.InvalidValueError,
    pulse.InvalidTypeError,
    pulse.NotFoundError,
    pulse.IndexOutOfRangeError,
    pulse.UnsupportedError,
    pulse.CompilationError,
    pulse.TimingError,
    pulse.ExecutionError,
    pulse.InternalError,
    pulse.DeviceDefinitionError,
    UnknownTargetError,
    BackendNotFoundError,
    UnsupportedInterpreterError,
    GpuRuntimeError,
]


@pytest.mark.parametrize("cls", ALL, ids=lambda c: c.__name__)
def test_each_error_survives_pickling_with_its_problems(cls):
    problem = pulse.Problem("must be positive", path="shots", value=0, line=2)
    error = cls("1 problem", problems=[problem])
    copy = pickle.loads(pickle.dumps(error))
    assert type(copy) is cls
    assert str(copy) == str(error)
    assert copy.problems == (problem,)
    assert copy.problems[0].value == 0


def test_unset_survives_pickling_as_itself():
    problem = pickle.loads(pickle.dumps(pulse.Problem("x")))
    assert problem.value is pulse.Problem.UNSET


def test_a_symbolic_value_error_survives_pickling():
    error = SymbolicValueError("needs the value of 'theta'", ("theta",), "int()")
    copy = pickle.loads(pickle.dumps(error))
    assert (str(copy), copy.names, copy.operation) == (str(error), ("theta",), "int()")


def test_an_internal_error_asks_for_a_report():
    error = pulse.InternalError("the slot table is empty")
    text = str(error)
    assert text.startswith("the slot table is empty; this is a bug in cudaq-pulse ")
    assert text.endswith(
        "; please report it at https://github.com/NVIDIA/cudaq-pulse/issues "
        "with the kernel that raised it"
    )
    assert str(pickle.loads(pickle.dumps(error))) == text


def test_library_errors_are_pulse_errors_of_their_kind():
    from cudaq_pulse.classical import emulation
    from cudaq_pulse.targets import Calibration, transmon_generic

    with pytest.raises(pulse.NotFoundError, match="no calibration 'missing'"):
        Calibration.load("missing")
    with pytest.raises(pulse.InvalidValueError, match="n_qubits must be >= 1"):
        transmon_generic(0)
    with pytest.raises(pulse.UnsupportedError, match="not available yet"):
        emulation.device_call_text(None)
    with pytest.raises(pulse.CompilationError, match="inside a @cudaq_pulse.kernel"):
        pulse.ops.wait(None, 4)

    @pulse.kernel
    def empty():
        pass

    with pytest.raises(pulse.InvalidValueError, match="Did you mean 'fuse_pulses'"):
        pulse.compile(empty, passes=("fuse_pulse",))
    with pytest.raises(pulse.InvalidValueError, match="not the string 'verify'"):
        pulse.compile(empty, passes="verify")


# The modules that still raise built-in errors, and how many; every other
# module raises library errors only.
BUILTIN_RAISES = {
    "_loops.py": 1,
    "_waits.py": 6,
    "compiler.py": 14,
    "controller.py": 11,
    "kernel/_kernel.py": 7,
}


def test_only_known_modules_raise_builtin_errors():
    import re
    from pathlib import Path

    raise_builtin = re.compile(
        r"\braise (ValueError|TypeError|KeyError|RuntimeError|FileNotFoundError|"
        r"NotImplementedError)\("
    )
    root = Path(pulse.__file__).parent
    found = {}
    for path in sorted(root.rglob("*.py")):
        count = len(raise_builtin.findall(path.read_text()))
        if count:
            found[path.relative_to(root).as_posix()] = count
    assert found == BUILTIN_RAISES


def test_the_specialization_warning_is_a_pulse_warning_and_a_runtime_warning():
    from tests._handles import Q

    @pulse.kernel
    def k(q, x: float):
        d = get_drive_line(q)
        drive(d, square(int(x), 0.1))

    with pytest.warns(pulse.PulseWarning, match="float argument 'x'") as record:
        pulse.compile(k).lower(Q[0], 8.0)
    assert issubclass(record[0].category, RuntimeWarning)


def test_pulse_warnings_can_be_made_errors_at_once():
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=pulse.PulseWarning)
        with pytest.raises(pulse.UpgradeNotice):
            warnings.warn("use the newer format", pulse.UpgradeNotice, stacklevel=1)
