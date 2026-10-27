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


def test_pulse_warnings_can_be_made_errors_at_once():
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=pulse.PulseWarning)
        with pytest.raises(pulse.UpgradeNotice):
            warnings.warn("use the newer format", pulse.UpgradeNotice, stacklevel=1)
