# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Validating a controller backend against the export test corpus."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import Controller, _validation, backends
from cudaq_pulse.compiler import MachineModel
from cudaq_pulse.controller import _EVENT_KINDS, _FORMATS, unroll

_PLUGINS = Path(__file__).resolve().parent / "plugins"
_KNOWN = _EVENT_KINDS | {"drive", "readout", "submit"}


class ToyBackend:
    """A backend that emits each controller's events as tuples, reports them
    back, and emulates every measurement with outcome 0."""

    name = "toy"
    formats = _FORMATS

    def __init__(self, limits: Controller | None = None, shift: int = 0) -> None:
        self.limits = limits or Controller(clock_ghz=1.0)
        self.shift = shift

    def machine_model(self) -> MachineModel:
        return MachineModel(controller=self.limits)

    def emit(self, program: Mapping[str, Any]) -> list[list[dict[str, Any]]]:
        if program.get("format") not in _FORMATS:
            raise backends.IncompatibleBackendError(f"cannot read {program.get('format')!r}")
        flat = unroll(program)
        unknown = {e["kind"] for e in flat["events"]} - _KNOWN
        if unknown:
            raise backends.IncompatibleBackendError(f"cannot emit {sorted(unknown)}")
        return [
            [
                {**e, "start_cycles": e["start_cycles"] + (self.shift if k == 1 else 0)}
                for k, e in enumerate(flat["events"])
            ]
        ]

    def events(self, artifact: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
        return artifact

    def emulate(self, artifact: Any, *, shots: int) -> list[list[int]]:
        readouts = sum(1 for e in artifact[0] if e["kind"] == "readout")
        return [[0] * readouts for _ in range(shots)]

    def run(self, artifact: Any, *, shots: int = 1) -> Any:
        return self.emulate(artifact, shots=shots)


@pytest.fixture
def ideal(monkeypatch):
    """Every simulated measurement gives 0, as the toy emulator does."""
    monkeypatch.setattr(_validation, "_simulated", lambda lowered: [0.0] * _readouts(lowered))


def _readouts(lowered):
    return sum(
        1 for e in unroll(lowered.export_controller(version=2))["events"] if e["kind"] == "readout"
    )


def test_the_corpus_has_a_program_for_every_event_kind():
    assert set(pulse.Event) <= set(_validation._corpus())


def test_a_conforming_backend_passes_in_every_format_it_reads(ideal):
    report = backends.validate(ToyBackend())
    assert report.ok, str(report)
    lines = str(report).splitlines()
    assert "PASS play (/1)" in lines and "PASS loop (/2)" in lines
    assert "PASS unsupported_event" in lines and "PASS newer_format" in lines
    assert lines[-1] == "backend 'toy': 24 passed"


def test_the_corpus_tells_where_a_run_time_frequency_change_is_made():
    dev = _validation._device(MachineModel())
    compiled = pulse.compile(_validation._corpus()["runtime_frequency"], device=dev)
    lowered = compiled.lower(*dev.qudit_handles[:2])
    events = lowered.export_controller()["events"]
    first, second = (e for e in events if e["kind"] == "drive")
    [change] = second["frame_updates"]
    # An emulator that took the change at the second pulse would miss the
    # phase the frame gains between them.
    assert change["op"] == "shift_frequency"
    assert change["start_cycles"] == first["end_cycles"] == second["start_cycles"] - 200
    (p,) = _validation._simulated(lowered)
    assert p < 0.1


def test_a_backend_that_accepts_what_it_cannot_read_fails(monkeypatch):
    monkeypatch.syspath_prepend(str(_PLUGINS))
    from dummy_controller import DummyBackend

    sys.modules.pop("dummy_controller")
    report = backends.validate(DummyBackend())
    assert not report.ok
    text = str(report)
    assert (
        "FAIL unsupported_event: emit() accepted 'forward_outcome'; "
        "it should raise IncompatibleBackendError" in text
    )
    assert "FAIL newer_format: emit() raised ValueError" in text


def test_timing_that_differs_from_the_controller_program_fails(ideal):
    report = backends.validate(ToyBackend(shift=3))
    assert not report.ok
    assert (
        "FAIL play (/1): line 1: event 0 is a play at 51 cycles; "
        "the controller program has a play at 48" in str(report)
    )


def test_a_wrong_number_of_programs_fails(ideal, monkeypatch):
    backend = ToyBackend()
    monkeypatch.setattr(backend, "events", lambda artifact: artifact * 2)
    assert "FAIL play (/1): emitted 2 programs, expected 1" in str(backends.validate(backend))


def test_a_program_beyond_the_declared_controller_is_skipped(ideal, monkeypatch):
    backend = ToyBackend()
    monkeypatch.setattr(backend, "machine_model", lambda: MachineModel())
    no_loop = [c for c in pulse.Capability if c is not pulse.Capability.LOOP]
    limits = Controller(clock_ghz=1.0, max_pulses=2, supports=no_loop)
    report = backends.validate(backend, controller=limits)
    assert "SKIP play: exceeds the controller's limits" in str(report)
    assert report.ok


def test_emitted_pulses_must_respect_the_declared_limits():
    limits = Controller(instruction_cycles=4, max_pulses=2)
    close = {0: [("play", 0, 16), ("play", 2, 16)]}
    assert _validation._limits(limits, close) == (
        "line 0: two pulses start 2 cycles apart; the controller issues one per 4"
    )
    many = {0: [("play", 0, 4), ("play", 8, 4)], 1: [("acquire", 0, 4)]}
    assert _validation._limits(limits, many) is None
    no_loop = [c for c in pulse.Capability if c is not pulse.Capability.LOOP]
    unrolled = Controller(instruction_cycles=4, max_pulses=2, supports=no_loop)
    assert _validation._limits(unrolled, many) == "emitted 3 pulses; the controller plays at most 2"
    assert _validation._limits(None, many) is None


def test_an_emulator_that_disagrees_with_the_simulator_fails(monkeypatch):
    monkeypatch.setattr(_validation, "_simulated", lambda lowered: [0.5] * _readouts(lowered))
    report = backends.validate(ToyBackend(), shots=200)
    assert "FAIL acquire (/1): measurement 0: the emulator gives 1 in 0.000 of shots" in str(report)


def test_the_emulator_is_compared_with_the_simulator():
    lowered = pulse.compile(
        _validation._corpus()[pulse.Event.ACQUIRE], device=_validation._device(MachineModel())
    ).lower(*_validation._device(MachineModel()).qudit_handles[:2])
    (p,) = _validation._simulated(lowered)
    assert 0.0 < p < 1.0


def test_a_backend_that_refuses_everything_fails(ideal, monkeypatch):
    backend = ToyBackend()

    def refuse(program):
        raise backends.IncompatibleBackendError("no")

    monkeypatch.setattr(backend, "emit", refuse)
    report = backends.validate(backend)
    assert "REFUSED play (/1): no" in str(report)
    assert "FAIL corpus: the backend refused every program" in str(report)


def test_validate_checks_its_arguments():
    with pytest.raises(ValueError, match="shots must be positive"):
        backends.validate(ToyBackend(), shots=0)
    with pytest.raises(backends.IncompatibleBackendError, match="is not a"):
        backends.validate(object())  # type: ignore[arg-type]
    with pytest.raises(backends.BackendNotFoundError):
        backends.validate("no_such_backend")


def test_only_controllers_that_opt_in_are_validated(ideal):
    found = {"toy": ToyBackend()}
    limits = Controller(clock_ghz=1.0)
    reports = _validation.opted_in(
        [("vendor/a", {"backend": "toy"}, limits), ("vendor/b", {}, limits)], found.__getitem__
    )
    assert [name for name, _ in reports] == ["vendor/a"] and reports[0][1].ok


def test_validation_costs_nothing_until_it_is_called():
    code = (
        "import sys, cudaq_pulse as p; from cudaq_pulse import backends; "
        "print('cudaq_pulse._validation' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
