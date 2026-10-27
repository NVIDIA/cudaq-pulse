# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Finding controller backends by their entry points, with the toy backend."""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import backends

from tests._handles import ladder_target

_TOY = Path(__file__).resolve().parents[1] / "examples" / "controller_backend"


@pytest.fixture
def installed_toy(tmp_path, monkeypatch):
    """The toy backend as if installed: its package and a dist-info that
    declares its entry point, on sys.path."""
    info = tmp_path / "toy_controller-0.1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: toy-controller\nVersion: 0.1.0\n")
    (info / "entry_points.txt").write_text(f"[{backends.GROUP}]\ntoy = toy_controller:ToyBackend\n")
    monkeypatch.syspath_prepend(str(_TOY))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(backends, "_loaded", {})
    yield
    sys.modules.pop("toy_controller", None)


@pulse.kernel
def drive_and_measure(q):
    d, t = get_drive_line(q)
    r, rt = get_readout_line(q)
    drive(d, gaussian(40, 0.5, 10.0), t)
    sync(d, r)
    readout(r, square(200, 0.05), rt)


def _device():
    return pulse.device({"target": dataclasses.replace(ladder_target(4), clock_ghz=1.0)})


def test_an_installed_backend_is_listed_without_being_imported(installed_toy):
    assert "toy" in backends.available()
    assert "toy_controller" not in sys.modules


def test_an_installed_backend_compiles_emits_and_runs(installed_toy):
    backend = backends.get("toy")
    assert isinstance(backend, backends.Backend) and backend.name == "toy"
    assert backends.get("toy") is backend
    dev = _device()
    compiled = pulse.compile(drive_and_measure, device=dev, machine=backend.machine_model())
    program = backend.emit(compiled.lower(dev.qudit_handles[0]).export_controller())
    assert program.splitlines() == [
        "0 0 drive gaussian 0.5 5e+09",
        "1 40 readout square 0.05 5e+09",
    ]
    result = backend.run(program, shots=3)
    assert result.played == ((0, 0, "drive"), (1, 40, "readout"))
    assert result.outcomes == ((0,), (0,), (0,))


def test_the_backends_limits_reject_what_its_controller_cannot_play(installed_toy):
    @pulse.kernel
    def two_readouts(q, r):
        ra, ta = get_readout_line(q)
        rb, tb = get_readout_line(r)
        readout(ra, square(40, 0.05), ta)
        readout(rb, square(40, 0.05), tb)

    dev = _device()
    machine = backends.get("toy").machine_model()
    with pytest.raises(pulse.controller.ControllerLimitError, match="acquirer 0"):
        pulse.compile(two_readouts, device=dev, schedule="asap", machine=machine).lower(
            dev.qudit_handles[0], dev.qudit_handles[1]
        )


def test_an_unknown_backend_names_the_installed_ones(installed_toy):
    with pytest.raises(backends.BackendNotFoundError, match=r"no controller backend 'tape'.*toy"):
        backends.get("tape")


def test_a_registered_backend_takes_precedence_and_can_be_removed(installed_toy):
    class Other:
        name = "toy"

        def machine_model(self):
            return pulse.compiler.MachineModel()

        def emit(self, program):
            return "other"

        def run(self, artifact, *, shots=1):
            return artifact

    backends.register("toy", Other)
    try:
        assert backends.get("toy").emit({}) == "other"
    finally:
        backends.unregister("toy")
    assert backends.get("toy").__class__.__name__ == "ToyBackend"
    with pytest.raises(ValueError, match="identifier"):
        backends.register("not a name", Other)


def test_an_entry_point_that_is_not_a_backend_is_refused(tmp_path, monkeypatch):
    info = tmp_path / "broken-0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: broken\nVersion: 0.1\n")
    (info / "entry_points.txt").write_text(f"[{backends.GROUP}]\nbroken = json\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(backends, "_loaded", {})
    with pytest.raises(TypeError, match="not a cudaq_pulse.backends.Backend"):
        backends.get("broken")


def test_importing_cudaq_pulse_imports_neither_module():
    code = (
        "import sys, cudaq_pulse; "
        "print('cudaq_pulse.backends' in sys.modules, 'cudaq_pulse.controller' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "False"]
