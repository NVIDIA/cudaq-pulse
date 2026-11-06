# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Finding controller backends by their entry points, with a dummy backend."""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import backends
from cudaq_pulse.controller import ControllerLimitError

from tests._handles import ladder_target

_PLUGINS = Path(__file__).resolve().parent / "plugins"


def _install(tmp_path, monkeypatch, entries: dict[str, str], name: str = "dummy-controller"):
    """A distribution declaring *entries* in the backends group, as if
    installed: its dist-info on sys.path, beside the dummy plugin's module."""
    info = tmp_path / f"{name.replace('-', '_')}-0.1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 0.1.0\n")
    lines = "".join(f"{key} = {value}\n" for key, value in entries.items())
    (info / "entry_points.txt").write_text(f"[{backends.GROUP}]\n{lines}")
    monkeypatch.syspath_prepend(str(_PLUGINS))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(backends, "_loaded", {})


@pytest.fixture
def installed(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch, {"dummy": "dummy_controller:DummyBackend"})
    yield
    sys.modules.pop("dummy_controller", None)


@pulse.kernel
def drive_and_measure(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, gaussian(40, 0.5, 10.0))
    sync(d, r)
    readout(r, square(200, 0.05))


@pulse.kernel
def two_readouts(q, r):
    ra = get_readout_line(q)
    rb = get_readout_line(r)
    readout(ra, square(40, 0.05))
    readout(rb, square(40, 0.05))


def _device():
    return pulse.device({"target": ladder_target(4), "clock_ghz": 1.0})


def test_an_installed_backend_is_listed_without_being_imported(installed):
    assert "dummy" in backends.available()
    assert "dummy_controller" not in sys.modules


def test_an_installed_backend_compiles_emits_and_runs(installed):
    backend = backends.get("dummy")
    assert isinstance(backend, backends.Backend) and backend.name == "dummy"
    assert backends.get("dummy") is backend
    dev = _device()
    compiled = pulse.compile(drive_and_measure, device=dev, machine=backend.machine_model())
    artifact = backend.emit(compiled.lower(dev.qudit_handles[0]).export_controller())
    assert artifact == [(0, 0, "drive", "gaussian"), (40, 1, "readout", "square")]
    assert backend.run(artifact, shots=3)["outcomes"] == [[0], [0], [0]]


def test_the_backends_limits_reject_what_its_controller_cannot_play(installed):
    dev = _device()
    machine = backends.get("dummy").machine_model()
    with pytest.raises(ControllerLimitError, match="detector 0"):
        pulse.compile(two_readouts, device=dev, schedule="asap", machine=machine).lower(
            dev.qudit_handles[0], dev.qudit_handles[1]
        )
    # Serialized by the scheduler, with the detector's hold time between them.
    machine = dataclasses.replace(machine, line_switch_penalty_cycles=10)
    pulse.compile(two_readouts, device=dev, schedule="rcp", machine=machine).lower(
        dev.qudit_handles[0], dev.qudit_handles[1]
    )


def test_an_unknown_backend_names_the_installed_ones(installed):
    with pytest.raises(backends.BackendNotFoundError, match=r"no controller backend 'tape'.*dummy"):
        backends.get("tape")
    assert issubclass(backends.BackendNotFoundError, LookupError)


def test_a_registered_backend_takes_precedence_and_can_be_removed(installed):
    class Other:
        name = "dummy"

        def machine_model(self):
            return pulse.compiler.MachineModel()

        def emit(self, program):
            return "other"

        def run(self, artifact, *, shots=1):
            return artifact

    backends.register("dummy", Other)
    try:
        assert backends.get("dummy").emit({}) == "other"
        assert "dummy_controller" not in sys.modules
    finally:
        backends.unregister("dummy")
    assert type(backends.get("dummy")).__name__ == "DummyBackend"
    with pytest.raises(ValueError, match="identifier"):
        backends.register("not a name", Other)


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("json", "not a cudaq_pulse.backends.Backend"),
        ("dummy_controller:OldBackend", r"reads cudaq-pulse.controller/0; .* exports"),
        ("no_such_module:Backend", r"failed to load: ModuleNotFoundError"),
    ],
    ids=["not-a-backend", "old-format", "import-error"],
)
def test_an_incompatible_backend_is_refused(tmp_path, monkeypatch, target, message):
    _install(tmp_path, monkeypatch, {"broken": target}, name="broken")
    with pytest.raises(backends.IncompatibleBackendError, match=message) as caught:
        backends.get("broken")
    assert isinstance(caught.value, backends.BackendError)
    assert isinstance(caught.value, TypeError)
    sys.modules.pop("dummy_controller", None)


def test_importing_cudaq_pulse_looks_for_no_backend():
    code = (
        "import sys, importlib.metadata as m\n"
        "calls = []\n"
        "real = m.entry_points\n"
        "m.entry_points = lambda **k: calls.append(k) or real(**k)\n"
        "import cudaq_pulse\n"
        "cudaq_pulse.compile\n"
        "print('cudaq_pulse.backends' in sys.modules, 'cudaq_pulse.controller' in sys.modules,"
        " any(k.get('group') == 'cudaq_pulse.backends' for k in calls))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "False", "False"]
