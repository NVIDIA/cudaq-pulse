# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata
import io
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cudaq_pulse
import pytest
from cudaq_pulse import _build_config, backends, doctor
from cudaq_pulse.runtime import gpu

compiler_only = pytest.mark.skipif(
    _build_config.GPU_RUNTIME_ENABLED, reason="checks the compiler-only configuration"
)


@pytest.fixture(autouse=True)
def _fresh_guard(monkeypatch):
    monkeypatch.delenv("CUDM_RUNTIME_LIB", raising=False)
    monkeypatch.delenv("CUDAQ_PULSE_CUDA", raising=False)
    monkeypatch.delenv("CUDAQ_PULSE_CONTEXT_MAX_MODULES", raising=False)
    monkeypatch.delenv("CUDAQ_PULSE_CONTEXT_MAX_WORDS", raising=False)
    monkeypatch.setattr(gpu, "_runtime", None)


def _run(*argv):
    out = io.StringIO()
    return doctor.main(list(argv), out=out), out.getvalue()


@compiler_only
def test_a_compiler_only_build_passes_on_the_cpu(monkeypatch):
    for argv in ([], ["--allow-no-gpu"]):
        code, output = _run(*argv)
        assert code == 0, output
        assert "Compiler (MLIR JIT): ok" in output
        assert "CPU simulation (NumPy reference): ok" in output
        assert "GPU simulation: not set up (GpuSupportNotBuiltError)" in output
        assert 'pip install "cudaq-pulse[cu12]" (NVIDIA driver 525.60.13 or newer)' in output
        assert 'pip install "cudaq-pulse[cu13]" (NVIDIA driver 580 or newer)' in output
        assert output.rstrip().endswith(
            "Result: OK, compilation and CPU simulation work; GPU simulation is not set up."
        )
    assert _run("--expect-no-device")[0] == 1
    # Asking for a CUDA runtime the build does not have is a failure.
    monkeypatch.setenv("CUDAQ_PULSE_CUDA", "12")
    code, output = _run()
    assert code == 1
    assert "GPU simulation: unavailable (GpuSupportNotBuiltError)" in output
    assert "Result: FAILED, CUDAQ_PULSE_CUDA is set but GPU simulation does not work" in output
    assert _run("--allow-no-gpu")[0] == 0


def test_cpu_simulation_failure_fails_every_mode(monkeypatch):
    monkeypatch.setattr(doctor, "_check_cpu_simulation", lambda: "RuntimeError: boom")
    monkeypatch.setattr(doctor, "_diagnose_gpu", lambda write: gpu.GpuSupportNotBuiltError())
    for argv in ([], ["--allow-no-gpu"], ["--expect-no-device"]):
        code, output = _run(*argv)
        assert code == 1
        assert "CPU simulation (NumPy reference): FAILED\n  RuntimeError: boom" in output
        assert output.rstrip().endswith("Result: FAILED, CPU simulation does not work.")


def test_a_driver_too_old_for_the_installed_extra_fails(monkeypatch):
    message = (
        "The installed NVIDIA driver supports CUDA up to 12.8, but cudaq-pulse[cu13] "
        "needs driver 580 or newer (CUDA 13). Upgrade the driver."
    )

    def old_driver():
        raise gpu.GpuDeviceUnavailableError(message)

    monkeypatch.setattr(gpu, "_installed_cuda_majors", lambda: [13])
    monkeypatch.setattr(gpu, "require_gpu_runtime", old_driver)
    code, output = _run()
    assert code == 1
    assert "GPU simulation: unavailable (GpuDeviceUnavailableError)" in output
    assert message in output
    assert output.rstrip().endswith(
        "Result: FAILED, cudaq-pulse[cu13] is installed but GPU simulation does not work "
        "(CPU simulation does)."
    )
    assert _run("--expect-no-device")[0] == 0


@compiler_only
def test_module_entry_point(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(Path(cudaq_pulse.__file__).parent.parent))
    # The suite's own target is registered only inside the test process.
    env.pop("CUDAQ_PULSE_QUANTUM_SYSTEM", None)
    result = subprocess.run(
        [sys.executable, "-m", "cudaq_pulse.doctor"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("cudaq-pulse ")


def test_compiler_failure_fails_every_mode(monkeypatch):
    monkeypatch.setattr(doctor, "_check_compiler", lambda: "RuntimeError: boom")
    monkeypatch.setattr(doctor, "_diagnose_gpu", lambda write: gpu.GpuSupportNotBuiltError())
    for argv in ([], ["--allow-no-gpu"], ["--expect-no-device"]):
        code, output = _run(*argv)
        assert code == 1
        assert "boom" in output
        assert "CPU simulation (NumPy reference): not checked" in output


def test_flags_are_mutually_exclusive():
    with pytest.raises(SystemExit) as info:
        _run("--allow-no-gpu", "--expect-no-device")
    assert info.value.code == 2


@pytest.mark.parametrize(
    ("outcome", "installed", "env", "flags", "expected"),
    [
        (None, [12], {}, {}, 0),
        # Nothing asks for GPU simulation: CPU simulation is enough.
        (gpu.GpuSupportNotBuiltError(), [], {}, {}, 0),
        (gpu.GpuRuntimeLoadError(), [], {}, {}, 0),
        # A CUDA extra or a runtime variable asks for it, so it must work.
        (gpu.GpuDeviceUnavailableError(), [12], {}, {}, 1),
        (gpu.GpuDeviceUnavailableError(), [13], {}, {}, 1),
        (gpu.GpuRuntimeLoadError(), [12, 13], {}, {}, 1),
        (gpu.GpuSupportNotBuiltError(), [], {"CUDAQ_PULSE_CUDA": "12"}, {}, 1),
        (gpu.GpuRuntimeLoadError(), [], {"CUDM_RUNTIME_LIB": "/none"}, {}, 1),
        (gpu.GpuSupportNotBuiltError(), [], {}, {"allow_no_gpu": True}, 0),
        (gpu.GpuRuntimeLoadError(), [], {}, {"allow_no_gpu": True}, 0),
        (gpu.GpuRuntimeLoadError(), [], {"CUDM_RUNTIME_LIB": "/none"}, {"allow_no_gpu": True}, 0),
        (gpu.GpuRuntimeLoadError(), [13], {}, {"allow_no_gpu": True}, 1),
        (gpu.GpuRuntimeLoadError(), [12, 13], {}, {"allow_no_gpu": True}, 1),
        (gpu.GpuDeviceUnavailableError(), [13], {}, {"allow_no_gpu": True}, 0),
        (gpu.GpuDeviceUnavailableError(), [12], {}, {"expect_no_device": True}, 0),
        (gpu.GpuRuntimeLoadError(), [12], {}, {"expect_no_device": True}, 1),
        (gpu.GpuSupportNotBuiltError(), [], {}, {"expect_no_device": True}, 1),
        (None, [12], {}, {"expect_no_device": True}, 1),
    ],
)
def test_exit_codes(monkeypatch, outcome, installed, env, flags, expected):
    monkeypatch.setattr(gpu, "_installed_cuda_majors", lambda: installed)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    lines = []
    code = doctor._exit_code(
        outcome,
        allow_no_gpu=flags.get("allow_no_gpu", False),
        expect_no_device=flags.get("expect_no_device", False),
        write=lines.append,
    )
    assert code == expected
    assert lines[-1].startswith("Result: ")


def _installed(monkeypatch, tmp_path, driver_version, script):
    """Pretend cudaq-pulse 1.2.3 is installed, with cudaq-pulse-tools *driver_version*
    (``None``: not installed) whose cudaq-pulse-opt runs *script* (``None``: absent)."""
    files = []
    if script is not None:
        executable = tmp_path / "bin" / "cudaq-pulse-opt"
        executable.parent.mkdir()
        executable.write_text(f"#!/bin/sh\n{script}\n")
        executable.chmod(0o755)
        files.append(importlib.metadata.PackagePath("bin/cudaq-pulse-opt"))
    distributions = {
        "cudaq-pulse": SimpleNamespace(version="1.2.3", read_text=lambda name: None),
        "cudaq-pulse-tools": None
        if driver_version is None
        else SimpleNamespace(
            version=driver_version, files=files, locate_file=lambda path: tmp_path / path
        ),
    }
    monkeypatch.setattr(doctor, "_distribution", distributions.get)
    monkeypatch.setattr(doctor, "_diagnose_gpu", lambda write: gpu.GpuSupportNotBuiltError())


@pytest.mark.parametrize(
    ("driver_version", "script", "message"),
    [
        (None, "true", "cudaq-pulse-tools is not installed, and cudaq-pulse 1.2.3 requires it"),
        ("1.2.2", "true", "cudaq-pulse-tools 1.2.2 is installed, and cudaq-pulse 1.2.3 requires"),
        ("1.2.3", None, "pip install --force-reinstall cudaq-pulse-tools==1.2.3"),
        ("1.2.3", "echo broken >&2; exit 3", "--version exited with 3: broken"),
        (
            "1.2.3",
            "echo 'LLVM version 22.1.8'",
            "--version printed 'LLVM version 22.1.8', not "
            "'cudaq-pulse-opt <version> (LLVM <version>)'",
        ),
        (
            "1.2.3",
            "echo 'cudaq-pulse-opt 1.2.2 (LLVM 22.1.8)'",
            "reports version 1.2.2, but its distribution is 1.2.3: "
            "pip install --force-reinstall cudaq-pulse-tools==1.2.3",
        ),
    ],
)
def test_a_missing_or_broken_driver_fails(monkeypatch, tmp_path, driver_version, script, message):
    _installed(monkeypatch, tmp_path, driver_version, script)
    for argv in ([], ["--allow-no-gpu"], ["--expect-no-device"]):
        code, output = _run(*argv)
        assert code == 1
        assert "MLIR driver (cudaq-pulse-opt): FAILED" in output
        assert message in output
        assert output.rstrip().endswith("Result: FAILED, cudaq-pulse-opt does not work.")
    if driver_version != "1.2.3":
        assert "pip install cudaq-pulse-tools==1.2.3" in output


def test_the_installed_driver_is_reported(monkeypatch, tmp_path):
    _installed(monkeypatch, tmp_path, "1.2.3", "echo 'cudaq-pulse-opt 1.2.3 (LLVM 22.1.8)'")
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    code, output = _run("--allow-no-gpu")
    assert code == 0, output
    executable = (tmp_path / "bin" / "cudaq-pulse-opt").resolve()
    assert f"MLIR driver (cudaq-pulse-opt): 1.2.3 (LLVM 22.1.8) at {executable}\n" in output
    monkeypatch.setenv("PATH", "")
    code, output = _run("--allow-no-gpu")
    assert code == 0, output
    assert f"not on PATH; add {executable.parent} to PATH" in output


def test_a_build_tree_driver_is_only_reported(monkeypatch):
    monkeypatch.setattr(doctor, "_distribution", lambda name: None)
    monkeypatch.setattr(doctor, "_diagnose_gpu", lambda write: gpu.GpuSupportNotBuiltError())
    monkeypatch.setenv("PATH", "")
    code, output = _run("--allow-no-gpu")
    assert code == 0, output
    assert (
        "MLIR driver (cudaq-pulse-opt): not on PATH (source or build tree; not checked)" in output
    )


@compiler_only
def test_reports_the_mlir_context_limits(monkeypatch):
    code, output = _run("--allow-no-gpu")
    assert code == 0
    assert "MLIR context reuse: up to 256 modules or 1048576 packed words" in output
    monkeypatch.setenv("CUDAQ_PULSE_CONTEXT_MAX_MODULES", "8")
    code, output = _run("--allow-no-gpu")
    assert code == 0
    assert "up to 8 modules" in output
    assert "CUDAQ_PULSE_CONTEXT_MAX_MODULES=8" in output


def test_invalid_mlir_context_limits_fail(monkeypatch):
    monkeypatch.setenv("CUDAQ_PULSE_CONTEXT_MAX_WORDS", "lots")
    code, output = _run("--allow-no-gpu")
    assert code == 1
    assert "MLIR context reuse: INVALID" in output
    assert "CUDAQ_PULSE_CONTEXT_MAX_WORDS=lots is not valid" in output


def test_the_process_target_and_kernel_cache_are_reported(monkeypatch):
    monkeypatch.setattr(doctor, "_diagnose_gpu", lambda write: gpu.GpuSupportNotBuiltError())
    monkeypatch.setenv("CUDAQ_PULSE_QUANTUM_SYSTEM", "transmon_lattice:3")
    monkeypatch.setenv("CUDAQ_PULSE_CACHE_SIZE", "8")
    code, out = _run("--allow-no-gpu")
    assert code == 0
    assert (
        "Process quantum system: transmon_lattice:3 "
        "(env CUDAQ_PULSE_QUANTUM_SYSTEM=transmon_lattice:3" in out
    )
    assert "Kernel cache: 8 specializations per kernel (CUDAQ_PULSE_CACHE_SIZE=8)" in out


def test_the_installed_controller_backends_are_listed(monkeypatch):
    monkeypatch.setattr(doctor, "_diagnose_gpu", lambda write: gpu.GpuSupportNotBuiltError())
    backends.register("doctor_listing_probe", object)
    try:
        code, out = _run("--allow-no-gpu")
    finally:
        backends.unregister("doctor_listing_probe")
    assert code == 0
    (line,) = [line for line in out.splitlines() if line.startswith("Controller backends")]
    assert "doctor_listing_probe" in line.split(": ", 1)[1].split(", ")


@pytest.mark.parametrize(
    ("name", "value"), [("CUDAQ_PULSE_QUANTUM_SYSTEM", "nope"), ("CUDAQ_PULSE_CACHE_SIZE", "0")]
)
def test_invalid_kernel_settings_fail(monkeypatch, name, value):
    monkeypatch.setattr(doctor, "_diagnose_gpu", lambda write: gpu.GpuSupportNotBuiltError())
    monkeypatch.setenv(name, value)
    code, out = _run("--allow-no-gpu")
    assert code == 1
    assert "Result: FAILED, invalid kernel settings." in out


def test_system_lists_every_problem_of_a_device_file(tmp_path):
    path = tmp_path / "lab.json"
    path.write_text('{"name": "transmon_surface17",\n "layout": [4, 99], "measurement_run": "x"}\n')
    code, output = _run("--system", str(path))
    assert code == 1
    assert output.splitlines() == [
        f"{path}:2:16: error: layout[1]: carrier 99 is not in quantum system "
        "'transmon_surface17' (it has 0..16)",
        f"{path}:2:40: error: measurement_run: 'x' is given without a calibration; add "
        '"calibration", a shipped name or a file path, or remove "measurement_run"',
        f"{path}: 2 problems",
    ]


def test_system_json_report(tmp_path):
    import json

    path = tmp_path / "lab.json"
    path.write_text('{"name": "transmon_surface17", "layout": [4, 99]}')
    code, output = _run("--system", str(path), "--json")
    assert code == 1
    report = json.loads(output)
    assert report["ok"] is False and report["quantum_system"] is None
    assert report["problems"] == [
        {
            "message": "carrier 99 is not in quantum system 'transmon_surface17' (it has 0..16)",
            "path": "layout[1]",
            "value": 99,
            "fix": "",
            "file": str(path),
            "line": 1,
            "column": 46,
        }
    ]
    path.write_text('{"name": "transmon_surface17", "layout": [4, 5]}')
    code, output = _run("--system", str(path), "--json")
    assert code == 0
    assert json.loads(output) == {
        "file": str(path),
        "ok": True,
        "quantum_system": {"name": "transmon_surface17", "qudits": 2, "layout": [4, 5]},
        "problems": [],
    }


def test_system_reports_a_good_or_unreadable_file(tmp_path):
    path = tmp_path / "lab.json"
    path.write_text('{"name": "transmon_surface17", "layout": [4, 5]}')
    assert _run("--system", str(path)) == (
        0,
        f"{path}: ok, quantum system transmon_surface17 with 2 qudits\n",
    )
    code, output = _run("--system", str(tmp_path / "absent.json"))
    assert code == 1
    assert output.startswith(f"{tmp_path / 'absent.json'}: error: cannot read the file: ")


def test_json_needs_system():
    with pytest.raises(SystemExit) as info:
        _run("--json")
    assert info.value.code == 2
    with pytest.raises(SystemExit) as info:
        _run("--system", "a.json", "--allow-no-gpu")
    assert info.value.code == 2
