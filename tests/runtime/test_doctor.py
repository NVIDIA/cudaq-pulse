# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import io
import os
import subprocess
import sys
from pathlib import Path

import cudaq_pulse
import pytest
from cudaq_pulse import _build_config, doctor
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
def test_compiler_only_build_passes_only_with_allow_no_gpu():
    code, output = _run("--allow-no-gpu")
    assert code == 0
    assert "Compiler (MLIR JIT): ok" in output
    assert "GpuSupportNotBuiltError" in output
    assert _run()[0] == 1
    assert _run("--expect-no-device")[0] == 1


@compiler_only
def test_module_entry_point(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(Path(cudaq_pulse.__file__).parent.parent))
    # The suite's own target is registered only inside the test process.
    env.pop("CUDAQ_PULSE_TARGET", None)
    result = subprocess.run(
        [sys.executable, "-m", "cudaq_pulse.doctor", "--allow-no-gpu"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
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


def test_flags_are_mutually_exclusive():
    with pytest.raises(SystemExit) as info:
        _run("--allow-no-gpu", "--expect-no-device")
    assert info.value.code == 2


@pytest.mark.parametrize(
    ("outcome", "installed", "flags", "expected"),
    [
        (None, [12], {}, 0),
        (gpu.GpuDeviceUnavailableError(), [12], {}, 1),
        (gpu.GpuSupportNotBuiltError(), [], {}, 1),
        (gpu.GpuSupportNotBuiltError(), [], {"allow_no_gpu": True}, 0),
        (gpu.GpuRuntimeLoadError(), [], {"allow_no_gpu": True}, 0),
        (gpu.GpuRuntimeLoadError(), [13], {"allow_no_gpu": True}, 1),
        (gpu.GpuDeviceUnavailableError(), [13], {"allow_no_gpu": True}, 0),
        (gpu.GpuDeviceUnavailableError(), [12], {"expect_no_device": True}, 0),
        (gpu.GpuRuntimeLoadError(), [12], {"expect_no_device": True}, 1),
        (gpu.GpuSupportNotBuiltError(), [], {"expect_no_device": True}, 1),
        (None, [12], {"expect_no_device": True}, 1),
    ],
)
def test_exit_codes(monkeypatch, outcome, installed, flags, expected):
    monkeypatch.setattr(gpu, "_installed_cuda_majors", lambda: installed)
    lines = []
    code = doctor._exit_code(
        outcome,
        allow_no_gpu=flags.get("allow_no_gpu", False),
        expect_no_device=flags.get("expect_no_device", False),
        write=lines.append,
    )
    assert code == expected
    assert lines[-1].startswith("Result: ")


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
    monkeypatch.setenv("CUDAQ_PULSE_TARGET", "transmon_generic:3")
    monkeypatch.setenv("CUDAQ_PULSE_CACHE_SIZE", "8")
    code, out = _run("--allow-no-gpu")
    assert code == 0
    assert "Process target: transmon_generic:3 (env CUDAQ_PULSE_TARGET=transmon_generic:3" in out
    assert "Kernel cache: 8 specializations per kernel (CUDAQ_PULSE_CACHE_SIZE=8)" in out


@pytest.mark.parametrize(
    ("name", "value"), [("CUDAQ_PULSE_TARGET", "nope"), ("CUDAQ_PULSE_CACHE_SIZE", "0")]
)
def test_invalid_kernel_settings_fail(monkeypatch, name, value):
    monkeypatch.setattr(doctor, "_diagnose_gpu", lambda write: gpu.GpuSupportNotBuiltError())
    monkeypatch.setenv(name, value)
    code, out = _run("--allow-no-gpu")
    assert code == 1
    assert "Result: FAILED, invalid kernel settings." in out


def test_system_lists_every_problem_of_a_device_file(tmp_path):
    path = tmp_path / "lab.json"
    path.write_text('{"target": "transmon_krinner_17q",\n "layout": [4, 99], "cooldown": "x"}\n')
    code, output = _run("--system", str(path))
    assert code == 1
    assert output.splitlines() == [
        f"{path}:2:16: error: layout[1]: carrier 99 is not in target "
        "'transmon_krinner_17q' (it has 0..16)",
        f"{path}:2:33: error: cooldown: 'x' is given without a calibration; add "
        '"calibration", a shipped name or a file path, or remove "cooldown"',
        f"{path}: 2 problems",
    ]


def test_system_json_report(tmp_path):
    import json

    path = tmp_path / "lab.json"
    path.write_text('{"target": "transmon_krinner_17q", "layout": [4, 99]}')
    code, output = _run("--system", str(path), "--json")
    assert code == 1
    report = json.loads(output)
    assert report["ok"] is False and report["device"] is None
    assert report["problems"] == [
        {
            "message": "carrier 99 is not in target 'transmon_krinner_17q' (it has 0..16)",
            "path": "layout[1]",
            "value": 99,
            "fix": "",
            "file": str(path),
            "line": 1,
            "column": 50,
        }
    ]
    path.write_text('{"target": "transmon_krinner_17q", "layout": [4, 5]}')
    code, output = _run("--system", str(path), "--json")
    assert code == 0
    assert json.loads(output) == {
        "file": str(path),
        "ok": True,
        "device": {"name": "transmon_krinner_17q", "qudits": 2, "layout": [4, 5]},
        "problems": [],
    }


def test_system_reports_a_good_or_unreadable_file(tmp_path):
    path = tmp_path / "lab.json"
    path.write_text('{"target": "transmon_krinner_17q", "layout": [4, 5]}')
    assert _run("--system", str(path)) == (
        0,
        f"{path}: ok, device transmon_krinner_17q with 2 qudits\n",
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
