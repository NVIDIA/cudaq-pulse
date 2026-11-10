# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import _build_config
from cudaq_pulse._native._cudaq_pulse_native import MLIRPipeline
from cudaq_pulse.quantum_systems.base import PhysicalQudit
from cudaq_pulse.runtime import gpu

from tests._handles import Q
from tests.runtime._lowering import evolve, on

compiler_only = pytest.mark.skipif(
    _build_config.GPU_RUNTIME_ENABLED, reason="checks the compiler-only configuration"
)


@pytest.fixture(autouse=True)
def _fresh_guard(monkeypatch):
    monkeypatch.delenv("CUDM_RUNTIME_LIB", raising=False)
    monkeypatch.setattr(gpu, "_runtime", None)


def test_error_classes_are_distinct():
    errors = (gpu.GpuSupportNotBuiltError, gpu.GpuRuntimeLoadError, gpu.GpuDeviceUnavailableError)
    for error in errors:
        assert issubclass(error, gpu.GpuRuntimeError)
        assert all(not issubclass(error, other) for other in errors if other is not error)


@compiler_only
def test_compiler_only_build_reports_missing_gpu_support():
    with pytest.raises(gpu.GpuSupportNotBuiltError, match=r"cudaq-pulse\[cu12\].*\[cu13\]"):
        gpu.require_gpu_runtime()


@compiler_only
def test_evolve_on_compiler_only_build_raises_not_built():

    @pulse.kernel
    def kernel(q0):
        d0 = get_drive_line(q0)
        drive(d0, gaussian(40, 0.3, 10.0))

    target = pulse.quantum_system(
        name="guard",
        qudits={
            0: PhysicalQudit(
                index=0, frequency_hz=5.0e9, anharmonicity_hz=-200.0e6, t1_us=0.0, t2_star_us=0.0
            )
        },
    )
    with pytest.raises(gpu.GpuSupportNotBuiltError):
        evolve(on(target, kernel, Q[0]), t_start=0.0, t_end=20.0, step_count=10)


def test_override_pointing_nowhere_is_a_load_error(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDM_RUNTIME_LIB", str(tmp_path / "missing.so"))
    with pytest.raises(gpu.GpuRuntimeLoadError, match="not a file"):
        gpu.require_gpu_runtime()


def test_unloadable_library_is_a_load_error(monkeypatch, tmp_path):
    bogus = tmp_path / "libcudm-runtime.so"
    bogus.write_bytes(b"not a shared library")
    monkeypatch.setenv("CUDM_RUNTIME_LIB", str(bogus))
    with pytest.raises(gpu.GpuRuntimeLoadError) as info:
        gpu.require_gpu_runtime()
    assert isinstance(info.value.__cause__, OSError)


def _fake_site(root, relatives):
    for relative in relatives:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


@pytest.mark.parametrize("major", [12, 13])
def test_cuda_wheel_libraries_are_found_in_load_order(tmp_path, major):
    # Namespace packages may be split across site directories.
    libraries = gpu._CUDA_WHEEL_LIBRARIES[major]
    nvidia = [r for r in libraries if r.startswith("nvidia/")]
    others = [r for r in libraries if r not in nvidia]
    _fake_site(tmp_path / "site_a", nvidia)
    _fake_site(tmp_path / "site_b", others)
    roots = {
        top: [tmp_path / "site_a" / top, tmp_path / "site_b" / top]
        for top in ("nvidia", "cutensor", "cuquantum")
    }

    found, missing = gpu._locate_cuda_wheel_libraries(libraries, lambda top: roots[top])

    assert missing == []
    assert [p.name for p in found] == [r.rsplit("/", 1)[1] for r in libraries]


@pytest.mark.parametrize(
    ("major", "cublas"),
    [
        (12, ["libcublasLt.so.12", "libcublas.so.12"]),
        (13, ["libcublasLt.so.13", "libcublas.so.13"]),
    ],
)
def test_cuda_wheel_libraries_report_what_is_not_installed(tmp_path, major, cublas):
    libraries = gpu._CUDA_WHEEL_LIBRARIES[major]
    present = [r for r in libraries if "cublas" not in r]
    _fake_site(tmp_path, present)

    found, missing = gpu._locate_cuda_wheel_libraries(libraries, lambda top: [tmp_path / top])

    assert missing == cublas
    assert len(found) == len(present)


def test_load_error_names_libraries_missing_from_wheels(tmp_path):
    bogus = tmp_path / "libcudm-runtime.so"
    bogus.write_bytes(b"not a shared library")
    with pytest.raises(gpu.GpuRuntimeLoadError, match="libcublas.so.12"):
        gpu._load(bogus, ["libcublas.so.12"])


def test_in_process_jit_executes_llvm_dialect():
    module = MLIRPipeline().jit_llvm_dialect(
        """
        llvm.func @add(%a: i64, %b: i64) -> i64 {
          %sum = llvm.add %a, %b : i64
          llvm.return %sum : i64
        }
        """,
        [],
    )
    add = ctypes.CFUNCTYPE(ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)(module.lookup("add"))
    assert add(40, 2) == 42


def test_in_process_jit_reports_unknown_symbols():
    module = MLIRPipeline().jit_llvm_dialect("llvm.func @noop() { llvm.return }", [])
    with pytest.raises(RuntimeError, match="missing_entry"):
        module.lookup("missing_entry")
