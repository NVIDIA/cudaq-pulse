# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""CUDA-major selection and driver checks of the GPU loader, without a GPU."""

import pytest
from cudaq_pulse import _build_config
from cudaq_pulse.runtime import gpu


@pytest.fixture(autouse=True)
def _fresh_guard(monkeypatch):
    monkeypatch.delenv("CUDM_RUNTIME_LIB", raising=False)
    monkeypatch.delenv("CUDAQ_PULSE_CUDA", raising=False)
    monkeypatch.setattr(gpu, "_runtime", None)


def _build(monkeypatch, libraries):
    monkeypatch.setattr(_build_config, "GPU_RUNTIME_ENABLED", bool(libraries))
    monkeypatch.setattr(_build_config, "GPU_RUNTIME_LIBRARIES", libraries)


def _install(monkeypatch, *majors):
    names = {f"cudensitymat-cu{major}" for major in majors}
    monkeypatch.setattr(gpu, "_distribution_installed", lambda name: name in names)


BOTH = {12: "lib/libcudm-runtime-cu12.so", 13: "lib/libcudm-runtime-cu13.so"}


@pytest.mark.parametrize("major", [12, 13])
def test_the_installed_extra_selects_its_runtime(monkeypatch, major):
    _build(monkeypatch, BOTH)
    _install(monkeypatch, major)
    assert gpu._select_cuda_major() == major


def test_no_installed_extra_names_both_extras(monkeypatch):
    _build(monkeypatch, BOTH)
    _install(monkeypatch)
    with pytest.raises(gpu.GpuRuntimeLoadError, match=r"\[cu12\].*\[cu13\]"):
        gpu._select_cuda_major()


def test_both_installed_extras_are_rejected(monkeypatch):
    _build(monkeypatch, BOTH)
    _install(monkeypatch, 12, 13)
    with pytest.raises(gpu.GpuRuntimeLoadError, match="cannot coexist"):
        gpu._select_cuda_major()


def test_installed_extra_without_a_matching_runtime(monkeypatch):
    _build(monkeypatch, {12: BOTH[12]})
    _install(monkeypatch, 13)
    with pytest.raises(gpu.GpuRuntimeLoadError, match=r"only contains runtimes for CUDA 12"):
        gpu._select_cuda_major()


def test_environment_forces_a_major(monkeypatch):
    _build(monkeypatch, BOTH)
    _install(monkeypatch, 12, 13)
    monkeypatch.setenv("CUDAQ_PULSE_CUDA", "13")
    assert gpu._select_cuda_major() == 13


@pytest.mark.parametrize("value", ["11", "cu12", "14"])
def test_environment_rejects_unknown_majors(monkeypatch, value):
    _build(monkeypatch, BOTH)
    monkeypatch.setenv("CUDAQ_PULSE_CUDA", value)
    with pytest.raises(gpu.GpuRuntimeLoadError, match="must be one of 12, 13"):
        gpu._select_cuda_major()


def test_environment_cannot_force_a_runtime_that_was_not_built(monkeypatch):
    _build(monkeypatch, {12: BOTH[12]})
    monkeypatch.setenv("CUDAQ_PULSE_CUDA", "13")
    with pytest.raises(gpu.GpuRuntimeLoadError, match="only contains runtimes for CUDA 12"):
        gpu._select_cuda_major()


def test_missing_bundled_runtime_names_the_major(monkeypatch, tmp_path):
    _build(monkeypatch, {13: str(tmp_path / "libcudm-runtime-cu13.so")})
    _install(monkeypatch, 13)
    with pytest.raises(gpu.GpuRuntimeLoadError, match="CUDA 13 runtime library is missing"):
        gpu.require_gpu_runtime()


def test_bundled_runtime_of_the_selected_major_is_resolved(monkeypatch, tmp_path):
    runtimes = {major: tmp_path / f"libcudm-runtime-cu{major}.so" for major in (12, 13)}
    for path in runtimes.values():
        path.write_bytes(b"")
    _build(monkeypatch, {major: str(path) for major, path in runtimes.items()})
    _install(monkeypatch, 13)
    assert gpu._resolve_runtime() == (13, runtimes[13])


def test_override_uses_the_installed_major_for_preloading(monkeypatch, tmp_path):
    override = tmp_path / "libcudm-runtime.so"
    override.write_bytes(b"")
    monkeypatch.setenv("CUDM_RUNTIME_LIB", str(override))
    _install(monkeypatch, 12)
    assert gpu._resolve_runtime() == (12, override)
    _install(monkeypatch, 12, 13)
    assert gpu._resolve_runtime() == (None, override)


def test_cuda13_libraries_share_one_wheel_directory():
    cu12, cu13 = gpu._CUDA_WHEEL_LIBRARIES[12], gpu._CUDA_WHEEL_LIBRARIES[13]
    assert all(r.startswith("nvidia/cu13/lib/") for r in cu13 if r.startswith("nvidia/"))
    assert [r for r in cu12 if not r.startswith("nvidia/")] == [
        r for r in cu13 if not r.startswith("nvidia/")
    ]


def test_missing_driver_is_a_device_error(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu, "_driver_version", lambda: None)
    with pytest.raises(gpu.GpuDeviceUnavailableError, match="no NVIDIA driver"):
        gpu._check_driver(12, tmp_path)


def test_old_driver_names_the_required_version(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu, "_driver_version", lambda: 12080)
    with pytest.raises(gpu.GpuDeviceUnavailableError) as info:
        gpu._check_driver(13, tmp_path)
    message = str(info.value)
    assert "CUDA up to 12.8" in message
    assert "driver 580 or newer" in message
    assert "cudaq-pulse[cu12]" in message


def test_driver_older_than_every_major(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu, "_driver_version", lambda: 11080)
    with pytest.raises(gpu.GpuDeviceUnavailableError, match=r"525\.60\.13 or newer \(CUDA 12\)\."):
        gpu._check_driver(12, tmp_path)


@pytest.mark.parametrize(("major", "version"), [(12, 12000), (12, 13020), (13, 13000)])
def test_new_enough_driver_passes(monkeypatch, tmp_path, major, version):
    monkeypatch.setattr(gpu, "_driver_version", lambda: version)
    gpu._check_driver(major, tmp_path)


class _VersionedRuntime:
    def __init__(self, version):
        self.version = version

    def cudm_runtime_version(self):
        return self.version


@pytest.mark.parametrize(
    ("built", "loaded"), [("0.7.0", 700), ("0.7.0", 702), ("1.2.0", 10300), ("", 500)]
)
def test_a_cudensitymat_with_the_built_abi_is_accepted(monkeypatch, tmp_path, built, loaded):
    monkeypatch.setattr(_build_config, "CUDENSITYMAT_VERSIONS", {12: built})
    gpu._check_cudensitymat_version(_VersionedRuntime(loaded), tmp_path, 12)


@pytest.mark.parametrize(("built", "loaded"), [("0.7.0", 600), ("0.7.0", 800), ("1.2.0", 20000)])
def test_a_cudensitymat_with_another_abi_is_refused(monkeypatch, tmp_path, built, loaded):
    monkeypatch.setattr(_build_config, "CUDENSITYMAT_VERSIONS", {12: built})
    with pytest.raises(gpu.GpuRuntimeLoadError, match=f"built against cuDensityMat {built}"):
        gpu._check_cudensitymat_version(_VersionedRuntime(loaded), tmp_path, 12)
