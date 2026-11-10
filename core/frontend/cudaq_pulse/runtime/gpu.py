# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single gate for every GPU execution path.

Whether the GPU runtime exists is recorded by CMake in ``_build_config``; it
is never discovered by searching the filesystem. ``require_gpu_runtime()``
either returns a loaded, device-backed runtime or raises one of three
distinct errors, so "built without GPU support", "runtime failed to load",
and "no usable device" are never confused.

A build carries one runtime per CUDA major. The one used is chosen by which
``cudensitymat-cuNN`` distribution the ``[cu12]`` or ``[cu13]`` extra
installed. Its CUDA libraries live in separate wheels whose directories are
not all on cuDensityMat's RUNPATH, so they are loaded from those wheels
first, in dependency order. Everything is loaded RTLD_LOCAL: a later
library's dependencies resolve by soname to the copies already loaded,
while nothing joins the process's global symbol scope, where it could
interpose on another CUDA stack in the same process (another framework's
cuBLAS, say).
"""

from __future__ import annotations

import ctypes
import importlib.metadata
import importlib.util
import os
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .. import _build_config
from .._errors import ExecutionError

__all__ = [
    "GpuDeviceUnavailableError",
    "GpuRuntime",
    "GpuRuntimeError",
    "GpuRuntimeLoadError",
    "GpuSupportNotBuiltError",
    "require_gpu_runtime",
]

_OVERRIDE_ENV = "CUDM_RUNTIME_LIB"
_SELECT_ENV = "CUDAQ_PULSE_CUDA"

_SUPPORTED_MAJORS = (12, 13)

# Wheel-relative paths per CUDA major, ordered so that each library's
# DT_NEEDED entries are already resident when it loads.
_CUDA_WHEEL_LIBRARIES: dict[int, tuple[str, ...]] = {
    12: (
        "nvidia/nvjitlink/lib/libnvJitLink.so.12",
        "nvidia/cuda_runtime/lib/libcudart.so.12",
        "nvidia/cublas/lib/libcublasLt.so.12",
        "nvidia/cublas/lib/libcublas.so.12",
        "nvidia/cusparse/lib/libcusparse.so.12",
        "nvidia/cusolver/lib/libcusolver.so.11",
        "nvidia/curand/lib/libcurand.so.10",
        "cutensor/lib/libcutensor.so.2",
        "cuquantum/lib/libcutensornet.so.2",
        "cuquantum/lib/libcudensitymat.so.0",
    ),
    13: (
        "nvidia/cu13/lib/libnvJitLink.so.13",
        "nvidia/cu13/lib/libcudart.so.13",
        "nvidia/cu13/lib/libcublasLt.so.13",
        "nvidia/cu13/lib/libcublas.so.13",
        "nvidia/cu13/lib/libcusparse.so.12",
        "nvidia/cu13/lib/libcusolver.so.12",
        "nvidia/cu13/lib/libcurand.so.10",
        "cutensor/lib/libcutensor.so.2",
        "cuquantum/lib/libcutensornet.so.2",
        "cuquantum/lib/libcudensitymat.so.0",
    ),
}

# Oldest Linux driver for each CUDA major, as (cuDriverGetVersion value,
# driver release), under CUDA minor-version compatibility.
_MINIMUM_DRIVER: dict[int, tuple[int, str]] = {
    12: (12000, "525.60.13"),
    13: (13000, "580"),
}


class GpuRuntimeError(ExecutionError):
    """Base class for every reason GPU execution cannot proceed."""


class GpuSupportNotBuiltError(GpuRuntimeError):
    """This cudaq-pulse build does not contain the GPU runtime."""


class GpuRuntimeLoadError(GpuRuntimeError):
    """The GPU runtime is part of this build but could not be loaded."""


class GpuDeviceUnavailableError(GpuRuntimeError):
    """The GPU runtime loaded, but no usable NVIDIA device or driver exists."""


@dataclass(frozen=True)
class GpuRuntime:
    """A loaded cuDensityMat runtime with at least one usable device."""

    library: ctypes.CDLL
    path: Path
    device_count: int
    cuda_major: int | None = None


_lock = threading.Lock()
_runtime: GpuRuntime | None = None
_forked_after_cuda_init = False


def _after_fork_in_child() -> None:
    """Reset the guard in a forked child.

    The lock may have been held by a thread that does not exist in the
    child. A runtime loaded by the parent has initialized CUDA, which a
    forked child cannot use, so it is dropped and the child refuses GPU
    work instead of failing inside the driver.
    """
    global _lock, _runtime, _forked_after_cuda_init
    _lock = threading.Lock()
    if _runtime is not None:
        _runtime = None
        _forked_after_cuda_init = True


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


def require_gpu_runtime() -> GpuRuntime:
    """Return the loaded GPU runtime, or raise a specific ``GpuRuntimeError``.

    A process forked after its parent used the GPU cannot use it: start GPU
    workers with the ``spawn`` or ``forkserver`` multiprocessing method, or
    fork before the first GPU simulation.
    """
    global _runtime
    with _lock:
        if _forked_after_cuda_init:
            raise GpuDeviceUnavailableError(
                "This process was forked from one that had already initialized "
                "CUDA, and CUDA cannot be used in a forked child. Start GPU "
                "workers with multiprocessing's 'spawn' or 'forkserver' method "
                "(multiprocessing.get_context('spawn')), or fork before the "
                "first GPU simulation."
            )
        if _runtime is None:
            major, path = _resolve_runtime()
            not_in_wheels: list[str] = []
            if major is not None and sys.platform.startswith("linux"):
                not_in_wheels = _preload_cuda_wheels(major)
            library = _load(path, not_in_wheels, major)
            if major is not None:
                _check_cudensitymat_version(library, path, major)
                _check_driver(major, path)
            _runtime = GpuRuntime(
                library=library,
                path=path,
                device_count=_device_count(library, path),
                cuda_major=major,
            )
        return _runtime


def _extra_hint(majors: Sequence[int]) -> str:
    return " or ".join(f'`pip install "cudaq-pulse[cu{major}]"`' for major in majors)


def _distribution_installed(name: str) -> bool:
    try:
        importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _installed_cuda_majors() -> list[int]:
    """CUDA majors whose ``cudensitymat-cuNN`` distribution is installed."""
    return [m for m in _SUPPORTED_MAJORS if _distribution_installed(f"cudensitymat-cu{m}")]


def _forced_major() -> int | None:
    forced = os.environ.get(_SELECT_ENV)
    if not forced:
        return None
    try:
        major = int(forced)
    except ValueError:
        major = -1
    if major not in _SUPPORTED_MAJORS:
        raise GpuRuntimeLoadError(
            f"{_SELECT_ENV} is set to {forced!r}; it must be one of "
            f"{', '.join(map(str, _SUPPORTED_MAJORS))}."
        )
    return major


def _select_cuda_major() -> int:
    """Pick the bundled runtime that matches the installed CUDA extra."""
    built = sorted(_build_config.GPU_RUNTIME_LIBRARIES)
    forced = _forced_major()
    if forced is not None:
        if forced not in built:
            raise GpuRuntimeLoadError(
                f"{_SELECT_ENV}={forced} selects the CUDA {forced} runtime, but this "
                f"cudaq-pulse build only contains runtimes for CUDA "
                f"{', '.join(map(str, built))}."
            )
        return forced

    installed = _installed_cuda_majors()
    if not installed:
        raise GpuRuntimeLoadError(
            "cudaq-pulse includes GPU support, but no CUDA libraries are "
            f"installed for it. Install them with {_extra_hint(built)}."
        )
    if len(installed) > 1:
        names = ", ".join(f"cudensitymat-cu{m}" for m in installed)
        raise GpuRuntimeLoadError(
            f"Both the CUDA 12 and CUDA 13 GPU libraries are installed ({names}). "
            "Their wheels write the same cuquantum/lib and cutensor/lib files, so "
            "they cannot coexist: use a fresh environment with only one of "
            f"{_extra_hint(_SUPPORTED_MAJORS)}."
        )
    major = installed[0]
    if major not in built:
        raise GpuRuntimeLoadError(
            f"cudensitymat-cu{major} is installed, but this cudaq-pulse build only "
            f"contains runtimes for CUDA {', '.join(map(str, built))}. Install "
            f"{_extra_hint(built)} instead."
        )
    return major


def _resolve_runtime() -> tuple[int | None, Path]:
    """Return the CUDA major (``None`` if unknown) and the runtime to load."""
    override = os.environ.get(_OVERRIDE_ENV)
    if override:
        path = Path(override)
        if not path.is_file():
            raise GpuRuntimeLoadError(
                f"{_OVERRIDE_ENV} is set to {override!r}, which is not a file."
            )
        major = _forced_major()
        if major is None:
            installed = _installed_cuda_majors()
            major = installed[0] if len(installed) == 1 else None
        return major, path
    if not _build_config.GPU_RUNTIME_ENABLED:
        raise GpuSupportNotBuiltError(
            "This cudaq-pulse build does not include GPU support, so it can "
            "compile pulse programs but not execute them. Install a Linux wheel "
            'with GPU support via `pip install "cudaq-pulse[cu12]"` or '
            '`pip install "cudaq-pulse[cu13]"`, or rebuild from source with '
            "-DCUDAQ_PULSE_ENABLE_GPU=ON."
        )
    major = _select_cuda_major()
    # Not resolve(): a source build stages the Python sources as symlinks back
    # into core/frontend, and resolving them looks for the runtime library
    # there instead of in the staged package that holds it.
    path = Path(__file__).parent.parent / _build_config.GPU_RUNTIME_LIBRARIES[major]
    if not path.is_file():
        raise GpuRuntimeLoadError(
            f"cudaq-pulse was built with GPU support, but its CUDA {major} "
            f"runtime library is missing from {path}. The installation is "
            f"incomplete; reinstall cudaq-pulse."
        )
    return major, path


def _locate_cuda_wheel_libraries(
    libraries: Sequence[str],
    package_roots: Callable[[str], Sequence[Path]],
) -> tuple[list[Path], list[str]]:
    """Split wheel-relative ``libraries`` into files found under the installed
    packages (in load order) and the names of those that are not installed."""
    found: list[Path] = []
    missing: list[str] = []
    for relative in libraries:
        top, _, rest = relative.partition("/")
        candidates = [root / rest for root in package_roots(top)]
        match = next((c for c in candidates if c.is_file()), None)
        if match is None:
            missing.append(Path(relative).name)
        else:
            found.append(match)
    return found, missing


def _installed_package_roots(top: str) -> list[Path]:
    spec = importlib.util.find_spec(top)
    if spec is None or spec.submodule_search_locations is None:
        return []
    return [Path(location) for location in spec.submodule_search_locations]


def _preload_cuda_wheels(major: int) -> list[str]:
    found, missing = _locate_cuda_wheel_libraries(
        _CUDA_WHEEL_LIBRARIES[major], _installed_package_roots
    )
    for library in found:
        try:
            ctypes.CDLL(str(library), mode=ctypes.RTLD_LOCAL)
        except OSError as error:
            raise GpuRuntimeLoadError(
                f"cudaq-pulse could not load the CUDA {major} library {library}: "
                f"{error}. Reinstall with {_extra_hint([major])}."
            ) from error
    return missing


def _declare_prototypes(library: ctypes.CDLL) -> None:
    """Set the ctypes prototypes of the runtime's C API once, at load.

    The library object is shared by every thread, so the prototypes are not
    reassigned per call.
    """
    library.cudm_runtime_version.argtypes = []
    library.cudm_runtime_version.restype = ctypes.c_int64
    library.cudm_set_initial_state.argtypes = [ctypes.c_void_p, ctypes.c_int64]
    library.cudm_set_initial_state.restype = ctypes.c_int
    library.cudm_last_result_size.argtypes = []
    library.cudm_last_result_size.restype = ctypes.c_int64
    library.cudm_last_result_copy.argtypes = [ctypes.c_void_p, ctypes.c_int64]
    library.cudm_last_result_copy.restype = ctypes.c_int
    library.cudm_last_error_message.argtypes = []
    library.cudm_last_error_message.restype = ctypes.c_char_p


def _load(path: Path, not_in_wheels: Sequence[str], major: int | None = None) -> ctypes.CDLL:
    try:
        library = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
    except OSError as error:
        detail = ""
        if not_in_wheels:
            detail = (
                f" These CUDA libraries are not installed as Python "
                f"packages, so they had to come from the system library "
                f"path: {', '.join(not_in_wheels)}."
            )
        majors = [major] if major is not None else sorted(_build_config.GPU_RUNTIME_LIBRARIES)
        hint = _extra_hint(majors) if majors else _extra_hint(_SUPPORTED_MAJORS)
        raise GpuRuntimeLoadError(
            f"cudaq-pulse could not load its GPU runtime {path}: {error}.{detail} "
            f"Install the GPU dependencies with {hint}."
        ) from error
    try:
        _declare_prototypes(library)
    except AttributeError as error:
        raise GpuRuntimeLoadError(
            f"{path} is not a cudaq-pulse GPU runtime: {error}. Reinstall cudaq-pulse."
        ) from error
    return library


def _abi_version(major: int, minor: int) -> tuple[int, ...]:
    """The part of a cuDensityMat version that fixes its ABI: before 1.0 the
    minor version, afterwards the major version alone."""
    return (major, minor) if major == 0 else (major,)


def _check_cudensitymat_version(library: ctypes.CDLL, path: Path, major: int) -> None:
    """Refuse a cuDensityMat whose ABI differs from the one the runtime was built with."""
    built = _build_config.CUDENSITYMAT_VERSIONS.get(major, "")
    try:
        built_major, built_minor = (int(part) for part in built.split(".")[:2])
    except ValueError:
        return
    version = int(library.cudm_runtime_version())
    loaded = (version // 10000, version // 100 % 100, version % 100)
    if _abi_version(*loaded[:2]) != _abi_version(built_major, built_minor):
        raise GpuRuntimeLoadError(
            f"The cudaq-pulse CUDA {major} runtime ({path}) was built against "
            f"cuDensityMat {built}, but cuDensityMat {'.'.join(map(str, loaded))} "
            f"is installed, and its ABI differs. Reinstall with {_extra_hint([major])}."
        )


def _driver_version() -> int | None:
    """``cuDriverGetVersion`` from the driver's libcuda, or ``None`` without one."""
    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return None
    get_version = libcuda.cuDriverGetVersion
    get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get_version.restype = ctypes.c_int
    version = ctypes.c_int(0)
    if get_version(ctypes.byref(version)) != 0:
        return None
    return version.value


def _format_cuda_version(version: int) -> str:
    return f"{version // 1000}.{version % 1000 // 10}"


def _check_driver(major: int, path: Path) -> None:
    version = _driver_version()
    minimum, release = _MINIMUM_DRIVER[major]
    if version is None:
        raise GpuDeviceUnavailableError(
            f"The cudaq-pulse CUDA {major} runtime ({path}) loaded, but no NVIDIA "
            f"driver was found (libcuda.so.1 is not loadable). Install NVIDIA "
            f"driver {release} or newer."
        )
    if version < minimum:
        older = [m for m in _SUPPORTED_MAJORS if m < major and _MINIMUM_DRIVER[m][0] <= version]
        alternative = f", or use {_extra_hint(older)} instead" if older else ""
        raise GpuDeviceUnavailableError(
            f"The installed NVIDIA driver supports CUDA up to "
            f"{_format_cuda_version(version)}, but cudaq-pulse[cu{major}] needs "
            f"driver {release} or newer (CUDA {major}). Upgrade the driver{alternative}."
        )


def _device_count(library: ctypes.CDLL, path: Path) -> int:
    get_count = library.cudaGetDeviceCount
    get_count.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get_count.restype = ctypes.c_int
    error_string = library.cudaGetErrorString
    error_string.argtypes = [ctypes.c_int]
    error_string.restype = ctypes.c_char_p

    count = ctypes.c_int(0)
    status = get_count(ctypes.byref(count))
    if status != 0:
        raw = error_string(status)
        reason = raw.decode("utf-8", errors="replace") if raw else "unknown"
        raise GpuDeviceUnavailableError(
            f"The cudaq-pulse GPU runtime ({path}) loaded, but CUDA reports no "
            f"usable device: cudaGetDeviceCount returned error {status} "
            f"({reason}). Check the NVIDIA driver and CUDA_VISIBLE_DEVICES."
        )
    if count.value == 0:
        raise GpuDeviceUnavailableError(
            f"The cudaq-pulse GPU runtime ({path}) loaded, but CUDA reports "
            f"zero devices. Check the NVIDIA driver and CUDA_VISIBLE_DEVICES."
        )
    return count.value
