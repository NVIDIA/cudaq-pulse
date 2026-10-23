# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Report what this cudaq-pulse installation contains and whether it works.

    python -m cudaq_pulse.doctor                     # exit 0 only if GPU execution works
    python -m cudaq_pulse.doctor --allow-no-gpu      # exit 0 if the compiler works
    python -m cudaq_pulse.doctor --expect-no-device  # exit 0 if GPU libraries load
                                                     # but no device exists

The compiler check always runs; any failure there exits non-zero.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import os
import platform
import sys
from collections.abc import Callable, Sequence
from typing import TextIO

from . import _build_config
from .runtime import gpu

_DISTRIBUTION = "cudaq-pulse"


def _version() -> str:
    try:
        return importlib.metadata.version(_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        from . import __version__

        return f"{__version__} (not installed as a distribution; source or build tree)"


def _wheel_tags() -> str:
    try:
        wheel = importlib.metadata.distribution(_DISTRIBUTION).read_text("WHEEL")
    except importlib.metadata.PackageNotFoundError:
        wheel = None
    if not wheel:
        return "none (not installed from a wheel)"
    tags = [line.split(":", 1)[1].strip() for line in wheel.splitlines() if line.startswith("Tag:")]
    return ", ".join(tags) or "unknown"


def _check_compiler() -> str | None:
    """JIT a trivial LLVM-dialect module; return an error message on failure."""
    try:
        from ._native._cudaq_pulse_native import MLIRPipeline

        module = MLIRPipeline().jit_llvm_dialect(
            "llvm.func @answer() -> i64 {\n"
            "  %0 = llvm.mlir.constant(42 : i64) : i64\n"
            "  llvm.return %0 : i64\n"
            "}",
            [],
        )
        answer = ctypes.CFUNCTYPE(ctypes.c_int64)(module.lookup("answer"))()
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    if answer != 42:
        return f"the JIT-compiled check returned {answer}, expected 42"
    return None


def _context_budget(write: Callable[[str], None]) -> bool:
    """Print the shared MLIR context limits; return False if they are invalid."""
    from ._native import _cudaq_pulse_native as native

    try:
        modules, words = native.shared_context_budget()
    except ValueError as error:
        write("MLIR context reuse: INVALID")
        write(f"  {error}")
        return False
    sources = [
        f"{name}={os.environ[name]}"
        for name in ("CUDAQ_PULSE_CONTEXT_MAX_MODULES", "CUDAQ_PULSE_CONTEXT_MAX_WORDS")
        if os.environ.get(name)
    ]
    write(
        f"MLIR context reuse: up to {modules} modules or {words} packed words per "
        f"context ({', '.join(sources) if sources else 'defaults'})"
    )
    return True


def _kernel_settings(write: Callable[[str], None]) -> bool:
    """Print the process target and cache size; return False if either is invalid."""
    from .kernel._kernel import CACHE_SIZE_ENV, _cache_size
    from .targets._registry import ENV_VAR, available_targets, current_target

    ok = True
    write(f"Targets available: {', '.join(available_targets())}")
    if not os.environ.get(ENV_VAR):
        write(f"Process target: none ({ENV_VAR} is not set; kernels run on their qudits' device)")
    else:
        try:
            device = current_target()
        except Exception as error:
            write("Process target: INVALID")
            write(f"  {error}")
            ok = False
        else:
            write(f"Process target: {device.name} ({device.source}, {len(device.layout)} qudits)")
    try:
        size = _cache_size(None)
    except ValueError as error:
        write("Kernel cache: INVALID")
        write(f"  {error}")
        return False
    source = (
        f"{CACHE_SIZE_ENV}={os.environ[CACHE_SIZE_ENV]}"
        if os.environ.get(CACHE_SIZE_ENV)
        else "default"
    )
    write(f"Kernel cache: {size} specializations per kernel ({source})")
    return ok


def _installed_extras() -> list[str]:
    installed = []
    for major in gpu._SUPPORTED_MAJORS:
        name = f"cudensitymat-cu{major}"
        try:
            installed.append(f"[cu{major}] ({name} {importlib.metadata.version(name)})")
        except importlib.metadata.PackageNotFoundError:
            pass
    return installed


def _report_libraries(major: int, write: Callable[[str], None]) -> None:
    write(f"  CUDA {major} libraries:")
    libraries = gpu._CUDA_WHEEL_LIBRARIES[major]
    found, _ = gpu._locate_cuda_wheel_libraries(libraries, gpu._installed_package_roots)
    by_name = {path.name: path for path in found}
    for relative in libraries:
        name = relative.rsplit("/", 1)[1]
        path = by_name.get(name)
        if path is None:
            write(f"    {name:<24} not installed as a Python package")
            continue
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        except OSError as error:
            write(f"    {name:<24} FAILED: {error}")
        else:
            write(f"    {name:<24} {path}")


def _diagnose_gpu(write: Callable[[str], None]) -> gpu.GpuRuntimeError | None:
    """Print the GPU path step by step; return its outcome (``None`` if usable)."""
    built = sorted(_build_config.GPU_RUNTIME_LIBRARIES)
    write(f"GPU runtime built in: {'yes' if _build_config.GPU_RUNTIME_ENABLED else 'no'}")
    for major in built:
        write(
            f"  CUDA {major}: {_build_config.GPU_RUNTIME_LIBRARIES[major]} "
            f"(CUDA {_build_config.CUDA_VERSIONS.get(major, '?')}, "
            f"cuDensityMat {_build_config.CUDENSITYMAT_VERSIONS.get(major, '?')})"
        )
    extras = _installed_extras()
    write(f"CUDA extra installed: {', '.join(extras) if extras else 'none'}")

    if _build_config.GPU_RUNTIME_ENABLED and sys.platform.startswith("linux"):
        try:
            major = gpu._select_cuda_major()
        except gpu.GpuRuntimeError:
            pass
        else:
            _report_libraries(major, write)

    if sys.platform.startswith("linux"):
        driver = gpu._driver_version()
        write(
            "NVIDIA driver: "
            + (
                f"supports CUDA {gpu._format_cuda_version(driver)}"
                if driver is not None
                else "not found (libcuda.so.1 is not loadable)"
            )
        )

    try:
        runtime = gpu.require_gpu_runtime()
    except gpu.GpuRuntimeError as error:
        write(f"GPU execution: unavailable ({type(error).__name__})")
        write(f"  {error}")
        return error
    write(f"GPU execution: ready, CUDA {runtime.cuda_major} runtime {runtime.path}")
    write(f"  devices: {runtime.device_count}")
    return None


def _exit_code(
    outcome: gpu.GpuRuntimeError | None,
    *,
    allow_no_gpu: bool,
    expect_no_device: bool,
    write: Callable[[str], None],
) -> int:
    if expect_no_device:
        if isinstance(outcome, gpu.GpuDeviceUnavailableError):
            write("Result: OK, every GPU library loads and, as expected, no device is usable.")
            return 0
        if outcome is None:
            write("Result: FAILED, --expect-no-device was given but a GPU device is usable.")
        else:
            write(
                "Result: FAILED, the GPU libraries did not load (expected only a missing device)."
            )
        return 1
    if outcome is None:
        write("Result: OK, compilation and GPU execution work.")
        return 0
    if allow_no_gpu and (
        isinstance(outcome, (gpu.GpuSupportNotBuiltError, gpu.GpuDeviceUnavailableError))
        or not gpu._installed_cuda_majors()
    ):
        write("Result: OK, compilation works; GPU execution is unavailable (allowed).")
        return 0
    write("Result: FAILED, GPU execution is unavailable.")
    return 1


def main(argv: Sequence[str] | None = None, out: TextIO | None = None) -> int:
    """Run the diagnostics and return the process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m cudaq_pulse.doctor",
        description="Report what this cudaq-pulse installation contains and whether it works.",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--allow-no-gpu",
        action="store_true",
        help="succeed when the compiler works, even if GPU execution is unavailable "
        "because the build is compiler-only, no CUDA extra is installed, or no device exists",
    )
    modes.add_argument(
        "--expect-no-device",
        action="store_true",
        help="succeed only if every GPU library loads and the sole problem is that "
        "no usable GPU device exists",
    )
    args = parser.parse_args(argv)
    stream = out if out is not None else sys.stdout

    def write(line: str) -> None:
        print(line, file=stream)

    write(f"cudaq-pulse {_version()}")
    write(
        f"Python {platform.python_version()} ({sys.implementation.name}) on "
        f"{platform.system()} {platform.machine()}"
    )
    write(f"Wheel tags: {_wheel_tags()}")

    compiler_error = _check_compiler()
    write(f"Compiler (MLIR JIT): {'ok' if compiler_error is None else 'FAILED'}")
    if compiler_error is not None:
        write(f"  {compiler_error}")
    if not _context_budget(write) and compiler_error is None:
        compiler_error = "invalid MLIR context limits"
    if not _kernel_settings(write) and compiler_error is None:
        compiler_error = "invalid kernel settings"

    outcome = _diagnose_gpu(write)
    if compiler_error is not None:
        settings = compiler_error.startswith("invalid ")
        write(f"Result: FAILED, {compiler_error if settings else 'the compiler does not work'}.")
        return 1
    return _exit_code(
        outcome,
        allow_no_gpu=args.allow_no_gpu,
        expect_no_device=args.expect_no_device,
        write=write,
    )


if __name__ == "__main__":
    sys.exit(main())
