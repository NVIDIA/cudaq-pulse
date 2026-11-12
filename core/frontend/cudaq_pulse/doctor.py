# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report what this cudaq-pulse installation contains and whether it works.

    python -m cudaq_pulse.doctor                     # exit 0 only if GPU execution works
    python -m cudaq_pulse.doctor --allow-no-gpu      # exit 0 if the compiler works
    python -m cudaq_pulse.doctor --expect-no-device  # exit 0 if GPU libraries load
                                                     # but no quantum system exists
    python -m cudaq_pulse.doctor --setup-editor      # configure mypy and ruff for
                                                     # the kernel language
    python -m cudaq_pulse.doctor --system lab.json   # check a quantum system file
    python -m cudaq_pulse.doctor --system lab.json --json

The compiler check always runs; any failure there exits non-zero. So does
one of the cudaq-pulse-opt MLIR driver, from the cudaq-pulse-tools
distribution, which an installed cudaq-pulse requires at its own version,
and whose ``--version`` must print it.
``--system FILE`` instead runs the checks of ``pulse.QuantumSystem.load(FILE)`` and
prints each problem as ``FILE:LINE:COLUMN: error: PATH: MESSAGE``, or with
``--json`` as one JSON object; it exits 1 if the file has a problem.
``--setup-editor`` instead shows the change to the mypy and ruff
configuration of the project in the current directory (or ``--project``)
and writes it only when confirmed, or with ``--yes``. ``--system`` loads a
system file (``QuantumSystem.as_dict()``), lists every problem of its definition
with its line, and shows its controllers, links and whether each
controller's backend is installed; it exits non-zero if the file has a
problem or names a backend that is not installed.
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import importlib.metadata
import os
import platform
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TextIO

from . import _build_config
from ._errors import UNSET, Problem, QuantumSystemDefinitionError
from .runtime import gpu

__all__ = ["main"]

_DISTRIBUTION = "cudaq-pulse"


def _distribution(name: str) -> importlib.metadata.Distribution | None:
    try:
        return importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _check_driver(write: Callable[[str], None], version: str | None) -> str | None:
    """Locate and run cudaq-pulse-opt; return an error message on failure.

    An installed cudaq-pulse needs the cudaq-pulse-tools distribution of its
    version (pip installs it unless told ``--no-deps``); a source or build
    tree is only reported.
    """
    import shutil

    command, tools = "cudaq-pulse-opt", "cudaq-pulse-tools"
    heading = f"MLIR driver ({command})"
    on_path = shutil.which(command)
    if version is None:
        write(f"{heading}: {on_path or 'not on PATH'} (source or build tree; not checked)")
        return None
    install = f"pip install {tools}=={version}"
    driver = _distribution(tools)
    error = None
    executable = None
    if driver is None:
        error = (
            f"{tools} is not installed, and cudaq-pulse {version} requires it "
            f"(installed with --no-deps?): {install}"
        )
    elif driver.version != version:
        error = (
            f"{tools} {driver.version} is installed, and cudaq-pulse {version} "
            f"requires {version}: {install}"
        )
    else:
        files = [f for f in driver.files or () if f.name == command]
        executable = Path(str(driver.locate_file(files[0]))).resolve() if files else None
        if executable is None or not executable.is_file():
            error = (
                f"the {tools} {version} distribution has no {command} executable: "
                f"pip install --force-reinstall {tools}=={version}"
            )
    if error is not None or executable is None:
        error = error or f"{command} has no executable"
        write(f"{heading}: FAILED")
        write(f"  {error}")
        return error
    import re
    import subprocess

    failure: str | None = None
    llvm = ""
    try:
        result = subprocess.run(
            [str(executable), "--version"], capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        failure = f"{executable} --version failed: {exc}"
    else:
        output = result.stdout.strip()
        printed = re.fullmatch(rf"{command} (\S+) \(LLVM (\S+)\)", output)
        if result.returncode != 0:
            output = result.stderr.strip() or output
            failure = f"{executable} --version exited with {result.returncode}: {output}"
        elif printed is None:
            failure = (
                f"{executable} --version printed {output!r}, not "
                f"'{command} <version> (LLVM <version>)'"
            )
        elif printed.group(1) != version:
            failure = (
                f"{executable} reports version {printed.group(1)}, but its distribution is "
                f"{version}: pip install --force-reinstall {tools}=={version}"
            )
        else:
            llvm = printed.group(2)
    if failure is not None:
        write(f"{heading}: FAILED")
        write(f"  {failure}")
        return failure
    write(f"{heading}: {version} (LLVM {llvm}) at {executable}")
    if on_path is None:
        write(f"  not on PATH; add {executable.parent} to PATH to run it by name")
    elif Path(on_path).resolve() != executable:
        write(f"  PATH runs another {command} first: {on_path}")
    return None


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
    """Print the process quantum system and cache size; return False if either is invalid."""
    from .kernel._kernel import CACHE_SIZE_ENV, _cache_size
    from .quantum_systems._registry import (
        ENV_VAR,
        available_quantum_systems,
        current_quantum_system,
    )

    ok = True
    write(f"Quantum systems available: {', '.join(available_quantum_systems())}")
    if not os.environ.get(ENV_VAR):
        write(
            f"Process quantum system: none ({ENV_VAR} is not set; "
            "kernels run on their qudits' quantum system)"
        )
    else:
        try:
            quantum_system = current_quantum_system()
        except Exception as error:
            write("Process quantum system: INVALID")
            write(f"  {error}")
            ok = False
        else:
            write(
                f"Process quantum system: {quantum_system.name} "
                f"({quantum_system.source}, {len(quantum_system.layout)} qudits)"
            )
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
            ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
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


def _problem_line(problem: Problem) -> str:
    text = str(dataclasses.replace(problem, file=None))
    if problem.file is None:
        return f"error: {text}"
    where = [problem.file]
    if problem.line is not None:
        where.append(str(problem.line))
        if problem.column is not None:
            where.append(str(problem.column))
    return f"{':'.join(where)}: error: {text}"


def _problem_json(problem: Problem) -> dict[str, Any]:
    import json

    fields = dataclasses.asdict(problem)
    value = fields.pop("value")
    if value is not UNSET:
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            value = repr(value)
        fields["value"] = value
    return fields


def _check_system(file: Path, as_json: bool, write: Callable[[str], None]) -> int:
    """Check the quantum system or system file *file*, and that the backends of its
    controllers are installed; return the exit code."""
    from ._device import QuantumSystem

    problems: tuple[Problem, ...] = ()
    summary: dict[str, Any] | None = None
    try:
        dev = QuantumSystem.load(file)
    except QuantumSystemDefinitionError as error:
        problems = error.problems
    except OSError as error:
        problems = (Problem(f"cannot read the file: {error.strerror or error}", file=str(file)),)
    else:
        summary = {"name": dev.name, "qudits": len(dev.layout), "layout": list(dev.layout)}
        if dev.control_system is not None:
            summary["control_system"], problems = _control_summary(dev.control_system, file)
    if as_json:
        import json

        report = {
            "file": str(file),
            "ok": not problems,
            "quantum_system": summary,
            "problems": [_problem_json(p) for p in problems],
        }
        write(json.dumps(report, indent=2))
        return 1 if problems else 0
    if summary is not None:
        state = "ok" if not problems else "loads"
        write(f"{file}: {state}, quantum system {summary['name']} with {summary['qudits']} qudits")
        for line in _control_lines(summary.get("control_system")):
            write(f"  {line}")
    for problem in problems:
        write(_problem_line(problem))
    if problems:
        write(f"{file}: {len(problems)} problem{'' if len(problems) == 1 else 's'}")
    return 1 if problems else 0


def _control_summary(system: Any, file: Path) -> tuple[dict[str, Any], tuple[Problem, ...]]:
    """The controllers and links of *system* as data, and a problem for each
    backend of a controller that is not installed."""
    from . import backends

    installed = set(backends.available())
    controllers = []
    missing = []
    for index, controller in enumerate(system.controllers):
        entry: dict[str, Any] = {
            "model": controller.model,
            "version": controller.version,
            "drive_lines": [q for line in controller.drive_lines for q in line.qudits]
            if system._shares
            else [line[0] for line in controller.drive_lines],
            "readout_lines": [q for line in controller.readout_lines for q in line.qudits]
            if system._shares
            else [line[0] for line in controller.readout_lines],
            "backend": controller.backend,
        }
        if controller.backend is not None:
            entry["backend_installed"] = controller.backend in installed
            if not entry["backend_installed"]:
                missing.append(
                    Problem(
                        f"backend {controller.backend!r} is not installed",
                        path=f"control_system.controllers[{index}].backend",
                        value=controller.backend,
                        file=str(file),
                        fix="install the package that provides it; python -m "
                        "cudaq_pulse.doctor lists the installed backends",
                    )
                )
        controllers.append(entry)
    links = {
        name: None if link.controllers is None else [system._index(c) for c in link.controllers]
        for name, link in system.links.items()
    }
    return {"controllers": controllers, "links": links}, tuple(missing)


def _control_lines(summary: dict[str, Any] | None) -> list[str]:
    if summary is None:
        return []
    from ._errors import _ranges

    lines = []
    for index, c in enumerate(summary["controllers"]):
        name = c["model"] or "unnamed"
        version = f" version {c['version']}" if c["version"] else ""
        text = (
            f"controllers[{index}]: {name}{version}, drive lines "
            f"{_ranges(c['drive_lines']) if c['drive_lines'] else 'none'}, readout lines "
            f"{_ranges(c['readout_lines']) if c['readout_lines'] else 'none'}"
        )
        if c["backend"] is not None:
            found = "installed" if c["backend_installed"] else "not installed"
            text += f", backend {c['backend']} ({found})"
        lines.append(text)
    for name, reached in summary["links"].items():
        whom = "every controller" if reached is None else f"controllers {reached}"
        lines.append(f"link {name!r}: reaches {whom}")
    return lines


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
    modes.add_argument(
        "--setup-editor",
        action="store_true",
        help="show the mypy and ruff configuration the kernel language needs, and "
        "write it when confirmed (pyright-based editors need none)",
    )
    modes.add_argument(
        "--system",
        type=Path,
        metavar="FILE",
        help="check the system file FILE as pulse.QuantumSystem.load(FILE) does: every "
        "problem, its controllers and links, and whether their backends are installed",
    )
    parser.add_argument(
        "--json", action="store_true", help="with --system, print the report as JSON"
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="project directory for --setup-editor (default: the current directory)",
    )
    parser.add_argument(
        "--yes", action="store_true", help="with --setup-editor, write without asking"
    )
    args = parser.parse_args(argv)
    stream = out if out is not None else sys.stdout
    if (args.yes or args.project is not None) and not args.setup_editor:
        parser.error("--yes and --project apply to --setup-editor")
    if args.json and args.system is None:
        parser.error("--json applies to --system")
    if args.system is not None:
        return _check_system(args.system, args.json, lambda line: print(line, file=stream))
    if args.setup_editor:
        from ._editor_setup import setup_editor

        def confirm(prompt: str) -> bool:
            return input(prompt).strip().lower() in ("y", "yes")

        return setup_editor(
            args.project if args.project is not None else Path.cwd(),
            assume_yes=args.yes,
            out=stream,
            confirm=confirm if out is None and sys.stdin.isatty() else None,
        )

    def write(line: str) -> None:
        print(line, file=stream)

    package = _distribution(_DISTRIBUTION)
    if package is not None:
        version = package.version
    else:
        from . import __version__

        version = f"{__version__} (not installed as a distribution; source or build tree)"
    write(f"cudaq-pulse {version}")
    write(
        f"Python {platform.python_version()} ({sys.implementation.name}) on "
        f"{platform.system()} {platform.machine()}"
    )
    wheel = package.read_text("WHEEL") if package is not None else None
    if wheel:
        tags = [
            line.split(":", 1)[1].strip() for line in wheel.splitlines() if line.startswith("Tag:")
        ]
        write(f"Wheel tags: {', '.join(tags) or 'unknown'}")
    else:
        write("Wheel tags: none (not installed from a wheel)")

    compiler_error = _check_compiler()
    write(f"Compiler (MLIR JIT): {'ok' if compiler_error is None else 'FAILED'}")
    if compiler_error is not None:
        write(f"  {compiler_error}")
    if not _context_budget(write) and compiler_error is None:
        compiler_error = "invalid MLIR context limits"
    if not _kernel_settings(write) and compiler_error is None:
        compiler_error = "invalid kernel settings"
    driver_error = _check_driver(write, None if package is None else package.version)

    outcome = _diagnose_gpu(write)
    if compiler_error is not None:
        settings = compiler_error.startswith("invalid ")
        write(f"Result: FAILED, {compiler_error if settings else 'the compiler does not work'}.")
        return 1
    if driver_error is not None:
        write("Result: FAILED, cudaq-pulse-opt does not work.")
        return 1
    return _exit_code(
        outcome,
        allow_no_gpu=args.allow_no_gpu,
        expect_no_device=args.expect_no_device,
        write=write,
    )


if __name__ == "__main__":
    sys.exit(main())
