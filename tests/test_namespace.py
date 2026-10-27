# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The public namespace is frozen: these names, and only these."""

import subprocess
import sys

import cudaq_pulse as pulse
import pytest

TOP_LEVEL = {
    "kernel",
    "Kernel",
    "compile",
    "CompiledKernel",
    "Lowered",
    "simulate",
    "SimulationResult",
    "device",
    "get_device",
    "Device",
    "Qudit",
    "Register",
    "Target",
    "PulseError",
    "CompilationError",
    "NoTargetError",
    "NoDeviceError",
    "DeviceMismatchError",
    "GpuRuntimeError",
    "viz",
    "classical",
    "controller",
    "backends",
    "__version__",
}

DSL = {
    "get_drive_line",
    "get_readout_line",
    "drive",
    "readout",
    "wait",
    "sync",
    "shift_phase",
    "set_phase",
    "shift_frequency",
    "set_frequency",
    "gaussian",
    "square",
    "drag",
    "cosine",
    "tanh_ramp",
    "gaussian_square",
    "custom",
    "custom_samples",
    "wf_add",
    "wf_sub",
    "wf_mul",
    "wf_scale",
    "wf_neg",
}


def test_dir_is_exactly_the_public_names():
    assert set(dir(pulse)) == TOP_LEVEL
    assert set(pulse.__all__) == TOP_LEVEL - {"__version__"}


def test_every_public_name_resolves():
    for name in TOP_LEVEL:
        assert getattr(pulse, name) is not None, name


@pytest.mark.parametrize(
    "name",
    [
        "qudit_ref",
        "qvec_ref",
        "QuditRef",
        "QvecRef",
        "drive",
        "gaussian",
        "CompileMetrics",
        "ScheduledEvent",
        "MachineModel",
        "DEFAULT_PASSES",
        "Parameter",
    ],
)
def test_names_that_moved_or_went_are_not_top_level(name):
    with pytest.raises(AttributeError):
        getattr(pulse, name)


def test_the_kernel_language_is_in_ops():
    assert DSL <= set(dir(pulse.ops))
    assert len(DSL) == 23


def test_ops_all_is_the_kernel_language_the_tracer_lowers():
    from cudaq_pulse.kernel.ir_builder import OP_TABLE

    assert set(pulse.ops.__all__) == DSL == set(OP_TABLE)
    assert len(pulse.ops.__all__) == len(DSL)


def test_ruff_builtins_are_the_kernel_language():
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    builtins = tomllib.loads(pyproject.read_text())["tool"]["ruff"]["builtins"]
    assert sorted(builtins) == sorted(pulse.ops.__all__)


def _stub_imports(stub):
    return [ln for ln in stub.read_text().splitlines() if ln.startswith("from ")]


def test_the_installed_builtins_stub_reexports_exactly_ops_all():
    from pathlib import Path

    # Beside the package, at the root of site-packages (or of build/*/python).
    stub = Path(pulse.__file__).parent.parent / "__builtins__.pyi"
    assert _stub_imports(stub) == [
        f"from cudaq_pulse.ops import {n} as {n}" for n in pulse.ops.__all__
    ]


def test_builtins_stub_reexports_exactly_ops_all(tmp_path):
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    generator = root / "cmake" / "write_builtins_stub.py"
    if not generator.exists():
        pytest.skip("needs the source tree; a wheel's tests check the installed stub")
    spec = importlib.util.spec_from_file_location("write_builtins_stub", generator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "__builtins__.pyi"
    module.main(
        [
            "write_builtins_stub.py",
            str(root / "core/frontend/cudaq_pulse/ops/__init__.py"),
            str(out),
        ]
    )
    assert _stub_imports(out) == [
        f"from cudaq_pulse.ops import {n} as {n}" for n in pulse.ops.__all__
    ]


def test_compiler_types_are_in_the_compiler_module():
    from cudaq_pulse import compiler

    for name in ("CompileMetrics", "ScheduledEvent", "MachineModel", "DEFAULT_PASSES"):
        assert hasattr(compiler, name)


def test_the_target_description_is_not_the_target_enum():
    from cudaq_pulse.targets import TargetDescription

    assert pulse.Target.TRANSMON_KRINNER_17Q.value == "transmon_krinner_17q"
    assert not issubclass(TargetDescription, type(pulse.Target.TRANSMON_KRINNER_17Q))
    import cudaq_pulse.targets as targets

    assert not hasattr(targets, "Target")


def test_viz_loads_on_first_use():
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, cudaq_pulse as p; before = 'cudaq_pulse.viz' in sys.modules; "
            "p.viz; print(before, 'cudaq_pulse.viz' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out == "False True"


def test_every_public_error_is_a_pulse_error_and_keeps_its_builtin_base():
    import cudaq_pulse.backends
    import cudaq_pulse.compiler
    import cudaq_pulse.controller
    import cudaq_pulse.runtime
    import cudaq_pulse.targets

    builtin = {
        "CompilationError": Exception,
        "UnsupportedInterpreterError": RuntimeError,
        "NoTargetError": RuntimeError,
        "NoDeviceError": RuntimeError,
        "DeviceMismatchError": RuntimeError,
        "UnknownTargetError": LookupError,
        "GpuRuntimeError": RuntimeError,
        "GpuSupportNotBuiltError": RuntimeError,
        "GpuRuntimeLoadError": RuntimeError,
        "GpuDeviceUnavailableError": RuntimeError,
        "BackendError": Exception,
        "BackendNotFoundError": LookupError,
        "IncompatibleBackendError": TypeError,
        "ControllerLimitError": Exception,
    }
    errors = {
        name: getattr(module, name)
        for module in (
            pulse,
            cudaq_pulse.backends,
            cudaq_pulse.compiler,
            cudaq_pulse.controller,
            cudaq_pulse.runtime,
            cudaq_pulse.targets,
        )
        for name in module.__all__
        if isinstance(getattr(module, name), type)
        and issubclass(getattr(module, name), BaseException)
    }
    assert set(errors) == set(builtin) | {"PulseError"}
    for name, base in builtin.items():
        assert issubclass(errors[name], pulse.PulseError), name
        assert issubclass(errors[name], base), name
