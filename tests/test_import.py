# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Importing and compiling stay light; simulation loads its dependencies lazily."""

import importlib
import importlib.metadata
import os
import re
import subprocess
import sys
from pathlib import Path

import cudaq_pulse as pulse
import numpy as np
import pytest

from tests._handles import Q

_ROOT = str(Path(pulse.__file__).parent.parent)


def _run(code: str, tmp_path: Path) -> str:
    script = tmp_path / "probe.py"
    script.write_text(code)
    env = dict(os.environ, PYTHONPATH=_ROOT)
    result = subprocess.run(
        [sys.executable, str(script)], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_import_and_compile_do_not_load_numpy_or_the_gpu_runtime(tmp_path):
    loaded = _run(
        "import sys, cudaq_pulse as p\n"
        "@p.kernel\n"
        "def k(q):\n"
        "    d = get_drive_line(q)\n"
        "    drive(d, square(40, 0.1))\n"
        "dev = p.device('transmon_krinner_17q')\n"
        "p.compile(k).lower(dev.qudit_handles[0])\n"
        "print(sorted(m for m in ('numpy', 'cudaq_pulse.runtime.gpu',"
        " 'cudaq_pulse.runtime._jit', 'importlib.metadata') if m in sys.modules))",
        tmp_path,
    )
    assert loaded == "[]"


def test_simulation_exports_load_on_first_use(tmp_path):
    out = _run(
        "import sys, cudaq_pulse as p\n"
        "before = 'cudaq_pulse.runtime.simulate' in sys.modules\n"
        "f = p.simulate\n"
        "print(before, 'cudaq_pulse.runtime.simulate' in sys.modules, f.__name__,"
        " p.SimulationResult.__name__, 'simulate' in dir(p))",
        tmp_path,
    )
    assert out == "False True simulate SimulationResult True"


def test_dir_lists_no_internals():
    names = set(dir(pulse))
    leaked = {"_device", "_native", "_build_config", "annotations", "TYPE_CHECKING"}
    assert not names & leaked
    assert not {n for n in names if n.startswith("_") and not n.startswith("__")}
    assert set(pulse.__all__) <= names
    assert "__version__" in names
    # Hidden, not removed.
    assert pulse._device.Device is pulse.Device


@pytest.mark.parametrize(
    "module",
    [
        "cudaq_pulse",
        "cudaq_pulse._device",
        "cudaq_pulse.doctor",
        "cudaq_pulse.compiler",
        "cudaq_pulse.ops",
        "cudaq_pulse.runtime",
        "cudaq_pulse.runtime.simulate",
        "cudaq_pulse.targets",
        "cudaq_pulse.targets.transmon_model",
        "cudaq_pulse.viz",
    ],
)
def test_all_names_resolve(module):
    mod = importlib.import_module(module)
    assert mod.__all__
    for name in mod.__all__:
        assert getattr(mod, name) is not None, name


def test_runtime_exports_the_gpu_errors_without_loading_them(tmp_path):
    out = _run(
        "import sys, cudaq_pulse.runtime as r\n"
        "before = 'cudaq_pulse.runtime.gpu' in sys.modules\n"
        "from cudaq_pulse.runtime import gpu\n"
        "print(before, r.GpuRuntimeError is gpu.GpuRuntimeError,"
        " issubclass(r.GpuSupportNotBuiltError, r.GpuRuntimeError))",
        tmp_path,
    )
    assert out == "False True True"


def test_version_comes_from_the_build():
    from cudaq_pulse import _build_config

    assert pulse.__version__ == _build_config.PACKAGE_VERSION
    assert re.fullmatch(
        r"\d+(\.\d+)+((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?(\+[\w.]+)?", pulse.__version__
    )
    try:
        installed = importlib.metadata.version("cudaq-pulse")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    assert installed in (None, pulse.__version__)


def test_unknown_attribute_is_an_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'not_a_thing'"):
        pulse.not_a_thing  # noqa: B018


def test_numpy_scalars_and_arrays_are_accepted_without_numpy_in_the_tracer():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        drive(d, square(np.int64(40), np.float64(0.1)))
        drive(d, square(40, np.complex128(0.1 + 0.2j)))
        drive(d, square(40, np.array([0.1, 0.2])))
        drive(d, gaussian(np.int32(40), np.float32(0.25), 10.0))
        drive(d, custom_samples(np.linspace(0.0, 0.1, 8)))

    text = pulse.compile(k, passes=()).lower(Q[0]).mlir()
    assert text.count("= pulse.drive ") == 5
    assert "2.000000e-01" in text
