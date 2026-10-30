# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cudaq-pulse-opt --version prints the package's version and its LLVM's."""

import shutil
import subprocess
from pathlib import Path

import cudaq_pulse
import pytest
from cudaq_pulse import _build_config

# A build tree stages the package in build/<dir>/python and cudaq-pulse-opt in
# build/<dir>/bin; an installation puts the cudaq-pulse-tools wheel's on PATH.
_BUILD_DRIVER = Path(cudaq_pulse.__file__).parent.parent.parent / "bin" / "cudaq-pulse-opt"


def test_the_driver_prints_the_package_version():
    driver = str(_BUILD_DRIVER) if _BUILD_DRIVER.exists() else shutil.which("cudaq-pulse-opt")
    if driver is None:
        pytest.skip("cudaq-pulse-opt is neither in the build tree nor on PATH")
    result = subprocess.run(
        [driver, "--version"], capture_output=True, text=True, timeout=60, check=True
    )
    assert result.stdout == (
        f"cudaq-pulse-opt {cudaq_pulse.__version__} (LLVM {_build_config.LLVM_VERSION})\n"
    )
