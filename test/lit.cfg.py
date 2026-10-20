# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os

import lit.formats

config.name = "cudaq-pulse"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)

# Tests resolve tools only from this build and the pinned MLIR toolchain.
config.environment["PATH"] = os.pathsep.join([config.cudaq_pulse_tools_dir, config.llvm_tools_dir])
config.substitutions.append(("%pulse_example_plugin", config.pulse_example_plugin))
