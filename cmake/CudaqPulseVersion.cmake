# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

# The version comes from the git tags through setuptools-scm, configured in
# pyproject.toml, for wheels and plain CMake builds alike; the Python package
# and cudaq-pulse-opt both report it. A CMake build reads it at configure
# time, so re-run configure after tagging.
if(SKBUILD)
  set(CUDAQ_PULSE_PACKAGE_VERSION "${SKBUILD_PROJECT_VERSION_FULL}")
else()
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m setuptools_scm
      --root "${CUDAQ_PULSE_SOURCE_DIR}"
      --config "${CUDAQ_PULSE_SOURCE_DIR}/pyproject.toml"
    OUTPUT_VARIABLE CUDAQ_PULSE_PACKAGE_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _cudaq_pulse_scm_result)
  if(NOT _cudaq_pulse_scm_result EQUAL 0 OR CUDAQ_PULSE_PACKAGE_VERSION STREQUAL "")
    message(STATUS "setuptools-scm could not determine the version (install the "
      "`build` dependency group, or build from a git checkout); using 0.0.0+unknown")
    set(CUDAQ_PULSE_PACKAGE_VERSION "0.0.0+unknown")
  endif()
endif()
message(STATUS "cudaq-pulse version: ${CUDAQ_PULSE_PACKAGE_VERSION}")
