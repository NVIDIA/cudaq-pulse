# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

include_guard(DIRECTORY)

include(CudaqPulseStagePython)

# The build tree mirrors the installed package: one PYTHONPATH entry exposes
# the staged Python sources, the native extension, and the GPU runtime.
file(GLOB_RECURSE _cudaq_pulse_python_sources
  RELATIVE "${CUDAQ_PULSE_FRONTEND_DIR}"
  "${CUDAQ_PULSE_FRONTEND_DIR}/*.py"
  "${CUDAQ_PULSE_FRONTEND_DIR}/*.pyi"
  "${CUDAQ_PULSE_FRONTEND_DIR}/*.json")
list(APPEND _cudaq_pulse_python_sources py.typed)

cudaq_pulse_stage_python_sources(CudaqPulsePythonStaging
  ROOT_DIR "${CUDAQ_PULSE_FRONTEND_DIR}"
  OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/cudaq_pulse"
  SOURCES ${_cudaq_pulse_python_sources})
add_dependencies(_cudaq_pulse_native CudaqPulsePythonStaging)

set(CUDAQ_PULSE_PYTHONPATH "${CMAKE_BINARY_DIR}/python")

# The version comes from the git tags through setuptools-scm, configured in
# pyproject.toml, for wheels and plain CMake builds alike. A CMake build reads
# it at configure time, so re-run configure after tagging.
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

set(_cudaq_pulse_build_targets
  cudaq-pulse-opt
  _cudaq_pulse_native
  _cudaq_pulse_native_stub
  CudaqPulsePythonStaging)

set(_cudaq_pulse_runtime_libraries "")
set(_cudaq_pulse_cudensitymat_versions "")
set(_cudaq_pulse_cuda_versions "")
foreach(_major IN LISTS CUDAQ_PULSE_CUDA_MAJORS)
  list(APPEND _cudaq_pulse_build_targets cudm_runtime_cu${_major})
  list(APPEND _cudaq_pulse_runtime_libraries
    "${_major}: \"lib/${CMAKE_SHARED_LIBRARY_PREFIX}cudm-runtime-cu${_major}${CMAKE_SHARED_LIBRARY_SUFFIX}\"")
  list(APPEND _cudaq_pulse_cudensitymat_versions
    "${_major}: \"${CUDAQ_PULSE_CUDA${_major}_CUDENSITYMAT_VERSION}\"")
  list(APPEND _cudaq_pulse_cuda_versions
    "${_major}: \"${CUDAQ_PULSE_CUDA${_major}_VERSION}\"")
endforeach()
list(JOIN _cudaq_pulse_runtime_libraries ", " _cudaq_pulse_runtime_libraries)
list(JOIN _cudaq_pulse_cudensitymat_versions ", " _cudaq_pulse_cudensitymat_versions)
list(JOIN _cudaq_pulse_cuda_versions ", " _cudaq_pulse_cuda_versions)
if(CUDAQ_PULSE_ENABLE_GPU)
  set(CUDAQ_PULSE_GPU_RUNTIME_ENABLED_PY True)
else()
  set(CUDAQ_PULSE_GPU_RUNTIME_ENABLED_PY False)
endif()
set(CUDAQ_PULSE_GPU_RUNTIME_LIBRARIES_PY "{${_cudaq_pulse_runtime_libraries}}")
set(CUDAQ_PULSE_CUDENSITYMAT_VERSIONS_PY "{${_cudaq_pulse_cudensitymat_versions}}")
set(CUDAQ_PULSE_CUDA_VERSIONS_PY "{${_cudaq_pulse_cuda_versions}}")

configure_file(
  "${CUDAQ_PULSE_SOURCE_DIR}/cmake/_build_config.py.in"
  "${CMAKE_BINARY_DIR}/python/cudaq_pulse/_build_config.py"
  @ONLY)

add_custom_target(pulse DEPENDS ${_cudaq_pulse_build_targets})

install(DIRECTORY "${CUDAQ_PULSE_FRONTEND_DIR}"
  DESTINATION .
  COMPONENT CudaqPulse
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE)
install(FILES "${CMAKE_BINARY_DIR}/python/cudaq_pulse/_build_config.py"
  DESTINATION cudaq_pulse
  COMPONENT CudaqPulse)
