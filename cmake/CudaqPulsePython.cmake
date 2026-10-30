# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(DIRECTORY)

include(CudaqPulseStagePython)

# The build tree mirrors the installed package: one PYTHONPATH entry exposes
# the staged Python sources, the native extension, and the GPU runtime.
file(GLOB_RECURSE _cudaq_pulse_python_sources
  RELATIVE "${CUDAQ_PULSE_FRONTEND_DIR}"
  "${CUDAQ_PULSE_FRONTEND_DIR}/*.py"
  "${CUDAQ_PULSE_FRONTEND_DIR}/*.pyi"
  "${CUDAQ_PULSE_FRONTEND_DIR}/*.json"
  "${CUDAQ_PULSE_FRONTEND_DIR}/*.toml")
list(APPEND _cudaq_pulse_python_sources py.typed)

cudaq_pulse_stage_python_sources(CudaqPulsePythonStaging
  ROOT_DIR "${CUDAQ_PULSE_FRONTEND_DIR}"
  OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/cudaq_pulse"
  SOURCES ${_cudaq_pulse_python_sources})
add_dependencies(_cudaq_pulse_native CudaqPulsePythonStaging)

set(CUDAQ_PULSE_PYTHONPATH "${CMAKE_BINARY_DIR}/python")

# A hash of those sources, written at build time so that import reads it
# rather than computes it (see cmake/write_source_stamp.cmake).
set(_cudaq_pulse_source_list "${CMAKE_BINARY_DIR}/python_sources.txt")
list(JOIN _cudaq_pulse_python_sources "\n" _cudaq_pulse_source_lines)
file(CONFIGURE OUTPUT "${_cudaq_pulse_source_list}" CONTENT "${_cudaq_pulse_source_lines}\n")
list(TRANSFORM _cudaq_pulse_python_sources PREPEND "${CUDAQ_PULSE_FRONTEND_DIR}/"
  OUTPUT_VARIABLE _cudaq_pulse_python_paths)
set(_cudaq_pulse_source_stamp "${CMAKE_BINARY_DIR}/python/cudaq_pulse/_source_stamp.py")
add_custom_command(
  OUTPUT "${_cudaq_pulse_source_stamp}"
  COMMAND "${CMAKE_COMMAND}" "-DROOT=${CUDAQ_PULSE_FRONTEND_DIR}"
    "-DSOURCES=${_cudaq_pulse_source_list}" "-DOUTPUT=${_cudaq_pulse_source_stamp}"
    -P "${CUDAQ_PULSE_SOURCE_DIR}/cmake/write_source_stamp.cmake"
  DEPENDS ${_cudaq_pulse_python_paths} "${_cudaq_pulse_source_list}"
    "${CUDAQ_PULSE_SOURCE_DIR}/cmake/write_source_stamp.cmake"
  COMMENT "Hashing the Python sources into _source_stamp.py")
add_custom_target(CudaqPulseSourceStamp DEPENDS "${_cudaq_pulse_source_stamp}")
add_dependencies(_cudaq_pulse_native CudaqPulseSourceStamp)

# The kernel language as builtins for type checkers, beside the package at
# the root of site-packages (see cmake/write_builtins_stub.py).
set(_cudaq_pulse_builtins_stub "${CMAKE_BINARY_DIR}/python/__builtins__.pyi")
add_custom_command(
  OUTPUT "${_cudaq_pulse_builtins_stub}"
  COMMAND "${Python3_EXECUTABLE}" "${CUDAQ_PULSE_SOURCE_DIR}/cmake/write_builtins_stub.py"
    "${CUDAQ_PULSE_FRONTEND_DIR}/ops/__init__.py" "${_cudaq_pulse_builtins_stub}"
  DEPENDS "${CUDAQ_PULSE_SOURCE_DIR}/cmake/write_builtins_stub.py"
    "${CUDAQ_PULSE_FRONTEND_DIR}/ops/__init__.py"
  COMMENT "Generating __builtins__.pyi from cudaq_pulse.ops.__all__")
add_custom_target(CudaqPulseBuiltinsStub DEPENDS "${_cudaq_pulse_builtins_stub}")
add_dependencies(_cudaq_pulse_native CudaqPulseBuiltinsStub)

set(_cudaq_pulse_build_targets
  _cudaq_pulse_native
  _cudaq_pulse_native_stub
  CudaqPulsePythonStaging
  CudaqPulseSourceStamp)
if(CUDAQ_PULSE_BUILD_OPT)
  list(APPEND _cudaq_pulse_build_targets cudaq-pulse-opt)
endif()

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

include(CudaqPulseNotices)
set(_cudaq_pulse_shipped_binaries _cudaq_pulse_native)
foreach(_major IN LISTS CUDAQ_PULSE_CUDA_MAJORS)
  list(APPEND _cudaq_pulse_shipped_binaries cudm_runtime_cu${_major})
endforeach()
cudaq_pulse_third_party_notices(
  NOTICES "${CUDAQ_PULSE_SOURCE_DIR}/THIRD_PARTY_NOTICES"
  COMPONENT CudaqPulse
  TARGETS ${_cudaq_pulse_shipped_binaries})

install(DIRECTORY "${CUDAQ_PULSE_FRONTEND_DIR}"
  DESTINATION .
  COMPONENT CudaqPulse
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE)
install(FILES "${CMAKE_BINARY_DIR}/python/cudaq_pulse/_build_config.py"
  "${_cudaq_pulse_source_stamp}"
  DESTINATION cudaq_pulse
  COMPONENT CudaqPulse)
install(FILES "${_cudaq_pulse_builtins_stub}"
  DESTINATION .
  COMPONENT CudaqPulse)
