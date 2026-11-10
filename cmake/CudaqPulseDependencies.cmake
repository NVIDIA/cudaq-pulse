# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(DIRECTORY)

# Build-time dependency discovery for the GPU runtime, Python, and nanobind.
# MLIR/LLVM is resolved in the top-level CMakeLists.txt.

# The GPU runtime is an explicit choice, never auto-detected: OFF configures
# the compiler-only package, ON builds one runtime per CUDA major whose root
# is set and makes every dependency of those runtimes REQUIRED.
option(CUDAQ_PULSE_ENABLE_GPU
  "Build the cuDensityMat GPU runtime for each CUDA major with a CUDAQ_PULSE_CUDA<major>_ROOT"
  OFF)
set(CUDAQ_PULSE_CUDA_SUPPORTED_MAJORS 12 13)
foreach(_major IN LISTS CUDAQ_PULSE_CUDA_SUPPORTED_MAJORS)
  set(CUDAQ_PULSE_CUDA${_major}_ROOT "$ENV{CUDAQ_PULSE_CUDA${_major}_ROOT}" CACHE STRING
    "Directories (;-list) providing the CUDA ${_major} runtime and cuDensityMat for the GPU runtime")
endforeach()

set(CUDAQ_PULSE_CUDA_MAJORS "")
if(CUDAQ_PULSE_ENABLE_GPU)
  include(CudaqPulseCudaMajor)
  foreach(_major IN LISTS CUDAQ_PULSE_CUDA_SUPPORTED_MAJORS)
    if(CUDAQ_PULSE_CUDA${_major}_ROOT)
      cudaq_pulse_find_cuda_major(${_major})
      list(APPEND CUDAQ_PULSE_CUDA_MAJORS ${_major})
    endif()
  endforeach()
  if(NOT CUDAQ_PULSE_CUDA_MAJORS)
    message(FATAL_ERROR
      "CUDAQ_PULSE_ENABLE_GPU=ON needs at least one of CUDAQ_PULSE_CUDA12_ROOT "
      "or CUDAQ_PULSE_CUDA13_ROOT. For a single-major build, install that "
      "major's extra (e.g. `uv pip install -r pyproject.toml --extra cu12`) "
      "and pass a CUDA toolkit plus that site-packages directory, e.g. "
      "-DCUDAQ_PULSE_CUDA12_ROOT=\"/usr/local/cuda-12.9;$(python -c "
      "'import sysconfig; print(sysconfig.get_path(\"purelib\"))')\". "
      "Without a toolkit, also install the header wheels listed in "
      "tools/ci/manylinux-before-all.sh into that site-packages.")
  endif()
  add_subdirectory(core/runtime)
endif()

# nanobind resolves its Python dependency through FindPython, whereas MLIR and
# the top-level project use FindPython3. Seed the former from the latter so both
# modules agree on a single interpreter.
if(NOT Python_EXECUTABLE)
  set(Python_EXECUTABLE "${Python3_EXECUTABLE}")
endif()
# scikit-build-core sets SKBUILD_SABI_COMPONENT when it tags the wheel abi3
# (wheel.py-api, Python >= 3.12), which nanobind's STABLE_ABI requires.
find_package(Python 3.12 REQUIRED
  COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT})

# nanobind ships its own CMake package inside the Python wheel. Ask the
# interpreter where it is, while still letting -Dnanobind_DIR override.
if(NOT nanobind_DIR)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE _cudaq_pulse_nanobind_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE _cudaq_pulse_nanobind_error
    ERROR_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _cudaq_pulse_nanobind_status)
  if(NOT _cudaq_pulse_nanobind_status EQUAL 0)
    message(FATAL_ERROR
      "nanobind was not found by ${Python3_EXECUTABLE}: "
      "${_cudaq_pulse_nanobind_error}\n"
      "Install it with `pip install \"nanobind>=2.12\"`, or point at an "
      "existing CMake package with -Dnanobind_DIR=<dir>.")
  endif()
  set(nanobind_DIR "${_cudaq_pulse_nanobind_dir}")
endif()
find_package(nanobind CONFIG REQUIRED)
