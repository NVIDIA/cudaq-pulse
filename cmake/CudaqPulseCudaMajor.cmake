# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

# Directory, relative to site-packages, holding each CUDA major's runtime
# wheel files. CUDA 12 gives every component its own directory; CUDA 13
# wheels share one.
set(_cudaq_pulse_cuda12_wheel_dir "nvidia/cuda_runtime")
set(_cudaq_pulse_cuda13_wheel_dir "nvidia/cu13")
# CUDA 12 ships crt/host_defines.h with nvcc; CUDA 13 has a separate crt wheel
# that installs next to the runtime headers.
set(_cudaq_pulse_cuda12_crt_dir "nvidia/cuda_nvcc")
set(_cudaq_pulse_cuda13_crt_dir "nvidia/cu13")

# cudaq_pulse_find_cuda_major(<major>)
#
# Resolves the CUDA runtime and cuDensityMat for one CUDA major from the
# directories listed in CUDAQ_PULSE_CUDA<major>_ROOT. Each directory may be a
# toolkit-style prefix (include/, lib/) or a site-packages tree populated by
# that major's pip wheels. Defines the INTERFACE target cudaq_pulse_cuda<major>
# and, in the caller's scope, CUDAQ_PULSE_CUDA<major>_VERSION,
# CUDAQ_PULSE_CUDA<major>_CUDENSITYMAT_VERSION, and
# CUDAQ_PULSE_CUDA<major>_LIBRARY_DIRS.
function(cudaq_pulse_find_cuda_major major)
  set(_roots ${CUDAQ_PULSE_CUDA${major}_ROOT})
  set(_wheel "${_cudaq_pulse_cuda${major}_wheel_dir}")
  set(_crt "${_cudaq_pulse_cuda${major}_crt_dir}")
  set(_prefix CUDAQ_PULSE_CUDA${major})

  find_path(${_prefix}_CUDART_INCLUDE_DIR cuda_runtime_api.h
    HINTS ${_roots} PATH_SUFFIXES include "${_wheel}/include" NO_DEFAULT_PATH)
  find_path(${_prefix}_CRT_INCLUDE_DIR crt/host_defines.h
    HINTS ${_roots} PATH_SUFFIXES include "${_crt}/include" NO_DEFAULT_PATH)
  find_path(${_prefix}_CUDENSITYMAT_INCLUDE_DIR cudensitymat.h
    HINTS ${_roots} PATH_SUFFIXES include cuquantum/include NO_DEFAULT_PATH)
  find_library(${_prefix}_CUDART_LIBRARY
    NAMES libcudart.so.${major} cudart
    HINTS ${_roots} PATH_SUFFIXES lib lib64 "${_wheel}/lib" NO_DEFAULT_PATH)
  find_library(${_prefix}_CUDENSITYMAT_LIBRARY
    NAMES libcudensitymat.so.0 cudensitymat
    HINTS ${_roots}
    PATH_SUFFIXES lib lib64 lib/${major} lib64/${major} cuquantum/lib
    NO_DEFAULT_PATH)

  set(_missing "")
  foreach(_var CUDART_INCLUDE_DIR CRT_INCLUDE_DIR CUDENSITYMAT_INCLUDE_DIR
               CUDART_LIBRARY CUDENSITYMAT_LIBRARY)
    if(NOT ${_prefix}_${_var})
      list(APPEND _missing ${_prefix}_${_var})
    endif()
  endforeach()
  if(_missing)
    list(JOIN _missing ", " _missing)
    message(FATAL_ERROR
      "CUDA ${major} GPU runtime requested (CUDAQ_PULSE_CUDA${major}_ROOT="
      "\"${_roots}\"), but these were not found there: ${_missing}. The root "
      "must contain the CUDA runtime (cuda_runtime_api.h, crt/host_defines.h, "
      "libcudart.so.${major}) and cuDensityMat (cudensitymat.h, "
      "libcudensitymat.so.0), either as a toolkit-style include/ and lib/ "
      "prefix or as a site-packages tree; tools/ci/manylinux-before-all.sh "
      "shows the pip wheels that provide them.")
  endif()

  file(STRINGS "${${_prefix}_CUDART_INCLUDE_DIR}/cuda_runtime_api.h" _line
    REGEX "^#define CUDART_VERSION +[0-9]+")
  string(REGEX MATCH "[0-9]+$" _cudart "${_line}")
  math(EXPR _found_major "${_cudart} / 1000")
  math(EXPR _found_minor "(${_cudart} % 1000) / 10")
  if(NOT _found_major EQUAL major)
    message(FATAL_ERROR
      "CUDAQ_PULSE_CUDA${major}_ROOT provides the CUDA ${_found_major}.${_found_minor} "
      "runtime headers (${${_prefix}_CUDART_INCLUDE_DIR}), not CUDA ${major}.")
  endif()

  file(STRINGS "${${_prefix}_CUDENSITYMAT_INCLUDE_DIR}/cudensitymat.h" _lines
    REGEX "^#define CUDENSITYMAT_(MAJOR|MINOR|PATCH) +[0-9]+")
  set(_cudm_version "")
  foreach(_part MAJOR MINOR PATCH)
    string(REGEX MATCH "CUDENSITYMAT_${_part} +([0-9]+)" _ "${_lines}")
    list(APPEND _cudm_version "${CMAKE_MATCH_1}")
  endforeach()
  list(JOIN _cudm_version "." _cudm_version)

  add_library(cudaq_pulse_cuda${major} INTERFACE)
  target_include_directories(cudaq_pulse_cuda${major} SYSTEM INTERFACE
    "${${_prefix}_CUDART_INCLUDE_DIR}"
    "${${_prefix}_CRT_INCLUDE_DIR}"
    "${${_prefix}_CUDENSITYMAT_INCLUDE_DIR}")
  target_link_libraries(cudaq_pulse_cuda${major} INTERFACE
    "${${_prefix}_CUDENSITYMAT_LIBRARY}"
    "${${_prefix}_CUDART_LIBRARY}")

  get_filename_component(_cudart_dir "${${_prefix}_CUDART_LIBRARY}" DIRECTORY)
  get_filename_component(_cudm_dir "${${_prefix}_CUDENSITYMAT_LIBRARY}" DIRECTORY)
  set(_library_dirs "${_cudm_dir}" "${_cudart_dir}")
  list(REMOVE_DUPLICATES _library_dirs)

  set(${_prefix}_VERSION "${_found_major}.${_found_minor}" PARENT_SCOPE)
  set(${_prefix}_CUDENSITYMAT_VERSION "${_cudm_version}" PARENT_SCOPE)
  set(${_prefix}_LIBRARY_DIRS "${_library_dirs}" PARENT_SCOPE)
  message(STATUS
    "cudaq-pulse GPU runtime for CUDA ${major}: CUDA ${_found_major}.${_found_minor}, "
    "cuDensityMat ${_cudm_version} (${${_prefix}_CUDENSITYMAT_LIBRARY})")
endfunction()
