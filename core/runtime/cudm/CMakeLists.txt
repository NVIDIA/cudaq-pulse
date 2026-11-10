# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The installed libraries live at site-packages/cudaq_pulse/lib and resolve
# cuDensityMat and the CUDA runtime from the wheels named by that major's
# [cuNN] extra. CUDA 13 component wheels all install into nvidia/cu13/lib.
set(_cudm_runtime_cu12_install_rpath
  "$ORIGIN/../../cuquantum/lib;$ORIGIN/../../nvidia/cuda_runtime/lib")
set(_cudm_runtime_cu13_install_rpath
  "$ORIGIN/../../cuquantum/lib;$ORIGIN/../../nvidia/cu13/lib")

foreach(_major IN LISTS CUDAQ_PULSE_CUDA_MAJORS)
  set(_target cudm_runtime_cu${_major})
  add_library(${_target} SHARED
    cudm_runtime.cpp
  )

  # The runtime is loaded by JIT-compiled code, not linked by installed
  # consumers, so its header is only used inside this build.
  target_include_directories(${_target} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

  target_link_libraries(${_target} PRIVATE
    cudaq_pulse_cuda${_major}
    ${CMAKE_DL_LIBS})

  set_target_properties(${_target} PROPERTIES
    OUTPUT_NAME "cudm-runtime-cu${_major}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/cudaq_pulse/lib"
    BUILD_RPATH "${CUDAQ_PULSE_CUDA${_major}_LIBRARY_DIRS}"
    INSTALL_RPATH "${_cudm_runtime_cu${_major}_install_rpath}"
  )

  target_link_options(${_target} PRIVATE
    "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.linux.map"
    "LINKER:--exclude-libs,ALL")
  set_property(TARGET ${_target} APPEND PROPERTY LINK_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/exports.linux.map")

  install(TARGETS ${_target}
    LIBRARY DESTINATION cudaq_pulse/lib
    COMPONENT CudaqPulse
  )
endforeach()
