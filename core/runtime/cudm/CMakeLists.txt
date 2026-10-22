# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

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

  # Use build/install-interface generator expressions: the raw source path is
  # only valid while building from this source tree, whereas consumers of an
  # installed cudaq-pulse resolve the header from the install prefix.
  target_include_directories(${_target} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
  )

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
