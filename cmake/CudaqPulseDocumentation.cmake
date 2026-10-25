# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

include_guard(DIRECTORY)

option(CUDAQ_PULSE_BUILD_DOCS "Build the cudaq-pulse documentation" OFF)
if(NOT CUDAQ_PULSE_BUILD_DOCS)
  return()
endif()

get_filename_component(_cudaq_pulse_python_bin
  "${Python_EXECUTABLE}" DIRECTORY)
find_program(CUDAQ_PULSE_SPHINX_EXECUTABLE NAMES sphinx-build
  HINTS "${_cudaq_pulse_python_bin}"
  REQUIRED)

# Sphinx reads a copy of docs/ with the dialect reference pages, which
# mlir-tblgen generates from the ODS definitions, added under
# architecture/dialects/.
set(_cudaq_pulse_docs_source "${CUDAQ_PULSE_BINARY_DIR}/docs/source")
add_custom_target(pulse-docs
  COMMAND ${CMAKE_COMMAND} -E rm -rf ${_cudaq_pulse_docs_source}
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CUDAQ_PULSE_SOURCE_DIR}/docs ${_cudaq_pulse_docs_source}
  COMMAND ${CMAKE_COMMAND}
          -DGENERATED_DIR=${CUDAQ_PULSE_BINARY_DIR}/core/mlir/include/cudaq-pulse/Dialect
          -DOUTPUT_DIR=${_cudaq_pulse_docs_source}/architecture/dialects
          -P ${CUDAQ_PULSE_SOURCE_DIR}/cmake/CudaqPulseDialectDocs.cmake
  COMMAND ${CMAKE_COMMAND} -E env
          "PYTHONPATH=${CUDAQ_PULSE_PYTHONPATH}"
          ${CUDAQ_PULSE_SPHINX_EXECUTABLE} -W --keep-going -b html
          ${_cudaq_pulse_docs_source}
          ${CUDAQ_PULSE_BINARY_DIR}/docs/html
  DEPENDS pulse CudaqPulsePulseDocGen CudaqPulseQOpDocGen
          CudaqPulseCuDensityMatDocGen
  COMMENT "Building cudaq-pulse documentation"
  USES_TERMINAL)
