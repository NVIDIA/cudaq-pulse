# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

# The source of the Pulse and Classical dialects (see
# cmake/write_dialect_source.cmake): what the bytecode this build writes
# states in its producer string, and what pulse.load holds a bytecode module
# that states one to.
#
# CUDAQ_PULSE_DIALECT_DEFINITIONS  the files the source is the digest of
# CUDAQ_PULSE_DIALECT_LIST         a file that lists them: an input of the
#                                  hash, so that a file added or removed
#                                  changes it
# CudaqPulseDialectSourceInc       the target that writes the source as a C++
#                                  string literal, included as
#                                  "cudaq-pulse/Dialect/DialectSource.inc"
file(GLOB_RECURSE CUDAQ_PULSE_DIALECT_DEFINITIONS CONFIGURE_DEPENDS
  "${CUDAQ_PULSE_SOURCE_DIR}/core/mlir/include/cudaq-pulse/Dialect/Pulse/*"
  "${CUDAQ_PULSE_SOURCE_DIR}/core/mlir/include/cudaq-pulse/Dialect/Classical/*"
  "${CUDAQ_PULSE_SOURCE_DIR}/core/mlir/dialects/Pulse/*"
  "${CUDAQ_PULSE_SOURCE_DIR}/core/mlir/dialects/Classical/*")
list(FILTER CUDAQ_PULSE_DIALECT_DEFINITIONS INCLUDE REGEX "\\.(td|h|cpp)$")
set(CUDAQ_PULSE_DIALECT_LIST "${CUDAQ_PULSE_BINARY_DIR}/dialect_definitions.txt")
list(JOIN CUDAQ_PULSE_DIALECT_DEFINITIONS "\n" _cudaq_pulse_dialect_lines)
file(CONFIGURE OUTPUT "${CUDAQ_PULSE_DIALECT_LIST}" CONTENT "${_cudaq_pulse_dialect_lines}\n")

set(_cudaq_pulse_dialect_source_inc
  "${CUDAQ_PULSE_BINARY_DIR}/core/mlir/include/cudaq-pulse/Dialect/DialectSource.inc")
add_custom_command(
  OUTPUT "${_cudaq_pulse_dialect_source_inc}"
  COMMAND "${CMAKE_COMMAND}" "-DROOT=${CUDAQ_PULSE_SOURCE_DIR}"
    "-DLLVM_VERSION=${LLVM_PACKAGE_VERSION}" "-DOUTPUT=${_cudaq_pulse_dialect_source_inc}"
    -P "${CUDAQ_PULSE_SOURCE_DIR}/cmake/write_dialect_source.cmake"
  DEPENDS ${CUDAQ_PULSE_DIALECT_DEFINITIONS} "${CUDAQ_PULSE_DIALECT_LIST}"
    "${CUDAQ_PULSE_SOURCE_DIR}/cmake/write_dialect_source.cmake"
  COMMENT "Hashing the dialect definitions into DialectSource.inc")
add_custom_target(CudaqPulseDialectSourceInc DEPENDS "${_cudaq_pulse_dialect_source_inc}")
