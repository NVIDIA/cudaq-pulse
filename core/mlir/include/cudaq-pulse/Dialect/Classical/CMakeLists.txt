# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(LLVM_TARGET_DEFINITIONS ClassicalOps.td)
mlir_tablegen(ClassicalOps.h.inc -gen-op-decls)
mlir_tablegen(ClassicalOps.cpp.inc -gen-op-defs)
mlir_tablegen(ClassicalDialect.h.inc -gen-dialect-decls)
mlir_tablegen(ClassicalDialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS ClassicalTypes.td)
mlir_tablegen(ClassicalTypes.h.inc -gen-typedef-decls)
mlir_tablegen(ClassicalTypes.cpp.inc -gen-typedef-defs)

add_public_tablegen_target(CudaqPulseClassicalIncGen)

set(LLVM_TARGET_DEFINITIONS ClassicalOps.td)
mlir_tablegen(ClassicalDialect.md -gen-dialect-doc -dialect=classical)
add_custom_target(CudaqPulseClassicalDocGen
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/ClassicalDialect.md)
