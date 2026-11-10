// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "cudaq-pulse/Dialect/Classical/ClassicalDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Classical/ClassicalTypes.h.inc"

#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Classical/ClassicalOps.h.inc"

using namespace mlir;

#include "cudaq-pulse/Dialect/Classical/ClassicalDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Classical/ClassicalTypes.cpp.inc"

namespace classical {

void ClassicalDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "cudaq-pulse/Dialect/Classical/ClassicalTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "cudaq-pulse/Dialect/Classical/ClassicalOps.cpp.inc"
      >();
}

} // namespace classical
