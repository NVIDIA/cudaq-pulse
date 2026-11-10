// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The SSA types `classical.wait` gives for the results of a callable: a
// leading `i1` in timeout mode, then per result descriptor `i1` for each
// bit, `i64` for each integer or level, and `f64` for each float or IQ
// component.
//
// Include after the Classical op headers.

#pragma once

#include "cudaq-pulse/Dialect/Classical/Schema.h"

#include "mlir/IR/Builders.h"

namespace classical {

inline llvm::SmallVector<mlir::Type> answerTypes(CallableOp callable,
                                                 bool timeout) {
  mlir::Builder b(callable.getContext());
  llvm::SmallVector<mlir::Type> types;
  if (timeout || callable.getMode() == "timeout")
    types.push_back(b.getI1Type());
  for (mlir::Attribute entry : callable.getResults()) {
    Descriptor desc =
        *parseDescriptor(mlir::cast<mlir::StringAttr>(entry).getValue());
    mlir::Type element = desc.kind == Kind::Bits ? mlir::Type(b.getI1Type())
                         : desc.isFloat()        ? mlir::Type(b.getF64Type())
                                                 : mlir::Type(b.getI64Type());
    int64_t n = desc.kind == Kind::IQ ? 2 * desc.count : desc.count;
    types.append(static_cast<size_t>(n), element);
  }
  return types;
}

} // namespace classical
