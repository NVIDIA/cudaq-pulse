/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// Schedule attributes of timed pulse ops inside structured control flow.
//
// A scheduled drive, readout or wait carries `start_vtu` and `duration_vtu`.
// Inside `scf.for` loops it also carries `period_vtu`, one entry per
// enclosing loop (outermost first): in iteration (k_0, ..., k_n) of those
// loops the op starts at start_vtu + sum_l k_l * period_vtu[l].

#pragma once

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"

#include <optional>

namespace pulse {

inline std::optional<int64_t> constantIndex(mlir::Value value) {
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
    if (auto attr = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
      return attr.getInt();
  return std::nullopt;
}

/// The number of iterations of `loop`, if its bounds are constants.
inline std::optional<int64_t> constantTripCount(mlir::scf::ForOp loop) {
  auto lb = constantIndex(loop.getLowerBound());
  auto ub = constantIndex(loop.getUpperBound());
  auto step = constantIndex(loop.getStep());
  if (!lb || !ub || !step || *step <= 0)
    return std::nullopt;
  return *ub > *lb ? (*ub - *lb + *step - 1) / *step : 0;
}

/// The `scf.for` ops enclosing `op`, outermost first; with `root`, only
/// those inside it.
inline llvm::SmallVector<mlir::scf::ForOp, 2>
enclosingLoops(mlir::Operation *op, mlir::Operation *root = nullptr) {
  llvm::SmallVector<mlir::scf::ForOp, 2> loops;
  for (mlir::Operation *parent = op->getParentOp(); parent && parent != root;
       parent = parent->getParentOp())
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(parent))
      loops.insert(loops.begin(), loop);
  return loops;
}

/// The start of `op` in iteration `iteration` of its enclosing loops, if it
/// is scheduled.
inline std::optional<int64_t>
scheduledStart(mlir::Operation *op, llvm::ArrayRef<int64_t> iteration) {
  auto start = op->getAttrOfType<mlir::IntegerAttr>("start_vtu");
  if (!start)
    return std::nullopt;
  int64_t time = start.getInt();
  if (auto periods = op->getAttrOfType<mlir::DenseI64ArrayAttr>("period_vtu"))
    for (auto [k, period] : llvm::zip(iteration, periods.asArrayRef()))
      time += k * period;
  return time;
}

inline std::optional<int64_t> scheduledDuration(mlir::Operation *op) {
  if (auto duration = op->getAttrOfType<mlir::IntegerAttr>("duration_vtu"))
    return duration.getInt();
  return std::nullopt;
}

} // namespace pulse
