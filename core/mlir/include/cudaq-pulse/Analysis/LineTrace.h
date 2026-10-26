/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// Follow a linear pulse value (line or tone) back to the allocation it is a
// version of, through pulse ops and structured control flow.
//
// Include after the Pulse dialect op and type headers.

#pragma once

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace pulse {

/// A `classical` link or call handle: linear and timed, like a line.
inline bool isClassicalType(mlir::Type type) {
  return type.getDialect().getNamespace() == "classical";
}

inline bool isLinearPulseType(mlir::Type type) {
  return mlir::isa<DriveLineType, ReadoutLineType, ToneType>(type) ||
         isClassicalType(type);
}

/// The value `value` is the next version of, or null when `value` comes
/// straight from an allocation (or from something this analysis does not
/// model).
///
/// - An `scf.for` iteration argument and result continue the matching init
///   operand.
/// - An `scf.if` result continues the matching `then` yield operand; the
///   `else` arm yields a version of the same allocation.
/// - A `pulse.schedule_block` argument continues its operand, and a result
///   continues the value the body yields for it.
/// - Result `k` of type `T` of any other op continues its `k`-th operand of
///   type `T` (a drive's line result continues its line operand, a sync's
///   third line continues its third input line).
inline mlir::Value linearPredecessor(mlir::Value value) {
  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Operation *parent = argument.getOwner()->getParentOp();
    if (auto block = mlir::dyn_cast_or_null<ScheduleBlockOp>(parent))
      return block.getInputs()[argument.getArgNumber()];
    auto loop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(parent);
    if (!loop || argument.getArgNumber() == 0)
      return {};
    return loop.getInitArgs()[argument.getArgNumber() - 1];
  }
  auto result = mlir::cast<mlir::OpResult>(value);
  mlir::Operation *op = result.getOwner();
  unsigned index = result.getResultNumber();
  if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op))
    return loop.getInitArgs()[index];
  if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op))
    return branch.thenYield().getOperand(index);
  // A schedule block result continues the value its body yields.
  if (auto block = mlir::dyn_cast<ScheduleBlockOp>(op))
    return block.getBody().front().getTerminator()->getOperand(index);
  if (mlir::isa<GetDriveLineOp, GetReadoutLineOp>(op) ||
      op->getName().getStringRef() == "classical.get_link")
    return {};
  mlir::Type type = value.getType();
  unsigned rank = 0;
  for (mlir::OpResult earlier : op->getResults().take_front(index))
    rank += earlier.getType() == type;
  for (mlir::Value operand : op->getOperands())
    if (operand.getType() == type && rank-- == 0)
      return operand;
  return {};
}

/// The `pulse.get_drive_line` / `pulse.get_readout_line` op whose line or
/// tone `value` is a version of (or the `classical.get_link` of a link), or
/// null.
inline mlir::Operation *traceAllocation(mlir::Value value) {
  while (value) {
    if (mlir::Operation *def = value.getDefiningOp())
      if (mlir::isa<GetDriveLineOp, GetReadoutLineOp>(def) ||
          def->getName().getStringRef() == "classical.get_link")
        return def;
    value = linearPredecessor(value);
  }
  return nullptr;
}

} // namespace pulse
