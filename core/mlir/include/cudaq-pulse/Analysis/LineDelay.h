/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// The calibrated delay of each line, in whole program cycles: every drive,
// readout and wait on the line starts that much later than scheduled.
//
// A module without delays has no `pulse.line_delays` attribute, and reading
// it is one attribute lookup. Otherwise it is a dictionary whose `drive` and
// `readout` entries (i64 arrays indexed by qudit) give each line's delay.
// A wait whose length comes from a classical answer is a barrier of every
// line and keeps its time.
//
// Include after the Pulse dialect op and type headers and LineTrace.h.

#pragma once

#include "cudaq-pulse/Analysis/LineTrace.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

namespace pulse {

/// Shift the scheduled ops of every delayed line in `function` by its delay.
inline void applyLineDelays(mlir::Operation *function) {
  auto module = function->getParentOfType<mlir::ModuleOp>();
  auto delays =
      module ? module->getAttrOfType<mlir::DictionaryAttr>("pulse.line_delays")
             : mlir::DictionaryAttr();
  if (!delays)
    return;
  auto drive = delays.getAs<mlir::DenseI64ArrayAttr>("drive");
  auto readout = delays.getAs<mlir::DenseI64ArrayAttr>("readout");
  mlir::MLIRContext *context = function->getContext();
  auto i64 = mlir::IntegerType::get(context, 64);
  AllocationTracer trace;
  function->walk([&](mlir::Operation *op) {
    if (!mlir::isa<DriveOp, ReadoutOp, WaitOp>(op) ||
        op->hasAttr("classical.variable"))
      return;
    auto start = op->getAttrOfType<mlir::IntegerAttr>("start_cycles");
    if (!start)
      return;
    mlir::Operation *allocation = trace(op->getOperand(0));
    if (!allocation)
      return;
    auto qubit = allocation->getAttrOfType<mlir::IntegerAttr>("qubit");
    mlir::DenseI64ArrayAttr shifts =
        mlir::isa<GetDriveLineOp>(allocation) ? drive : readout;
    if (!qubit || !shifts || qubit.getInt() < 0 ||
        qubit.getInt() >= static_cast<int64_t>(shifts.size()))
      return;
    int64_t shift = shifts[qubit.getInt()];
    if (shift != 0)
      op->setAttr("start_cycles",
                  mlir::IntegerAttr::get(i64, start.getInt() + shift));
  });
}

} // namespace pulse
