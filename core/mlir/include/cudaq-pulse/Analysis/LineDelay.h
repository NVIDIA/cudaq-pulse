// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The calibrated delay of each line, in whole program cycles: every drive,
// readout and wait on the line starts that much later than scheduled.
//
// A module without delays has no `pulse.line_delays` attribute (read with its
// other clock attributes, see ClockAttributes in LineGrid.h). Otherwise it is a
// dictionary whose `drive` and `readout` entries (i64 arrays indexed by qudit)
// give each line's delay. A wait whose length comes from a classical answer is
// a barrier of every line and keeps its time.
//
// Include after the Pulse and Classical dialect op headers.

#pragma once

#include "cudaq-pulse/Analysis/LineTrace.h"
#include "cudaq-pulse/Analysis/Timing.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

namespace pulse {

/// Shift the scheduled ops of every delayed line in `function` by its delay
/// in `delays`, its module's `pulse.line_delays` (none when null).
inline void applyLineDelays(mlir::Operation *function,
                            mlir::DictionaryAttr delays) {
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
    auto start = startCyclesAttr(op);
    if (!start)
      return;
    mlir::Operation *allocation = trace(op->getOperand(0));
    if (!allocation)
      return;
    int64_t qudit = quditOf(allocation);
    mlir::DenseI64ArrayAttr shifts =
        mlir::isa<GetDriveLineOp>(allocation) ? drive : readout;
    if (!shifts || qudit < 0 || qudit >= static_cast<int64_t>(shifts.size()))
      return;
    int64_t shift = shifts[qudit];
    if (shift != 0)
      setStartCycles(op, mlir::IntegerAttr::get(i64, start.getInt() + shift));
  });
}

} // namespace pulse
