// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The calibrated delay of each line, in whole program cycles: every drive,
// readout and wait on the line starts that much later than scheduled.
//
// A module's control system (`pulse.control_system`) gives each delayed
// line's `delay_cycles`, read with its other line attributes (see
// ClockAttributes in LineGrid.h). A wait whose length comes from a classical
// answer is a barrier of every line and keeps its time.
//
// Include after the Pulse and Classical dialect op headers.

#pragma once

#include "cudaq-pulse/Analysis/LineGrid.h"
#include "cudaq-pulse/Analysis/LineTrace.h"
#include "cudaq-pulse/Analysis/Timing.h"
#include "mlir/IR/BuiltinOps.h"

namespace pulse {

/// Shift the scheduled ops of every delayed line in `function` by its delay
/// in `clocks`, its module's. Whether some op moved.
inline bool applyLineDelays(mlir::Operation *function,
                            const ClockAttributes &clocks) {
  if (!clocks.delayed())
    return false;
  bool moved = false;
  AllocationTracer trace;
  function->walk([&](mlir::Operation *op) {
    if (!mlir::isa<DriveOp, ReadoutOp, WaitOp>(op) ||
        op->hasAttr("classical.variable"))
      return;
    std::optional<int64_t> start = startCycles(op);
    if (!start)
      return;
    mlir::Operation *allocation = trace(op->getOperand(0));
    if (!allocation)
      return;
    int64_t qudit = quditOf(allocation);
    const llvm::SmallVector<int64_t> &shifts =
        mlir::isa<GetDriveLineOp>(allocation) ? clocks.driveDelay
                                              : clocks.readoutDelay;
    if (qudit < 0 || qudit >= static_cast<int64_t>(shifts.size()))
      return;
    int64_t shift = shifts[qudit];
    if (shift != 0) {
      setStartCycles(op, *start + shift);
      moved = true;
    }
  });
  return moved;
}

} // namespace pulse
