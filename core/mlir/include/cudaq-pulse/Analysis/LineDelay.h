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

#include "cudaq-pulse/Analysis/Variant.h"

#include "cudaq-pulse/Analysis/LineGrid.h"
#include "cudaq-pulse/Analysis/LineTrace.h"
#include "cudaq-pulse/Analysis/Timing.h"
#include "mlir/IR/BuiltinOps.h"

namespace pulse {
PULSE_VARIANT_OPEN

/// The delay of each line of a module's control system, as `clocks` reads
/// it.
class LineDelays {
public:
  explicit LineDelays(const ClockAttributes &clocks)
      : drive(clocks.driveDelay), readout(clocks.readoutDelay) {}

  /// The delay of the line scheduled `op` plays on: 0 for an op no delay
  /// moves.
  int64_t of(mlir::Operation *op) {
    if (!(mlir::isa<DriveOp, ReadoutOp, WaitOp>(op) ||
          (kCoupled && mlir::isa<DriveBasebandOp>(op))) ||
        op->hasAttr("classical.variable"))
      return 0;
    mlir::Operation *allocation = trace(op->getOperand(0));
    if (!allocation)
      return 0;
    if (kCoupled && mlir::isa<GetCouplingLineOp>(allocation))
      return couplingFact(allocation, "delay_cycles", 0);
    int64_t qudit = quditOf(allocation);
    const llvm::SmallVector<int64_t> &shifts =
        mlir::isa<GetDriveLineOp>(allocation) ? drive : readout;
    if (qudit < 0 || qudit >= static_cast<int64_t>(shifts.size()))
      return 0;
    return shifts[qudit];
  }

private:
  const llvm::SmallVector<int64_t> &drive, &readout;
  AllocationTracer trace;
};

/// Shift the scheduled ops of every delayed line in `function` by its delay
/// in `clocks`, its module's. Whether some op moved.
inline bool applyLineDelays(mlir::Operation *function,
                            const ClockAttributes &clocks) {
  if (!clocks.delayed())
    return false;
  bool moved = false;
  LineDelays lines(clocks);
  function->walk([&](mlir::Operation *op) {
    std::optional<int64_t> start = startCycles(op);
    if (!start)
      return;
    int64_t shift = lines.of(op);
    if (shift != 0) {
      setStartCycles(op, *start + shift);
      moved = true;
    }
  });
  return moved;
}

PULSE_VARIANT_CLOSE
} // namespace pulse
