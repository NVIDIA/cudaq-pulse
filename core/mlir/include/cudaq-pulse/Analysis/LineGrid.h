// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The time grid of each line: the program cycles between instants its
// controller's own clock can start an operation.
//
// A module's control system (`pulse.control_system`) gives each controller's
// grid (`grid_cycles`, 1 by default) and the controller of each line it
// lists. With one controller every line is on its grid; with several, a line
// it does not list (a runtime qudit's) is on the least common multiple of
// theirs. A module whose lines all start on every program cycle pays one
// attribute lookup to find out.
//
// Include after the Pulse dialect op and type headers and LineTrace.h.

#pragma once

#include "cudaq-pulse/Analysis/LineTrace.h"
#include "cudaq-pulse/Dialect/Pulse/PulseAttrs.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinOps.h"
#include <numeric>

namespace pulse {

/// What a module's control system says of its lines, by qudit, read once.
struct ClockAttributes {
  ControlSystemAttr system;
  /// The least common multiple of the controllers' grids.
  int64_t grid = 1;
  /// By qudit, the grid of its drive and readout line (0: `grid`), only
  /// when the lines are on several controllers' grids.
  llvm::SmallVector<int64_t> driveGrid, readoutGrid;
  /// By qudit, the delay of its drive and readout line; empty without one.
  llvm::SmallVector<int64_t> driveDelay, readoutDelay;
  /// By qudit, the controller of its drive and readout line (-1: none), on
  /// several controllers.
  llvm::SmallVector<int64_t> driveOwner, readoutOwner;

  ClockAttributes() = default;

  explicit ClockAttributes(mlir::ModuleOp module)
      : system(module ? controlSystemOf(module) : ControlSystemAttr()) {
    if (!system)
      return;
    llvm::ArrayRef<ControllerAttr> controllers = system.getControllers();
    for (ControllerAttr controller : controllers)
      grid = std::lcm(grid, controller.getGridCycles());
    bool several = controllers.size() > 1;
    bool gridded = several && grid > 1;
    for (LineAttr line : system.getLines()) {
      bool drive = line.getKind() == LineKind::Drive;
      int64_t controller = line.getController();
      for (int64_t qudit : line.getQudits()) {
        if (line.getDelayCycles() != 0)
          set(drive ? driveDelay : readoutDelay, qudit, line.getDelayCycles(),
              0);
        if (gridded)
          set(drive ? driveGrid : readoutGrid, qudit,
              controllers[controller].getGridCycles(), 0);
        if (several)
          set(drive ? driveOwner : readoutOwner, qudit, controller, -1);
      }
    }
  }

  /// Whether some line has a delay.
  bool delayed() const { return !driveDelay.empty() || !readoutDelay.empty(); }

private:
  static void set(llvm::SmallVector<int64_t> &table, int64_t qudit,
                  int64_t value, int64_t fill) {
    if (table.size() <= static_cast<size_t>(qudit))
      table.resize(qudit + 1, fill);
    table[qudit] = value;
  }
};

class LineGrid {
public:
  LineGrid() = default;

  explicit LineGrid(mlir::ModuleOp module)
      : LineGrid(ClockAttributes(module)) {}

  explicit LineGrid(const ClockAttributes &attrs)
      : fallback(attrs.grid), common(attrs.grid), drive(attrs.driveGrid),
        readout(attrs.readoutGrid) {}

  /// Whether some line has a grid coarser than one program cycle.
  bool enabled() const { return common > 1; }

  /// The least common multiple of every line's grid: where all can meet.
  int64_t lcm() const { return common; }

  /// Whether lines have different grids, so `of` traces them.
  bool perLine() const { return !drive.empty() || !readout.empty(); }

  /// The grid of the line `line` is a version of; `trace` must not outlive
  /// a change to the IR.
  int64_t of(mlir::Value line, AllocationTracer &trace) const {
    if (!perLine())
      return fallback;
    mlir::Operation *allocation = trace(line);
    if (!allocation)
      return fallback;
    int64_t qudit = quditOf(allocation);
    const llvm::SmallVector<int64_t> &grids =
        mlir::isa<GetDriveLineOp>(allocation) ? drive : readout;
    if (qudit < 0 || qudit >= static_cast<int64_t>(grids.size()))
      return fallback;
    int64_t g = grids[qudit];
    return g > 0 ? g : fallback;
  }

  static int64_t roundUp(int64_t time, int64_t grid) {
    int64_t r = time % grid;
    return r == 0 ? time : time + (r > 0 ? grid - r : -r);
  }

  static int64_t roundDown(int64_t time, int64_t grid) {
    int64_t r = time % grid;
    return r == 0 ? time : time - (r > 0 ? r : grid + r);
  }

private:
  int64_t fallback = 1;
  int64_t common = 1;
  llvm::SmallVector<int64_t> drive, readout;
};

} // namespace pulse
