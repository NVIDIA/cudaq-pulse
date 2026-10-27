/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// The time grid of each line: the program cycles between instants its
// controller's own clock can start an operation at.
//
// A module whose lines all start on every program cycle has no grid
// attributes. Otherwise `pulse.line_grid` (i64) is the grid of every line,
// and `pulse.drive_line_grid` / `pulse.readout_line_grid` (i64 arrays indexed
// by qudit, 0 for the default) override it per line.
//
// Include after the Pulse dialect op and type headers and LineTrace.h.

#pragma once

#include "cudaq-pulse/Analysis/LineTrace.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include <numeric>

namespace pulse {

class LineGrid {
public:
  LineGrid() = default;

  explicit LineGrid(mlir::ModuleOp module) {
    if (!module)
      return;
    if (auto attr = module->getAttrOfType<mlir::IntegerAttr>("pulse.line_grid"))
      fallback = std::max<int64_t>(1, attr.getInt());
    drive =
        module->getAttrOfType<mlir::DenseI64ArrayAttr>("pulse.drive_line_grid");
    readout = module->getAttrOfType<mlir::DenseI64ArrayAttr>(
        "pulse.readout_line_grid");
    common = fallback;
    for (mlir::DenseI64ArrayAttr grids : {drive, readout})
      if (grids)
        for (int64_t g : grids.asArrayRef())
          if (g > 0)
            common = std::lcm(common, g);
  }

  /// Whether some line has a grid coarser than one program cycle.
  bool enabled() const { return common > 1; }

  /// The least common multiple of every line's grid: where all can meet.
  int64_t lcm() const { return common; }

  /// Whether lines have different grids, so `of` traces them.
  bool perLine() const { return drive || readout; }

  /// The grid of the line `line` is a version of; `trace` must not outlive
  /// a change to the IR.
  int64_t of(mlir::Value line, AllocationTracer &trace) const {
    if (!perLine())
      return fallback;
    mlir::Operation *allocation = trace(line);
    if (!allocation)
      return fallback;
    auto qubit = allocation->getAttrOfType<mlir::IntegerAttr>("qubit");
    mlir::DenseI64ArrayAttr grids =
        mlir::isa<GetDriveLineOp>(allocation) ? drive : readout;
    if (!qubit || !grids || qubit.getInt() < 0 ||
        qubit.getInt() >= static_cast<int64_t>(grids.size()))
      return fallback;
    int64_t g = grids[qubit.getInt()];
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
  mlir::DenseI64ArrayAttr drive, readout;
};

} // namespace pulse
