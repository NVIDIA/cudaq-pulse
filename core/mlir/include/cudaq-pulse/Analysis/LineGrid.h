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
// attributes, and reading its grid is one pass over its attributes. Otherwise
// `pulse.line_grid` (i64) is the grid of every line, and
// `pulse.drive_line_grid` / `pulse.readout_line_grid` (i64 arrays indexed by
// qudit, 0 for the default) override it per line; they are read only beside
// `pulse.line_grid`.
//
// Include after the Pulse dialect op and type headers and LineTrace.h.

#pragma once

#include "cudaq-pulse/Analysis/LineTrace.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include <numeric>

namespace pulse {

/// The attributes of a module that put its lines on other clocks or
/// controllers, read in one pass over its attributes: a module without them
/// pays only that.
struct ClockAttributes {
  mlir::IntegerAttr grid;
  mlir::DenseI64ArrayAttr driveGrid, readoutGrid;
  mlir::DictionaryAttr delays;
  mlir::DictionaryAttr controllers;

  ClockAttributes() = default;

  explicit ClockAttributes(mlir::ModuleOp module) {
    if (!module)
      return;
    for (mlir::NamedAttribute attr :
         module->getDiscardableAttrDictionary().getValue()) {
      llvm::StringRef name = attr.getName().getValue();
      if (!name.consume_front("pulse."))
        continue;
      mlir::Attribute value = attr.getValue();
      if (name == "line_grid")
        grid = mlir::dyn_cast<mlir::IntegerAttr>(value);
      else if (name == "drive_line_grid")
        driveGrid = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(value);
      else if (name == "readout_line_grid")
        readoutGrid = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(value);
      else if (name == "line_delays")
        delays = mlir::dyn_cast<mlir::DictionaryAttr>(value);
      else if (name == "controllers")
        controllers = mlir::dyn_cast<mlir::DictionaryAttr>(value);
    }
  }
};

class LineGrid {
public:
  LineGrid() = default;

  explicit LineGrid(mlir::ModuleOp module)
      : LineGrid(ClockAttributes(module)) {}

  explicit LineGrid(const ClockAttributes &attrs) {
    if (!attrs.grid)
      return;
    fallback = std::max<int64_t>(1, attrs.grid.getInt());
    drive = attrs.driveGrid;
    readout = attrs.readoutGrid;
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
