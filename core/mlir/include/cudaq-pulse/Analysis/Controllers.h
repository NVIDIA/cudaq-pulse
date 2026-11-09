// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The controllers of a program whose lines several controllers play: which
// one plays each line, and what crossing between two of them costs.
//
// The module attribute `pulse.controllers` is a dictionary of
// `drive` / `readout` (i64 arrays indexed by qudit: the index of the
// controller that plays the line, -1 for none), `outcome_latency` (i64, n*n,
// reader-major: how long after a readout on controller r ends its outcome is
// usable on controller c, in program cycles), `spread` (i64, n*n: how far
// apart in time two controllers' starts may be, in program cycles, 0 on the
// diagonal) and `uncertainty_ns` (f64, n: how well each controller's start
// is known). A line of no known controller, such as a runtime qudit's, may
// be on any. A module without it has one controller, and reading it is one
// attribute lookup.
//
// Include after the Pulse dialect op and type headers and LineTrace.h.

#pragma once

#include "cudaq-pulse/Analysis/LineTrace.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include <algorithm>
#include <cstdint>

namespace pulse {

class Controllers {
public:
  Controllers() = default;

  explicit Controllers(mlir::ModuleOp module)
      : Controllers(module ? module->getAttrOfType<mlir::DictionaryAttr>(
                                 "pulse.controllers")
                           : mlir::DictionaryAttr()) {}

  /// Of the module's `pulse.controllers` attribute `attr`, if any.
  explicit Controllers(mlir::DictionaryAttr attr) {
    if (!attr)
      return;
    auto uncertainty = attr.getAs<mlir::DenseF64ArrayAttr>("uncertainty_ns");
    auto outcome = attr.getAs<mlir::DenseI64ArrayAttr>("outcome_latency");
    auto spreads = attr.getAs<mlir::DenseI64ArrayAttr>("spread");
    if (!uncertainty || !outcome || !spreads)
      return;
    int64_t size = uncertainty.size();
    if (size < 2 || outcome.size() != size * size ||
        spreads.size() != size * size)
      return;
    n = size;
    uncertaintyNs.assign(uncertainty.asArrayRef().begin(),
                         uncertainty.asArrayRef().end());
    latency.assign(outcome.asArrayRef().begin(), outcome.asArrayRef().end());
    spreadOf.assign(spreads.asArrayRef().begin(), spreads.asArrayRef().end());
    drive = attr.getAs<mlir::DenseI64ArrayAttr>("drive");
    readout = attr.getAs<mlir::DenseI64ArrayAttr>("readout");
  }

  /// Whether the program's lines are on several controllers.
  bool enabled() const { return n > 0; }

  int64_t count() const { return n; }

  /// The controller that plays the line `line` is a version of, or -1.
  int64_t of(mlir::Value line, AllocationTracer &trace) const {
    mlir::Operation *allocation = trace(line);
    if (!allocation)
      return -1;
    int64_t qudit = quditOf(allocation);
    mlir::DenseI64ArrayAttr owners =
        mlir::isa<GetDriveLineOp>(allocation) ? drive : readout;
    if (!owners || qudit < 0 || qudit >= static_cast<int64_t>(owners.size()))
      return -1;
    int64_t owner = owners[qudit];
    return owner >= 0 && owner < n ? owner : -1;
  }

  /// After a readout on `reader` (-1: any) ends, when its outcome is usable
  /// on the reader itself.
  int64_t own(int64_t reader) const {
    if (reader >= 0)
      return at(latency, reader, reader);
    int64_t most = 0;
    for (int64_t r = 0; r < n; ++r)
      most = std::max(most, at(latency, r, r));
    return most;
  }

  /// How much later than on `reader` itself the outcome of its readout is
  /// usable on `user`; either may be -1, for any.
  int64_t extra(int64_t reader, int64_t user) const {
    int64_t most = 0;
    for (int64_t r = reader < 0 ? 0 : reader; r < (reader < 0 ? n : reader + 1);
         ++r)
      for (int64_t c = user < 0 ? 0 : user; c < (user < 0 ? n : user + 1); ++c)
        most = std::max(most, at(latency, r, c) - at(latency, r, r));
    return most;
  }

  /// How far apart the starts of `a` and `b` may be; either may be -1, for
  /// any other controller.
  int64_t spread(int64_t a, int64_t b) const {
    if (a >= 0 && b >= 0)
      return at(spreadOf, a, b);
    int64_t most = 0;
    for (int64_t x = a < 0 ? 0 : a; x < (a < 0 ? n : a + 1); ++x)
      for (int64_t y = b < 0 ? 0 : b; y < (b < 0 ? n : b + 1); ++y)
        most = std::max(most, at(spreadOf, x, y));
    return most;
  }

  /// How well the start of `controller` is known, in ns.
  double uncertainty(int64_t controller) const {
    return uncertaintyNs[controller];
  }

private:
  int64_t at(const llvm::SmallVector<int64_t> &table, int64_t a,
             int64_t b) const {
    return table[a * n + b];
  }

  int64_t n = 0;
  llvm::SmallVector<double> uncertaintyNs;
  llvm::SmallVector<int64_t> latency, spreadOf;
  mlir::DenseI64ArrayAttr drive, readout;
};

} // namespace pulse
