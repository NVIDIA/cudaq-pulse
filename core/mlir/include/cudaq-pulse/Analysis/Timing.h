// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The schedule of timed ops inside structured control flow.
//
// A scheduled drive, readout, wait, frame change, schedule block or classical
// submit or wait holds `start_cycles` and `duration_cycles` in its properties
// (PulseTiming.td). Inside `scf.for` loops it also holds `period_cycles`, one
// entry per enclosing loop (outermost first): in iteration (k_0, ..., k_n) of
// those loops the op starts at start_cycles + sum_l k_l * period_cycles[l].
//
// Include after the Pulse and Classical dialect op headers.

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

/// constantInteger for a value that is not a plain integer constant: integer
/// arithmetic on constants is folded with the ops' own folders.
[[gnu::noinline]] inline std::optional<int64_t>
foldedInteger(mlir::Value value) {
  mlir::Operation *op = value.getDefiningOp();
  if (!op)
    return std::nullopt;
  if (auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(op)) {
    if (auto attr = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
      return attr.getInt();
    return std::nullopt;
  }
  if (!mlir::isa<mlir::arith::ArithDialect>(op->getDialect()) ||
      op->getNumResults() != 1 || op->getNumRegions() != 0)
    return std::nullopt;
  llvm::SmallVector<mlir::Attribute, 2> operands;
  for (mlir::Value operand : op->getOperands()) {
    if (!mlir::isa<mlir::IntegerType, mlir::IndexType>(operand.getType()))
      return std::nullopt;
    std::optional<int64_t> known = foldedInteger(operand);
    if (!known)
      return std::nullopt;
    operands.push_back(mlir::IntegerAttr::get(operand.getType(), *known));
  }
  llvm::SmallVector<mlir::OpFoldResult, 1> folded;
  if (mlir::failed(op->fold(operands, folded)) || folded.size() != 1)
    return std::nullopt;
  if (auto same = mlir::dyn_cast<mlir::Value>(folded.front()))
    return same == value ? std::nullopt : foldedInteger(same);
  if (auto attr = mlir::dyn_cast<mlir::IntegerAttr>(
          mlir::cast<mlir::Attribute>(folded.front())))
    return attr.getInt();
  return std::nullopt;
}

/// The integer `value` when it is a constant, or integer arithmetic on
/// constants not folded yet (as lowering a select leaves it in each arm):
/// what folding it would give.
inline std::optional<int64_t> constantInteger(mlir::Value value) {
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantIntOp>())
    return constant.value();
  return foldedInteger(value);
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

/// `fn` applied to `op` as the op class that holds timing properties, or
/// `otherwise` when `op` holds none. The classes are tried in the order they
/// are most common in.
template <typename R, typename Fn>
inline R withTiming(mlir::Operation *op, R otherwise, Fn &&fn) {
  if (auto timed = mlir::dyn_cast<DriveOp>(op))
    return fn(timed);
  if (auto timed = mlir::dyn_cast<WaitOp>(op))
    return fn(timed);
  if (auto timed = mlir::dyn_cast<ReadoutOp>(op))
    return fn(timed);
  if (auto timed = mlir::dyn_cast<classical::SubmitOp>(op))
    return fn(timed);
  if (auto timed = mlir::dyn_cast<classical::WaitOp>(op))
    return fn(timed);
  if (auto timed = mlir::dyn_cast<ShiftPhaseOp>(op))
    return fn(timed);
  if (auto timed = mlir::dyn_cast<SetPhaseOp>(op))
    return fn(timed);
  if (auto timed = mlir::dyn_cast<ShiftFrequencyOp>(op))
    return fn(timed);
  if (auto timed = mlir::dyn_cast<SetFrequencyOp>(op))
    return fn(timed);
  if (auto timed = mlir::dyn_cast<ScheduleBlockOp>(op))
    return fn(timed);
  return otherwise;
}

inline mlir::IntegerAttr startCyclesAttr(mlir::Operation *op) {
  return withTiming(op, mlir::IntegerAttr(),
                    [](auto timed) { return timed.getStartCyclesAttr(); });
}

inline mlir::IntegerAttr durationCyclesAttr(mlir::Operation *op) {
  return withTiming(op, mlir::IntegerAttr(),
                    [](auto timed) { return timed.getDurationCyclesAttr(); });
}

inline mlir::DenseI64ArrayAttr periodCyclesAttr(mlir::Operation *op) {
  return withTiming(op, mlir::DenseI64ArrayAttr(),
                    [](auto timed) { return timed.getPeriodCyclesAttr(); });
}

/// Whether `op` has a start: it is scheduled and runs.
inline bool hasStart(mlir::Operation *op) {
  return static_cast<bool>(startCyclesAttr(op));
}

/// Sets the timing of `op`, a timed op; a null attribute removes that part.
inline void setTiming(mlir::Operation *op, mlir::IntegerAttr start,
                      mlir::IntegerAttr duration,
                      mlir::DenseI64ArrayAttr period) {
  withTiming(op, 0, [&](auto timed) {
    auto &properties = timed.getProperties();
    properties.start_cycles = start;
    properties.duration_cycles = duration;
    properties.period_cycles = period;
    return 0;
  });
}

inline void setStartCycles(mlir::Operation *op, mlir::IntegerAttr start) {
  withTiming(op, 0, [&](auto timed) {
    timed.getProperties().start_cycles = start;
    return 0;
  });
}

inline void setPeriodCycles(mlir::Operation *op,
                            mlir::DenseI64ArrayAttr period) {
  withTiming(op, 0, [&](auto timed) {
    timed.getProperties().period_cycles = period;
    return 0;
  });
}

/// The start of `op` in iteration `iteration` of its enclosing loops, if it
/// is scheduled.
inline std::optional<int64_t>
scheduledStart(mlir::Operation *op, llvm::ArrayRef<int64_t> iteration) {
  return withTiming(
      op, std::optional<int64_t>(), [&](auto timed) -> std::optional<int64_t> {
        mlir::IntegerAttr start = timed.getStartCyclesAttr();
        if (!start)
          return std::nullopt;
        int64_t time = start.getInt();
        if (auto periods = timed.getPeriodCyclesAttr())
          for (auto [k, period] : llvm::zip(iteration, periods.asArrayRef()))
            time += k * period;
        return time;
      });
}

inline std::optional<int64_t> scheduledDuration(mlir::Operation *op) {
  if (auto duration = durationCyclesAttr(op))
    return duration.getInt();
  return std::nullopt;
}

} // namespace pulse
