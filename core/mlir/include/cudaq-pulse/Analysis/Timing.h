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

#include "cudaq-pulse/Analysis/Variant.h"

// Included once per variant (see Variant.h).
#if PULSE_COUPLED == 0
#ifdef CUDAQ_PULSE_ANALYSIS_TIMING_H
#define CUDAQ_PULSE_TIMING_SEEN
#else
#define CUDAQ_PULSE_ANALYSIS_TIMING_H
#endif
#else
#ifdef CUDAQ_PULSE_ANALYSIS_TIMING_COUPLED_H
#define CUDAQ_PULSE_TIMING_SEEN
#else
#define CUDAQ_PULSE_ANALYSIS_TIMING_COUPLED_H
#endif
#endif
#ifndef CUDAQ_PULSE_TIMING_SEEN

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"

#include <optional>
#include <utility>
#include <vector>

namespace pulse {
PULSE_VARIANT_OPEN

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
#if PULSE_COUPLED
  if (kCoupled)
    if (auto timed = mlir::dyn_cast<DriveBasebandOp>(op))
      return fn(timed);
#endif
  return otherwise;
}

inline std::optional<int64_t> startCycles(mlir::Operation *op) {
  return withTiming(op, std::optional<int64_t>(),
                    [](auto timed) { return timed.getStartCycles(); });
}

inline std::optional<int64_t> durationCycles(mlir::Operation *op) {
  return withTiming(op, std::optional<int64_t>(),
                    [](auto timed) { return timed.getDurationCycles(); });
}

/// The periods of `op`, one per enclosing loop it is scheduled in; empty
/// outside loops or when it is not scheduled.
inline llvm::ArrayRef<int64_t> periodCycles(mlir::Operation *op) {
  return withTiming(op, llvm::ArrayRef<int64_t>(),
                    [](auto timed) { return timed.getPeriodCycles(); });
}

/// Whether `op` has a start: it is scheduled and runs.
inline bool hasStart(mlir::Operation *op) {
  return startCycles(op).has_value();
}

/// Sets the timing of `op`, a timed op; nullopt or no periods removes that
/// part.
inline void setTiming(mlir::Operation *op, std::optional<int64_t> start,
                      std::optional<int64_t> duration,
                      llvm::ArrayRef<int64_t> periods) {
  withTiming(op, 0, [&](auto timed) {
    timed.setStartCycles(start);
    timed.setDurationCycles(duration);
    timed.setPeriodCycles(periods);
    return 0;
  });
}

inline void setStartCycles(mlir::Operation *op, std::optional<int64_t> start) {
  withTiming(op, 0, [&](auto timed) {
    timed.setStartCycles(start);
    return 0;
  });
}

inline void setPeriodCycles(mlir::Operation *op,
                            llvm::ArrayRef<int64_t> periods) {
  withTiming(op, 0, [&](auto timed) {
    timed.setPeriodCycles(periods);
    return 0;
  });
}

/// The start of `op` in iteration `iteration` of its enclosing loops, if it
/// is scheduled.
inline std::optional<int64_t>
scheduledStart(mlir::Operation *op, llvm::ArrayRef<int64_t> iteration) {
  return withTiming(
      op, std::optional<int64_t>(), [&](auto timed) -> std::optional<int64_t> {
        std::optional<int64_t> start = timed.getStartCycles();
        if (!start)
          return std::nullopt;
        int64_t time = *start;
        for (auto [k, period] : llvm::zip(iteration, timed.getPeriodCycles()))
          time += k * period;
        return time;
      });
}

inline std::optional<int64_t> scheduledDuration(mlir::Operation *op) {
  return durationCycles(op);
}

/// A readout's number in the kernel, written when the module is built from
/// a kernel in which a pass can copy a readout (one in a loop, or one after
/// an if or a select), or, in a module without it, when scheduling first
/// copies a readout of its function; without it, its place among the
/// readouts.
inline constexpr llvm::StringLiteral kReadoutNumberAttr =
    "pulse.readout_number";
/// How a readout's loop iterations map to the kernel's, written when a loop
/// around it is peeled or unrolled: per loop of the kernel around it,
/// outermost first, three numbers (depth, scale, offset). Its iteration is
/// scale * k + offset, k the iteration of its enclosing loop at `depth`
/// (outermost 0), or offset where `depth` is -1 (that loop's iteration was
/// copied out of it). Without it, its loops are the kernel's.
inline constexpr llvm::StringLiteral kReadoutIterationAttr =
    "pulse.readout_iteration";

/// Number the readouts in `scope` in walk order from `first`, unless one of
/// them is numbered already: before anything copies one (an op copied into
/// both arms of an if, a peeled or unrolled loop), every copy then keeps
/// the kernel's number.
inline void numberReadouts(mlir::Operation *scope, int64_t first = 0) {
  auto isReadout = [](mlir::Operation *op) {
    return op->getName().getStringRef() == "pulse.readout" ||
           op->getName().getStringRef() == "pulse.iq_acquire";
  };
  llvm::SmallVector<mlir::Operation *> readouts;
  bool numbered = false;
  scope->walk([&](mlir::Operation *op) {
    if (isReadout(op)) {
      readouts.push_back(op);
      numbered = numbered || op->hasAttr(kReadoutNumberAttr);
    }
  });
  if (numbered)
    return;
  auto i64 = mlir::IntegerType::get(scope->getContext(), 64);
  for (mlir::Operation *op : readouts)
    op->setAttr(kReadoutNumberAttr, mlir::IntegerAttr::get(i64, first++));
}

/// The number and loop iteration the kernel gave readout `op` executed in
/// iteration `iteration` of its enclosing loops, `position` being its place
/// among the readouts in walk order.
inline std::pair<int64_t, std::vector<int64_t>>
kernelReadoutKey(mlir::Operation *op, llvm::ArrayRef<int64_t> iteration,
                 int64_t position) {
  auto number = op->getAttrOfType<mlir::IntegerAttr>(kReadoutNumberAttr);
  auto map = op->getAttrOfType<mlir::DenseI64ArrayAttr>(kReadoutIterationAttr);
  std::pair<int64_t, std::vector<int64_t>> key{
      number ? number.getInt() : position, {}};
  if (!map) {
    key.second.assign(iteration.begin(), iteration.end());
    return key;
  }
  for (size_t i = 0; i + 2 < static_cast<size_t>(map.size()); i += 3) {
    int64_t depth = map[i];
    int64_t k = depth >= 0 && depth < static_cast<int64_t>(iteration.size())
                    ? iteration[depth]
                    : 0;
    key.second.push_back(map[i + 1] * k + map[i + 2]);
  }
  return key;
}

PULSE_VARIANT_CLOSE
} // namespace pulse

#endif // CUDAQ_PULSE_TIMING_SEEN
#undef CUDAQ_PULSE_TIMING_SEEN
