/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// Module-level verification pass for the Pulse dialect.
// Checks linearity, monotone time, measurement causality, and waveform
// validity beyond what individual op verifiers can check, through structured
// control flow: the arms of an `scf.if` are alternative paths, and an
// `scf.for` body runs once per iteration.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseOps.h.inc"

#include "cudaq-pulse/Analysis/LineTrace.h"
#include "cudaq-pulse/Analysis/Timing.h"

#include <algorithm>
#include <iterator>

namespace {

static std::optional<int64_t> traceConstantI64(mlir::Value v) {
  if (auto cst = v.getDefiningOp<mlir::arith::ConstantIntOp>())
    return cst.value();
  if (auto cst = v.getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
      return ia.getInt();
  }
  return std::nullopt;
}

/// Whether `a` and `b` lie in different arms of some `scf.if`, so at most
/// one of them runs.
static bool mutuallyExclusive(mlir::Operation *a, mlir::Operation *b) {
  llvm::DenseMap<mlir::Operation *, mlir::Region *> armsOfA;
  for (mlir::Region *region = a->getParentRegion(); region;
       region = region->getParentOp()->getParentRegion())
    if (mlir::isa<mlir::scf::IfOp>(region->getParentOp()))
      armsOfA[region->getParentOp()] = region;
  for (mlir::Region *region = b->getParentRegion(); region;
       region = region->getParentOp()->getParentRegion()) {
    auto it = armsOfA.find(region->getParentOp());
    if (it != armsOfA.end() && it->second != region)
      return true;
  }
  return false;
}

static bool isTimedOp(mlir::Operation *op) {
  return mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp>(op);
}

/// The measurement results a branch condition is computed from.
static void conditionMeasurements(mlir::Value value,
                                  llvm::SmallVectorImpl<mlir::Value> &found,
                                  llvm::DenseSet<mlir::Value> &seen) {
  if (!seen.insert(value).second)
    return;
  if (mlir::isa<pulse::MeasurementType>(value.getType())) {
    found.push_back(value);
    return;
  }
  if (mlir::Operation *def = value.getDefiningOp())
    for (mlir::Value operand : def->getOperands())
      conditionMeasurements(operand, found, seen);
}

/// Checks the schedule attributes of one function, walking the program in
/// execution order. Loops are checked on their first two iterations and
/// their last: with one period per line, those bound every iteration.
class TimingChecker {
  pulse::AllocationTracer trace;

public:
  bool hadError = false;

  void checkBlock(mlir::Block &block) {
    for (mlir::Operation &op : block.without_terminator())
      checkOp(&op);
  }

private:
  void error(mlir::Operation *op, const llvm::Twine &message) {
    op->emitError(message);
    hadError = true;
  }

  void checkOp(mlir::Operation *op) {
    if (isTimedOp(op))
      return checkTimed(op);
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op))
      return checkLoop(loop);
    if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op))
      return checkIf(branch);
    for (mlir::Region &region : op->getRegions())
      for (mlir::Block &block : region)
        checkBlock(block);
  }

  void checkTimed(mlir::Operation *op) {
    auto start = pulse::scheduledStart(op, iteration);
    auto duration = pulse::scheduledDuration(op);
    if (!start || !duration)
      return;
    auto periods = op->getAttrOfType<mlir::DenseI64ArrayAttr>("period_vtu");
    if (static_cast<size_t>(periods ? periods.size() : 0) != iteration.size()) {
      error(op, "scheduled operation needs one period_vtu entry per enclosing "
                "scf.for");
      return;
    }
    if (*start < 0) {
      op->emitError("operation is scheduled before time zero: start ")
          << *start;
      hadError = true;
    }
    mlir::Operation *physicalLine = trace(op->getOperand(0));
    if (!physicalLine) {
      error(op, "cannot trace timed operation to a physical line");
      return;
    }
    if (*start < measured) {
      op->emitError("operation in a measurement-conditioned branch starts at ")
          << *start << ", before its measurement completes at " << measured;
      hadError = true;
    }
    auto it = lastEnd.find(physicalLine);
    if (it != lastEnd.end() && *start < it->second) {
      op->emitError("operation overlaps or precedes its physical-line "
                    "predecessor: start ")
          << *start << " < prior end " << it->second;
      hadError = true;
    }
    lastEnd[physicalLine] =
        std::max(lastEnd.lookup(physicalLine), *start + *duration);
  }

  void checkLoop(mlir::scf::ForOp loop) {
    auto trips = pulse::constantTripCount(loop);
    if (!trips) {
      for (mlir::Operation &inner : loop.getBody()->without_terminator())
        if (isTimedOp(&inner) && inner.hasAttr("start_vtu")) {
          error(&inner, "a scheduled loop needs constant bounds");
          return;
        }
      return;
    }
    checkPeriods(loop);
    iteration.push_back(0);
    for (int64_t k = 0; k < *trips; ++k) {
      if (k == 2 && *trips > 3)
        k = *trips - 1;
      iteration.back() = k;
      checkBlock(*loop.getBody());
    }
    iteration.pop_back();
  }

  /// Within one iteration of `loop`, every op on a physical line must move
  /// by the same non-negative period from one iteration to the next.
  void checkPeriods(mlir::scf::ForOp loop) {
    size_t level = iteration.size();
    llvm::DenseMap<mlir::Operation *, int64_t> periodOf;
    loop.getBody()->walk([&](mlir::Operation *op) {
      if (!isTimedOp(op))
        return;
      auto periods = op->getAttrOfType<mlir::DenseI64ArrayAttr>("period_vtu");
      if (!periods || periods.size() <= level)
        return;
      int64_t period = periods[level];
      if (period < 0)
        return error(op, "period_vtu must not be negative");
      mlir::Operation *line = trace(op->getOperand(0));
      auto [it, inserted] = periodOf.try_emplace(line, period);
      if (!inserted && it->second != period)
        error(op, "operations on one line in a loop must share its period");
    });
  }

  void checkIf(mlir::scf::IfOp branch) {
    int64_t savedMeasured = measured;
    llvm::SmallVector<mlir::Value> measurements;
    llvm::DenseSet<mlir::Value> seen;
    conditionMeasurements(branch.getCondition(), measurements, seen);
    for (mlir::Value measurement : measurements)
      if (auto readout = measurement.getDefiningOp<pulse::ReadoutOp>()) {
        size_t depth = pulse::enclosingLoops(readout).size();
        auto start = pulse::scheduledStart(
            readout, llvm::ArrayRef(iteration).take_front(depth));
        auto duration = pulse::scheduledDuration(readout);
        if (start && duration)
          measured = std::max(measured, *start + *duration);
      }

    auto before = lastEnd;
    checkBlock(*branch.thenBlock());
    if (!branch.getElseRegion().empty()) {
      auto afterThen = std::move(lastEnd);
      lastEnd = std::move(before);
      checkBlock(*branch.elseBlock());
      for (auto [line, end] : afterThen)
        lastEnd[line] = std::max(lastEnd.lookup(line), end);
    }
    measured = savedMeasured;
  }

  llvm::DenseMap<mlir::Operation *, int64_t> lastEnd;
  llvm::SmallVector<int64_t, 2> iteration;
  int64_t measured = std::numeric_limits<int64_t>::min();
};

struct PulseVerifyPass
    : public mlir::PassWrapper<PulseVerifyPass,
                               mlir::OperationPass<mlir::ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PulseVerifyPass)

  llvm::StringRef getArgument() const override { return "pulse-verify"; }
  llvm::StringRef getDescription() const override {
    return "Module-level verification of Pulse IR constraints";
  }

  void runOnOperation() override {
    auto module = getOperation();
    bool hadError = false;

    // Check 1: Linearity -- lines must be consumed. The warning names the
    // qubit instead of attaching the op: printing an op numbers every value
    // of its function, which is linear in the program per warning.
    auto warnUnused = [](mlir::Operation *op, mlir::Value line,
                         llvm::StringRef kind) {
      if (!line.use_empty())
        return;
      auto diagnostic = mlir::emitWarning(op->getLoc())
                        << kind << " line allocated but never used";
      if (auto qubit = op->getAttrOfType<mlir::IntegerAttr>("qubit"))
        diagnostic << " (qubit " << qubit.getInt() << ")";
    };
    module.walk([&](pulse::GetDriveLineOp op) {
      warnUnused(op, op.getLine(), "drive");
    });
    module.walk([&](pulse::GetReadoutLineOp op) {
      warnUnused(op, op.getLine(), "readout");
    });
    // A line or tone is consumed at most once on every path: its uses must
    // lie in different arms of a conditional, and none inside a loop that
    // does not carry it.
    auto checkLinear = [&](mlir::Value value, mlir::Operation *where) {
      if (!pulse::isLinearPulseType(value.getType()))
        return;
      llvm::SmallVector<mlir::Operation *> users;
      for (mlir::OpOperand &use : value.getUses())
        users.push_back(use.getOwner());
      for (size_t i = 0; i < users.size(); ++i)
        for (size_t j = i + 1; j < users.size(); ++j)
          if (!mutuallyExclusive(users[i], users[j])) {
            where->emitError("linear pulse value is consumed more than once "
                             "on one path");
            hadError = true;
            return;
          }
      for (mlir::Operation *user : users)
        for (mlir::Operation *loop = user->getParentOp(); loop;
             loop = loop->getParentOp()) {
          if (loop == where || loop->isAncestor(where))
            break;
          if (mlir::isa<mlir::scf::ForOp>(loop)) {
            user->emitError("linear pulse value from outside an scf.for is "
                            "used in its body; carry it through iter_args");
            hadError = true;
            return;
          }
        }
    };
    module.walk([&](mlir::Operation *op) {
      for (mlir::Value result : op->getResults())
        checkLinear(result, op);
      for (mlir::Region &region : op->getRegions())
        for (mlir::Block &block : region)
          for (mlir::Value arg : block.getArguments())
            checkLinear(arg, op);
    });
    // Timing and lowering identify a line by the op that acquires it, so an
    // acquisition in a loop body cannot stand for a new line per iteration.
    module.walk([&](mlir::Operation *op) {
      if (!mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(op) ||
          !op->getParentOfType<mlir::scf::ForOp>())
        return;
      op->emitError("a line cannot be acquired inside an scf.for; acquire it "
                    "before the loop and carry it through iter_args");
      hadError = true;
    });

    // Check 2: Waveform validity -- trace SSA duration to constant, skip
    // parametric (block arg) values which are checked at evaluation time.
    auto checkDuration = [&](mlir::Operation *op, mlir::Value durVal) {
      if (auto dur = traceConstantI64(durVal)) {
        if (*dur <= 0) {
          op->emitError("waveform duration must be positive, got ") << *dur;
          hadError = true;
        }
      }
    };
    module.walk([&](pulse::GaussianPulseOp op) {
      checkDuration(op, op.getDuration());
    });
    module.walk(
        [&](pulse::SquarePulseOp op) { checkDuration(op, op.getDuration()); });
    module.walk(
        [&](pulse::DRAGPulseOp op) { checkDuration(op, op.getDuration()); });
    // A wait whose length is not a constant is scheduled at a placeholder
    // length until its arguments are bound; only a variable wait, which
    // carries its bounds, may be.
    module.walk([&](pulse::WaitOp op) {
      auto conversion =
          op.getDuration().getDefiningOp<pulse::DurationFromIntOp>();
      if (op->hasAttr("pulse.variable") ||
          (conversion && traceConstantI64(conversion.getCycles())))
        return;
      op.emitError("a wait whose length is not a constant must be marked "
                   "pulse.variable, with its bounds in pulse.min_vtu and "
                   "pulse.max_vtu");
      hadError = true;
    });

    // Check 3: Monotone, non-overlapping time on each physical line, and no
    // conditional operation before its measurement. SSA line values change
    // after every operation, so trace them back to allocation.
    module.walk([&](mlir::func::FuncOp funcOp) {
      if (funcOp.isExternal())
        return;
      TimingChecker checker;
      checker.checkBlock(funcOp.getBody().front());
      hadError |= checker.hadError;
    });

    if (hadError)
      signalPassFailure();
  }
};

} // namespace

namespace pulse {
std::unique_ptr<mlir::Pass> createPulseVerifyPass() {
  return std::make_unique<PulseVerifyPass>();
}
} // namespace pulse
