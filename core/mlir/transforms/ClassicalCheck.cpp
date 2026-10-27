/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// Checks on the classical calls of a scheduled pulse program:
//
// - every request and answer fits a slot of its link;
// - a call with a deadline can meet it: its submission and budgeted latency
//   end within `deadline_ns` of the start of the submission;
// - every answer is taken no earlier than it is due (S7), and a future is
//   waited on in the loop iteration that submitted it;
// - at no time does a link hold more requests than `max_in_flight` (S8);
// - the submissions on one link go out one after another, in program order
//   (S9).

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseOps.h.inc"

#include "cudaq-pulse/Dialect/Classical/ClassicalDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Classical/ClassicalTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Classical/ClassicalOps.h.inc"

#include "cudaq-pulse/Analysis/ClassicalTiming.h"
#include "cudaq-pulse/Analysis/Timing.h"

#include <algorithm>
#include <functional>
#include <map>
#include <vector>

namespace {

struct Occurrence {
  int64_t start;
  int64_t end;
  mlir::Operation *op;
};

/// Every execution of every submission, per link, in program order. Both
/// arms of a branch are listed, so capacity is checked for the worst case.
static void collectSubmissions(
    mlir::Block &block, llvm::SmallVectorImpl<int64_t> &iteration,
    std::map<int64_t, std::vector<Occurrence>> &perLink, bool inBranch,
    std::map<int64_t, std::vector<Occurrence>> &ordered) {
  for (mlir::Operation &op : block) {
    if (mlir::isa<classical::SubmitOp>(op)) {
      auto start = pulse::scheduledStart(&op, iteration);
      if (!start)
        continue;
      int64_t device = classical::intAttr(&op, "classical.device");
      int64_t duration = pulse::scheduledDuration(&op).value_or(1);
      int64_t budget = classical::intAttr(&op, "classical.budget_vtu");
      perLink[device].push_back({*start, *start + duration + budget, &op});
      if (!inBranch)
        ordered[device].push_back({*start, *start + duration, &op});
    } else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
      int64_t trips = pulse::constantTripCount(loop).value_or(0);
      iteration.push_back(0);
      for (int64_t k = 0; k < trips; ++k) {
        iteration.back() = k;
        collectSubmissions(*loop.getBody(), iteration, perLink, inBranch,
                           ordered);
      }
      iteration.pop_back();
    } else {
      bool branch = inBranch || mlir::isa<mlir::scf::IfOp>(op);
      for (mlir::Region &region : op.getRegions())
        for (mlir::Block &nested : region)
          collectSubmissions(nested, iteration, perLink, branch, ordered);
    }
  }
}

struct PulseClassicalCheckPass
    : public mlir::PassWrapper<PulseClassicalCheckPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PulseClassicalCheckPass)

  llvm::StringRef getArgument() const override {
    return "pulse-classical-check";
  }
  llvm::StringRef getDescription() const override {
    return "Check the classical calls of a scheduled pulse program: slot "
           "sizes, deadlines, answer timing, link capacity and order";
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    if (!classical::declaresClassicalCalls(module))
      return;
    bool failed = false;
    double clock = classical::moduleClockGHz(module);

    for (auto callable : module.getOps<classical::CallableOp>()) {
      classical::LinkModel link =
          classical::linkModel(module, callable.getDevice());
      int64_t request = classical::requestBytes(callable);
      int64_t answer = classical::answerBytes(callable);
      if (request > link.slotBytes || answer > link.slotBytes) {
        callable.emitError(request > link.slotBytes ? "a request" : "an answer")
            << " of '" << callable.getSymName() << "' takes "
            << std::max(request, answer) << " bytes, but a slot of link "
            << callable.getDevice() << " holds " << link.slotBytes;
        failed = true;
      }
      if (auto deadline = callable.getDeadlineNs()) {
        int64_t due = classical::submitVtu(callable, link, clock) +
                      classical::budgetVtu(callable, link, clock);
        int64_t limit = classical::ceilVtu(deadline->convertToDouble(), clock);
        if (due > limit) {
          callable.emitError("'")
              << callable.getSymName() << "' cannot meet its deadline: its "
              << "answer is due " << due << " cycles after the submission "
              << "starts, but the deadline is " << limit << " (slack "
              << limit - due
              << "); raise deadline_ns, lower max_latency_ns or use a "
                 "faster link";
          failed = true;
        }
      }
    }

    module.walk([&](mlir::func::FuncOp function) {
      if (function.isExternal())
        return;
      classical::annotateClassicalCalls(function);
      // S7: each answer is taken no earlier than it is due. Start times are
      // affine in the iteration, so checking the corners of the iteration
      // box covers every iteration.
      function.walk([&](classical::WaitOp wait) {
        mlir::Value future = wait.getFuture();
        auto submit = future.getDefiningOp<classical::SubmitOp>();
        if (!submit) {
          auto argument = mlir::dyn_cast<mlir::BlockArgument>(future);
          if (future.getDefiningOp<mlir::scf::ForOp>() ||
              (argument && mlir::isa<mlir::scf::ForOp>(
                               argument.getOwner()->getParentOp()))) {
            wait.emitError("the answer of '")
                << wait.getCallee()
                << "' is taken in another loop iteration than its submission";
            failed = true;
          }
          return;
        }
        if (!submit->hasAttr("start_vtu") || !wait->hasAttr("start_vtu"))
          return;
        auto loops = pulse::enclosingLoops(wait, function);
        if (loops != pulse::enclosingLoops(submit, function)) {
          wait.emitError("the answer of '")
              << wait.getCallee()
              << "' is taken in another loop than its submission";
          failed = true;
          return;
        }
        int64_t duration = pulse::scheduledDuration(submit).value_or(1);
        int64_t budget = classical::intAttr(submit, "classical.budget_vtu");
        size_t depth = loops.size();
        llvm::SmallVector<int64_t, 2> iteration(depth, 0);
        for (uint64_t corner = 0; corner < (uint64_t{1} << depth); ++corner) {
          bool empty = false;
          for (size_t level = 0; level < depth; ++level) {
            int64_t trips = pulse::constantTripCount(loops[level]).value_or(0);
            empty |= trips == 0;
            iteration[level] = (corner >> level) & 1 ? trips - 1 : 0;
          }
          if (empty)
            return;
          int64_t due =
              *pulse::scheduledStart(submit, iteration) + duration + budget;
          int64_t taken = *pulse::scheduledStart(wait, iteration);
          if (taken < due) {
            wait.emitError("the answer of '")
                << wait.getCallee() << "' is taken at " << taken
                << ", before it is due at " << due;
            failed = true;
            return;
          }
        }
      });

      std::map<int64_t, std::vector<Occurrence>> perLink, ordered;
      llvm::SmallVector<int64_t> iteration;
      collectSubmissions(function.getBody().front(), iteration, perLink, false,
                         ordered);
      // S8: requests in flight at once, by a sweep over interval ends.
      for (auto &[device, occurrences] : perLink) {
        if (occurrences.empty())
          continue;
        int64_t capacity = classical::intAttr(occurrences.front().op,
                                              "classical.max_in_flight", 64);
        std::vector<std::pair<int64_t, int>> events;
        events.reserve(2 * occurrences.size());
        for (const Occurrence &o : occurrences) {
          events.push_back({o.start, +1});
          events.push_back({o.end, -1});
        }
        // An interval is half-open: a slot freed at t can be reused at t.
        std::sort(events.begin(), events.end());
        int64_t inFlight = 0, worst = 0, when = 0;
        for (auto [time, delta] : events) {
          inFlight += delta;
          if (inFlight > worst) {
            worst = inFlight;
            when = time;
          }
        }
        if (worst > capacity) {
          occurrences.front().op->emitError("link ")
              << device << " holds " << worst << " requests at once at " << when
              << ", more than its " << capacity << " slots";
          failed = true;
        }
      }
      // S9: one submission at a time on a link, in program order.
      for (auto &[device, occurrences] : ordered)
        for (size_t i = 1; i < occurrences.size(); ++i)
          if (occurrences[i].start < occurrences[i - 1].end) {
            occurrences[i].op->emitError("submission on link ")
                << device << " starts at " << occurrences[i].start
                << ", before the one before it in program order ends at "
                << occurrences[i - 1].end;
            failed = true;
            break;
          }
    });
    if (failed)
      signalPassFailure();
  }
};

} // namespace

namespace pulse {
std::unique_ptr<mlir::Pass> createPulseClassicalCheckPass() {
  return std::make_unique<PulseClassicalCheckPass>();
}
} // namespace pulse
