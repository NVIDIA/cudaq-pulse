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
// - a call with a reaction time reacts in it: its answer is due within
//   `reaction_deadline_ns` of the end of the readouts its request carries;
// - at no time does a link hold more requests than `max_in_flight` (S8);
// - the submissions on one link go out one after another, in program order
//   (S9).
//
// Where lines wait independently (`classical.timeline`), a link's slots and
// order are per timeline: each holds no more requests than its share and
// sends its own in program order.

#include "llvm/ADT/StringMap.h"
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

#include "llvm/ADT/SetVector.h"
#include "mlir/IR/SymbolTable.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <vector>

namespace {

struct Occurrence {
  int64_t start;
  int64_t end;
  mlir::Operation *op;
};

/// A link, its instance on one controller (-1: the link shared), and one
/// timeline's share of that (-1: all of it).
using LinkKey = std::tuple<int64_t, int64_t, int64_t>;

/// Every execution of every submission, per link, in program order. Both
/// arms of a branch are listed, so capacity is checked for the worst case.
static void collectSubmissions(
    mlir::Block &block, llvm::SmallVectorImpl<int64_t> &iteration,
    std::map<LinkKey, std::vector<Occurrence>> &perLink, bool inBranch,
    std::map<LinkKey, std::vector<Occurrence>> &ordered) {
  for (mlir::Operation &op : block) {
    if (mlir::isa<classical::SubmitOp>(op)) {
      auto start = pulse::scheduledStart(&op, iteration);
      if (!start)
        continue;
      LinkKey device{classical::intAttr(&op, "classical.device"),
                     classical::intAttr(&op, classical::kInstanceAttr, -1),
                     classical::intAttr(&op, classical::kTimelineAttr, -1)};
      int64_t duration = pulse::scheduledDuration(&op).value_or(1);
      int64_t budget = classical::intAttr(&op, "classical.budget_cycles");
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

/// The readouts whose measurements reach `values` through ops without
/// regions, into `readouts`; false if one comes from elsewhere (a branch
/// result or a loop argument), whose readout is not known statically.
static bool carriedReadouts(mlir::ValueRange values,
                            llvm::SetVector<mlir::Operation *> &readouts) {
  llvm::SmallVector<mlir::Value> work(values.begin(), values.end());
  llvm::DenseSet<mlir::Value> seen;
  while (!work.empty()) {
    mlir::Value value = work.pop_back_val();
    if (!seen.insert(value).second)
      continue;
    mlir::Operation *def = value.getDefiningOp();
    if (def && mlir::isa<pulse::ReadoutOp>(def)) {
      readouts.insert(def);
      continue;
    }
    if (!def || def->getNumRegions() > 0) {
      if (mlir::isa<pulse::MeasurementType>(value.getType()))
        return false;
      continue;
    }
    work.append(def->getOperands().begin(), def->getOperands().end());
  }
  return true;
}

/// How many cycles longer than scheduled `op` can take: 0 for an op of
/// known length, -1 for a wait without a highest length.
static int64_t stretchOf(mlir::Operation *op) {
  if (mlir::isa<classical::WaitOp>(op))
    return op->hasAttr("classical.blocking") ? -1 : 0;
  auto wait = mlir::dyn_cast<pulse::WaitOp>(op);
  if (!wait || !op->hasAttr("pulse.variable_wait"))
    return 0;
  if (auto conversion =
          wait.getDuration().getDefiningOp<pulse::DurationFromIntOp>())
    if (pulse::constantIndex(conversion.getCycles()))
      return 0;
  auto highest = op->getAttrOfType<mlir::IntegerAttr>("pulse.max_cycles");
  if (!highest)
    return -1;
  return std::max<int64_t>(0, highest.getInt() -
                                  pulse::scheduledDuration(op).value_or(0));
}

/// The ops of a function whose length is not known until it runs, in
/// program order, for what they can add between two ops.
struct Stretches {
  explicit Stretches(mlir::func::FuncOp function) : function(function) {
    int64_t position = 0;
    function.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
      order[op] = position++;
      if (int64_t stretch = stretchOf(op))
        ops.push_back({op, stretch});
    });
  }

  /// What the ops that can run after `from` and before `to` can add, per
  /// execution of the innermost loop holding both: those between them in
  /// program order, and every one in a loop that holds `to` but not
  /// `from`. -1 if one has no bound.
  int64_t between(mlir::Operation *from, mlir::Operation *to) {
    if (ops.empty())
      return 0;
    mlir::Operation *root = function, *span = nullptr;
    for (mlir::scf::ForOp loop : pulse::enclosingLoops(to, function)) {
      if (!loop->isProperAncestor(from)) {
        span = loop;
        break;
      }
      root = loop;
    }
    int64_t low = order.lookup(from), high = order.lookup(to), total = 0;
    for (auto [op, stretch] : ops) {
      int64_t at = order.lookup(op);
      bool inSpan = span && span->isProperAncestor(op);
      if (!root->isProperAncestor(op) || (!inSpan && (at <= low || at >= high)))
        continue;
      if (stretch < 0)
        return -1;
      int64_t count = 1;
      for (mlir::scf::ForOp loop : pulse::enclosingLoops(op, root))
        count *= pulse::constantTripCount(loop).value_or(0);
      total += stretch * count;
    }
    return total;
  }

  mlir::func::FuncOp function;
  llvm::DenseMap<mlir::Operation *, int64_t> order;
  llvm::SmallVector<std::pair<mlir::Operation *, int64_t>> ops;
};

/// Check that every execution of `submit`, a call to `callable` with
/// `reaction_deadline_ns`, has its answer due within it of the end of the
/// latest readout it carries, or of its own start. The submission and each
/// readout start affinely in the iteration, so each difference is checked
/// at the corners of the submission's iteration box; the latest readout is
/// at least as late as each, so the smallest of their worst cases bounds
/// the reaction. A wait between a readout and the submission whose length
/// is known only when the program runs counts at its highest length.
static bool checkReaction(classical::SubmitOp submit,
                          classical::CallableOp callable,
                          mlir::func::FuncOp function, double clock,
                          Stretches &stretches) {
  if (!submit->hasAttr("start_cycles"))
    return true;
  double reactionNs = callable.getReactionDeadlineNs()->convertToDouble();
  int64_t limit = static_cast<int64_t>(std::floor(reactionNs * clock + 1e-9));
  llvm::SetVector<mlir::Operation *> readouts;
  if (!carriedReadouts(submit.getArgs(), readouts)) {
    submit.emitError("'")
        << callable.getSymName()
        << "' has reaction_deadline_ns, but a measurement it carries comes out "
           "of a "
           "branch or a loop iteration, so its readout is not known; read it "
           "out where the request is made, or drop reaction_deadline_ns";
    return false;
  }
  auto loops = pulse::enclosingLoops(submit, function);
  size_t depth = loops.size();
  llvm::SmallVector<int64_t, 2> trips;
  for (mlir::scf::ForOp loop : loops) {
    trips.push_back(pulse::constantTripCount(loop).value_or(0));
    if (trips.back() == 0)
      return true;
  }
  int64_t duration = pulse::scheduledDuration(submit).value_or(1);
  int64_t budget = classical::intAttr(submit, "classical.budget_cycles");
  // (worst reaction, its iteration) from each reference: the readouts, or
  // the submission itself.
  llvm::SmallVector<mlir::Operation *> references(readouts.begin(),
                                                  readouts.end());
  if (references.empty())
    references.push_back(submit);
  std::optional<int64_t> bound;
  llvm::SmallVector<int64_t, 2> worstAt;
  llvm::SmallVector<int64_t, 2> iteration(depth, 0);
  for (mlir::Operation *reference : references) {
    size_t own = pulse::enclosingLoops(reference, function).size();
    if (!reference->hasAttr("start_cycles") || own > depth)
      continue;
    int64_t stretch = reference == submit.getOperation()
                          ? 0
                          : stretches.between(reference, submit);
    if (stretch < 0) {
      submit.emitError("'")
          << callable.getSymName()
          << "' has reaction_deadline_ns, but between a readout it carries and "
             "the "
             "submission the program waits for a time with no bound (a "
             "blocking answer, or a wait without a highest length); submit "
             "before it, bound the wait with bounds=(lowest, highest), or "
             "drop reaction_deadline_ns";
      return false;
    }
    std::optional<int64_t> worst;
    llvm::SmallVector<int64_t, 2> at;
    for (uint64_t corner = 0; corner < (uint64_t{1} << depth); ++corner) {
      for (size_t level = 0; level < depth; ++level)
        iteration[level] = (corner >> level) & 1 ? trips[level] - 1 : 0;
      int64_t due =
          *pulse::scheduledStart(submit, iteration) + duration + budget;
      llvm::ArrayRef<int64_t> prefix(iteration.data(), own);
      int64_t from = *pulse::scheduledStart(reference, prefix);
      if (reference != submit.getOperation())
        from += pulse::scheduledDuration(reference).value_or(0);
      if (!worst || due + stretch - from > *worst) {
        worst = due + stretch - from;
        at.assign(iteration.begin(), iteration.end());
      }
    }
    if (worst && (!bound || *worst < *bound)) {
      bound = worst;
      worstAt = at;
    }
  }
  if (!bound || *bound <= limit)
    return true;
  auto diagnostic = submit.emitError("'")
                    << callable.getSymName() << "' cannot react in time: ";
  if (depth) {
    diagnostic << "in iteration (";
    llvm::interleaveComma(worstAt, diagnostic);
    diagnostic << ") ";
  }
  diagnostic << "its answer is due " << *bound << " cycles after "
             << (readouts.empty() ? "the submission starts"
                                  : "the readouts it carries end")
             << ", but reaction_deadline_ns allows " << limit << " (slack "
             << limit - *bound
             << "); submit it sooner, lower max_latency_ns, use a faster link "
                "or raise reaction_deadline_ns";
  return false;
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

    // On several controllers, the most start uncertainty a function's calls
    // add to its budget (the link's when a call's is not known).
    pulse::Controllers controllers(module);
    llvm::StringMap<std::optional<double>> uncertainty;
    if (controllers.enabled()) {
      pulse::AllocationTracer trace;
      module.walk([&](classical::SubmitOp submit) {
        auto found = classical::callUncertaintyNs(submit, controllers, trace);
        auto [it, inserted] =
            uncertainty.try_emplace(submit.getCallee(), found);
        if (!inserted && it->second)
          it->second =
              found ? std::optional<double>(std::max(*it->second, *found))
                    : std::nullopt;
      });
    }

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
        auto calls = uncertainty.find(callable.getSymName());
        int64_t due =
            classical::submitCycles(callable, link, clock) +
            classical::budgetCycles(callable, link, clock, std::nullopt,
                                    calls == uncertainty.end() ? std::nullopt
                                                               : calls->second);
        int64_t limit =
            classical::ceilCycles(deadline->convertToDouble(), clock);
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

    bool reactions =
        llvm::any_of(module.getOps<classical::CallableOp>(),
                     [](classical::CallableOp callable) {
                       return callable.getReactionDeadlineNs().has_value();
                     });
    mlir::SymbolTable symbols(module);

    module.walk([&](mlir::func::FuncOp function) {
      if (function.isExternal())
        return;
      if (mlir::failed(classical::annotateClassicalCalls(function))) {
        failed = true;
        return;
      }
      if (reactions) {
        Stretches stretches(function);
        function.walk([&](classical::SubmitOp submit) {
          auto callable =
              symbols.lookup<classical::CallableOp>(submit.getCallee());
          if (callable && callable.getReactionDeadlineNs() &&
              !checkReaction(submit, callable, function, clock, stretches))
            failed = true;
        });
      }
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
        if (!submit->hasAttr("start_cycles") || !wait->hasAttr("start_cycles"))
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
        int64_t budget = classical::intAttr(submit, "classical.budget_cycles");
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

      std::map<LinkKey, std::vector<Occurrence>> perLink, ordered;
      llvm::SmallVector<int64_t> iteration;
      collectSubmissions(function.getBody().front(), iteration, perLink, false,
                         ordered);
      // S8: requests in flight at once, by a sweep over interval ends.
      for (auto &[key, occurrences] : perLink) {
        auto [device, instance, timeline] = key;
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
          auto diag = occurrences.front().op->emitError("link ");
          diag << device;
          if (instance >= 0)
            diag << " on controllers[" << instance << "]";
          diag << " holds " << worst << " requests";
          if (timeline >= 0)
            diag << " for lines that wait on their own";
          diag << " at once at " << when << ", more than ";
          if (timeline >= 0)
            diag << "the " << capacity << " of its slots they have";
          else
            diag << "its " << capacity << " slots";
          failed = true;
        }
      }
      // S9: one submission at a time on a link, in program order.
      for (auto &[key, occurrences] : ordered)
        for (size_t i = 1; i < occurrences.size(); ++i)
          if (occurrences[i].start < occurrences[i - 1].end) {
            occurrences[i].op->emitError("submission on link ")
                << std::get<0>(key) << " starts at " << occurrences[i].start
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
