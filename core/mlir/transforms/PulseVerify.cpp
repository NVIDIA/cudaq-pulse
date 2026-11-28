// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Module-level verification pass for the Pulse dialect.
// Checks linearity, monotone time, measurement causality, and waveform
// validity beyond what individual op verifiers can check, through structured
// control flow: the arms of an `scf.if` are alternative paths, and an
// `scf.for` body runs once per iteration.

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include <functional>

#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"
#include "cudaq-pulse/Dialect/Pulse/PulseEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseAttrs.h"
#include "cudaq-pulse/Dialect/Pulse/PulseOps.h.inc"

#include "cudaq-pulse/Dialect/Classical/ClassicalDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Classical/ClassicalTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Classical/ClassicalOps.h.inc"

#include "cudaq-pulse/Analysis/ClassicalTiming.h"
#include "cudaq-pulse/Analysis/Durations.h"
#include "cudaq-pulse/Analysis/LineTrace.h"
#include "cudaq-pulse/Analysis/Timing.h"
#include "cudaq-pulse/Analysis/Untimed.h"

#include <algorithm>
#include <cmath>
#include <iterator>

namespace pulse {
/// Whether `module` holds a pulse.call (PulseCallCheck.cpp).
bool moduleHasCalls(mlir::ModuleOp module);
/// pulse-verify's check of a module with placed calls (PulseCallCheck.cpp).
mlir::LogicalResult checkCalls(mlir::ModuleOp module);
} // namespace pulse

namespace {

static bool isTimedOp(mlir::Operation *op) {
  return mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp>(op);
}

/// The measurement results a branch condition is computed from, not
/// through a link or an answer (see the scheduler's conditionMeasurements):
/// pulse-classical-check checks a branch on an answer.
static void conditionMeasurements(mlir::Value value,
                                  llvm::SmallVectorImpl<mlir::Value> &found,
                                  llvm::DenseSet<mlir::Value> &seen) {
  // The values still to follow, the next one last: an operation's operands
  // in their order, each with all it is computed from before the next.
  llvm::SmallVector<mlir::Value, 16> todo{value};
  while (!todo.empty()) {
    mlir::Value next = todo.pop_back_val();
    if (mlir::isa<classical::LinkType>(next.getType()) ||
        !seen.insert(next).second)
      continue;
    if (mlir::isa<pulse::MeasurementType>(next.getType())) {
      found.push_back(next);
      continue;
    }
    mlir::Operation *def = next.getDefiningOp();
    if (!def || mlir::isa<classical::WaitOp>(def))
      continue;
    for (mlir::Value operand : llvm::reverse(def->getOperands()))
      todo.push_back(operand);
  }
}

/// Checks the schedule attributes of one function: every drive, readout
/// and wait has its times, starts no earlier than time zero, than the
/// measurements the branches around it wait for, and than the operation
/// before it on its physical line ends, in every iteration of the loops
/// around it.
///
/// What is reported is what a walk of the program in execution order
/// reports, each loop at its first two iterations and its last, one inside
/// the other: an operation once, at the first of those visits at which it
/// is wrong, in the order of the visits. That walk takes three to the
/// power of the loops' depth; this one takes the operations times the
/// depth (times, for an operation in a branch inside loops, the readouts
/// it waits for that move differently), from three facts. With one period per
/// line per loop, which a loop is held to before it is walked (checkPeriods),
/// and none negative:
///
/// - A start only grows with the iteration, so one before time zero is
///   before it at the first visit.
/// - The time from a measurement's end to a start is affine in the
///   iteration, so the first visit at which it is negative is found a loop
///   at a time (startsBeforeMeasured).
/// - What a line's operations end at only grows along the walk (but for
///   the arm of a branch that is walked second, which starts from where
///   the first started), and by a line's period from one iteration to the
///   next. So an operation that
///   starts before the latest end on its line at any visit does so at the
///   first visit, or at a visit with exactly one loop at its second
///   iteration and the others at their first: any later visit repeats the
///   comparisons of one of those, moved by whole periods, against ends
///   that were already there.
///
/// So the walk makes one pass with every loop at its first iteration,
/// and, at each loop it meets there, one more of that loop's body at its
/// second iteration with the loops inside it at their first. When a loop
/// is left, each line it touched ends where its operations end with the
/// loop and those inside it at their last iterations, as after the walk
/// of every visit.
class TimingChecker {
  pulse::AllocationTracer trace;

public:
  bool hadError = false;

  /// A timed or placed function (pulse.schedule_state) is scheduled as a
  /// whole: every timed op that runs has a schedule.
  void checkFunction(mlir::func::FuncOp function) {
    checkBlock(function.getBody().front());
    // In the order of the visits.
    llvm::stable_sort(reports, [](const Report &a, const Report &b) {
      return std::lexicographical_compare(a.visit.begin(), a.visit.end(),
                                          b.visit.begin(), b.visit.end());
    });
    for (const Report &found : reports)
      found.op->emitError(found.message);
    pulse::ScheduleState state = pulse::scheduleStateOf(function);
    if (state != pulse::ScheduleState::Unscheduled && unscheduled) {
      hadError = true;
      if (!reportOf.count(unscheduled))
        unscheduled->emitError(
            "operation is not scheduled, but its function is ")
            << pulse::stringifyScheduleState(state);
    }
  }

private:
  /// The kinds of fault of an operation, in the order they are looked for
  /// at one visit: the first found is the one reported.
  enum Fault : int64_t {
    NoStart,
    NoDuration,
    NoLength,
    WrongDuration,
    PeriodCount,
    BeforeZero,
    NoLine,
    BeforeMeasured,
    BeforeEnd,
  };

  /// A fault of `op` and where in the walk it is found: the place of each
  /// block and operation around it among its siblings, the iteration rank
  /// of each loop, and the kind of fault.
  struct Report {
    llvm::SmallVector<int64_t, 12> visit;
    mlir::Operation *op;
    std::string message;
  };

  /// Keeps the fault of `op` found at `visit` if it is its first in the
  /// order of the visits.
  void report(mlir::Operation *op, llvm::SmallVector<int64_t, 12> visit,
              const llvm::Twine &message) {
    hadError = true;
    auto [it, inserted] = reportOf.try_emplace(op, reports.size());
    if (inserted) {
      reports.push_back({std::move(visit), op, message.str()});
      return;
    }
    Report &old = reports[it->second];
    if (std::lexicographical_compare(visit.begin(), visit.end(),
                                     old.visit.begin(), old.visit.end())) {
      old.visit = std::move(visit);
      old.message = message.str();
    }
  }

  /// The visit being made, for a fault of kind `fault`.
  llvm::SmallVector<int64_t, 12> here(int64_t fault) const {
    llvm::SmallVector<int64_t, 12> visit(place.begin(), place.end());
    visit.push_back(fault);
    return visit;
  }

  void error(mlir::Operation *op, Fault fault, const llvm::Twine &message) {
    report(op, here(fault), message);
  }

  void checkBlock(mlir::Block &block) {
    place.push_back(0);
    for (mlir::Operation &op : block.without_terminator()) {
      checkOp(&op);
      ++place.back();
    }
    place.pop_back();
  }

  void checkOp(mlir::Operation *op) {
    if (isTimedOp(op))
      return checkTimed(op);
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op))
      return checkLoop(loop);
    if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op))
      return checkIf(branch);
    if (op->getNumRegions() == 0)
      return;
    place.push_back(0);
    for (mlir::Region &region : op->getRegions())
      for (mlir::Block &block : region) {
        checkBlock(block);
        ++place.back();
      }
    place.pop_back();
  }

  /// Leaves `op` out of the second iteration of the loops around it.
  void passOver(mlir::Operation *op) {
    if (!around.empty())
      untimed.insert(op);
  }

  void checkTimed(mlir::Operation *op) {
    // The loop at its second iteration in this pass, if any; the others
    // are at their first.
    bool first = !second;
    if (!first && !untimed.empty() && untimed.count(op))
      return;
    auto start = pulse::startCycles(op);
    auto duration = pulse::scheduledDuration(op);
    if (!start && !duration) {
      if (first && !unscheduled)
        unscheduled = op;
      return;
    }
    if (first) {
      if (!start) {
        passOver(op);
        return error(op, NoStart, "a scheduled operation needs start_cycles");
      }
      if (!duration) {
        passOver(op);
        return error(op, NoDuration,
                     "a scheduled operation needs duration_cycles");
      }
      if (!op->hasAttr("classical.variable")) {
        std::optional<int64_t> cycles = cyclesOf(op);
        if (!cycles) {
          passOver(op);
          return error(op, NoLength,
                       "a scheduled operation needs a known length");
        }
        if (*cycles != *duration)
          error(op, WrongDuration,
                "duration_cycles is " + llvm::Twine(*duration) +
                    ", but the operation takes " + llvm::Twine(*cycles) +
                    " cycles");
      }
    }
    llvm::ArrayRef<int64_t> periods = pulse::periodCycles(op);
    if (first && periods.size() != around.size()) {
      passOver(op);
      return error(op, PeriodCount,
                   "scheduled operation needs one period_cycles entry per "
                   "enclosing scf.for");
    }
    int64_t at = *start + (first ? 0 : periods[*second]);
    if (first && at < 0)
      error(op, BeforeZero,
            "operation is scheduled before time zero: start " +
                llvm::Twine(at));
    mlir::Operation *physicalLine = trace(op->getOperand(0));
    if (!physicalLine) {
      if (first) {
        passOver(op);
        error(op, NoLine, "cannot trace timed operation to a physical line");
      }
      return;
    }
    if (around.empty()) {
      if (at < measured)
        error(op, BeforeMeasured,
              "operation in a measurement-conditioned branch starts at " +
                  llvm::Twine(at) + ", before its measurement completes at " +
                  llvm::Twine(measured));
    } else if (first && !readouts.empty()) {
      startsBeforeMeasured(op, *start, periods);
    }
    int64_t end = at + *duration;
    auto [it, inserted] = lastEnd.try_emplace(physicalLine, 0);
    if (!inserted && at < it->second)
      error(op, BeforeEnd,
            "operation overlaps or precedes its physical-line predecessor: "
            "start " +
                llvm::Twine(at) + " < prior end " + llvm::Twine(it->second));
    it->second = std::max(it->second, end);
    if (!inside.empty()) {
      auto [held, fresh] = inside.back().try_emplace(physicalLine, end, op);
      if (!fresh && end > held->second.first)
        held->second = {end, op};
    }
  }

  /// The iterations of a loop of `trips` that the walk of every visit
  /// makes, in its order: the first two and the last.
  static llvm::SmallVector<int64_t, 3> visited(int64_t trips) {
    llvm::SmallVector<int64_t, 3> iterations;
    for (int64_t k = 0; k < trips; ++k) {
      if (k == 2 && trips > 3)
        k = trips - 1;
      iterations.push_back(k);
    }
    return iterations;
  }

  /// Reports `op`, which starts at `start` in the first iteration of the
  /// loops around it and moves by `periods`, at the first visit at which it
  /// starts before a measurement the branches around it wait for
  /// completes. The time between is affine in the iteration: per readout,
  /// a loop at a time from the outermost, the earliest iteration is taken
  /// from which the loops inside can still make it negative.
  void startsBeforeMeasured(mlir::Operation *op, int64_t start,
                            llvm::ArrayRef<int64_t> periods) {
    size_t depth = around.size();
    llvm::SmallVector<int64_t, 4> moves(depth), least(depth + 1, 0);
    llvm::SmallVector<int64_t, 4> best, ranks(depth), chosen(depth);
    for (const Waited &waited : readouts) {
      llvm::ArrayRef<int64_t> its = waited.periods;
      size_t shared = its.size();
      int64_t between = start - waited.end;
      for (size_t level = depth; level-- > 0;) {
        moves[level] = periods[level] - (level < shared ? its[level] : 0);
        int64_t low = 0;
        for (int64_t k : around[level].iterations)
          low = std::min(low, moves[level] * k);
        least[level] = least[level + 1] + low;
      }
      if (between + least[0] >= 0)
        continue;
      for (size_t level = 0; level < depth; ++level) {
        llvm::ArrayRef<int64_t> iterations = around[level].iterations;
        for (size_t rank = 0; rank < iterations.size(); ++rank) {
          int64_t next = between + moves[level] * iterations[rank];
          if (next + least[level + 1] < 0) {
            between = next;
            ranks[level] = rank;
            break;
          }
        }
      }
      if (best.empty() ||
          std::lexicographical_compare(ranks.begin(), ranks.end(), best.begin(),
                                       best.end()))
        best.assign(ranks.begin(), ranks.end());
    }
    if (best.empty())
      return;
    for (size_t level = 0; level < depth; ++level)
      chosen[level] = around[level].iterations[best[level]];
    int64_t at = start;
    for (size_t level = 0; level < depth; ++level)
      at += chosen[level] * periods[level];
    int64_t completes = std::numeric_limits<int64_t>::min();
    for (const Waited &waited : readouts) {
      int64_t end = waited.end;
      for (size_t level = 0; level < waited.periods.size(); ++level)
        end += chosen[level] * waited.periods[level];
      completes = std::max(completes, end);
    }
    llvm::SmallVector<int64_t, 12> visit = here(BeforeMeasured);
    for (size_t level = 0; level < depth; ++level)
      visit[around[level].rank] = best[level];
    report(op, std::move(visit),
           "operation in a measurement-conditioned branch starts at " +
               llvm::Twine(at) + ", before its measurement completes at " +
               llvm::Twine(completes));
  }

  void checkLoop(mlir::scf::ForOp loop) {
    bool first = !second;
    if (!first && !unwalked.empty() && unwalked.count(loop))
      return;
    auto trips = pulse::constantTripCount(loop);
    if (!trips) {
      unwalked.insert(loop);
      for (mlir::Operation &inner : loop.getBody()->without_terminator())
        if (isTimedOp(&inner) && pulse::hasStart(&inner)) {
          place.push_back(-1);
          error(&inner, NoStart, "a scheduled loop needs constant bounds");
          place.pop_back();
          return;
        }
      return;
    }
    // Checking iterations relies on one period per line.
    if (first && !checkPeriods(loop)) {
      unwalked.insert(loop);
      return;
    }
    if (*trips == 0)
      return;
    size_t level = around.size();
    around.push_back({visited(*trips), place.size()});
    // The first iteration; what each line ends at in it, the loops inside
    // at their last.
    inside.emplace_back();
    place.push_back(0);
    checkBlock(*loop.getBody());
    if (first && *trips > 1) {
      // The second, with the loops inside at their first.
      second = level;
      inside.emplace_back();
      place.back() = 1;
      checkBlock(*loop.getBody());
      inside.pop_back();
      second.reset();
    }
    place.pop_back();
    around.pop_back();
    // Left: each line ends where it does with the loop at its last
    // iteration, a whole number of its periods after the first.
    LineEnds ends = std::move(inside.back());
    inside.pop_back();
    for (auto &[line, held] : ends) {
      auto [end, op] = held;
      end += (*trips - 1) * pulse::periodCycles(op)[level];
      int64_t &last = lastEnd[line];
      last = std::max(last, end);
      if (!inside.empty()) {
        auto [outer, fresh] = inside.back().try_emplace(line, end, op);
        if (!fresh && end > outer->second.first)
          outer->second = {end, op};
      }
    }
  }

  /// Within one iteration of `loop`, every op on a physical line must move
  /// by the same non-negative period from one iteration to the next.
  bool checkPeriods(mlir::scf::ForOp loop) {
    size_t level = around.size();
    bool ok = true;
    int64_t seen = 0;
    llvm::DenseMap<mlir::Operation *, int64_t> periodOf;
    loop.getBody()->walk([&](mlir::Operation *op) {
      if (!isTimedOp(op))
        return;
      llvm::ArrayRef<int64_t> periods = pulse::periodCycles(op);
      if (periods.size() <= level)
        return;
      int64_t period = periods[level];
      // Before the loop's own visits, in the order of its operations.
      auto fault = [&](const char *message) {
        ok = false;
        llvm::SmallVector<int64_t, 12> visit(place.begin(), place.end());
        visit.push_back(-1);
        visit.push_back(seen);
        report(op, std::move(visit), message);
      };
      ++seen;
      if (period < 0)
        return fault("period_cycles must not be negative");
      mlir::Operation *line = trace(op->getOperand(0));
      auto [it, inserted] = periodOf.try_emplace(line, period);
      if (!inserted && it->second != period)
        fault("operations on one line in a loop must share its period");
    });
    return ok;
  }

  void checkIf(mlir::scf::IfOp branch) {
    int64_t savedMeasured = measured;
    size_t savedReadouts = readouts.size();
    llvm::SmallVector<mlir::Value> measurements;
    llvm::DenseSet<mlir::Value> seen;
    conditionMeasurements(branch.getCondition(), measurements, seen);
    for (mlir::Value measurement : measurements)
      if (auto readout = measurement.getDefiningOp<pulse::ReadoutOp>()) {
        auto start = pulse::startCycles(readout);
        auto duration = pulse::scheduledDuration(readout);
        if (!start || !duration)
          continue;
        if (around.empty())
          measured = std::max(measured, *start + *duration);
        waitFor(readout, *start + *duration, savedReadouts);
      }

    auto before = lastEnd;
    place.push_back(0);
    checkBlock(*branch.thenBlock());
    if (!branch.getElseRegion().empty()) {
      auto afterThen = std::move(lastEnd);
      lastEnd = std::move(before);
      place.back() = 1;
      checkBlock(*branch.elseBlock());
      for (auto [line, end] : afterThen)
        lastEnd[line] = std::max(lastEnd.lookup(line), end);
    }
    place.pop_back();
    measured = savedMeasured;
    readouts.truncate(savedReadouts);
  }

  /// Adds `readout`, which ends at `end` in the first iteration of the
  /// loops around it, to the readouts the operations visited wait for;
  /// `own` is where those of the branch being entered begin. Of two that
  /// move alike from one iteration to the next, the one that ends later is
  /// the later at every visit: the other is not kept, or, if it is one of
  /// this branch's, gives way.
  void waitFor(mlir::Operation *readout, int64_t end, size_t own) {
    // Outside every loop no readout moves.
    llvm::ArrayRef<int64_t> its;
    if (!around.empty())
      its = pulse::periodCycles(readout).take_front(
          std::min(pulse::enclosingLoops(readout).size(), around.size()));
    for (size_t i = readouts.size(); i-- > 0;) {
      Waited &other = readouts[i];
      if (other.periods != its)
        continue;
      if (other.end >= end)
        return;
      if (i >= own) {
        other.end = end;
        return;
      }
    }
    readouts.push_back({its, end});
  }

  /// The cycles a drive, readout or wait takes, as the scheduler places it.
  static std::optional<int64_t> cyclesOf(mlir::Operation *op) {
    if (auto drive = mlir::dyn_cast<pulse::DriveOp>(op))
      return pulse::waveformCycles(drive.getPulse());
    if (auto readout = mlir::dyn_cast<pulse::ReadoutOp>(op))
      return pulse::waveformCycles(readout.getPulse());
    return pulse::waitCycles(mlir::cast<pulse::WaitOp>(op));
  }

  /// A loop around the operation visited: the iterations the walk of every
  /// visit makes of it, and where its rank is in `place`.
  struct Around {
    llvm::SmallVector<int64_t, 3> iterations;
    size_t rank;
  };
  /// Per physical line, the latest end of its operations and one of them.
  using LineEnds =
      llvm::SmallDenseMap<mlir::Operation *,
                          std::pair<int64_t, mlir::Operation *>, 4>;

  llvm::DenseMap<mlir::Operation *, int64_t> lastEnd;
  std::vector<Report> reports;
  llvm::DenseMap<mlir::Operation *, size_t> reportOf;
  /// The first timed op without a schedule, if any.
  mlir::Operation *unscheduled = nullptr;
  /// The loops around the operation visited, outermost first; the one at
  /// its second iteration in this pass, if any; and, per loop being walked
  /// at its first iteration, what its lines end at in it.
  llvm::SmallVector<Around, 4> around;
  std::optional<size_t> second;
  llvm::SmallVector<LineEnds, 4> inside;
  /// Where the walk is: see Report.
  llvm::SmallVector<int64_t, 12> place;
  /// Timed ops that end no line (a fault said why at their first visit),
  /// and loops that are not walked.
  llvm::DenseSet<mlir::Operation *> untimed, unwalked;
  /// A readout the branches around the operation visited wait for: its
  /// periods in the loops around it that are around the operation too, and
  /// its end in their first iteration.
  struct Waited {
    llvm::ArrayRef<int64_t> periods;
    int64_t end;
  };
  /// Those readouts, but the ones another is later than at every visit;
  /// outside every loop, the latest of their ends.
  llvm::SmallVector<Waited, 2> readouts;
  int64_t measured = std::numeric_limits<int64_t>::min();
};

/// What an op verifier cannot check without looking at the ops that compute
/// its operands. A duration, sigma or risefall is checked once it is a
/// constant (or arithmetic on constants: folding must not make valid IR
/// invalid); one computed from the function's arguments is checked when they
/// are bound.
class WaveformCheck {
public:
  bool hadError = false;

  void visit(mlir::Operation *op) {
    if (op->getNumResults() == 1 &&
        mlir::isa<pulse::WaveformType>(op->getResult(0).getType()))
      return visitWaveform(op);
    if (auto outcome = mlir::dyn_cast<pulse::OutcomeOp>(op))
      return checkMeasurement(op, outcome.getMeasurement());
    if (auto isSet = mlir::dyn_cast<pulse::IsSetOp>(op))
      return checkMeasurement(op, isSet.getMeasurement());
    if (auto wait = mlir::dyn_cast<pulse::WaitOp>(op))
      return checkBounds(wait);
  }

private:
  void visitWaveform(mlir::Operation *op) {
    llvm::TypeSwitch<mlir::Operation *>(op)
        .Case<pulse::SquarePulseOp, pulse::CosinePulseOp, pulse::CustomOp>(
            [&](auto op) { checkDuration(op, op.getDuration()); })
        .Case<pulse::GaussianPulseOp, pulse::DRAGPulseOp, pulse::TanhRampOp>(
            [&](auto op) {
              checkDuration(op, op.getDuration());
              checkSigma(op, op.getSigma());
            })
        .Case([&](pulse::GaussianSquarePulseOp op) {
          checkDuration(op, op.getDuration());
          checkSigma(op, op.getSigma());
          checkRisefall(op);
        })
        .Case<pulse::PulseAddOp, pulse::PulseSubOp, pulse::PulseMulOp>(
            [&](auto op) { checkEqualLengths(op, op.getLhs(), op.getRhs()); });
  }

  void checkDuration(mlir::Operation *op, mlir::Value duration) {
    auto cycles = pulse::constantInteger(duration);
    if (cycles && *cycles <= 0) {
      op->emitOpError("duration must be positive, got ") << *cycles;
      hadError = true;
    }
  }

  /// NaN compares false with everything, so `sigma <= 0` would pass it.
  void checkSigma(mlir::Operation *op, mlir::Value sigma) {
    auto constant = sigma.getDefiningOp<mlir::arith::ConstantOp>();
    auto value = constant ? mlir::dyn_cast<mlir::FloatAttr>(constant.getValue())
                          : nullptr;
    if (!value)
      return;
    double number = value.getValueAsDouble();
    if (!(number > 0.0) || !std::isfinite(number)) {
      op->emitOpError("sigma must be positive and finite, got ") << number;
      hadError = true;
    }
  }

  void checkRisefall(pulse::GaussianSquarePulseOp op) {
    auto risefall = pulse::constantInteger(op.getRisefall());
    if (!risefall)
      return;
    if (*risefall <= 0) {
      op.emitOpError("risefall must be positive, got ") << *risefall;
      hadError = true;
      return;
    }
    auto duration = pulse::constantInteger(op.getDuration());
    if (duration && *duration > 0 && *risefall > *duration / 2) {
      op.emitOpError("risefall (")
          << *risefall << ") exceeds half the duration (" << *duration << ")";
      hadError = true;
    }
  }

  /// Waveform arithmetic is element-wise, so both operands must have the
  /// same length; otherwise one would be silently truncated.
  void checkEqualLengths(mlir::Operation *op, mlir::Value lhs,
                         mlir::Value rhs) {
    auto left = lengthOf(lhs);
    auto right = lengthOf(rhs);
    if (left && right && *left != *right) {
      op->emitOpError("operands have durations ")
          << *left << " and " << *right
          << "; waveform arithmetic needs equal durations";
      hadError = true;
    }
  }

  /// The length of `waveform` in cycles, if known: from either operand of
  /// waveform arithmetic. Each value is computed once.
  std::optional<int64_t> lengthOf(mlir::Value waveform) {
    auto known = lengths.find(waveform);
    if (known != lengths.end())
      return known->second;
    std::optional<int64_t> length;
    mlir::Operation *def = waveform.getDefiningOp();
    if (mlir::isa_and_nonnull<pulse::PulseAddOp, pulse::PulseSubOp,
                              pulse::PulseMulOp>(def)) {
      length = lengthOf(def->getOperand(0));
      if (!length)
        length = lengthOf(def->getOperand(1));
    } else if (mlir::isa_and_nonnull<pulse::PulseScaleOp, pulse::PulseNegOp>(
                   def)) {
      length = lengthOf(def->getOperand(0));
    } else {
      length = pulse::waveformCycles(waveform);
    }
    lengths[waveform] = length;
    return length;
  }

  /// A measurement comes from a readout on every path: through either arm
  /// of an scf.if, a loop's initial and yielded value (only the yielded one
  /// for the result of a loop that runs), a schedule block's operand and
  /// yielded value. A function argument is the caller's.
  void checkMeasurement(mlir::Operation *op, mlir::Value measurement) {
    if (measurement.getDefiningOp<pulse::ReadoutOp>())
      return;
    if (mlir::Operation *wrong = sourceOtherThanReadout(measurement)) {
      op->emitOpError("measurement must be produced by pulse.readout, got ")
          << wrong->getName();
      hadError = true;
    }
  }

  /// An op that is not a readout and that `measurement` may come from, or
  /// null. Each value is followed once; a loop's carried value is assumed
  /// to come from a readout while it is being followed.
  mlir::Operation *sourceOtherThanReadout(mlir::Value measurement) {
    auto [it, inserted] = sources.try_emplace(measurement, nullptr);
    if (!inserted)
      return it->second;
    auto follow = [&](mlir::Value a, mlir::Value b = {}) -> mlir::Operation * {
      if (mlir::Operation *wrong = sourceOtherThanReadout(a))
        return wrong;
      return b ? sourceOtherThanReadout(b) : nullptr;
    };
    mlir::Operation *wrong = nullptr;
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(measurement)) {
      mlir::Operation *parent = argument.getOwner()->getParentOp();
      unsigned index = argument.getArgNumber();
      if (auto loop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(parent)) {
        if (index > 0)
          wrong =
              follow(loop.getInitArgs()[index - 1],
                     loop.getBody()->getTerminator()->getOperand(index - 1));
      } else if (auto block =
                     mlir::dyn_cast_or_null<pulse::ScheduleBlockOp>(parent)) {
        wrong = follow(block.getInputs()[index]);
      }
    } else {
      auto result = mlir::cast<mlir::OpResult>(measurement);
      mlir::Operation *def = result.getOwner();
      unsigned index = result.getResultNumber();
      if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(def))
        wrong = follow(branch.thenYield().getOperand(index),
                       branch.elseYield().getOperand(index));
      else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(def)) {
        // After an iteration, the result is what the last one yielded.
        mlir::Value yielded =
            loop.getBody()->getTerminator()->getOperand(index);
        auto trips = pulse::constantTripCount(loop);
        wrong = trips && *trips > 0
                    ? follow(yielded)
                    : follow(loop.getInitArgs()[index], yielded);
      } else if (auto block = mlir::dyn_cast<pulse::ScheduleBlockOp>(def))
        wrong =
            follow(block.getBody().front().getTerminator()->getOperand(index));
      else if (!mlir::isa<pulse::ReadoutOp>(def))
        wrong = def;
    }
    sources[measurement] = wrong;
    return wrong;
  }

  /// A variable wait whose length is a constant must be within its bounds.
  void checkBounds(pulse::WaitOp wait) {
    auto lowest = wait->getAttrOfType<mlir::IntegerAttr>("pulse.min_cycles");
    auto conversion =
        wait.getDuration().getDefiningOp<pulse::DurationFromIntOp>();
    if (!lowest || !wait->hasAttr("pulse.variable_wait") || !conversion)
      return;
    auto highest = wait->getAttrOfType<mlir::IntegerAttr>("pulse.max_cycles");
    auto cycles = pulse::constantInteger(conversion.getCycles());
    if (!cycles || (*cycles >= lowest.getInt() &&
                    (!highest || *cycles <= highest.getInt())))
      return;
    wait.emitOpError("a pulse.variable_wait wait of ")
        << *cycles << " cycles is outside its bounds [" << lowest.getInt()
        << ", " << (highest ? std::to_string(highest.getInt()) : "none") << "]";
    hadError = true;
  }

  llvm::DenseMap<mlir::Value, std::optional<int64_t>> lengths;
  llvm::DenseMap<mlir::Value, mlir::Operation *> sources;
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
    // A module with pulse.call is checked as it is written out.
    if (pulse::moduleHasCalls(module)) [[unlikely]] {
      if (mlir::failed(pulse::checkCalls(module)))
        signalPassFailure();
      return;
    }
    bool hadError = false;

    // Check 1: Linearity -- lines must be consumed. The warning names the
    // qudit instead of attaching the op: printing an op numbers every value
    // of its function, which is linear in the program per warning.
    auto warnUnused = [](mlir::Operation *op, mlir::Value line,
                         llvm::StringRef kind) {
      if (!line.use_empty())
        return;
      auto diagnostic = mlir::emitWarning(op->getLoc())
                        << kind << " line allocated but never used";
      if (int64_t qudit = pulse::quditOf(op); qudit >= 0)
        diagnostic << " (qudit " << qudit << ")";
    };
    module.walk([&](pulse::GetDriveLineOp op) {
      warnUnused(op, op.getLine(), "drive");
    });
    module.walk([&](pulse::GetReadoutLineOp op) {
      warnUnused(op, op.getLine(), "readout");
    });
    // A line or tone is consumed at most once on every path: its uses must
    // lie in different arms of a conditional, and none inside a loop that
    // does not carry it. A use that only reads it (a time read) consumes
    // nothing.
    auto checkLinear = [&](mlir::Value value, mlir::Operation *where) {
      if (!pulse::isLinearPulseType(value.getType()))
        return;
      llvm::SmallVector<mlir::Operation *> users;
      for (mlir::OpOperand &use : value.getUses())
        if (!pulse::readsWithoutConsuming(use.getOwner()))
          users.push_back(use.getOwner());
      if (!pulse::allMutuallyExclusive(users)) {
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
    // One walk for each op's own checks: an op the schedule passes cannot
    // time, linearity, the values of waveforms
    // and of the waits and measurements that use them (WaveformCheck), and
    // acquisitions. Timing and lowering identify a line by the op that
    // acquires it, so an acquisition in a loop body cannot stand for a new
    // line per iteration.
    WaveformCheck waveforms;
    module.walk([&](mlir::Operation *op) {
      // The schedule passes would time an op they do not know as nothing.
      if (pulse::untimed(op)) {
        pulse::emitUntimed(op);
        hadError = true;
      }
      for (mlir::Value result : op->getResults())
        checkLinear(result, op);
      for (mlir::Region &region : op->getRegions())
        for (mlir::Block &block : region)
          for (mlir::Value arg : block.getArguments())
            checkLinear(arg, op);
      waveforms.visit(op);
      if (!mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(op))
        return;
      if (op->getParentOfType<mlir::scf::ForOp>()) {
        op->emitError("a line cannot be acquired inside an scf.for; acquire "
                      "it before the loop and carry it through iter_args");
        hadError = true;
      }
      if (mlir::failed(pulse::checkOneAcquisition(op)))
        hadError = true;
    });

    // Check 1b: classical calls. A line-like link cannot be acquired in a
    // loop either (V1 and V2 of the classical-call rules follow).
    module.walk([&](classical::GetLinkOp op) {
      if (!op->getParentOfType<mlir::scf::ForOp>())
        return;
      op->emitError("a link cannot be acquired inside an scf.for; acquire it "
                    "before the loop and carry it through iter_args");
      hadError = true;
    });
    // V1: an answer that arrives while the program runs cannot change the
    // timeline, so no duration or loop bound may be computed from one.
    llvm::DenseMap<mlir::Value, bool> fromAnswer;
    std::function<bool(mlir::Value)> dependsOnAnswer =
        [&](mlir::Value value) -> bool {
      auto [it, inserted] = fromAnswer.try_emplace(value, false);
      if (!inserted)
        return it->second;
      mlir::Operation *def = value.getDefiningOp();
      bool result = false;
      if (def && mlir::isa<classical::WaitOp>(def))
        result = true;
      else if (def && def->getNumRegions() == 0)
        result = llvm::any_of(def->getOperands(), dependsOnAnswer);
      fromAnswer[value] = result;
      return result;
    };
    bool hasAnswers = false;
    module.walk([&](classical::WaitOp) { hasAnswers = true; });
    if (hasAnswers) {
      auto rejectDuration = [&](mlir::Operation *op, mlir::Value duration) {
        if (!dependsOnAnswer(duration))
          return;
        op->emitError("a duration cannot depend on the answer of a classical "
                      "call: the answer arrives while the program runs, after "
                      "its timeline is fixed; branch on the answer to choose "
                      "between durations instead");
        hadError = true;
      };
      module.walk([&](mlir::Operation *op) {
        if (auto w = mlir::dyn_cast<pulse::SquarePulseOp>(op))
          rejectDuration(op, w.getDuration());
        else if (auto w = mlir::dyn_cast<pulse::GaussianPulseOp>(op))
          rejectDuration(op, w.getDuration());
        else if (auto w = mlir::dyn_cast<pulse::DRAGPulseOp>(op))
          rejectDuration(op, w.getDuration());
        else if (auto w = mlir::dyn_cast<pulse::CosinePulseOp>(op))
          rejectDuration(op, w.getDuration());
        else if (auto w = mlir::dyn_cast<pulse::TanhRampOp>(op))
          rejectDuration(op, w.getDuration());
        else if (auto w = mlir::dyn_cast<pulse::GaussianSquarePulseOp>(op)) {
          rejectDuration(op, w.getDuration());
          rejectDuration(op, w.getRisefall());
        } else if (auto w = mlir::dyn_cast<pulse::CustomOp>(op))
          rejectDuration(op, w.getDuration());
        else if (auto d = mlir::dyn_cast<pulse::DurationFromIntOp>(op)) {
          // A wait marked variable may take its length from an answer.
          bool variable = !d->use_empty() &&
                          llvm::all_of(d->getUsers(), [](mlir::Operation *u) {
                            return mlir::isa<pulse::WaitOp>(u) &&
                                   u->hasAttr("classical.variable");
                          });
          if (!variable)
            rejectDuration(op, d.getCycles());
        } else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op))
          for (mlir::Value bound :
               {loop.getLowerBound(), loop.getUpperBound(), loop.getStep()})
            if (dependsOnAnswer(bound)) {
              loop.emitError("a loop bound cannot depend on the answer of a "
                             "classical call");
              hadError = true;
              break;
            }
      });
    }
    // V2 for variable waits: a region whose length is known only at run time
    // cannot sit in a schedule block or in one arm of a branch.
    module.walk([&](pulse::WaitOp op) {
      if (!op->hasAttr("classical.variable"))
        return;
      if (!op->hasAttr("pulse.variable_wait")) {
        op.emitError(
            "a classical.variable wait must also be pulse.variable_wait, "
            "with its bounds in pulse.min_cycles and pulse.max_cycles");
        hadError = true;
      } else if (op->getParentOfType<pulse::ScheduleBlockOp>()) {
        op.emitError(classical::kVariableWaitInBlock);
        hadError = true;
      } else if (op->getParentOfType<mlir::scf::IfOp>()) {
        op.emitError("a variable wait cannot be inside one arm of an scf.if; "
                     "let the answer be 0 where no wait is wanted");
        hadError = true;
      }
    });
    // V2: a blocking wait stops the whole program until the answer arrives,
    // so it cannot sit in a schedule block, which has a fixed length, or in
    // one arm of a branch.
    mlir::SymbolTableCollection symbols;
    module.walk([&](classical::WaitOp op) {
      auto callable = symbols.lookupNearestSymbolFrom<classical::CallableOp>(
          op, op.getCalleeAttr());
      if (!callable || callable.getMode() != "blocking")
        return;
      if (op->getParentOfType<pulse::ScheduleBlockOp>()) {
        op.emitError(classical::kBlockingWaitInBlock);
        hadError = true;
      } else if (op->getParentOfType<mlir::scf::IfOp>()) {
        op.emitError("a blocking wait cannot be inside one arm of an scf.if; "
                     "wait before or after the branch");
        hadError = true;
      }
    });
    // A call of a controller that blocks on calls holds every line, which a
    // schedule block, placed whole, cannot.
    if (pulse::blocksOnCalls(module))
      module.walk([&](classical::SubmitOp op) {
        if (op->getParentOfType<pulse::ScheduleBlockOp>()) {
          op.emitError(classical::kBlockingCallInBlock);
          hadError = true;
        }
      });

    hadError |= waveforms.hadError;
    // A wait whose length is not a constant is scheduled at a placeholder
    // length until its arguments are bound; only a variable wait, which
    // carries its bounds, may be.
    module.walk([&](pulse::WaitOp op) {
      auto conversion =
          op.getDuration().getDefiningOp<pulse::DurationFromIntOp>();
      if (op->hasAttr("pulse.variable_wait") ||
          (conversion && pulse::constantInteger(conversion.getCycles())))
        return;
      op.emitError(
          "a wait whose length is not a constant must be marked "
          "pulse.variable_wait, with its bounds in pulse.min_cycles and "
          "pulse.max_cycles");
      hadError = true;
    });

    // Check 3: Monotone, non-overlapping time on each physical line, and no
    // conditional operation before its measurement. SSA line values change
    // after every operation, so trace them back to allocation.
    module.walk([&](mlir::func::FuncOp funcOp) {
      if (funcOp.isExternal())
        return;
      TimingChecker checker;
      checker.checkFunction(funcOp);
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
