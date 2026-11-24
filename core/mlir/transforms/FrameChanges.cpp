// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Frame changes on the timeline of a scheduled program.
//
// The four frame changes (shift_phase, set_phase, shift_frequency and
// set_frequency) take no time. Once the program is scheduled each gets a
// place on its tone's timeline, as `start_cycles`, `duration_cycles` = 0 and
// `period_cycles` in loops: where it is made, as the frame accrues phase at
// its new frequency from there. That is where its tone is free, the end of
// the latest drive or readout on it, or, when later, where the program
// reaches it on the tone's line (its own line, else the line it last or
// next plays on), after the waits, pulses and syncs before it there, and
// after the readouts it depends on (in an arm of a branch on one, or with
// a value computed from one); never later than the drive or readout that
// plays with it. So a frequency changed before a wait and changed back
// after it runs at the changed frequency through the wait. Where that time
// is not one pattern over the iterations of a loop (a change before the
// first pulse of an iteration on a tone the previous iteration played), the
// change is placed at the start of the next drive or readout that plays
// with it. In a loop whose iterations do not play its tone, a change is at
// the start of each iteration. A change takes no time: no drive, readout,
// loop or program starts, ends or lasts otherwise for it.
//
// A change that a drive or readout plays with, later in the shot or in a
// later iteration of a loop, is folded by the export into that pulse's
// frame. Every change is kept, including one that no drive or readout
// plays with on some path: the program says it, so it is meant. The export
// gives each such execution as a frame update of its own, from what this
// pass decides on the code alone:
//
// - A change no drive or readout can play with is marked `pulse.unconsumed`
//   and is a frame update where it is made.
// - The changes a loop carries out of its last iteration to nothing that
//   plays with them are listed on the loop (`pulse.exit_updates`), and are
//   frame updates after it.
// - The changes made before a branch that one arm, and nothing after the
//   branch, plays with are listed on the other arm (`pulse.then_updates`,
//   `pulse.else_updates`, also for an if without an else), and are frame
//   updates at its start.
//
// The listed changes carry an id (`pulse.carried`), and every loop around
// one lists it (`pulse.carried_in`). A change pending on a path this does
// not follow exactly (one that a loop carries through a branch, or out of
// a loop whose iterations do not play its tone) is marked
// `pulse.unconsumed` instead: a frame update at each execution, which the
// pulse that plays with it on another path takes as well. A controller
// whose frames persist from one shot to the next declares it with the
// module's `pulse.persistent_frames`: then nothing is marked, and the
// export gives instead the frame each tone ends the shot in.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

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

#include "cudaq-pulse/Analysis/Execution.h"
#include "cudaq-pulse/Analysis/LineTrace.h"
#include "cudaq-pulse/Analysis/Timing.h"
#include "cudaq-pulse/Transforms/Passes.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"

#include <array>
#include <optional>

#if PULSE_COUPLED == 0
namespace pulse {
namespace coupled {
/// The frame changes of a module that holds a coupling line: this file
/// compiled as the variant that knows its ops (Variant.h). Whether it
/// changed the module.
bool placeFrameChanges(mlir::ModuleOp module);
} // namespace coupled
} // namespace pulse
#endif

namespace {

bool isFrameChange(mlir::Operation *op) {
  return mlir::isa<pulse::ShiftPhaseOp, pulse::SetPhaseOp,
                   pulse::ShiftFrequencyOp, pulse::SetFrequencyOp>(op);
}

bool isPulse(mlir::Operation *op) {
  return mlir::isa<pulse::DriveOp, pulse::ReadoutOp>(op);
}

mlir::Value toneOf(mlir::Operation *pulse) {
  if (auto drive = mlir::dyn_cast<pulse::DriveOp>(pulse))
    return drive.getTone();
  return mlir::cast<pulse::ReadoutOp>(pulse).getTone();
}

mlir::Value lineOf(mlir::Operation *pulse) {
  if (auto drive = mlir::dyn_cast<pulse::DriveOp>(pulse))
    return drive.getLine();
  return mlir::cast<pulse::ReadoutOp>(pulse).getLine();
}

bool isLine(mlir::Value value) {
  return mlir::isa<pulse::DriveLineType, pulse::ReadoutLineType>(
      value.getType());
}

/// A time in every iteration of the enclosing loops: `base` plus k_l times
/// `periods[l]`, outermost first.
struct Time {
  int64_t base = 0;
  llvm::SmallVector<int64_t, 2> periods;
};

/// Whether `a` is at or after `b` in every iteration, both with a period
/// for each of the same loops.
bool notBefore(const Time &a, const Time &b) {
  if (a.base < b.base)
    return false;
  for (auto [x, y] : llvm::zip_equal(a.periods, b.periods))
    if (x < y)
      return false;
  return true;
}

/// Where the program makes each frame change on the line of its tone: the
/// version of that line current there, after the ops before the change on
/// it. The tone's line is its own, from `pulse.get_drive_line` or
/// `pulse.get_readout_line`; for a tone of its own (`pulse.tone`) it is the
/// line the tone last played on, else the one it next plays on.
class LinesAtChanges {
public:
  explicit LinesAtChanges(mlir::func::FuncOp function) {
    for (mlir::Block &block : function.getBody())
      walk(block);
  }

  mlir::Value at(mlir::Operation *change) const { return lines.lookup(change); }

private:
  pulse::AllocationTracer trace;
  /// The current version of each line, by its acquisition, and the earlier
  /// ones replaced, undone when a region is left.
  llvm::DenseMap<mlir::Operation *, mlir::Value> current;
  llvm::SmallVector<std::pair<mlir::Operation *, mlir::Value>> replaced;
  llvm::DenseMap<mlir::Operation *, mlir::Value> lines;

  void note(mlir::Value value) {
    if (!isLine(value))
      return;
    if (mlir::Operation *line = trace(value)) {
      mlir::Value &version = current[line];
      replaced.emplace_back(line, version);
      version = value;
    }
  }

  void walk(mlir::Block &block) {
    for (mlir::BlockArgument argument : block.getArguments())
      note(argument);
    for (mlir::Operation &op : block) {
      if (isFrameChange(&op)) {
        if (mlir::Operation *line = lineOfTone(&op))
          if (mlir::Value version = current.lookup(line))
            lines[&op] = version;
        continue;
      }
      for (mlir::Region &region : op.getRegions())
        for (mlir::Block &inner : region) {
          size_t mark = replaced.size();
          walk(inner);
          while (replaced.size() > mark) {
            auto [line, version] = replaced.pop_back_val();
            current[line] = version;
          }
        }
      for (mlir::Value result : op.getResults())
        note(result);
    }
  }

  /// The acquisition of the line `change` is made on, or null.
  mlir::Operation *lineOfTone(mlir::Operation *change) {
    mlir::Value tone = change->getOperand(0);
    if (mlir::Operation *line = trace(tone))
      return line;
    for (mlir::Value value = tone;;) {
      mlir::Operation *def = value.getDefiningOp();
      if (!def || !isFrameChange(def)) {
        if (def && isPulse(def))
          return trace(lineOf(def));
        break;
      }
      value = def->getOperand(0);
    }
    if (mlir::Operation *next = nextPulse(change->getResult(0)))
      return trace(lineOf(next));
    return nullptr;
  }

  static mlir::Operation *nextPulse(mlir::Value tone) {
    for (mlir::Operation *user : tone.getUsers()) {
      if (isPulse(user) && toneOf(user) == tone)
        return user;
      if (isFrameChange(user))
        if (mlir::Operation *next = nextPulse(user->getResult(0)))
          return next;
    }
    return nullptr;
  }
};

/// Whether iteration arg `index` of `loop` reaches the yield through frame
/// changes alone: no pulse in the body plays on that tone.
bool passesThrough(mlir::scf::ForOp loop, unsigned index) {
  mlir::Value value = loop.getBody()->getTerminator()->getOperand(index);
  while (mlir::Operation *def = value.getDefiningOp()) {
    if (!isFrameChange(def))
      return false;
    value = def->getOperand(0);
  }
  return value == loop.getRegionIterArg(index);
}

/// The names of the attributes the pass sets, interned once a run.
struct Names {
  explicit Names(mlir::MLIRContext *context)
      : unconsumed(mlir::StringAttr::get(context, "pulse.unconsumed")),
        carried(mlir::StringAttr::get(context, "pulse.carried")),
        carriedIn(mlir::StringAttr::get(context, "pulse.carried_in")),
        exitUpdates(mlir::StringAttr::get(context, "pulse.exit_updates")),
        thenUpdates(mlir::StringAttr::get(context, "pulse.then_updates")),
        elseUpdates(mlir::StringAttr::get(context, "pulse.else_updates")) {}

  /// The marks, which a run first removes.
  mlir::StringAttr unconsumed, carried, carriedIn, exitUpdates, thenUpdates,
      elseUpdates;
  std::array<mlir::StringAttr, 6> marks() const {
    return {unconsumed,  carried,     carriedIn,
            exitUpdates, thenUpdates, elseUpdates};
  }
};

class FramePlacement {
public:
  FramePlacement(mlir::MLIRContext *context, const Names &names)
      : i64(mlir::IntegerType::get(context, 64)), context(context),
        names(names) {}

  /// Places `change`, made where `line` (null if none) is the version of
  /// its tone's line.
  void place(mlir::Operation *change, mlir::Value line) {
    size_t depth = pulse::enclosingLoops(change).size();
    mlir::Operation *source = nullptr;
    std::optional<Time> time = freeAt(change->getOperand(0));
    if (time) {
      source = sourceOf(change->getOperand(0));
      time->periods.resize(depth, 0);
      std::optional<Time> reached = line ? freeAt(line) : std::nullopt;
      mlir::Operation *by = nullptr;
      if (reached) {
        reached->periods.resize(depth, 0);
        by = sourceOf(line);
      }
      mlir::Operation *readout = nullptr;
      if (std::optional<Time> known = measured(change, depth, readout);
          known && later(*known, reached)) {
        reached = std::move(known);
        by = readout;
      }
      // Later than its tone is free, and no later than the drive or readout
      // that plays with it (on another line, it can be earlier).
      if (reached && later(*reached, time)) {
        std::optional<Time> next = nextPulse(change->getResult(0), depth);
        if (!next || notBefore(*next, *reached)) {
          time = std::move(reached);
          source = by;
        }
      }
    } else if ((time = nextPulse(change->getResult(0), depth)))
      source = change;
    else
      time = Time{};
    time->periods.resize(depth, 0);
    if (depth) {
      llvm::SmallVector<mlir::scf::ForOp> loops = pulse::enclosingLoops(change);
      for (size_t l = 0; l < depth; ++l) {
        if (time->periods[l] != 0 || (source && loops[l]->isAncestor(source)))
          continue;
        if (std::optional<Time> first = firstIteration(loops[l])) {
          // At the start of each iteration: when the tone is free can be
          // later, and a change there could end after the loop.
          time->base = first->base;
          std::copy_n(first->periods.begin(), l + 1, time->periods.begin());
          std::fill(time->periods.begin() + l + 1, time->periods.end(), 0);
        }
      }
    }
    // Spacing for an update bus holds a change back (`pulse.hold_cycles`),
    // never past the drive or readout that plays with it.
    if (auto hold =
            change->getAttrOfType<mlir::IntegerAttr>("pulse.hold_cycles")) {
      Time held = *time;
      held.base += hold.getInt();
      std::optional<Time> next = nextPulse(change->getResult(0), depth);
      if (!next || notBefore(*next, held))
        time = std::move(held);
    }
    pulse::setTiming(change, time->base, 0, time->periods);
  }

private:
  /// Whether `a` is later than `b` (none) in some iteration and not earlier
  /// in any, both with periods for the same loops.
  static bool later(const Time &a, const std::optional<Time> &b) {
    return !b || (notBefore(a, *b) && !notBefore(*b, a));
  }

  /// Where the measurements end that `change` depends on directly, those
  /// the branches around it test and those its value is computed from, and
  /// the `readout` that ends last: a change the measurement chooses, in an
  /// arm or by its value, is made once it is in, as if its line were synced
  /// with the readout's.
  std::optional<Time> measured(mlir::Operation *change, size_t depth,
                               mlir::Operation *&readout) {
    std::optional<Time> latest;
    llvm::SmallVector<mlir::Value, 4> work{change->getOperand(1)};
    for (mlir::Operation *op = change->getParentOp();
         op && !mlir::isa<mlir::func::FuncOp>(op); op = op->getParentOp())
      if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op))
        work.push_back(branch.getCondition());
    llvm::DenseSet<mlir::Value> seen;
    while (!work.empty()) {
      mlir::Value value = work.pop_back_val();
      mlir::Operation *def = value.getDefiningOp();
      if (!def || !seen.insert(value).second)
        continue;
      if (mlir::isa<pulse::ReadoutOp>(def)) {
        if (!pulse::hasStart(def))
          continue;
        Time end = timeOf(def, pulse::scheduledDuration(def).value_or(0));
        end.periods.resize(depth, 0);
        if (later(end, latest)) {
          latest = std::move(end);
          readout = def;
        }
      } else if (def->getNumRegions() == 0 &&
                 !mlir::isa<classical::ClassicalDialect>(def->getDialect())) {
        llvm::append_range(work, def->getOperands());
      }
    }
    return latest;
  }

  Time timeOf(mlir::Operation *op, int64_t offset) const {
    Time time{*pulse::startCycles(op) + offset, {}};
    llvm::ArrayRef<int64_t> periods = pulse::periodCycles(op);
    time.periods.assign(periods.begin(), periods.end());
    return time;
  }

  /// The start of the first iteration of `loop`, at the first of each loop
  /// around it, and the period of every scheduled op in it (`loop`'s at its
  /// depth); none for a loop without one.
  std::optional<Time> firstIteration(mlir::scf::ForOp loop) {
    auto [it, inserted] = firsts.try_emplace(loop);
    if (!inserted)
      return it->second;
    size_t level = pulse::enclosingLoops(loop).size();
    std::optional<Time> first;
    loop.getBody()->walk([&](mlir::Operation *op) {
      if (isFrameChange(op) || !pulse::hasStart(op))
        return;
      Time time = timeOf(op, 0);
      if (time.periods.size() <= level)
        return;
      if (!first || time.base < first->base)
        first = time;
    });
    it->second = first;
    return first;
  }

  /// The op `freeAt(tone)` times: the drive, readout, loop or branch before
  /// the frame changes `tone` follows and the loops it passes through, or
  /// none for the function's tone. For a line, the op that made the version.
  static mlir::Operation *sourceOf(mlir::Value tone) {
    while (true) {
      if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(tone)) {
        auto loop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(
            argument.getOwner()->getParentOp());
        if (!loop || argument.getArgNumber() == 0)
          return nullptr;
        tone = loop.getInitArgs()[argument.getArgNumber() - 1];
        continue;
      }
      mlir::Operation *def = tone.getDefiningOp();
      if (!isFrameChange(def))
        return def;
      tone = def->getOperand(0);
    }
  }

  /// When `tone` is free, as a pattern over the loops around its block; for
  /// a line, when the ops before it on the line end.
  /// Each value's is found once: changes are placed in program order and a
  /// tone is free at what comes before it, so it does not change after.
  /// Else each change after a run of branches would look through them all.
  /// The values findFree reads are found first, from a stack: a run of
  /// branches as long as the program is no deeper than one.
  std::optional<Time> freeAt(mlir::Value tone) {
    if (auto found = free.find(tone); found != free.end())
      return found->second;
    llvm::SmallVector<mlir::Value, 8> work{tone};
    llvm::SmallVector<mlir::Value, 2> reads;
    while (!work.empty()) {
      mlir::Value value = work.back();
      if (free.contains(value)) {
        work.pop_back();
        continue;
      }
      reads.clear();
      readsOf(value, reads);
      bool ready = true;
      for (mlir::Value read : reads)
        if (!free.contains(read)) {
          work.push_back(read);
          ready = false;
        }
      if (!ready)
        continue;
      work.pop_back();
      std::optional<Time> time = findFree(value);
      free.try_emplace(value, std::move(time));
    }
    return free.find(tone)->second;
  }

  /// The values findFree(`tone`) asks freeAt for.
  static void readsOf(mlir::Value tone,
                      llvm::SmallVectorImpl<mlir::Value> &reads) {
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(tone)) {
      auto loop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(
          argument.getOwner()->getParentOp());
      if (!loop || argument.getArgNumber() == 0)
        return;
      unsigned index = argument.getArgNumber() - 1;
      if (passesThrough(loop, index))
        reads.push_back(loop.getInitArgs()[index]);
      return;
    }
    mlir::Operation *def = tone.getDefiningOp();
    unsigned index = mlir::cast<mlir::OpResult>(tone).getResultNumber();
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(def)) {
      if (pulse::constantTripCount(loop).value_or(0) == 0 ||
          passesThrough(loop, index))
        reads.push_back(loop.getInitArgs()[index]);
      else
        reads.push_back(loop.getBody()->getTerminator()->getOperand(index));
    } else if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(def)) {
      if (branch.getElseRegion().empty())
        return;
      reads.push_back(branch.thenYield().getOperand(index));
      reads.push_back(branch.elseYield().getOperand(index));
    } else if (auto sync = mlir::dyn_cast<pulse::SyncOp>(def)) {
      llvm::append_range(reads, sync.getLines());
    }
  }

  std::optional<Time> findFree(mlir::Value tone) {
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(tone)) {
      auto loop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(
          argument.getOwner()->getParentOp());
      if (!loop) {
        if (mlir::isa_and_nonnull<mlir::func::FuncOp>(
                argument.getOwner()->getParentOp()))
          return Time{};
        return std::nullopt;
      }
      if (argument.getArgNumber() == 0)
        return std::nullopt;
      unsigned index = argument.getArgNumber() - 1;
      if (!passesThrough(loop, index))
        return std::nullopt;
      std::optional<Time> outside = freeAt(loop.getInitArgs()[index]);
      if (outside)
        outside->periods.resize(pulse::enclosingLoops(loop).size() + 1, 0);
      return outside;
    }
    mlir::Operation *def = tone.getDefiningOp();
#if PULSE_COUPLED
    if (isPulse(def) || mlir::isa<pulse::WaitOp>(def) ||
        (pulse::kCoupled && mlir::isa<pulse::DriveBasebandOp>(def))) {
#else
    if (isPulse(def) || mlir::isa<pulse::WaitOp>(def)) {
#endif
      if (!pulse::hasStart(def))
        return std::nullopt;
      return timeOf(def, pulse::scheduledDuration(def).value_or(0));
    }
    if (isFrameChange(def)) {
      if (!pulse::hasStart(def))
        return std::nullopt;
      return timeOf(def, 0);
    }
    unsigned index = mlir::cast<mlir::OpResult>(tone).getResultNumber();
    size_t depth = pulse::enclosingLoops(def).size();
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(def)) {
      int64_t trips = pulse::constantTripCount(loop).value_or(0);
      std::optional<Time> time;
      if (trips == 0 || passesThrough(loop, index)) {
        time = freeAt(loop.getInitArgs()[index]);
      } else {
        time = freeAt(loop.getBody()->getTerminator()->getOperand(index));
        if (time && time->periods.size() == depth + 1) {
          time->base += (trips - 1) * time->periods.back();
          time->periods.pop_back();
        } else {
          time.reset();
        }
      }
      if (time)
        time->periods.resize(depth, 0);
      return time;
    }
    if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(def)) {
      if (branch.getElseRegion().empty())
        return std::nullopt;
      std::optional<Time> then = freeAt(branch.thenYield().getOperand(index));
      std::optional<Time> otherwise =
          freeAt(branch.elseYield().getOperand(index));
      if (!then || !otherwise)
        return std::nullopt;
      then->periods.resize(depth, 0);
      otherwise->periods.resize(depth, 0);
      if (then->periods != otherwise->periods)
        return std::nullopt;
      then->base = std::max(then->base, otherwise->base);
      return then;
    }
    // A sync's lines are all free where the latest of them is.
    if (auto sync = mlir::dyn_cast<pulse::SyncOp>(def)) {
      std::optional<Time> latest;
      for (mlir::Value line : sync.getLines()) {
        std::optional<Time> time = freeAt(line);
        if (!time)
          return std::nullopt;
        time->periods.resize(depth, 0);
        if (latest && latest->periods != time->periods)
          return std::nullopt;
        if (!latest || time->base > latest->base)
          latest = std::move(time);
      }
      return latest;
    }
    if (mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp,
#if PULSE_COUPLED
                  pulse::ToneOp>(def) ||
        (pulse::kCoupled && mlir::isa<pulse::GetCouplingLineOp>(def)))
#else
                  pulse::ToneOp>(def))
#endif
      return Time{};
    return std::nullopt;
  }

  /// The start of the first drive or readout that plays with the change
  /// whose result is `tone`, in iteration 0 of the loops inside `depth`.
  std::optional<Time> nextPulse(mlir::Value tone, size_t depth) {
    for (mlir::Operation *user : tone.getUsers()) {
      if (isPulse(user) && toneOf(user) == tone && pulse::hasStart(user)) {
        Time time = timeOf(user, 0);
        time.periods.resize(depth, 0);
        return time;
      }
      if (isFrameChange(user))
        if (auto time = nextPulse(user->getResult(0), depth))
          return time;
    }
    return std::nullopt;
  }

  mlir::IntegerType i64;
  mlir::MLIRContext *context;
  const Names &names;
  llvm::DenseMap<mlir::Operation *, std::optional<Time>> firsts;
  llvm::DenseMap<mlir::Value, std::optional<Time>> free;
};

/// The tone values a drive or readout plays with, later on some path: the
/// tones pulses play on, and what they continue through frame changes,
/// loops and branches.
llvm::DenseSet<mlir::Value> liveTones(mlir::func::FuncOp function) {
  llvm::DenseSet<mlir::Value> live;
  llvm::SmallVector<mlir::Value> work;
  auto keep = [&](mlir::Value value) {
    if (value && live.insert(value).second)
      work.push_back(value);
  };
  function.walk([&](mlir::Operation *op) {
    if (isFrameChange(op))
      return;
    if (isPulse(op)) {
      keep(toneOf(op));
      return;
    }
    // A tone that leaves the function or reaches an op this does not know
    // is kept as it is.
    if (mlir::isa<mlir::scf::YieldOp, mlir::scf::ForOp>(op))
      return;
    for (mlir::Value operand : op->getOperands())
      if (mlir::isa<pulse::ToneType>(operand.getType()))
        keep(operand);
  });
  while (!work.empty()) {
    mlir::Value value = work.pop_back_val();
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      auto loop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(
          argument.getOwner()->getParentOp());
      if (loop && argument.getArgNumber() > 0) {
        unsigned index = argument.getArgNumber() - 1;
        keep(loop.getInitArgs()[index]);
        keep(loop.getBody()->getTerminator()->getOperand(index));
      }
      continue;
    }
    mlir::Operation *def = value.getDefiningOp();
    unsigned index = mlir::cast<mlir::OpResult>(value).getResultNumber();
    if (isFrameChange(def)) {
      keep(def->getOperand(0));
    } else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(def)) {
      keep(loop.getBody()->getTerminator()->getOperand(index));
      keep(loop.getInitArgs()[index]);
    } else if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(def)) {
      keep(branch.thenYield().getOperand(index));
      if (!branch.getElseRegion().empty())
        keep(branch.elseYield().getOperand(index));
    }
  }
  return live;
}

/// Whether `user` runs after `before` on the same path: after the op
/// holding `before` in a block around both.
bool follows(mlir::Operation *before, mlir::Operation *user) {
  for (mlir::Operation *op = user; op; op = op->getParentOp()) {
    mlir::Block *block = op->getBlock();
    if (!block)
      return false;
    if (mlir::Operation *ancestor = block->findAncestorOpInBlock(*before))
      return ancestor != op && ancestor->isBeforeInBlock(op);
  }
  return false;
}

/// The frame changes of a function that some path does not play with, and
/// where the export gives them (see the top of this file).
class PendingChanges {
public:
  PendingChanges(mlir::func::FuncOp function, const Names &names,
                 int64_t &nextId)
      : live(liveTones(function)), names(names), nextId(nextId) {}

  llvm::DenseSet<mlir::Value> live;
  /// The changes that are frame updates where they are made.
  llvm::SetVector<mlir::Operation *> inPlace;
  /// The id of each listed change, and the lists of each loop and branch:
  /// (op, attribute) -> ids in program order.
  llvm::MapVector<mlir::Operation *, int64_t> ids;
  llvm::MapVector<std::pair<mlir::Operation *, mlir::StringAttr>,
                  llvm::SmallVector<int64_t>>
      lists;

  void unconsumed(mlir::Operation *change) {
    if (!live.contains(change->getResult(0)))
      inPlace.insert(change);
  }

  /// A loop's tones that its last iteration leaves to nothing that plays.
  void exits(mlir::scf::ForOp loop) {
    for (unsigned j = 0; j < loop.getNumResults(); ++j) {
      mlir::Value yielded = loop.getBody()->getTerminator()->getOperand(j);
      if (mlir::isa<pulse::ToneType>(yielded.getType()) &&
          !live.contains(loop.getResult(j)) && live.contains(yielded))
        list(loop, names.exitUpdates, yielded);
    }
  }

  /// The tones from before a branch that an arm leaves to nothing that
  /// plays, while the other arm plays them.
  void arms(mlir::scf::IfOp branch) {
    llvm::SetVector<mlir::Value> outside;
    branch->walk([&](mlir::Operation *op) {
      for (mlir::Value operand : op->getOperands())
        if (mlir::isa<pulse::ToneType>(operand.getType()) &&
            !branch->isAncestor(operand.getParentRegion()->getParentOp()))
          outside.insert(operand);
    });
    for (mlir::Value tone : outside) {
      if (!live.contains(tone) || playsAfter(tone, branch))
        continue;
      if (!playsIn(branch.getThenRegion(), tone))
        list(branch, names.thenUpdates, tone);
      if (!playsIn(branch.getElseRegion(), tone))
        list(branch, names.elseUpdates, tone);
    }
  }

private:
  const Names &names;
  int64_t &nextId;

  /// Whether `user` of `tone` plays with it, or passes it on to what does.
  bool plays(mlir::Operation *user, mlir::Value tone) const {
    if (isPulse(user))
      return toneOf(user) == tone;
    if (isFrameChange(user))
      return live.contains(user->getResult(0));
    if (mlir::isa<mlir::scf::YieldOp, mlir::scf::ForOp>(user)) {
      mlir::Operation *parent =
          mlir::isa<mlir::scf::YieldOp>(user) ? user->getParentOp() : user;
      for (mlir::OpOperand &use : user->getOpOperands()) {
        if (use.get() != tone)
          continue;
        unsigned j = use.getOperandNumber();
        if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(parent)) {
          if (user == parent) {
            if (j < 3)
              continue;
            j -= 3;
          }
          if (live.contains(loop.getResult(j)) ||
              live.contains(loop.getRegionIterArg(j)))
            return true;
        } else if (mlir::isa<mlir::scf::IfOp>(parent)) {
          if (live.contains(parent->getResult(j)))
            return true;
        } else {
          return true;
        }
      }
      return false;
    }
    return true;
  }

  bool playsIn(mlir::Region &region, mlir::Value tone) const {
    return llvm::any_of(tone.getUsers(), [&](mlir::Operation *user) {
      return region.isAncestor(user->getParentRegion()) && plays(user, tone);
    });
  }

  bool playsAfter(mlir::Value tone, mlir::Operation *branch) const {
    return llvm::any_of(tone.getUsers(), [&](mlir::Operation *user) {
      return !branch->isAncestor(user) && follows(branch, user) &&
             plays(user, tone);
    });
  }

  /// The changes pending on `tone`, in program order: since the latest
  /// pulse on it, through loops whose iterations play it (each loop's
  /// last). False where the path is not one: through a branch, or a loop
  /// that does not play the tone or runs a number of times not known.
  static bool tail(mlir::Value tone,
                   llvm::SmallVectorImpl<mlir::Operation *> &out) {
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(tone))
      return mlir::isa_and_nonnull<mlir::func::FuncOp>(
          argument.getOwner()->getParentOp());
    mlir::Operation *def = tone.getDefiningOp();
    if (isPulse(def) || mlir::isa<pulse::GetDriveLineOp,
                                  pulse::GetReadoutLineOp, pulse::ToneOp>(def))
      return true;
    if (isFrameChange(def)) {
      if (!tail(def->getOperand(0), out))
        return false;
      out.push_back(def);
      return true;
    }
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(def)) {
      unsigned j = mlir::cast<mlir::OpResult>(tone).getResultNumber();
      std::optional<int64_t> trips = pulse::constantTripCount(loop);
      if (!trips)
        return false;
      if (*trips == 0)
        return tail(loop.getInitArgs()[j], out);
      if (passesThrough(loop, j))
        return false;
      return tail(loop.getBody()->getTerminator()->getOperand(j), out);
    }
    return false;
  }

  /// Every change on the paths to `tone` since the pulses before it.
  void fallback(mlir::Value tone) {
    llvm::SmallVector<mlir::Value> work{tone};
    llvm::DenseSet<mlir::Value> seen;
    while (!work.empty()) {
      mlir::Value value = work.pop_back_val();
      if (!value || !seen.insert(value).second)
        continue;
      if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
        auto loop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(
            argument.getOwner()->getParentOp());
        if (loop && argument.getArgNumber() > 0) {
          unsigned j = argument.getArgNumber() - 1;
          work.push_back(loop.getInitArgs()[j]);
          work.push_back(loop.getBody()->getTerminator()->getOperand(j));
        }
        continue;
      }
      mlir::Operation *def = value.getDefiningOp();
      unsigned j = mlir::cast<mlir::OpResult>(value).getResultNumber();
      if (isFrameChange(def)) {
        inPlace.insert(def);
        work.push_back(def->getOperand(0));
      } else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(def)) {
        work.push_back(loop.getInitArgs()[j]);
        work.push_back(loop.getBody()->getTerminator()->getOperand(j));
      } else if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(def)) {
        work.push_back(branch.thenYield().getOperand(j));
        if (!branch.getElseRegion().empty())
          work.push_back(branch.elseYield().getOperand(j));
      }
    }
  }

  void list(mlir::Operation *site, mlir::StringAttr name, mlir::Value tone) {
    llvm::SmallVector<mlir::Operation *> changes;
    if (!tail(tone, changes)) {
      fallback(tone);
      return;
    }
    if (changes.empty())
      return;
    llvm::SmallVector<int64_t> &into = lists[{site, name}];
    for (mlir::Operation *change : changes) {
      auto [it, inserted] = ids.try_emplace(change, nextId);
      if (inserted)
        ++nextId;
      into.push_back(it->second);
    }
  }
};

/// Places the frame changes of `module` and marks those some path does not
/// play with; whether it changed the module.
LLVM_ATTRIBUTE_ALWAYS_INLINE static bool
placeFrameChanges(mlir::ModuleOp module) {
#if PULSE_COUPLED == 0
  // A module that holds a coupling line is placed by the variant of this
  // file that knows its ops (Variant.h).
  if (module->hasAttr(pulse::kCouplingLinesAttr)) [[unlikely]]
    return pulse::coupled::placeFrameChanges(module);
#endif
  mlir::MLIRContext *context = module.getContext();
  bool persistent = pulse::persistentFrames(module);
  auto mark = mlir::UnitAttr::get(context);
  auto i64 = mlir::IntegerType::get(context, 64);
  int64_t nextId = 0;
  const Names names(context);
  bool changed = false;
  for (auto function : module.getOps<mlir::func::FuncOp>()) {
    FramePlacement placement(context, names);
    std::optional<LinesAtChanges> lines;
    bool changes = false;
    function.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
      bool change = isFrameChange(op);
      // Only a frame change, loop or branch carries a mark.
      if (change || mlir::isa<mlir::scf::ForOp, mlir::scf::IfOp>(op)) {
        mlir::DictionaryAttr attributes = op->getRawDictionaryAttrs();
        if (!attributes.empty())
          for (mlir::StringAttr name : names.marks())
            if (attributes.contains(name)) {
              op->removeAttr(name);
              changed = true;
            }
      }
      if (change) {
        if (!lines)
          lines.emplace(function);
        placement.place(op, lines->at(op));
        changes = true;
      }
    });
    changed |= changes;
    // Without a frame change, none is pending anywhere.
    if (persistent || !changes)
      continue;
    PendingChanges pending(function, names, nextId);
    function.walk([&](mlir::Operation *op) {
      if (isFrameChange(op))
        pending.unconsumed(op);
      else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op))
        pending.exits(loop);
      else if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op))
        pending.arms(branch);
    });
    for (mlir::Operation *change : pending.inPlace)
      change->setAttr(names.unconsumed, mark);
    llvm::MapVector<mlir::Operation *, llvm::SmallVector<int64_t>> within;
    for (auto [change, id] : pending.ids) {
      change->setAttr(names.carried, mlir::IntegerAttr::get(i64, id));
      for (mlir::scf::ForOp loop : pulse::enclosingLoops(change))
        within[loop].push_back(id);
    }
    for (auto &[site, ids] : pending.lists)
      site.first->setAttr(site.second,
                          mlir::DenseI64ArrayAttr::get(context, ids));
    for (auto &[loop, ids] : within)
      loop->setAttr(names.carriedIn,
                    mlir::DenseI64ArrayAttr::get(context, ids));
  }
  return changed;
}

#if PULSE_COUPLED == 0
struct PulseFrameChangesPass
    : public mlir::PassWrapper<PulseFrameChangesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PulseFrameChangesPass)

  llvm::StringRef getArgument() const override { return "pulse-frame-changes"; }
  llvm::StringRef getDescription() const override {
    return "Place the frame changes of a scheduled program on their tones' "
           "timelines, and mark those some path does not play with";
  }

  void runOnOperation() override {
    // An unchanged module needs no verifying again.
    if (!placeFrameChanges(getOperation()))
      markAllAnalysesPreserved();
  }
};
#endif

} // namespace

#if PULSE_COUPLED == 0
std::unique_ptr<mlir::Pass> pulse::createPulseFrameChangesPass() {
  return std::make_unique<PulseFrameChangesPass>();
}

bool pulse::hasFrameChangeWork(mlir::Operation *op) {
  return op
      ->walk([](mlir::Operation *op) {
        if (isFrameChange(op))
          return mlir::WalkResult::interrupt();
        if (mlir::isa<mlir::scf::ForOp, mlir::scf::IfOp>(op))
          for (llvm::StringRef name :
               {"pulse.carried_in", "pulse.exit_updates", "pulse.then_updates",
                "pulse.else_updates"})
            if (op->hasAttr(name))
              return mlir::WalkResult::interrupt();
        return mlir::WalkResult::advance();
      })
      .wasInterrupted();
}
#else
namespace pulse {
inline namespace coupled {
bool placeFrameChanges(mlir::ModuleOp module) {
  return ::placeFrameChanges(module);
}
} // namespace coupled
} // namespace pulse
#endif
