// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Deterministic ASAP, ALAP and resource-constrained scheduling passes for the
// Pulse dialect, over structured control flow.
//
// A pass simulates the program in order. Every iteration of an `scf.for` is
// simulated until the loop-carried lines all advance by the same amount per
// iteration, after which the remaining iterations are known to repeat that
// shift. The two arms of a measurement-conditioned `scf.if` both start once
// the measurement is available, and the lines leaving the `if` are ready when
// the later arm is done.
//
// Results are written as `start_cycles` / `duration_cycles`, plus
// `period_cycles` for ops in loops (see Analysis/Timing.h). When an op's start
// is not affine in its iteration (a loop's first or last iterations behave
// differently), that iteration is peeled off the loop and the function is
// scheduled again.
//
// A classical call is a `classical.submit`, which holds its link for a few
// cycles and makes a future available a budgeted latency after it ends, and a
// `classical.wait`, a zero-length op at the time the answer is taken. In a
// function that makes calls, every value carries the time it is available, so
// a pulse whose amplitude, tone or branch condition is computed from an
// answer starts after it; a function without calls skips that bookkeeping.
// Eager submissions keep their ASAP time under ALAP. A blocking wait, like a
// variable wait, is a barrier: nothing after it starts before it, nothing
// before it ends after.
//
// A barrier holds back only its timeline (see Timelines below): the lines
// that depend on it. Each timeline keeps its own floor, end and ceiling, its
// own order on each link it shares and its own lanes under rcp, and a loop
// advances the timelines its body plays on. With one timeline (a function
// on one line, or without a run-time wait) a barrier holds back every line.
//
// A `pulse.schedule_block` is scheduled first, innermost first, on its own
// with its own policy. Its shape (length, and when each operand is first
// needed and each result is available) makes it one rigid op for the
// enclosing schedule. Once that is placed, the block's ops are moved to
// their place in it and the block is inlined.

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallBitVector.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>

#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"
#include "cudaq-pulse/Dialect/Pulse/PulseEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseOps.h.inc"

#include "cudaq-pulse/Dialect/Classical/ClassicalDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Classical/ClassicalTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Classical/ClassicalOps.h.inc"

#include "cudaq-pulse/Analysis/Affine.h"
#include "cudaq-pulse/Analysis/ClassicalTiming.h"
#include "cudaq-pulse/Analysis/Controllers.h"
#include "cudaq-pulse/Analysis/Durations.h"
#include "cudaq-pulse/Analysis/LineDelay.h"
#include "cudaq-pulse/Analysis/LineGrid.h"
#include "cudaq-pulse/Analysis/LineTrace.h"
#include "cudaq-pulse/Analysis/Timing.h"

#include <map>

namespace {

constexpr int64_t kNever = std::numeric_limits<int64_t>::max() / 4;

static std::optional<int64_t> traceConstantI64(mlir::Value value) {
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantIntOp>())
    return constant.value();
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
    if (auto attr = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
      return attr.getInt();
  return std::nullopt;
}

static bool isTimedOp(mlir::Operation *op) {
  return mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp,
                   classical::SubmitOp, classical::WaitOp>(op);
}

/// The shape a scheduled `pulse.schedule_block` presents to its parent, in
/// the block's local time. An offset of -1 marks an operand nothing in the
/// block waits for, or a result that passes operand `source` through.
struct BlockShape {
  int64_t length = 0;
  llvm::SmallVector<int64_t> in, out, source;
};

static BlockShape shapeOf(pulse::ScheduleBlockOp block) {
  BlockShape shape;
  auto array = [&](llvm::StringRef name) {
    llvm::SmallVector<int64_t> values;
    if (auto attr = block->getAttrOfType<mlir::DenseI64ArrayAttr>(name))
      values.assign(attr.asArrayRef().begin(), attr.asArrayRef().end());
    return values;
  };
  if (auto length = block->getAttrOfType<mlir::IntegerAttr>("length_cycles"))
    shape.length = length.getInt();
  shape.in = array("in_offset_cycles");
  shape.out = array("out_offset_cycles");
  shape.source = array("out_source");
  return shape;
}

/// The cycles `op` takes. The forward pass checks (placedDuration) that
/// every pulse and wait it places has a known length.
static int64_t durationOf(mlir::Operation *op) {
  if (auto block = mlir::dyn_cast<pulse::ScheduleBlockOp>(op))
    return shapeOf(block).length;
  if (auto drive = mlir::dyn_cast<pulse::DriveOp>(op))
    return pulse::waveformCycles(drive.getPulse()).value_or(0);
  if (auto readout = mlir::dyn_cast<pulse::ReadoutOp>(op))
    return pulse::waveformCycles(readout.getPulse()).value_or(0);
  if (mlir::isa<classical::SubmitOp>(op))
    return classical::intAttr(op, "classical.submit_cycles", 1);
  if (mlir::isa<classical::WaitOp>(op))
    return 0;
  // A variable wait is scheduled at its lowest length, or at its length,
  // clamped to its bounds, once an answer has been bound to it.
  if (op->hasAttr("classical.variable")) {
    int64_t lowest = classical::intAttr(op, "pulse.min_cycles");
    int64_t length = lowest;
    if (auto conversion = mlir::cast<pulse::WaitOp>(op)
                              .getDuration()
                              .getDefiningOp<pulse::DurationFromIntOp>())
      if (auto cycles = pulse::boundNumber(conversion.getCycles()))
        length = std::max(lowest, static_cast<int64_t>(*cycles));
    if (auto highest = op->getAttrOfType<mlir::IntegerAttr>("pulse.max_cycles"))
      length = std::min(length, highest.getInt());
    return length;
  }
  return pulse::waitCycles(mlir::cast<pulse::WaitOp>(op)).value_or(0);
}

/// The cycles a drive, readout or wait takes as the forward pass places it,
/// once it is known to be valid: the schedule places an op for the cycles it
/// takes, so an unknown length is an error rather than a placement that
/// assumes some length.
static mlir::FailureOr<int64_t> placedDuration(mlir::Operation *op) {
  auto checkPulse = [&](mlir::Value waveform) -> mlir::FailureOr<int64_t> {
    std::optional<int64_t> cycles = pulse::waveformCycles(waveform);
    if (!cycles)
      return op->emitError("cannot determine the duration of this pulse; "
                           "specialize the waveform to a constant duration "
                           "first");
    if (*cycles <= 0)
      return op->emitError("a pulse must last at least one cycle, got ")
             << *cycles;
    return *cycles;
  };
  if (auto drive = mlir::dyn_cast<pulse::DriveOp>(op))
    return checkPulse(drive.getPulse());
  if (auto readout = mlir::dyn_cast<pulse::ReadoutOp>(op))
    return checkPulse(readout.getPulse());
  auto wait = mlir::dyn_cast<pulse::WaitOp>(op);
  if (!wait || op->hasAttr("classical.variable"))
    return durationOf(op);
  std::optional<int64_t> cycles = pulse::waitCycles(wait);
  if (!cycles)
    return wait.emitError("cannot determine the length of this wait; give "
                          "it a constant length, or make it a "
                          "pulse.variable_wait with its bounds");
  if (*cycles < 0)
    return wait.emitError("a wait cannot last ") << *cycles << " cycles";
  return *cycles;
}

/// From the end of a submission until its future is available.
static int64_t budgetOf(mlir::Operation *submit) {
  return classical::intAttr(submit, "classical.budget_cycles");
}

/// How many times a submission in the body of `loop` is in flight at once
/// in steady state when the loop's iterations shift by `period`: once per
/// run of it in an iteration (the trips of the loops between them), for
/// ceil(busy / period) iterations each. None when that does not fit.
static std::optional<int64_t> inFlight(mlir::Operation *submit,
                                       mlir::scf::ForOp loop, int64_t period) {
  int64_t busy = durationOf(submit) + budgetOf(submit);
  int64_t runs = (busy + period - 1) / period;
  for (auto inner = submit->getParentOfType<mlir::scf::ForOp>();
       inner && inner != loop;
       inner = inner->getParentOfType<mlir::scf::ForOp>()) {
    auto trips = pulse::constantTripCount(inner);
    if (!trips)
      return std::nullopt;
    if (__builtin_mul_overflow(runs, *trips, &runs))
      return std::nullopt;
  }
  return runs;
}

/// Whether `loop`'s body makes a classical call.
static bool holdsCalls(mlir::scf::ForOp loop) {
  return loop.getBody()
      ->walk([](classical::SubmitOp) { return mlir::WalkResult::interrupt(); })
      .wasInterrupted();
}

/// Lines, tones and measurement results: the values that become available
/// at a time. A tone is busy while a drive or readout plays on it, so pulses
/// that share a tone run in the order the tone is threaded through them.
static bool isTimedType(mlir::Type type) {
  return mlir::isa<pulse::DriveLineType, pulse::ReadoutLineType,
                   pulse::ToneType, pulse::MeasurementType>(type) ||
         pulse::isClassicalType(type);
}

static bool isLineType(mlir::Type type) {
  return mlir::isa<pulse::DriveLineType, pulse::ReadoutLineType>(type);
}

/// Visit the timed ops and schedule blocks of `region`, but not the ops in
/// those blocks: they carry a schedule local to their block.
template <typename Fn>
static void forEachScheduled(mlir::Region &region, Fn &&fn) {
  region.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (mlir::isa<pulse::ScheduleBlockOp>(op)) {
      fn(op);
      return mlir::WalkResult::skip();
    }
    if (isTimedOp(op))
      fn(op);
    return mlir::WalkResult::advance();
  });
}

/// How a loop-carried value takes part in the steady-state test.
enum class Carried : uint8_t {
  Untimed,
  /// Must advance by the loop's period every iteration.
  Timed,
  /// A tone that no pulse in the body plays on keeps its time.
  Tone,
};

static mlir::Value toneOperand(mlir::Operation *op) {
  if (auto drive = mlir::dyn_cast<pulse::DriveOp>(op))
    return drive.getTone();
  if (auto readout = mlir::dyn_cast<pulse::ReadoutOp>(op))
    return readout.getTone();
  return {};
}

static mlir::Value toneResult(mlir::Operation *op) {
  if (auto drive = mlir::dyn_cast<pulse::DriveOp>(op))
    return drive.getUpdatedTone();
  if (auto readout = mlir::dyn_cast<pulse::ReadoutOp>(op))
    return readout.getUpdatedTone();
  return {};
}

/// The measurement results a branch condition is computed from. A link is
/// not followed: the calls before a call on its link end before it starts,
/// and each starts after the measurements it sends, as
/// pulse-classical-check checks (S10, S11), so theirs are ready before
/// its own. Without `throughAnswers`, neither is an answer: a call starts
/// after what it sends (S10) and its answer is due later still, before any
/// op of a branch on it starts (S12), so a branch on an answer is bounded
/// by the answer alone.
static void conditionMeasurements(mlir::Value value,
                                  llvm::SmallVectorImpl<mlir::Value> &found,
                                  llvm::DenseSet<mlir::Value> &seen,
                                  bool throughAnswers) {
  if (mlir::isa<classical::LinkType>(value.getType()) ||
      !seen.insert(value).second)
    return;
  if (mlir::isa<pulse::MeasurementType>(value.getType())) {
    found.push_back(value);
    return;
  }
  mlir::Operation *def = value.getDefiningOp();
  if (!def || (!throughAnswers && mlir::isa<classical::WaitOp>(def)))
    return;
  for (mlir::Value operand : def->getOperands())
    conditionMeasurements(operand, found, seen, throughAnswers);
}

static llvm::SmallVector<mlir::Value>
conditionMeasurements(mlir::Value cond, bool throughAnswers = true) {
  llvm::SmallVector<mlir::Value> found;
  llvm::DenseSet<mlir::Value> seen;
  conditionMeasurements(cond, found, seen, throughAnswers);
  return found;
}

struct Resources {
  bool enabled = false;
  int64_t maxDrives = 0;
  int64_t maxReadouts = 0;
  int64_t readoutLatency = 0;
  int64_t switchPenalty = 0;
};

// ── Timelines ────────────────────────────────────────────────────────────

/// The lines of a function in timelines: a run-time wait (a variable wait,
/// or a blocking wait for an answer) holds back the lines of its timeline
/// only. Two lines share a timeline when one waits for the other: an op
/// joins the timelines of the lines, tones, measurements and answers it is
/// computed from, an op in a branch joins the condition's, a schedule block
/// (placed whole) or a sync joins its inputs', and a loop-carried value
/// joins what it starts from and what each iteration yields.
///
/// Other lines are held back too where their schedule could not just move
/// with the wait: under rcp/alap_rcp when there are more lines than lanes, a
/// blocking wait whose answer nothing uses (it stops everything), and a link
/// with more timelines than slots, which merges them until each has one.
struct Timelines {
  /// The number of timelines, or 0 when one barrier would hold back every
  /// line: a function without run-time waits, or with one timeline.
  int64_t count = 0;
  /// Under rcp/alap_rcp, the drive and readout lines of each timeline: its
  /// own lanes.
  llvm::SmallVector<int64_t> drives, readouts;
};

class TimelineAnalysis {
public:
  TimelineAnalysis(mlir::func::FuncOp function, const Resources &resources)
      : function(function), resources(resources) {}

  /// Find the timelines and write `classical.timeline` on the allocations,
  /// calls and schedule blocks of the function, or remove it if there is
  /// one. Returns whether the calls' attributes changed.
  bool run(Timelines &result) {
    // Every call of a controller that blocks on calls holds every line.
    everyLine = function->getParentOfType<mlir::ModuleOp>()->hasAttr(
        classical::kBlocksOnCallsAttr);
    // Without a run-time wait there is one timeline: skip the analysis.
    bool numbered = false;
    auto scan = function.walk([&](mlir::Operation *op) {
      if (op->hasAttr("classical.variable") ||
          (mlir::isa<classical::WaitOp>(op) &&
           op->hasAttr("classical.blocking")))
        return mlir::WalkResult::interrupt();
      numbered |= op->hasAttr(classical::kTimelineAttr);
      return mlir::WalkResult::advance();
    });
    if (!scan.wasInterrupted()) {
      if (numbered)
        function.walk([](mlir::Operation *op) {
          op->removeAttr(classical::kTimelineAttr);
        });
      return numbered;
    }
    visitBlock(function.getBody().front(), kNone);
    result = decide();
    bool changed = stale;
    if (result.count == 0) {
      if (stale)
        for (auto [op, c] : nodes)
          op->removeAttr(classical::kTimelineAttr);
      return changed;
    }
    mlir::Builder b(function.getContext());
    for (auto [op, c] : nodes)
      op->setAttr(classical::kTimelineAttr, b.getI64IntegerAttr(idOf[find(c)]));
    return true;
  }

private:
  static constexpr unsigned kNone = ~0u;

  unsigned make() {
    parent.push_back(static_cast<unsigned>(parent.size()));
    return parent.back();
  }
  unsigned find(unsigned x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  }
  /// The joined class; the earlier one names it, so numbering is stable.
  unsigned join(unsigned a, unsigned b) {
    if (a == kNone)
      return b;
    if (b == kNone)
      return a;
    a = find(a);
    b = find(b);
    if (a == b)
      return a;
    if (b < a)
      std::swap(a, b);
    parent[b] = a;
    return a;
  }
  unsigned of(mlir::Value value) const {
    auto it = classOf.find(value);
    return it == classOf.end() ? kNone : it->second;
  }
  static bool isLink(mlir::Value value) {
    return mlir::isa<classical::LinkType>(value.getType());
  }
  unsigned operandClass(mlir::Operation *op, unsigned context) {
    unsigned c = context;
    for (mlir::Value operand : op->getOperands())
      if (!isLink(operand))
        c = join(c, of(operand));
    return c;
  }
  void give(mlir::ValueRange values, unsigned c) {
    if (c == kNone)
      return;
    for (mlir::Value value : values)
      if (!isLink(value))
        classOf[value] = c;
  }
  void note(mlir::Operation *op, unsigned c) {
    stale |= op->hasAttr(classical::kTimelineAttr);
    nodes.push_back({op, c});
  }
  /// A call with no inputs that carry time goes with what uses its answer.
  unsigned orphan(unsigned c) { return c != kNone ? c : make(); }

  void visitBlock(mlir::Block &block, unsigned context) {
    for (mlir::Operation &op : block)
      if (!op.hasTrait<mlir::OpTrait::IsTerminator>())
        visitOp(&op, context);
  }

  void visitOp(mlir::Operation *op, unsigned context) {
    if (mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(op)) {
      unsigned c = join(make(), context);
      lines.push_back({op, c});
      note(op, c);
      give(op->getResults(), c);
      return;
    }
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
      auto yield = loop.getBody()->getTerminator();
      auto trips = pulse::constantTripCount(loop);
      bool runs = trips && *trips > 0;
      for (auto [init, arg, result] :
           llvm::zip(loop.getInitArgs(), loop.getRegionIterArgs(),
                     loop.getResults())) {
        if (isLink(arg))
          continue;
        // A value the body does not read, in a loop that runs, reaches
        // nothing from before the loop: each iteration yields its own.
        bool reaches = !runs || !arg.use_empty();
        unsigned c = join(make(), reaches ? join(context, of(init)) : context);
        classOf[arg] = c;
        classOf[result] = c;
      }
      visitBlock(*loop.getBody(), context);
      for (auto [arg, yielded] :
           llvm::zip(loop.getRegionIterArgs(), yield->getOperands()))
        if (!isLink(arg))
          join(of(arg), of(yielded));
      return;
    }
    if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
      unsigned c = join(context, of(branch.getCondition()));
      visitBlock(*branch.thenBlock(), c);
      if (!branch.getElseRegion().empty())
        visitBlock(*branch.elseBlock(), c);
      for (auto [index, result] : llvm::enumerate(branch.getResults())) {
        unsigned r = join(c, of(branch.thenYield().getOperand(index)));
        if (!branch.getElseRegion().empty())
          r = join(r, of(branch.elseYield().getOperand(index)));
        give(result, r);
      }
      return;
    }
    if (auto block = mlir::dyn_cast<pulse::ScheduleBlockOp>(op)) {
      unsigned c = operandClass(op, context);
      if (c == kNone)
        c = make();
      give(block.getBody().getArguments(), c);
      visitBlock(block.getBody().front(), c);
      for (mlir::Value yielded :
           block.getBody().front().getTerminator()->getOperands())
        if (!isLink(yielded))
          c = join(c, of(yielded));
      give(op->getResults(), c);
      note(op, c);
      return;
    }
    if (auto submit = mlir::dyn_cast<classical::SubmitOp>(op)) {
      unsigned c = orphan(operandClass(op, context));
      give(submit.getFuture(), c);
      note(op, c);
      return;
    }
    if (auto wait = mlir::dyn_cast<classical::WaitOp>(op)) {
      unsigned c = orphan(operandClass(op, context));
      give(wait.getValues(), c);
      note(op, c);
      if (op->hasAttr("classical.blocking")) {
        runtimeWait = true;
        everyLine |= llvm::all_of(wait.getValues(),
                                  [](mlir::Value v) { return v.use_empty(); });
      }
      return;
    }
    runtimeWait |= op->hasAttr("classical.variable");
    unsigned c = operandClass(op, context);
    for (mlir::Region &region : op->getRegions())
      for (mlir::Block &nested : region)
        visitBlock(nested, c);
    give(op->getResults(), c);
  }

  /// Number the classes that hold lines in allocation order; a call or
  /// block whose class holds none goes with the first line.
  void number() {
    llvm::DenseSet<unsigned> withLines;
    for (auto [op, c] : lines)
      withLines.insert(find(c));
    for (auto [op, c] : nodes)
      if (!withLines.count(find(c)))
        join(c, lines.front().second);
    idOf.clear();
    for (auto [op, c] : lines)
      idOf.try_emplace(find(c), static_cast<int64_t>(idOf.size()));
  }

  Timelines decide() {
    Timelines result;
    if (!runtimeWait || everyLine || lines.size() < 2)
      return result;
    if (resources.enabled) {
      int64_t drives = 0, readouts = 0;
      for (auto [op, c] : lines)
        ++(mlir::isa<pulse::GetDriveLineOp>(op) ? drives : readouts);
      if (drives > resources.maxDrives || readouts > resources.maxReadouts)
        return result;
    }
    auto module = function->getParentOfType<mlir::ModuleOp>();
    // Merge the timelines of a link round-robin until it has a slot for each.
    while (true) {
      number();
      std::map<std::pair<int64_t, int64_t>, llvm::SmallVector<unsigned>>
          perLink;
      for (auto [op, c] : nodes) {
        if (!mlir::isa<classical::SubmitOp>(op))
          continue;
        auto &list =
            perLink[{classical::intAttr(op, "classical.device"),
                     classical::intAttr(op, classical::kInstanceAttr, -1)}];
        if (!llvm::is_contained(list, find(c)))
          list.push_back(find(c));
      }
      bool merged = false;
      for (auto &[key, list] : perLink) {
        auto slots = static_cast<size_t>(std::max<int64_t>(
            1, classical::linkModel(module, key.first).slots));
        for (size_t k = slots; k < list.size(); ++k) {
          join(list[k], list[k % slots]);
          merged = true;
        }
      }
      if (!merged)
        break;
    }
    number();
    if (idOf.size() < 2)
      return result;
    result.count = static_cast<int64_t>(idOf.size());
    if (resources.enabled) {
      result.drives.assign(idOf.size(), 0);
      result.readouts.assign(idOf.size(), 0);
      for (auto [op, c] : lines)
        ++(mlir::isa<pulse::GetDriveLineOp>(op)
               ? result.drives
               : result.readouts)[idOf[find(c)]];
    }
    return result;
  }

  mlir::func::FuncOp function;
  Resources resources;
  llvm::SmallVector<unsigned> parent;
  llvm::DenseMap<mlir::Value, unsigned> classOf;
  llvm::SmallVector<std::pair<mlir::Operation *, unsigned>> lines, nodes;
  llvm::DenseMap<unsigned, int64_t> idOf;
  bool runtimeWait = false, everyLine = false, stale = false;
};

struct Sample {
  llvm::SmallVector<int64_t, 2> iteration;
  int64_t start;
};

/// A loop run whose iterations from `from` on were not simulated because
/// each one shifts the previous by `period`; or, with `periods`, shifts
/// each timed op by the period of its own (see LoopGroups).
struct Extrapolation {
  mlir::scf::ForOp loop;
  int64_t period;
  std::shared_ptr<const llvm::DenseMap<mlir::Operation *, int64_t>> periods;

  std::optional<int64_t> periodOf(mlir::Operation *op) const {
    if (!periods)
      return period;
    auto it = periods->find(op);
    if (it == periods->end())
      return std::nullopt;
    return it->second;
  }
};

/// The carried values of a loop in groups that no op of its body relates:
/// each op's timing depends on the values of one group only, so each group
/// may move by a period of its own. `ofArg` has each carried value's group
/// and `ofOp` each timed op's in the body, nested ones included; `outside`
/// the measurements from before the loop that its branches test, which
/// hold back the ops in them until they are ready.
///
/// A branch relates the group of its condition and those of what it
/// decides one way only: forward, the condition holds back the arms and
/// what follows them; backward, they hold back the condition. Each of
/// `branches` is such a pair, the condition's group first. A group so held
/// back may move faster than the one holding it back: the constraint
/// between them, once slack, only grows slacker. (Groups that hold each
/// other back can then only move together.)
///
/// Backward, a call's wait moves with what its answer decides rather than
/// with its request, and so does a lazy request, due by its wait, with the
/// link it threads: `ofWaitBackward` has the group of each such wait and
/// request, `ofArgBackward` each carried value's group backward, and
/// `mixedWaits` whether some wait's answer decides several groups or some
/// link is due by several.
struct LoopGroups {
  unsigned count = 0;
  /// Where a link is due by several groups (so `mixedWaits`), the groups
  /// backward with each lazy request whose link goes on to an eager one
  /// moving with that one, which keeps its forward time and bounds it in
  /// link order: steady while that bound binds, so the group of its waits
  /// is held to move no faster than the eager request's (a branch, the
  /// waits' group deciding). None when that leaves a link due by several.
  std::shared_ptr<const LoopGroups> linkBound;
  llvm::SmallVector<unsigned> ofArg, ofArgBackward;
  llvm::DenseMap<mlir::Operation *, unsigned> ofOp;
  llvm::DenseMap<mlir::Operation *, unsigned> ofWaitBackward;
  bool mixedWaits = false;
  /// The groups of the drives, and of the readouts, in the body, which take
  /// lanes under resources.
  llvm::SmallBitVector driveGroups, readoutGroups;
  llvm::SmallVector<mlir::Value> outside;
  llvm::SmallVector<std::pair<unsigned, unsigned>> branches;
  /// Of `linkBound`'s branches, each lazy request and the eager one after
  /// it on the link: forward right after it, the eager one bounds it
  /// backward too (its answer's due time is never before its forward
  /// start), so without resources the pair holds however its groups move.
  struct LinkPair {
    std::pair<unsigned, unsigned> branch;
    mlir::Operation *lazy, *eager;
  };
  llvm::SmallVector<LinkPair> linkPairs;
  /// Of `branches`, those some op relates other than through answers (a
  /// branch on a measurement), and the lazy requests whose answers relate
  /// each other one.
  llvm::SmallVector<std::pair<unsigned, unsigned>> strictBranches;
  llvm::SmallVector<std::pair<std::pair<unsigned, unsigned>,
                              llvm::SmallVector<mlir::Operation *, 2>>>
      lazyBranches;

  /// The group of a timed op in the body, the way the pass runs.
  unsigned groupOf(mlir::Operation *op, bool backward) const {
    if (backward)
      if (auto it = ofWaitBackward.find(op); it != ofWaitBackward.end())
        return it->second;
    return ofOp.lookup(op);
  }
  /// The group of each carried value, the way the pass runs.
  llvm::ArrayRef<unsigned> argGroups(bool backward) const {
    return backward ? ofArgBackward : ofArg;
  }
};

/// The waits of calls whose answers `value` is computed from.
static void conditionWaits(mlir::Value value,
                           llvm::SmallVectorImpl<mlir::Operation *> &found,
                           llvm::DenseSet<mlir::Value> &seen) {
  if (!seen.insert(value).second)
    return;
  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return;
  if (mlir::isa<classical::WaitOp>(def)) {
    found.push_back(def);
    return;
  }
  for (mlir::Value operand : def->getOperands())
    conditionWaits(operand, found, seen);
}

/// Whether `value` is computed from a measurement other than through the
/// answer of a call.
static bool readsMeasurement(mlir::Value value,
                             llvm::DenseSet<mlir::Value> &seen) {
  if (!seen.insert(value).second)
    return false;
  if (mlir::isa<pulse::MeasurementType>(value.getType()))
    return true;
  mlir::Operation *def = value.getDefiningOp();
  if (!def || mlir::isa<classical::WaitOp>(def))
    return false;
  return llvm::any_of(def->getOperands(), [&](mlir::Value operand) {
    return readsMeasurement(operand, seen);
  });
}

/// The groups of `loop`'s carried values, or none when their independence
/// cannot be shown: a carried call value other than a link, or a carried
/// measurement the body reads; a call op that keeps state the values it
/// reads do not carry; a timed value other than a measurement from outside
/// the loop read in it (other than through what it carries); a schedule
/// block or another region op; a timed op that reads no carried value; or a
/// line value that does not reach the loop's yield through timed ops and
/// syncs. In a body with a loop of its own, a branch joins the groups it
/// relates.
static std::shared_ptr<const LoopGroups> loopGroups(mlir::scf::ForOp loop) {
  unsigned n = loop.getNumRegionIterArgs();
  for (mlir::Value arg : loop.getRegionIterArgs()) {
    mlir::Type type = arg.getType();
    if (mlir::isa<pulse::MeasurementType>(type)
            ? !arg.use_empty()
            : isTimedType(type) && !isLineType(type) &&
                  !mlir::isa<pulse::ToneType, classical::LinkType>(type))
      return nullptr;
  }
  // A call's state off its values: a barrier, a wait of a run-time length,
  // or a link order of an instance's own. And a wait in a loop of the body
  // whose answers nothing reads: only its own time marks where it ends.
  if (loop.getBody()
          ->walk([&](mlir::Operation *op) {
            bool unread = mlir::isa<classical::WaitOp>(op) &&
                          op->getParentOfType<mlir::scf::ForOp>() != loop &&
                          op->use_empty();
            return unread || op->hasAttr("classical.blocking") ||
                           op->hasAttr("classical.variable") ||
                           op->hasAttr(classical::kInstanceAttr)
                       ? mlir::WalkResult::interrupt()
                       : mlir::WalkResult::advance();
          })
          .wasInterrupted())
    return nullptr;
  llvm::SmallVector<mlir::Value> outside;
  llvm::DenseMap<mlir::Value, llvm::SmallBitVector> roots;
  for (auto [i, arg] : llvm::enumerate(loop.getRegionIterArgs())) {
    llvm::SmallBitVector bits(n);
    bits.set(static_cast<unsigned>(i));
    roots[arg] = std::move(bits);
  }
  auto rootsOf = [&](mlir::Value value) {
    auto found = roots.find(value);
    return found == roots.end() ? llvm::SmallBitVector(n) : found->second;
  };
  auto inside = [&](mlir::Value value) {
    return loop->isAncestor(value.getParentRegion()->getParentOp());
  };
  auto reachesYield = [](mlir::Value line) {
    if (line.use_empty())
      return false;
    return llvm::all_of(line.getUsers(), [](mlir::Operation *user) {
      return mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp,
                       pulse::SyncOp, mlir::scf::YieldOp, mlir::scf::ForOp>(
          user);
    });
  };
  // The ops that relate the values they read: timed ops and syncs.
  llvm::SmallVector<std::pair<mlir::Operation *, llvm::SmallBitVector>> related;
  // What each branch's condition and what it decides read, whether the
  // one holds back the other only through answers, and the lazy requests
  // among those.
  struct Decides {
    llvm::SmallBitVector condition, decided;
    bool answers;
    llvm::SmallVector<mlir::Operation *, 2> lazy;
  };
  llvm::SmallVector<Decides> decides;
  auto eagerAnswer = [](mlir::Operation *wait) {
    auto submit =
        mlir::cast<classical::WaitOp>(wait).getFuture().getDefiningOp();
    return submit && submit->hasAttr("classical.eager");
  };
  auto lazyOf = [&](llvm::ArrayRef<mlir::Operation *> waits) {
    llvm::SmallVector<mlir::Operation *, 2> lazy;
    for (mlir::Operation *wait : waits)
      if (!eagerAnswer(wait))
        if (mlir::Operation *submit =
                mlir::cast<classical::WaitOp>(wait).getFuture().getDefiningOp())
          lazy.push_back(submit);
    return lazy;
  };
  // What the answers of each wait decide.
  llvm::DenseMap<mlir::Operation *, llvm::SmallBitVector> waitDecides;
  bool directed = true;
  loop.getBody()->walk([&](mlir::scf::ForOp) { directed = false; });
  // In a body without loops of its own an answer does not join what it is
  // a value of with its call: it holds it back, as a branch's condition
  // does what it decides. The waits each value is computed from, and what
  // each wait's call reads.
  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Operation *, 1>> answers;
  llvm::DenseMap<mlir::Operation *, llvm::SmallBitVector> callOf;
  // What `value` reads, its answers' calls included.
  auto readsOf = [&](mlir::Value value) {
    llvm::SmallBitVector bits = rootsOf(value);
    bits.resize(n);
    if (auto it = answers.find(value); it != answers.end())
      for (mlir::Operation *wait : it->second)
        bits |= callOf.lookup(wait);
    return bits;
  };
  bool ok = true;
  std::function<void(mlir::Block &, const llvm::SmallBitVector &)> visit =
      [&](mlir::Block &block, const llvm::SmallBitVector &ambient) {
        for (mlir::Operation &op : block.without_terminator()) {
          for (mlir::Value operand : op.getOperands())
            if (isTimedType(operand.getType()) && !inside(operand)) {
              if (mlir::isa<pulse::MeasurementType>(operand.getType()))
                outside.push_back(operand);
              else
                ok = false;
            }
          for (mlir::Value result : op.getResults())
            if (isLineType(result.getType()) && !reachesYield(result))
              ok = false;
          if (!ok)
            return;
          if (auto inner = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
            llvm::SmallVector<llvm::SmallBitVector> state;
            for (mlir::Value init : inner.getInitArgs())
              state.push_back(readsOf(init) | ambient);
            auto yield = inner.getBody()->getTerminator();
            size_t mark = related.size();
            // Each round adds roots, of which there are `n`.
            for (bool changed = true; changed;) {
              related.resize(mark, {nullptr, llvm::SmallBitVector()});
              for (auto [arg, bits] :
                   llvm::zip(inner.getRegionIterArgs(), state))
                roots[arg] = bits;
              visit(*inner.getBody(), ambient);
              if (!ok)
                return;
              changed = false;
              for (auto [bits, value] :
                   llvm::zip(state, yield->getOperands())) {
                llvm::SmallBitVector next = bits | rootsOf(value);
                if (next != bits) {
                  bits = std::move(next);
                  changed = true;
                }
              }
            }
            for (auto [result, bits] : llvm::zip(inner.getResults(), state))
              roots[result] = bits;
            continue;
          }
          if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
            for (mlir::Value measurement :
                 conditionMeasurements(branch.getCondition()))
              if (!inside(measurement))
                outside.push_back(measurement);
            llvm::SmallBitVector condition =
                readsOf(branch.getCondition()) | ambient;
            size_t mark = related.size();
            visit(*branch.thenBlock(), condition);
            if (!branch.getElseRegion().empty())
              visit(*branch.elseBlock(), condition);
            if (!ok)
              return;
            llvm::SmallBitVector decided(n);
            for (auto [index, result] : llvm::enumerate(branch.getResults())) {
              llvm::SmallBitVector bits =
                  readsOf(branch.thenYield().getOperand(index));
              if (!branch.getElseRegion().empty())
                bits |= readsOf(branch.elseYield().getOperand(index));
              if (!directed)
                bits |= condition;
              decided |= bits;
              roots[result] = std::move(bits);
            }
            if (directed) {
              for (size_t i = mark; i < related.size(); ++i)
                decided |= related[i].second;
              llvm::SmallVector<mlir::Operation *> waits;
              llvm::DenseSet<mlir::Value> seen;
              conditionWaits(branch.getCondition(), waits, seen);
              for (mlir::Operation *wait : waits) {
                auto [it, inserted] = waitDecides.try_emplace(wait, n);
                it->second |= decided;
              }
              llvm::DenseSet<mlir::Value> read;
              bool answers = ambient.none() && !waits.empty() &&
                             !readsMeasurement(branch.getCondition(), read);
              decides.push_back(
                  {condition, std::move(decided), answers, lazyOf(waits)});
            }
            continue;
          }
          if (op.getNumRegions()) {
            ok = false;
            return;
          }
          llvm::SmallBitVector bits =
              directed ? llvm::SmallBitVector(n) : ambient;
          llvm::SmallVector<mlir::Operation *, 1> from;
          for (mlir::Value operand : op.getOperands()) {
            bits |= directed ? rootsOf(operand) : readsOf(operand);
            if (auto it = answers.find(operand);
                directed && it != answers.end())
              for (mlir::Operation *wait : it->second)
                if (!llvm::is_contained(from, wait))
                  from.push_back(wait);
          }
          if (directed && mlir::isa<classical::WaitOp>(op)) {
            callOf[&op] = bits;
            for (mlir::Value result : op.getResults())
              answers[result] = {&op};
          } else {
            // A line or tone an answer holds back moves with what it is.
            for (mlir::Value result : op.getResults()) {
              roots[result] = bits;
              if (!from.empty() && !isTimedType(result.getType()))
                answers[result] = from;
            }
          }
          if (bits.any())
            for (mlir::Operation *wait : from) {
              decides.push_back(
                  {callOf.lookup(wait), bits, true, lazyOf(wait)});
              auto [it, inserted] = waitDecides.try_emplace(wait, n);
              it->second |= bits;
            }
          if (isTimedOp(&op) || mlir::isa<pulse::SyncOp>(op))
            related.emplace_back(&op, std::move(bits));
        }
      };
  visit(*loop.getBody(), llvm::SmallBitVector(n));
  if (!ok)
    return nullptr;

  llvm::SmallVector<unsigned> parent(n);
  std::iota(parent.begin(), parent.end(), 0u);
  auto find = [&](unsigned i) {
    while (parent[i] != i)
      i = parent[i] = parent[parent[i]];
    return i;
  };
  auto join = [&](const llvm::SmallBitVector &bits) {
    int first = bits.find_first();
    for (unsigned bit : bits.set_bits())
      parent[find(bit)] = find(static_cast<unsigned>(first));
  };
  for (auto &[op, bits] : related) {
    if (bits.none())
      return nullptr;
    join(bits);
  }
  auto yield = loop.getBody()->getTerminator();
  for (auto [i, value] : llvm::enumerate(yield->getOperands())) {
    llvm::SmallBitVector bits = readsOf(value);
    bits.set(static_cast<unsigned>(i));
    join(bits);
  }
  auto groups = std::make_shared<LoopGroups>();
  llvm::DenseMap<unsigned, unsigned> index;
  for (unsigned i = 0; i < n; ++i) {
    auto [it, inserted] = index.try_emplace(find(i), groups->count);
    if (inserted)
      ++groups->count;
    groups->ofArg.push_back(it->second);
  }
  for (auto &[op, bits] : related)
    if (isTimedOp(op))
      groups->ofOp[op] =
          index.lookup(find(static_cast<unsigned>(bits.find_first())));
  for (auto &[condition, decided, answers, lazy] : decides)
    for (unsigned c : condition.set_bits())
      for (unsigned d : decided.set_bits()) {
        std::pair<unsigned, unsigned> branch{index.lookup(find(c)),
                                             index.lookup(find(d))};
        if (branch.first == branch.second)
          continue;
        if (!llvm::is_contained(groups->branches, branch))
          groups->branches.push_back(branch);
        if (!answers && !llvm::is_contained(groups->strictBranches, branch))
          groups->strictBranches.push_back(branch);
        auto it = llvm::find_if(groups->lazyBranches, [&](const auto &entry) {
          return entry.first == branch;
        });
        if (it == groups->lazyBranches.end())
          it = groups->lazyBranches.insert(it, {branch, {}});
        for (mlir::Operation *submit : lazy)
          if (!llvm::is_contained(it->second, submit))
            it->second.push_back(submit);
      }
  groups->driveGroups.resize(groups->count);
  groups->readoutGroups.resize(groups->count);
  for (auto [op, group] : groups->ofOp) {
    if (mlir::isa<pulse::DriveOp>(op))
      groups->driveGroups.set(group);
    else if (mlir::isa<pulse::ReadoutOp>(op))
      groups->readoutGroups.set(group);
  }
  for (auto &[wait, decided] : waitDecides) {
    std::optional<unsigned> only;
    for (unsigned d : decided.set_bits()) {
      unsigned group = index.lookup(find(d));
      if (only && *only != group)
        groups->mixedWaits = true;
      only = group;
    }
    if (only)
      groups->ofWaitBackward[wait] = *only;
  }
  bool waitsMixed = groups->mixedWaits;
  auto variant = std::make_shared<LoopGroups>(*groups);
  // Which groups each carried link and lazy request are due by backward;
  // `bound`, each lazy request going on to an eager one with that one.
  auto assignDue = [&](LoopGroups &into, bool bound) {
    into.ofArgBackward = into.ofArg;
    llvm::SmallVector<std::optional<unsigned>> dueBy(n);
    loop.getBody()->walk([&](classical::SubmitOp submit) {
      mlir::Operation *op = submit;
      std::optional<unsigned> group;
      if (!op->hasAttr("classical.eager"))
        for (mlir::Operation *user : submit.getFuture().getUsers())
          if (auto it = into.ofWaitBackward.find(user);
              it != into.ofWaitBackward.end()) {
            if (group && *group != it->second)
              into.mixedWaits = true;
            group = it->second;
          }
      if (bound && group && submit.getUpdatedLink().hasOneUse())
        if (auto next = mlir::dyn_cast<classical::SubmitOp>(
                *submit.getUpdatedLink().getUsers().begin());
            next && next->getBlock() == op->getBlock() &&
            next->hasAttr("classical.eager")) {
          unsigned eager = into.ofOp.lookup(next);
          std::pair<unsigned, unsigned> held{eager, *group};
          if (eager != *group) {
            if (!llvm::is_contained(into.branches, held))
              into.branches.push_back(held);
            into.linkPairs.push_back({held, op, next});
            auto it = llvm::find_if(into.lazyBranches, [&](const auto &entry) {
              return entry.first == held;
            });
            if (it == into.lazyBranches.end())
              it = into.lazyBranches.insert(it, {held, {}});
            if (!llvm::is_contained(it->second, op))
              it->second.push_back(op);
          }
          group = eager;
        }
      if (group)
        into.ofWaitBackward[op] = *group;
      unsigned own = group.value_or(into.ofOp.lookup(op));
      llvm::SmallBitVector roots = rootsOf(submit.getLink());
      for (unsigned i : roots.set_bits()) {
        if (!mlir::isa<classical::LinkType>(
                loop.getRegionIterArgs()[i].getType()))
          continue;
        if (dueBy[i] && *dueBy[i] != own)
          into.mixedWaits = true;
        dueBy[i] = own;
      }
    });
    for (auto [i, group] : llvm::enumerate(dueBy))
      if (group)
        into.ofArgBackward[i] = *group;
  };
  assignDue(*groups, /*bound=*/false);
  if (groups->mixedWaits && !waitsMixed) {
    assignDue(*variant, /*bound=*/true);
    if (!variant->mixedWaits) {
      variant->outside = outside;
      groups->linkBound = std::move(variant);
    }
  }
  groups->outside = std::move(outside);
  return groups;
}

/// Where a Scheduler without grids would trace lines: nowhere.
struct NoLineTrace {};

/// The most iterations after which a loop's iterations may repeat for the
/// scheduler to run that many in each (see Scheduler::regroup).
constexpr int64_t kMostRegrouped = 8;

/// The most iterations of a loop as written regrouping runs in each, over
/// every time it regroups the loop.
constexpr int64_t kMostRegroupedInAll = 16;

/// The carried times after each of a loop's last iterations simulated, up
/// to the most two runs of kMostRegrouped iterations compare.
class RecentStates {
public:
  void push(llvm::ArrayRef<int64_t> state) {
    if (states.size() < kSize)
      states.emplace_back(state.begin(), state.end());
    else
      states[next].assign(state.begin(), state.end());
    next = (next + 1) % kSize;
    ++pushed;
  }

  /// The state pushed `back` pushes before the last one.
  llvm::ArrayRef<int64_t> before(int64_t back) const {
    return states[(next + kSize - 1 - static_cast<size_t>(back)) % kSize];
  }

  int64_t pushed = 0;

private:
  static constexpr size_t kSize = 2 * kMostRegrouped + 1;
  llvm::SmallVector<llvm::SmallVector<int64_t, 8>, 0> states;
  size_t next = 0;
};

/// Schedules one region: a function body, or a schedule block's body with
/// every block argument available at time 0. `Gridded` when some line is on
/// a grid coarser than one program cycle (see LineGrid) or, under rcp, lines
/// are on several controllers (see Controllers): both trace lines to their
/// allocations. Without, a schedule does no grid or controller work at all.
template <bool Gridded>
class Scheduler {
public:
  Scheduler(mlir::Region &region, Resources resources, bool classical,
            const pulse::LineGrid &grid, const pulse::Controllers &controllers,
            const Timelines *timelines = nullptr)
      : region(&region), resources(resources), classical(classical),
        lanesPer(timelines && timelines->count > 1 ? timelines : nullptr),
        grid(grid),
        controllers(resources.enabled ? controllers : pulse::Controllers()) {
    auto module = region.getParentOp()->getParentOfType<mlir::ModuleOp>();
    if (mlir::Attribute held =
            module ? module->getAttr(classical::kBlocksOnCallsAttr) : nullptr)
      holdControllers(held, controllers);
    auto count = static_cast<size_t>(isolated() ? lanesPer->count : 1);
    floors.assign(count, 0);
    ends.assign(count, 0);
    iterationEnds.assign(count, -1);
    ceilings.assign(count, kNever);
    if (this->several())
      armStarts.assign(static_cast<size_t>(this->controllers.count()), kNever);
  }

  mlir::LogicalResult forward() {
    resetLanes(0);
    if (mlir::failed(forwardBlock(region->front())))
      return mlir::failure();
    return mlir::success();
  }

  mlir::LogicalResult backward() {
    if (classical) {
      // Eager submissions and unused answers keep their forward times.
      for (auto &[op, list] : samples)
        if (mlir::isa<classical::SubmitOp, classical::WaitOp>(op) ||
            op->hasAttr("classical.variable"))
          for (const Sample &sample : list)
            forwardStarts[{op, sample.iteration}] = sample.start;
      forwardExtrapolations = extrapolations;
    }
    samples.clear();
    extrapolations.clear();
    resetLanes(makespan);
    return backwardBlock(region->front());
  }

  int64_t makespan = 0;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<Sample, 1>> samples;
  llvm::SmallVector<Extrapolation> extrapolations;
  /// An iteration the backward pass did not simulate starts an op before
  /// zero.
  bool skippedNegative = false;
  /// The backward pass takes an answer before it is due.
  bool answeredEarly = false;
  /// The backward pass keeps a request at its forward start where its link
  /// has no slot free for it, as slots handed out in reverse program order
  /// tell: requests holding theirs for different times may still fit
  /// (heldTooMany).
  bool slotsOverrun = false;

  /// Whether the requests the pass placed, as sampled, hold more of a
  /// link's slots at once than it has: those of the iterations it did not
  /// simulate held more.
  bool heldTooMany() {
    // Per link, its slots and when each request takes (+1) and frees (-1)
    // one.
    std::map<LinkKey,
             std::pair<int64_t, llvm::SmallVector<std::pair<int64_t, int>>>>
        spans;
    auto hold = [&](mlir::Operation *submit, int64_t start, int64_t busy) {
      auto &[slots, found] = spans[linkOf(submit)];
      slots = classical::intAttr(submit, "classical.slots", 64);
      found.push_back({start, 1});
      found.push_back({start + busy, -1});
    };
    forEachScheduled(*region, [&](mlir::Operation *op) {
      auto it = samples.find(op);
      if (it == samples.end())
        return;
      auto block = mlir::dyn_cast<pulse::ScheduleBlockOp>(op);
      if (!block && !mlir::isa<classical::SubmitOp>(op))
        return;
      for (const Sample &sample : it->second) {
        if (!block) {
          hold(op, sample.start, durationOf(op) + budgetOf(op));
          continue;
        }
        for (const BlockRequest &request : requestsOf(block, false))
          hold(request.op, sample.start + request.start, request.busy);
      }
    });
    for (auto &[key, entry] : spans) {
      auto &[slots, found] = entry;
      // A slot freed at a time is free for a request starting then.
      llvm::sort(found);
      int64_t held = 0;
      for (auto [at, step] : found) {
        held += step;
        if (held > slots)
          return true;
      }
    }
    return false;
  }

  /// Whether every call holds every line from its submission until its
  /// answer is due (`pulse.blocks_on_calls`): lines are done before it, and
  /// none starts again before then.
  bool blocksOnCalls = false;

  /// On several controllers, some of which block on calls: per controller,
  /// whether it does, when its lines end and may start again (forward), and
  /// when they must be done by (backward). Empty otherwise.
  llvm::SmallVector<bool, 4> blocking;
  llvm::SmallVector<int64_t, 4> heldEnds, heldFloors, heldCeilings;

  /// Whether a loop whose iterations repeat only every few of them stops
  /// the pass, to be run that many iterations at a time (`regroup`).
  bool mayRegroup = true;
  /// Such a loop and how many iterations it repeats after, found when it
  /// had more than two runs of them left: the pass stopped there, so its
  /// schedule is not complete.
  struct Regroup {
    mlir::scf::ForOp loop;
    int64_t iterations;
  };
  std::optional<Regroup> regroup;
  /// How many iterations as written each loop regrouped before runs in each.
  llvm::DenseMap<mlir::Operation *, int64_t> regroupedBy;
  /// Whether the backward pass, finding a loop's first iterations kept
  /// forward starts that do not repeat, stops to have them peeled
  /// (`headPeel`) rather than simulating every iteration.
  bool mayPeelHead = true;
  /// Such a loop and how many of its first iterations to peel.
  std::optional<Regroup> headPeel;

private:
  /// The earliest start of each group's ops in the iteration being
  /// simulated, for a loop whose groups move by periods of their own.
  struct Track {
    const LoopGroups *groups = nullptr;
    llvm::SmallVector<int64_t, 4> earliest;
    /// Without groups yet, every start recorded, for them once built.
    llvm::SmallVector<std::pair<mlir::Operation *, int64_t>, 8> starts;
  };

  /// The carried times before and after an iteration whose values moved by
  /// different shifts: only two such in a row can repeat group by group, so
  /// a loop's groups wait for the second.
  struct Unshifted {
    llvm::SmallVector<int64_t> before, after;
  };

  // ── Shared state ──────────────────────────────────────────────────────

  void resetLanes(int64_t value) {
    linkLanes.clear();
    laneReset = value;
    if (!resources.enabled)
      return;
    // Timelines that wait independently each use lanes of their own: as
    // many as they have lines, which the hardware has.
    size_t count = floors.size();
    driveLanes.assign(count, {});
    readoutLanes.assign(count, {});
    driveUses.assign(count, 0);
    readoutUses.assign(count, 0);
    for (size_t t = 0; t < count; ++t) {
      int64_t drives = isolated() ? lanesPer->drives[t] : resources.maxDrives;
      int64_t readouts =
          isolated() ? lanesPer->readouts[t] : resources.maxReadouts;
      driveLanes[t].assign(static_cast<size_t>(std::max<int64_t>(1, drives)),
                           value);
      readoutLanes[t].assign(
          static_cast<size_t>(std::max<int64_t>(1, readouts)), value);
    }
    if (several()) {
      driveOwners = driveLanes;
      readoutOwners = readoutLanes;
      for (Lanes *owners : {&driveOwners, &readoutOwners})
        for (auto &list : *owners)
          std::fill(list.begin(), list.end(), kNoOwner);
    }
  }

  // ── Controllers ───────────────────────────────────────────────────────

  /// A lane's owner before any op takes it, and one that may be any
  /// controller.
  static constexpr int64_t kNoOwner = -2, kAnyOwner = -1;

  /// Whether the lines are on several controllers: never without line
  /// tracing.
  bool several() const {
    if constexpr (Gridded)
      return controllers.enabled();
    else
      return false;
  }

  /// The controller that plays the line `line` is a version of, or -1.
  int64_t controllerOfLine(mlir::Value line) {
    if constexpr (Gridded)
      return controllers.of(line, lines);
    else
      return kAnyOwner;
  }

  /// The controller that plays the line `op` is on, or -1.
  int64_t controllerOf(mlir::Operation *op) {
    return controllerOfLine(op->getOperand(0));
  }

  /// The controller that read `measurement`, or -1.
  int64_t readerOf(mlir::Value measurement) {
    auto readout = measurement.getDefiningOp<pulse::ReadoutOp>();
    return readout ? controllerOf(readout) : -1;
  }

  /// In a branch on readouts, no op of `controller` (-1: any) starts before
  /// their outcomes reach it.
  int64_t armFloorOf(int64_t controller) const {
    if (armFloors.empty())
      return 0;
    return controller < 0 ? *llvm::max_element(armFloors)
                          : armFloors[controller];
  }

  /// How long after a lane's op of `owner` an op of `controller` may start
  /// on it: their starts may be that far apart.
  int64_t laneGap(int64_t owner, int64_t controller) const {
    if (owner == kNoOwner || (owner >= 0 && owner == controller))
      return 0;
    return controllers.spread(owner, controller);
  }

  // ── Calls that hold controllers ───────────────────────────────────────

  /// From `pulse.blocks_on_calls`: every call holds every line, or, on
  /// several controllers whose lines are traced, the lines of those it
  /// lists that the call involves (`heldBy`).
  void holdControllers(mlir::Attribute held, const pulse::Controllers &all) {
    auto listed = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(held);
    if (!Gridded || !listed || !all.enabled()) {
      blocksOnCalls = true;
      return;
    }
    owners = all;
    auto count = static_cast<size_t>(all.count());
    blocking.assign(count, false);
    for (int64_t c : listed.asArrayRef())
      if (c >= 0 && c < all.count())
        blocking[c] = true;
    heldEnds.assign(count, 0);
    heldFloors.assign(count, 0);
    heldCeilings.assign(count, kNever);
  }

  bool holding() const { return !blocking.empty(); }

  /// The controller of the line timed op `op` is on, or -1, whatever the
  /// policy.
  int64_t heldOwnerOf(mlir::Operation *op) {
    if constexpr (Gridded)
      return owners.of(op->getOperand(0), lines);
    else
      return kAnyOwner;
  }

  /// No op of a held controller starts before the answers it waits for
  /// are due; a line of no known controller waits for every one.
  int64_t heldFloorOf(mlir::Operation *op) {
    int64_t c = heldOwnerOf(op);
    return c >= 0 ? heldFloors[c] : *llvm::max_element(heldFloors);
  }

  int64_t heldCeilingOf(mlir::Operation *op) {
    int64_t c = heldOwnerOf(op);
    return c >= 0 ? heldCeilings[c] : *llvm::min_element(heldCeilings);
  }

  /// Move the times an iteration changed by `distance`, for the iterations
  /// not simulated.
  static void moveChanged(llvm::MutableArrayRef<int64_t> times,
                          llvm::ArrayRef<int64_t> before, int64_t distance) {
    for (auto [now, was] : llvm::zip(times, before))
      if (now != was)
        now += distance;
  }

  void noteHeldEnd(mlir::Operation *op, int64_t end) {
    int64_t c = heldOwnerOf(op);
    if (c >= 0) {
      heldEnds[c] = std::max(heldEnds[c], end);
      return;
    }
    for (int64_t &at : heldEnds)
      at = std::max(at, end);
  }

  /// The controllers that block on calls which `submit` holds: the one
  /// that issues it (whose instance of a per-controller link runs it, else
  /// those that read the measurements it sends, else any), and those that
  /// play an op that uses its answer, computes with it or branches on it. A
  /// line of no known controller, or an answer used in a way not followed
  /// here, holds them all.
  llvm::ArrayRef<int64_t> heldBy(classical::SubmitOp submit) {
    auto [it, inserted] = heldCalls.try_emplace(submit);
    if (!inserted)
      return it->second;
    llvm::SmallVector<bool, 4> held(blocking.size(), false);
    auto hold = [&](int64_t c) {
      if (c >= 0)
        held[c] = true;
      else
        std::fill(held.begin(), held.end(), true);
    };
    auto played = [](mlir::Operation *op) {
      return mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp>(op);
    };
    int64_t instance = classical::intAttr(submit, classical::kInstanceAttr, -1);
    llvm::SmallVector<mlir::Value> sent = valueMeasurements(submit);
    if (instance >= 0 && instance < static_cast<int64_t>(held.size()))
      held[instance] = true;
    else if (sent.empty())
      hold(kAnyOwner);
    for (mlir::Value measurement : sent) {
      auto readout = measurement.getDefiningOp<pulse::ReadoutOp>();
      hold(readout ? heldOwnerOf(readout) : kAnyOwner);
    }
    llvm::SmallVector<mlir::Value> answers;
    for (mlir::Operation *user : submit.getFuture().getUsers()) {
      if (auto wait = mlir::dyn_cast<classical::WaitOp>(user))
        llvm::append_range(answers, wait.getValues());
      else
        hold(kAnyOwner);
    }
    llvm::DenseSet<mlir::Value> seen;
    while (!answers.empty()) {
      mlir::Value value = answers.pop_back_val();
      if (!seen.insert(value).second)
        continue;
      for (mlir::OpOperand &use : value.getUses()) {
        mlir::Operation *user = use.getOwner();
        unsigned index = use.getOperandNumber();
        if (played(user)) {
          hold(heldOwnerOf(user));
        } else if (mlir::isa<classical::SubmitOp>(user)) {
          // An answer sent on to another call is that call's.
        } else if (mlir::isa<mlir::scf::YieldOp>(user)) {
          mlir::Operation *parent = user->getParentOp();
          if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(parent)) {
            answers.push_back(loop.getRegionIterArgs()[index]);
            answers.push_back(loop->getResult(index));
          } else if (mlir::isa<mlir::scf::IfOp>(parent)) {
            answers.push_back(parent->getResult(index));
          } else {
            hold(kAnyOwner);
          }
        } else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(user);
                   loop && index >= loop.getNumControlOperands()) {
          index -= loop.getNumControlOperands();
          answers.push_back(loop.getRegionIterArgs()[index]);
          answers.push_back(loop->getResult(index));
        } else {
          // A branch on it, a loop over it or a block given it holds what
          // it plays.
          if (user->getNumRegions())
            user->walk([&](mlir::Operation *inner) {
              if (played(inner))
                hold(heldOwnerOf(inner));
            });
          llvm::append_range(answers, user->getResults());
        }
      }
    }
    for (auto [c, is] : llvm::enumerate(held))
      if (is && blocking[c])
        it->second.push_back(static_cast<int64_t>(c));
    return it->second;
  }

  /// The measurements a value is computed from, not through other timed
  /// values or answers.
  static void valueMeasurements(mlir::Value value,
                                llvm::SmallVectorImpl<mlir::Value> &found,
                                llvm::DenseSet<mlir::Value> &seen) {
    if (!seen.insert(value).second)
      return;
    if (mlir::isa<pulse::MeasurementType>(value.getType())) {
      found.push_back(value);
      return;
    }
    if (isTimedType(value.getType()))
      return;
    mlir::Operation *def = value.getDefiningOp();
    if (!def || mlir::isa<classical::WaitOp>(def))
      return;
    if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(def)) {
      valueMeasurements(branch.getCondition(), found, seen);
      auto index = mlir::cast<mlir::OpResult>(value).getResultNumber();
      valueMeasurements(branch.thenYield().getOperand(index), found, seen);
      if (!branch.getElseRegion().empty())
        valueMeasurements(branch.elseYield().getOperand(index), found, seen);
      return;
    }
    if (def->getNumRegions())
      return;
    for (mlir::Value operand : def->getOperands())
      valueMeasurements(operand, found, seen);
  }

  /// The measurements the operands of timed op `op` other than its line and
  /// tone are computed from.
  static llvm::SmallVector<mlir::Value> valueMeasurements(mlir::Operation *op) {
    llvm::SmallVector<mlir::Value> found;
    llvm::DenseSet<mlir::Value> seen;
    mlir::Value tone = toneOperand(op);
    for (mlir::Value operand : op->getOperands().drop_front())
      if (operand != tone)
        valueMeasurements(operand, found, seen);
    return found;
  }

  // ── Timelines ─────────────────────────────────────────────────────────

  bool isolated() const { return lanesPer != nullptr; }

  /// The timeline of a call, block, line or tone allocation, or of the
  /// first line or tone `op` plays on.
  int64_t timelineOf(mlir::Operation *op) {
    if (!isolated())
      return 0;
    auto [it, inserted] = timelineCache.try_emplace(op, 0);
    if (!inserted)
      return it->second;
    if (op->hasAttr(classical::kTimelineAttr)) {
      it->second = classical::intAttr(op, classical::kTimelineAttr);
    } else {
      for (mlir::Value operand : op->getOperands())
        if (mlir::isa<pulse::DriveLineType, pulse::ReadoutLineType,
                      pulse::ToneType>(operand.getType())) {
          it->second = timelineOfValue(operand);
          break;
        }
    }
    return it->second;
  }

  int64_t timelineOfValue(mlir::Value value) {
    if (!isolated())
      return 0;
    mlir::Operation *allocation = trace(value);
    return allocation ? classical::intAttr(allocation, classical::kTimelineAttr)
                      : 0;
  }

  /// No op of timeline `t` starts before this: its last run-time wait, and
  /// the condition of the branch the op is in.
  int64_t floorOf(int64_t t) const { return std::max(floor, floors[t]); }

  /// Where a timeline's lines end if nothing later uses them.
  int64_t endOf(int64_t t) const { return isolated() ? ends[t] : makespan; }

  /// A link as one timeline sees it, on one controller for a link with an
  /// instance per controller: its share of the slots, and its own order of
  /// submissions.
  using LinkKey = std::tuple<int64_t, int64_t, int64_t>;
  LinkKey linkOf(mlir::Operation *submit) {
    return {classical::intAttr(submit, "classical.device"),
            classical::intAttr(submit, classical::kInstanceAttr, -1),
            timelineOf(submit)};
  }

  /// Whether a submission keeps an order of its own on its link
  /// (`linkReady`, `linkLate`) rather than the link value threaded through
  /// the program: with timelines, or on its controller's instance.
  bool ownOrder(mlir::Operation *submit) const {
    return isolated() || submit->hasAttr(classical::kInstanceAttr);
  }

  void record(mlir::Operation *op, int64_t start) {
    samples[op].push_back({iteration, start});
    // Only the backward pass tracks the earliest starts.
    for (Track *track : tracks)
      if (!track->groups)
        track->starts.push_back({op, start});
      else if (track->groups->ofOp.contains(op)) {
        int64_t &earliest =
            track->earliest[track->groups->groupOf(op, /*backward=*/true)];
        earliest = std::min(earliest, start);
      }
  }

  /// Lower the earliest start `arm` of a branch's arm to `simulated`, a
  /// start in the iteration last simulated, moved by `distance` to where it
  /// is in an iteration the backward pass extrapolated.
  void lowerArmStart(int64_t &arm, int64_t simulated, int64_t distance) {
    if (simulated == kNever)
      return;
    int64_t start;
    if (__builtin_add_overflow(simulated, distance, &start)) {
      skippedNegative = true;
      return;
    }
    if (start < 0)
      skippedNegative = true;
    arm = std::min(arm, start);
  }

  // ── Lanes over a loop's iterations ─────────────────────────────────────

  using Lanes = llvm::SmallVector<llvm::SmallVector<int64_t>, 1>;

  /// The lanes, and how many ops took each timeline's, before an iteration.
  struct LaneState {
    Lanes drives, readouts, driveOwners, readoutOwners;
    llvm::SmallVector<int64_t, 1> driveUses, readoutUses;
  };

  LaneState laneState() const {
    return {driveLanes,    readoutLanes, driveOwners,
            readoutOwners, driveUses,    readoutUses};
  }

  /// Whether lanes of equal times are interchangeable for the rest of the
  /// schedule: on one controller, with no branch whose arms both take lanes,
  /// as their lanes merge lane by lane. An arm that takes none leaves each
  /// lane where the other arm's is or before it.
  bool lanesByTime() {
    if (!lanesKnown) {
      lanesKnown = true;
      auto takesLanes = [](mlir::Region &arm) {
        return arm
            .walk([](mlir::Operation *op) {
              return isTimedOp(op) && !mlir::isa<pulse::WaitOp>(op)
                         ? mlir::WalkResult::interrupt()
                         : mlir::WalkResult::advance();
            })
            .wasInterrupted();
      };
      lanesInterchangeable =
          !several() && !region
                             ->walk([&](mlir::scf::IfOp branch) {
                               return takesLanes(branch.getThenRegion()) &&
                                              takesLanes(branch.getElseRegion())
                                          ? mlir::WalkResult::interrupt()
                                          : mlir::WalkResult::advance();
                             })
                             .wasInterrupted();
    }
    return lanesInterchangeable;
  }

  /// Whether an iteration moved the lanes by `shift`, so that the next one
  /// plays as it did moved by `shift`: on each timeline, the lanes of a kind
  /// an op took are those before moved by `shift`, as a multiset when lanes
  /// of equal times are interchangeable, else lane by lane with the same
  /// owners.
  bool lanesMoved(const LaneState &before, int64_t shift) {
    return lanesMovedBy(before, shift, shift);
  }

  /// lanesMoved, the drive lanes by `driveShift` and the readout lanes by
  /// `readoutShift`.
  bool lanesMovedBy(const LaneState &before, int64_t driveShift,
                    int64_t readoutShift) {
    bool byTime = lanesByTime();
    auto moved = [&](const Lanes &was, const Lanes &now, const Lanes &wasOwners,
                     const Lanes &nowOwners, llvm::ArrayRef<int64_t> wasUses,
                     llvm::ArrayRef<int64_t> nowUses, int64_t shift) {
      for (size_t t = 0; t < now.size(); ++t) {
        if (wasUses[t] == nowUses[t])
          continue;
        if (byTime) {
          llvm::SmallVector<int64_t, 4> a(was[t].begin(), was[t].end());
          llvm::SmallVector<int64_t, 4> b(now[t].begin(), now[t].end());
          llvm::sort(a);
          llvm::sort(b);
          for (auto [x, y] : llvm::zip(a, b))
            if (y != x + shift)
              return false;
          continue;
        }
        if (several() && wasOwners[t] != nowOwners[t])
          return false;
        for (auto [x, y] : llvm::zip(was[t], now[t]))
          if (y != x + shift)
            return false;
      }
      return true;
    };
    return moved(before.drives, driveLanes, before.driveOwners, driveOwners,
                 before.driveUses, driveUses, driveShift) &&
           moved(before.readouts, readoutLanes, before.readoutOwners,
                 readoutOwners, before.readoutUses, readoutUses, readoutShift);
  }

  /// How many iterations moving as the one before a loop's lanes are
  /// compared over, past which they are not: lanes that will move as the
  /// iterations do have once each was taken again (none without resources).
  int64_t laneTriesPerLoop() const {
    if (!resources.enabled)
      return 0;
    size_t most = 0;
    for (auto [drives, readouts] : llvm::zip(driveLanes, readoutLanes))
      most = std::max(most, drives.size() + readouts.size());
    return 2 * static_cast<int64_t>(most) + 2;
  }

  /// Add the lanes now to `history`, keeping the last kMostRegrouped.
  void pushLanes(llvm::SmallVectorImpl<LaneState> &history) const {
    if (static_cast<int64_t>(history.size()) == kMostRegrouped)
      history.erase(history.begin());
    history.push_back(laneState());
  }

  /// lanesMoved, counting the tries left down when not.
  bool lanesSteady(const LaneState &before, int64_t shift, int64_t &tries) {
    if (tries <= 0)
      return false;
    if (lanesMoved(before, shift))
      return true;
    --tries;
    return false;
  }

  /// Moves the lanes an iteration took by `distance`, as the iterations not
  /// simulated do.
  void moveLanes(const LaneState &before, int64_t distance) {
    moveLanesBy(before, distance, distance);
  }

  /// moveLanes, the drive lanes by `driveDistance` and the readout lanes by
  /// `readoutDistance`.
  void moveLanesBy(const LaneState &before, int64_t driveDistance,
                   int64_t readoutDistance) {
    for (size_t t = 0; t < driveLanes.size(); ++t) {
      if (driveUses[t] != before.driveUses[t])
        for (int64_t &lane : driveLanes[t])
          lane += driveDistance;
      if (readoutUses[t] != before.readoutUses[t])
        for (int64_t &lane : readoutLanes[t])
          lane += readoutDistance;
    }
  }

  // ── Loops whose lines move by periods of their own ────────────────────

  /// Whether a loop's carried values may move by a period per group (see
  /// LoopGroups): without timelines, whose state the groups share, or
  /// resources on several controllers.
  bool groupsApply() const {
    return !lanesBound && !isolated() && !(resources.enabled && several()) &&
           !holding();
  }

  const LoopGroups *groupsOf(mlir::scf::ForOp loop) {
    auto [it, inserted] = groupCache.try_emplace(loop, nullptr);
    if (inserted)
      it->second = loopGroups(loop);
    return it->second.get();
  }

  using GroupShifts = llvm::SmallVector<std::optional<int64_t>, 4>;

  /// slotsHoldSteadyState for a loop whose groups move by `shifts`, the
  /// way the pass runs (`sign`): each submission by its group's.
  bool slotsHoldGroups(mlir::scf::ForOp loop, const LoopGroups &groups,
                       const GroupShifts &shifts, int64_t sign) {
    if (!classical)
      return true;
    std::map<LinkKey, std::pair<int64_t, int64_t>> perLink;
    bool ok = true;
    loop.getBody()->walk([&](classical::SubmitOp submit) {
      mlir::Operation *op = submit;
      int64_t period =
          groups.ofOp.contains(op)
              ? sign * shifts[groups.groupOf(op, sign < 0)].value_or(0)
              : 0;
      std::optional<int64_t> held;
      if (period > 0)
        held = inFlight(op, loop, period);
      auto &[need, slots] = perLink[linkOf(op)];
      if (!held) {
        ok = false;
        return;
      }
      if (__builtin_add_overflow(need, *held, &need)) {
        ok = false;
        return;
      }
      slots = classical::intAttr(op, "classical.slots", 64);
    });
    for (auto &[device, entry] : perLink)
      ok = ok && entry.first <= entry.second;
    return ok;
  }

  /// The shift of the groups of a loop's submissions `ops`, the way the pass
  /// runs, each that of the op of the body it is in; none when they differ.
  static std::optional<int64_t>
  requestsShift(mlir::scf::ForOp loop, const LoopGroups &groups,
                const GroupShifts &shifts,
                llvm::ArrayRef<mlir::Operation *> ops, bool backward) {
    std::optional<int64_t> shift;
    for (mlir::Operation *op : ops) {
      mlir::Operation *timed = op;
      while (timed != loop && !groups.ofOp.contains(timed))
        timed = timed->getParentOp();
      if (timed == loop)
        return std::nullopt;
      int64_t moved = shifts[groups.groupOf(timed, backward)].value_or(0);
      if (shift && *shift != moved)
        return std::nullopt;
      shift = moved;
    }
    return shift;
  }

  /// Under resources, whether an iteration of a loop whose groups moved by
  /// `shifts` moved its lanes from `before` by as much: each kind's by the
  /// one shift of the groups its ops are in. True without resources.
  bool groupLanesMoved(const LaneState &before, const LoopGroups &groups,
                       const GroupShifts &shifts) {
    if (!resources.enabled)
      return true;
    auto shiftOf =
        [&](const llvm::SmallBitVector &kind) -> std::optional<int64_t> {
      std::optional<int64_t> shift;
      for (unsigned group : kind.set_bits()) {
        int64_t moved = shifts[group].value_or(0);
        if (shift && *shift != moved)
          return std::nullopt;
        shift = moved;
      }
      return shift.value_or(0);
    };
    std::optional<int64_t> drives = shiftOf(groups.driveGroups),
                           readouts = shiftOf(groups.readoutGroups);
    return drives && readouts && lanesMovedBy(before, *drives, *readouts);
  }

  /// Move the lanes as the iterations a loop's groups skip `distances`
  /// do (see groupLanesMoved).
  void moveGroupLanes(const LaneState &before, const LoopGroups &groups,
                      llvm::ArrayRef<int64_t> distances) {
    if (!resources.enabled)
      return;
    auto distanceOf = [&](const llvm::SmallBitVector &kind) -> int64_t {
      int first = kind.find_first();
      return first < 0 ? 0 : distances[static_cast<unsigned>(first)];
    };
    moveLanesBy(before, distanceOf(groups.driveGroups),
                distanceOf(groups.readoutGroups));
  }

  /// How far each group of carried values moved from `before` to `after`
  /// (none: nothing timed in it moved), or none when a group's values moved
  /// by different amounts; `backward`, by their groups backward.
  static std::optional<GroupShifts> groupShifts(const LoopGroups &groups,
                                                llvm::ArrayRef<int64_t> before,
                                                llvm::ArrayRef<int64_t> after,
                                                llvm::ArrayRef<Carried> kinds,
                                                bool backward) {
    GroupShifts shifts(groups.count);
    for (auto [group, b, a, kind] :
         llvm::zip(groups.argGroups(backward), before, after, kinds)) {
      if (kind == Carried::Untimed || (kind == Carried::Tone && a == b))
        continue;
      std::optional<int64_t> &shift = shifts[group];
      if (shift && *shift != a - b)
        return std::nullopt;
      shift = a - b;
    }
    return shifts;
  }

  /// Each group's shift times `copies`, or none past 2**63.
  static std::optional<llvm::SmallVector<int64_t, 4>>
  groupDistances(const GroupShifts &shifts, int64_t copies) {
    llvm::SmallVector<int64_t, 4> distances(shifts.size(), 0);
    for (auto [shift, distance] : llvm::zip(shifts, distances))
      if (shift)
        if (__builtin_mul_overflow(copies, *shift, &distance))
          return std::nullopt;
    return distances;
  }

  /// Advance each carried value that moves by its group's distance; none,
  /// leaving `carried` as it was, past 2**63.
  static bool extrapolateGroups(llvm::MutableArrayRef<int64_t> carried,
                                llvm::ArrayRef<int64_t> previous,
                                llvm::ArrayRef<Carried> kinds,
                                const LoopGroups &groups, bool backward,
                                llvm::ArrayRef<int64_t> distances) {
    llvm::SmallVector<int64_t> moved(carried.begin(), carried.end());
    for (auto [time, now, before, kind, group] :
         llvm::zip(moved, carried, previous, kinds, groups.argGroups(backward)))
      if (kind == Carried::Timed || (kind == Carried::Tone && now != before))
        if (__builtin_add_overflow(now, distances[group], &time))
          return false;
    llvm::copy(moved, carried.begin());
    return true;
  }

  /// Each timed op's period under `shifts` of its group, times `sign`: 0
  /// in a group none of whose values move (a tone nothing plays on).
  static std::shared_ptr<const llvm::DenseMap<mlir::Operation *, int64_t>>
  opPeriods(const LoopGroups &groups, const GroupShifts &shifts, int64_t sign) {
    auto periods =
        std::make_shared<llvm::DenseMap<mlir::Operation *, int64_t>>();
    for (mlir::Operation *op : llvm::make_first_range(groups.ofOp))
      (*periods)[op] = sign * shifts[groups.groupOf(op, sign < 0)].value_or(0);
    return periods;
  }

  /// Whether each group's shift is whole steps of every grid and points
  /// the way the pass runs (`sign` 1 forward, -1 backward), and some group
  /// moves.
  bool groupsMove(const GroupShifts &shifts, int64_t sign) const {
    bool moves = false;
    for (const std::optional<int64_t> &shift : shifts) {
      if (!shift)
        continue;
      if (sign * *shift < 0)
        return false;
      if constexpr (Gridded)
        if (*shift % grid.lcm() != 0)
          return false;
      moves = true;
    }
    return moves;
  }

  /// Whether, backward without resources, `branch` holds nothing in its
  /// condition's group back: it relates the groups only through answers
  /// (no branch on a measurement), and each lazy request among them ended
  /// forward, in this iteration, right where the eager one after it on its
  /// link starts (see LoopGroups::linkPairs). An eager request keeps its
  /// forward time, and so does such a lazy one, which it bounds; no wait is
  /// placed before its forward time, after its answer is due.
  bool answersHoldNothing(const LoopGroups &groups,
                          std::pair<unsigned, unsigned> branch) const {
    if (llvm::is_contained(groups.strictBranches, branch))
      return false;
    auto it = llvm::find_if(groups.lazyBranches, [&](const auto &entry) {
      return entry.first == branch;
    });
    if (it == groups.lazyBranches.end())
      return false;
    for (mlir::Operation *lazy : it->second) {
      auto pair = llvm::find_if(groups.linkPairs,
                                [&](const auto &p) { return p.lazy == lazy; });
      if (pair == groups.linkPairs.end() ||
          forwardStartOf(lazy) + durationOf(lazy) !=
              forwardStartOf(pair->eager))
        return false;
    }
    return true;
  }

  /// Whether groupsMove, each group a branch holds back moves with the
  /// group holding it back or ahead of it, with every op in the body where
  /// its group puts it (backward, each wait with what its answers decide):
  /// then what holds it back stays behind.
  bool groupsSteady(const LoopGroups &groups, const GroupShifts &shifts,
                    int64_t sign) const {
    if (!groupsMove(shifts, sign) || (sign < 0 && groups.mixedWaits))
      return false;
    bool ahead = false;
    for (auto [condition, decided] : groups.branches) {
      if (sign < 0 && !resources.enabled && !lanesBound &&
          answersHoldNothing(groups, std::pair(condition, decided)))
        continue;
      auto [holding, held] = sign > 0 ? std::pair(condition, decided)
                                      : std::pair(decided, condition);
      int64_t from = shifts[holding].value_or(0);
      int64_t to = shifts[held].value_or(0);
      if (to == from)
        continue;
      if (sign * to < sign * from)
        return false;
      ahead = true;
    }
    if (!ahead)
      return true;
    // Each op's last two samples, this iteration's and the one before.
    for (mlir::Operation *op : llvm::make_first_range(groups.ofOp)) {
      unsigned group = groups.groupOf(op, sign < 0);
      auto found = samples.find(op);
      if (found == samples.end() || found->second.size() < 2)
        return false;
      const Sample &now = found->second.back();
      const Sample &before = found->second[found->second.size() - 2];
      if (llvm::ArrayRef(now.iteration) != llvm::ArrayRef(iteration) ||
          llvm::ArrayRef(before.iteration).drop_back() !=
              llvm::ArrayRef(iteration).drop_back() ||
          before.iteration.back() != iteration.back() - sign ||
          now.start - before.start != shifts[group].value_or(0))
        return false;
    }
    return true;
  }

  mlir::LogicalResult tripCount(mlir::scf::ForOp loop, int64_t &count) {
    auto trips = pulse::constantTripCount(loop);
    if (!trips)
      return loop.emitError("cannot schedule a loop with non-constant bounds");
    count = *trips;
    return mlir::success();
  }

  /// All carried timed values advanced by the same `period`; tones may also
  /// stay where they were.
  static bool uniformShift(llvm::ArrayRef<int64_t> before,
                           llvm::ArrayRef<int64_t> after,
                           llvm::ArrayRef<Carried> kinds, int64_t &period) {
    std::optional<int64_t> shift;
    for (auto [b, a, kind] : llvm::zip(before, after, kinds)) {
      if (kind == Carried::Untimed || (kind == Carried::Tone && a == b))
        continue;
      if (shift && *shift != a - b)
        return false;
      shift = a - b;
    }
    if (!shift)
      return false;
    period = *shift;
    return true;
  }

  /// The most iterations of `loop` regrouping may run in each now.
  int64_t mostRegrouped(mlir::scf::ForOp loop) const {
    return std::min(kMostRegrouped,
                    kMostRegroupedInAll /
                        std::max<int64_t>(1, regroupedBy.lookup(loop)));
  }

  /// Note the carried times after an iteration of `loop` that did not
  /// repeat the one before, with `left` iterations after it; if the last
  /// two runs of some fewer than kMostRegrouped iterations, from two on,
  /// each moved every timed value by one shift the way the pass runs
  /// (`sign`), and more than two such runs are left, ask for the loop to be
  /// regrouped by that many.
  void noteRecent(RecentStates &recent, mlir::scf::ForOp loop,
                  llvm::ArrayRef<int64_t> state, llvm::ArrayRef<Carried> kinds,
                  int64_t left, int64_t sign) {
    recent.push(state);
    const LoopGroups *groups = nullptr;
    bool grouped = false;
    int64_t most = mostRegrouped(loop);
    for (int64_t n = 2; n <= most && 2 * n < recent.pushed; ++n) {
      if (left < 2 * n)
        return;
      int64_t first, second;
      if (uniformShift(recent.before(2 * n), recent.before(n), kinds, first) &&
          uniformShift(recent.before(n), state, kinds, second) &&
          first == second && sign * first > 0 &&
          (!Gridded || first % grid.lcm() == 0)) {
        regroup = Regroup{loop, n};
        return;
      }
      // Or each group by a shift of its own.
      if (!grouped) {
        grouped = true;
        groups = groupsApply() ? groupsOf(loop) : nullptr;
      }
      if (!groups)
        continue;
      auto earlier = groupShifts(*groups, recent.before(2 * n),
                                 recent.before(n), kinds, sign < 0);
      auto later =
          groupShifts(*groups, recent.before(n), state, kinds, sign < 0);
      if (earlier && later && *earlier == *later &&
          groupsMove(*earlier, sign)) {
        regroup = Regroup{loop, n};
        return;
      }
    }
  }

  /// Note the lanes before an iteration of `loop` (`history`, the last
  /// kMostRegrouped of them, this one's last), after which its carried
  /// times moved by `shift` as they did in the one before but its lanes did
  /// not; if they moved by as much each over the last run of some fewer
  /// than kMostRegrouped iterations, from two on, and more than two such
  /// runs are left (`left` iterations after this one), ask for the loop to
  /// be regrouped by that many.
  void noteRecentLanes(llvm::ArrayRef<LaneState> history, mlir::scf::ForOp loop,
                       int64_t shift, int64_t left) {
    int64_t most =
        std::min(mostRegrouped(loop), static_cast<int64_t>(history.size()));
    for (int64_t n = 2; n <= most; ++n) {
      if (left < 2 * n)
        return;
      if (lanesMoved(history[history.size() - n], n * shift)) {
        regroup = Regroup{loop, n};
        return;
      }
    }
  }

  /// Advance the carried values that move with the loop by `distance`.
  static void extrapolate(llvm::MutableArrayRef<int64_t> carried,
                          llvm::ArrayRef<int64_t> previous,
                          llvm::ArrayRef<Carried> kinds, int64_t distance) {
    for (auto [time, before, kind] : llvm::zip(carried, previous, kinds))
      if (kind == Carried::Timed || (kind == Carried::Tone && time != before))
        time += distance;
  }

  static Carried carriedKind(mlir::Type type, bool timed) {
    // A link, like a tone, keeps its time through a body that sends nothing.
    if (mlir::isa<pulse::ToneType, classical::LinkType>(type))
      return Carried::Tone;
    return timed ? Carried::Timed : Carried::Untimed;
  }

  // ── Grids ─────────────────────────────────────────────────────────────

  /// The grid every line of `sync` is on.
  int64_t syncGrid(pulse::SyncOp sync) {
    if (!grid.perLine())
      return grid.lcm();
    int64_t step = 1;
    for (mlir::Value line : sync.getLines())
      step = std::lcm(step, grid.of(line, lines));
    return step;
  }

  // ── Forward (ASAP) ────────────────────────────────────────────────────

  int64_t ready(mlir::Value value) const { return readyAt.lookup(value); }

  /// On several controllers, the measurements a branch condition comes
  /// from, through answers, each with the controller that reads it out; and
  /// per such controller, the latest its outcomes may be ready by the
  /// branches on the condition the backward pass has passed (late() reads it
  /// with each measurement's own bound): a branch bounds the condition once,
  /// not every measurement.
  struct AnswerGroup {
    mlir::Value condition;
    llvm::SmallVector<std::pair<mlir::Value, int64_t>, 4> measured;
    llvm::SmallVector<int64_t, 2> readers;
    llvm::SmallVector<int64_t, 2> latest;
  };

  AnswerGroup &answerGroup(mlir::Value condition) {
    auto [it, inserted] = groupOf.try_emplace(
        condition, static_cast<unsigned>(answerGroups.size()));
    if (!inserted)
      return answerGroups[it->second];
    AnswerGroup &group = answerGroups.emplace_back();
    group.condition = condition;
    for (mlir::Value measurement : conditionMeasurements(condition)) {
      int64_t reader = readerOf(measurement);
      auto slot = static_cast<unsigned>(llvm::find(group.readers, reader) -
                                        group.readers.begin());
      if (slot == group.readers.size()) {
        group.readers.push_back(reader);
        group.latest.push_back(kNever);
      }
      group.measured.push_back({measurement, reader});
      memberOf[measurement].push_back({it->second, slot});
    }
    return group;
  }

  /// `time`, or the latest the groups of `measurement` let it be ready
  /// (only measurements are in groups).
  int64_t withGroups(mlir::Value measurement, int64_t time) const {
    if (memberOf.empty())
      return time;
    auto it = memberOf.find(measurement);
    if (it == memberOf.end())
      return time;
    for (auto [group, slot] : it->second)
      time = std::min(time, answerGroups[group].latest[slot]);
    return time;
  }

  /// Bound the measurements of the groups on a condition inside `loop` by
  /// what its branches asked, before the loop's next iteration forgets the
  /// bounds of its body's values.
  void settleGroups(mlir::Operation *loop) {
    for (AnswerGroup &group : answerGroups) {
      mlir::Value condition = group.condition;
      mlir::Operation *at = condition.getDefiningOp();
      if (!at)
        at = condition.getParentBlock()->getParentOp();
      if (!loop->isAncestor(at))
        continue;
      for (auto [slot, latest] : llvm::enumerate(group.latest)) {
        if (latest == kNever)
          continue;
        for (auto [measurement, reader] : group.measured)
          if (reader == group.readers[slot])
            bound(measurement, latest);
        latest = kNever;
      }
    }
  }

  /// On several controllers, when the latest measurement a branch condition
  /// comes from (through answers) is ready, and when the outcomes reach each
  /// controller: what forwardIf takes from them, found once per condition
  /// while none of them is placed again (forwardLoop forgets them at each
  /// iteration).
  struct ConditionReady {
    bool any = false;
    int64_t ready = 0;
    llvm::SmallVector<int64_t, 4> reached;
  };

  const ConditionReady &conditionReadyOf(mlir::Value condition) {
    auto [it, inserted] = conditionReady.try_emplace(condition);
    ConditionReady &at = it->second;
    if (!inserted)
      return at;
    for (auto [measurement, reader] : answerGroup(condition).measured) {
      int64_t time = ready(measurement);
      if (!at.any) {
        at.any = true;
        at.ready = time;
        at.reached.assign(static_cast<size_t>(controllers.count()), 0);
      }
      at.ready = std::max(at.ready, time);
      for (auto [c, reached] : llvm::enumerate(at.reached))
        reached = std::max(
            reached, time + controllers.extra(reader, static_cast<int64_t>(c)));
    }
    return at;
  }

  /// When every operand of `op` is available: data from answers included.
  /// A submission with its own order on a link (`ownOrder`) does not wait
  /// for the link it is threaded through.
  int64_t operandsReady(mlir::Operation *op) const {
    int64_t time = 0;
    bool own = ownOrder(op);
    for (mlir::Value operand : op->getOperands())
      if (!own || !mlir::isa<classical::LinkType>(operand.getType()))
        time = std::max(time, ready(operand));
    return time;
  }

  /// A link's request slots: a heap of the times they become free (forward,
  /// min-heap) or must be free from (backward, max-heap). A link's slots
  /// bound every policy: they are hardware. Also the start of the request
  /// placed on it last in the pass's order, the latest forward and the
  /// earliest backward: every request placed after it starts no earlier
  /// (forward) or ends its submission by it (backward), so only the slots
  /// held past it (heldPast) can delay one.
  struct LinkSlots {
    llvm::SmallVector<int64_t> free;
    std::optional<int64_t> last;
  };

  LinkSlots &slotsOf(mlir::Operation *submit) {
    auto [it, inserted] = linkLanes.try_emplace(linkOf(submit));
    if (inserted)
      it->second.free.assign(
          static_cast<size_t>(std::max<int64_t>(
              1, classical::intAttr(submit, "classical.slots", 64))),
          laneReset);
    return it->second;
  }

  llvm::SmallVector<int64_t> &lanesOf(mlir::Operation *submit) {
    return slotsOf(submit).free;
  }

  /// The links a loop's requests use, each with its submissions.
  using LoopLinks = llvm::SmallVector<
      std::pair<LinkKey, llvm::SmallVector<mlir::Operation *, 2>>, 1>;

  LoopLinks linksOf(mlir::scf::ForOp loop) {
    LoopLinks links;
    if (!classical)
      return links;
    loop.getBody()->walk([&](classical::SubmitOp submit) {
      LinkKey key = linkOf(submit);
      auto it = llvm::find_if(
          links, [&](const auto &entry) { return entry.first == key; });
      if (it == links.end())
        it =
            &links.emplace_back(key, llvm::SmallVector<mlir::Operation *, 2>{});
      it->second.push_back(submit);
    });
    return links;
  }

  /// The largest budget of a request on a link, which bounds how long past
  /// its submission one holds its slot.
  int64_t linkBudget(const LinkKey &key) {
    if (linkBudgets.empty())
      region->walk([&](classical::SubmitOp submit) {
        int64_t &most = linkBudgets[linkOf(submit)];
        most = std::max(most, budgetOf(submit));
      });
    return linkBudgets.lookup(key);
  }

  /// Whether a slot free at (forward) or from (backward) `free` can delay a
  /// request placed after the link's last: forward, one free past its
  /// start; backward, one to be free before its start plus the largest
  /// budget, where every request placed after it has its slot free again.
  bool heldPast(int64_t free, const LinkSlots &slots, int64_t budget,
                bool backward) const {
    if (!slots.last)
      return true;
    return backward ? free < *slots.last + budget : free > *slots.last;
  }

  /// Per link of a loop, its last request and the slots held past it, in
  /// order.
  struct HeldSlots {
    std::optional<int64_t> last;
    llvm::SmallVector<int64_t, 4> held;
  };

  llvm::SmallVector<HeldSlots, 1> heldSlots(const LoopLinks &links,
                                            bool backward) {
    llvm::SmallVector<HeldSlots, 1> state(links.size());
    for (auto [link, slots] : llvm::zip(links, state)) {
      auto it = linkLanes.find(link.first);
      if (it == linkLanes.end())
        continue;
      int64_t budget = backward ? linkBudget(link.first) : 0;
      slots.last = it->second.last;
      for (int64_t free : it->second.free)
        if (heldPast(free, it->second, budget, backward))
          slots.held.push_back(free);
      llvm::sort(slots.held);
    }
    return state;
  }

  /// Whether an iteration moved each link's last request and held slots
  /// from `before` by the shift its requests move by (`shiftOf`; none when
  /// they move by several), so that the iterations after it hold the slots
  /// as it did, moved as they are.
  bool slotsMoved(const LoopLinks &links, llvm::ArrayRef<HeldSlots> before,
                  bool backward,
                  llvm::function_ref<
                      std::optional<int64_t>(llvm::ArrayRef<mlir::Operation *>)>
                      shiftOf) {
    if (before.size() != links.size())
      return false;
    auto now = heldSlots(links, backward);
    for (auto [link, was, is] : llvm::zip(links, before, now)) {
      std::optional<int64_t> shift = shiftOf(link.second);
      if (!shift || !was.last || !is.last || *is.last != *was.last + *shift ||
          is.held.size() != was.held.size())
        return false;
      for (auto [x, y] : llvm::zip(was.held, is.held))
        if (y != x + *shift)
          return false;
    }
    return true;
  }

  /// Move each link's last request and held slots by `distanceOf` its
  /// requests, as the iterations not simulated do (see slotsMoved).
  void moveSlots(const LoopLinks &links, bool backward,
                 llvm::function_ref<int64_t(llvm::ArrayRef<mlir::Operation *>)>
                     distanceOf) {
    for (const auto &[key, ops] : links) {
      auto it = linkLanes.find(key);
      if (it == linkLanes.end() || !it->second.last)
        continue;
      int64_t distance = distanceOf(ops);
      int64_t budget = backward ? linkBudget(key) : 0;
      // Only held slots move, the way the heap orders them: it stays one.
      for (int64_t &free : it->second.free)
        if (heldPast(free, it->second, budget, backward))
          free += distance;
      *it->second.last += distance;
    }
  }

  /// Whether a loop whose iterations shift by `period` keeps every link's
  /// requests within its slots in steady state, so that extrapolating it
  /// cannot exceed them: per link, the submissions of one iteration stay in
  /// flight for ceil(busy / period) iterations each.
  bool slotsHoldSteadyState(mlir::scf::ForOp loop, int64_t period) {
    if (!classical)
      return true;
    std::map<LinkKey, std::pair<int64_t, int64_t>> perLink;
    bool ok = true;
    loop.getBody()->walk([&](classical::SubmitOp submit) {
      mlir::Operation *op = submit;
      std::optional<int64_t> held;
      if (period > 0)
        held = inFlight(op, loop, period);
      auto &[need, slots] = perLink[linkOf(op)];
      if (!held) {
        ok = false;
        return;
      }
      if (__builtin_add_overflow(need, *held, &need)) {
        ok = false;
        return;
      }
      slots = classical::intAttr(op, "classical.slots", 64);
    });
    for (auto &[device, entry] : perLink)
      ok = ok && entry.first <= entry.second;
    return ok;
  }

  /// Take the earliest free slot of a link for a request starting no earlier
  /// than `start`; it is busy until the answer is due.
  int64_t takeSlot(mlir::Operation *submit, int64_t start, int64_t busy) {
    LinkSlots &slots = slotsOf(submit);
    auto &heap = slots.free;
    auto later = std::greater<int64_t>();
    std::pop_heap(heap.begin(), heap.end(), later);
    if (heap.back() > start) {
      start = heap.back();
      lanesBound = true;
    }
    heap.back() = start + busy;
    std::push_heap(heap.begin(), heap.end(), later);
    slots.last = std::max(slots.last.value_or(start), start);
    return start;
  }

  /// The backward mirror: the latest a request ending its slot by the slot's
  /// bound may start, for one that is busy for `busy`.
  int64_t takeSlotBackward(mlir::Operation *submit, int64_t start, int64_t busy,
                           bool pinned) {
    LinkSlots &slots = slotsOf(submit);
    auto &heap = slots.free;
    std::pop_heap(heap.begin(), heap.end());
    if (heap.back() - busy < start) {
      // A slot no later request took is free however long this one holds
      // it, past the end of the program too.
      if (pinned) {
        if (heap.back() < laneReset)
          slotsOverrun = true;
      } else {
        start = heap.back() - busy;
        lanesBound = true;
      }
    }
    heap.back() = start;
    std::push_heap(heap.begin(), heap.end());
    slots.last = std::min(slots.last.value_or(start), start);
    return start;
  }

  /// One execution of a submission in a scheduled block, in the block's
  /// local time: when it starts and how long it holds its link's slot.
  struct BlockRequest {
    int64_t start, busy;
    mlir::Operation *op;
  };

  static void collectRequests(mlir::Block &body,
                              llvm::SmallVectorImpl<int64_t> &iteration,
                              llvm::SmallVectorImpl<BlockRequest> &found) {
    for (mlir::Operation &op : body) {
      if (mlir::isa<classical::SubmitOp>(op)) {
        if (auto start = pulse::scheduledStart(&op, iteration))
          found.push_back({*start, durationOf(&op) + budgetOf(&op), &op});
      } else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
        int64_t trips = pulse::constantTripCount(loop).value_or(0);
        iteration.push_back(0);
        for (int64_t k = 0; k < trips; ++k) {
          iteration.back() = k;
          collectRequests(*loop.getBody(), iteration, found);
        }
        iteration.pop_back();
      } else {
        // Both arms of a branch hold their slots, as the check counts them.
        for (mlir::Region &region : op.getRegions())
          for (mlir::Block &nested : region)
            collectRequests(nested, iteration, found);
      }
    }
  }

  /// The requests of `block`, ordered by local start (forward) or by local
  /// end, latest first (backward).
  llvm::ArrayRef<BlockRequest> requestsOf(pulse::ScheduleBlockOp block,
                                          bool backward) {
    auto [it, inserted] = blockRequests.try_emplace(block.getOperation());
    auto &[forwardOrder, backwardOrder] = it->second;
    if (inserted) {
      llvm::SmallVector<int64_t, 2> iteration;
      collectRequests(block.getBody().front(), iteration, forwardOrder);
      llvm::stable_sort(forwardOrder, [](const auto &a, const auto &b) {
        return a.start < b.start;
      });
      backwardOrder = forwardOrder;
      llvm::stable_sort(backwardOrder, [](const auto &a, const auto &b) {
        return a.start + a.busy > b.start + b.busy;
      });
    }
    return backward ? backwardOrder : forwardOrder;
  }

  /// Move a block placed at `start` until the requests in it fit the slots
  /// its links have left, later (forward) or earlier (backward), and take
  /// those slots. The block's own schedule fits empty links, so moving it
  /// past the requests around it always succeeds.
  int64_t takeBlockSlots(pulse::ScheduleBlockOp block, int64_t start,
                         bool backward) {
    llvm::ArrayRef<BlockRequest> requests = requestsOf(block, backward);
    if (requests.empty())
      return start;
    while (true) {
      std::map<LinkKey, llvm::SmallVector<int64_t>> trial;
      int64_t shift = 0;
      for (const BlockRequest &request : requests) {
        LinkKey key = linkOf(request.op);
        auto it = trial.find(key);
        if (it == trial.end())
          it = trial.try_emplace(key, lanesOf(request.op)).first;
        auto &heap = it->second;
        int64_t at = start + request.start;
        if (backward) {
          std::pop_heap(heap.begin(), heap.end());
          shift = at + request.busy - heap.back();
          heap.back() = at;
          std::push_heap(heap.begin(), heap.end());
        } else {
          auto later = std::greater<int64_t>();
          std::pop_heap(heap.begin(), heap.end(), later);
          shift = heap.back() - at;
          heap.back() = at + request.busy;
          std::push_heap(heap.begin(), heap.end(), later);
        }
        if (shift > 0)
          break;
      }
      if (shift <= 0) {
        for (auto &[key, heap] : trial)
          linkLanes[key].free = std::move(heap);
        for (const BlockRequest &request : requests) {
          std::optional<int64_t> &last = linkLanes[linkOf(request.op)].last;
          int64_t at = start + request.start;
          last = backward ? std::min(last.value_or(at), at)
                          : std::max(last.value_or(at), at);
        }
        return start;
      }
      start += backward ? -shift : shift;
      if constexpr (Gridded)
        start = backward ? pulse::LineGrid::roundDown(start, grid.lcm())
                         : pulse::LineGrid::roundUp(start, grid.lcm());
      lanesBound = true;
    }
  }

  void forwardSubmit(classical::SubmitOp submit) {
    mlir::Operation *op = submit;
    int64_t t = timelineOf(op);
    bool own = ownOrder(op);
    int64_t link = own ? linkReady[linkOf(op)] : ready(submit.getLink());
    int64_t start =
        std::max({link, floorOf(t), operandsReady(op), armFloorOf(kAnyOwner)});
    if (blocksOnCalls)
      start = std::max(start, endOf(t));
    llvm::ArrayRef<int64_t> held;
    if (holding()) {
      held = heldBy(submit);
      for (int64_t c : held)
        start = std::max(start, heldEnds[c]);
    }
    int64_t duration = durationOf(op), budget = budgetOf(op);
    start = takeSlot(op, start, duration + budget);
    record(op, start);
    readyAt[submit.getUpdatedLink()] = start + duration;
    if (own)
      linkReady[linkOf(op)] = start + duration;
    readyAt[submit.getFuture()] = start + duration + budget;
    if (blocksOnCalls)
      floors[t] = std::max(floors[t], start + duration + budget);
    for (int64_t c : held)
      heldFloors[c] = std::max(heldFloors[c], start + duration + budget);
    noteEnd(start + duration, t);
  }

  void forwardWait(classical::WaitOp wait) {
    int64_t t = timelineOf(wait);
    int64_t start =
        std::max({ready(wait.getFuture()), floorOf(t), armFloorOf(kAnyOwner)});
    // A blocking wait is a barrier: every line of its timeline is done
    // before it.
    if (wait->hasAttr("classical.blocking"))
      start = std::max(start, endOf(t));
    record(wait, start);
    for (mlir::Value value : wait.getValues())
      readyAt[value] = start;
    if (wait->hasAttr("classical.blocking"))
      floors[t] = std::max(floors[t], start);
    noteEnd(start, t);
  }

  void place(mlir::Operation *op, int64_t start, int64_t t, int64_t duration) {
    record(op, start);
    int64_t end = start + duration;
    int64_t lineEnd = end;
    if (auto readout = mlir::dyn_cast<pulse::ReadoutOp>(op)) {
      int64_t latency = several() ? controllers.own(controllerOf(op))
                                  : resources.readoutLatency;
      if (resources.enabled)
        lineEnd += latency;
      readyAt[readout.getResult()] = end + latency;
    }
    readyAt[op->getResult(0)] = lineEnd;
    if (mlir::Value tone = toneResult(op))
      readyAt[tone] = end;
    if (holding())
      noteHeldEnd(op, lineEnd);
    noteEnd(lineEnd, t);
  }

  void noteEnd(int64_t end, int64_t t) {
    makespan = std::max(makespan, end);
    iterationEnd = std::max(iterationEnd, end);
    ends[t] = std::max(ends[t], end);
    iterationEnds[t] = std::max(iterationEnds[t], end);
  }

  mlir::LogicalResult forwardBlock(mlir::Block &block) {
    for (mlir::Operation &op : block.without_terminator())
      if (mlir::failed(forwardOp(&op)))
        return mlir::failure();
    return mlir::success();
  }

  mlir::LogicalResult forwardOp(mlir::Operation *op) {
    if (classical) {
      if (auto submit = mlir::dyn_cast<classical::SubmitOp>(op)) {
        forwardSubmit(submit);
        return mlir::success();
      }
      if (auto wait = mlir::dyn_cast<classical::WaitOp>(op)) {
        forwardWait(wait);
        return mlir::success();
      }
    }
    if (isTimedOp(op)) {
      mlir::FailureOr<int64_t> duration = placedDuration(op);
      if (mlir::failed(duration))
        return mlir::failure();
      int64_t t = timelineOf(op);
      int64_t start = std::max(ready(op->getOperand(0)), floorOf(t));
      int64_t owner = several() ? controllerOf(op) : kAnyOwner;
      if (!armFloors.empty())
        start = std::max(start, armFloorOf(owner));
      if (holding())
        start = std::max(start, heldFloorOf(op));
      if (mlir::Value tone = toneOperand(op))
        start = std::max(start, ready(tone));
      if (classical) {
        start = std::max(start, operandsReady(op));
        // An outcome it computes with reaches its controller later than the
        // reader's.
        if (several())
          for (mlir::Value measurement : valueMeasurements(op))
            start = std::max(
                start, ready(measurement) +
                           controllers.extra(readerOf(measurement), owner));
        // A variable wait is a barrier: every line of its timeline is done
        // before it.
        if (op->hasAttr("classical.variable"))
          start = std::max(start, endOf(t));
      }
      int64_t step = 1;
      if constexpr (Gridded)
        step = grid.of(op->getOperand(0), lines);
      if (resources.enabled && !mlir::isa<pulse::WaitOp>(op)) {
        bool isDrive = mlir::isa<pulse::DriveOp>(op);
        auto &lanes = isDrive ? driveLanes[t] : readoutLanes[t];
        ++(isDrive ? driveUses : readoutUses)[t];
        if (several()) {
          // Another controller's op ends a lane only as surely as the two
          // starts are known.
          auto &owners = isDrive ? driveOwners[t] : readoutOwners[t];
          size_t best = 0;
          int64_t free = kNever;
          for (size_t i = 0; i < lanes.size(); ++i) {
            int64_t at = lanes[i] + laneGap(owners[i], owner);
            if (at < free) {
              free = at;
              best = i;
            }
          }
          start = std::max(start, free);
          if (step > 1)
            start = pulse::LineGrid::roundUp(start, step);
          lanes[best] = start + *duration + resources.switchPenalty;
          owners[best] = owner;
        } else {
          auto lane = std::min_element(lanes.begin(), lanes.end());
          start = std::max(start, *lane);
          if (step > 1)
            start = pulse::LineGrid::roundUp(start, step);
          *lane = start + *duration + resources.switchPenalty;
        }
      }
      if (step > 1)
        start = pulse::LineGrid::roundUp(start, step);
      place(op, start, t, *duration);
      if (classical && op->hasAttr("classical.variable"))
        floors[t] = std::max(floors[t], start + *duration);
      return mlir::success();
    }
    if (auto block = mlir::dyn_cast<pulse::ScheduleBlockOp>(op)) {
      forwardScheduleBlock(block);
      return mlir::success();
    }
    if (auto sync = mlir::dyn_cast<pulse::SyncOp>(op)) {
      int64_t t = timelineOf(op);
      int64_t time = floorOf(t);
      for (mlir::Value line : sync.getLines()) {
        time = std::max(time, ready(line));
        if constexpr (Gridded)
          if (!armFloors.empty())
            time = std::max(time, armFloorOf(controllerOfLine(line)));
      }
      if constexpr (Gridded)
        time = pulse::LineGrid::roundUp(time, syncGrid(sync));
      for (mlir::Value result : sync.getResults())
        readyAt[result] = time;
      noteEnd(time, t);
      return mlir::success();
    }
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op))
      return forwardLoop(loop);
    if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op))
      return forwardIf(branch);
    // Any other op releases its timed results when its timed operands are;
    // with calls, every result is released when all its operands are.
    if (classical) {
      int64_t time = operandsReady(op);
      for (mlir::Value result : op->getResults())
        readyAt[result] = time;
      return mlir::success();
    }
    int64_t time = 0;
    for (mlir::Value operand : op->getOperands())
      if (isTimedType(operand.getType()))
        time = std::max(time, ready(operand));
    for (mlir::Value result : op->getResults())
      if (isTimedType(result.getType()))
        readyAt[result] = time;
    return mlir::success();
  }

  /// Start the block once every operand is there when the block needs it.
  void forwardScheduleBlock(pulse::ScheduleBlockOp block) {
    BlockShape shape = shapeOf(block);
    auto inputs = block.getInputs();
    int64_t t = timelineOf(block);
    int64_t start = floorOf(t);
    for (auto [input, offset] : llvm::zip(inputs, shape.in))
      if (offset >= 0)
        start = std::max(start, ready(input) - offset);
    // A block placed whole is held on every controller.
    if (holding())
      start = std::max(start, *llvm::max_element(heldFloors));
    if constexpr (Gridded)
      start = pulse::LineGrid::roundUp(start, grid.lcm());
    if (classical)
      start = takeBlockSlots(block, start, /*backward=*/false);
    record(block, start);
    for (auto [result, offset, source] :
         llvm::zip(block.getOutputs(), shape.out, shape.source))
      readyAt[result] = offset >= 0 ? start + offset : ready(inputs[source]);
    for (int64_t &at : heldEnds)
      at = std::max(at, start + shape.length);
    noteEnd(start + shape.length, t);
  }

  mlir::LogicalResult forwardLoop(mlir::scf::ForOp loop) {
    int64_t count;
    if (mlir::failed(tripCount(loop, count)))
      return mlir::failure();
    auto args = loop.getRegionIterArgs();
    auto yield =
        mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
    llvm::SmallVector<Carried> kinds;
    llvm::SmallVector<int64_t> carried;
    for (mlir::Value init : loop.getInitArgs()) {
      kinds.push_back(carriedKind(init.getType(), isTimedType(init.getType())));
      carried.push_back(ready(init));
    }
    // Times the body reads from outside the loop; like the enclosing floor,
    // they stop mattering once every carried value is past them.
    int64_t outside = floor;
    loop.getBody()->walk([&](mlir::Operation *inner) {
      for (mlir::Value operand : inner->getOperands())
        if (isTimedType(operand.getType()) &&
            !loop->isAncestor(operand.getParentRegion()->getParentOp()))
          outside = std::max(outside, ready(operand));
    });
    // So do the floors of the timelines carried values are on.
    llvm::SmallVector<int64_t> outsideOf;
    for (mlir::Value init : loop.getInitArgs())
      outsideOf.push_back(
          std::max(outside, isLineType(init.getType()) || !isolated()
                                ? floors[timelineOfValue(init)]
                                : *llvm::max_element(floors)));

    iteration.push_back(0);
    std::optional<int64_t> lastShift;
    int64_t laneTries = laneTriesPerLoop();
    const LoopGroups *groups = nullptr;
    bool grouped = false;
    std::optional<Unshifted> unshifted;
    std::optional<GroupShifts> lastGroups;
    RecentStates recent;
    // Lanes that repeat only every few iterations regroup a loop of calls
    // (a loop without is summarized by the controller check as it is).
    bool regroupLanes = resources.enabled && mayRegroup && holdsCalls(loop);
    llvm::SmallVector<LaneState, 0> laneHistory;
    LoopLinks links = linksOf(loop);
    for (int64_t k = 0; k < count; ++k) {
      iteration.back() = k;
      // Each iteration places its readouts again, so the conditions on them
      // are found again.
      if (!conditionReady.empty())
        conditionReady.clear();
      // Slots held past an iteration's last request may delay the next
      // ones: their iterations repeat only if the iteration moved them too.
      llvm::SmallVector<HeldSlots, 1> slotsBefore;
      if (!links.empty() && !lanesBound)
        slotsBefore = heldSlots(links, false);
      int64_t savedEnd = iterationEnd;
      iterationEnd = 0;
      auto savedEnds = iterationEnds;
      std::fill(iterationEnds.begin(), iterationEnds.end(), -1);
      auto floorsBefore = floors;
      auto heldFloorsBefore = heldFloors, heldEndsBefore = heldEnds;
      auto linksBefore = linkReady;
      // Only an iteration moving as the one before can be steady.
      LaneState lanesBefore;
      if (lastShift && laneTries > 0)
        lanesBefore = laneState();
      if (regroupLanes)
        pushLanes(laneHistory);
      LaneState groupLanes;
      if (resources.enabled && groupsApply())
        groupLanes = laneState();
      for (auto [arg, time] : llvm::zip(args, carried))
        readyAt[arg] = time;
      if (mlir::failed(forwardBlock(*loop.getBody())))
        return mlir::failure();
      llvm::SmallVector<int64_t> next;
      for (mlir::Value value : yield.getOperands())
        next.push_back(ready(value));
      int64_t thisEnd = iterationEnd;
      iterationEnd = std::max(savedEnd, iterationEnd);
      auto theseEnds = iterationEnds;
      for (auto [end, saved] : llvm::zip(iterationEnds, savedEnds))
        end = std::max(end, saved);

      int64_t shift;
      bool shifted = uniformShift(carried, next, kinds, shift);
      bool repeated = shifted && lastShift == shift;
      bool steady =
          !lanesBound && repeated && shift >= 0 &&
          (!Gridded || shift % grid.lcm() == 0) &&
          slotsHoldSteadyState(loop, shift) &&
          (!resources.enabled || lanesSteady(lanesBefore, shift, laneTries)) &&
          slotsMoved(links, slotsBefore, false,
                     [&](llvm::ArrayRef<mlir::Operation *>) { return shift; });
      bool bounded = true;
      for (auto [before, after, kind, bound] :
           llvm::zip(carried, next, kinds, outsideOf))
        if (kind == Carried::Timed ||
            (kind == Carried::Tone && after != before))
          bounded = bounded && after >= bound;
      steady = steady && bounded;
      lastShift = shifted ? std::optional<int64_t>(shift) : std::nullopt;
      // Lines that move by periods of their own repeat too, group by group.
      std::optional<GroupShifts> nowGroups;
      if (!shifted && k + 1 < count && groupsApply()) [[unlikely]] {
        if (!grouped && !unshifted) {
          unshifted = Unshifted{carried, next};
        } else if (!grouped) {
          grouped = true;
          groups = groupsOf(loop);
          if (groups)
            lastGroups = groupShifts(*groups, unshifted->before,
                                     unshifted->after, kinds, false);
          unshifted.reset();
        }
        if (groups)
          nowGroups = groupShifts(*groups, carried, next, kinds, false);
      } else if (!grouped) {
        unshifted.reset();
      }
      bool mixed = nowGroups && bounded && lastGroups == nowGroups &&
                   groupsSteady(*groups, *nowGroups, 1) &&
                   slotsHoldGroups(loop, *groups, *nowGroups, 1) &&
                   groupLanesMoved(groupLanes, *groups, *nowGroups) &&
                   slotsMoved(links, slotsBefore, false,
                              [&](llvm::ArrayRef<mlir::Operation *> ops) {
                                return requestsShift(loop, *groups, *nowGroups,
                                                     ops, false);
                              });
      bool groupsRepeated = nowGroups && lastGroups == nowGroups;
      lastGroups = std::move(nowGroups);
      llvm::SmallVector<int64_t> previous = std::move(carried);
      carried = std::move(next);
      if (mixed) [[unlikely]] {
        auto distances = groupDistances(*lastGroups, count - 1 - k);
        if (distances && extrapolateGroups(carried, previous, kinds, *groups,
                                           false, *distances)) {
          moveGroupLanes(groupLanes, *groups, *distances);
          moveSlots(links, false, [&](llvm::ArrayRef<mlir::Operation *> ops) {
            return *requestsShift(loop, *groups, *lastGroups, ops, false) *
                   (count - 1 - k);
          });
          // Each line ends its last iteration where it is carried to, and
          // each wait in the body where its group moves it.
          for (auto [init, time, group] :
               llvm::zip(loop.getInitArgs(), carried, groups->ofArg))
            if ((isLineType(init.getType()) ||
                 mlir::isa<classical::LinkType>(init.getType())) &&
                (*distances)[group] > 0)
              noteEnd(time, 0);
          for (auto [op, group] : groups->ofOp) {
            if (!mlir::isa<classical::WaitOp>(op) ||
                op->template getParentOfType<mlir::scf::ForOp>() != loop)
              continue;
            auto found = samples.find(op);
            if (found != samples.end() && !found->second.empty() &&
                llvm::ArrayRef(found->second.back().iteration) ==
                    llvm::ArrayRef(iteration))
              noteEnd(found->second.back().start + (*distances)[group], 0);
          }
          extrapolations.push_back(
              {loop, 0, opPeriods(*groups, *lastGroups, 1)});
          break;
        }
      }
      if (steady && k + 1 < count) {
        int64_t remaining = count - 1 - k;
        int64_t distance = remaining * shift;
        extrapolate(carried, previous, kinds, distance);
        moveSlots(links, false,
                  [&](llvm::ArrayRef<mlir::Operation *>) { return distance; });
        // What the iterations not simulated move: the ends and run-time
        // waits of the timelines the body plays on, and their links.
        if (!isolated()) {
          noteEnd(thisEnd + distance, 0);
        } else {
          for (auto [t, end] : llvm::enumerate(theseEnds))
            if (end >= 0)
              noteEnd(end + distance, static_cast<int64_t>(t));
        }
        for (auto &[key, time] : linkReady) {
          auto before = linksBefore.find(key);
          if (before == linksBefore.end() || before->second != time)
            time += distance;
        }
        for (auto [now, before] : llvm::zip(floors, floorsBefore))
          if (now != before)
            now += distance;
        moveChanged(heldFloors, heldFloorsBefore, distance);
        moveChanged(heldEnds, heldEndsBefore, distance);
        if (resources.enabled)
          moveLanes(lanesBefore, distance);
        extrapolations.push_back({loop, shift, nullptr});
        break;
      }
      if (regroup)
        break;
      // Regrouping is for a loop that repeats only every few iterations:
      // one whose groups or lanes repeat every iteration is left to the
      // checks above.
      if (k >= 2 && !repeated && !groupsRepeated && mayRegroup) {
        noteRecent(recent, loop, carried, kinds, count - 1 - k, 1);
        if (regroup)
          break;
      }
      if (repeated && bounded && !lanesBound && laneTries <= 0 &&
          !laneHistory.empty() && !lanesMoved(laneHistory.back(), shift)) {
        noteRecentLanes(laneHistory, loop, shift, count - 1 - k);
        if (regroup)
          break;
      }
    }
    iteration.pop_back();
    for (auto [result, time] : llvm::zip(loop.getResults(), carried))
      readyAt[result] = time;
    return mlir::success();
  }

  /// After a branch, each lane is as busy as the busier arm left it.
  static void mergeLanes(Lanes &lanes, const Lanes &other, bool later) {
    for (auto [mine, theirs] : llvm::zip(lanes, other))
      for (auto [a, b] : llvm::zip(mine, theirs))
        a = later ? std::max(a, b) : std::min(a, b);
  }

  /// After a branch, a lane's last op may be either arm's.
  static void mergeOwners(Lanes &owners, const Lanes &other) {
    for (auto [mine, theirs] : llvm::zip(owners, other))
      for (auto [a, b] : llvm::zip(mine, theirs))
        if (a != b)
          a = a == kNoOwner ? b : b == kNoOwner ? a : kAnyOwner;
  }

  /// Whether what `branch` yields is there only once its condition is: when
  /// the condition reads a measurement, or comes from a classical call, an
  /// arm may set a tone or a parameter without a timed op, and the op that
  /// reads it after the if still waits for the condition.
  bool yieldsAfterCondition(mlir::scf::IfOp branch) const {
    if (classical)
      return true;
    return !conditionMeasurements(branch.getCondition()).empty();
  }

  mlir::LogicalResult forwardIf(mlir::scf::IfOp branch) {
    int64_t savedFloor = floor;
    auto savedArmFloors = armFloors;
    // On several controllers each outcome reaches each controller apart,
    // so the measurements an answer comes from bound the arm too, and each
    // controller's ops wait until the outcomes reach it.
    if (several()) {
      const ConditionReady &at = conditionReadyOf(branch.getCondition());
      if (at.any) {
        floor = std::max(floor, at.ready);
        if (armFloors.empty())
          armFloors.assign(static_cast<size_t>(controllers.count()), 0);
        for (auto [arm, reached] : llvm::zip(armFloors, at.reached))
          arm = std::max(arm, reached);
      }
    } else {
      for (mlir::Value measurement :
           conditionMeasurements(branch.getCondition(), false))
        floor = std::max(floor, ready(measurement));
    }
    if (classical)
      floor = std::max(floor, ready(branch.getCondition()));

    auto savedDrive = driveLanes, savedReadout = readoutLanes;
    auto savedDriveOwners = driveOwners, savedReadoutOwners = readoutOwners;
    auto savedLinks = linkReady;
    if (mlir::failed(forwardBlock(*branch.thenBlock())))
      return mlir::failure();
    llvm::SmallVector<int64_t> results;
    for (mlir::Value value : branch.thenYield().getOperands())
      results.push_back(ready(value));
    if (!branch.getElseRegion().empty()) {
      auto thenDrive = driveLanes, thenReadout = readoutLanes;
      auto thenDriveOwners = std::move(driveOwners);
      auto thenReadoutOwners = std::move(readoutOwners);
      auto thenLinks = std::move(linkReady);
      driveLanes = savedDrive;
      readoutLanes = savedReadout;
      driveOwners = std::move(savedDriveOwners);
      readoutOwners = std::move(savedReadoutOwners);
      linkReady = std::move(savedLinks);
      if (mlir::failed(forwardBlock(*branch.elseBlock())))
        return mlir::failure();
      for (auto [time, value] :
           llvm::zip(results, branch.elseYield().getOperands()))
        time = std::max(time, ready(value));
      mergeLanes(driveLanes, thenDrive, /*later=*/true);
      mergeLanes(readoutLanes, thenReadout, /*later=*/true);
      mergeOwners(driveOwners, thenDriveOwners);
      mergeOwners(readoutOwners, thenReadoutOwners);
      for (auto &[key, time] : thenLinks) {
        auto [it, inserted] = linkReady.try_emplace(key, time);
        if (!inserted)
          it->second = std::max(it->second, time);
      }
    }
    if (yieldsAfterCondition(branch))
      for (int64_t &time : results)
        time = std::max({time, floor, armFloorOf(kAnyOwner)});
    for (auto [result, time] : llvm::zip(branch.getResults(), results))
      readyAt[result] = time;
    floor = savedFloor;
    armFloors = std::move(savedArmFloors);
    return mlir::success();
  }

  // ── Backward (ALAP) ───────────────────────────────────────────────────

  /// The latest time `value` may become available without delaying the
  /// schedule. Lines left unconsumed may run until the makespan, or until
  /// the end of their timeline: a later run-time wait elsewhere does not
  /// move them. Nothing waits for an unused measurement.
  int64_t late(mlir::Value value) {
    auto it = lateAt.find(value);
    if (it != lateAt.end())
      return withGroups(value, it->second);
    mlir::Type type = value.getType();
    if (mlir::isa<pulse::MeasurementType>(type))
      return withGroups(value, kNever);
    // Nothing waits for an unused future or untimed value; a line, tone or
    // link may run until the makespan.
    if (classical && !mlir::isa<pulse::DriveLineType, pulse::ReadoutLineType,
                                pulse::ToneType, classical::LinkType>(type))
      return kNever;
    if (isolated() && !mlir::isa<classical::LinkType>(type))
      return ends[timelineOfValue(value)];
    return makespan;
  }

  /// The forward start of a call op in the current iteration, also for an
  /// iteration the forward pass extrapolated.
  int64_t forwardStartOf(mlir::Operation *op) const {
    auto loops = pulse::enclosingLoops(op, region->getParentOp());
    return forwardStartAt(op, iteration, loops);
  }

  /// forwardStartOf in iteration `at` of the loops around `op`, `loops`.
  int64_t forwardStartAt(mlir::Operation *op, llvm::ArrayRef<int64_t> at,
                         llvm::ArrayRef<mlir::scf::ForOp> loops) const {
    std::pair<mlir::Operation *, llvm::SmallVector<int64_t, 2>> key{
        op, llvm::SmallVector<int64_t, 2>(at)};
    auto it = forwardStarts.find(key);
    if (it != forwardStarts.end())
      return it->second;
    // The latest iteration simulated before this one of the innermost loop
    // the forward pass extrapolated, in the same iteration of the loops
    // around it, shifted by its period.
    for (int64_t level = static_cast<int64_t>(at.size()) - 1; level >= 0;
         --level) {
      if (level >= static_cast<int64_t>(loops.size()))
        continue;
      std::optional<int64_t> period;
      for (const Extrapolation &e : forwardExtrapolations)
        if (e.loop == loops[level]) {
          period = e.periodOf(op).value_or(0);
          break;
        }
      if (!period)
        continue;
      key.second.assign(at.begin(), at.begin() + level + 1);
      auto before = forwardStarts.lower_bound(key);
      if (before == forwardStarts.begin())
        continue;
      --before;
      const auto &seen = before->first.second;
      if (before->first.first != op ||
          static_cast<int64_t>(seen.size()) <= level ||
          !std::equal(at.begin(), at.begin() + level, seen.begin()))
        continue;
      llvm::SmallVector<int64_t, 2> moved(at);
      moved[level] = seen[level];
      return forwardStartAt(op, moved, loops) +
             (at[level] - seen[level]) * *period;
    }
    // Otherwise the latest simulated iteration before this one.
    key.second.assign(at.begin(), at.end());
    auto before = forwardStarts.lower_bound(key);
    if (before == forwardStarts.begin())
      return 0;
    --before;
    if (before->first.first != op || at.empty())
      return 0;
    return before->second;
  }

  /// How many of the first iterations of `loop`, the one the backward pass
  /// is in, an extrapolation from iteration `k` moving each op by
  /// `shiftOf(op)` each would misplace: those holding an op kept at its
  /// forward start that the forward pass started off that line.
  int64_t pinnedHead(mlir::scf::ForOp loop, int64_t k,
                     llvm::function_ref<int64_t(mlir::Operation *)> shiftOf) {
    size_t depth = iteration.size() - 1;
    llvm::SmallVector<int64_t, 2> saved = iteration;
    int64_t head = 0;
    for (mlir::Operation *op : pinned) {
      if (!loop->isProperAncestor(op))
        continue;
      std::pair<mlir::Operation *, llvm::SmallVector<int64_t, 2>> key{
          op,
          llvm::SmallVector<int64_t, 2>(saved.begin(), saved.begin() + depth)};
      for (auto it = forwardStarts.lower_bound(key);
           it != forwardStarts.end() && it->first.first == op; ++it) {
        const auto &at = it->first.second;
        if (at.size() <= depth ||
            !std::equal(saved.begin(), saved.begin() + depth, at.begin()) ||
            at[depth] >= k)
          break;
        iteration.assign(at.begin(), at.end());
        iteration[depth] = k;
        if (it->second != forwardStartOf(op) + (k - at[depth]) * shiftOf(op))
          head = std::max(head, at[depth] + 1);
      }
    }
    iteration = std::move(saved);
    return head;
  }

  void backwardSubmit(classical::SubmitOp submit) {
    mlir::Operation *op = submit;
    int64_t t = timelineOf(op);
    int64_t duration = durationOf(op), budget = budgetOf(op);
    int64_t start;
    // A call that holds every line keeps its forward time, as a blocking
    // wait does: what follows starts after its answer is due.
    bool kept = blocksOnCalls || op->hasAttr("classical.eager");
    llvm::ArrayRef<int64_t> held;
    if (holding()) {
      held = heldBy(submit);
      kept = kept || !held.empty();
    }
    if (kept) {
      start = forwardStartOf(op);
      pinned.insert(op);
    } else {
      int64_t link = late(submit.getUpdatedLink());
      if (ownOrder(op)) {
        auto it = linkLate.find(linkOf(op));
        link = it == linkLate.end() ? endOf(t) : it->second;
      }
      int64_t end = std::min(link, ceilings[t]);
      int64_t needed = late(submit.getFuture());
      if (needed != kNever)
        end = std::min(end, needed - budget);
      start = end - duration;
    }
    start = takeSlotBackward(op, start, duration + budget, kept);
    record(op, start);
    armStart = std::min(armStart, start);
    for (mlir::Value operand : op->getOperands())
      bound(operand, start);
    if (blocksOnCalls)
      ceilings[t] = std::min(ceilings[t], start);
    for (int64_t c : held)
      heldCeilings[c] = std::min(heldCeilings[c], start);
    if (ownOrder(op)) {
      auto [it, inserted] = linkLate.try_emplace(linkOf(op), start);
      if (!inserted)
        it->second = std::min(it->second, start);
    }
  }

  void backwardWait(classical::WaitOp wait) {
    int64_t t = timelineOf(wait);
    int64_t start = kNever;
    for (mlir::Value value : wait.getValues())
      start = std::min(start, late(value));
    if (start == kNever || wait->hasAttr("classical.blocking")) {
      start = forwardStartOf(wait);
      pinned.insert(wait);
    }
    // A timeline's share of a link holds only the requests it times; an
    // answer it took late would hold its slot after its due time.
    else if (isolated()) {
      start = std::min(start, forwardStartOf(wait));
      pinned.insert(wait);
    }
    start = std::min(start, ceilings[t]);
    // An eager request keeps its forward time, so where lanes or slots
    // handed out in reverse moved what its answer decides earlier, the
    // answer would be taken before it is due.
    if (auto submit = wait.getFuture().getDefiningOp<classical::SubmitOp>();
        submit && submit->hasAttr("classical.eager") &&
        start < forwardStartOf(submit) + durationOf(submit) + budgetOf(submit))
      answeredEarly = true;
    record(wait, start);
    armStart = std::min(armStart, start);
    bound(wait.getFuture(), start);
    if (wait->hasAttr("classical.blocking"))
      ceilings[t] = std::min(ceilings[t], start);
  }

  void bound(mlir::Value value, int64_t time) {
    auto [it, inserted] = lateAt.try_emplace(value, time);
    if (!inserted)
      it->second = std::min(it->second, time);
  }

  mlir::LogicalResult backwardBlock(mlir::Block &block) {
    for (mlir::Operation &op : llvm::reverse(block.without_terminator()))
      if (mlir::failed(backwardOp(&op)))
        return mlir::failure();
    return mlir::success();
  }

  mlir::LogicalResult backwardOp(mlir::Operation *op) {
    if (classical) {
      if (auto submit = mlir::dyn_cast<classical::SubmitOp>(op)) {
        backwardSubmit(submit);
        return mlir::success();
      }
      if (auto wait = mlir::dyn_cast<classical::WaitOp>(op)) {
        backwardWait(wait);
        return mlir::success();
      }
    }
    if (classical && op->hasAttr("classical.variable")) {
      // A variable wait keeps its forward time; nothing before it ends later.
      int64_t start = forwardStartOf(op);
      pinned.insert(op);
      record(op, start);
      armStart = std::min(armStart, start);
      bound(op->getOperand(0), start);
      int64_t &ceiling = ceilings[timelineOf(op)];
      ceiling = std::min(ceiling, start);
      return mlir::success();
    }
    if (isTimedOp(op)) {
      int64_t t = timelineOf(op);
      int64_t duration = durationOf(op);
      int64_t end = late(op->getResult(0));
      if (classical)
        end = std::min(end, ceilings[t]);
      if (holding())
        end = std::min(end, heldCeilingOf(op));
      bool isReadout = mlir::isa<pulse::ReadoutOp>(op);
      int64_t owner = several() ? controllerOf(op) : kAnyOwner;
      if (isReadout) {
        int64_t latency =
            several() ? controllers.own(owner) : resources.readoutLatency;
        if (resources.enabled)
          end -= latency;
        int64_t needed = late(mlir::cast<pulse::ReadoutOp>(op).getResult());
        if (needed != kNever)
          end = std::min(end, needed - latency);
      }
      if (mlir::Value tone = toneResult(op))
        end = std::min(end, late(tone));
      int64_t step = 1;
      if constexpr (Gridded)
        step = grid.of(op->getOperand(0), lines);
      if (resources.enabled && !mlir::isa<pulse::WaitOp>(op)) {
        auto &lanes = isReadout ? readoutLanes[t] : driveLanes[t];
        ++(isReadout ? readoutUses : driveUses)[t];
        if (several()) {
          auto &owners = isReadout ? readoutOwners[t] : driveOwners[t];
          size_t best = 0;
          int64_t free = -kNever;
          for (size_t i = 0; i < lanes.size(); ++i) {
            int64_t at = lanes[i] - laneGap(owners[i], owner);
            if (at > free) {
              free = at;
              best = i;
            }
          }
          end = std::min(end, free);
          if (step > 1)
            end = pulse::LineGrid::roundDown(end - duration, step) + duration;
          lanes[best] = end - duration - resources.switchPenalty;
          owners[best] = owner;
        } else {
          auto lane = std::max_element(lanes.begin(), lanes.end());
          end = std::min(end, *lane);
          if (step > 1)
            end = pulse::LineGrid::roundDown(end - duration, step) + duration;
          *lane = end - duration - resources.switchPenalty;
        }
      }
      int64_t start = end - duration;
      if (step > 1)
        start = pulse::LineGrid::roundDown(start, step);
      record(op, start);
      if (owner >= 0)
        armStarts[owner] = std::min(armStarts[owner], start);
      else
        armStart = std::min(armStart, start);
      bound(op->getOperand(0), start);
      if (mlir::Value tone = toneOperand(op))
        bound(tone, start);
      if (classical) {
        for (mlir::Value operand : op->getOperands())
          bound(operand, start);
        if (several())
          for (mlir::Value measurement : valueMeasurements(op))
            bound(measurement,
                  start - controllers.extra(readerOf(measurement), owner));
      }
      return mlir::success();
    }
    if (auto block = mlir::dyn_cast<pulse::ScheduleBlockOp>(op)) {
      backwardScheduleBlock(block);
      return mlir::success();
    }
    if (auto sync = mlir::dyn_cast<pulse::SyncOp>(op)) {
      int64_t time = endOf(timelineOf(op));
      for (mlir::Value result : sync.getResults())
        time = std::min(time, late(result));
      if constexpr (Gridded)
        time = pulse::LineGrid::roundDown(time, syncGrid(sync));
      for (mlir::Value line : sync.getLines())
        bound(line, time);
      if (!several() || sync.getLines().empty())
        armStart = std::min(armStart, time);
      else
        for (mlir::Value line : sync.getLines()) {
          int64_t owner = controllerOfLine(line);
          if (owner >= 0)
            armStarts[owner] = std::min(armStarts[owner], time);
          else
            armStart = std::min(armStart, time);
        }
      return mlir::success();
    }
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op))
      return backwardLoop(loop);
    if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op))
      return backwardIf(branch);
    int64_t time = kNever;
    for (mlir::Value result : op->getResults())
      if (classical || isTimedType(result.getType()))
        time = std::min(time, late(result));
    if (time != kNever)
      for (mlir::Value operand : op->getOperands())
        if (classical || isTimedType(operand.getType()))
          bound(operand, time);
    return mlir::success();
  }

  /// Start the block as late as every result's consumer allows.
  void backwardScheduleBlock(pulse::ScheduleBlockOp block) {
    BlockShape shape = shapeOf(block);
    auto inputs = block.getInputs();
    int64_t t = timelineOf(block);
    // A block before a run-time wait ends by its start, like any op.
    int64_t start = endOf(t) - shape.length;
    if (classical)
      start = std::min(start, ceilings[t] - shape.length);
    if (holding())
      start = std::min(start, *llvm::min_element(heldCeilings) - shape.length);
    for (auto [result, offset, source] :
         llvm::zip(block.getOutputs(), shape.out, shape.source)) {
      int64_t needed = late(result);
      if (needed == kNever)
        continue;
      if (offset >= 0)
        start = std::min(start, needed - offset);
      else
        bound(inputs[source], needed);
    }
    if constexpr (Gridded)
      start = pulse::LineGrid::roundDown(start, grid.lcm());
    if (classical)
      start = takeBlockSlots(block, start, /*backward=*/true);
    record(block, start);
    armStart = std::min(armStart, start);
    for (auto [input, offset] : llvm::zip(inputs, shape.in))
      if (offset >= 0)
        bound(input, start + offset);
  }

  mlir::LogicalResult backwardLoop(mlir::scf::ForOp loop) {
    int64_t count;
    if (mlir::failed(tripCount(loop, count)))
      return mlir::failure();
    auto args = loop.getRegionIterArgs();
    auto yield =
        mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
    llvm::SmallVector<mlir::Value> bodyValues;
    // Whether the body bounds a value from before the loop.
    bool readsOutside = false;
    loop.getBody()->walk([&](mlir::Operation *inner) {
      for (mlir::Value result : inner->getResults())
        bodyValues.push_back(result);
      for (mlir::Region &region : inner->getRegions())
        for (mlir::Block &nested : region)
          for (mlir::Value arg : nested.getArguments())
            bodyValues.push_back(arg);
      for (mlir::Value operand : inner->getOperands())
        if ((isTimedType(operand.getType()) ||
             (classical && !mlir::matchPattern(operand, mlir::m_Constant()))) &&
            !loop->isAncestor(operand.getParentRegion()->getParentOp()))
          readsOutside = true;
    });
    llvm::SmallVector<Carried> kinds;
    llvm::SmallVector<int64_t> carried;
    for (mlir::Value result : loop.getResults()) {
      kinds.push_back(
          carriedKind(result.getType(), isLineType(result.getType())));
      carried.push_back(late(result));
    }

    iteration.push_back(0);
    std::optional<int64_t> lastShift;
    int64_t laneTries = laneTriesPerLoop();
    const LoopGroups *groups = nullptr;
    bool grouped = false;
    std::optional<Unshifted> unshifted;
    std::optional<GroupShifts> lastGroups;
    Track track;
    bool simulatedFirst = false;
    RecentStates recent;
    // Lanes that repeat only every few iterations regroup a loop of calls
    // (a loop without is summarized by the controller check as it is).
    bool regroupLanes = resources.enabled && mayRegroup && holdsCalls(loop);
    llvm::SmallVector<LaneState, 0> laneHistory;
    LoopLinks links = linksOf(loop);
    for (int64_t k = count - 1; k >= 0; --k) {
      iteration.back() = k;
      llvm::SmallVector<HeldSlots, 1> slotsBefore;
      if (!links.empty() && !lanesBound)
        slotsBefore = heldSlots(links, true);
      if (!answerGroups.empty())
        settleGroups(loop);
      for (mlir::Value value : bodyValues)
        lateAt.erase(value);
      for (mlir::Value arg : args)
        lateAt.erase(arg);
      for (auto [value, time] : llvm::zip(yield.getOperands(), carried))
        bound(value, time);
      auto ceilingsBefore = ceilings;
      auto heldCeilingsBefore = heldCeilings;
      auto linksBefore = linkLate;
      // Only an iteration moving as the one before can be steady.
      LaneState lanesBefore;
      if (lastShift && laneTries > 0)
        lanesBefore = laneState();
      if (regroupLanes)
        pushLanes(laneHistory);
      LaneState groupLanes;
      if (resources.enabled && groupsApply())
        groupLanes = laneState();
      if (groups) {
        track.earliest.assign(groups->count, kNever);
        tracks.push_back(&track);
      } else if (unshifted) {
        track.starts.clear();
        tracks.push_back(&track);
      }
      bool tracked = groups || unshifted;
      // The earliest starts of this iteration, in the arm the loop is in.
      int64_t armBefore = std::exchange(armStart, kNever);
      llvm::SmallVector<int64_t, 4> armStartsBefore = armStarts;
      std::fill(armStarts.begin(), armStarts.end(), kNever);
      mlir::LogicalResult simulated = backwardBlock(*loop.getBody());
      if (tracked)
        tracks.pop_back();
      if (mlir::failed(simulated))
        return mlir::failure();
      int64_t iterationArmStart = armStart;
      llvm::SmallVector<int64_t, 4> iterationArmStarts = armStarts;
      armStart = std::min(armBefore, armStart);
      for (auto [start, saved] : llvm::zip(armStarts, armStartsBefore))
        start = std::min(start, saved);
      llvm::SmallVector<int64_t> next;
      for (mlir::Value arg : args)
        next.push_back(late(arg));

      int64_t shift;
      bool shifted = uniformShift(carried, next, kinds, shift);
      bool repeated = shifted && lastShift == shift;
      bool steady =
          !lanesBound && repeated && shift <= 0 &&
          (!Gridded || shift % grid.lcm() == 0) &&
          slotsHoldSteadyState(loop, -shift) &&
          (!resources.enabled || lanesSteady(lanesBefore, shift, laneTries)) &&
          slotsMoved(links, slotsBefore, true,
                     [&](llvm::ArrayRef<mlir::Operation *>) { return shift; });
      lastShift = shifted ? std::optional<int64_t>(shift) : std::nullopt;
      // Lines that move by periods of their own repeat too, group by group.
      std::optional<GroupShifts> nowGroups;
      if (!shifted && k > 0 && groupsApply()) [[unlikely]] {
        if (!grouped && !unshifted) {
          unshifted = Unshifted{carried, next};
        } else if (!grouped) {
          grouped = true;
          groups = groupsOf(loop);
          if (groups && groups->mixedWaits && groups->linkBound)
            groups = groups->linkBound.get();
          track.groups = groups;
          if (groups) {
            lastGroups = groupShifts(*groups, unshifted->before,
                                     unshifted->after, kinds, true);
            track.earliest.assign(groups->count, kNever);
            for (auto [op, start] : track.starts)
              if (groups->ofOp.contains(op)) {
                int64_t &earliest =
                    track.earliest[groups->groupOf(op, /*backward=*/true)];
                earliest = std::min(earliest, start);
              }
          }
          track.starts.clear();
          unshifted.reset();
        }
        if (groups)
          nowGroups = groupShifts(*groups, carried, next, kinds, true);
      } else if (!grouped) {
        unshifted.reset();
      }
      // Each group that moves has an op to move: the syncs of one without
      // would not start where its ops do.
      bool mixed = nowGroups && !track.earliest.empty() &&
                   lastGroups == nowGroups &&
                   groupsSteady(*groups, *nowGroups, -1);
      if (mixed)
        for (auto [shift, earliest] : llvm::zip(*nowGroups, track.earliest))
          if (shift.value_or(0) != 0 && earliest == kNever)
            mixed = false;
      mixed = mixed && slotsHoldGroups(loop, *groups, *nowGroups, -1) &&
              groupLanesMoved(groupLanes, *groups, *nowGroups) &&
              slotsMoved(links, slotsBefore, true,
                         [&](llvm::ArrayRef<mlir::Operation *> ops) {
                           return requestsShift(loop, *groups, *nowGroups, ops,
                                                true);
                         });
      if (mixed && k > 0 && !pinned.empty()) {
        const GroupShifts &shifts = *nowGroups;
        int64_t head = pinnedHead(loop, k, [&](mlir::Operation *op) {
          return groups->ofOp.contains(op)
                     ? shifts[groups->groupOf(op, /*backward=*/true)].value_or(
                           0)
                     : 0;
        });
        if (head > 0 && mayPeelHead) {
          headPeel = Regroup{loop, head};
          break;
        }
        mixed = head == 0;
      }
      bool groupsRepeated = nowGroups && lastGroups == nowGroups;
      lastGroups = std::move(nowGroups);
      llvm::SmallVector<int64_t> previous = std::move(carried);
      carried = std::move(next);
      if (mixed) [[unlikely]] {
        // A measurement from before the loop is bounded by every
        // iteration's branches, the first iteration's the earliest: skip to
        // it and simulate it.
        bool first = !groups->outside.empty();
        int64_t copies = first ? k - 1 : k;
        auto distances = groupDistances(*lastGroups, copies);
        if (distances && extrapolateGroups(carried, previous, kinds, *groups,
                                           true, *distances)) {
          moveGroupLanes(groupLanes, *groups, *distances);
          moveSlots(links, true, [&](llvm::ArrayRef<mlir::Operation *> ops) {
            return *requestsShift(loop, *groups, *lastGroups, ops, true) *
                   copies;
          });
          // The iterations not simulated start their ops earlier, by
          // their group's distance in the first, and so the arm it is in.
          for (auto [earliest, distance] :
               llvm::zip(track.earliest, *distances))
            lowerArmStart(armStart, earliest, distance);
          extrapolations.push_back(
              {loop, 0, opPeriods(*groups, *lastGroups, -1)});
          if (first) {
            simulatedFirst = true;
            k = 1;
            continue;
          }
          break;
        }
      }
      // Ops kept at their forward starts repeat only where the forward
      // pass's did: peel the first iterations that do not.
      if (steady && k > 0 && !pinned.empty()) {
        int64_t head =
            pinnedHead(loop, k, [&](mlir::Operation *) { return shift; });
        if (head > 0 && mayPeelHead) {
          headPeel = Regroup{loop, head};
          break;
        }
        steady = head == 0;
      }
      // A value from before the loop is bounded by every iteration, the
      // first the earliest: skip to it and simulate it.
      if (steady && k > (readsOutside ? 1 : 0)) {
        int64_t distance = (readsOutside ? k - 1 : k) * shift;
        // The first iteration skipped starts the arm earliest.
        lowerArmStart(armStart, iterationArmStart, distance);
        for (auto [start, inIteration] :
             llvm::zip(armStarts, iterationArmStarts))
          lowerArmStart(start, inIteration, distance);
        extrapolate(carried, previous, kinds, distance);
        moveSlots(links, true,
                  [&](llvm::ArrayRef<mlir::Operation *>) { return distance; });
        for (auto [now, before] : llvm::zip(ceilings, ceilingsBefore))
          if (now != before)
            now += distance;
        moveChanged(heldCeilings, heldCeilingsBefore, distance);
        for (auto &[key, time] : linkLate) {
          auto before = linksBefore.find(key);
          if (before == linksBefore.end() || before->second != time)
            time += distance;
        }
        if (resources.enabled)
          moveLanes(lanesBefore, distance);
        extrapolations.push_back({loop, -shift, nullptr});
        if (readsOutside) {
          simulatedFirst = true;
          k = 1;
          continue;
        }
        break;
      }
      if (regroup || headPeel)
        break;
      if (count - 1 - k >= 2 && !repeated && !groupsRepeated && mayRegroup) {
        noteRecent(recent, loop, carried, kinds, k, -1);
        if (regroup)
          break;
      }
      if (repeated && !lanesBound && laneTries <= 0 && !laneHistory.empty() &&
          !lanesMoved(laneHistory.back(), shift)) {
        noteRecentLanes(laneHistory, loop, shift, k);
        if (regroup)
          break;
      }
    }
    // The first iteration was simulated for what it bounds before the loop;
    // its samples would leave a gap the extrapolation already spans.
    if (simulatedFirst) [[unlikely]] {
      size_t depth = iteration.size() - 1;
      llvm::ArrayRef<int64_t> outer =
          llvm::ArrayRef(iteration).take_front(depth);
      loop.getBody()->walk([&](mlir::Operation *op) {
        auto found = samples.find(op);
        if (found == samples.end())
          return;
        auto &list = found->second;
        while (!list.empty() && list.back().iteration.size() > depth &&
               list.back().iteration[depth] == 0 &&
               llvm::ArrayRef(list.back().iteration).take_front(depth) ==
                   outer) {
          if (list.back().start < 0)
            skippedNegative = true;
          list.pop_back();
        }
      });
    }
    iteration.pop_back();
    for (auto [init, time] : llvm::zip(loop.getInitArgs(), carried))
      bound(init, time);
    return mlir::success();
  }

  mlir::LogicalResult backwardIf(mlir::scf::IfOp branch) {
    int64_t savedArmStart = armStart;
    armStart = kNever;
    auto savedArmStarts = armStarts;
    std::fill(armStarts.begin(), armStarts.end(), kNever);
    auto savedDrive = driveLanes, savedReadout = readoutLanes;
    auto savedDriveOwners = driveOwners, savedReadoutOwners = readoutOwners;
    auto savedLinks = linkLate;
    for (auto [value, result] :
         llvm::zip(branch.thenYield().getOperands(), branch.getResults()))
      bound(value, late(result));
    if (mlir::failed(backwardBlock(*branch.thenBlock())))
      return mlir::failure();
    if (!branch.getElseRegion().empty()) {
      auto thenDrive = driveLanes, thenReadout = readoutLanes;
      auto thenDriveOwners = std::move(driveOwners);
      auto thenReadoutOwners = std::move(readoutOwners);
      auto thenLinks = std::move(linkLate);
      driveLanes = savedDrive;
      readoutLanes = savedReadout;
      driveOwners = std::move(savedDriveOwners);
      readoutOwners = std::move(savedReadoutOwners);
      linkLate = std::move(savedLinks);
      for (auto [value, result] :
           llvm::zip(branch.elseYield().getOperands(), branch.getResults()))
        bound(value, late(result));
      if (mlir::failed(backwardBlock(*branch.elseBlock())))
        return mlir::failure();
      mergeLanes(driveLanes, thenDrive, /*later=*/false);
      mergeLanes(readoutLanes, thenReadout, /*later=*/false);
      mergeOwners(driveOwners, thenDriveOwners);
      mergeOwners(readoutOwners, thenReadoutOwners);
      for (auto &[key, time] : thenLinks) {
        auto [it, inserted] = linkLate.try_emplace(key, time);
        if (!inserted)
          it->second = std::min(it->second, time);
      }
    }
    if (several()) {
      // Each outcome reaches every controller of an arm before its ops.
      int64_t earliest = armStart;
      for (int64_t start : armStarts)
        earliest = std::min(earliest, start);
      if (earliest != kNever) {
        AnswerGroup &group = answerGroup(branch.getCondition());
        for (auto [reader, bounded] : llvm::zip(group.readers, group.latest)) {
          int64_t latest = armStart == kNever
                               ? kNever
                               : armStart - controllers.extra(reader, -1);
          for (auto [c, start] : llvm::enumerate(armStarts))
            if (start != kNever)
              latest = std::min(
                  latest,
                  start - controllers.extra(reader, static_cast<int64_t>(c)));
          bounded = std::min(bounded, latest);
        }
        if (classical)
          bound(branch.getCondition(), earliest);
      }
    } else if (armStart != kNever) {
      for (mlir::Value measurement :
           conditionMeasurements(branch.getCondition(), false))
        bound(measurement, armStart);
      if (classical)
        bound(branch.getCondition(), armStart);
    }
    if (yieldsAfterCondition(branch)) {
      int64_t needed = kNever;
      for (mlir::Value result : branch.getResults())
        needed = std::min(needed, late(result));
      if (needed != kNever) {
        if (several()) {
          AnswerGroup &group = answerGroup(branch.getCondition());
          for (auto [reader, bounded] : llvm::zip(group.readers, group.latest))
            bounded = std::min(bounded, needed - controllers.extra(reader, -1));
        } else
          for (mlir::Value measurement :
               conditionMeasurements(branch.getCondition(), false))
            bound(measurement, needed);
        bound(branch.getCondition(), needed);
      }
    }
    armStart = std::min(savedArmStart, armStart);
    for (auto [start, saved] : llvm::zip(armStarts, savedArmStarts))
      start = std::min(start, saved);
    return mlir::success();
  }

  mlir::Region *region;
  Resources resources;
  llvm::DenseMap<mlir::Value, int64_t> readyAt;
  llvm::DenseMap<mlir::Value, int64_t> lateAt;
  /// Per branch condition, on several controllers: answerGroup, and the
  /// groups each measurement is in (group, slot of its reader);
  /// conditionReadyOf.
  std::vector<AnswerGroup> answerGroups;
  llvm::DenseMap<mlir::Value, unsigned> groupOf;
  llvm::DenseMap<mlir::Value,
                 llvm::SmallVector<std::pair<unsigned, unsigned>, 1>>
      memberOf;
  llvm::DenseMap<mlir::Value, ConditionReady> conditionReady;
  /// Under rcp, the drive and readout lanes of each timeline.
  Lanes driveLanes, readoutLanes;
  llvm::SmallVector<int64_t, 2> iteration;
  /// No op starts before the condition of the branch it is in.
  int64_t floor = 0;
  int64_t iterationEnd = 0;
  int64_t armStart = kNever;
  // Classical calls.
  bool classical = false;
  /// The lines' timelines, when there is more than one.
  const Timelines *lanesPer = nullptr;
  pulse::AllocationTracer trace;
  llvm::DenseMap<mlir::Operation *, int64_t> timelineCache;
  /// Per timeline: no op starts before its last run-time wait (forward),
  /// where its ops end (in the current loop iteration: -1 for none), and
  /// under ALAP nothing before a run-time wait ends after it.
  llvm::SmallVector<int64_t, 1> floors, ends, iterationEnds, ceilings;
  /// For submissions with their own order (`ownOrder`), when their link is
  /// free (forward) or must be free from (backward), in that order.
  std::map<LinkKey, int64_t> linkReady, linkLate;
  /// Request slots per link and timeline.
  llvm::DenseMap<LinkKey, LinkSlots> linkLanes;
  llvm::DenseMap<LinkKey, int64_t> linkBudgets;
  int64_t laneReset = 0;
  /// A link's slots delayed a request: loop iterations may differ, so they
  /// are all simulated.
  bool lanesBound = false;
  llvm::DenseMap<mlir::Operation *, std::shared_ptr<const LoopGroups>>
      groupCache;
  /// The loops whose ops `record` notes the earliest start of, innermost
  /// last.
  llvm::SmallVector<Track *, 2> tracks;
  /// The requests of each block, by local start and by local end.
  llvm::DenseMap<mlir::Operation *, std::pair<llvm::SmallVector<BlockRequest>,
                                              llvm::SmallVector<BlockRequest>>>
      blockRequests;
  std::map<std::pair<mlir::Operation *, llvm::SmallVector<int64_t, 2>>, int64_t>
      forwardStarts;
  /// The ops the backward pass placed at their forward starts.
  llvm::SmallPtrSet<mlir::Operation *, 8> pinned;
  llvm::SmallVector<Extrapolation> forwardExtrapolations;
  // Lines whose controllers start operations only every few program cycles.
  pulse::LineGrid grid;
  [[no_unique_address]] std::conditional_t<Gridded, pulse::AllocationTracer,
                                           NoLineTrace> lines;
  /// Under rcp on several controllers: which plays each line, and what
  /// crossing between two costs.
  pulse::Controllers controllers;
  /// When calls hold controllers (`holding`), which plays each line under
  /// any policy, and the controllers each call holds.
  pulse::Controllers owners;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<int64_t, 2>> heldCalls;
  /// In a branch on readouts, per controller: when their outcomes reach it
  /// (forward; empty outside one).
  llvm::SmallVector<int64_t, 4> armFloors;
  /// Per controller, the earliest start of an op of it in the arm being
  /// scheduled; `armStart` has those of no one controller (backward).
  llvm::SmallVector<int64_t, 4> armStarts;
  /// The controller of each lane's last op.
  Lanes driveOwners, readoutOwners;
  /// Per timeline, how many ops have taken its drive and readout lanes.
  llvm::SmallVector<int64_t, 1> driveUses, readoutUses;
  bool lanesKnown = false, lanesInterchangeable = false;
};

// ── Writing the schedule ─────────────────────────────────────────────────

/// Why an op's samples are not affine in its iterations: the iterations of
/// `loop` to peel off its front and back, or none to unroll it fully.
struct NotAffine {
  mlir::scf::ForOp loop;
  int64_t peelFront = 0;
  int64_t peelBack = 0;
};

/// The iterations to peel off `loop` so that of the starts `line` it plays
/// one op at, by iteration, one run of equal steps between consecutive
/// iterations is left: the one over the most iterations, counting for the
/// run at an end past which the scheduler extrapolated the iterations it
/// did not simulate, as they repeat it; any other run has two steps or
/// more. None when there is no such run.
static NotAffine peelAround(mlir::scf::ForOp loop,
                            llvm::ArrayRef<std::pair<int64_t, int64_t>> line,
                            int64_t trips) {
  bool before = line.front().first > 0, after = line.back().first < trips - 1;
  if (before && after)
    return {loop};
  size_t first = 0, last = 0;
  int64_t longest = 0;
  bool extended = false;
  for (size_t from = 0; from + 1 < line.size();) {
    size_t to = from + 1;
    if (line[to].first != line[from].first + 1) {
      from = to;
      continue;
    }
    int64_t step = line[to].second - line[from].second;
    while (to + 1 < line.size() && line[to + 1].first == line[to].first + 1 &&
           line[to + 1].second - line[to].second == step)
      ++to;
    // The run at the end extrapolated past spans the iterations past it too.
    bool extends = (before && from == 0) || (after && to == line.size() - 1);
    int64_t span = (extends && before ? 0 : line[from].first);
    span = (extends && after ? trips - 1 : line[to].first) - span;
    if (extends ? span >= 1 && span >= longest : span >= 2 && span > longest) {
      first = from;
      last = to;
      longest = span;
      extended = extends;
    }
    from = to;
  }
  if (longest == 0)
    return {loop};
  return {loop, extended && before ? 0 : line[first].first,
          extended && after ? 0 : trips - 1 - line[last].first};
}

static std::optional<NotAffine>
writeTiming(mlir::Operation *op, mlir::Operation *root,
            llvm::ArrayRef<Sample> samples,
            llvm::ArrayRef<Extrapolation> extrapolations) {
  auto loops = pulse::enclosingLoops(op, root);
  int64_t duration = durationOf(op);
  auto write = [&](std::optional<int64_t> start,
                   llvm::ArrayRef<int64_t> periods) {
    pulse::setTiming(op, start, duration, periods);
  };
  if (samples.empty()) {
    // Never executed (inside a zero-trip loop).
    write(std::nullopt, {});
    return std::nullopt;
  }
  if (loops.empty() && samples.size() == 1) {
    write(samples.front().start, {});
    return std::nullopt;
  }

  llvm::DenseMap<llvm::ArrayRef<int64_t>, int64_t> at;
  for (const Sample &sample : samples)
    at[sample.iteration] = sample.start;

  // The per-iteration shift along each loop, from any two simulated
  // neighbors, else from the extrapolation that skipped them.
  llvm::SmallVector<int64_t, 2> periods(loops.size(), 0);
  for (auto [level, loop] : llvm::enumerate(loops)) {
    bool found = false;
    for (const Sample &sample : samples) {
      llvm::SmallVector<int64_t, 2> next(sample.iteration.begin(),
                                         sample.iteration.end());
      ++next[level];
      auto it = at.find(next);
      if (it != at.end()) {
        periods[level] = it->second - sample.start;
        found = true;
        break;
      }
    }
    for (const Extrapolation &e : extrapolations)
      if (e.loop == loop && !found)
        periods[level] = e.periodOf(op).value_or(0);
  }
  int64_t base = samples.front().start;
  for (auto [k, period] : llvm::zip(samples.front().iteration, periods))
    base -= k * period;

  for (const Sample &sample : samples) {
    int64_t expected = base;
    for (auto [k, period] : llvm::zip(sample.iteration, periods))
      expected += k * period;
    if (expected == sample.start)
      continue;
    // Find the innermost loop along which this sample breaks the pattern,
    // and whether only the loop's first or last iteration differs.
    for (int64_t level = static_cast<int64_t>(loops.size()) - 1; level >= 0;
         --level) {
      auto trips = pulse::constantTripCount(loops[level]).value_or(0);
      // The simulated iterations along this loop, from the samples: an
      // extrapolated loop has few of its trips among them.
      llvm::SmallVector<std::pair<int64_t, int64_t>> line;
      for (const Sample &other : samples) {
        bool along = true;
        for (auto [i, k] : llvm::enumerate(other.iteration))
          if (static_cast<int64_t>(i) != level && k != sample.iteration[i])
            along = false;
        if (along && other.iteration[level] >= 0 &&
            other.iteration[level] < trips)
          line.push_back({other.iteration[level], at.lookup(other.iteration)});
      }
      llvm::sort(line);
      line.erase(std::unique(line.begin(), line.end(),
                             [](const auto &a, const auto &b) {
                               return a.first == b.first;
                             }),
                 line.end());
      llvm::SmallVector<int64_t> steps;
      for (size_t i = 1; i < line.size(); ++i)
        steps.push_back(line[i].second - line[i - 1].second);
      if (llvm::all_equal(steps) &&
          (steps.empty() || steps[0] == periods[level]))
        continue;
      bool front = line.front().first == 0 && steps.size() >= 2 &&
                   llvm::all_equal(llvm::ArrayRef(steps).drop_front());
      bool back = line.back().first == trips - 1 && steps.size() >= 2 &&
                  llvm::all_equal(llvm::ArrayRef(steps).drop_back());
      if (front || back)
        return NotAffine{loops[level], front, !front && back};
      return peelAround(loops[level], line, trips);
    }
    return NotAffine{loops.back()};
  }
  for (const Extrapolation &e : extrapolations)
    for (auto [level, loop] : llvm::enumerate(loops))
      if (e.loop == loop && periods[level] != e.periodOf(op))
        return NotAffine{loop};

  write(base, periods);
  return std::nullopt;
}

static mlir::Value indexConstant(mlir::OpBuilder &builder, mlir::Location loc,
                                 int64_t value) {
  return mlir::arith::ConstantIndexOp::create(builder, loc, value);
}

/// Execute the first iteration of `loop` before it.
static void peelFront(mlir::scf::ForOp loop) {
  int64_t lb = *pulse::constantIndex(loop.getLowerBound());
  int64_t step = *pulse::constantIndex(loop.getStep());
  mlir::OpBuilder builder(loop);
  mlir::IRMapping map;
  map.map(loop.getInductionVar(), loop.getLowerBound());
  for (auto [arg, init] :
       llvm::zip(loop.getRegionIterArgs(), loop.getInitArgs()))
    map.map(arg, init);
  for (mlir::Operation &op : loop.getBody()->without_terminator())
    builder.clone(op, map);
  llvm::SmallVector<mlir::Value> next;
  for (mlir::Value value : loop.getBody()->getTerminator()->getOperands())
    next.push_back(map.lookupOrDefault(value));
  loop.getInitArgsMutable().assign(next);
  loop.setLowerBound(indexConstant(builder, loop.getLoc(), lb + step));
  if (pulse::constantTripCount(loop) == 0) {
    loop.replaceAllUsesWith(next);
    loop.erase();
  }
}

/// Execute the last iteration of `loop` after it.
static void peelBack(mlir::scf::ForOp loop) {
  int64_t lb = *pulse::constantIndex(loop.getLowerBound());
  int64_t step = *pulse::constantIndex(loop.getStep());
  int64_t trips = *pulse::constantTripCount(loop);
  mlir::OpBuilder builder(loop->getContext());
  builder.setInsertionPointAfter(loop);
  mlir::IRMapping map;
  map.map(loop.getInductionVar(),
          indexConstant(builder, loop.getLoc(), lb + (trips - 1) * step));
  for (auto [arg, result] :
       llvm::zip(loop.getRegionIterArgs(), loop.getResults()))
    map.map(arg, result);
  llvm::SmallPtrSet<mlir::Operation *, 8> clones;
  for (mlir::Operation &op : loop.getBody()->without_terminator())
    clones.insert(builder.clone(op, map));
  for (auto [result, value] : llvm::zip(
           loop.getResults(), loop.getBody()->getTerminator()->getOperands())) {
    mlir::Value last = map.lookupOrDefault(value);
    result.replaceUsesWithIf(last, [&](mlir::OpOperand &use) {
      for (mlir::Operation *owner = use.getOwner(); owner;
           owner = owner->getParentOp())
        if (clones.contains(owner))
          return false;
      return true;
    });
  }
  mlir::OpBuilder bounds(loop);
  loop.setUpperBound(
      indexConstant(bounds, loop.getLoc(), lb + (trips - 1) * step));
  if (trips - 1 == 0) {
    loop.replaceAllUsesWith(loop.getInitArgs());
    loop.erase();
  }
}

/// Run `factor` iterations of `loop` in each of its iterations, executing
/// those left over after it. The loop has at least 2 x `factor` trips.
static void unrollBy(mlir::scf::ForOp loop, int64_t factor) {
  for (int64_t left = *pulse::constantTripCount(loop) % factor; left > 0;
       --left)
    peelBack(loop);
  int64_t step = *pulse::constantIndex(loop.getStep());
  mlir::Block *body = loop.getBody();
  mlir::Operation *yield = body->getTerminator();
  llvm::SmallVector<mlir::Operation *> ops;
  for (mlir::Operation &op : body->without_terminator())
    ops.push_back(&op);
  llvm::SmallVector<mlir::Value> yielded(yield->getOperands());
  llvm::SmallVector<mlir::Value> carried = yielded;
  mlir::OpBuilder builder(yield);
  mlir::Value iv = loop.getInductionVar();
  for (int64_t copy = 1; copy < factor; ++copy) {
    mlir::IRMapping map;
    if (!iv.use_empty())
      map.map(iv, mlir::arith::AddIOp::create(
                      builder, loop.getLoc(), iv,
                      indexConstant(builder, loop.getLoc(), copy * step)));
    for (auto [arg, value] : llvm::zip(loop.getRegionIterArgs(), carried))
      map.map(arg, value);
    for (mlir::Operation *op : ops)
      builder.clone(*op, map);
    for (auto [value, original] : llvm::zip(carried, yielded))
      value = map.lookupOrDefault(original);
  }
  yield->setOperands(carried);
  mlir::OpBuilder bounds(loop);
  loop.setStep(indexConstant(bounds, loop.getLoc(), step * factor));
}

static void unrollFully(mlir::scf::ForOp loop) {
  while (auto trips = pulse::constantTripCount(loop)) {
    if (*trips == 0) {
      loop.replaceAllUsesWith(loop.getInitArgs());
      loop.erase();
      return;
    }
    bool last = *trips == 1;
    peelFront(loop);
    if (last)
      return;
  }
}

/// A loop in which two ops on one physical line move by different periods
/// per iteration. Each op's timing is affine, but the line's timeline is not
/// one pattern repeated, which happens when the first iteration differs.
static std::optional<NotAffine> periodMismatch(mlir::Region &region,
                                               mlir::Operation *root) {
  std::optional<NotAffine> problem;
  pulse::AllocationTracer trace;
  region.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (problem || mlir::isa<pulse::ScheduleBlockOp>(op))
      return mlir::WalkResult::skip();
    auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op);
    if (!loop)
      return mlir::WalkResult::advance();
    size_t level = pulse::enclosingLoops(loop, root).size();
    llvm::DenseMap<mlir::Operation *, int64_t> periodOf;
    auto check = [&](mlir::Value line, int64_t period) {
      auto [it, inserted] = periodOf.try_emplace(trace(line), period);
      if (!inserted && it->second != period)
        problem = NotAffine{loop, 1};
    };
    forEachScheduled(loop.getRegion(), [&](mlir::Operation *timed) {
      llvm::ArrayRef<int64_t> periods = pulse::periodCycles(timed);
      if (problem || periods.size() <= level)
        return;
      if (mlir::isa<classical::WaitOp>(timed))
        return;
      if (!mlir::isa<pulse::ScheduleBlockOp>(timed))
        return check(timed->getOperand(0), periods[level]);
      for (mlir::Value input : timed->getOperands())
        if (isLineType(input.getType()))
          check(input, periods[level]);
    });
    return mlir::WalkResult::advance();
  });
  return problem;
}

// ── Schedule blocks ──────────────────────────────────────────────────────

/// The schedule blocks in `region` that no other block in it encloses, once
/// each line in it, outside those blocks, has one timeline
/// (checkOneAcquisition).
static mlir::FailureOr<llvm::SmallVector<pulse::ScheduleBlockOp>>
placeableBlocks(mlir::Region &region) {
  llvm::SmallVector<pulse::ScheduleBlockOp> blocks;
  mlir::WalkResult walk =
      region.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
        if (auto block = mlir::dyn_cast<pulse::ScheduleBlockOp>(op)) {
          blocks.push_back(block);
          return mlir::WalkResult::skip();
        }
        if (mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(op) &&
            mlir::failed(pulse::checkOneAcquisition(op)))
          return mlir::WalkResult::interrupt();
        return mlir::WalkResult::advance();
      });
  if (walk.wasInterrupted())
    return mlir::failure();
  return blocks;
}

/// The schedule blocks in `region` that no other block in it encloses.
static llvm::SmallVector<pulse::ScheduleBlockOp>
outermostBlocks(mlir::Region &region) {
  llvm::SmallVector<pulse::ScheduleBlockOp> blocks;
  region.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (auto block = mlir::dyn_cast<pulse::ScheduleBlockOp>(op)) {
      blocks.push_back(block);
      return mlir::WalkResult::skip();
    }
    return mlir::WalkResult::advance();
  });
  return blocks;
}

/// When a value from outside a block is first needed in it: the earliest
/// local start of a timed op that depends on it, or of a branch arm
/// conditioned on it.
class FirstUse {
public:
  explicit FirstUse(pulse::ScheduleBlockOp block) : block(block) {}

  int64_t of(mlir::Value value) {
    earliest = kNever;
    visited.clear();
    visitUses(value);
    return earliest;
  }

private:
  void visitUses(mlir::Value value) {
    for (mlir::OpOperand &use : value.getUses())
      visit(use);
  }

  void note(mlir::Operation *op) {
    if (std::optional<int64_t> start = pulse::startCycles(op))
      earliest = std::min(earliest, *start);
  }

  void visit(mlir::OpOperand &use) {
    mlir::Operation *op = use.getOwner();
    if (op->getParentOp() == block &&
        op->hasTrait<mlir::OpTrait::IsTerminator>())
      return;
    if (!visited.insert(&use).second)
      return;
    // Later ops in the chain start no earlier than this one.
    if (isTimedOp(op))
      return note(op);
    unsigned index = use.getOperandNumber();
    if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(op)) {
      mlir::Operation *parent = yield->getParentOp();
      visitUses(parent->getResult(index));
      if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(parent))
        visitUses(loop.getRegionIterArgs()[index]);
      return;
    }
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
      if (index >= loop.getNumControlOperands())
        visitUses(
            loop.getRegionIterArgs()[index - loop.getNumControlOperands()]);
      return;
    }
    if (op->getNumRegions() > 0) {
      // A branch condition: both arms wait for it.
      op->walk([&](mlir::Operation *inner) {
        if (isTimedOp(inner))
          note(inner);
      });
      return;
    }
    for (mlir::Value result : op->getResults())
      visitUses(result);
  }

  pulse::ScheduleBlockOp block;
  llvm::DenseSet<mlir::OpOperand *> visited;
  int64_t earliest = kNever;
};

/// When each value in a scheduled block becomes available, in local time,
/// or which operand it passes through unchanged.
class Availability {
public:
  struct State {
    int64_t time = 0;
    int source = -1;
  };

  Availability(pulse::ScheduleBlockOp block,
               llvm::MutableArrayRef<int64_t> firstUse)
      : block(block), firstUse(firstUse) {
    for (auto [index, arg] : llvm::enumerate(block.getBody().getArguments()))
      state[arg] = State{0, static_cast<int>(index)};
  }

  void run() { visitBlock(block.getBody().front()); }

  State of(mlir::Value value) const { return state.lookup(value); }

  /// The local time every op of the block is done by.
  int64_t length = 0;

private:
  /// The latest of `states`: the source they all pass through, else a
  /// time. An operand reached here is needed by now if nothing needs it
  /// earlier.
  State join(llvm::ArrayRef<State> states) {
    if (!states.empty() && llvm::all_of(states, [&](const State &s) {
          return s.source >= 0 && s.source == states.front().source;
        }))
      return states.front();
    int64_t time = 0;
    for (const State &s : states)
      time = std::max(time, s.source >= 0
                                ? std::max<int64_t>(firstUse[s.source], 0)
                                : s.time);
    for (const State &s : states)
      if (s.source >= 0 && firstUse[s.source] < 0)
        firstUse[s.source] = time;
    return State{time, -1};
  }

  State joinValues(mlir::ValueRange values) {
    llvm::SmallVector<State> states;
    for (mlir::Value value : values)
      if (isTimedType(value.getType()))
        states.push_back(state.lookup(value));
    return join(states);
  }

  void visitBlock(mlir::Block &body) {
    for (mlir::Operation &op : body.without_terminator())
      visitOp(&op);
  }

  void visitOp(mlir::Operation *op) {
    if (isTimedOp(op)) {
      auto start = pulse::scheduledStart(op, iteration);
      if (!start) {
        State through = joinValues(op->getOperands());
        for (mlir::Value result : op->getResults())
          state[result] = through;
        return;
      }
      int64_t end = *start + durationOf(op);
      length = std::max(length, end);
      for (mlir::Value result : op->getResults())
        state[result] = State{end, -1};
      if (auto submit = mlir::dyn_cast<classical::SubmitOp>(op))
        state[submit.getFuture()] = State{end + budgetOf(op), -1};
      return;
    }
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
      int64_t trips = pulse::constantTripCount(loop).value_or(0);
      llvm::SmallVector<State> carried;
      for (mlir::Value init : loop.getInitArgs())
        carried.push_back(state.lookup(init));
      // The first iteration, then the last from what the first carries:
      // the ops that run in every iteration override what it carries.
      llvm::SmallVector<int64_t, 2> runs;
      if (trips > 0)
        runs.push_back(0);
      if (trips > 1)
        runs.push_back(trips - 1);
      iteration.push_back(0);
      for (int64_t k : runs) {
        iteration.back() = k;
        for (auto [arg, s] : llvm::zip(loop.getRegionIterArgs(), carried))
          state[arg] = s;
        visitBlock(*loop.getBody());
        for (auto [s, value] :
             llvm::zip(carried, loop.getBody()->getTerminator()->getOperands()))
          s = state.lookup(value);
      }
      iteration.pop_back();
      for (auto [result, s] : llvm::zip(loop.getResults(), carried))
        state[result] = s;
      return;
    }
    if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
      visitBlock(*branch.thenBlock());
      llvm::SmallVector<State> results;
      for (mlir::Value value : branch.thenYield().getOperands())
        results.push_back(state.lookup(value));
      if (!branch.getElseRegion().empty()) {
        visitBlock(*branch.elseBlock());
        for (auto [s, value] :
             llvm::zip(results, branch.elseYield().getOperands()))
          s = join({s, state.lookup(value)});
      }
      for (auto [result, s] : llvm::zip(branch.getResults(), results))
        state[result] = s;
      return;
    }
    State through = joinValues(op->getOperands());
    for (mlir::Value result : op->getResults())
      if (isTimedType(result.getType()))
        state[result] = through;
  }

  pulse::ScheduleBlockOp block;
  llvm::MutableArrayRef<int64_t> firstUse;
  llvm::DenseMap<mlir::Value, State> state;
  llvm::SmallVector<int64_t, 2> iteration;
};

/// Record the shape of `block`, whose body has a local schedule.
static void writeShape(pulse::ScheduleBlockOp block) {
  FirstUse firstUse(block);
  llvm::SmallVector<int64_t> in;
  for (mlir::Value arg : block.getBody().getArguments()) {
    int64_t first = firstUse.of(arg);
    in.push_back(first == kNever ? -1 : first);
  }
  Availability availability(block, in);
  availability.run();
  llvm::SmallVector<int64_t> out, source;
  int64_t length = availability.length;
  for (mlir::Value value :
       block.getBody().front().getTerminator()->getOperands()) {
    Availability::State s = availability.of(value);
    out.push_back(s.source >= 0 ? -1 : s.time);
    source.push_back(s.source);
    length = std::max(length, s.source >= 0 ? 0 : s.time);
  }
  mlir::MLIRContext *context = block.getContext();
  block->setAttr(
      "length_cycles",
      mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64), length));
  block->setAttr("in_offset_cycles", mlir::DenseI64ArrayAttr::get(context, in));
  block->setAttr("out_offset_cycles",
                 mlir::DenseI64ArrayAttr::get(context, out));
  block->setAttr("out_source", mlir::DenseI64ArrayAttr::get(context, source));
}

/// Replace `block` by its body.
static void inlineBody(pulse::ScheduleBlockOp block) {
  mlir::Block &body = block.getBody().front();
  mlir::Operation *yield = body.getTerminator();
  llvm::SmallVector<mlir::Value> results(yield->getOperands());
  yield->erase();
  mlir::IRRewriter rewriter(block.getContext());
  rewriter.inlineBlockBefore(&body, block, block.getInputs());
  rewriter.replaceOp(block, results);
}

/// Move the ops of the placed `block` to its start and period, then inline
/// it.
static void inlineScheduleBlock(pulse::ScheduleBlockOp block) {
  std::optional<int64_t> start = pulse::startCycles(block);
  llvm::ArrayRef<int64_t> outer = pulse::periodCycles(block);
  block.getBody().walk([&](mlir::Operation *op) {
    if (!isTimedOp(op))
      return;
    std::optional<int64_t> local = pulse::startCycles(op);
    if (!start || !local) {
      // The block or this op never runs.
      pulse::setStartCycles(op, std::nullopt);
      pulse::setPeriodCycles(op, {});
      return;
    }
    pulse::setStartCycles(op, *start + *local);
    llvm::SmallVector<int64_t> periods(outer);
    llvm::append_range(periods, pulse::periodCycles(op));
    pulse::setPeriodCycles(op, periods);
  });
  inlineBody(block);
}

// ── Duration budgets ─────────────────────────────────────────────────────
//
// `max_duration_cycles` on a schedule block, or `pulse.max_duration_cycles`
// on the module, bounds how long every execution of the block, or the whole
// program, may take. A wait whose length is not known yet is scheduled at
// its current length; the check adds what each execution of it can add.

/// How many cycles longer than scheduled `op` can take: 0 for an op of
/// known length, -1 for a wait without a highest length.
static int64_t stretchOf(mlir::Operation *op) {
  if (mlir::isa<classical::WaitOp>(op))
    return op->hasAttr("classical.blocking") ? -1 : 0;
  auto wait = mlir::dyn_cast<pulse::WaitOp>(op);
  if (!wait || !op->hasAttr("pulse.variable_wait"))
    return 0;
  if (auto conversion =
          wait.getDuration().getDefiningOp<pulse::DurationFromIntOp>()) {
    bool known = op->hasAttr("classical.variable")
                     ? pulse::boundNumber(conversion.getCycles()).has_value()
                     : traceConstantI64(conversion.getCycles()).has_value();
    if (known)
      return 0;
  }
  auto highest = op->getAttrOfType<mlir::IntegerAttr>("pulse.max_cycles");
  if (!highest)
    return -1;
  return std::max<int64_t>(0, highest.getInt() - durationOf(op));
}

/// The executions of `op` per execution of `root`.
static int64_t executionsIn(mlir::Operation *op, mlir::Operation *root) {
  int64_t count = 1;
  for (mlir::scf::ForOp loop : pulse::enclosingLoops(op, root))
    count *= pulse::constantTripCount(loop).value_or(0);
  return count;
}

/// What the ops in `root` whose length is not known yet can add to it, or
/// the first one with no bound.
static std::pair<int64_t, mlir::Operation *> stretchIn(mlir::Operation *root) {
  int64_t total = 0;
  mlir::Operation *unbounded = nullptr;
  root->walk([&](mlir::Operation *op) {
    if (!isTimedOp(op))
      return mlir::WalkResult::advance();
    int64_t stretch = stretchOf(op);
    if (stretch < 0) {
      unbounded = op;
      return mlir::WalkResult::interrupt();
    }
    if (stretch > 0)
      total += stretch * executionsIn(op, root);
    return mlir::WalkResult::advance();
  });
  return {total, unbounded};
}

/// Fail with the error for `what` (a kernel, quoted), whose executions take
/// `length` cycles as scheduled, against `budget`, if it can take longer.
static mlir::LogicalResult checkBudget(mlir::Operation *root,
                                       mlir::Operation *at,
                                       llvm::StringRef what, int64_t length,
                                       int64_t budget) {
  auto [stretch, unbounded] = stretchIn(root);
  if (unbounded && mlir::isa<classical::WaitOp>(unbounded))
    return unbounded->emitError()
           << what << " has max_duration_cycles=" << budget
           << ", but it waits for an answer of a blocking classical function, "
              "which has no bound; use latency_mode=\"bounded\" or "
              "\"timeout\", or drop max_duration_cycles";
  if (unbounded)
    return unbounded->emitError()
           << what << " has max_duration_cycles=" << budget
           << ", but this wait in it has no highest length; bound it with "
              "wait(..., bounds=(lowest, highest)), or drop "
              "max_duration_cycles";
  if (length + stretch <= budget)
    return mlir::success();
  auto diagnostic = at->emitError()
                    << what << " takes " << (stretch ? "up to " : "")
                    << length + stretch << " cycles";
  if (stretch)
    diagnostic << " (" << length << " as scheduled, its variable waits add up "
               << "to " << stretch << ")";
  diagnostic << ", more than max_duration_cycles=" << budget
             << "; let its operations overlap, shorten them or raise "
                "max_duration_cycles";
  return mlir::failure();
}

static mlir::LogicalResult checkBlockBudget(pulse::ScheduleBlockOp block) {
  auto budget = block->getAttrOfType<mlir::IntegerAttr>("max_duration_cycles");
  if (!budget)
    return mlir::success();
  std::string what =
      "each call of kernel '" + block.getCallee().value_or("block").str() + "'";
  return checkBudget(block, block, what, shapeOf(block).length,
                     budget.getInt());
}

/// The end of the last execution of the last op of the scheduled
/// `function`.
static int64_t programLength(mlir::func::FuncOp function) {
  int64_t length = 0;
  function.walk([&](mlir::Operation *op) {
    if (!isTimedOp(op))
      return;
    std::optional<int64_t> start = pulse::startCycles(op);
    if (!start)
      return;
    auto loops = pulse::enclosingLoops(op, function);
    llvm::SmallVector<int64_t, 2> last;
    for (mlir::scf::ForOp loop : loops) {
      int64_t trips = pulse::constantTripCount(loop).value_or(0);
      if (trips == 0)
        return;
      last.push_back(trips - 1);
    }
    int64_t end = *start;
    for (auto [k, period] : llvm::zip(last, pulse::periodCycles(op)))
      end += std::max<int64_t>(0, k * period);
    end += pulse::scheduledDuration(op).value_or(0);
    length = std::max(length, end);
  });
  return length;
}

static mlir::LogicalResult checkProgramBudget(mlir::func::FuncOp function) {
  auto module = function->getParentOfType<mlir::ModuleOp>();
  auto budget = module ? module->getAttrOfType<mlir::IntegerAttr>(
                             "pulse.max_duration_cycles")
                       : mlir::IntegerAttr();
  if (!budget)
    return mlir::success();
  std::string what =
      "kernel '" + module.getSymName().value_or("main").str() + "'";
  return checkBudget(function, function, what, programLength(function),
                     budget.getInt());
}

// ── Driver ───────────────────────────────────────────────────────────────

/// Rounds of a region's scheduling that may peel several iterations at once.
constexpr int64_t kWidePeelRounds = 8;

/// Rounds of a region's scheduling that may regroup a loop's iterations.
constexpr int64_t kRegroupRounds = 4;

/// Schedule `region`, its schedule blocks done, in rounds until its schedule
/// is affine.
template <bool Gridded>
static mlir::LogicalResult
scheduleRounds(mlir::Region &region, mlir::Operation *root, Resources resources,
               bool isAlap, bool classical, const Timelines &timelines,
               const pulse::LineGrid &grid,
               const pulse::Controllers &controllers) {
  // Each round either writes an affine schedule, removes an iteration from
  // a loop, or (a bounded number of times) regroups one, so this
  // terminates.
  int64_t widePeels = 0, regroups = 0, headPeels = 0;
  llvm::DenseMap<mlir::Operation *, int64_t> regroupedBy;
  while (true) {
    Scheduler<Gridded> scheduler(region, resources, classical, grid,
                                 controllers, &timelines);
    scheduler.mayRegroup = regroups < kRegroupRounds;
    scheduler.mayPeelHead = headPeels < kRegroupRounds;
    scheduler.regroupedBy = regroupedBy;
    auto regrouped = [&] {
      if (scheduler.headPeel) {
        ++headPeels;
        for (int64_t i = 0; i < scheduler.headPeel->iterations; ++i)
          peelFront(scheduler.headPeel->loop);
        return true;
      }
      if (!scheduler.regroup)
        return false;
      ++regroups;
      auto [loop, factor] = *scheduler.regroup;
      int64_t &by = regroupedBy.try_emplace(loop, 1).first->second;
      by *= factor;
      unrollBy(loop, factor);
      return true;
    };
    if (mlir::failed(scheduler.forward()))
      return mlir::failure();
    if (regrouped())
      continue;
    if (isAlap) {
      if (mlir::failed(scheduler.backward()))
        return mlir::failure();
      if (regrouped())
        continue;
      // Handing out resources in reverse program order can need time before
      // zero, take an answer before it is due, or leave a request kept at
      // its forward start no slot; the forward resource schedule is always
      // feasible.
      bool feasible = !scheduler.skippedNegative && !scheduler.answeredEarly &&
                      !(scheduler.slotsOverrun && scheduler.heldTooMany()) &&
                      llvm::all_of(scheduler.samples, [](const auto &entry) {
                        return llvm::all_of(entry.second, [](const Sample &s) {
                          return s.start >= 0;
                        });
                      });
      if (!feasible) {
        scheduler = Scheduler<Gridded>(region, resources, classical, grid,
                                       controllers, &timelines);
        scheduler.mayRegroup = false;
        scheduler.mayPeelHead = false;
        if (mlir::failed(scheduler.forward()))
          return mlir::failure();
      }
    }

    // The first problem of each loop that neither holds nor is held by the
    // loop of one before it: peeling or unrolling one leaves the others'
    // loops in place, and copies of a loop that peeling made (one per
    // iteration peeled around it) are then all fixed in one round.
    llvm::SmallVector<NotAffine> problems;
    auto apart = [&](mlir::scf::ForOp loop) {
      return llvm::none_of(problems, [&](const NotAffine &other) {
        return other.loop->isAncestor(loop) || loop->isAncestor(other.loop);
      });
    };
    forEachScheduled(region, [&](mlir::Operation *op) {
      auto it = scheduler.samples.find(op);
      llvm::ArrayRef<Sample> samples;
      if (it != scheduler.samples.end())
        samples = it->second;
      if (auto problem =
              writeTiming(op, root, samples, scheduler.extrapolations))
        if (apart(problem->loop))
          problems.push_back(*problem);
    });
    if (problems.empty())
      if (auto problem = periodMismatch(region, root))
        problems.push_back(*problem);
    if (problems.empty())
      break;
    // Each round schedules the whole region again: a few may peel several
    // iterations of a loop, the rest at most one or unroll.
    bool wide = llvm::any_of(problems, [](const NotAffine &problem) {
      return problem.peelFront + problem.peelBack > 1;
    });
    bool narrow = wide && ++widePeels > kWidePeelRounds;
    for (NotAffine &problem : problems) {
      if (narrow && problem.peelFront + problem.peelBack > 1)
        problem.peelFront = problem.peelBack = 0;
      for (int64_t i = 0; i < problem.peelFront; ++i)
        peelFront(problem.loop);
      for (int64_t i = 0; i < problem.peelBack; ++i)
        peelBack(problem.loop);
      if (!problem.peelFront && !problem.peelBack)
        unrollFully(problem.loop);
    }
  }

  return mlir::success();
}

/// Whether the module of `op` lists the controllers that block on calls.
static bool holdsControllers(mlir::Operation *op) {
  auto module = op->getParentOfType<mlir::ModuleOp>();
  return module && mlir::isa_and_present<mlir::DenseI64ArrayAttr>(
                       module->getAttr(classical::kBlocksOnCallsAttr));
}

static mlir::LogicalResult
scheduleRegion(mlir::Region &region, mlir::Operation *root, Resources resources,
               bool isAlap, bool classical, const Timelines &timelines,
               const pulse::LineGrid &grid,
               const pulse::Controllers &controllers) {
  auto blocks = placeableBlocks(region);
  if (mlir::failed(blocks))
    return mlir::failure();
  for (pulse::ScheduleBlockOp block : *blocks) {
    if (mlir::failed(
            scheduleRegion(block.getBody(), block, Resources{},
                           block.getPolicy() == pulse::SchedulePolicy::ALAP,
                           classical, timelines, grid, controllers)))
      return mlir::failure();
    writeShape(block);
    if (mlir::failed(checkBlockBudget(block)))
      return mlir::failure();
  }
  // Calls that hold some of several controllers trace lines under any
  // policy.
  bool traced =
      grid.enabled() ||
      (controllers.enabled() && (resources.enabled || holdsControllers(root)));
  if (mlir::failed(
          traced
              ? scheduleRounds<true>(region, root, resources, isAlap, classical,
                                     timelines, grid, controllers)
              : scheduleRounds<false>(region, root, resources, isAlap,
                                      classical, timelines, grid, controllers)))
    return mlir::failure();

  // Peeling may have copied blocks, so collect them again.
  for (pulse::ScheduleBlockOp block : outermostBlocks(region))
    inlineScheduleBlock(block);
  return mlir::success();
}

static mlir::LogicalResult schedule(mlir::func::FuncOp function,
                                    Resources resources, bool isAlap) {
  if (resources.enabled &&
      (resources.maxDrives <= 0 || resources.maxReadouts <= 0))
    return function.emitError("resource limits must be positive");
  if (resources.enabled) {
    auto blocks = outermostBlocks(function.getBody());
    if (!blocks.empty())
      return blocks.front().emitError(
          "resource-constrained scheduling (rcp, alap_rcp) does not support "
          "kernels with their own schedule yet; drop schedule= from the "
          "called kernels, or flatten them with flatten_schedules=True");
  }
  mlir::FailureOr<bool> annotated = classical::annotateClassicalCalls(function);
  if (mlir::failed(annotated))
    return mlir::failure();
  bool classical = *annotated;
  bool blocksOnCalls = function->getParentOfType<mlir::ModuleOp>()->hasAttr(
      classical::kBlocksOnCallsAttr);
  if (classical)
    for (pulse::ScheduleBlockOp block : outermostBlocks(function.getBody())) {
      mlir::WalkResult found = block.walk([blocksOnCalls](mlir::Operation *op) {
        if (op->hasAttr("classical.variable")) {
          op->emitError(classical::kVariableWaitInBlock);
          return mlir::WalkResult::interrupt();
        }
        if (mlir::isa<classical::WaitOp>(op) &&
            op->hasAttr("classical.blocking")) {
          op->emitError(classical::kBlockingWaitInBlock);
          return mlir::WalkResult::interrupt();
        }
        if (blocksOnCalls && mlir::isa<classical::SubmitOp>(op)) {
          op->emitError(classical::kBlockingCallInBlock);
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      });
      if (found.wasInterrupted())
        return mlir::failure();
    }
  Timelines timelines;
  if (classical && TimelineAnalysis(function, resources).run(timelines) &&
      mlir::failed(classical::annotateClassicalCalls(function)))
    return mlir::failure();
  // Ops that read a value a measurement selects wait for it, as ops that
  // read an answer do.
  bool data = classical || function->hasAttr("pulse.selected_values");
  const pulse::ClockAttributes clocks(
      function->getParentOfType<mlir::ModuleOp>());
  if (mlir::failed(scheduleRegion(
          function.getBody(), function, resources, isAlap, data, timelines,
          pulse::LineGrid(clocks), pulse::Controllers(clocks.controllers))) ||
      mlir::failed(checkProgramBudget(function)))
    return mlir::failure();
  pulse::applyLineDelays(function, clocks.delays);
  return mlir::success();
}

template <bool IsAlap>
struct PulseSchedulePass
    : public mlir::PassWrapper<PulseSchedulePass<IsAlap>,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PulseSchedulePass<IsAlap>)

  llvm::StringRef getArgument() const override {
    return IsAlap ? "pulse-schedule-alap" : "pulse-schedule-asap";
  }

  llvm::StringRef getDescription() const override {
    return IsAlap ? "Assign a dependency-correct ALAP pulse schedule"
                  : "Assign a dependency-correct ASAP pulse schedule";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
  }

  void runOnOperation() override {
    if (mlir::failed(schedule(this->getOperation(), Resources{}, IsAlap)))
      this->signalPassFailure();
  }
};

struct PulseScheduleRcpPass
    : public mlir::PassWrapper<PulseScheduleRcpPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PulseScheduleRcpPass)

  PulseScheduleRcpPass(int64_t maxDrives, int64_t maxReadouts,
                       int64_t readoutLatency, int64_t switchPenalty,
                       bool isAlap)
      : resources{true, maxDrives, maxReadouts, readoutLatency, switchPenalty},
        isAlap(isAlap) {}

  llvm::StringRef getArgument() const override { return "pulse-schedule-rcp"; }
  llvm::StringRef getDescription() const override {
    return "Assign a resource-constrained pulse schedule";
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
  }
  void runOnOperation() override {
    if (mlir::failed(schedule(getOperation(), resources, isAlap)))
      signalPassFailure();
  }

  Resources resources;
  bool isAlap;
};

/// pulse-schedule-rcp as cudaq-pulse-opt runs it, with the machine model as
/// options. The extension builds PulseScheduleRcpPass instead: options cost
/// a command-line parser per pass.
struct PulseScheduleRcpOptionsPass
    : public mlir::PassWrapper<PulseScheduleRcpOptionsPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PulseScheduleRcpOptionsPass)

  PulseScheduleRcpOptionsPass() = default;
  PulseScheduleRcpOptionsPass(const PulseScheduleRcpOptionsPass &other)
      : PassWrapper(other) {}

  llvm::StringRef getArgument() const override { return "pulse-schedule-rcp"; }
  llvm::StringRef getDescription() const override {
    return "Assign a resource-constrained pulse schedule";
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
  }
  void runOnOperation() override {
    Resources resources{true, maxDrives, maxReadouts, readoutLatency,
                        switchPenalty};
    if (mlir::failed(schedule(getOperation(), resources, alap)))
      signalPassFailure();
  }

  Option<int64_t> maxDrives{*this, "max-drives",
                            llvm::cl::desc("Drives that may play at once"),
                            llvm::cl::init(4)};
  Option<int64_t> maxReadouts{*this, "max-readouts",
                              llvm::cl::desc("Readouts that may play at once"),
                              llvm::cl::init(2)};
  Option<int64_t> readoutLatency{
      *this, "readout-latency",
      llvm::cl::desc(
          "Cycles after a readout before its line and result are ready"),
      llvm::cl::init(0)};
  Option<int64_t> switchPenalty{
      *this, "switch-penalty",
      llvm::cl::desc("Cycles a drive or readout unit stays busy after each op"),
      llvm::cl::init(0)};
  Option<bool> alap{*this, "alap",
                    llvm::cl::desc("Place each op as late as it may go "
                                   "(alap_rcp)"),
                    llvm::cl::init(false)};
};

/// Drop every schedule block, so that its ops are scheduled with the
/// enclosing ones.
struct PulseInlineScheduleBlocksPass
    : public mlir::PassWrapper<PulseInlineScheduleBlocksPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PulseInlineScheduleBlocksPass)

  llvm::StringRef getArgument() const override {
    return "pulse-inline-schedule-blocks";
  }
  llvm::StringRef getDescription() const override {
    return "Inline pulse.schedule_block ops, ignoring their policies";
  }
  void runOnOperation() override {
    llvm::SmallVector<pulse::ScheduleBlockOp> blocks;
    getOperation().walk(
        [&](pulse::ScheduleBlockOp block) { blocks.push_back(block); });
    for (pulse::ScheduleBlockOp block : blocks)
      inlineBody(block);
  }
};

/// Whether `loop` holds a classical call or a variable wait.
static bool holdsClassical(mlir::scf::ForOp loop) {
  return loop.getBody()
      ->walk([](mlir::Operation *op) {
        if (op->getDialect() && op->getDialect()->getNamespace() == "classical")
          return mlir::WalkResult::interrupt();
        return mlir::WalkResult::advance();
      })
      .wasInterrupted();
}

/// Unroll every loop that holds classical calls, so each execution of a call
/// is an op of its own: simulation binds answers one execution at a time.
struct PulseUnrollClassicalLoopsPass
    : public mlir::PassWrapper<PulseUnrollClassicalLoopsPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PulseUnrollClassicalLoopsPass)

  llvm::StringRef getArgument() const override {
    return "pulse-unroll-classical-loops";
  }
  llvm::StringRef getDescription() const override {
    return "Fully unroll loops that hold classical calls";
  }
  void runOnOperation() override {
    while (true) {
      llvm::SmallVector<mlir::scf::ForOp> loops;
      getOperation().walk<mlir::WalkOrder::PreOrder>(
          [&](mlir::scf::ForOp loop) {
            if (!holdsClassical(loop))
              return mlir::WalkResult::advance();
            loops.push_back(loop);
            return mlir::WalkResult::skip();
          });
      if (loops.empty())
        return;
      for (mlir::scf::ForOp loop : loops) {
        if (!pulse::constantTripCount(loop)) {
          loop.emitError("cannot unroll a loop with non-constant bounds");
          return signalPassFailure();
        }
        unrollFully(loop);
      }
    }
  }
};

} // namespace

namespace pulse {
std::unique_ptr<mlir::Pass> createPulseUnrollClassicalLoopsPass() {
  return std::make_unique<PulseUnrollClassicalLoopsPass>();
}

std::unique_ptr<mlir::Pass> createPulseScheduleAsapPass() {
  return std::make_unique<PulseSchedulePass<false>>();
}

std::unique_ptr<mlir::Pass> createPulseScheduleAlapPass() {
  return std::make_unique<PulseSchedulePass<true>>();
}

std::unique_ptr<mlir::Pass> createPulseScheduleRcpPass(int64_t maxDrives,
                                                       int64_t maxReadouts,
                                                       int64_t readoutLatency,
                                                       int64_t switchPenalty,
                                                       bool isAlap) {
  return std::make_unique<PulseScheduleRcpPass>(
      maxDrives, maxReadouts, readoutLatency, switchPenalty, isAlap);
}

std::unique_ptr<mlir::Pass> createPulseScheduleRcpOptionsPass() {
  return std::make_unique<PulseScheduleRcpOptionsPass>();
}

std::unique_ptr<mlir::Pass> createPulseInlineScheduleBlocksPass() {
  return std::make_unique<PulseInlineScheduleBlocksPass>();
}
} // namespace pulse
