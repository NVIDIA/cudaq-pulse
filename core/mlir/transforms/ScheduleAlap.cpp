/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

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
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <algorithm>
#include <limits>

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

#include "cudaq-pulse/Analysis/Affine.h"
#include "cudaq-pulse/Analysis/ClassicalTiming.h"
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

static int64_t getWaveformDuration(mlir::Value waveform) {
  auto *definition = waveform.getDefiningOp();
  if (!definition)
    return 0;

  mlir::Value duration;
  if (auto op = mlir::dyn_cast<pulse::GaussianPulseOp>(definition))
    duration = op.getDuration();
  else if (auto op = mlir::dyn_cast<pulse::SquarePulseOp>(definition))
    duration = op.getDuration();
  else if (auto op = mlir::dyn_cast<pulse::DRAGPulseOp>(definition))
    duration = op.getDuration();
  else if (auto op = mlir::dyn_cast<pulse::CosinePulseOp>(definition))
    duration = op.getDuration();
  else if (auto op = mlir::dyn_cast<pulse::TanhRampOp>(definition))
    duration = op.getDuration();
  else if (auto op = mlir::dyn_cast<pulse::GaussianSquarePulseOp>(definition))
    duration = op.getDuration();
  else if (auto op = mlir::dyn_cast<pulse::CustomOp>(definition))
    duration = op.getDuration();
  else if (auto op = mlir::dyn_cast<pulse::CustomSamplesOp>(definition))
    return op.getDurationCycles();
  else if (auto op = mlir::dyn_cast<pulse::PulseAddOp>(definition))
    return getWaveformDuration(op.getLhs());
  else if (auto op = mlir::dyn_cast<pulse::PulseSubOp>(definition))
    return getWaveformDuration(op.getLhs());
  else if (auto op = mlir::dyn_cast<pulse::PulseMulOp>(definition))
    return getWaveformDuration(op.getLhs());
  else if (auto op = mlir::dyn_cast<pulse::PulseScaleOp>(definition))
    return getWaveformDuration(op.getPulse());
  else if (auto op = mlir::dyn_cast<pulse::PulseNegOp>(definition))
    return getWaveformDuration(op.getPulse());
  else
    return 0;

  return traceConstantI64(duration).value_or(0);
}

/// A variable wait whose length is not bound yet is scheduled at its lowest
/// length.
static int64_t getWaitDuration(pulse::WaitOp wait) {
  if (auto conversion =
          wait.getDuration().getDefiningOp<pulse::DurationFromIntOp>())
    if (auto cycles = traceConstantI64(conversion.getCycles()))
      return *cycles;
  if (auto lowest = wait->getAttrOfType<mlir::IntegerAttr>("pulse.min_cycles"))
    return lowest.getInt();
  return 0;
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

static int64_t durationOf(mlir::Operation *op) {
  if (auto block = mlir::dyn_cast<pulse::ScheduleBlockOp>(op))
    return shapeOf(block).length;
  if (auto drive = mlir::dyn_cast<pulse::DriveOp>(op))
    return getWaveformDuration(drive.getPulse());
  if (auto readout = mlir::dyn_cast<pulse::ReadoutOp>(op))
    return getWaveformDuration(readout.getPulse());
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
  return getWaitDuration(mlir::cast<pulse::WaitOp>(op));
}

/// From the end of a submission until its future is available.
static int64_t budgetOf(mlir::Operation *submit) {
  return classical::intAttr(submit, "classical.budget_cycles");
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

static llvm::SmallVector<mlir::Value> conditionMeasurements(mlir::Value cond) {
  llvm::SmallVector<mlir::Value> found;
  llvm::DenseSet<mlir::Value> seen;
  conditionMeasurements(cond, found, seen);
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
      for (auto [init, arg, result] :
           llvm::zip(loop.getInitArgs(), loop.getRegionIterArgs(),
                     loop.getResults())) {
        if (isLink(arg))
          continue;
        unsigned c = join(make(), join(context, of(init)));
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
      std::map<int64_t, llvm::SmallVector<unsigned>> perLink;
      for (auto [op, c] : nodes) {
        if (!mlir::isa<classical::SubmitOp>(op))
          continue;
        auto &list = perLink[classical::intAttr(op, "classical.device")];
        if (!llvm::is_contained(list, find(c)))
          list.push_back(find(c));
      }
      bool merged = false;
      for (auto &[device, list] : perLink) {
        auto slots = static_cast<size_t>(std::max<int64_t>(
            1, classical::linkModel(module, device).maxInFlight));
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
/// each one shifts the previous by `period`.
struct Extrapolation {
  mlir::scf::ForOp loop;
  int64_t period;
};

/// Schedules one region: a function body, or a schedule block's body with
/// every block argument available at time 0.
class Scheduler {
public:
  Scheduler(mlir::Region &region, Resources resources, bool classical,
            const pulse::LineGrid &grid, const Timelines *timelines = nullptr)
      : region(&region), resources(resources), classical(classical),
        lanesPer(timelines && timelines->count > 1 ? timelines : nullptr),
        grid(grid) {
    auto count = static_cast<size_t>(isolated() ? lanesPer->count : 1);
    floors.assign(count, 0);
    ends.assign(count, 0);
    iterationEnds.assign(count, -1);
    ceilings.assign(count, kNever);
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

private:
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
    for (size_t t = 0; t < count; ++t) {
      int64_t drives = isolated() ? lanesPer->drives[t] : resources.maxDrives;
      int64_t readouts =
          isolated() ? lanesPer->readouts[t] : resources.maxReadouts;
      driveLanes[t].assign(static_cast<size_t>(std::max<int64_t>(1, drives)),
                           value);
      readoutLanes[t].assign(
          static_cast<size_t>(std::max<int64_t>(1, readouts)), value);
    }
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

  /// A link as one timeline sees it: its share of the slots, and its own
  /// order of submissions.
  using LinkKey = std::pair<int64_t, int64_t>;
  LinkKey linkOf(mlir::Operation *submit) {
    return {classical::intAttr(submit, "classical.device"), timelineOf(submit)};
  }

  void record(mlir::Operation *op, int64_t start) {
    samples[op].push_back({iteration, start});
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

  /// When every operand of `op` is available: data from answers included.
  /// Timelines keep their own order on a link, so the link a submission
  /// is threaded through says nothing about when it may start.
  int64_t operandsReady(mlir::Operation *op) const {
    int64_t time = 0;
    for (mlir::Value operand : op->getOperands())
      if (!isolated() || !mlir::isa<classical::LinkType>(operand.getType()))
        time = std::max(time, ready(operand));
    return time;
  }

  /// The request slots of the link a submission uses: a heap of the times
  /// they become free (forward, min-heap) or must be free from (backward,
  /// max-heap). A link's slots bound every policy: they are hardware.
  llvm::SmallVector<int64_t> &lanesOf(mlir::Operation *submit) {
    auto [it, inserted] = linkLanes.try_emplace(linkOf(submit));
    if (inserted)
      it->second.assign(
          static_cast<size_t>(std::max<int64_t>(
              1, classical::intAttr(submit, "classical.max_in_flight", 64))),
          laneReset);
    return it->second;
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
      int64_t busy = durationOf(op) + budgetOf(op);
      if (period <= 0) {
        ok = false;
        return;
      }
      auto &[need, slots] = perLink[linkOf(op)];
      need += (busy + period - 1) / period;
      slots = classical::intAttr(op, "classical.max_in_flight", 64);
    });
    for (auto &[device, entry] : perLink)
      ok = ok && entry.first <= entry.second;
    return ok;
  }

  /// Take the earliest free slot of a link for a request starting no earlier
  /// than `start`; it is busy until the answer is due.
  int64_t takeSlot(mlir::Operation *submit, int64_t start, int64_t busy) {
    auto &heap = lanesOf(submit);
    auto later = std::greater<int64_t>();
    std::pop_heap(heap.begin(), heap.end(), later);
    if (heap.back() > start) {
      start = heap.back();
      lanesBound = true;
    }
    heap.back() = start + busy;
    std::push_heap(heap.begin(), heap.end(), later);
    return start;
  }

  /// The backward mirror: the latest a request ending its slot by the slot's
  /// bound may start, for one that is busy for `busy`.
  int64_t takeSlotBackward(mlir::Operation *submit, int64_t start, int64_t busy,
                           bool pinned) {
    auto &heap = lanesOf(submit);
    std::pop_heap(heap.begin(), heap.end());
    if (!pinned && heap.back() - busy < start) {
      start = heap.back() - busy;
      lanesBound = true;
    }
    heap.back() = start;
    std::push_heap(heap.begin(), heap.end());
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
          linkLanes[key] = std::move(heap);
        return start;
      }
      start += backward ? -shift : shift;
      if (grid.enabled())
        start = backward ? pulse::LineGrid::roundDown(start, grid.lcm())
                         : pulse::LineGrid::roundUp(start, grid.lcm());
      lanesBound = true;
    }
  }

  void forwardSubmit(classical::SubmitOp submit) {
    mlir::Operation *op = submit;
    int64_t t = timelineOf(op);
    int64_t link = isolated() ? linkReady[linkOf(op)] : ready(submit.getLink());
    int64_t start = std::max({link, floorOf(t), operandsReady(op)});
    int64_t duration = durationOf(op), budget = budgetOf(op);
    start = takeSlot(op, start, duration + budget);
    record(op, start);
    readyAt[submit.getUpdatedLink()] = start + duration;
    if (isolated())
      linkReady[linkOf(op)] = start + duration;
    readyAt[submit.getFuture()] = start + duration + budget;
    noteEnd(start + duration, t);
  }

  void forwardWait(classical::WaitOp wait) {
    int64_t t = timelineOf(wait);
    int64_t start = std::max(ready(wait.getFuture()), floorOf(t));
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

  void place(mlir::Operation *op, int64_t start, int64_t t) {
    int64_t duration = durationOf(op);
    record(op, start);
    int64_t end = start + duration;
    int64_t lineEnd = end;
    if (auto readout = mlir::dyn_cast<pulse::ReadoutOp>(op)) {
      if (resources.enabled)
        lineEnd += resources.readoutLatency;
      readyAt[readout.getResult()] = end + resources.readoutLatency;
    }
    readyAt[op->getResult(0)] = lineEnd;
    if (mlir::Value tone = toneResult(op))
      readyAt[tone] = end;
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
      int64_t t = timelineOf(op);
      int64_t start = std::max(ready(op->getOperand(0)), floorOf(t));
      if (mlir::Value tone = toneOperand(op))
        start = std::max(start, ready(tone));
      if (classical) {
        start = std::max(start, operandsReady(op));
        // A variable wait is a barrier: every line of its timeline is done
        // before it.
        if (op->hasAttr("classical.variable"))
          start = std::max(start, endOf(t));
      }
      int64_t step = grid.enabled() ? grid.of(op->getOperand(0), lines) : 1;
      if (resources.enabled && !mlir::isa<pulse::WaitOp>(op)) {
        auto &lanes =
            mlir::isa<pulse::DriveOp>(op) ? driveLanes[t] : readoutLanes[t];
        auto lane = std::min_element(lanes.begin(), lanes.end());
        start = std::max(start, *lane);
        if (step > 1)
          start = pulse::LineGrid::roundUp(start, step);
        *lane = start + durationOf(op) + resources.switchPenalty;
      }
      if (step > 1)
        start = pulse::LineGrid::roundUp(start, step);
      place(op, start, t);
      if (classical && op->hasAttr("classical.variable"))
        floors[t] = std::max(floors[t], start + durationOf(op));
      return mlir::success();
    }
    if (auto block = mlir::dyn_cast<pulse::ScheduleBlockOp>(op)) {
      forwardScheduleBlock(block);
      return mlir::success();
    }
    if (auto sync = mlir::dyn_cast<pulse::SyncOp>(op)) {
      int64_t t = timelineOf(op);
      int64_t time = floorOf(t);
      for (mlir::Value line : sync.getLines())
        time = std::max(time, ready(line));
      if (grid.enabled())
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
    if (grid.enabled())
      start = pulse::LineGrid::roundUp(start, grid.lcm());
    if (classical)
      start = takeBlockSlots(block, start, /*backward=*/false);
    record(block, start);
    for (auto [result, offset, source] :
         llvm::zip(block.getOutputs(), shape.out, shape.source))
      readyAt[result] = offset >= 0 ? start + offset : ready(inputs[source]);
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
    for (int64_t k = 0; k < count; ++k) {
      iteration.back() = k;
      int64_t savedEnd = iterationEnd;
      iterationEnd = 0;
      auto savedEnds = iterationEnds;
      std::fill(iterationEnds.begin(), iterationEnds.end(), -1);
      auto floorsBefore = floors;
      auto linksBefore = linkReady;
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
      bool steady = !resources.enabled && !lanesBound && shifted &&
                    lastShift == shift && shift >= 0 &&
                    shift % grid.lcm() == 0 &&
                    slotsHoldSteadyState(loop, shift);
      for (auto [before, after, kind, bound] :
           llvm::zip(carried, next, kinds, outsideOf))
        if (kind == Carried::Timed ||
            (kind == Carried::Tone && after != before))
          steady = steady && after >= bound;
      lastShift = shifted ? std::optional<int64_t>(shift) : std::nullopt;
      llvm::SmallVector<int64_t> previous = std::move(carried);
      carried = std::move(next);
      if (steady && k + 1 < count) {
        int64_t remaining = count - 1 - k;
        int64_t distance = remaining * shift;
        extrapolate(carried, previous, kinds, distance);
        // What the iterations not simulated move: the ends and run-time
        // waits of the timelines the body plays on, and their links.
        if (!isolated()) {
          noteEnd(thisEnd + distance, 0);
        } else {
          for (auto [t, end] : llvm::enumerate(theseEnds))
            if (end >= 0)
              noteEnd(end + distance, static_cast<int64_t>(t));
          for (auto &[key, time] : linkReady) {
            auto before = linksBefore.find(key);
            if (before == linksBefore.end() || before->second != time)
              time += distance;
          }
        }
        for (auto [now, before] : llvm::zip(floors, floorsBefore))
          if (now != before)
            now += distance;
        extrapolations.push_back({loop, shift});
        break;
      }
    }
    iteration.pop_back();
    for (auto [result, time] : llvm::zip(loop.getResults(), carried))
      readyAt[result] = time;
    return mlir::success();
  }

  using Lanes = llvm::SmallVector<llvm::SmallVector<int64_t>, 1>;

  /// After a branch, each lane is as busy as the busier arm left it.
  static void mergeLanes(Lanes &lanes, const Lanes &other, bool later) {
    for (auto [mine, theirs] : llvm::zip(lanes, other))
      for (auto [a, b] : llvm::zip(mine, theirs))
        a = later ? std::max(a, b) : std::min(a, b);
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
    for (mlir::Value measurement : conditionMeasurements(branch.getCondition()))
      floor = std::max(floor, ready(measurement));
    if (classical)
      floor = std::max(floor, ready(branch.getCondition()));

    auto savedDrive = driveLanes, savedReadout = readoutLanes;
    auto savedLinks = linkReady;
    if (mlir::failed(forwardBlock(*branch.thenBlock())))
      return mlir::failure();
    llvm::SmallVector<int64_t> results;
    for (mlir::Value value : branch.thenYield().getOperands())
      results.push_back(ready(value));
    if (!branch.getElseRegion().empty()) {
      auto thenDrive = driveLanes, thenReadout = readoutLanes;
      auto thenLinks = std::move(linkReady);
      driveLanes = savedDrive;
      readoutLanes = savedReadout;
      linkReady = std::move(savedLinks);
      if (mlir::failed(forwardBlock(*branch.elseBlock())))
        return mlir::failure();
      for (auto [time, value] :
           llvm::zip(results, branch.elseYield().getOperands()))
        time = std::max(time, ready(value));
      mergeLanes(driveLanes, thenDrive, /*later=*/true);
      mergeLanes(readoutLanes, thenReadout, /*later=*/true);
      for (auto &[key, time] : thenLinks) {
        auto [it, inserted] = linkReady.try_emplace(key, time);
        if (!inserted)
          it->second = std::max(it->second, time);
      }
    }
    if (yieldsAfterCondition(branch))
      for (int64_t &time : results)
        time = std::max(time, floor);
    for (auto [result, time] : llvm::zip(branch.getResults(), results))
      readyAt[result] = time;
    floor = savedFloor;
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
      return it->second;
    mlir::Type type = value.getType();
    if (mlir::isa<pulse::MeasurementType>(type))
      return kNever;
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
    std::pair<mlir::Operation *, llvm::SmallVector<int64_t, 2>> key{op,
                                                                    iteration};
    auto it = forwardStarts.find(key);
    if (it != forwardStarts.end())
      return it->second;
    // The latest simulated iteration before this one, shifted by the period
    // of the innermost loop the forward pass extrapolated.
    auto before = forwardStarts.lower_bound(key);
    if (before == forwardStarts.begin())
      return 0;
    --before;
    if (before->first.first != op || iteration.empty())
      return 0;
    const auto &seen = before->first.second;
    auto loops = pulse::enclosingLoops(op, region->getParentOp());
    for (int64_t level = static_cast<int64_t>(iteration.size()) - 1; level >= 0;
         --level) {
      if (level >= static_cast<int64_t>(loops.size()) ||
          !std::equal(seen.begin(), seen.begin() + level, iteration.begin()))
        continue;
      for (const Extrapolation &e : forwardExtrapolations)
        if (e.loop == loops[level])
          return before->second + (iteration[level] - seen[level]) * e.period;
    }
    return before->second;
  }

  void backwardSubmit(classical::SubmitOp submit) {
    mlir::Operation *op = submit;
    int64_t t = timelineOf(op);
    int64_t duration = durationOf(op), budget = budgetOf(op);
    int64_t start;
    if (op->hasAttr("classical.eager")) {
      start = forwardStartOf(op);
    } else {
      int64_t link = late(submit.getUpdatedLink());
      if (isolated()) {
        auto it = linkLate.find(linkOf(op));
        link = it == linkLate.end() ? ends[t] : it->second;
      }
      int64_t end = std::min(link, ceilings[t]);
      int64_t needed = late(submit.getFuture());
      if (needed != kNever)
        end = std::min(end, needed - budget);
      start = end - duration;
    }
    start = takeSlotBackward(op, start, duration + budget,
                             op->hasAttr("classical.eager"));
    record(op, start);
    armStart = std::min(armStart, start);
    for (mlir::Value operand : op->getOperands())
      bound(operand, start);
    if (isolated()) {
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
    if (start == kNever || wait->hasAttr("classical.blocking"))
      start = forwardStartOf(wait);
    // A timeline's share of a link holds only the requests it times; an
    // answer it took late would hold its slot after its due time.
    else if (isolated())
      start = std::min(start, forwardStartOf(wait));
    start = std::min(start, ceilings[t]);
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
      bool isReadout = mlir::isa<pulse::ReadoutOp>(op);
      if (isReadout) {
        int64_t latency = resources.readoutLatency;
        if (resources.enabled)
          end -= latency;
        int64_t needed = late(mlir::cast<pulse::ReadoutOp>(op).getResult());
        if (needed != kNever)
          end = std::min(end, needed - latency);
      }
      if (mlir::Value tone = toneResult(op))
        end = std::min(end, late(tone));
      int64_t step = grid.enabled() ? grid.of(op->getOperand(0), lines) : 1;
      if (resources.enabled && !mlir::isa<pulse::WaitOp>(op)) {
        auto &lanes = isReadout ? readoutLanes[t] : driveLanes[t];
        auto lane = std::max_element(lanes.begin(), lanes.end());
        end = std::min(end, *lane);
        if (step > 1)
          end = pulse::LineGrid::roundDown(end - duration, step) + duration;
        *lane = end - duration - resources.switchPenalty;
      }
      int64_t start = end - duration;
      if (step > 1)
        start = pulse::LineGrid::roundDown(start, step);
      record(op, start);
      armStart = std::min(armStart, start);
      bound(op->getOperand(0), start);
      if (mlir::Value tone = toneOperand(op))
        bound(tone, start);
      if (classical)
        for (mlir::Value operand : op->getOperands())
          bound(operand, start);
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
      if (grid.enabled())
        time = pulse::LineGrid::roundDown(time, syncGrid(sync));
      for (mlir::Value line : sync.getLines())
        bound(line, time);
      armStart = std::min(armStart, time);
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
    if (grid.enabled())
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
    loop.getBody()->walk([&](mlir::Operation *inner) {
      for (mlir::Value result : inner->getResults())
        bodyValues.push_back(result);
      for (mlir::Region &region : inner->getRegions())
        for (mlir::Block &nested : region)
          for (mlir::Value arg : nested.getArguments())
            bodyValues.push_back(arg);
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
    for (int64_t k = count - 1; k >= 0; --k) {
      iteration.back() = k;
      for (mlir::Value value : bodyValues)
        lateAt.erase(value);
      for (mlir::Value arg : args)
        lateAt.erase(arg);
      for (auto [value, time] : llvm::zip(yield.getOperands(), carried))
        bound(value, time);
      auto ceilingsBefore = ceilings;
      auto linksBefore = linkLate;
      if (mlir::failed(backwardBlock(*loop.getBody())))
        return mlir::failure();
      llvm::SmallVector<int64_t> next;
      for (mlir::Value arg : args)
        next.push_back(late(arg));

      int64_t shift;
      bool shifted = uniformShift(carried, next, kinds, shift);
      bool steady = !resources.enabled && !lanesBound && shifted &&
                    lastShift == shift && shift <= 0 &&
                    shift % grid.lcm() == 0 &&
                    slotsHoldSteadyState(loop, -shift);
      lastShift = shifted ? std::optional<int64_t>(shift) : std::nullopt;
      llvm::SmallVector<int64_t> previous = std::move(carried);
      carried = std::move(next);
      if (steady && k > 0) {
        extrapolate(carried, previous, kinds, k * shift);
        for (auto [now, before] : llvm::zip(ceilings, ceilingsBefore))
          if (now != before)
            now += k * shift;
        for (auto &[key, time] : linkLate) {
          auto before = linksBefore.find(key);
          if (before == linksBefore.end() || before->second != time)
            time += k * shift;
        }
        extrapolations.push_back({loop, -shift});
        break;
      }
    }
    iteration.pop_back();
    for (auto [init, time] : llvm::zip(loop.getInitArgs(), carried))
      bound(init, time);
    return mlir::success();
  }

  mlir::LogicalResult backwardIf(mlir::scf::IfOp branch) {
    int64_t savedArmStart = armStart;
    armStart = kNever;
    auto savedDrive = driveLanes, savedReadout = readoutLanes;
    auto savedLinks = linkLate;
    for (auto [value, result] :
         llvm::zip(branch.thenYield().getOperands(), branch.getResults()))
      bound(value, late(result));
    if (mlir::failed(backwardBlock(*branch.thenBlock())))
      return mlir::failure();
    if (!branch.getElseRegion().empty()) {
      auto thenDrive = driveLanes, thenReadout = readoutLanes;
      auto thenLinks = std::move(linkLate);
      driveLanes = savedDrive;
      readoutLanes = savedReadout;
      linkLate = std::move(savedLinks);
      for (auto [value, result] :
           llvm::zip(branch.elseYield().getOperands(), branch.getResults()))
        bound(value, late(result));
      if (mlir::failed(backwardBlock(*branch.elseBlock())))
        return mlir::failure();
      mergeLanes(driveLanes, thenDrive, /*later=*/false);
      mergeLanes(readoutLanes, thenReadout, /*later=*/false);
      for (auto &[key, time] : thenLinks) {
        auto [it, inserted] = linkLate.try_emplace(key, time);
        if (!inserted)
          it->second = std::min(it->second, time);
      }
    }
    if (armStart != kNever) {
      for (mlir::Value measurement :
           conditionMeasurements(branch.getCondition()))
        bound(measurement, armStart);
      if (classical)
        bound(branch.getCondition(), armStart);
    }
    if (yieldsAfterCondition(branch)) {
      int64_t needed = kNever;
      for (mlir::Value result : branch.getResults())
        needed = std::min(needed, late(result));
      if (needed != kNever) {
        for (mlir::Value measurement :
             conditionMeasurements(branch.getCondition()))
          bound(measurement, needed);
        bound(branch.getCondition(), needed);
      }
    }
    armStart = std::min(savedArmStart, armStart);
    return mlir::success();
  }

  mlir::Region *region;
  Resources resources;
  llvm::DenseMap<mlir::Value, int64_t> readyAt;
  llvm::DenseMap<mlir::Value, int64_t> lateAt;
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
  /// With timelines, when each one's link is free (forward) or must be
  /// free from (backward), in its own order.
  std::map<LinkKey, int64_t> linkReady, linkLate;
  /// Request slots per link and timeline, each the time it is free
  /// (forward) or must be free from (backward).
  llvm::DenseMap<LinkKey, llvm::SmallVector<int64_t>> linkLanes;
  int64_t laneReset = 0;
  /// A link's slots delayed a request: loop iterations may differ, so they
  /// are all simulated.
  bool lanesBound = false;
  /// The requests of each block, by local start and by local end.
  llvm::DenseMap<mlir::Operation *, std::pair<llvm::SmallVector<BlockRequest>,
                                              llvm::SmallVector<BlockRequest>>>
      blockRequests;
  std::map<std::pair<mlir::Operation *, llvm::SmallVector<int64_t, 2>>, int64_t>
      forwardStarts;
  llvm::SmallVector<Extrapolation> forwardExtrapolations;
  // Lines whose controllers start operations only every few program cycles.
  pulse::LineGrid grid;
  pulse::AllocationTracer lines;
};

// ── Writing the schedule ─────────────────────────────────────────────────

/// Why an op's samples are not affine in its iterations.
struct NotAffine {
  mlir::scf::ForOp loop;
  bool peelFront = false;
  bool peelBack = false;
};

static std::optional<NotAffine>
writeTiming(mlir::Operation *op, mlir::Operation *root,
            llvm::ArrayRef<Sample> samples,
            llvm::ArrayRef<Extrapolation> extrapolations) {
  auto loops = pulse::enclosingLoops(op, root);
  mlir::MLIRContext *context = op->getContext();
  auto i64 = mlir::IntegerType::get(context, 64);
  int64_t duration = durationOf(op);
  // One attribute-dictionary rebuild per op.
  auto write = [&](std::optional<int64_t> start,
                   llvm::ArrayRef<int64_t> periods) {
    mlir::NamedAttrList attributes(op->getAttrDictionary());
    if (start)
      attributes.set("start_cycles", mlir::IntegerAttr::get(i64, *start));
    else
      attributes.erase("start_cycles");
    attributes.set("duration_cycles", mlir::IntegerAttr::get(i64, duration));
    if (periods.empty())
      attributes.erase("period_cycles");
    else
      attributes.set("period_cycles",
                     mlir::DenseI64ArrayAttr::get(context, periods));
    op->setAttrs(attributes.getDictionary(context));
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
        periods[level] = e.period;
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
      llvm::SmallVector<int64_t, 2> probe(sample.iteration.begin(),
                                          sample.iteration.end());
      llvm::SmallVector<std::pair<int64_t, int64_t>> line;
      for (int64_t k = 0; k < trips; ++k) {
        probe[level] = k;
        auto found = at.find(probe);
        if (found != at.end())
          line.push_back({k, found->second});
      }
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
      return NotAffine{loops[level], front, !front && back};
    }
    return NotAffine{loops.back(), false, false};
  }
  for (const Extrapolation &e : extrapolations)
    for (auto [level, loop] : llvm::enumerate(loops))
      if (e.loop == loop && periods[level] != e.period)
        return NotAffine{loop, false, false};

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
        problem = NotAffine{loop, true, false};
    };
    forEachScheduled(loop.getRegion(), [&](mlir::Operation *timed) {
      auto periods =
          timed->getAttrOfType<mlir::DenseI64ArrayAttr>("period_cycles");
      if (problem || !periods || static_cast<size_t>(periods.size()) <= level)
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
    if (auto start = op->getAttrOfType<mlir::IntegerAttr>("start_cycles"))
      earliest = std::min(earliest, start.getInt());
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
  mlir::MLIRContext *context = block.getContext();
  auto start = block->getAttrOfType<mlir::IntegerAttr>("start_cycles");
  llvm::SmallVector<int64_t> outer;
  if (auto periods =
          block->getAttrOfType<mlir::DenseI64ArrayAttr>("period_cycles"))
    outer.assign(periods.asArrayRef().begin(), periods.asArrayRef().end());
  block.getBody().walk([&](mlir::Operation *op) {
    if (!isTimedOp(op))
      return;
    auto local = op->getAttrOfType<mlir::IntegerAttr>("start_cycles");
    if (!start || !local) {
      // The block or this op never runs.
      op->removeAttr("start_cycles");
      op->removeAttr("period_cycles");
      return;
    }
    op->setAttr("start_cycles",
                mlir::IntegerAttr::get(local.getType(),
                                       start.getInt() + local.getInt()));
    llvm::SmallVector<int64_t> periods = outer;
    if (auto inner =
            op->getAttrOfType<mlir::DenseI64ArrayAttr>("period_cycles"))
      periods.append(inner.asArrayRef().begin(), inner.asArrayRef().end());
    if (periods.empty())
      op->removeAttr("period_cycles");
    else
      op->setAttr("period_cycles",
                  mlir::DenseI64ArrayAttr::get(context, periods));
  });
  inlineBody(block);
}

// ── Driver ───────────────────────────────────────────────────────────────

static mlir::LogicalResult
scheduleRegion(mlir::Region &region, mlir::Operation *root, Resources resources,
               bool isAlap, bool classical, const Timelines &timelines,
               const pulse::LineGrid &grid) {
  for (pulse::ScheduleBlockOp block : outermostBlocks(region)) {
    if (mlir::failed(scheduleRegion(block.getBody(), block, Resources{},
                                    block.getPolicy() == "alap", classical,
                                    timelines, grid)))
      return mlir::failure();
    writeShape(block);
  }

  // Each round either writes an affine schedule or removes an iteration from
  // a loop, so this terminates.
  while (true) {
    Scheduler scheduler(region, resources, classical, grid, &timelines);
    if (mlir::failed(scheduler.forward()))
      return mlir::failure();
    if (isAlap) {
      if (mlir::failed(scheduler.backward()))
        return mlir::failure();
      // Handing out resources in reverse program order can need time before
      // zero; the forward resource schedule is always feasible.
      bool feasible = llvm::all_of(scheduler.samples, [](const auto &entry) {
        return llvm::all_of(entry.second,
                            [](const Sample &s) { return s.start >= 0; });
      });
      if (!feasible) {
        scheduler = Scheduler(region, resources, classical, grid, &timelines);
        if (mlir::failed(scheduler.forward()))
          return mlir::failure();
      }
    }

    std::optional<NotAffine> problem;
    forEachScheduled(region, [&](mlir::Operation *op) {
      if (problem)
        return;
      auto it = scheduler.samples.find(op);
      llvm::ArrayRef<Sample> samples;
      if (it != scheduler.samples.end())
        samples = it->second;
      problem = writeTiming(op, root, samples, scheduler.extrapolations);
    });
    if (!problem)
      problem = periodMismatch(region, root);
    if (!problem)
      break;
    if (problem->peelFront)
      peelFront(problem->loop);
    else if (problem->peelBack)
      peelBack(problem->loop);
    else
      unrollFully(problem->loop);
  }

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
  bool classical = classical::annotateClassicalCalls(function);
  if (classical)
    for (pulse::ScheduleBlockOp block : outermostBlocks(function.getBody())) {
      mlir::WalkResult found = block.walk([](mlir::Operation *op) {
        if (op->hasAttr("classical.variable")) {
          op->emitError(classical::kVariableWaitInBlock);
          return mlir::WalkResult::interrupt();
        }
        if (mlir::isa<classical::WaitOp>(op) &&
            op->hasAttr("classical.blocking")) {
          op->emitError(classical::kBlockingWaitInBlock);
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      });
      if (found.wasInterrupted())
        return mlir::failure();
    }
  Timelines timelines;
  if (classical && TimelineAnalysis(function, resources).run(timelines))
    classical::annotateClassicalCalls(function);
  // Ops that read a value a measurement selects wait for it, as ops that
  // read an answer do.
  bool data = classical || function->hasAttr("pulse.selected_values");
  pulse::LineGrid grid(function->getParentOfType<mlir::ModuleOp>());
  return scheduleRegion(function.getBody(), function, resources, isAlap, data,
                        timelines, grid);
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
