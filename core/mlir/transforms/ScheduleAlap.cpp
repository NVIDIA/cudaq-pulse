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
// Results are written as `start_vtu` / `duration_vtu`, plus `period_vtu` for
// ops in loops (see Analysis/Timing.h). When an op's start is not affine in
// its iteration (a loop's first or last iterations behave differently), that
// iteration is peeled off the loop and the function is scheduled again.

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include <algorithm>
#include <limits>

#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseOps.h.inc"

#include "cudaq-pulse/Analysis/Timing.h"

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
    return static_cast<int64_t>(op.getSamples().size());
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

static int64_t getWaitDuration(pulse::WaitOp wait) {
  if (auto conversion =
          wait.getDuration().getDefiningOp<pulse::DurationFromIntOp>())
    return traceConstantI64(conversion.getCycles()).value_or(0);
  return 0;
}

static bool isTimedOp(mlir::Operation *op) {
  return mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp>(op);
}

static int64_t durationOf(mlir::Operation *op) {
  if (auto drive = mlir::dyn_cast<pulse::DriveOp>(op))
    return getWaveformDuration(drive.getPulse());
  if (auto readout = mlir::dyn_cast<pulse::ReadoutOp>(op))
    return getWaveformDuration(readout.getPulse());
  return getWaitDuration(mlir::cast<pulse::WaitOp>(op));
}

/// Lines and measurement results: the values that become available at a time.
static bool isTimedType(mlir::Type type) {
  return mlir::isa<pulse::DriveLineType, pulse::ReadoutLineType,
                   pulse::MeasurementType>(type);
}

static bool isLineType(mlir::Type type) {
  return mlir::isa<pulse::DriveLineType, pulse::ReadoutLineType>(type);
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

class Scheduler {
public:
  Scheduler(mlir::func::FuncOp function, Resources resources)
      : function(function), resources(resources) {}

  mlir::LogicalResult forward() {
    resetLanes(0);
    if (mlir::failed(forwardBlock(function.getBody().front())))
      return mlir::failure();
    return mlir::success();
  }

  mlir::LogicalResult backward() {
    samples.clear();
    extrapolations.clear();
    resetLanes(makespan);
    return backwardBlock(function.getBody().front());
  }

  int64_t makespan = 0;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<Sample, 1>> samples;
  llvm::SmallVector<Extrapolation> extrapolations;

private:
  // ── Shared state ──────────────────────────────────────────────────────

  void resetLanes(int64_t value) {
    if (!resources.enabled)
      return;
    driveLanes.assign(static_cast<size_t>(resources.maxDrives), value);
    readoutLanes.assign(static_cast<size_t>(resources.maxReadouts), value);
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

  /// All carried lines advanced by the same `period`, twice in a row.
  static bool uniformShift(llvm::ArrayRef<int64_t> before,
                           llvm::ArrayRef<int64_t> after,
                           llvm::ArrayRef<bool> isLine, int64_t &period) {
    std::optional<int64_t> shift;
    for (auto [b, a, line] : llvm::zip(before, after, isLine)) {
      if (!line)
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

  // ── Forward (ASAP) ────────────────────────────────────────────────────

  int64_t ready(mlir::Value value) const { return readyAt.lookup(value); }

  void place(mlir::Operation *op, int64_t start) {
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
    noteEnd(lineEnd);
  }

  void noteEnd(int64_t end) {
    makespan = std::max(makespan, end);
    iterationEnd = std::max(iterationEnd, end);
  }

  mlir::LogicalResult forwardBlock(mlir::Block &block) {
    for (mlir::Operation &op : block.without_terminator())
      if (mlir::failed(forwardOp(&op)))
        return mlir::failure();
    return mlir::success();
  }

  mlir::LogicalResult forwardOp(mlir::Operation *op) {
    if (isTimedOp(op)) {
      int64_t start = std::max(ready(op->getOperand(0)), floor);
      if (resources.enabled && !mlir::isa<pulse::WaitOp>(op)) {
        auto &lanes = mlir::isa<pulse::DriveOp>(op) ? driveLanes : readoutLanes;
        auto lane = std::min_element(lanes.begin(), lanes.end());
        start = std::max(start, *lane);
        *lane = start + durationOf(op) + resources.switchPenalty;
      }
      place(op, start);
      return mlir::success();
    }
    if (auto sync = mlir::dyn_cast<pulse::SyncOp>(op)) {
      int64_t time = floor;
      for (mlir::Value line : sync.getLines())
        time = std::max(time, ready(line));
      for (mlir::Value result : sync.getResults())
        readyAt[result] = time;
      noteEnd(time);
      return mlir::success();
    }
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op))
      return forwardLoop(loop);
    if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op))
      return forwardIf(branch);
    // Any other op releases its timed results when its timed operands are.
    int64_t time = 0;
    for (mlir::Value operand : op->getOperands())
      if (isTimedType(operand.getType()))
        time = std::max(time, ready(operand));
    for (mlir::Value result : op->getResults())
      if (isTimedType(result.getType()))
        readyAt[result] = time;
    return mlir::success();
  }

  mlir::LogicalResult forwardLoop(mlir::scf::ForOp loop) {
    int64_t count;
    if (mlir::failed(tripCount(loop, count)))
      return mlir::failure();
    auto args = loop.getRegionIterArgs();
    auto yield =
        mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
    llvm::SmallVector<bool> isLine;
    llvm::SmallVector<int64_t> carried;
    for (mlir::Value init : loop.getInitArgs()) {
      isLine.push_back(isTimedType(init.getType()));
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

    iteration.push_back(0);
    std::optional<int64_t> lastShift;
    for (int64_t k = 0; k < count; ++k) {
      iteration.back() = k;
      int64_t savedEnd = iterationEnd;
      iterationEnd = 0;
      for (auto [arg, time] : llvm::zip(args, carried))
        readyAt[arg] = time;
      if (mlir::failed(forwardBlock(*loop.getBody())))
        return mlir::failure();
      llvm::SmallVector<int64_t> next;
      for (mlir::Value value : yield.getOperands())
        next.push_back(ready(value));
      int64_t thisEnd = iterationEnd;
      iterationEnd = std::max(savedEnd, iterationEnd);

      int64_t shift;
      bool steady = !resources.enabled &&
                    uniformShift(carried, next, isLine, shift) &&
                    lastShift == shift && shift >= 0 &&
                    llvm::all_of(next, [&](int64_t t) { return t >= outside; });
      lastShift = uniformShift(carried, next, isLine, shift)
                      ? std::optional<int64_t>(shift)
                      : std::nullopt;
      carried = std::move(next);
      if (steady && k + 1 < count) {
        int64_t remaining = count - 1 - k;
        for (auto [time, line] : llvm::zip(carried, isLine))
          if (line)
            time += remaining * shift;
        noteEnd(thisEnd + remaining * shift);
        extrapolations.push_back({loop, shift});
        break;
      }
    }
    iteration.pop_back();
    for (auto [result, time] : llvm::zip(loop.getResults(), carried))
      readyAt[result] = time;
    return mlir::success();
  }

  mlir::LogicalResult forwardIf(mlir::scf::IfOp branch) {
    int64_t savedFloor = floor;
    for (mlir::Value measurement : conditionMeasurements(branch.getCondition()))
      floor = std::max(floor, ready(measurement));

    auto savedDrive = driveLanes, savedReadout = readoutLanes;
    if (mlir::failed(forwardBlock(*branch.thenBlock())))
      return mlir::failure();
    llvm::SmallVector<int64_t> results;
    for (mlir::Value value : branch.thenYield().getOperands())
      results.push_back(ready(value));
    if (!branch.getElseRegion().empty()) {
      auto thenDrive = driveLanes, thenReadout = readoutLanes;
      driveLanes = savedDrive;
      readoutLanes = savedReadout;
      if (mlir::failed(forwardBlock(*branch.elseBlock())))
        return mlir::failure();
      for (auto [time, value] :
           llvm::zip(results, branch.elseYield().getOperands()))
        time = std::max(time, ready(value));
      for (auto [a, b] : llvm::zip(driveLanes, thenDrive))
        a = std::max(a, b);
      for (auto [a, b] : llvm::zip(readoutLanes, thenReadout))
        a = std::max(a, b);
    }
    for (auto [result, time] : llvm::zip(branch.getResults(), results))
      readyAt[result] = time;
    floor = savedFloor;
    return mlir::success();
  }

  // ── Backward (ALAP) ───────────────────────────────────────────────────

  /// The latest time `value` may become available without delaying the
  /// schedule. Lines left unconsumed may run until the makespan; nothing
  /// waits for an unused measurement.
  int64_t late(mlir::Value value) const {
    auto it = lateAt.find(value);
    if (it != lateAt.end())
      return it->second;
    return mlir::isa<pulse::MeasurementType>(value.getType()) ? kNever
                                                              : makespan;
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
    if (isTimedOp(op)) {
      int64_t duration = durationOf(op);
      int64_t end = late(op->getResult(0));
      bool isReadout = mlir::isa<pulse::ReadoutOp>(op);
      if (isReadout) {
        int64_t latency = resources.readoutLatency;
        if (resources.enabled)
          end -= latency;
        int64_t needed = late(mlir::cast<pulse::ReadoutOp>(op).getResult());
        if (needed != kNever)
          end = std::min(end, needed - latency);
      }
      if (resources.enabled && !mlir::isa<pulse::WaitOp>(op)) {
        auto &lanes = isReadout ? readoutLanes : driveLanes;
        auto lane = std::max_element(lanes.begin(), lanes.end());
        end = std::min(end, *lane);
        *lane = end - duration - resources.switchPenalty;
      }
      int64_t start = end - duration;
      record(op, start);
      armStart = std::min(armStart, start);
      bound(op->getOperand(0), start);
      return mlir::success();
    }
    if (auto sync = mlir::dyn_cast<pulse::SyncOp>(op)) {
      int64_t time = makespan;
      for (mlir::Value result : sync.getResults())
        time = std::min(time, late(result));
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
      if (isTimedType(result.getType()))
        time = std::min(time, late(result));
    if (time != kNever)
      for (mlir::Value operand : op->getOperands())
        if (isTimedType(operand.getType()))
          bound(operand, time);
    return mlir::success();
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
    llvm::SmallVector<bool> isLine;
    llvm::SmallVector<int64_t> carried;
    for (mlir::Value result : loop.getResults()) {
      isLine.push_back(isLineType(result.getType()));
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
      if (mlir::failed(backwardBlock(*loop.getBody())))
        return mlir::failure();
      llvm::SmallVector<int64_t> next;
      for (mlir::Value arg : args)
        next.push_back(late(arg));

      int64_t shift;
      bool shifted = uniformShift(carried, next, isLine, shift);
      bool steady =
          !resources.enabled && shifted && lastShift == shift && shift <= 0;
      lastShift = shifted ? std::optional<int64_t>(shift) : std::nullopt;
      carried = std::move(next);
      if (steady && k > 0) {
        for (auto [time, line] : llvm::zip(carried, isLine))
          if (line)
            time += k * shift;
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
    for (auto [value, result] :
         llvm::zip(branch.thenYield().getOperands(), branch.getResults()))
      bound(value, late(result));
    if (mlir::failed(backwardBlock(*branch.thenBlock())))
      return mlir::failure();
    if (!branch.getElseRegion().empty()) {
      auto thenDrive = driveLanes, thenReadout = readoutLanes;
      driveLanes = savedDrive;
      readoutLanes = savedReadout;
      for (auto [value, result] :
           llvm::zip(branch.elseYield().getOperands(), branch.getResults()))
        bound(value, late(result));
      if (mlir::failed(backwardBlock(*branch.elseBlock())))
        return mlir::failure();
      for (auto [a, b] : llvm::zip(driveLanes, thenDrive))
        a = std::min(a, b);
      for (auto [a, b] : llvm::zip(readoutLanes, thenReadout))
        a = std::min(a, b);
    }
    if (armStart != kNever)
      for (mlir::Value measurement :
           conditionMeasurements(branch.getCondition()))
        bound(measurement, armStart);
    armStart = std::min(savedArmStart, armStart);
    return mlir::success();
  }

  mlir::func::FuncOp function;
  Resources resources;
  llvm::DenseMap<mlir::Value, int64_t> readyAt;
  llvm::DenseMap<mlir::Value, int64_t> lateAt;
  llvm::SmallVector<int64_t> driveLanes, readoutLanes;
  llvm::SmallVector<int64_t, 2> iteration;
  int64_t floor = 0;
  int64_t iterationEnd = 0;
  int64_t armStart = kNever;
};

// ── Writing the schedule ─────────────────────────────────────────────────

/// Why an op's samples are not affine in its iterations.
struct NotAffine {
  mlir::scf::ForOp loop;
  bool peelFront = false;
  bool peelBack = false;
};

static std::optional<NotAffine>
writeTiming(mlir::Operation *op, llvm::ArrayRef<Sample> samples,
            llvm::ArrayRef<Extrapolation> extrapolations) {
  auto loops = pulse::enclosingLoops(op);
  auto i64 = mlir::IntegerType::get(op->getContext(), 64);
  int64_t duration = durationOf(op);
  if (samples.empty()) {
    // Never executed (inside a zero-trip loop).
    op->removeAttr("start_vtu");
    op->removeAttr("period_vtu");
    op->setAttr("duration_vtu", mlir::IntegerAttr::get(i64, duration));
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

  op->setAttr("start_vtu", mlir::IntegerAttr::get(i64, base));
  op->setAttr("duration_vtu", mlir::IntegerAttr::get(i64, duration));
  if (loops.empty())
    op->removeAttr("period_vtu");
  else
    op->setAttr("period_vtu",
                mlir::DenseI64ArrayAttr::get(op->getContext(), periods));
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

static mlir::LogicalResult schedule(mlir::func::FuncOp function,
                                    Resources resources, bool isAlap) {
  if (resources.enabled &&
      (resources.maxDrives <= 0 || resources.maxReadouts <= 0))
    return function.emitError("resource limits must be positive");

  // Each round either writes an affine schedule or removes an iteration from
  // a loop, so this terminates.
  while (true) {
    Scheduler scheduler(function, resources);
    if (mlir::failed(scheduler.forward()))
      return mlir::failure();
    if (isAlap && mlir::failed(scheduler.backward()))
      return mlir::failure();

    std::optional<NotAffine> problem;
    function.walk([&](mlir::Operation *op) {
      if (!isTimedOp(op) || problem)
        return;
      auto it = scheduler.samples.find(op);
      llvm::ArrayRef<Sample> samples;
      if (it != scheduler.samples.end())
        samples = it->second;
      problem = writeTiming(op, samples, scheduler.extrapolations);
    });
    if (!problem)
      return mlir::success();
    if (problem->peelFront)
      peelFront(problem->loop);
    else if (problem->peelBack)
      peelBack(problem->loop);
    else
      unrollFully(problem->loop);
  }
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

} // namespace

namespace pulse {
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
} // namespace pulse
