/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// The walk behind Lowered.export_controller() and the controller limit
// checks, included by bindings.cpp inside its anonymous namespace.
//
// The walk lists every timed op, sync and classical call with the frame of
// its tone and the branches it lies in. It runs in one of three ways:
//
// - Every: once per loop iteration (format /1).
// - Sampled: a loop whose iterations repeat (the same frequencies from
//   iteration 1 on, every op moving by one period, no iteration overlapping
//   the next, no reference into the previous iteration) is walked at
//   iterations 0, 1 and the last only; the records of iteration 1 stand for
//   iterations 1 to n - 2. The limit checks run on these records.
// - Rolled: such a loop becomes one record followed by its body at its first
//   iteration, with each phase's step per iteration; an iteration 0 that
//   differs is peeled off first (format /2).
//
// A loop that does not repeat, and one holding an epoch boundary, is walked
// at every iteration in every way, so all three describe the same program.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

struct ControllerRecord {
  using Condition = std::tuple<int64_t, int64_t, std::vector<int64_t>,
                               std::vector<int64_t>, std::string>;

  const char *kind = "";
  mlir::Operation *op = nullptr;
  int64_t line = -1;
  int64_t qudit = -1;
  std::optional<int64_t> start;
  std::optional<int64_t> duration;
  // The iteration of each enclosing loop; -1 for a loop kept rolled.
  std::vector<int64_t> iteration;
  double frequency = std::numeric_limits<double>::quiet_NaN();
  double phase = std::numeric_limits<double>::quiet_NaN();
  std::vector<Condition> conditions;
  std::vector<int64_t> joined;
  std::vector<int64_t> refs;
  std::string callee;
  std::string mode;
  double timeout = std::numeric_limits<double>::quiet_NaN();
  int64_t epoch = 0;
  // Sampled: the executions this record stands for, and (loop instance, 1
  // for its iteration 1, 2 for its last) for each sampled loop around it.
  int64_t weight = 1;
  std::vector<std::pair<int64_t, int8_t>> samples;
  // Rolled: the phase's step per iteration of each rolled loop around it,
  // outermost first. A loop record has its iterations, first iteration,
  // period, the number of records of its body after it, and the end of its
  // last iteration.
  std::vector<double> phaseSteps;
  int64_t trips = 0;
  int64_t first = 0;
  int64_t period = 0;
  int64_t size = 0;
  std::optional<int64_t> end;

  std::optional<int64_t> stop() const {
    if (end)
      return end;
    if (start && duration)
      return *start + *duration;
    return std::nullopt;
  }
  bool played() const {
    return kind[0] == 'd' || (kind[0] == 'r' && kind[2] == 'a');
  }
};

struct ControllerWalk {
  using WaveformNode =
      std::tuple<std::string, std::vector<double>, std::vector<int64_t>>;
  using Condition = ControllerRecord::Condition;
  enum class Walk { Every, Sampled, Rolled };

  /// A tone's frame as the simulator tracks it (pulse::ToneState): `phase`
  /// at `timeNs`, the end of its latest drive; a tone detuned from its
  /// frame frequency gains phase from there on.
  struct Frame {
    double frequencyHz = 0.0;
    double frameFrequencyHz = 0.0;
    double phase = 0.0;
    double timeNs = 0.0;

    double phaseAt(double timeNs) const {
      double detuning = frequencyHz - frameFrequencyHz;
      if (detuning == 0.0)
        return phase;
      return phase + detuning * (2.0 * M_PI) * 1.0e-9 * (timeNs - this->timeNs);
    }
    /// Whether two arms leave the frame the same: without a detuning, the
    /// time of the last drive does not matter.
    bool operator==(const Frame &other) const {
      return same(frequencyHz, other.frequencyHz) &&
             same(frameFrequencyHz, other.frameFrequencyHz) &&
             same(phase, other.phase) &&
             (same(timeNs, other.timeNs) || frequencyHz == frameFrequencyHz);
    }
  };

  static bool same(double a, double b) {
    return a == b || (std::isnan(a) && std::isnan(b));
  }

  ControllerWalk(double clockGHz, Walk walk) : clockGHz(clockGHz), walk(walk) {}

  double clockGHz;
  Walk walk;
  llvm::DenseMap<mlir::Operation *, int64_t> lineIndex;
  std::vector<ControllerRecord> result;

  static constexpr double kUnknown = std::numeric_limits<double>::quiet_NaN();

  /// A number computed from constants alone, else NaN.
  static double number(mlir::Value value) {
    if (auto constant = pulse::constantNumber(value))
      return *constant;
    if (auto affine = pulse::affineNumber(value);
        affine && affine->isConstant())
      return affine->evaluate({});
    return kUnknown;
  }

  pulse::AllocationTracer trace;
  pulse::ToneRoots toneRoots;
  llvm::SmallVector<int64_t, 2> iteration;
  std::vector<Condition> branches;
  int64_t nextBranch = 0;
  std::map<mlir::Operation *, Frame> frames;
  // The time each line has reached (the end of its latest event), or none
  // when an event on it is not scheduled.
  std::map<mlir::Operation *, std::optional<int64_t>> reached;
  // The record of the latest execution of each readout and classical wait.
  llvm::DenseMap<mlir::Operation *, int64_t> latest;
  // Sampled: the sampled loops open now.
  std::vector<std::pair<int64_t, int8_t>> sampling;
  int64_t nextInstance = 0;
  llvm::DenseMap<mlir::Operation *, bool> boundaries;
  // The epoch of each event, when the program has classical calls. A loop
  // holding an epoch boundary is walked at every iteration, so the walk
  // visits the boundaries in program order, once per execution.
  std::optional<EpochWalk> epochs;

  static Frame initialFrame(mlir::Operation *root) {
    if (!root)
      return {};
    if (auto frequency = root->getAttrOfType<mlir::FloatAttr>("frequency_hz"))
      return {frequency.getValueAsDouble(), frequency.getValueAsDouble(), 0.0,
              0.0};
    if (root->getName().getStringRef() == "pulse.tone" &&
        root->getNumOperands() == 2) {
      double frequency = number(root->getOperand(0));
      return {frequency, frequency, number(root->getOperand(1)), 0.0};
    }
    return {};
  }

  Frame &frameOf(mlir::Value tone) {
    mlir::Operation *root = toneRoots(tone);
    auto [it, inserted] = frames.try_emplace(root);
    if (inserted)
      it->second = initialFrame(root);
    return it->second;
  }

  /// The waveform expression of `value`, children before their parent.
  static int64_t waveform(mlir::Value value, std::vector<WaveformNode> &nodes) {
    mlir::Operation *def = value.getDefiningOp();
    if (!def) {
      nodes.emplace_back("argument", std::vector<double>{},
                         std::vector<int64_t>{});
      return static_cast<int64_t>(nodes.size()) - 1;
    }
    std::vector<double> numbers;
    std::vector<int64_t> children;
    for (mlir::Value operand : def->getOperands()) {
      if (mlir::isa<pulse::WaveformType>(operand.getType()))
        children.push_back(waveform(operand, nodes));
      else
        numbers.push_back(number(operand));
    }
    if (auto samples = def->getAttrOfType<mlir::ArrayAttr>("samples"))
      for (mlir::Attribute sample : samples)
        if (auto number = mlir::dyn_cast<mlir::FloatAttr>(sample))
          numbers.push_back(number.getValueAsDouble());
    std::string name = def->getName().stripDialect().str();
    if (auto custom = def->getAttrOfType<mlir::FlatSymbolRefAttr>("callee"))
      name += ":" + custom.getValue().str();
    nodes.emplace_back(name, std::move(numbers), std::move(children));
    return static_cast<int64_t>(nodes.size()) - 1;
  }

  /// The readout records and answer records `value` is computed from.
  void inputs(mlir::Value value, std::vector<int64_t> &readouts,
              std::vector<int64_t> &answers) {
    llvm::SmallVector<mlir::Value> work{value};
    llvm::DenseSet<mlir::Value> seen;
    while (!work.empty()) {
      mlir::Value v = work.pop_back_val();
      if (!seen.insert(v).second)
        continue;
      mlir::Operation *def = v.getDefiningOp();
      if (!def || mlir::isa<classical::SubmitOp>(def))
        continue;
      if (mlir::isa<pulse::ReadoutOp, classical::WaitOp>(def)) {
        auto found = latest.find(def);
        if (found != latest.end()) {
          auto &list = mlir::isa<pulse::ReadoutOp>(def) ? readouts : answers;
          if (llvm::find(list, found->second) == list.end())
            list.push_back(found->second);
        }
        continue;
      }
      if (def->getNumRegions() == 0)
        work.append(def->getOperands().begin(), def->getOperands().end());
    }
  }

  int64_t lineOf(mlir::Operation *allocation) const {
    return allocation ? lineIndex.lookup(allocation) : -1;
  }

  ControllerRecord &record(const char *kind, mlir::Operation *op) {
    ControllerRecord &r = result.emplace_back();
    r.kind = kind;
    r.op = op;
    r.iteration.assign(iteration.begin(), iteration.end());
    r.conditions = branches;
    if (walk == Walk::Sampled)
      r.samples = sampling;
    if (epochs)
      r.epoch = epochs->visit(op, iteration);
    return r;
  }

  void block(mlir::Block &body) {
    for (mlir::Operation &op : body)
      step(&op);
  }

  void timed(mlir::Operation *op) {
    mlir::Operation *allocation = trace(op->getOperand(0));
    int64_t qubit = -1;
    if (allocation)
      if (auto attr = allocation->getAttrOfType<mlir::IntegerAttr>("qubit"))
        qubit = attr.getInt();
    auto start = pulse::scheduledStart(op, iteration);
    auto duration = pulse::scheduledDuration(op);
    double frequency = kUnknown, phase = kUnknown;
    if (!mlir::isa<pulse::WaitOp>(op)) {
      Frame &frame = frameOf(op->getOperand(2));
      bool drive = mlir::isa<pulse::DriveOp>(op);
      double offset = 0.0;
      if (drive) {
        if (auto shift =
                op->getAttrOfType<mlir::FloatAttr>("frame_phase_offset"))
          frame.phase += shift.getValueAsDouble();
        if (auto own = op->getAttrOfType<mlir::FloatAttr>("phase_offset"))
          offset = own.getValueAsDouble();
      }
      frequency = frame.frequencyHz;
      if (start)
        phase = frame.phaseAt(static_cast<double>(*start) / clockGHz) + offset;
      if (drive) {
        if (start && duration) {
          double endNs = static_cast<double>(*start + *duration) / clockGHz;
          frame.phase = frame.phaseAt(endNs);
          frame.timeNs = endNs;
        } else {
          frame = {kUnknown, kUnknown, kUnknown, frame.timeNs};
        }
      }
    }
    if (mlir::isa<pulse::ReadoutOp>(op))
      latest[op] = static_cast<int64_t>(result.size());
    auto line = reached.try_emplace(allocation, 0).first;
    if (start && duration && line->second)
      line->second = std::max(*line->second, *start + *duration);
    else
      line->second = std::nullopt;
    const char *kind = mlir::isa<pulse::DriveOp>(op)     ? "drive"
                       : mlir::isa<pulse::ReadoutOp>(op) ? "readout"
                                                         : "wait";
    ControllerRecord &r = record(kind, op);
    r.line = lineOf(allocation);
    r.qudit = qubit;
    r.start = start;
    r.duration = duration;
    r.frequency = frequency;
    r.phase = phase;
  }

  void sync(pulse::SyncOp op) {
    std::vector<int64_t> lines;
    std::optional<int64_t> time = 0;
    for (mlir::Value value : op.getLines()) {
      mlir::Operation *allocation = trace(value);
      lines.push_back(lineOf(allocation));
      auto found = reached.find(allocation);
      if (found == reached.end())
        continue;
      if (!found->second || !time)
        time = std::nullopt;
      else
        time = std::max(*time, *found->second);
    }
    if (time)
      for (mlir::Value value : op.getLines())
        reached[trace(value)] = time;
    ControllerRecord &r = record("sync", op);
    r.start = time;
    r.duration = time ? std::optional<int64_t>(0) : std::optional<int64_t>();
    r.joined = std::move(lines);
  }

  /// A submission (with the readout records whose measurements it sends) or
  /// the wait for its answer, on its link.
  void call(mlir::Operation *op) {
    bool submit = mlir::isa<classical::SubmitOp>(op);
    std::vector<int64_t> refs, answers;
    if (submit)
      for (mlir::Value operand : op->getOperands())
        inputs(operand, refs, answers);
    std::string callee;
    if (auto symbol = op->getAttrOfType<mlir::FlatSymbolRefAttr>("callee"))
      callee = symbol.getValue().str();
    std::string mode;
    double timeoutNs = kUnknown;
    if (!submit) {
      latest[op] = static_cast<int64_t>(result.size());
      mode = op->hasAttr("classical.blocking") ? "blocking" : "bounded";
      if (auto timeout = op->getAttrOfType<mlir::FloatAttr>("timeout_ns")) {
        mode = "timeout";
        timeoutNs = timeout.getValueAsDouble();
      }
    }
    ControllerRecord &r = record(submit ? "submit" : "answer", op);
    r.line = classical::intAttr(op, "classical.device");
    r.start = pulse::scheduledStart(op, iteration);
    r.duration = pulse::scheduledDuration(op);
    r.refs = std::move(refs);
    r.callee = std::move(callee);
    r.mode = std::move(mode);
    r.timeout = timeoutNs;
  }

  void step(mlir::Operation *op) {
    llvm::StringRef name = op->getName().getStringRef();
    if (mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp>(op)) {
      timed(op);
      return;
    }
    if (auto barrier = mlir::dyn_cast<pulse::SyncOp>(op)) {
      sync(barrier);
      return;
    }
    if (mlir::isa<classical::SubmitOp, classical::WaitOp>(op)) {
      call(op);
      return;
    }
    if (name == "pulse.shift_phase" || name == "pulse.set_phase" ||
        name == "pulse.shift_frequency" || name == "pulse.set_frequency") {
      double value = number(op->getOperand(1));
      Frame &frame = frameOf(op->getOperand(0));
      if (name == "pulse.shift_phase")
        frame.phase += value;
      else if (name == "pulse.set_phase")
        frame.phase = value;
      else if (name == "pulse.shift_frequency")
        frame.frequencyHz += value;
      else
        frame.frequencyHz = value;
      return;
    }
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
      int64_t trips = pulse::constantTripCount(loop).value_or(0);
      if (walk == Walk::Every || trips < 2 || holdsBoundary(loop))
        everyIteration(loop, trips);
      else if (walk == Walk::Sampled)
        sampled(loop, trips);
      else
        rolled(loop, trips);
      return;
    }
    if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
      mlir::Value condition = branch.getCondition();
      std::vector<int64_t> readouts, answers;
      inputs(condition, readouts, answers);
      std::string test = "expression";
      if (mlir::Operation *def = condition.getDefiningOp()) {
        if (mlir::isa<pulse::IsSetOp>(def) && readouts.size() == 1)
          test = "is_set";
        else if (mlir::isa<classical::WaitOp>(def) && answers.size() == 1)
          test = "answer";
      }
      int64_t id = nextBranch++;
      auto framesBefore = frames;
      auto reachedBefore = reached;
      branches.emplace_back(id, 1, std::move(readouts), std::move(answers),
                            std::move(test));
      block(*branch.thenBlock());
      auto framesThen = std::move(frames);
      auto reachedThen = std::move(reached);
      frames = std::move(framesBefore);
      reached = std::move(reachedBefore);
      std::get<1>(branches.back()) = 0;
      if (!branch.getElseRegion().empty())
        block(*branch.elseBlock());
      branches.pop_back();
      mergeFrames(framesThen);
      for (auto &[allocation, time] : reachedThen) {
        auto [it, inserted] = reached.try_emplace(allocation, time);
        if (inserted)
          continue;
        if (!time || !it->second)
          it->second = std::nullopt;
        else
          it->second = std::max(*it->second, *time);
      }
      return;
    }
    for (mlir::Region &region : op->getRegions())
      for (mlir::Block &nested : region)
        block(nested);
  }

  /// The frames after a branch: those both arms leave the same, else
  /// unknown from here on.
  void mergeFrames(const std::map<mlir::Operation *, Frame> &other) {
    for (auto &[root, frame] : other)
      if (!frames.count(root))
        frames[root] = initialFrame(root);
    for (auto &[root, frame] : frames) {
      auto found = other.find(root);
      Frame theirs = found == other.end() ? initialFrame(root) : found->second;
      if (!(theirs == frame))
        frame = {kUnknown, kUnknown, kUnknown, frame.timeNs};
      else
        frame.timeNs = std::max(frame.timeNs, theirs.timeNs);
    }
  }

  // ── Loops ────────────────────────────────────────────────────────────

  void everyIteration(mlir::scf::ForOp loop, int64_t trips) {
    iteration.push_back(0);
    for (int64_t k = 0; k < trips; ++k) {
      iteration.back() = k;
      block(*loop.getBody());
    }
    iteration.pop_back();
  }

  /// Whether `loop` holds a variable or blocking wait, which ends an epoch
  /// in every iteration.
  bool holdsBoundary(mlir::scf::ForOp loop) {
    auto [it, inserted] = boundaries.try_emplace(loop, false);
    if (inserted)
      loop->walk([&](mlir::Operation *op) {
        if (op->hasAttr("classical.variable") ||
            op->hasAttr("classical.blocking"))
          it->second = true;
      });
    return it->second;
  }

  struct State {
    std::map<mlir::Operation *, Frame> frames;
    std::map<mlir::Operation *, std::optional<int64_t>> reached;
    llvm::DenseMap<mlir::Operation *, int64_t> latest;
    int64_t nextBranch;
    size_t size;
  };

  State save() const {
    return {frames, reached, latest, nextBranch, result.size()};
  }

  void restore(State &state) {
    frames = std::move(state.frames);
    reached = std::move(state.reached);
    latest = std::move(state.latest);
    nextBranch = state.nextBranch;
    result.resize(state.size);
  }

  /// Whether two frame states have the same frequencies, so that the body
  /// moves every phase by the same amount from one to the next.
  static bool sameFrequencies(const std::map<mlir::Operation *, Frame> &a,
                              const std::map<mlir::Operation *, Frame> &b) {
    auto get = [](const std::map<mlir::Operation *, Frame> &frames,
                  mlir::Operation *root) {
      auto found = frames.find(root);
      return found == frames.end() ? initialFrame(root) : found->second;
    };
    for (const auto *frames : {&a, &b})
      for (const auto &[root, frame] : *frames) {
        Frame x = get(a, root), y = get(b, root);
        if (!same(x.frequencyHz, y.frequencyHz) ||
            !same(x.frameFrequencyHz, y.frameFrequencyHz))
          return false;
      }
    return true;
  }

  /// The period by which records [b, e) follow records [a, b), or none when
  /// they do not match one to one, move by different amounts, or (with
  /// `apart`) one iteration overlaps the next.
  std::optional<int64_t> periodOf(size_t a, size_t b, size_t e, bool apart) {
    if (b - a != e - b)
      return std::nullopt;
    std::optional<int64_t> period;
    std::optional<int64_t> earliest, latestEnd;
    for (size_t i = 0; i < b - a; ++i) {
      const ControllerRecord &x = result[a + i], &y = result[b + i];
      if (std::strcmp(x.kind, y.kind) != 0 || x.op != y.op ||
          x.start.has_value() != y.start.has_value() || x.trips != y.trips ||
          x.first != y.first || x.size != y.size ||
          x.phaseSteps.size() != y.phaseSteps.size())
        return std::nullopt;
      if (!x.start)
        continue;
      int64_t shift = *y.start - *x.start;
      if (period && *period != shift)
        return std::nullopt;
      period = shift;
      earliest = earliest ? std::min(*earliest, *x.start) : *x.start;
      if (auto stop = x.stop())
        latestEnd = latestEnd ? std::max(*latestEnd, *stop) : *stop;
    }
    if (!period)
      return int64_t{0};
    if (apart && latestEnd && *latestEnd - *earliest > *period)
      return std::nullopt;
    return period;
  }

  /// Whether a record of [b, e) refers to one of [a, b): an iteration that
  /// uses the previous one's readouts or answers.
  bool refersBack(size_t a, size_t b, size_t e) const {
    auto into = [&](const std::vector<int64_t> &refs) {
      return llvm::any_of(refs, [&](int64_t ref) {
        return ref >= static_cast<int64_t>(a) && ref < static_cast<int64_t>(b);
      });
    };
    for (size_t i = b; i < e; ++i) {
      if (into(result[i].refs))
        return true;
      for (const Condition &c : result[i].conditions)
        if (into(std::get<2>(c)) || into(std::get<3>(c)))
          return true;
    }
    return false;
  }

  void sampled(mlir::scf::ForOp loop, int64_t trips) {
    if (trips < 3)
      return everyIteration(loop, trips);
    State entry = save();
    int64_t instance = nextInstance++;
    iteration.push_back(0);
    size_t a = result.size();
    block(*loop.getBody());
    auto after = frames;
    size_t b = result.size();
    iteration.back() = 1;
    sampling.emplace_back(instance, 1);
    block(*loop.getBody());
    sampling.pop_back();
    size_t e = result.size();
    if (!sameFrequencies(after, frames) || !periodOf(a, b, e, true) ||
        refersBack(a, b, e)) {
      iteration.pop_back();
      restore(entry);
      return everyIteration(loop, trips);
    }
    for (size_t i = b; i < e; ++i)
      result[i].weight *= trips - 2;
    iteration.back() = trips - 1;
    sampling.emplace_back(instance, 2);
    block(*loop.getBody());
    sampling.pop_back();
    iteration.pop_back();
  }

  static double extend(double before, double after, int64_t times) {
    if (std::isnan(after) || std::isnan(before))
      return after;
    return after + static_cast<double>(times) * (after - before);
  }

  void rolled(mlir::scf::ForOp loop, int64_t trips) {
    State entry = save();
    size_t depth = iteration.size();
    iteration.push_back(0);
    auto frames0 = frames;
    size_t a0 = result.size();
    block(*loop.getBody());
    // The states before and after the first iteration exported rolled.
    auto framesFirst = frames0, framesNext = frames;
    auto reachedFirst = entry.reached, reachedNext = reached;
    auto latestNext = latest;
    size_t a1 = result.size();
    iteration.back() = 1;
    block(*loop.getBody());
    size_t a2 = result.size();
    int64_t first = 0;
    size_t a = a0, b = a1, e = a2;
    if (!sameFrequencies(framesFirst, framesNext)) {
      first = 1;
      bool repeats = trips >= 3 && sameFrequencies(framesNext, frames);
      if (repeats) {
        framesFirst = std::move(framesNext);
        framesNext = frames;
        reachedFirst = std::move(reachedNext);
        reachedNext = reached;
        latestNext = latest;
        iteration.back() = 2;
        block(*loop.getBody());
        a = a1, b = a2, e = result.size();
      } else {
        e = 0;
      }
    }
    std::optional<int64_t> period =
        e ? periodOf(a, b, e, false) : std::optional<int64_t>();
    std::vector<double> steps;
    for (size_t i = 0; period && i < b - a; ++i) {
      const ControllerRecord &x = result[a + i], &y = result[b + i];
      double step = y.phase - x.phase;
      if (!same(x.frequency, y.frequency) ||
          std::isnan(x.phase) != std::isnan(y.phase))
        period.reset();
      for (size_t l = 0; period && l < x.phaseSteps.size(); ++l)
        if (!same(x.phaseSteps[l], y.phaseSteps[l]))
          period.reset();
      steps.push_back(step);
    }
    if (!period) {
      iteration.pop_back();
      restore(entry);
      return everyIteration(loop, trips);
    }
    // The state after the last iteration: every number moves by the same
    // amount per iteration from the first exported one on.
    int64_t more = trips - first - 2;
    auto framesAfter = frames;
    for (auto &[root, frame] : framesAfter) {
      auto found = framesNext.find(root);
      Frame next =
          found == framesNext.end() ? initialFrame(root) : found->second;
      auto was = framesFirst.find(root);
      Frame before =
          was == framesFirst.end() ? initialFrame(root) : was->second;
      frame = next;
      frame.phase = extend(before.phase, next.phase, more + 1);
      frame.timeNs = extend(before.timeNs, next.timeNs, more + 1);
    }
    for (auto &[allocation, time] : reached) {
      auto next = reachedNext.find(allocation);
      auto before = reachedFirst.find(allocation);
      std::optional<int64_t> now =
          next == reachedNext.end() ? time : next->second;
      if (now && before != reachedFirst.end() && before->second)
        now = *now + (more + 1) * (*now - *before->second);
      time = now;
    }
    frames = std::move(framesAfter);
    result.resize(b);
    ControllerRecord node;
    node.kind = "loop";
    node.op = loop;
    node.iteration.assign(iteration.begin(), iteration.begin() + depth);
    node.conditions = branches;
    node.trips = trips - first;
    node.first = first;
    node.period = *period;
    node.size = static_cast<int64_t>(b - a);
    for (size_t i = a; i < b; ++i) {
      ControllerRecord &r = result[i];
      r.iteration[depth] = -1;
      r.phaseSteps.insert(r.phaseSteps.begin(), steps[i - a]);
      if (r.start && !node.start)
        node.start = r.start;
      else if (r.start)
        node.start = std::min(*node.start, *r.start);
      if (auto stop = r.stop()) {
        int64_t last = *stop + (node.trips - 1) * node.period;
        node.end = node.end ? std::max(*node.end, last) : last;
      }
    }
    node.duration = node.start && node.end
                        ? std::optional<int64_t>(*node.end - *node.start)
                        : std::nullopt;
    // The body moves one place on; references into it follow.
    auto shift = [&](int64_t &ref) {
      if (ref >= static_cast<int64_t>(a))
        ++ref;
    };
    for (size_t i = a; i < b; ++i) {
      ControllerRecord &r = result[i];
      for (int64_t &ref : r.refs)
        shift(ref);
      for (Condition &c : r.conditions) {
        for (int64_t &ref : std::get<2>(c))
          shift(ref);
        for (int64_t &ref : std::get<3>(c))
          shift(ref);
      }
    }
    for (auto &[op, ref] : latestNext)
      shift(ref);
    latest = std::move(latestNext);
    result.insert(result.begin() + static_cast<std::ptrdiff_t>(a),
                  std::move(node));
    iteration.pop_back();
  }
};

// ── Waveforms ─────────────────────────────────────────────────────────────

/// The distinct waveforms the drives and readouts of a walk play, and which
/// one each op plays: ops with the same expression share one.
struct ControllerWaveforms {
  using Nodes = std::vector<ControllerWalk::WaveformNode>;

  std::vector<Nodes> distinct;
  llvm::DenseMap<mlir::Operation *, int64_t> of;

  void add(mlir::Operation *op) {
    if (of.count(op))
      return;
    Nodes nodes;
    ControllerWalk::waveform(op->getOperand(1), nodes);
    std::string key;
    for (const auto &[name, numbers, children] : nodes) {
      key += name;
      key += '\0';
      key.append(reinterpret_cast<const char *>(numbers.data()),
                 numbers.size() * sizeof(double));
      key += '\1';
      key.append(reinterpret_cast<const char *>(children.data()),
                 children.size() * sizeof(int64_t));
      key += '\2';
    }
    auto [it, inserted] = index.try_emplace(
        std::move(key), static_cast<int64_t>(distinct.size()));
    if (inserted)
      distinct.push_back(std::move(nodes));
    of[op] = it->second;
  }

private:
  std::unordered_map<std::string, int64_t> index;
};

// ── Limit checks ──────────────────────────────────────────────────────────

/// What `ControllerLimits` checks on the schedule; -1 for a limit not set.
struct ControllerLimitSpec {
  int64_t issue = 1;
  int64_t gap = 1;
  // Samples per cycle: samplesNum every samplesDen cycles.
  int64_t samplesNum = 1;
  int64_t samplesDen = 1;
  int64_t waveformSamples = -1;
  // The waveform memory: its allocation unit, whether all lines share one,
  // whether one stored shape plays at any amplitude, the built-in shapes it
  // plays without memory, and whether a constant one needs none.
  int64_t granule = 1;
  bool sharedMemory = false;
  bool scaling = true;
  std::set<std::string> synthesized;
  bool hold = false;
  int64_t tonesPerLine = -1;
  int64_t maxEvents = -1;
  std::map<int64_t, int64_t> acquirers;
  int64_t acquirerHold = 0;
  int64_t feedbackLatency = 0;
  bool answerTimeouts = true;
  std::set<std::string> timeoutFunctions;
};

/// The key of `shape_key` in cudaq_pulse.controller: equal for waveforms
/// that differ only in overall amplitude, known or not; none when another
/// number is not known. Without `scaled`, equal only for equal waveforms,
/// and none when any number is not known.
inline std::optional<std::string>
shapeKey(const std::vector<ControllerWalk::WaveformNode> &nodes, int64_t index,
         bool scaled = true) {
  static const std::map<std::string, std::vector<std::string>> parameters = {
      {"square", {"duration", "amplitude", "amplitude_imag"}},
      {"gaussian", {"duration", "amplitude", "sigma"}},
      {"gaussian_square", {"duration", "amplitude", "sigma", "risefall"}},
      {"drag", {"duration", "amplitude", "sigma", "beta"}},
      {"cosine", {"duration", "amplitude"}},
      {"tanh_ramp", {"duration", "amplitude", "sigma"}},
      {"custom", {"duration"}},
      {"scale", {"factor"}},
  };
  static const std::map<std::string, std::set<size_t>> amplitudes = {
      {"square", {1, 2}}, {"gaussian", {1}}, {"gaussian_square", {1}},
      {"drag", {1}},      {"cosine", {1}},   {"tanh_ramp", {1}},
  };
  auto text = [](double v) {
    char buffer[32];
    std::snprintf(buffer, sizeof buffer, "%.17g", v == 0.0 ? 0.0 : v);
    return std::string(buffer);
  };
  const auto &[name, numbers, children] = nodes[index];
  std::string shape = name.substr(0, name.find(':'));
  std::string custom = name.find(':') == std::string::npos
                           ? ""
                           : name.substr(name.find(':') + 1);
  if (!scaled) {
    std::string key = name + "(";
    for (double v : numbers) {
      if (std::isnan(v))
        return std::nullopt;
      key += text(v) + ",";
    }
    key += ")[";
    for (int64_t child : children) {
      auto inner = shapeKey(nodes, child, false);
      if (!inner)
        return std::nullopt;
      key += *inner + ";";
    }
    return key + "]";
  }
  if (shape == "scale" || shape == "neg")
    return children.empty() ? std::nullopt : shapeKey(nodes, children[0]);
  if (shape == "custom_samples") {
    double peak = 0.0;
    for (double v : numbers)
      peak = std::max(peak, std::fabs(v));
    if (peak == 0.0)
      peak = 1.0;
    std::string key = shape;
    for (double v : numbers)
      key += "," + text(std::round(v / peak * 1e12) / 1e12);
    return key;
  }
  auto names = parameters.find(shape);
  if (names == parameters.end() || names->second.size() != numbers.size()) {
    std::string key = shape + "(";
    for (double v : numbers)
      key += (std::isnan(v) ? "None" : text(v)) + ",";
    key += ")[";
    for (int64_t child : children) {
      auto inner = shapeKey(nodes, child);
      if (!inner)
        return std::nullopt;
      key += *inner + ";";
    }
    return key + "]";
  }
  std::string key = shape + "|" + custom;
  auto skip = amplitudes.find(shape);
  for (size_t i = 0; i < numbers.size(); ++i)
    if (skip == amplitudes.end() || !skip->second.count(i)) {
      if (std::isnan(numbers[i]))
        return std::nullopt;
      key += "|" + names->second[i] + "=" + text(numbers[i]);
    }
  return key;
}

/// The op name of a waveform's root, through scale and neg.
inline std::string
rootShape(const std::vector<ControllerWalk::WaveformNode> &nodes) {
  int64_t index = static_cast<int64_t>(nodes.size()) - 1;
  while (index >= 0) {
    const auto &[name, numbers, children] = nodes[index];
    if ((name != "scale" && name != "neg") || children.empty())
      return name;
    index = children[0];
  }
  return "";
}

/// The limits `records` exceed. With `unresolved`, a tone or waveform whose
/// value is not known counts only where the limit is exceeded whatever its
/// value; `*unresolved` is set when only some values would exceed one.
inline std::vector<std::string>
checkControllerLimits(const std::vector<ControllerRecord> &records,
                      const ControllerLimitSpec &limits,
                      bool *unresolved = nullptr) {
  std::vector<std::string> found;
  std::vector<size_t> timed;
  int64_t played = 0;
  for (size_t i = 0; i < records.size(); ++i) {
    const ControllerRecord &r = records[i];
    if (r.played() && r.start && r.duration) {
      timed.push_back(i);
      played += r.weight;
    }
  }
  if (limits.maxEvents >= 0 && played > limits.maxEvents)
    found.push_back(std::to_string(played) +
                    " drives and readouts; the controller plays at most " +
                    std::to_string(limits.maxEvents));

  auto exclusive = [&](const ControllerRecord &a, const ControllerRecord &b) {
    if (a.conditions.empty() || b.conditions.empty() ||
        a.iteration != b.iteration)
      return false;
    for (const auto &c : b.conditions)
      for (const auto &d : a.conditions)
        if (std::get<0>(c) == std::get<0>(d) &&
            std::get<1>(c) != std::get<1>(d))
          return true;
    return false;
  };
  // A pair from a sampled loop's iteration 1 to its last stands for the
  // pair from one iteration to the next, which iterations 0 and 1 show.
  auto jumped = [](const ControllerRecord &a, const ControllerRecord &b) {
    for (const auto &[instance, sample] : a.samples)
      if (sample == 1)
        for (const auto &[other, later] : b.samples)
          if (other == instance && later == 2)
            return true;
    return false;
  };
  // Each record with the latest earlier one it can follow on one path.
  auto pairs = [&](std::vector<size_t> items) {
    std::stable_sort(items.begin(), items.end(), [&](size_t x, size_t y) {
      return *records[x].start < *records[y].start;
    });
    std::vector<std::pair<size_t, size_t>> out;
    for (size_t k = 1; k < items.size(); ++k)
      for (size_t j = k; j-- > 0;) {
        if (jumped(records[items[j]], records[items[k]]))
          break;
        if (!exclusive(records[items[j]], records[items[k]])) {
          out.emplace_back(items[j], items[k]);
          break;
        }
      }
    return out;
  };

  // A record of a sampled loop's iteration 1 stands for iterations 1 to
  // n - 2: its message says so.
  auto times = [](const ControllerRecord &r) {
    return r.weight > 1 ? " (in each of " + std::to_string(r.weight) +
                              " iterations of a loop)"
                        : std::string();
  };
  std::map<int64_t, std::vector<size_t>> byLine;
  for (size_t i : timed)
    byLine[records[i].line].push_back(i);
  ControllerWaveforms waveforms;
  std::vector<std::optional<std::optional<std::string>>> keys;
  for (auto &[line, items] : byLine) {
    for (auto [p, q] : pairs(items)) {
      const ControllerRecord &previous = records[p], &event = records[q];
      int64_t distance = *event.start - *previous.start;
      if (distance != limits.issue && distance < limits.gap)
        found.push_back(
            "line " + std::to_string(line) + ": " + event.kind + " at " +
            std::to_string(*event.start) + " starts " +
            std::to_string(distance) + " cycles after the " + previous.kind +
            " at " + std::to_string(*previous.start) +
            "; the controller needs exactly " + std::to_string(limits.issue) +
            " or at least " + std::to_string(limits.gap) + times(event));
    }
    if (limits.tonesPerLine >= 0) {
      std::set<double> tones;
      int64_t unknown = 0;
      for (size_t i : items) {
        if (!std::isnan(records[i].frequency))
          tones.insert(records[i].frequency);
        else if (records[i].played())
          unknown += records[i].weight;
      }
      int64_t known = static_cast<int64_t>(tones.size());
      if (known > limits.tonesPerLine)
        found.push_back("line " + std::to_string(line) + ": " +
                        std::to_string(tones.size()) +
                        " tone frequencies; the controller holds " +
                        std::to_string(limits.tonesPerLine) + " per line");
      else if (unresolved && known + unknown > limits.tonesPerLine)
        *unresolved = true;
    }
  }
  if (limits.waveformSamples >= 0) {
    // One memory per line, or one for all.
    std::map<int64_t, std::vector<size_t>> memories;
    for (auto &[line, items] : byLine)
      for (size_t i : items)
        if (records[i].played())
          memories[limits.sharedMemory ? -1 : line].push_back(i);
    std::vector<int8_t> costless;
    for (auto &[line, items] : memories) {
      std::map<std::string, int64_t> shapes;
      int64_t unshared = 0, unknown = 0;
      for (size_t i : items) {
        const ControllerRecord &r = records[i];
        waveforms.add(r.op);
        size_t shape = static_cast<size_t>(waveforms.of.lookup(r.op));
        if (keys.size() <= shape) {
          keys.resize(shape + 1);
          costless.resize(shape + 1, -1);
        }
        const auto &nodes = waveforms.distinct[shape];
        if (costless[shape] < 0) {
          std::string root = rootShape(nodes);
          costless[shape] = limits.synthesized.count(root) ||
                            (limits.hold && root == "square");
        }
        if (costless[shape])
          continue;
        if (!keys[shape])
          keys[shape] = shapeKey(nodes, static_cast<int64_t>(nodes.size()) - 1,
                                 limits.scaling);
        const std::optional<std::string> &key = *keys[shape];
        int64_t samples =
            (*r.duration * limits.samplesNum + limits.samplesDen - 1) /
            limits.samplesDen;
        samples =
            (samples + limits.granule - 1) / limits.granule * limits.granule;
        if (key)
          shapes[*key + "@" + std::to_string(*r.duration)] = samples;
        else if (unresolved)
          unknown += samples * r.weight;
        else
          unshared += samples * r.weight;
      }
      int64_t used = unshared;
      for (auto &[key, samples] : shapes)
        used += samples;
      if (used > limits.waveformSamples)
        found.push_back(
            (limits.sharedMemory
                 ? std::string("its waveforms take ")
                 : "line " + std::to_string(line) + ": its waveforms take ") +
            std::to_string(used) + " samples; the controller's " +
            (limits.sharedMemory ? "shared " : "") + "waveform memory holds " +
            std::to_string(limits.waveformSamples));
      else if (unresolved && used + unknown > limits.waveformSamples)
        *unresolved = true;
    }
  }
  if (!limits.acquirers.empty()) {
    std::map<int64_t, std::vector<size_t>> byAcquirer;
    for (size_t i : timed) {
      const ControllerRecord &r = records[i];
      auto acquirer = limits.acquirers.find(r.qudit);
      if (std::strcmp(r.kind, "readout") == 0 &&
          acquirer != limits.acquirers.end())
        byAcquirer[acquirer->second].push_back(i);
    }
    for (auto &[acquirer, items] : byAcquirer)
      for (auto [p, q] : pairs(items)) {
        const ControllerRecord &previous = records[p], &event = records[q];
        int64_t free = *previous.stop() + limits.acquirerHold;
        if (*event.start < free)
          found.push_back(
              "acquirer " + std::to_string(acquirer) +
              ": the readout of qudit " + std::to_string(event.qudit) + " at " +
              std::to_string(*event.start) +
              " starts before the acquirer is free at " + std::to_string(free) +
              " (the readout of qudit " + std::to_string(previous.qudit) +
              " at " + std::to_string(*previous.start) + ")" + times(event));
      }
  }
  if (limits.feedbackLatency)
    for (size_t i : timed) {
      const ControllerRecord &event = records[i];
      for (const auto &condition : event.conditions)
        for (int64_t ref : std::get<2>(condition)) {
          auto end = records[ref].stop();
          if (!end || records[ref].end)
            continue;
          int64_t ready = *end + limits.feedbackLatency;
          if (*event.start < ready)
            found.push_back(
                "line " + std::to_string(event.line) + ": the " + event.kind +
                " at " + std::to_string(*event.start) +
                " depends on the readout ending at " + std::to_string(*end) +
                "; the controller needs " +
                std::to_string(limits.feedbackLatency) +
                " cycles of feedback latency (start at " +
                std::to_string(ready) +
                " or later; schedule with rcp and readout_latency_cycles >= "
                "feedback_latency_cycles)" +
                times(event));
        }
    }
  if (!limits.answerTimeouts)
    for (const ControllerRecord &r : records)
      if (std::strcmp(r.kind, "answer") == 0 &&
          (r.mode == "timeout" || limits.timeoutFunctions.count(r.callee)))
        found.push_back(
            "the wait for " + r.callee + " at " +
            (r.start ? std::to_string(*r.start) : "None") +
            " has a timeout; the controller cannot stop waiting for "
            "an answer, so take it with result() (no timeout)" +
            times(r));
  return found;
}

// ── Python objects ────────────────────────────────────────────────────────

/// Builds the events of a walk as the dictionaries of the export format,
/// without reading the IR. `waveform` turns the nodes of a waveform into
/// (expression, amplitude); it is called once per distinct waveform, and
/// the events that play one share what it returns.
class ControllerEvents {
public:
  /// `compact` (format /2) leaves out what is derivable or the default: the
  /// index (an event's place in pre-order), the qudit (its line's), the end,
  /// an empty iteration, epoch 0, empty conditions and lists, a sync's or
  /// call's line, a timeout that is none and a first iteration of 0; its
  /// `waveform` returns (index into the program's waveforms, None).
  ControllerEvents(const std::vector<ControllerRecord> &records,
                   const ControllerWaveforms &waveforms, nb::callable waveform,
                   bool compact)
      : records(records), waveforms(waveforms), waveform(std::move(waveform)),
        compact(compact), made(waveforms.distinct.size()) {}

  nb::list build() {
    size_t at = 0;
    return range(at, records.size());
  }

private:
  const std::vector<ControllerRecord> &records;
  const ControllerWaveforms &waveforms;
  nb::callable waveform;
  bool compact;
  std::vector<std::optional<std::pair<nb::object, nb::object>>> made;

  static nb::object number(double v) {
    return std::isnan(v) ? nb::none() : nb::object(nb::float_(v));
  }
  static nb::object optional(const std::optional<int64_t> &v) {
    return v ? nb::object(nb::int_(*v)) : nb::none();
  }
  static nb::list ints(const std::vector<int64_t> &values) {
    nb::list out;
    for (int64_t v : values)
      out.append(v);
    return out;
  }

  nb::list range(size_t &at, size_t stop) {
    nb::list out;
    while (at < stop) {
      size_t index = at++;
      const ControllerRecord &r = records[index];
      nb::dict event = common(r, index);
      if (std::strcmp(r.kind, "loop") == 0) {
        event["trips"] = r.trips;
        if (!compact || r.first)
          event["first_iteration"] = r.first;
        event["period_cycles"] = r.period;
        event["body"] = range(at, at + static_cast<size_t>(r.size));
      } else {
        details(r, event);
      }
      out.append(event);
    }
    return out;
  }

  nb::dict common(const ControllerRecord &r, size_t index) {
    nb::dict event;
    bool sync = std::strcmp(r.kind, "sync") == 0;
    bool loop = std::strcmp(r.kind, "loop") == 0;
    bool call = (r.kind[0] == 's' && !sync) || r.kind[0] == 'a';
    if (!compact)
      event["index"] = index;
    event["kind"] = r.kind;
    if (!compact)
      event["line"] = sync || loop ? nb::none() : nb::object(nb::int_(r.line));
    else if (!sync && !loop && !call)
      event["line"] = r.line;
    if (!compact)
      event["qudit"] =
          sync || loop ? nb::none() : nb::object(nb::int_(r.qudit));
    event["start_cycles"] = optional(r.start);
    event["duration_cycles"] = optional(r.duration);
    if (!compact)
      event["end_cycles"] = optional(r.stop());
    if (!compact || !r.iteration.empty()) {
      nb::list iteration;
      for (int64_t k : r.iteration)
        iteration.append(k < 0 ? nb::none() : nb::object(nb::int_(k)));
      event["iteration"] = iteration;
    }
    int64_t epoch = r.start ? r.epoch : 0;
    if (!compact || epoch)
      event["epoch"] = epoch;
    if (!compact || !r.conditions.empty()) {
      nb::list conditions;
      for (const auto &[branch, arm, readouts, answers, test] : r.conditions) {
        nb::dict condition;
        condition["branch"] = branch;
        condition["arm"] = static_cast<bool>(arm);
        if (!compact || !readouts.empty())
          condition["readouts"] = ints(readouts);
        if (!compact || !answers.empty())
          condition["answers"] = ints(answers);
        condition["test"] = test;
        conditions.append(condition);
      }
      event["conditions"] = conditions;
    }
    return event;
  }

  void details(const ControllerRecord &r, nb::dict &event) {
    if (r.played()) {
      auto &shape = made[static_cast<size_t>(waveforms.of.lookup(r.op))];
      if (!shape) {
        nb::tuple pair = nb::cast<nb::tuple>(
            waveform(waveforms.distinct[waveforms.of.lookup(r.op)]));
        shape.emplace(pair[0], pair[1]);
      }
      event["waveform"] = shape->first;
      if (!compact) {
        event["amplitude"] = shape->second;
        event["frequency_hz"] = number(r.frequency);
        event["phase_rad"] = number(r.phase);
        return;
      }
      // A list, so that an event may later play more than one carrier.
      nb::dict tone;
      tone["frequency_hz"] = number(r.frequency);
      tone["phase_rad"] = number(r.phase);
      if (!r.phaseSteps.empty()) {
        nb::list steps;
        for (double step : r.phaseSteps)
          steps.append(number(step));
        tone["phase_step_rad"] = steps;
      }
      nb::list tones;
      tones.append(tone);
      event["tones"] = tones;
    } else if (std::strcmp(r.kind, "sync") == 0) {
      event["lines"] = ints(r.joined);
    } else if (r.kind[0] == 's' || r.kind[0] == 'a') {
      event["link"] = r.line;
      event["function"] = r.callee;
      if (r.kind[0] == 's') {
        event["measurements"] = ints(r.refs);
      } else {
        event["wait"] = r.mode;
        if (!compact || !std::isnan(r.timeout))
          event["timeout_ns"] = number(r.timeout);
      }
    }
  }
};
