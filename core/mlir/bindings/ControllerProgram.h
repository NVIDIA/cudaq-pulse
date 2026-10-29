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
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
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
  // What the event has that is known only at run time: its index in the
  // walk's `known`, else -1.
  int32_t known = -1;
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

  /// With `track`, the arguments of `main` each record's tone frequency,
  /// frame phase and waveform depend on, parallel to `result`. A qref
  /// argument stands for its qudit's native frequency.
  bool track = false;
  using Dependencies = std::array<std::vector<int64_t>, 3>;
  std::vector<Dependencies> dependencies;
  /// With `track`, parallel to `result`: the qref argument whose native
  /// frequency a record's tone plays (its line was never retuned), or -1.
  std::vector<int64_t> natives;
  /// What a tone's frequency, the frequency of its frame (where it
  /// started) and its phase depend on.
  struct FrameDependencies {
    std::vector<int64_t> frequency, frame, phase;
    bool retuned = false;
  };
  std::map<mlir::Operation *, FrameDependencies> frameDependencies;

  /// The arguments of `main` *value* is computed from.
  static void argumentsOf(mlir::Value value, std::vector<int64_t> &out) {
    llvm::SmallVector<mlir::Value> work{value};
    llvm::DenseSet<mlir::Value> seen;
    while (!work.empty()) {
      mlir::Value v = work.pop_back_val();
      if (!seen.insert(v).second)
        continue;
      if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(v)) {
        mlir::Block *owner = argument.getOwner();
        if (owner->isEntryBlock() &&
            mlir::isa_and_nonnull<mlir::func::FuncOp>(owner->getParentOp()))
          out.push_back(argument.getArgNumber());
        continue;
      }
      mlir::Operation *def = v.getDefiningOp();
      if (def && def->getNumRegions() == 0)
        work.append(def->getOperands().begin(), def->getOperands().end());
    }
  }

  static void merge(std::vector<int64_t> &into,
                    const std::vector<int64_t> &from) {
    into.insert(into.end(), from.begin(), from.end());
    std::sort(into.begin(), into.end());
    into.erase(std::unique(into.begin(), into.end()), into.end());
  }

  FrameDependencies &frameDependenciesOf(mlir::Value tone) {
    mlir::Operation *root = toneRoots(tone);
    auto [it, inserted] = frameDependencies.try_emplace(root);
    if (inserted && root) {
      if (mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(root) &&
          !root->hasAttr("frequency_hz"))
        argumentsOf(root->getOperand(0), it->second.frequency);
      else if (root->getName().getStringRef() == "pulse.tone" &&
               root->getNumOperands() == 2) {
        argumentsOf(root->getOperand(0), it->second.frequency);
        argumentsOf(root->getOperand(1), it->second.phase);
      }
      merge(it->second.frequency, {});
      merge(it->second.phase, {});
      it->second.frame = it->second.frequency;
    }
    return it->second;
  }

  /// The dependencies and native tone of drive or readout `op`.
  LLVM_ATTRIBUTE_NOINLINE void trackPlayed(mlir::Operation *op) {
    Dependencies &found = dependencies.back();
    FrameDependencies &frame = frameDependenciesOf(op->getOperand(2));
    mlir::Operation *root = toneRoots(op->getOperand(2));
    if (!frame.retuned && root &&
        mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(root) &&
        !root->hasAttr("frequency_hz"))
      if (auto argument =
              mlir::dyn_cast<mlir::BlockArgument>(root->getOperand(0)))
        if (argument.getOwner()->isEntryBlock() &&
            mlir::isa_and_nonnull<mlir::func::FuncOp>(
                argument.getOwner()->getParentOp()))
          natives.back() = argument.getArgNumber();
    found[0] = frame.frequency;
    found[1] = frame.phase;
    // A retuned tone gains phase at its detuning from the frame.
    if (frame.retuned) {
      merge(found[1], frame.frequency);
      merge(found[1], frame.frame);
    }
    argumentsOf(op->getOperand(1), found[2]);
    merge(found[2], {});
  }

  /// What frame change `op` (named `name`) makes its tone depend on.
  LLVM_ATTRIBUTE_NOINLINE void trackFrameChange(mlir::Operation *op,
                                                llvm::StringRef name) {
    FrameDependencies &frame = frameDependenciesOf(op->getOperand(0));
    std::vector<int64_t> found;
    argumentsOf(op->getOperand(1), found);
    bool phase = name == "pulse.shift_phase" || name == "pulse.set_phase";
    auto &field = phase ? frame.phase : frame.frequency;
    if (name.starts_with("pulse.set_"))
      field.clear();
    merge(field, found);
    frame.retuned |= !phase;
  }

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
    // A runtime qudit's line: its frequency is the carrier's.
    if (mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(root))
      return {kUnknown, kUnknown, 0.0, 0.0};
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

  // ── Run-time values ──────────────────────────────────────────────────
  //
  // What an event has that is known only at run time: on a tone whose frame
  // depends on run time, the frame changes since its latest drive or readout;
  // the expressions of the waveform operands, a variable wait's length and a
  // submission's operands computed from readouts and answers. Looked for
  // only where the module can compute something at run time (`runTime`).

  bool runTime = false;

  /// Whether `value` depends on a readout or an answer.
  static bool fromRunTime(mlir::Value value) {
    mlir::Operation *first = value.getDefiningOp();
    if (!first || first->hasTrait<mlir::OpTrait::ConstantLike>())
      return false;
    llvm::SmallVector<mlir::Value> work{value};
    llvm::DenseSet<mlir::Value> seen;
    while (!work.empty()) {
      mlir::Value v = work.pop_back_val();
      if (!seen.insert(v).second)
        continue;
      mlir::Operation *def = v.getDefiningOp();
      if (!def)
        continue;
      if (mlir::isa<pulse::ReadoutOp, classical::WaitOp, mlir::scf::IfOp>(def))
        return true;
      work.append(def->getOperands().begin(), def->getOperands().end());
    }
    return false;
  }

  /// Whether a measurement reaches anything but branch conditions and
  /// submissions; true past a few ops, which only costs a slower walk.
  static bool measurementIsUsed(mlir::Value value, int depth = 0) {
    for (mlir::Operation *user : value.getUsers()) {
      if (mlir::isa<mlir::scf::IfOp, classical::SubmitOp>(user))
        continue;
      if (depth == 4 ||
          (!mlir::isa<pulse::IsSetOp, pulse::OutcomeOp>(user) &&
           !mlir::isa<mlir::arith::ArithDialect>(user->getDialect())))
        return true;
      for (mlir::Value next : user->getResults())
        if (measurementIsUsed(next, depth + 1))
          return true;
    }
    return false;
  }

  /// The tones whose frame a controller tracks at run time: those with a
  /// frame change computed from a readout or an answer, or made in a branch.
  llvm::DenseSet<mlir::Operation *> tracked;
  /// Whether any event can have something known only at run time: a tracked
  /// tone, a classical call, or a measurement used other than as a branch
  /// condition.
  bool detail = false;

  void scan(mlir::Operation *op) {
    if (mlir::isa<pulse::ShiftPhaseOp, pulse::SetPhaseOp,
                  pulse::ShiftFrequencyOp, pulse::SetFrequencyOp>(op)) {
      if (fromRunTime(op->getOperand(1)) ||
          op->getParentOfType<mlir::scf::IfOp>()) {
        tracked.insert(toneRoots(op->getOperand(0)));
        detail = true;
      }
    } else if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
      // A virtual Z folded into a drive in an arm is a frame change there.
      branch->walk([&](pulse::DriveOp drive) {
        if (drive->hasAttr("frame_phase_offset")) {
          tracked.insert(toneRoots(drive->getOperand(2)));
          detail = true;
        }
      });
    } else if (mlir::isa<classical::SubmitOp>(op)) {
      detail = true;
    } else if (mlir::isa<pulse::ReadoutOp>(op) && !detail) {
      detail = measurementIsUsed(op->getResult(2));
    }
  }

  /// How a value is computed at run time, children before their parent: a
  /// number (a frame change's value), a constant, an unknown (`name`: what),
  /// the result of a readout or answer record, or an op (`name`, and a
  /// comparison's `predicate`) over the expressions `of`.
  struct Expression {
    enum Kind : uint8_t { Number, Constant, Unknown, Readout, Answer, Op };
    Kind kind = Number;
    double number = 0.0;
    int64_t event = 0, result = 0;
    llvm::StringRef name, predicate;
    std::vector<int32_t> of;
  };
  std::vector<Expression> expressions;

  struct FrameUpdate {
    uint64_t seq;
    llvm::StringRef op;
    int32_t value; // an expression
    std::vector<Condition> conditions;
  };
  // Each tracked tone's id, in the order tones are first played, and the
  // frame changes on it since its latest drive or readout.
  llvm::DenseMap<mlir::Operation *, int64_t> toneIds;
  std::map<mlir::Operation *, std::vector<FrameUpdate>> pending;
  uint64_t nextUpdate = 0;

  /// The frame changes still pending once the walk is done, which no drive
  /// or readout took, in program order with their tone's id; a tone never
  /// played is numbered after those that are.
  std::vector<std::pair<int64_t, FrameUpdate>> unapplied() {
    std::vector<std::pair<mlir::Operation *, FrameUpdate>> all;
    for (auto &[root, updates] : pending)
      for (FrameUpdate &update : updates)
        all.emplace_back(root, std::move(update));
    pending.clear();
    std::sort(all.begin(), all.end(), [](const auto &a, const auto &b) {
      return a.second.seq < b.second.seq;
    });
    std::vector<std::pair<int64_t, FrameUpdate>> out;
    for (auto &[root, update] : all)
      out.emplace_back(
          toneIds.try_emplace(root, static_cast<int64_t>(toneIds.size()))
              .first->second,
          std::move(update));
    return out;
  }

  /// On a tracked tone (`tone` at least 0), the frame changes since its
  /// latest drive or readout and its own phase offset; the waveform
  /// operands computed from a readout or an answer, as (node, operand
  /// index, expression); a variable wait's length (at least 0) and bounds;
  /// a submission's operands.
  struct Known {
    int64_t tone = -1;
    std::vector<FrameUpdate> updates;
    std::optional<double> offset;
    std::vector<std::tuple<int64_t, int64_t, int32_t>> waveform;
    int32_t length = -1;
    std::optional<int64_t> minCycles, maxCycles;
    std::optional<std::vector<int32_t>> operands;
  };
  std::vector<Known> known;

  int32_t add(Expression expression) {
    expressions.push_back(std::move(expression));
    return static_cast<int32_t>(expressions.size()) - 1;
  }

  int32_t add(Expression::Kind kind, double number, int64_t event = 0,
              int64_t result = 0, llvm::StringRef name = {}) {
    Expression expression;
    expression.kind = kind;
    expression.number = number;
    expression.event = event;
    expression.result = result;
    expression.name = name;
    return add(std::move(expression));
  }

  /// How `value` is computed at run time: a constant, a result of a readout
  /// or answer record, or an op (by its MLIR name) over such expressions.
  int32_t expression(mlir::Value value, int depth = 0) {
    double constant = number(value);
    if (!std::isnan(constant))
      return add(Expression::Constant, constant);
    mlir::Operation *def = value.getDefiningOp();
    if (!def)
      return add(Expression::Unknown, 0.0, 0, 0, "argument");
    llvm::StringRef name = def->getName().getStringRef();
    if (mlir::isa<pulse::ReadoutOp, classical::WaitOp>(def)) {
      auto found = latest.find(def);
      if (found == latest.end())
        return add(Expression::Unknown, 0.0, 0, 0, name);
      if (mlir::isa<pulse::ReadoutOp>(def))
        return add(Expression::Readout, 0.0, found->second);
      return add(Expression::Answer, 0.0, found->second,
                 static_cast<int64_t>(
                     mlir::cast<mlir::OpResult>(value).getResultNumber()));
    }
    if (depth > 64 || def->getNumRegions() != 0 || def->getNumResults() != 1)
      return add(Expression::Unknown, 0.0, 0, 0, name);
    Expression op;
    op.kind = Expression::Op;
    op.name = name;
    for (mlir::Value operand : def->getOperands())
      op.of.push_back(expression(operand, depth + 1));
    if (auto cmp = mlir::dyn_cast<mlir::arith::CmpFOp>(def))
      op.predicate = mlir::arith::stringifyCmpFPredicate(cmp.getPredicate());
    else if (auto cmp = mlir::dyn_cast<mlir::arith::CmpIOp>(def))
      op.predicate = mlir::arith::stringifyCmpIPredicate(cmp.getPredicate());
    return add(std::move(op));
  }

  /// Whether expression `y` is `x` one iteration on: records [a, b) moved
  /// to [b, 2b - a), the rest the same.
  bool sameExpression(int32_t x, int32_t y, int64_t a, int64_t b) const {
    const Expression &p = expressions[x], &q = expressions[y];
    if (p.kind != q.kind || p.name != q.name || p.predicate != q.predicate ||
        p.result != q.result || p.of.size() != q.of.size() ||
        !same(p.number, q.number))
      return false;
    if (p.kind == Expression::Readout || p.kind == Expression::Answer) {
      int64_t moved = p.event >= a && p.event < b ? p.event + (b - a) : p.event;
      if (q.event != moved)
        return false;
    }
    for (size_t i = 0; i < p.of.size(); ++i)
      if (!sameExpression(p.of[i], q.of[i], a, b))
        return false;
    return true;
  }

  /// Whether record `y`'s run-time values are record `x`'s one iteration on
  /// (see sameExpression); a tracked tone never repeats.
  bool sameKnown(const ControllerRecord &x, const ControllerRecord &y,
                 int64_t a, int64_t b) const {
    if (x.known < 0 || y.known < 0)
      return x.known == y.known;
    const Known &p = known[x.known], &q = known[y.known];
    if (p.tone >= 0 || q.tone >= 0 || p.waveform.size() != q.waveform.size() ||
        (p.length < 0) != (q.length < 0) || p.minCycles != q.minCycles ||
        p.maxCycles != q.maxCycles ||
        p.operands.has_value() != q.operands.has_value())
      return false;
    for (size_t i = 0; i < p.waveform.size(); ++i) {
      const auto &[n, k, e] = p.waveform[i];
      const auto &[m, l, f] = q.waveform[i];
      if (n != m || k != l || !sameExpression(e, f, a, b))
        return false;
    }
    if (p.length >= 0 && !sameExpression(p.length, q.length, a, b))
      return false;
    if (p.operands) {
      if (p.operands->size() != q.operands->size())
        return false;
      for (size_t i = 0; i < p.operands->size(); ++i)
        if (!sameExpression((*p.operands)[i], (*q.operands)[i], a, b))
          return false;
    }
    return true;
  }

  /// Moves the record references of record `r`'s run-time values with
  /// `shift`.
  template <typename Shift>
  void shiftKnown(const ControllerRecord &r, Shift shift) {
    if (r.known < 0)
      return;
    auto move = [&](int32_t index, auto &self) -> void {
      Expression &e = expressions[index];
      if (e.kind == Expression::Readout || e.kind == Expression::Answer)
        shift(e.event);
      for (int32_t child : e.of)
        self(child, self);
    };
    Known &k = known[r.known];
    for (auto &[node, operand, expr] : k.waveform)
      move(expr, move);
    if (k.length >= 0)
      move(k.length, move);
    if (k.operands)
      for (int32_t expr : *k.operands)
        move(expr, move);
  }

  /// A frame change's value: its number, else its expression.
  int32_t frameValue(mlir::Value value) {
    double constant = number(value);
    return std::isnan(constant) ? expression(value)
                                : add(Expression::Number, constant);
  }

  void frameUpdate(mlir::Operation *root, llvm::StringRef op, int32_t value) {
    pending[root].push_back({nextUpdate++, op, value, branches});
  }

  // The run-time work of the walk stays out of line, so that the per-op
  // path of a walk without it is unchanged.

  /// A frame change `op` (named `name`), on a tone that may be tracked.
  LLVM_ATTRIBUTE_NOINLINE void frameChange(mlir::Operation *op,
                                           llvm::StringRef name) {
    if (mlir::Operation *root = toneRoots(op->getOperand(0));
        tracked.contains(root))
      frameUpdate(root, name, frameValue(op->getOperand(1)));
  }

  // The frame changes pending before each open branch, and after its then
  // arm.
  std::vector<std::pair<decltype(pending), decltype(pending)>> pendingArms;

  LLVM_ATTRIBUTE_NOINLINE void enterBranch() {
    pendingArms.emplace_back(pending, decltype(pending){});
  }

  LLVM_ATTRIBUTE_NOINLINE void enterElse() {
    auto &[before, then] = pendingArms.back();
    then = std::move(pending);
    pending = before;
  }

  LLVM_ATTRIBUTE_NOINLINE void leaveBranch(const Condition &arms) {
    auto [before, then] = std::move(pendingArms.back());
    pendingArms.pop_back();
    mergePending(before, then, arms);
  }

  /// The frame changes pending after a branch: those of both arms, by
  /// program order. One pending before the branch that a drive or readout
  /// in one arm took is pending only on the other arm's path.
  void
  mergePending(std::map<mlir::Operation *, std::vector<FrameUpdate>> &before,
               std::map<mlir::Operation *, std::vector<FrameUpdate>> &then,
               const Condition &branch) {
    std::set<mlir::Operation *> roots;
    for (auto *side : {&before, &then, &pending})
      for (auto &entry : *side)
        roots.insert(entry.first);
    auto seqs = [](std::map<mlir::Operation *, std::vector<FrameUpdate>> &side,
                   mlir::Operation *root) {
      std::set<uint64_t> out;
      auto found = side.find(root);
      if (found != side.end())
        for (auto &update : found->second)
          out.insert(update.seq);
      return out;
    };
    for (mlir::Operation *root : roots) {
      std::set<uint64_t> old = seqs(before, root), inThen = seqs(then, root),
                         inElse = seqs(pending, root);
      std::map<uint64_t, FrameUpdate> merged;
      auto take =
          [&](std::map<mlir::Operation *, std::vector<FrameUpdate>> &side,
              bool arm, const std::set<uint64_t> &other) {
            auto found = side.find(root);
            if (found == side.end())
              return;
            for (auto &update : found->second) {
              if (merged.count(update.seq))
                continue;
              if (old.count(update.seq) && !other.count(update.seq)) {
                update.conditions.push_back(branch);
                std::get<1>(update.conditions.back()) = arm;
              }
              merged.emplace(update.seq, std::move(update));
            }
          };
      take(then, true, inElse);
      take(pending, false, inThen);
      std::vector<FrameUpdate> out;
      for (auto &[seq, update] : merged)
        out.push_back(std::move(update));
      if (out.empty())
        pending.erase(root);
      else
        pending[root] = std::move(out);
    }
  }

  /// The operands of `value`'s waveform nodes (numbered as `waveform` does,
  /// `nodes` its result) computed from a readout or an answer, as (node,
  /// operand index, operand).
  static void
  runTimeOperands(mlir::Value value, const std::vector<WaveformNode> &nodes,
                  int64_t &next,
                  std::vector<std::tuple<int64_t, int64_t, mlir::Value>> &out) {
    mlir::Operation *def = value.getDefiningOp();
    if (!def) {
      ++next;
      return;
    }
    llvm::SmallVector<std::pair<int64_t, mlir::Value>, 4> numbers;
    for (mlir::Value operand : def->getOperands()) {
      if (mlir::isa<pulse::WaveformType>(operand.getType()))
        runTimeOperands(operand, nodes, next, out);
      else
        numbers.emplace_back(static_cast<int64_t>(numbers.size()), operand);
    }
    int64_t node = next++;
    const std::vector<double> &numbersKnown = std::get<1>(nodes[node]);
    for (auto &[index, operand] : numbers)
      if (std::isnan(numbersKnown[index]) && fromRunTime(operand))
        out.emplace_back(node, index, operand);
  }

  /// What the next record, a drive, readout or wait, has that is known only
  /// at run time: its index in `known`, else -1.
  LLVM_ATTRIBUTE_NOINLINE int32_t runTimeOf(mlir::Operation *op) {
    Known next;
    if (mlir::isa<pulse::WaitOp>(op)) {
      if (!fromRunTime(op->getOperand(1)) ||
          !op->hasAttr("pulse.variable_wait"))
        return -1;
      next.length = expression(op->getOperand(1));
      if (auto bound = op->getAttrOfType<mlir::IntegerAttr>("pulse.max_cycles"))
        next.maxCycles = bound.getInt();
      if (auto bound = op->getAttrOfType<mlir::IntegerAttr>("pulse.min_cycles"))
        next.minCycles = bound.getInt();
      known.push_back(std::move(next));
      return static_cast<int32_t>(known.size()) - 1;
    }
    std::vector<WaveformNode> nodes;
    waveform(op->getOperand(1), nodes);
    bool unknown = llvm::any_of(nodes, [](const WaveformNode &node) {
      return llvm::any_of(std::get<1>(node),
                          [](double v) { return std::isnan(v); });
    });
    if (unknown) {
      std::vector<std::tuple<int64_t, int64_t, mlir::Value>> operands;
      int64_t count = 0;
      runTimeOperands(op->getOperand(1), nodes, count, operands);
      for (auto &[node, index, operand] : operands)
        next.waveform.emplace_back(node, index, expression(operand));
    }
    mlir::Operation *root =
        tracked.empty() ? nullptr : toneRoots(op->getOperand(2));
    if (root && tracked.contains(root)) {
      next.tone =
          toneIds.try_emplace(root, static_cast<int64_t>(toneIds.size()))
              .first->second;
      if (mlir::isa<pulse::DriveOp>(op)) {
        if (auto shift =
                op->getAttrOfType<mlir::FloatAttr>("frame_phase_offset"))
          frameUpdate(root, "shift_phase",
                      add(Expression::Number, shift.getValueAsDouble()));
        if (auto own = op->getAttrOfType<mlir::FloatAttr>("phase_offset"))
          next.offset = own.getValueAsDouble();
      }
      if (auto found = pending.find(root); found != pending.end()) {
        next.updates = std::move(found->second);
        pending.erase(found);
      }
    }
    if (next.tone < 0 && next.waveform.empty())
      return -1;
    known.push_back(std::move(next));
    return static_cast<int32_t>(known.size()) - 1;
  }

  /// The operands of a submission, in argument order.
  LLVM_ATTRIBUTE_NOINLINE int32_t submission(classical::SubmitOp request) {
    Known next;
    next.operands.emplace();
    for (mlir::Value arg : request.getArgs())
      next.operands->push_back(expression(arg));
    known.push_back(std::move(next));
    return static_cast<int32_t>(known.size()) - 1;
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
    if (track) {
      dependencies.emplace_back();
      natives.push_back(-1);
    }
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
    int32_t knownIndex = runTime && detail ? runTimeOf(op) : -1;
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
    r.known = knownIndex;
    if (track && kind[0] != 'w')
      trackPlayed(op);
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
    int32_t knownIndex = -1;
    if (runTime)
      if (auto request = mlir::dyn_cast<classical::SubmitOp>(op))
        knownIndex = submission(request);
    ControllerRecord &r = record(submit ? "submit" : "answer", op);
    r.known = knownIndex;
    r.line = classical::intAttr(op, "classical.device");
    // A call has no qudit: its slot holds the controller whose instance of
    // a per-controller link runs it (-1: none).
    r.qudit = classical::intAttr(op, classical::kInstanceAttr, -1);
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
      if (track)
        trackFrameChange(op, name);
      double value = number(op->getOperand(1));
      Frame &frame = frameOf(op->getOperand(0));
      if (runTime && !tracked.empty())
        frameChange(op, name.drop_front(6));
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
      bool tracking = runTime && !tracked.empty();
      if (tracking)
        enterBranch();
      decltype(frameDependencies) dependenciesBefore, dependenciesThen;
      if (track)
        dependenciesBefore = frameDependencies;
      branches.emplace_back(id, 1, std::move(readouts), std::move(answers),
                            std::move(test));
      block(*branch.thenBlock());
      auto framesThen = std::move(frames);
      auto reachedThen = std::move(reached);
      frames = std::move(framesBefore);
      reached = std::move(reachedBefore);
      if (track) {
        dependenciesThen = std::move(frameDependencies);
        frameDependencies = std::move(dependenciesBefore);
      }
      if (tracking)
        enterElse();
      std::get<1>(branches.back()) = 0;
      if (!branch.getElseRegion().empty())
        block(*branch.elseBlock());
      if (tracking)
        leaveBranch(branches.back());
      branches.pop_back();
      mergeFrames(framesThen);
      // After the branch, a frame depends on what either arm set.
      for (auto &[root, theirs] : dependenciesThen) {
        auto &ours = frameDependencies[root];
        merge(ours.frequency, theirs.frequency);
        merge(ours.frame, theirs.frame);
        merge(ours.phase, theirs.phase);
        ours.retuned |= theirs.retuned;
      }
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
    std::map<mlir::Operation *, std::vector<FrameUpdate>> pending;
    llvm::DenseMap<mlir::Operation *, int64_t> toneIds;
    uint64_t nextUpdate;
    std::map<mlir::Operation *, FrameDependencies> frameDependencies;
  };

  State save() const {
    return {frames,
            reached,
            latest,
            nextBranch,
            result.size(),
            pending,
            toneIds,
            nextUpdate,
            track ? frameDependencies
                  : std::map<mlir::Operation *, FrameDependencies>()};
  }

  void restore(State &state) {
    frames = std::move(state.frames);
    reached = std::move(state.reached);
    latest = std::move(state.latest);
    nextBranch = state.nextBranch;
    result.resize(state.size);
    pending = std::move(state.pending);
    toneIds = std::move(state.toneIds);
    nextUpdate = state.nextUpdate;
    if (track) {
      dependencies.resize(state.size);
      natives.resize(state.size);
      frameDependencies = std::move(state.frameDependencies);
    }
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
          x.phaseSteps.size() != y.phaseSteps.size() ||
          !sameKnown(x, y, static_cast<int64_t>(a), static_cast<int64_t>(b)))
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
    // A body that changes or plays a tracked tone's frame lists frame
    // changes that differ from one iteration to the next.
    auto tracks = [&] {
      if (nextUpdate != entry.nextUpdate)
        return true;
      for (size_t i = entry.size; i < result.size(); ++i)
        if (result[i].known >= 0 && known[result[i].known].tone >= 0)
          return true;
      return false;
    };
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
        e && !tracks() ? periodOf(a, b, e, false) : std::optional<int64_t>();

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
    if (track) {
      // A record of the body stands for every iteration: it depends on
      // what any does. A shift adds to what a frame depends on and a set
      // replaces it, so from the second iteration on each depends alike.
      for (size_t i = a; i < b; ++i) {
        for (size_t f = 0; f < dependencies[i].size(); ++f)
          merge(dependencies[i][f], dependencies[i + (b - a)][f]);
        if (natives[i] != natives[i + (b - a)])
          natives[i] = -1;
      }
      dependencies.resize(b);
      natives.resize(b);
      dependencies.insert(dependencies.begin() + static_cast<std::ptrdiff_t>(a),
                          Dependencies());
      natives.insert(natives.begin() + static_cast<std::ptrdiff_t>(a), -1);
    }
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
      shiftKnown(r, shift);
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

// ── Controller clocks ─────────────────────────────────────────────────────

/// A line event that does not start and last whole multiples of its
/// controller's grid: (index, start, duration) in program cycles.
using OffGridEvent = std::tuple<int64_t, int64_t, int64_t>;

/// The records of a walk of every execution in the cycles of a controller
/// counting `num` of its cycles every `den` program cycles: each start,
/// duration and end rounded up, an end kept apart where it is not the start
/// plus the duration. Returns the drives, readouts and waits off its grid of
/// `grid` program cycles.
inline std::vector<OffGridEvent>
ownCycles(std::vector<ControllerRecord> &records, int64_t num, int64_t den,
          int64_t grid) {
  std::vector<OffGridEvent> offGrid;
  auto own = [&](std::optional<int64_t> &cycles) {
    if (cycles)
      *cycles = (*cycles * num + den - 1) / den;
  };
  for (size_t index = 0; index < records.size(); ++index) {
    ControllerRecord &r = records[index];
    if (grid > 1 && r.start && r.duration &&
        (std::strcmp(r.kind, "drive") == 0 ||
         std::strcmp(r.kind, "readout") == 0 ||
         std::strcmp(r.kind, "wait") == 0) &&
        (*r.start % grid || *r.duration % grid))
      offGrid.emplace_back(static_cast<int64_t>(index), *r.start, *r.duration);
    if (num == den)
      continue;
    std::optional<int64_t> end = r.stop();
    own(r.start);
    own(r.duration);
    own(end);
    if (end && *end != *r.start + *r.duration)
      r.end = end;
  }
  return offGrid;
}

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
/// the events that play one share what it returns. Without it, the
/// expression and amplitude are built here. `parameters` names the numbers
/// of each waveform shape, and `amplitudes` holds the shapes with an
/// overall amplitude.
class ControllerEvents {
public:
  /// `compact` (format /2) leaves out what is derivable or the default: the
  /// index (an event's place in pre-order), the qudit (its line's), the end,
  /// an empty iteration, epoch 0, empty conditions and lists, a sync's or
  /// call's line, a timeout that is none and a first iteration of 0; its
  /// `waveform` returns (index into the program's waveforms, None).
  ControllerEvents(const ControllerWalk &walk,
                   const ControllerWaveforms &waveforms, nb::object waveform,
                   nb::handle parameters, nb::handle amplitudes, bool compact)
      : records(walk.result), walk(walk), waveforms(waveforms),
        waveform(std::move(waveform)), parameters(parameters), compact(compact),
        made(waveforms.distinct.size()) {
    if (this->waveform.is_none())
      readShapes(amplitudes);
  }

  nb::list build() {
    size_t at = 0;
    return range(at, records.size());
  }

  /// Frame changes no drive or readout took, each with its `tone` first.
  nb::list unapplied(
      const std::vector<std::pair<int64_t, ControllerWalk::FrameUpdate>> &all)
      const {
    nb::list out;
    for (const auto &[tone, update] : all) {
      nb::dict item;
      item["tone"] = tone;
      nb::dict rest = nb::cast<nb::dict>(frameUpdates({update})[0]);
      for (auto [key, value] : rest)
        item[key] = value;
      out.append(item);
    }
    return out;
  }

private:
  using Expression = ControllerWalk::Expression;
  using Nodes = ControllerWaveforms::Nodes;
  using Operands = std::vector<std::tuple<int64_t, int64_t, int32_t>>;

  const std::vector<ControllerRecord> &records;
  const ControllerWalk &walk;
  const ControllerWaveforms &waveforms;
  nb::object waveform;
  nb::handle parameters;
  bool compact;
  std::vector<std::optional<std::pair<nb::object, nb::object>>> made;

  // The names of a waveform shape's numbers, where its amplitude and scale
  // factor are among them, and whether it has an overall amplitude.
  struct Shape {
    std::vector<nb::object> names;
    int real = -1, imag = -1, factor = -1;
    bool amplitude = false;
  };
  llvm::StringMap<Shape> shapes;

  void readShapes(nb::handle amplitudes) {
    for (nb::handle shape : parameters) {
      Shape &info = shapes[nb::cast<std::string>(shape)];
      info.amplitude = PySequence_Contains(amplitudes.ptr(), shape.ptr()) == 1;
      for (nb::handle name : parameters[shape]) {
        auto text = nb::cast<std::string>(name);
        int index = static_cast<int>(info.names.size());
        if (text == "amplitude")
          info.real = index;
        else if (text == "amplitude_imag")
          info.imag = index;
        else if (text == "factor")
          info.factor = index;
        info.names.push_back(nb::borrow(name));
      }
    }
  }

  /// The names of a node's numbers, if its shape names them all.
  const Shape *named(llvm::StringRef shape, size_t numbers) const {
    auto found = shapes.find(shape);
    if (found == shapes.end() || found->second.names.size() != numbers)
      return nullptr;
    return &found->second;
  }

  /// The waveform expression from node `index` down.
  nb::object tree(const Nodes &nodes, int64_t index) const {
    const auto &[name, numbers, children] = nodes[index];
    auto [shape, custom] = llvm::StringRef(name).split(':');
    nb::dict node;
    node["shape"] = nb::str(shape.data(), shape.size());
    if (!custom.empty())
      node["name"] = nb::str(custom.data(), custom.size());
    const Shape *names = named(shape, numbers.size());
    if (shape == "custom_samples") {
      nb::list samples;
      for (double v : numbers)
        samples.append(nb::float_(v));
      node["samples"] = samples;
    } else if (names) {
      nb::dict params;
      for (size_t i = 0; i < numbers.size(); ++i)
        params[names->names[i]] = number(numbers[i]);
      node["params"] = params;
    } else if (!numbers.empty()) {
      nb::list values;
      for (double v : numbers)
        values.append(number(v));
      node["values"] = values;
    }
    if (!children.empty()) {
      nb::list of;
      for (int64_t child : children)
        of.append(tree(nodes, child));
      node["of"] = of;
    }
    return node;
  }

  /// The overall complex amplitude of node `index`, through scale and neg,
  /// in Python's complex arithmetic; None where it is unknown.
  nb::object complexAmplitude(const Nodes &nodes, int64_t index) const {
    const auto &[name, numbers, children] = nodes[index];
    llvm::StringRef shape = llvm::StringRef(name).split(':').first;
    const Shape *names = named(shape, numbers.size());
    if (shape == "scale" || shape == "neg") {
      if (children.empty())
        return nb::none();
      nb::object inner = complexAmplitude(nodes, children[0]);
      if (inner.is_none())
        return inner;
      if (shape == "neg")
        return checked(PyNumber_Negative(inner.ptr()));
      if (!names || names->factor < 0 || std::isnan(numbers[names->factor]))
        return nb::none();
      nb::float_ factor(numbers[names->factor]);
      return checked(PyNumber_Multiply(inner.ptr(), factor.ptr()));
    }
    if (shape == "custom_samples") {
      if (numbers.empty())
        return nb::none();
      double peak = numbers[0];
      for (double sample : numbers)
        if (std::fabs(sample) > std::fabs(peak))
          peak = sample;
      return checked(PyComplex_FromDoubles(peak, 0.0));
    }
    if (!names || !names->amplitude || names->real < 0)
      return nb::none();
    double re = numbers[names->real];
    double im = names->imag < 0 ? 0.0 : numbers[names->imag];
    if (std::isnan(re) || std::isnan(im))
      return nb::none();
    return checked(PyComplex_FromDoubles(re, im));
  }

  static nb::object checked(PyObject *object) {
    if (!object)
      throw nb::python_error();
    return nb::steal(object);
  }

  /// (expression, amplitude as {re, im}) of a waveform, as the export has
  /// them.
  std::pair<nb::object, nb::object> built(const Nodes &nodes) const {
    if (nodes.empty())
      return {nb::none(), nb::none()};
    int64_t root = static_cast<int64_t>(nodes.size()) - 1;
    nb::object overall = complexAmplitude(nodes, root);
    if (overall.is_none())
      return {tree(nodes, root), nb::none()};
    nb::dict parts;
    parts["re"] = nb::float_(PyComplex_RealAsDouble(overall.ptr()));
    parts["im"] = nb::float_(PyComplex_ImagAsDouble(overall.ptr()));
    return {tree(nodes, root), parts};
  }

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
        if (r.known >= 0)
          runTime(r, walk.known[r.known], event);
      }
      out.append(event);
    }
    return out;
  }

  // The kind of an event in format /2, named for what the controller does
  // rather than for a modality.
  static const char *structuredKind(const char *kind) {
    if (std::strcmp(kind, "drive") == 0)
      return "play";
    if (std::strcmp(kind, "readout") == 0)
      return "acquire";
    if (std::strcmp(kind, "submit") == 0)
      return "call";
    return kind;
  }

  nb::dict common(const ControllerRecord &r, size_t index) {
    nb::dict event;
    bool sync = std::strcmp(r.kind, "sync") == 0;
    bool loop = std::strcmp(r.kind, "loop") == 0;
    bool call = (r.kind[0] == 's' && !sync) || r.kind[0] == 'a';
    if (!compact)
      event["index"] = index;
    event["kind"] = compact ? structuredKind(r.kind) : r.kind;
    if (!compact)
      event["line"] = sync || loop ? nb::none() : nb::object(nb::int_(r.line));
    else if (!sync && !loop && !call)
      event["line"] = r.line;
    if (!compact)
      event["qudit"] =
          sync || loop ? nb::none() : nb::object(nb::int_(call ? -1 : r.qudit));
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
        const Nodes &nodes = waveforms.distinct[waveforms.of.lookup(r.op)];
        if (waveform.is_none()) {
          shape.emplace(built(nodes));
        } else {
          nb::tuple pair = nb::cast<nb::tuple>(waveform(nodes));
          shape.emplace(pair[0], pair[1]);
        }
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
      if (r.qudit >= 0)
        event["controller"] = r.qudit;
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

  // ── Run-time values ──────────────────────────────────────────────────

  /// A number as the JSON text "%.17g" of it reads back: an int where that
  /// text has no fraction or exponent.
  static nb::object textNumber(double value) {
    if (std::isfinite(value) && value == std::trunc(value) &&
        std::fabs(value) < 1e17)
      return nb::int_(static_cast<int64_t>(value));
    return nb::float_(value);
  }

  nb::object expression(int32_t index) const {
    const Expression &e = walk.expressions[index];
    if (e.kind == Expression::Number)
      return textNumber(e.number);
    nb::dict out;
    switch (e.kind) {
    case Expression::Constant:
      out["const"] = textNumber(e.number);
      break;
    case Expression::Unknown:
      out["unknown"] = nb::str(e.name.data(), e.name.size());
      break;
    case Expression::Readout:
      out["readout"] = e.event;
      break;
    case Expression::Answer:
      out["answer"] = e.event;
      out["result"] = e.result;
      break;
    default: {
      nb::list of;
      for (int32_t child : e.of)
        of.append(expression(child));
      out["of"] = of;
      out["op"] = nb::str(e.name.data(), e.name.size());
      if (!e.predicate.empty())
        out["predicate"] = nb::str(e.predicate.data(), e.predicate.size());
    }
    }
    return out;
  }

  /// The name of number `operand` of a node: its parameter's, where the
  /// shape names all its numbers, else its index.
  nb::object parameter(const std::string &name, size_t numbers,
                       int64_t operand) const {
    nb::object names =
        parameters.attr("get")(nb::str(name.substr(0, name.find(':')).c_str()));
    if (!names.is_none() && nb::len(names) == numbers)
      return names[static_cast<size_t>(operand)];
    return nb::str(std::to_string(operand).c_str());
  }

  /// The run-time operands of node `index`, by parameter name; none when it
  /// has none.
  nb::object computed(const Nodes &nodes, int64_t index,
                      const Operands &operands) const {
    const auto &[name, numbers, children] = nodes[index];
    nb::dict out;
    for (const auto &[at, operand, expr] : operands)
      if (at == index)
        out[parameter(name, numbers.size(), operand)] = expression(expr);
    return nb::len(out) ? nb::object(out) : nb::none();
  }

  /// A copy of waveform `tree` (node `index` of `nodes`) with its operands
  /// known only at run time in the `runtime` of their nodes.
  nb::object withRunTime(const Nodes &nodes, int64_t index, nb::handle tree,
                         const Operands &operands) const {
    const auto &children = std::get<2>(nodes[index]);
    nb::dict node = nb::steal<nb::dict>(PyDict_Copy(tree.ptr()));
    if (!children.empty()) {
      nb::object old = node["of"];
      nb::list of;
      for (size_t i = 0; i < children.size(); ++i)
        of.append(withRunTime(nodes, children[i], old[i], operands));
      node["of"] = of;
    }
    nb::object here = computed(nodes, index, operands);
    if (!here.is_none())
      node["runtime"] = here;
    return node;
  }

  /// Format /2: the run-time operands of a waveform as a list of the paths
  /// to their nodes (the index in `of` at each node from the root) with
  /// their operands by parameter name.
  void paths(const Nodes &nodes, int64_t index, nb::list &path, nb::list &out,
             const Operands &operands) const {
    nb::object here = computed(nodes, index, operands);
    if (!here.is_none()) {
      nb::dict item;
      item["path"] = nb::list(path);
      item["params"] = here;
      out.append(item);
    }
    const auto &children = std::get<2>(nodes[index]);
    for (size_t i = 0; i < children.size(); ++i) {
      path.append(i);
      paths(nodes, children[i], path, out, operands);
      PyList_SetSlice(path.ptr(), nb::len(path) - 1, nb::len(path), nullptr);
    }
  }

  nb::object
  frameUpdates(const std::vector<ControllerWalk::FrameUpdate> &all) const {
    nb::list out;
    for (const auto &update : all) {
      nb::list conditions;
      for (const auto &[branch, arm, readouts, answers, test] :
           update.conditions) {
        nb::dict condition;
        if (!compact || !answers.empty())
          condition["answers"] = ints(answers);
        condition["arm"] = static_cast<bool>(arm);
        condition["branch"] = branch;
        if (!compact || !readouts.empty())
          condition["readouts"] = ints(readouts);
        condition["test"] = test;
        conditions.append(condition);
      }
      nb::dict item;
      if (!compact || nb::len(conditions))
        item["conditions"] = conditions;
      item["id"] = update.seq;
      item["op"] = nb::str(update.op.data(), update.op.size());
      item["value"] = expression(update.value);
      out.append(item);
    }
    return out;
  }

  /// What `r` has that is known only at run time, after the rest of
  /// `event`; format /1 adds the waveform's in place, in its nodes.
  void runTime(const ControllerRecord &r, const ControllerWalk::Known &known,
               nb::dict &event) {
    if (!known.waveform.empty() && r.played()) {
      const Nodes &nodes = waveforms.distinct[waveforms.of.lookup(r.op)];
      Operands sorted = known.waveform;
      std::stable_sort(sorted.begin(), sorted.end(),
                       [](const auto &a, const auto &b) {
                         return std::make_pair(std::get<0>(a), std::get<1>(a)) <
                                std::make_pair(std::get<0>(b), std::get<1>(b));
                       });
      int64_t root = static_cast<int64_t>(nodes.size()) - 1;
      if (!compact) {
        event["waveform"] = withRunTime(nodes, root, event["waveform"], sorted);
      } else {
        nb::list path, out;
        paths(nodes, root, path, out, sorted);
        event["runtime"] = out;
      }
    }
    if (known.tone >= 0) {
      event["tone"] = known.tone;
      event["frame_updates"] = frameUpdates(known.updates);
      if (known.offset)
        event["phase_offset_rad"] = textNumber(*known.offset);
    }
    if (known.length >= 0) {
      event["length"] = expression(known.length);
      if (!compact || known.minCycles)
        event["min_cycles"] = optional(known.minCycles);
      if (!compact || known.maxCycles)
        event["max_cycles"] = optional(known.maxCycles);
    }
    if (known.operands) {
      nb::list operands;
      for (int32_t expr : *known.operands)
        operands.append(expression(expr));
      event["operands"] = operands;
    }
  }
};
