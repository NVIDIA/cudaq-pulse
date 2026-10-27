/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// nanobind Python bindings for cudaq-pulse MLIR dialects and passes.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <bit>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <thread>

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/Support/TargetSelect.h"

#include <dlfcn.h>
#include <pthread.h>

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
#include "cudaq-pulse/Analysis/Execution.h"
#include "cudaq-pulse/Analysis/LineTrace.h"
#include "cudaq-pulse/Analysis/Timing.h"
#include "cudaq-pulse/Dialect/Classical/Answers.h"
#include "cudaq-pulse/Runtime/WaveformCallback.h"
#include "cudaq-pulse/Transforms/Passes.h"

#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatDialect.h.inc"
#include "cudaq-pulse/Dialect/QOp/QOpDialect.h.inc"

namespace nb = nanobind;

namespace {

mlir::DialectRegistry makeRegistry() {
  mlir::DialectRegistry reg;
  reg.insert<pulse::PulseDialect>();
  reg.insert<classical::ClassicalDialect>();
  reg.insert<qop::QOpDialect>();
  reg.insert<cudm::CuDensityMatDialect>();
  reg.insert<mlir::arith::ArithDialect>();
  reg.insert<mlir::func::FuncDialect>();
  reg.insert<mlir::scf::SCFDialect>();
  reg.insert<mlir::ub::UBDialect>();
  reg.insert<mlir::LLVM::LLVMDialect>();
  return reg;
}

// Packed-buffer opcodes; they must match the opcodes in
// core/frontend/cudaq_pulse/kernel/packed_ir_builder.py.
constexpr int kOpAllocDrive = 0;
constexpr int kOpAllocReadout = 1;
constexpr int kOpAllocTone = 2;
constexpr int kOpWfGaussian = 3;
constexpr int kOpWfSquare = 4;
constexpr int kOpWfDrag = 5;
constexpr int kOpWfCosine = 6;
constexpr int kOpWfTanhRamp = 7;
constexpr int kOpWfGaussSquare = 8;
constexpr int kOpWfCustom = 9;
constexpr int kOpDrive = 10;
constexpr int kOpReadout = 11;
constexpr int kOpSync = 12;
constexpr int kOpWait = 13;
constexpr int kOpShiftPhase = 14;
constexpr int kOpSetPhase = 15;
constexpr int kOpShiftFreq = 16;
constexpr int kOpSetFreq = 17;
constexpr int kOpParam = 18;
constexpr int kOpNumConst = 19;
constexpr int kOpNumBinary = 20;
constexpr int kOpNumNeg = 21;
constexpr int kOpNumCast = 22;
constexpr int kOpWfCustomSamples = 23;
constexpr int kOpWfAdd = 24;
constexpr int kOpWfSub = 25;
constexpr int kOpWfMul = 26;
constexpr int kOpWfScale = 27;
constexpr int kOpWfNeg = 28;
constexpr int kOpForBegin = 29;
constexpr int kOpForEnd = 30;
constexpr int kOpIfBegin = 31;
constexpr int kOpElse = 32;
constexpr int kOpIfEnd = 33;
constexpr int kOpBlockBegin = 34;
constexpr int kOpBlockEnd = 35;
constexpr int kOpClassicalDeclare = 36;
constexpr int kOpClassicalSubmit = 37;
constexpr int kOpClassicalWait = 38;
constexpr int kOpClassicalCompare = 39;
constexpr int kOpVariableWait = 40;
// A payload length of 0xFF means the length is in the next word.
constexpr int kExtendedPayload = 0xFF;

/// Collects the error diagnostics emitted while it is alive, so that a
/// failure can report them in the Python exception instead of on stderr.
class DiagnosticCollector {
public:
  explicit DiagnosticCollector(mlir::MLIRContext *ctx)
      : owner_(std::this_thread::get_id()),
        handler_(ctx, [this](mlir::Diagnostic &diagnostic) {
          // Modules from specialize() share a context; another thread's
          // pipeline may be running on it too.
          if (diagnostic.getSeverity() != mlir::DiagnosticSeverity::Error ||
              std::this_thread::get_id() != owner_)
            return mlir::failure();
          llvm::raw_string_ostream os(messages_);
          os << "\n  " << diagnostic.getLocation() << ": " << diagnostic;
          for (mlir::Diagnostic &note : diagnostic.getNotes())
            os << "\n    note: " << note;
          return mlir::success();
        }) {}

  [[noreturn]] void fail(const std::string &what) const {
    throw std::runtime_error(what + messages_);
  }

private:
  std::string messages_;
  std::thread::id owner_;
  mlir::ScopedDiagnosticHandler handler_;
};

/// Parse `text`, raising with the parser diagnostics on failure.
mlir::OwningOpRef<mlir::ModuleOp> parseModule(const std::string &text,
                                              mlir::MLIRContext *ctx,
                                              const std::string &what) {
  DiagnosticCollector diagnostics(ctx);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(text, ctx);
  if (!module)
    diagnostics.fail(what);
  return module;
}

/// Run `pm` on `op`, raising with the collected diagnostics on failure.
void runPipeline(mlir::PassManager &pm, mlir::Operation *op,
                 const std::string &what) {
  DiagnosticCollector diagnostics(pm.getContext());
  if (mlir::failed(pm.run(op)))
    diagnostics.fail(what);
}

/// The pass registered as `name`, or null. The registry is filled once, on
/// first use from any thread.
const mlir::PassInfo *lookupPass(llvm::StringRef name) {
  static const bool registered = [] {
    pulse::registerPulsePasses();
    mlir::registerLoopInvariantCodeMotionPass();
    return true;
  }();
  (void)registered;
  return mlir::PassInfo::lookup(name);
}

/// The passes of `entry`, split at the commas outside option braces.
llvm::SmallVector<llvm::StringRef, 4> splitPipeline(llvm::StringRef entry) {
  llvm::SmallVector<llvm::StringRef, 4> passes;
  int depth = 0;
  size_t start = 0;
  for (size_t i = 0; i < entry.size(); ++i) {
    if (entry[i] == '{')
      ++depth;
    else if (entry[i] == '}')
      --depth;
    else if (entry[i] == ',' && depth == 0) {
      passes.push_back(entry.slice(start, i).trim());
      start = i + 1;
    }
  }
  passes.push_back(entry.drop_front(start).trim());
  return passes;
}

inline double i2f(int64_t bits) {
  double d;
  std::memcpy(&d, &bits, sizeof(double));
  return d;
}

/// The walk behind PulseModule::controller_events(): every timed op, sync
/// and classical call once per loop iteration, both arms of every branch,
/// with the frame of its tone and the branches it lies in.
struct ControllerWalk {
  using WaveformNode =
      std::tuple<std::string, std::vector<double>, std::vector<int64_t>>;
  using Condition = std::tuple<int64_t, int64_t, std::vector<int64_t>,
                               std::vector<int64_t>, std::string>;
  using Event =
      std::tuple<std::string, int64_t, int64_t, std::optional<int64_t>,
                 std::optional<int64_t>, std::vector<int64_t>,
                 std::vector<WaveformNode>, double, double,
                 std::vector<Condition>, std::vector<int64_t>,
                 std::vector<int64_t>, std::string, std::string, double>;

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
      auto same = [](double a, double b) {
        return a == b || (std::isnan(a) && std::isnan(b));
      };
      return same(frequencyHz, other.frequencyHz) &&
             same(frameFrequencyHz, other.frameFrequencyHz) &&
             same(phase, other.phase) &&
             (same(timeNs, other.timeNs) || frequencyHz == frameFrequencyHz);
    }
  };

  explicit ControllerWalk(double clockGHz) : clockGHz(clockGHz) {}

  double clockGHz;
  llvm::DenseMap<mlir::Operation *, int64_t> lineIndex;
  std::vector<Event> result;

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
  // The event of the latest execution of each readout and classical wait.
  llvm::DenseMap<mlir::Operation *, int64_t> latest;

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

  /// The readout events and answer events `value` is computed from.
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
    std::vector<WaveformNode> nodes;
    double frequency = kUnknown, phase = kUnknown;
    if (!mlir::isa<pulse::WaitOp>(op)) {
      waveform(op->getOperand(1), nodes);
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
    result.emplace_back(
        op->getName().stripDialect().str(), lineOf(allocation), qubit, start,
        duration, std::vector<int64_t>(iteration.begin(), iteration.end()),
        std::move(nodes), frequency, phase, branches, std::vector<int64_t>{},
        std::vector<int64_t>{}, std::string(), std::string(), kUnknown);
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
    result.emplace_back(
        "sync", -1, -1, time,
        time ? std::optional<int64_t>(0) : std::optional<int64_t>(),
        std::vector<int64_t>(iteration.begin(), iteration.end()),
        std::vector<WaveformNode>{}, kUnknown, kUnknown, branches,
        std::move(lines), std::vector<int64_t>{}, std::string(), std::string(),
        kUnknown);
  }

  /// A submission (with the readout events whose measurements it sends) or
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
    result.emplace_back(
        submit ? "submit" : "answer",
        classical::intAttr(op, "classical.device"), -1,
        pulse::scheduledStart(op, iteration), pulse::scheduledDuration(op),
        std::vector<int64_t>(iteration.begin(), iteration.end()),
        std::vector<WaveformNode>{}, kUnknown, kUnknown, branches,
        std::vector<int64_t>{}, std::move(refs), std::move(callee),
        std::move(mode), timeoutNs);
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
      iteration.push_back(0);
      for (int64_t k = 0; k < trips; ++k) {
        iteration.back() = k;
        block(*loop.getBody());
      }
      iteration.pop_back();
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
};

/// The epochs of a scheduled program, in program order. A variable wait or a
/// blocking wait ends the current epoch of its timeline (the lines it holds
/// back) and opens the next one, which starts after it: epoch k > 0 is
/// opened by the k-th such wait, and follows `after`, the epoch its timeline
/// was in. Without timelines every epoch follows the one before.
struct EpochWalk {
  /// (kind, line or link, start, lowest, highest, after) per boundary: see
  /// PulseModule::epoch_boundaries().
  using Boundary =
      std::tuple<std::string, int64_t, int64_t, int64_t, int64_t, int64_t>;

  explicit EpochWalk(mlir::ModuleOp module) {
    module.walk([&](mlir::Operation *op) {
      if (mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(op))
        lineIndex.try_emplace(op, static_cast<int64_t>(lineIndex.size()));
    });
  }

  llvm::DenseMap<mlir::Operation *, int64_t> lineIndex;
  pulse::AllocationTracer trace;
  std::map<int64_t, int64_t> current;
  std::vector<Boundary> boundaries;

  int64_t timelineOf(mlir::Operation *op) {
    if (op->hasAttr(classical::kTimelineAttr))
      return classical::intAttr(op, classical::kTimelineAttr);
    for (mlir::Value operand : op->getOperands())
      if (mlir::isa<pulse::DriveLineType, pulse::ReadoutLineType,
                    pulse::ToneType>(operand.getType())) {
        mlir::Operation *allocation = trace(operand);
        return allocation
                   ? classical::intAttr(allocation, classical::kTimelineAttr)
                   : 0;
      }
    return 0;
  }

  /// The epoch `op` runs in, in iteration `iteration`; a boundary opens the
  /// next. A variable wait runs in the epoch it ends, a blocking wait's
  /// answer is taken in the one it opens.
  int64_t visit(mlir::Operation *op, llvm::ArrayRef<int64_t> iteration) {
    int64_t &epoch = current[timelineOf(op)];
    if (!op->hasAttr("classical.variable") &&
        !op->hasAttr("classical.blocking"))
      return epoch;
    auto start = pulse::scheduledStart(op, iteration);
    if (!start)
      return epoch;
    int64_t after = epoch;
    if (mlir::isa<pulse::WaitOp>(op)) {
      mlir::Operation *allocation = trace(op->getOperand(0));
      boundaries.emplace_back(
          "wait", allocation ? lineIndex.lookup(allocation) : -1, *start,
          classical::intAttr(op, "pulse.min_cycles"),
          classical::intAttr(op, "pulse.max_cycles", -1), after);
    } else {
      boundaries.emplace_back("answer",
                              classical::intAttr(op, "classical.device"),
                              *start, 0, -1, after);
    }
    epoch = static_cast<int64_t>(boundaries.size());
    return mlir::isa<pulse::WaitOp>(op) ? after : epoch;
  }

  /// Visit every op of `block` once per loop iteration, both arms of each
  /// branch, as the event listings do; `fn(op, epoch)` for each timed op,
  /// call and sync.
  template <typename Fn>
  void run(mlir::Block &block, Fn &&fn) {
    for (mlir::Operation &op : block) {
      if (mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp,
                    pulse::SyncOp, classical::SubmitOp, classical::WaitOp>(
              op)) {
        fn(&op, visit(&op, iteration));
      } else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
        int64_t trips = pulse::constantTripCount(loop).value_or(0);
        iteration.push_back(0);
        for (int64_t k = 0; k < trips; ++k) {
          iteration.back() = k;
          run(*loop.getBody(), fn);
        }
        iteration.pop_back();
      } else {
        for (mlir::Region &region : op.getRegions())
          for (mlir::Block &nested : region)
            run(nested, fn);
      }
    }
  }

  llvm::SmallVector<int64_t, 2> iteration;
};

// -----------------------------------------------------------------------
// PulseModule: opaque handle around an in-memory mlir::ModuleOp
// -----------------------------------------------------------------------
// The wrapper owns the MLIRContext and the ModuleOp; run_passes() looks
// passes up by their cudaq-pulse-opt names.
class PulseModule {
public:
  PulseModule(std::shared_ptr<mlir::MLIRContext> ctx,
              mlir::OwningOpRef<mlir::ModuleOp> mod)
      : ctx_(std::move(ctx)), module_(std::move(mod)),
        lock_(std::make_shared<std::recursive_mutex>()) {}

  std::string print() {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    std::string result;
    llvm::raw_string_ostream os(result);
    module_->print(os);
    return result;
  }

  /// Run the passes named in `pipeline`, in order: each entry is a pass as
  /// cudaq-pulse-opt names it, with its options in braces, or several
  /// separated by commas. A pass that runs on functions runs on each
  /// function of the module. Pattern passes in a row (pulse-fold,
  /// pulse-canonicalize, pulse-virtual-z, pulse-fusion) run as one rewrite
  /// with all their patterns: one walk of the module, to where none of them
  /// applies.
  void run_passes(const std::vector<std::string> &pipeline) {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    mlir::PassManager pm(ctx_.get(), mlir::ModuleOp::getOperationName(),
                         mlir::OpPassManager::Nesting::Implicit);
    llvm::SmallVector<const pulse::RewriteStage *, 4> stages;
    bool fold = false;
    auto flush = [&] {
      if (stages.empty() && fold)
        stages.push_back(pulse::findRewriteStage("pulse-fold"));
      if (!stages.empty())
        pm.addNestedPass<mlir::func::FuncOp>(
            pulse::createPulseRewritePass(stages, fold));
      stages.clear();
      fold = false;
    };
    for (const std::string &entry : pipeline)
      for (llvm::StringRef pass : splitPipeline(entry)) {
        auto [name, options] = pass.split('{');
        if (const pulse::RewriteStage *stage = pulse::findRewriteStage(name);
            stage && options.empty()) {
          if (!stage->populate) {
            fold = true;
            continue;
          }
          stages.push_back(stage);
          if (!llvm::is_contained(rewrites_, stage)) {
            rewrites_.push_back(stage);
            rewritePatterns_.reset();
          }
          continue;
        }
        flush();
        const mlir::PassInfo *info = lookupPass(name);
        if (!info)
          throw std::runtime_error("Unknown pass: " + name.str());
        if (mlir::failed(info->addToPipeline(
                pm, options.empty() ? options : options.drop_back(),
                [](const llvm::Twine &) { return mlir::failure(); })))
          throw std::runtime_error("Cannot add pass: " + pass.str());
      }
    flush();
    programs_.clear();
    runPipeline(pm, *module_, "MLIR pass pipeline failed");
  }

  std::string run_full_lowering() {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    // Lower a clone so inspection does not consume the scheduled Pulse module;
    // callers may lower repeatedly and still execute the original artifact.
    auto lowered = mlir::OwningOpRef<mlir::ModuleOp>(module_->clone());
    mlir::PassManager pm(ctx_.get());
    pm.addPass(pulse::createPulseToQOpPass());
    pm.addPass(qop::createQOpToCuDensityMatPass());
    pm.addPass(cudm::createCuDensityMatToLLVMPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    runPipeline(pm, *lowered, "MLIR full lowering pipeline failed");
    std::string result;
    llvm::raw_string_ostream os(result);
    lowered->print(os);
    return result;
  }

  void schedule(const std::string &policy, int64_t maxDrives = 4,
                int64_t maxReadouts = 2, int64_t readoutLatency = 0,
                int64_t switchPenalty = 0) {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    mlir::PassManager pm(ctx_.get());
    addSchedule(pm, *module_, true, policy, maxDrives, maxReadouts,
                readoutLatency, switchPenalty);
    programs_.clear();
    keepSource();
    runPipeline(pm, *module_, "pulse scheduling failed");
  }

  using Event = std::tuple<std::string, int64_t, int64_t,
                           std::optional<int64_t>, std::optional<int64_t>>;

  /// (kind, line, qubit, start_cycles, duration_cycles) for every drive,
  /// readout and wait, once per loop iteration and in program order. Both arms
  /// of a conditional are listed. Lines are numbered in allocation order. A
  /// classical call adds a "submit" and an "answer" event, whose line is the
  /// link id and whose qubit is -1.
  std::vector<Event> events() {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    llvm::DenseMap<mlir::Operation *, int64_t> lineIndex;
    module_->walk([&](mlir::Operation *op) {
      if (mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(op))
        lineIndex.try_emplace(op, static_cast<int64_t>(lineIndex.size()));
    });
    std::vector<Event> result;
    llvm::SmallVector<int64_t, 2> iteration;
    pulse::AllocationTracer trace;
    std::function<void(mlir::Block &)> collect = [&](mlir::Block &block) {
      for (mlir::Operation &op : block) {
        if (mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp>(op)) {
          mlir::Operation *allocation = trace(op.getOperand(0));
          int64_t line = allocation ? lineIndex.lookup(allocation) : -1;
          int64_t qubit = -1;
          if (allocation)
            if (auto attr =
                    allocation->getAttrOfType<mlir::IntegerAttr>("qubit"))
              qubit = attr.getInt();
          result.emplace_back(op.getName().stripDialect().str(), line, qubit,
                              pulse::scheduledStart(&op, iteration),
                              pulse::scheduledDuration(&op));
        } else if (mlir::isa<classical::SubmitOp, classical::WaitOp>(op)) {
          // A classical call's events are on its link; they have no qudit.
          result.emplace_back(mlir::isa<classical::SubmitOp>(op) ? "submit"
                                                                 : "answer",
                              classical::intAttr(&op, "classical.device"), -1,
                              pulse::scheduledStart(&op, iteration),
                              pulse::scheduledDuration(&op));
        } else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
          int64_t trips = pulse::constantTripCount(loop).value_or(0);
          iteration.push_back(0);
          for (int64_t k = 0; k < trips; ++k) {
            iteration.back() = k;
            collect(*loop.getBody());
          }
          iteration.pop_back();
        } else {
          for (mlir::Region &region : op.getRegions())
            for (mlir::Block &nested : region)
              collect(nested);
        }
      }
    };
    collect(*module_->getBody());
    return result;
  }

  using ControllerLine = std::tuple<std::string, int64_t>;
  using ControllerProgram = std::tuple<std::vector<ControllerLine>,
                                       std::vector<ControllerWalk::Event>>;

  /// The lines, (role, qubit) in allocation order, and the timed ops of
  /// `events()`, in the same order, every sync and every classical call,
  /// with what a controller needs to play them: (kind, line, qubit,
  /// start_cycles, duration_cycles, loop iteration, waveform, tone frequency in
  /// Hz and frame phase in radians at the op's start, the branches it lies in,
  /// the lines a sync joins, references, callee, wait mode, timeout). A
  /// waveform is its expression's nodes, children before their parent: (op name
  /// without the dialect, its numeric operands or the samples of
  /// `custom_samples`, NaN where one is not a number, the indices of its
  /// waveform operands). A branch is (branch id, 1 for the then arm and 0 for
  /// the else arm, the readout events and the answer events the condition is
  /// computed from, "is_set" when it is one readout's `pulse.is_set`, "answer"
  /// when it is one answer, else "expression"). The frame is tracked through
  /// both arms of a branch; where the arms leave it different, or a frame
  /// change is not a number, it is NaN from there on. A sync starts when the
  /// last of its lines is free and lasts 0 cycles. A classical call's line is
  /// its link: the references of a submission are the readout events whose
  /// measurements it sends, the callee is its function, and an answer's wait is
  /// "bounded", "blocking" or "timeout", with its timeout in ns (NaN for
  /// none).
  ControllerProgram controller_events() {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    ControllerWalk walk(clock());
    std::vector<ControllerLine> lines;
    module_->walk([&](mlir::Operation *op) {
      if (!mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(op))
        return;
      if (!walk.lineIndex
               .try_emplace(op, static_cast<int64_t>(walk.lineIndex.size()))
               .second)
        return;
      int64_t qubit = -1;
      if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("qubit"))
        qubit = attr.getInt();
      lines.emplace_back(
          mlir::isa<pulse::GetDriveLineOp>(op) ? "drive" : "readout", qubit);
    });
    walk.block(*module_->getBody());
    return {std::move(lines), std::move(walk.result)};
  }

  /// (function, link, iteration, submit start, submit duration, due,
  /// taken, first use, deadline, epoch) for every execution of a classical
  /// call, in program order; times in cycles. `taken` is the wait's start
  /// (absent for an answer never taken), `first use` the earliest start of a
  /// pulse computed from the answer, `deadline` the latest due time the
  /// function's deadline allows, and `epoch` the epoch it is submitted in.
  using ClassicalCall =
      std::tuple<std::string, int64_t, std::vector<int64_t>, int64_t, int64_t,
                 int64_t, std::optional<int64_t>, std::optional<int64_t>,
                 std::optional<int64_t>, int64_t>;

  std::vector<ClassicalCall> classical_calls() {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    std::vector<ClassicalCall> calls;
    if (!classical::declaresClassicalCalls(*module_))
      return calls;
    double clock = classical::moduleClockGHz(*module_);
    mlir::SymbolTable symbols(*module_);
    // The timed ops computed from each wait's answer, found once.
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *>>
        consumers;
    auto consumersOf = [&](classical::WaitOp wait) {
      auto [it, inserted] = consumers.try_emplace(wait);
      if (!inserted)
        return it->second;
      llvm::SmallVector<mlir::Value> work(wait.getValues().begin(),
                                          wait.getValues().end());
      llvm::DenseSet<mlir::Operation *> seen;
      while (!work.empty()) {
        mlir::Value value = work.pop_back_val();
        for (mlir::Operation *user : value.getUsers()) {
          if (!seen.insert(user).second)
            continue;
          if (mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp,
                        classical::SubmitOp>(user)) {
            it->second.push_back(user);
          } else if (user->getNumRegions() > 0) {
            user->walk([&](mlir::Operation *inner) {
              if (mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp,
                            classical::SubmitOp>(inner))
                it->second.push_back(inner);
            });
          } else {
            work.append(user->getResults().begin(), user->getResults().end());
          }
        }
      }
      return it->second;
    };
    auto startOf = [&](mlir::Operation *op, llvm::ArrayRef<int64_t> iteration) {
      llvm::SmallVector<int64_t, 2> at(iteration.begin(), iteration.end());
      at.resize(pulse::enclosingLoops(op).size(), 0);
      return pulse::scheduledStart(op, at);
    };
    EpochWalk epochs(*module_);
    llvm::SmallVector<int64_t, 2> iteration;
    std::function<void(mlir::Block &)> collect = [&](mlir::Block &block) {
      for (mlir::Operation &op : block) {
        if (auto submit = mlir::dyn_cast<classical::SubmitOp>(op)) {
          int64_t epoch = epochs.visit(&op, iteration);
          auto start = pulse::scheduledStart(&op, iteration);
          if (!start)
            continue;
          auto callable =
              symbols.lookup<classical::CallableOp>(submit.getCallee());
          int64_t duration = pulse::scheduledDuration(&op).value_or(1);
          int64_t due = *start + duration +
                        classical::intAttr(&op, "classical.budget_cycles");
          std::optional<int64_t> taken, firstUse, deadline;
          for (mlir::Operation *user : submit.getFuture().getUsers())
            if (auto wait = mlir::dyn_cast<classical::WaitOp>(user)) {
              taken = startOf(wait, iteration);
              for (mlir::Operation *consumer : consumersOf(wait))
                if (auto at = startOf(consumer, iteration))
                  firstUse = firstUse ? std::min(*firstUse, *at) : *at;
            }
          if (callable)
            if (auto limit = callable.getDeadlineNs())
              deadline = *start +
                         classical::ceilCycles(limit->convertToDouble(), clock);
          calls.emplace_back(
              submit.getCallee().str(),
              classical::intAttr(&op, "classical.device"),
              std::vector<int64_t>(iteration.begin(), iteration.end()), *start,
              duration, due, taken, firstUse, deadline, epoch);
        } else if (mlir::isa<classical::WaitOp, pulse::WaitOp>(op)) {
          epochs.visit(&op, iteration);
        } else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
          int64_t trips = pulse::constantTripCount(loop).value_or(0);
          iteration.push_back(0);
          for (int64_t k = 0; k < trips; ++k) {
            iteration.back() = k;
            collect(*loop.getBody());
          }
          iteration.pop_back();
        } else {
          for (mlir::Region &region : op.getRegions())
            for (mlir::Block &nested : region)
              collect(nested);
        }
      }
    };
    collect(*module_->getBody());
    return calls;
  }

  /// The boundaries between epochs, in program order, once per loop
  /// iteration: (kind, line or link, start, lowest, highest, after). A "wait"
  /// is a variable wait on a line, whose length lies in [lowest, highest]
  /// cycles (highest -1 if unbounded); an "answer" is a blocking wait on a
  /// link, whose length is the answer's latency (lowest 0, highest -1). The
  /// k-th boundary starts epoch k when it ends; `after` is the epoch it
  /// ends, of the lines it holds back.
  std::vector<EpochWalk::Boundary> epoch_boundaries() {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    if (!classical::declaresClassicalCalls(*module_))
      return {};
    EpochWalk walk(*module_);
    walk.run(*module_->getBody(), [](mlir::Operation *, int64_t) {});
    return std::move(walk.boundaries);
  }

  /// The epoch of each event of `events()` (with `syncs`, of
  /// `controller_events()`), in the same order.
  std::vector<int64_t> event_epochs(bool syncs) {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    std::vector<int64_t> epochs;
    EpochWalk walk(*module_);
    walk.run(*module_->getBody(), [&](mlir::Operation *op, int64_t epoch) {
      if (syncs || !mlir::isa<pulse::SyncOp>(op))
        epochs.push_back(epoch);
    });
    return epochs;
  }

  // ── Simulating classical calls ──────────────────────────────────────
  //
  // Simulation binds the answer of each call as a constant, one execution
  // at a time along a measurement path, and schedules again: the timeline
  // of a path is static once its answers are known.

  /// This module with loops that hold calls unrolled and scheduled again
  /// with `policy`, for simulation.
  PulseModule simulation_module(const std::string &policy, int64_t maxDrives,
                                int64_t maxReadouts, int64_t readoutLatency,
                                int64_t switchPenalty) {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    auto copy = mlir::OwningOpRef<mlir::ModuleOp>(source().clone());
    mlir::PassManager unroll(ctx_.get());
    unroll.addNestedPass<mlir::func::FuncOp>(
        pulse::createPulseUnrollClassicalLoopsPass());
    runPipeline(unroll, *copy, "simulation: unrolling classical loops failed");
    PulseModule result(ctx_, std::move(copy));
    result.keepSource();
    mlir::PassManager pm(ctx_.get());
    addSchedule(pm, *result.module_, false, policy, maxDrives, maxReadouts,
                readoutLatency, switchPenalty);
    runPipeline(pm, *result.module_, "simulation: scheduling failed");
    return result;
  }

  /// One argument of a call: (kind, readout index, iteration, value), kind 0
  /// for a measurement and 1 for a number.
  using CallArgument =
      std::tuple<int64_t, int64_t, std::vector<int64_t>, double>;
  /// (submit index, start ns, callee, arguments, answered, timeout ns or -1)
  using PendingCall = std::tuple<int64_t, double, std::string,
                                 std::vector<CallArgument>, bool, double>;

  /// The first call on `path` that has no answer bound yet, by start time.
  std::optional<PendingCall>
  next_call(const std::vector<std::vector<int64_t>> &path) {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    auto readouts = readoutOps();
    pulse::Path chosen = toPath(path, readouts);
    llvm::DenseMap<mlir::Operation *, int64_t> indexOf;
    module_->walk([&](classical::SubmitOp submit) {
      indexOf.try_emplace(submit, static_cast<int64_t>(indexOf.size()));
    });
    double clockGHz = clock();
    std::optional<PendingCall> next;
    pulse::ExecutionWalker walker(clockGHz, chosen);
    walker.onSubmit = [&](const pulse::Occurrence &occurrence) {
      mlir::Operation *op = occurrence.op;
      if (op->hasAttr("classical.bound"))
        return false;
      double start = occurrence.startCycles / clockGHz;
      if (next && std::get<1>(*next) <= start)
        return false;
      auto submit = mlir::cast<classical::SubmitOp>(op);
      std::vector<CallArgument> args;
      for (mlir::Value operand : submit.getArgs()) {
        if (auto key = walker.measurementKeyOf(operand)) {
          int64_t index = std::distance(readouts.begin(),
                                        llvm::find(readouts, key->readout));
          args.emplace_back(0, index, key->iteration, 0.0);
        } else if (auto value = pulse::ExecutionWalker::number(operand)) {
          args.emplace_back(1, -1, std::vector<int64_t>{}, *value);
        } else {
          throw std::runtime_error(
              "simulation: an argument of a classical call is neither a "
              "measurement nor a number known on this path");
        }
      }
      bool answered = false;
      double timeout = -1.0;
      for (classical::WaitOp wait : classical::waitsOf(submit)) {
        answered = true;
        if (auto t = wait.getTimeoutNs())
          timeout = t->convertToDouble();
      }
      next = PendingCall{indexOf.lookup(op), start,    submit.getCallee().str(),
                         std::move(args),    answered, timeout};
      return false;
    };
    auto function = getFuncOp();
    if (!function)
      throw std::runtime_error("no 'main' func in module");
    if (!walker.run(function))
      throw std::runtime_error(walker.error);
    return next;
  }

  /// This module with the answer `values` bound to execution `index` of a
  /// call (an empty `values` for a call whose answer is not taken), folded
  /// and scheduled again with `policy`.
  PulseModule bind_call(int64_t index, const std::vector<double> &values,
                        const std::string &policy, int64_t maxDrives,
                        int64_t maxReadouts, int64_t readoutLatency,
                        int64_t switchPenalty) {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    bool unscheduled = static_cast<bool>(source_);
    auto copy = mlir::OwningOpRef<mlir::ModuleOp>(source().clone());
    classical::SubmitOp found;
    int64_t seen = 0;
    copy->walk([&](classical::SubmitOp submit) {
      if (seen++ == index)
        found = submit;
    });
    if (!found)
      throw std::invalid_argument("bind_call: no call execution " +
                                  std::to_string(index));
    if (unscheduled) {
      int64_t scheduled = 0;
      module_->walk([&](classical::SubmitOp) { ++scheduled; });
      if (scheduled != seen)
        throw std::runtime_error(
            "bind_call: the module has " + std::to_string(scheduled) +
            " call executions and its source " + std::to_string(seen));
    }
    found->setAttr("classical.bound", mlir::UnitAttr::get(ctx_.get()));
    bool rebound = false;
    for (classical::WaitOp wait : classical::waitsOf(found)) {
      if (wait.getValues().size() != values.size())
        throw std::invalid_argument(
            "bind_call: " + found.getCallee().str() + " answers " +
            std::to_string(wait.getValues().size()) + " values, got " +
            std::to_string(values.size()));
      bindConstants(wait, values);
      rebound = true;
    }
    PulseModule result(ctx_, std::move(copy));
    if (rebound || unscheduled)
      result.schedule(policy, maxDrives, maxReadouts, readoutLatency,
                      switchPenalty);
    return result;
  }

  /// This module with every answer not bound yet replaced by zeros, for
  /// evolving up to the next call: those answers only affect what follows.
  PulseModule placeholders() {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    auto copy = mlir::OwningOpRef<mlir::ModuleOp>(module_->clone());
    llvm::SmallVector<classical::WaitOp> waits;
    copy->walk([&](classical::WaitOp wait) {
      if (!wait->hasAttr("classical.answers"))
        waits.push_back(wait);
    });
    for (classical::WaitOp wait : waits)
      bindConstants(wait, std::vector<double>(wait.getValues().size(), 0.0));
    return PulseModule(ctx_, std::move(copy));
  }

  using Measurement =
      std::tuple<int64_t, std::vector<int64_t>, int64_t, double, double>;

  /// The first measurement to complete on `path` (entries [readout index,
  /// iteration..., outcome]) that the path gives no outcome for, as
  /// (readout index, iteration, qubit, start ns, end ns).
  std::optional<Measurement>
  next_measurement(const std::vector<std::vector<int64_t>> &path) {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    auto readouts = readoutOps();
    pulse::Path chosen = toPath(path, readouts);
    double clockGHz = clock();
    std::optional<Measurement> next;
    pulse::ExecutionWalker walker(clockGHz, chosen);
    pulse::AllocationTracer trace;
    walker.onReadout = [&](const pulse::Occurrence &occurrence,
                           const pulse::MeasurementKey &key) {
      if (chosen.count(key))
        return false;
      double end =
          (occurrence.startCycles + occurrence.durationCycles) / clockGHz;
      if (next && std::get<4>(*next) <= end)
        return false;
      int64_t index =
          std::distance(readouts.begin(), llvm::find(readouts, occurrence.op));
      int64_t qubit = -1;
      if (auto *allocation = trace(occurrence.op->getOperand(0)))
        if (auto attr = allocation->getAttrOfType<mlir::IntegerAttr>("qubit"))
          qubit = attr.getInt();
      next = Measurement{index, key.iteration, qubit,
                         occurrence.startCycles / clockGHz, end};
      return false;
    };
    auto function = getFuncOp();
    if (!function)
      throw std::runtime_error("no 'main' func in module");
    if (!walker.run(function))
      throw std::runtime_error(walker.error);
    return next;
  }

  using ReadoutDrive =
      std::tuple<int64_t, std::vector<int64_t>, double, double>;

  /// Every readout on `path` as (readout index, iteration, tone frequency in
  /// Hz, peak amplitude of its pulse); NaN where a value is not a number.
  std::vector<ReadoutDrive>
  readout_drives(const std::vector<std::vector<int64_t>> &path) {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    auto readouts = readoutOps();
    pulse::Path chosen = toPath(path, readouts);
    std::vector<ReadoutDrive> result;
    pulse::ExecutionWalker walker(clock(), chosen);
    walker.readoutTones = true;
    walker.onReadout = [&](const pulse::Occurrence &occurrence,
                           const pulse::MeasurementKey &key) {
      int64_t index =
          std::distance(readouts.begin(), llvm::find(readouts, occurrence.op));
      const pulse::Affine &frequency = occurrence.tone.frequencyHz;
      double hz = frequency.isConstant()
                      ? frequency.evaluate({})
                      : std::numeric_limits<double>::quiet_NaN();
      result.emplace_back(index, key.iteration, hz,
                          peakAmplitude(occurrence.op->getOperand(1)));
      return false;
    };
    auto function = getFuncOp();
    if (!function)
      throw std::runtime_error("no 'main' func in module");
    if (!walker.run(function))
      throw std::runtime_error(walker.error);
    return result;
  }

  /// The peak amplitude of a built-in waveform (its operand 1), through
  /// `pulse.scale` by a real number and `pulse.neg`.
  static double peakAmplitude(mlir::Value waveform) {
    double factor = 1.0;
    for (mlir::Operation *op = waveform.getDefiningOp(); op;) {
      llvm::StringRef name = op->getName().getStringRef();
      if (name == "pulse.neg") {
        op = op->getOperand(0).getDefiningOp();
        continue;
      }
      if (name == "pulse.scale") {
        auto scale = pulse::constantNumber(op->getOperand(1));
        if (!scale)
          break;
        factor *= *scale;
        op = op->getOperand(0).getDefiningOp();
        continue;
      }
      if (op->getNumOperands() < 2)
        break;
      if (auto amplitude = pulse::constantNumber(op->getOperand(1)))
        return std::abs(*amplitude * factor);
      break;
    }
    return std::numeric_limits<double>::quiet_NaN();
  }

  using Factor = std::tuple<int64_t, int64_t, std::vector<double>>;
  using Term = std::tuple<double, double, int32_t, std::vector<double>,
                          std::vector<Factor>>;
  using OperatorProgram =
      std::tuple<std::vector<int64_t>, std::vector<Term>, std::vector<Term>,
                 double, double, int64_t, std::string>;

  /// The Hamiltonian and collapse operators pulse-to-qop builds, as data:
  /// (mode extents, Hamiltonian terms, collapse terms, t_start, t_end,
  /// num_steps, integrator). A term is (coefficient real, imaginary,
  /// callback kind or 0, callback parameters, factors), and a factor is
  /// (mode, dimension, column-major interleaved complex matrix). Callback
  /// parameters that depend on the program's float arguments are evaluated
  /// at `arguments`. The lowering is kept per set of module attributes, so
  /// evaluating it again at other arguments does not lower again.
  OperatorProgram operator_program(const std::vector<double> &arguments) {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    const void *key =
        module_->getOperation()->getAttrDictionary().getAsOpaquePointer();
    auto found = programs_.find(key);
    if (found == programs_.end()) {
      if (programs_.size() >= kMaxPrograms)
        programs_.clear();
      found = programs_.emplace(key, lowerOperatorProgram()).first;
    }
    const ProgramTemplate &program = found->second;
    if (program.arguments > arguments.size())
      throw std::invalid_argument("operator_program: the drives depend on " +
                                  std::to_string(program.arguments) +
                                  " float arguments, got " +
                                  std::to_string(arguments.size()));
    auto bind = [&](const std::vector<TermTemplate> &terms) {
      std::vector<Term> bound;
      bound.reserve(terms.size());
      for (const TermTemplate &term : terms) {
        bound.push_back(term.term);
        auto &parameters = std::get<3>(bound.back());
        for (const SymbolicParameter &symbolic : term.symbolic)
          parameters[symbolic.slot] +=
              symbolic.coefficient * arguments[symbolic.argument];
      }
      return bound;
    };
    return {program.extents,        bind(program.hamiltonian),
            bind(program.collapse), program.tStart,
            program.tEnd,           program.steps,
            program.integrator};
  }

  /// Merge a dictionary attribute, written in MLIR syntax, into the module
  /// attributes.
  void set_attributes(const std::string &dictionary) {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    auto attr = mlir::parseAttribute(dictionary, ctx_.get());
    auto entries = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(attr);
    if (!entries)
      throw std::invalid_argument("expected an MLIR dictionary attribute");
    for (mlir::NamedAttribute entry : entries)
      module_->getOperation()->setAttr(entry.getName(), entry.getValue());
  }

  std::vector<std::string> param_names() {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    std::vector<std::string> names;
    if (auto attr = module_->getOperation()->getAttrOfType<mlir::ArrayAttr>(
            "pulse.param_names")) {
      for (auto a : attr)
        names.push_back(mlir::cast<mlir::StringAttr>(a).getValue().str());
    }
    return names;
  }

  PulseModule specialize(const std::vector<double> &f64_args,
                         const std::vector<int64_t> &i64_args,
                         const std::string &schedule = "alap",
                         int64_t maxDrives = 4, int64_t maxReadouts = 2,
                         int64_t readoutLatency = 0,
                         int64_t switchPenalty = 0) {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    auto clonedModule = source().clone();
    auto funcOp = clonedModule.lookupSymbol<mlir::func::FuncOp>("main");
    if (!funcOp)
      throw std::runtime_error("specialize: no 'main' func in module");

    auto &entryBlock = funcOp.getBody().front();
    auto loc = funcOp.getLoc();
    mlir::OpBuilder builder(ctx_.get());
    builder.setInsertionPointToStart(&entryBlock);

    size_t f64Idx = 0, i64Idx = 0;
    auto f64Ty = builder.getF64Type();
    llvm::SmallVector<mlir::Value> bound;

    for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
      auto arg = entryBlock.getArgument(i);
      mlir::Value replacement;
      if (arg.getType().isInteger(64)) {
        if (i64Idx >= i64_args.size())
          throw std::runtime_error("specialize: not enough i64 arguments");
        replacement = mlir::arith::ConstantIntOp::create(
            builder, loc, i64_args[i64Idx++], 64);
      } else {
        if (f64Idx >= f64_args.size())
          throw std::runtime_error("specialize: not enough f64 arguments");
        replacement = mlir::arith::ConstantFloatOp::create(
            builder, loc, f64Ty, llvm::APFloat(f64_args[f64Idx++]));
        bound.push_back(replacement);
      }
      arg.replaceAllUsesWith(replacement);
    }

    // Remove block arguments (make func take no args)
    while (entryBlock.getNumArguments() > 0)
      entryBlock.eraseArgument(entryBlock.getNumArguments() - 1);
    funcOp.setType(builder.getFunctionType({}, {}));

    // Fold parameter expressions before deriving concrete durations.
    foldBinding(clonedModule, bound);
    auto owning = mlir::OwningOpRef<mlir::ModuleOp>(clonedModule);
    PulseModule result(ctx_, std::move(owning));
    result.keepSource();
    mlir::PassManager pm(ctx_.get());
    addSchedule(pm, *result.module_, true, schedule, maxDrives, maxReadouts,
                readoutLatency, switchPenalty);
    runPipeline(pm, *result.module_, "specialize: pass pipeline failed");
    return result;
  }

private:
  struct SymbolicParameter {
    size_t slot;
    size_t argument;
    double coefficient;
  };
  struct TermTemplate {
    Term term;
    std::vector<SymbolicParameter> symbolic;
  };
  struct ProgramTemplate {
    std::vector<int64_t> extents;
    std::vector<TermTemplate> hamiltonian, collapse;
    double tStart, tEnd;
    int64_t steps;
    std::string integrator;
    size_t arguments = 0;
  };
  static constexpr size_t kMaxPrograms = 64;

  ProgramTemplate lowerOperatorProgram() {
    auto lowered = mlir::OwningOpRef<mlir::ModuleOp>(module_->clone());
    mlir::PassManager pm(ctx_.get());
    pm.addPass(pulse::createPulseToQOpPass());
    runPipeline(pm, *lowered, "pulse-to-qop failed");

    size_t arguments = 0;
    auto constantI64 = [](mlir::Value value) -> int64_t {
      mlir::Attribute attr;
      if (mlir::matchPattern(value, mlir::m_Constant(&attr)))
        if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(attr))
          return integer.getInt();
      throw std::runtime_error("qop operator target is not a constant");
    };
    auto spinData = [](llvm::StringRef kind) -> std::vector<double> {
      if (kind == "spin_x")
        return {0, 0, 1, 0, 1, 0, 0, 0};
      if (kind == "spin_y")
        return {0, 0, 0, 1, 0, -1, 0, 0};
      if (kind == "spin_z")
        return {1, 0, 0, 0, 0, 0, -1, 0};
      if (kind == "spin_i")
        return {1, 0, 0, 0, 0, 0, 1, 0};
      if (kind == "spin_lowering")
        return {0, 0, 0, 0, 1, 0, 0, 0};
      if (kind == "spin_raising")
        return {0, 0, 1, 0, 0, 0, 0, 0};
      throw std::runtime_error("unsupported qop spin kind " + kind.str());
    };
    auto term = [&](mlir::Value productValue) -> TermTemplate {
      mlir::Operation *product = productValue.getDefiningOp();
      if (!product || product->getName().getStringRef() != "qop.make_product")
        throw std::runtime_error("qop sum term is not a qop.make_product");
      TermTemplate result{{1.0, 0.0, 0, {}, {}}, {}};
      Term &data = result.term;
      mlir::Operation *scalar = product->getOperand(0).getDefiningOp();
      if (scalar->getName().getStringRef() == "qop.const_scalar") {
        std::get<0>(data) = mlir::cast<mlir::FloatAttr>(scalar->getAttr("real"))
                                .getValueAsDouble();
        std::get<1>(data) = mlir::cast<mlir::FloatAttr>(scalar->getAttr("imag"))
                                .getValueAsDouble();
      } else {
        std::get<2>(data) =
            mlir::cast<mlir::IntegerAttr>(scalar->getAttr("cudm.callback_kind"))
                .getInt();
        auto params = mlir::cast<mlir::DenseF64ArrayAttr>(
            scalar->getAttr("cudm.callback_params"));
        std::get<3>(data).assign(params.asArrayRef().begin(),
                                 params.asArrayRef().end());
        if (auto slots = scalar->getAttrOfType<mlir::DenseI64ArrayAttr>(
                "cudm.callback_param_arguments")) {
          auto coefficients = scalar->getAttrOfType<mlir::DenseF64ArrayAttr>(
              "cudm.callback_param_coefficients");
          for (size_t i = 0; i < coefficients.size(); ++i) {
            auto argument = static_cast<size_t>(slots[2 * i + 1]);
            result.symbolic.push_back(
                {static_cast<size_t>(slots[2 * i]), argument, coefficients[i]});
            arguments = std::max(arguments, argument + 1);
          }
        }
      }
      for (mlir::Value factor : product->getOperands().drop_front()) {
        mlir::Operation *leaf = factor.getDefiningOp();
        if (leaf->getName().getStringRef() == "qop.spin") {
          std::string kind;
          llvm::raw_string_ostream os(kind);
          leaf->getAttr("kind").print(os);
          llvm::StringRef name(kind);
          name = name.substr(name.rfind(' ') + 1).rtrim('>');
          std::get<4>(data).emplace_back(constantI64(leaf->getOperand(0)), 2,
                                         spinData(name));
        } else {
          auto targets =
              mlir::cast<mlir::DenseI64ArrayAttr>(leaf->getAttr("targets"));
          auto dims =
              mlir::cast<mlir::DenseI64ArrayAttr>(leaf->getAttr("dimensions"));
          mlir::Attribute matrix;
          if (!mlir::matchPattern(leaf->getOperand(0),
                                  mlir::m_Constant(&matrix)))
            throw std::runtime_error("qop matrix leaf data is not a constant");
          auto values = mlir::cast<mlir::DenseFPElementsAttr>(matrix);
          std::get<4>(data).emplace_back(
              targets[0], dims[0],
              std::vector<double>(values.getValues<double>().begin(),
                                  values.getValues<double>().end()));
        }
      }
      return result;
    };

    mlir::Operation *lindblad = nullptr;
    lowered->walk([&](mlir::Operation *op) {
      if (op->getName().getStringRef() == "qop.lindblad")
        lindblad = op;
    });
    if (!lindblad)
      throw std::runtime_error("pulse-to-qop produced no qop.lindblad");
    auto sumTerms = [&](mlir::Value sum, std::vector<TermTemplate> &terms) {
      for (mlir::Value product : sum.getDefiningOp()->getOperands())
        terms.push_back(term(product));
    };
    ProgramTemplate program;
    sumTerms(lindblad->getOperand(0), program.hamiltonian);
    for (mlir::Value op : lindblad->getOperands().drop_front())
      sumTerms(op, program.collapse);

    mlir::Operation *top = lowered->getOperation();
    if (auto attr =
            top->getAttrOfType<mlir::DenseI64ArrayAttr>("qop.mode_extents"))
      program.extents.assign(attr.asArrayRef().begin(),
                             attr.asArrayRef().end());
    else {
      int64_t modes = 1;
      if (auto attr = top->getAttrOfType<mlir::IntegerAttr>("qop.n_qubits"))
        modes = attr.getInt();
      else if (auto attr =
                   top->getAttrOfType<mlir::IntegerAttr>("pulse.n_qubits"))
        modes = attr.getInt();
      for (const TermTemplate &t : program.hamiltonian)
        for (const Factor &f : std::get<4>(t.term))
          modes = std::max(modes, std::get<0>(f) + 1);
      program.extents.assign(static_cast<size_t>(modes), 2);
    }
    auto number = [&](llvm::StringRef name, double fallback) {
      if (auto attr = top->getAttrOfType<mlir::FloatAttr>(name))
        return attr.getValueAsDouble();
      return fallback;
    };
    program.tStart = number("qop.t_start", 0.0);
    program.tEnd = number("qop.t_end", 0.0);
    program.steps = 100;
    if (auto attr = top->getAttrOfType<mlir::IntegerAttr>("qop.num_steps"))
      program.steps = attr.getInt();
    program.integrator = "rk4";
    if (auto attr = top->getAttrOfType<mlir::StringAttr>("qop.integrator"))
      program.integrator = attr.getValue().str();
    program.arguments = arguments;
    return program;
  }

  /// Add scheduling `module` to `pm`: the schedule pass of `policy` with
  /// these resource limits, the check of its classical calls if `checkCalls`
  /// and it declares any, then the passes that run on a scheduled module.
  static void addSchedule(mlir::PassManager &pm, mlir::ModuleOp module,
                          bool checkCalls, const std::string &policy,
                          int64_t maxDrives, int64_t maxReadouts,
                          int64_t readoutLatency, int64_t switchPenalty) {
    if (policy == "asap")
      pm.addNestedPass<mlir::func::FuncOp>(
          pulse::createPulseScheduleAsapPass());
    else if (policy == "alap")
      pm.addNestedPass<mlir::func::FuncOp>(
          pulse::createPulseScheduleAlapPass());
    else if (policy == "rcp" || policy == "alap_rcp")
      pm.addNestedPass<mlir::func::FuncOp>(pulse::createPulseScheduleRcpPass(
          maxDrives, maxReadouts, readoutLatency, switchPenalty,
          policy == "alap_rcp"));
    else
      throw std::invalid_argument("unknown schedule policy: " + policy);
    if (checkCalls && classical::declaresClassicalCalls(module))
      pm.addPass(pulse::createPulseClassicalCheckPass());
    addAfterSchedulePasses(pm);
  }

  /// Passes that run on every module once it is scheduled and checked, by
  /// schedule(), for a call's values (specialize) and for simulation: a
  /// pass that needs the times of the ops, or that must see each binding,
  /// goes here. None yet.
  static void addAfterSchedulePasses(mlir::PassManager &) {}

  /// Bind `values` to the results of `wait`: the wait stays, so what it
  /// feeds keeps waiting for it, and its results read as the values.
  void bindConstants(classical::WaitOp wait,
                     const std::vector<double> &values) {
    std::vector<double> rounded(values);
    for (auto [value, type] : llvm::zip(rounded, wait.getValues().getTypes()))
      if (!mlir::isa<mlir::FloatType>(type))
        value = std::round(value);
    wait->setAttr("classical.answers",
                  mlir::DenseF64ArrayAttr::get(ctx_.get(), rounded));
  }

  llvm::SmallVector<mlir::Operation *> readoutOps() {
    llvm::SmallVector<mlir::Operation *> readouts;
    module_->walk([&](mlir::Operation *op) {
      if (mlir::isa<pulse::ReadoutOp>(op) ||
          op->getName().getStringRef() == "pulse.iq_acquire")
        readouts.push_back(op);
    });
    return readouts;
  }

  static pulse::Path toPath(const std::vector<std::vector<int64_t>> &entries,
                            llvm::ArrayRef<mlir::Operation *> readouts) {
    pulse::Path path;
    for (const auto &entry : entries) {
      if (entry.size() < 2 || entry.front() < 0 ||
          entry.front() >= static_cast<int64_t>(readouts.size()))
        throw std::invalid_argument("path entries are [readout index, "
                                    "iteration..., outcome]");
      path[{readouts[entry.front()],
            std::vector<int64_t>(entry.begin() + 1, entry.end() - 1)}] =
          entry.back();
    }
    return path;
  }

  double clock() {
    if (auto attr = module_->getOperation()->getAttrOfType<mlir::FloatAttr>(
            "pulse.clock_ghz"))
      return attr.getValueAsDouble();
    return 1.0;
  }

  mlir::func::FuncOp getFuncOp() {
    return module_->lookupSymbol<mlir::func::FuncOp>("main");
  }

  /// Keep a copy of the module before it is scheduled if it has schedule
  /// blocks: scheduling inlines them, and a module scheduled again (with
  /// values or answers bound) must place each block whole again.
  void keepSource() {
    bool blocks = false;
    module_->walk([&](pulse::ScheduleBlockOp) {
      blocks = true;
      return mlir::WalkResult::interrupt();
    });
    if (blocks)
      source_ = mlir::OwningOpRef<mlir::ModuleOp>(module_->clone());
  }

  /// What a module scheduled again starts from: the module before it was
  /// scheduled, if it had blocks, else the module.
  mlir::ModuleOp source() { return source_ ? *source_ : *module_; }

  std::shared_ptr<mlir::MLIRContext> ctx_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  // The module before scheduling, while it has schedule blocks (see
  // keepSource).
  mlir::OwningOpRef<mlir::ModuleOp> source_;
  /// Fold `module`, whose float arguments were replaced by the constants
  /// `values`: as the canonicalizer does, then with the pattern passes this
  /// module ran, on the only ops the values can open to them, frame changes
  /// by a constant phase and drives of a square pulse, if the values reach
  /// one. Both pattern sets are gathered once per module.
  void foldBinding(mlir::ModuleOp module, llvm::ArrayRef<mlir::Value> values) {
    bool phases = false, amplitudes = false;
    if (!rewrites_.empty()) {
      llvm::SetVector<mlir::Operation *> reached;
      for (mlir::Value value : values)
        mlir::getForwardSlice(value, &reached);
      for (mlir::Operation *op : reached) {
        phases |= mlir::isa<pulse::ShiftPhaseOp>(op);
        amplitudes |= mlir::isa<pulse::SquarePulseOp>(op);
      }
    }

    if (!canonicalPatterns_) {
      mlir::RewritePatternSet set(ctx_.get());
      for (mlir::Dialect *dialect : ctx_->getLoadedDialects())
        dialect->getCanonicalizationPatterns(set);
      for (mlir::RegisteredOperationName op : ctx_->getRegisteredOperations())
        op.getCanonicalizationPatterns(set, ctx_.get());
      canonicalPatterns_ =
          std::make_shared<const mlir::FrozenRewritePatternSet>(std::move(set));
    }
    // The canonicalizer pass's defaults.
    mlir::GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(
        mlir::GreedySimplifyRegionLevel::Normal);
    (void)mlir::applyPatternsGreedily(module, *canonicalPatterns_, config);
    if (!phases && !amplitudes)
      return;

    llvm::SmallVector<mlir::Operation *> ops;
    module.walk([&](mlir::Operation *op) {
      if (auto shift = mlir::dyn_cast<pulse::ShiftPhaseOp>(op); shift && phases)
        if (mlir::Operation *phase = shift.getPhaseRad().getDefiningOp();
            phase && mlir::isa<mlir::arith::ConstantFloatOp>(phase)) {
          ops.push_back(phase);
          ops.push_back(op);
        }
      if (auto drive = mlir::dyn_cast<pulse::DriveOp>(op); drive && amplitudes)
        if (drive.getPulse().getDefiningOp<pulse::SquarePulseOp>())
          ops.push_back(op);
    });
    if (ops.empty())
      return;
    if (!rewritePatterns_) {
      mlir::RewritePatternSet set(ctx_.get());
      for (const pulse::RewriteStage *stage : rewrites_)
        stage->populate(set);
      rewritePatterns_ =
          std::make_shared<const mlir::FrozenRewritePatternSet>(std::move(set));
    }
    mlir::GreedyRewriteConfig opConfig;
    opConfig.enableFolding(false).enableConstantCSE(false);
    opConfig.setStrictness(mlir::GreedyRewriteStrictness::ExistingOps);
    (void)mlir::applyOpPatternsGreedily(ops, *rewritePatterns_, opConfig);
  }

  // The pattern passes run_passes ran, in the order first run.
  llvm::SmallVector<const pulse::RewriteStage *, 4> rewrites_;
  // What foldBinding rewrites with, once gathered.
  std::shared_ptr<const mlir::FrozenRewritePatternSet> canonicalPatterns_;
  std::shared_ptr<const mlir::FrozenRewritePatternSet> rewritePatterns_;
  // Lowered operator programs by module attribute dictionary.
  std::map<const void *, ProgramTemplate> programs_;
  // Python threads may call into one module concurrently once the GIL is
  // released; MLIR does not allow concurrent mutation of one module.
  std::shared_ptr<std::recursive_mutex> lock_;
};

// -----------------------------------------------------------------------
// PulseModuleBuilder: decode a packed int64 buffer into in-memory Pulse IR
// -----------------------------------------------------------------------
/// Loading the dialects into a new context costs about 0.4 ms, so builders
/// share a context. A context keeps every attribute it ever uniqued (each
/// compile with new qubit frequencies, for instance, adds a few hundred bytes
/// per qubit), so it is retired after a number of builders or once the
/// programs built in it reach a number of packed words, whichever comes
/// first, and freed with its last module. Both limits can be set with
/// environment variables, read whenever a builder takes a context.
class SharedContextPool {
public:
  struct Budget {
    int64_t modules;
    int64_t words;
  };

  static constexpr const char *kModulesVariable =
      "CUDAQ_PULSE_CONTEXT_MAX_MODULES";
  static constexpr const char *kWordsVariable = "CUDAQ_PULSE_CONTEXT_MAX_WORDS";

  /// The limits in effect; throws std::invalid_argument for a malformed
  /// environment variable.
  static Budget budget() {
    return {limit(kModulesVariable, 256), limit(kWordsVariable, 1 << 20)};
  }

  static std::shared_ptr<mlir::MLIRContext> acquire() {
    const Budget limits = budget();
    std::lock_guard<std::mutex> guard(lock());
    auto &pool = state();
    if (!pool.current || ++pool.uses > limits.modules ||
        pool.words >= limits.words) {
      pool.current = std::make_shared<mlir::MLIRContext>(makeRegistry());
      pool.current->loadAllAvailableDialects();
      pool.uses = 1;
      pool.words = 0;
      ++pool.created;
    }
    return pool.current;
  }

  /// Count a program of `words` packed words against `context`'s budget.
  static void charge(const mlir::MLIRContext *context, int64_t words) {
    std::lock_guard<std::mutex> guard(lock());
    auto &pool = state();
    if (pool.current.get() == context)
      pool.words += words;
  }

  /// (contexts created, builders on the current one, its packed words).
  static std::tuple<int64_t, int64_t, int64_t> stats() {
    std::lock_guard<std::mutex> guard(lock());
    auto &pool = state();
    return {pool.created, pool.current ? pool.uses : 0, pool.words};
  }

  /// A forked child has only the forking thread. The pool's lock is held
  /// across fork so the child inherits it unlocked, and the child starts a
  /// new context: the current one's thread pool has no workers there. That
  /// context is abandoned rather than destroyed, since destroying it would
  /// wait for those workers.
  static void handleForks() {
    pthread_atfork([] { lock().lock(); }, [] { lock().unlock(); },
                   [] {
                     auto &pool = state();
                     if (pool.current)
                       new std::shared_ptr<mlir::MLIRContext>(
                           std::move(pool.current));
                     pool.uses = 0;
                     pool.words = 0;
                     lock().unlock();
                   });
  }

private:
  struct State {
    std::shared_ptr<mlir::MLIRContext> current;
    int64_t uses = 0;
    int64_t words = 0;
    int64_t created = 0;
  };

  static int64_t limit(const char *variable, int64_t fallback) {
    constexpr int64_t kMaximum = int64_t{1} << 40;
    const char *text = std::getenv(variable);
    if (!text || !*text)
      return fallback;
    int64_t value = 0;
    const char *end = text + std::strlen(text);
    auto [stop, error] = std::from_chars(text, end, value);
    if (error != std::errc() || stop != end || value < 1 || value > kMaximum)
      throw std::invalid_argument(
          std::string(variable) + "=" + text +
          " is not valid; it must be a whole number from 1 to " +
          std::to_string(kMaximum) + " (default " + std::to_string(fallback) +
          ")");
    return value;
  }

  static std::mutex &lock() {
    static std::mutex mutex;
    return mutex;
  }
  // Never destroyed: the last context must not be torn down during static
  // destruction, after LLVM's own statics.
  static State &state() {
    static auto *pool = new State();
    return *pool;
  }
};

bool isTimedPulseType(mlir::Type type) {
  return mlir::isa<pulse::DriveLineType, pulse::ReadoutLineType,
                   pulse::ToneType, pulse::MeasurementType>(type) ||
         pulse::isClassicalType(type);
}

/// Whether `value` is defined in `body` or in a region nested in it.
bool definedIn(mlir::Value value, mlir::Block *body) {
  for (mlir::Block *block = value.getParentBlock(); block;) {
    if (block == body)
      return true;
    mlir::Operation *parent = block->getParentOp();
    if (!parent)
      return false;
    block = parent->getBlock();
  }
  return false;
}

/// Wrap the ops traced into `body` in a `pulse.schedule_block` at the
/// builder's insertion point. The lines, tones and measurements the body
/// uses from outside become its operands; the latest version of every line
/// and tone it holds, and every measurement it takes, become its results.
pulse::ScheduleBlockOp buildScheduleBlock(mlir::OpBuilder &builder,
                                          mlir::Location loc,
                                          std::unique_ptr<mlir::Block> body,
                                          llvm::StringRef policy,
                                          llvm::StringRef callee) {
  // Untimed values (waveforms, constants) may be used after the block, so
  // they are defined before it.
  mlir::Block *parent = builder.getInsertionBlock();
  for (mlir::Operation &op : llvm::make_early_inc_range(*body)) {
    bool untimed = op.getNumRegions() == 0 &&
                   llvm::none_of(op.getOperandTypes(), isTimedPulseType) &&
                   llvm::none_of(op.getResultTypes(), isTimedPulseType) &&
                   llvm::none_of(op.getOperands(), [&](mlir::Value operand) {
                     return operand.getParentBlock() == body.get();
                   });
    if (untimed)
      op.moveBefore(parent, parent->end());
  }

  llvm::SmallVector<mlir::Value> inputs;
  llvm::DenseSet<mlir::Value> seen;
  body->walk([&](mlir::Operation *op) {
    for (mlir::Value operand : op->getOperands())
      if (isTimedPulseType(operand.getType()) &&
          !definedIn(operand, body.get()) && seen.insert(operand).second)
        inputs.push_back(operand);
  });
  llvm::SmallVector<mlir::Value> outputs;
  for (mlir::Operation &op : *body)
    for (mlir::Value result : op.getResults())
      if (mlir::isa<pulse::MeasurementType>(result.getType()) ||
          (pulse::isLinearPulseType(result.getType()) && result.use_empty()))
        outputs.push_back(result);

  llvm::SmallVector<mlir::Type> types;
  for (mlir::Value output : outputs)
    types.push_back(output.getType());
  auto block = pulse::ScheduleBlockOp::create(
      builder, loc, types, builder.getStringAttr(policy), inputs,
      callee.empty() ? mlir::StringAttr() : builder.getStringAttr(callee));
  block.getBody().push_back(body.release());
  mlir::Block *inner = &block.getBody().front();
  for (mlir::Value input : inputs) {
    mlir::BlockArgument arg = inner->addArgument(input.getType(), loc);
    input.replaceUsesWithIf(arg, [&](mlir::OpOperand &use) {
      return block->isProperAncestor(use.getOwner());
    });
  }
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(inner);
  pulse::YieldOp::create(builder, loc, outputs);
  return block;
}

class PulseModuleBuilder {
public:
  PulseModuleBuilder() : ctx_(SharedContextPool::acquire()) {
    builder_ = std::make_unique<mlir::OpBuilder>(ctx_.get());
  }

  // ─── Packed-buffer decoder (zero-copy path) ──────────────────────────
  PulseModule
  build_from_packed(nb::ndarray<int64_t, nb::ndim<1>, nb::c_contig> stream,
                    double clock_ghz, int64_t n_qubits,
                    nb::ndarray<double, nb::ndim<1>, nb::c_contig> qubit_freqs,
                    std::vector<std::string> param_names = {},
                    std::vector<std::string> param_types = {},
                    const std::string &name = "main") {
    std::lock_guard<std::mutex> guard(lock_);

    const int64_t *d = stream.data();
    const size_t len = stream.shape(0);
    SharedContextPool::charge(ctx_.get(), static_cast<int64_t>(len));
    auto loc = builder_->getUnknownLoc();

    if (clock_ghz <= 0.0)
      throw std::invalid_argument("packed: clock_ghz must be positive");
    if (n_qubits < 0)
      throw std::invalid_argument("packed: n_qubits cannot be negative");
    if (qubit_freqs.shape(0) != static_cast<size_t>(n_qubits))
      throw std::invalid_argument(
          "packed: qubit frequency count does not match n_qubits");
    if (param_names.size() != param_types.size())
      throw std::invalid_argument(
          "packed: parameter names and types have different lengths");

    auto module = mlir::ModuleOp::create(loc, mlir::StringRef(name));
    module->setAttr("pulse.n_qubits", builder_->getI64IntegerAttr(n_qubits));
    module->setAttr("pulse.clock_ghz", builder_->getF64FloatAttr(clock_ghz));

    // Encode qubit frequencies as module attribute
    const double *fdata = qubit_freqs.data();
    size_t flen = qubit_freqs.shape(0);
    llvm::SmallVector<double> freqVec(fdata, fdata + flen);
    module->setAttr("pulse.qubit_freq_hz",
                    builder_->getDenseF64ArrayAttr(freqVec));

    // Build func.func type: parameters become block arguments
    llvm::SmallVector<mlir::Type> argTypes;
    auto f64Ty = builder_->getF64Type();
    auto i64Ty = builder_->getIntegerType(64);
    for (auto &pt : param_types) {
      if (pt == "i64")
        argTypes.push_back(i64Ty);
      else if (pt == "f64")
        argTypes.push_back(f64Ty);
      else
        throw std::invalid_argument("packed: unsupported parameter type '" +
                                    pt + "'");
    }

    auto funcType = builder_->getFunctionType(argTypes, {});
    auto funcOp = mlir::func::FuncOp::create(loc, "main", funcType);
    module.push_back(funcOp);
    auto *entryBlock = funcOp.addEntryBlock();
    builder_->setInsertionPointToStart(entryBlock);

    // Store parameter name metadata on the module for later use
    if (!param_names.empty()) {
      llvm::SmallVector<mlir::Attribute> nameAttrs;
      for (auto &n : param_names)
        nameAttrs.push_back(builder_->getStringAttr(n));
      module->setAttr("pulse.param_names", builder_->getArrayAttr(nameAttrs));
    }

    auto qrefTy = pulse::QrefType::get(ctx_.get());
    auto driveTy = pulse::DriveLineType::get(ctx_.get());
    auto readoutTy = pulse::ReadoutLineType::get(ctx_.get());
    auto toneTy = pulse::ToneType::get(ctx_.get());
    auto wfTy = pulse::WaveformType::get(ctx_.get());
    auto durTy = pulse::DurationType::get(ctx_.get());
    auto measTy = pulse::MeasurementType::get(ctx_.get());

    llvm::DenseMap<int64_t, mlir::Value> ssa;
    llvm::DenseMap<int64_t, mlir::Value> qubitSSA;
    // The open schedule blocks as (policy, callee), innermost last, and the
    // ids of the values bound inside them, so that a block's results can
    // take them over when it closes.
    std::vector<std::pair<std::string, std::string>> blocks;
    llvm::DenseMap<mlir::Value, llvm::SmallVector<int64_t, 1>> blockVids;

    auto bind = [&](int64_t vid, mlir::Value v) {
      ssa[vid] = v;
      if (!blocks.empty())
        blockVids[v].push_back(vid);
    };
    auto look = [&](int64_t vid) -> mlir::Value {
      auto it = ssa.find(vid);
      if (it == ssa.end())
        throw std::runtime_error("packed: SSA value not found for vid=" +
                                 std::to_string(vid));
      return it->second;
    };
    // Region bodies are built in detached blocks and moved into their scf op
    // when the region closes; values they use from outside stay in scope.
    struct Region {
      bool isLoop;
      int64_t lb = 0, ub = 0, step = 1;
      mlir::Value condition;
      mlir::Block *parent;
      std::unique_ptr<mlir::Block> body;
      std::unique_ptr<mlir::Block> elseBody;
      // The links of classical calls as the region began, and as its then
      // arm ended: a link a region sends on is carried out of it.
      std::map<int64_t, mlir::Value> linksBefore = {}, linksThen = {};
    };
    std::vector<Region> regions;
    // A schedule block is a region with neither loop bounds nor a condition.
    auto isBlock = [](const Region &region) {
      return !region.isLoop && !region.condition;
    };

    auto getQubit = [&](int64_t q) -> mlir::Value {
      if (q < 0 || q >= n_qubits)
        throw std::invalid_argument("packed: qubit index out of range: " +
                                    std::to_string(q));
      auto it = qubitSSA.find(q);
      if (it != qubitSSA.end())
        return it->second;
      // Allocate outside any region, so every later use is dominated.
      mlir::OpBuilder::InsertionGuard guard(*builder_);
      if (!regions.empty())
        builder_->setInsertionPointToEnd(regions.front().parent);
      auto alloc = pulse::QuditAllocOp::create(*builder_, loc, qrefTy);
      alloc->setAttr("qubit", builder_->getI64IntegerAttr(q));
      auto v = alloc.getResult();
      qubitSSA[q] = v;
      return v;
    };

    // Classical calls: the declared functions by index, and the current
    // version of each link, created at function entry when first used.
    std::vector<classical::CallableOp> callables;
    std::map<int64_t, mlir::Value> links;
    auto linkOf = [&](int64_t device) -> mlir::Value {
      auto it = links.find(device);
      if (it != links.end())
        return it->second;
      mlir::OpBuilder::InsertionGuard guard(*builder_);
      if (!regions.empty())
        builder_->setInsertionPointToEnd(regions.front().parent);
      auto get = classical::GetLinkOp::create(
          *builder_, loc, classical::LinkType::get(ctx_.get()),
          builder_->getI64IntegerAttr(device));
      for (Region &open : regions)
        open.linksBefore.try_emplace(device, get.getLink());
      links[device] = get.getLink();
      return get.getLink();
    };
    auto callableAt = [&](int64_t index) -> classical::CallableOp {
      if (index < 0 || static_cast<size_t>(index) >= callables.size())
        throw std::invalid_argument("packed: unknown classical function " +
                                    std::to_string(index));
      return callables[static_cast<size_t>(index)];
    };
    // The links one arm or body changed, as (device, before, after).
    auto changedLinks = [](const std::map<int64_t, mlir::Value> &before,
                           const std::map<int64_t, mlir::Value> &a,
                           const std::map<int64_t, mlir::Value> &b) {
      llvm::SmallVector<std::tuple<int64_t, mlir::Value, mlir::Value>> out;
      for (auto &[device, value] : before) {
        auto ia = a.find(device), ib = b.find(device);
        mlir::Value va = ia == a.end() ? value : ia->second;
        mlir::Value vb = ib == b.end() ? value : ib->second;
        if (va != value || vb != value)
          out.push_back({device, va, vb});
      }
      return out;
    };

    size_t cur = 0;
    while (cur < len) {
      int64_t hdr = d[cur];
      int opcode = static_cast<int>(hdr & 0xFF);
      int64_t plen = (hdr >> 8) & 0xFF;
      int pmask = static_cast<int>((hdr >> 16) & 0xFFFF);
      size_t headerWords = 1;
      if (plen == kExtendedPayload) {
        if (cur + 1 >= len || d[cur + 1] < 0)
          throw std::invalid_argument("packed: truncated record at word " +
                                      std::to_string(cur));
        plen = d[cur + 1];
        headerWords = 2;
      }
      if (cur + headerWords + static_cast<size_t>(plen) > len)
        throw std::invalid_argument("packed: truncated record at word " +
                                    std::to_string(cur));
      const int64_t *p = d + cur + headerWords;

      auto requirePayload = [&](int64_t expected) {
        if (plen != expected)
          throw std::invalid_argument(
              "packed: opcode " + std::to_string(opcode) + " expects " +
              std::to_string(expected) + " payload words, got " +
              std::to_string(plen));
      };

      // Helper: get SSA value for a waveform arg slot. If the param_mask
      // bit is set, the slot holds a vid (block arg ref); otherwise it's
      // a literal that needs an arith.constant.
      auto getI64Arg = [&](int slot) -> mlir::Value {
        if (pmask & (1 << slot))
          return look(p[slot]);
        return mlir::arith::ConstantIntOp::create(*builder_, loc, p[slot], 64)
            .getResult();
      };
      auto getF64Arg = [&](int slot) -> mlir::Value {
        if (pmask & (1 << slot))
          return look(p[slot]);
        return mlir::arith::ConstantFloatOp::create(*builder_, loc, f64Ty,
                                                    llvm::APFloat(i2f(p[slot])))
            .getResult();
      };

      switch (opcode) {

      case kOpForBegin: {
        requirePayload(3);
        Region region{true,
                      p[0],
                      p[1],
                      p[2],
                      {},
                      builder_->getBlock(),
                      std::make_unique<mlir::Block>(),
                      nullptr};
        if (region.step <= 0)
          throw std::invalid_argument("packed: loop step must be positive");
        region.linksBefore = links;
        builder_->setInsertionPointToEnd(region.body.get());
        regions.push_back(std::move(region));
        break;
      }
      case kOpForEnd: {
        if (regions.empty() || !regions.back().isLoop || plen < 1 || p[0] < 0 ||
            plen != 1 + 3 * p[0])
          throw std::invalid_argument("packed: malformed loop end");
        Region region = std::move(regions.back());
        regions.pop_back();
        builder_->setInsertionPointToEnd(region.parent);
        const int64_t n = p[0];
        // A value carried under two names is one iteration argument.
        llvm::SmallVector<mlir::Value> inits, yields;
        llvm::SmallVector<size_t> argOf;
        llvm::DenseMap<int64_t, size_t> argOfInit;
        for (int64_t i = 0; i < n; ++i) {
          int64_t init = p[1 + i];
          mlir::Value next = look(p[1 + n + i]);
          if (init >= 0) {
            auto [it, inserted] = argOfInit.try_emplace(init, inits.size());
            if (!inserted) {
              if (yields[it->second] != next)
                throw std::invalid_argument(
                    "packed: one loop value carried to two different values");
              argOf.push_back(it->second);
              continue;
            }
          }
          argOf.push_back(inits.size());
          inits.push_back(init >= 0 ? look(init)
                                    : mlir::ub::PoisonOp::create(
                                          *builder_, loc, next.getType()));
          yields.push_back(next);
        }
        // Links the body sent on are carried by the loop.
        llvm::SmallVector<std::pair<int64_t, size_t>> carriedLinks;
        for (auto [device, after, unused] :
             changedLinks(region.linksBefore, links, links)) {
          (void)unused;
          carriedLinks.push_back({device, inits.size()});
          inits.push_back(region.linksBefore.at(device));
          yields.push_back(after);
        }
        auto index = [&](int64_t v) -> mlir::Value {
          return mlir::arith::ConstantIndexOp::create(*builder_, loc, v);
        };
        auto loop = mlir::scf::ForOp::create(*builder_, loc, index(region.lb),
                                             index(region.ub),
                                             index(region.step), inits);
        mlir::Block *body = loop.getBody();
        if (!body->empty())
          body->getTerminator()->erase();
        body->getOperations().splice(body->end(), region.body->getOperations());
        {
          mlir::OpBuilder::InsertionGuard guard(*builder_);
          builder_->setInsertionPointToEnd(body);
          mlir::scf::YieldOp::create(*builder_, loc, yields);
        }
        for (auto [init, arg] : llvm::zip(inits, loop.getRegionIterArgs()))
          init.replaceUsesWithIf(arg, [&](mlir::OpOperand &use) {
            return loop->isProperAncestor(use.getOwner());
          });
        for (int64_t i = 0; i < n; ++i)
          bind(p[1 + 2 * n + i], loop.getResult(argOf[i]));
        for (auto [device, index] : carriedLinks)
          links[device] = loop.getResult(index);
        break;
      }
      case kOpIfBegin: {
        requirePayload(1);
        Region region{false,
                      0,
                      0,
                      1,
                      look(p[0]),
                      builder_->getBlock(),
                      std::make_unique<mlir::Block>(),
                      nullptr};
        region.linksBefore = links;
        builder_->setInsertionPointToEnd(region.body.get());
        regions.push_back(std::move(region));
        break;
      }
      case kOpElse: {
        requirePayload(0);
        if (regions.empty() || regions.back().isLoop ||
            isBlock(regions.back()) || regions.back().elseBody)
          throw std::invalid_argument("packed: else outside an if");
        regions.back().elseBody = std::make_unique<mlir::Block>();
        regions.back().linksThen = links;
        links = regions.back().linksBefore;
        builder_->setInsertionPointToEnd(regions.back().elseBody.get());
        break;
      }
      case kOpIfEnd: {
        if (regions.empty() || regions.back().isLoop ||
            isBlock(regions.back()) || plen < 1 || p[0] < 0 ||
            plen != 1 + 3 * p[0])
          throw std::invalid_argument("packed: malformed if end");
        Region region = std::move(regions.back());
        regions.pop_back();
        builder_->setInsertionPointToEnd(region.parent);
        const int64_t n = p[0];
        llvm::SmallVector<mlir::Value> thenValues, elseValues;
        llvm::SmallVector<mlir::Type> types;
        for (int64_t i = 0; i < n; ++i) {
          thenValues.push_back(look(p[1 + i]));
          elseValues.push_back(look(p[1 + n + i]));
          types.push_back(thenValues.back().getType());
        }
        // A link either arm sent on leaves the branch as a result.
        const auto &thenLinks = region.elseBody ? region.linksThen : links;
        const auto &elseLinks = region.elseBody ? links : region.linksBefore;
        llvm::SmallVector<std::pair<int64_t, unsigned>> branchLinks;
        for (auto [device, a, b] :
             changedLinks(region.linksBefore, thenLinks, elseLinks)) {
          branchLinks.push_back({device, static_cast<unsigned>(types.size())});
          thenValues.push_back(a);
          elseValues.push_back(b);
          types.push_back(a.getType());
        }
        // A measurement is tested with pulse.is_set; an answer of a
        // classical call is already a boolean.
        mlir::Value condition = region.condition;
        if (mlir::isa<pulse::MeasurementType>(condition.getType()))
          condition = pulse::IsSetOp::create(*builder_, loc,
                                             builder_->getI1Type(), condition);
        auto branch = mlir::scf::IfOp::create(*builder_, loc, types, condition,
                                              /*withElseRegion=*/true);
        auto fill = [&](mlir::Block *arm, mlir::Block *source,
                        llvm::ArrayRef<mlir::Value> values) {
          if (!arm->empty())
            arm->getTerminator()->erase();
          if (source)
            arm->getOperations().splice(arm->end(), source->getOperations());
          mlir::OpBuilder::InsertionGuard guard(*builder_);
          builder_->setInsertionPointToEnd(arm);
          mlir::scf::YieldOp::create(*builder_, loc, values);
        };
        fill(branch.thenBlock(), region.body.get(), thenValues);
        fill(branch.elseBlock(), region.elseBody.get(), elseValues);
        for (int64_t i = 0; i < n; ++i)
          bind(p[1 + 2 * n + i], branch.getResult(i));
        for (auto [device, index] : branchLinks)
          links[device] = branch.getResult(index);
        break;
      }
      case kOpBlockBegin: {
        // [policy (0 = asap, 1 = alap), callee name bytes, name words...]
        if (plen < 2 || p[0] < 0 || p[0] > 1 || p[1] < 0 ||
            plen != 2 + (p[1] + 7) / 8)
          throw std::invalid_argument("packed: malformed schedule block");
        Region region{false,
                      0,
                      0,
                      1,
                      {},
                      builder_->getBlock(),
                      std::make_unique<mlir::Block>(),
                      nullptr};
        blocks.emplace_back(p[0] == 0 ? "asap" : "alap",
                            std::string(reinterpret_cast<const char *>(p + 2),
                                        static_cast<size_t>(p[1])));
        region.linksBefore = links;
        builder_->setInsertionPointToEnd(region.body.get());
        regions.push_back(std::move(region));
        break;
      }
      case kOpBlockEnd: {
        requirePayload(0);
        if (regions.empty() || !isBlock(regions.back()))
          throw std::invalid_argument(
              "packed: schedule block end without a begin");
        Region region = std::move(regions.back());
        regions.pop_back();
        auto [policy, callee] = std::move(blocks.back());
        blocks.pop_back();
        builder_->setInsertionPointToEnd(region.parent);
        auto block = buildScheduleBlock(*builder_, loc, std::move(region.body),
                                        policy, callee);
        auto yielded = block.getBody().front().getTerminator()->getOperands();
        // The last version of a link the block sent on is a block result.
        for (auto &[device, value] : links)
          for (auto [inner, result] : llvm::zip(yielded, block.getResults()))
            if (inner == value) {
              value = result;
              break;
            }
        for (auto [inner, result] : llvm::zip(yielded, block.getResults())) {
          auto it = blockVids.find(inner);
          if (it == blockVids.end())
            continue;
          for (int64_t vid : it->second)
            bind(vid, result);
        }
        if (blocks.empty())
          blockVids.clear();
        break;
      }

      case kOpAllocDrive: {
        requirePayload(3);
        auto qref = getQubit(p[0]);
        auto gdl = pulse::GetDriveLineOp::create(*builder_, loc, driveTy,
                                                 toneTy, qref);
        gdl->setAttr("qubit", builder_->getI64IntegerAttr(p[0]));
        gdl->setAttr("frequency_hz", builder_->getF64FloatAttr(freqVec[p[0]]));
        bind(p[1], gdl.getLine());
        bind(p[2], gdl.getTone());
        break;
      }

      case kOpAllocReadout: {
        requirePayload(3);
        auto qref = getQubit(p[0]);
        auto grl = pulse::GetReadoutLineOp::create(*builder_, loc, readoutTy,
                                                   toneTy, qref);
        grl->setAttr("qubit", builder_->getI64IntegerAttr(p[0]));
        grl->setAttr("frequency_hz", builder_->getF64FloatAttr(freqVec[p[0]]));
        bind(p[1], grl.getLine());
        bind(p[2], grl.getTone());
        break;
      }

      case kOpAllocTone: {
        requirePayload(3);
        double freq = i2f(p[1]);
        double phase = i2f(p[2]);
        auto fc = mlir::arith::ConstantFloatOp::create(
            *builder_, loc, builder_->getF64Type(), llvm::APFloat(freq));
        auto pc = mlir::arith::ConstantFloatOp::create(
            *builder_, loc, builder_->getF64Type(), llvm::APFloat(phase));
        auto t = pulse::ToneOp::create(*builder_, loc, toneTy, fc.getResult(),
                                       pc.getResult());
        bind(p[0], t.getResult());
        break;
      }

      case kOpWfGaussian: {
        requirePayload(4);
        auto op = pulse::GaussianPulseOp::create(
            *builder_, loc, wfTy, getI64Arg(1), getF64Arg(2), getF64Arg(3));
        bind(p[0], op.getResult());
        break;
      }

      case kOpWfSquare: {
        requirePayload(4);
        auto op = pulse::SquarePulseOp::create(
            *builder_, loc, wfTy, getI64Arg(1), getF64Arg(2), getF64Arg(3));
        bind(p[0], op.getResult());
        break;
      }

      case kOpWfDrag: {
        requirePayload(5);
        auto op = pulse::DRAGPulseOp::create(*builder_, loc, wfTy, getI64Arg(1),
                                             getF64Arg(2), getF64Arg(3),
                                             getF64Arg(4));
        bind(p[0], op.getResult());
        break;
      }

      case kOpWfCosine: {
        requirePayload(3);
        auto op = pulse::CosinePulseOp::create(*builder_, loc, wfTy,
                                               getI64Arg(1), getF64Arg(2));
        bind(p[0], op.getResult());
        break;
      }

      case kOpWfTanhRamp: {
        requirePayload(4);
        auto op = pulse::TanhRampOp::create(*builder_, loc, wfTy, getI64Arg(1),
                                            getF64Arg(2), getF64Arg(3));
        bind(p[0], op.getResult());
        break;
      }

      case kOpWfGaussSquare: {
        requirePayload(5);
        auto op = pulse::GaussianSquarePulseOp::create(
            *builder_, loc, wfTy, getI64Arg(1), getF64Arg(2), getF64Arg(3),
            getI64Arg(4));
        bind(p[0], op.getResult());
        break;
      }

      case kOpWfCustom: {
        if (plen < 4)
          throw std::invalid_argument(
              "packed: custom waveform record is too short");
        int64_t byteCount = p[2];
        if (byteCount <= 0)
          throw std::invalid_argument(
              "packed: custom waveform callback name is empty");
        size_t wordCount = (static_cast<size_t>(byteCount) + 7) / 8;
        if (3 + wordCount != static_cast<size_t>(plen))
          throw std::invalid_argument(
              "packed: custom waveform callback payload is malformed");
        std::string callbackName(reinterpret_cast<const char *>(p + 3),
                                 static_cast<size_t>(byteCount));
        auto callee = mlir::FlatSymbolRefAttr::get(ctx_.get(), callbackName);
        auto op =
            pulse::CustomOp::create(*builder_, loc, wfTy, callee, getI64Arg(1));
        bind(p[0], op.getResult());
        break;
      }

      case kOpWfCustomSamples: {
        if (plen < 2 || p[1] < 0 || p[1] + 2 != plen)
          throw std::invalid_argument(
              "packed: custom sample payload is malformed");
        llvm::SmallVector<double> samples;
        samples.reserve(static_cast<size_t>(p[1]));
        for (int64_t i = 0; i < p[1]; ++i)
          samples.push_back(i2f(p[2 + i]));
        auto op = pulse::CustomSamplesOp::create(
            *builder_, loc, wfTy,
            builder_->getF64ArrayAttr(llvm::ArrayRef<double>(samples)));
        bind(p[0], op.getResult());
        break;
      }

      case kOpParam: {
        requirePayload(2);
        // Parameter reference: p[0] = vid, p[1] = param_index
        // The value comes from the func.func block argument
        int64_t paramIdx = p[1];
        if (paramIdx < 0 ||
            paramIdx >= static_cast<int64_t>(entryBlock->getNumArguments()))
          throw std::invalid_argument("packed: parameter index out of range: " +
                                      std::to_string(paramIdx));
        bind(p[0], entryBlock->getArgument(paramIdx));
        break;
      }

      case kOpNumConst: {
        requirePayload(3);
        if (p[1] == 0) {
          bind(p[0],
               mlir::arith::ConstantIntOp::create(*builder_, loc, p[2], 64));
        } else if (p[1] == 1) {
          bind(p[0], mlir::arith::ConstantFloatOp::create(
                         *builder_, loc, f64Ty, llvm::APFloat(i2f(p[2]))));
        } else {
          throw std::invalid_argument("packed: invalid numeric constant type");
        }
        break;
      }

      case kOpNumBinary: {
        requirePayload(5);
        auto lhs = look(p[3]);
        auto rhs = look(p[4]);
        mlir::Value result;
        if (p[1] == 0) {
          switch (p[2]) {
          case 0:
            result = mlir::arith::AddIOp::create(*builder_, loc, lhs, rhs);
            break;
          case 1:
            result = mlir::arith::SubIOp::create(*builder_, loc, lhs, rhs);
            break;
          case 2:
            result = mlir::arith::MulIOp::create(*builder_, loc, lhs, rhs);
            break;
          case 3:
            result = mlir::arith::DivSIOp::create(*builder_, loc, lhs, rhs);
            break;
          case 4:
            result =
                mlir::arith::FloorDivSIOp::create(*builder_, loc, lhs, rhs);
            break;
          case 5:
            result = mlir::arith::RemSIOp::create(*builder_, loc, lhs, rhs);
            break;
          default:
            throw std::invalid_argument("packed: invalid integer binary op");
          }
        } else if (p[1] == 1) {
          switch (p[2]) {
          case 0:
            result = mlir::arith::AddFOp::create(*builder_, loc, lhs, rhs);
            break;
          case 1:
            result = mlir::arith::SubFOp::create(*builder_, loc, lhs, rhs);
            break;
          case 2:
            result = mlir::arith::MulFOp::create(*builder_, loc, lhs, rhs);
            break;
          case 3:
            result = mlir::arith::DivFOp::create(*builder_, loc, lhs, rhs);
            break;
          default:
            throw std::invalid_argument("packed: invalid float binary op");
          }
        } else {
          throw std::invalid_argument("packed: invalid binary numeric type");
        }
        bind(p[0], result);
        break;
      }

      case kOpNumNeg: {
        requirePayload(3);
        auto operand = look(p[2]);
        if (p[1] == 0) {
          auto zero = mlir::arith::ConstantIntOp::create(*builder_, loc, 0, 64);
          bind(p[0],
               mlir::arith::SubIOp::create(*builder_, loc, zero, operand));
        } else if (p[1] == 1) {
          bind(p[0], mlir::arith::NegFOp::create(*builder_, loc, operand));
        } else {
          throw std::invalid_argument("packed: invalid negation type");
        }
        break;
      }

      case kOpNumCast: {
        requirePayload(4);
        auto operand = look(p[3]);
        if (p[1] == 0 && p[2] == 1)
          bind(p[0],
               mlir::arith::SIToFPOp::create(*builder_, loc, f64Ty, operand));
        else if (p[1] == 1 && p[2] == 0)
          bind(p[0],
               mlir::arith::FPToSIOp::create(*builder_, loc, i64Ty, operand));
        else
          throw std::invalid_argument("packed: invalid numeric cast");
        break;
      }

      case kOpWfAdd:
      case kOpWfSub:
      case kOpWfMul: {
        requirePayload(3);
        mlir::Value result;
        if (opcode == kOpWfAdd)
          result = pulse::PulseAddOp::create(*builder_, loc, wfTy, look(p[1]),
                                             look(p[2]));
        else if (opcode == kOpWfSub)
          result = pulse::PulseSubOp::create(*builder_, loc, wfTy, look(p[1]),
                                             look(p[2]));
        else
          result = pulse::PulseMulOp::create(*builder_, loc, wfTy, look(p[1]),
                                             look(p[2]));
        bind(p[0], result);
        break;
      }

      case kOpWfScale: {
        requirePayload(3);
        auto op = pulse::PulseScaleOp::create(*builder_, loc, wfTy, look(p[1]),
                                              look(p[2]));
        bind(p[0], op.getResult());
        break;
      }

      case kOpWfNeg: {
        requirePayload(2);
        auto op = pulse::PulseNegOp::create(*builder_, loc, wfTy, look(p[1]));
        bind(p[0], op.getResult());
        break;
      }

      case kOpDrive: {
        requirePayload(7);
        auto driveOp =
            pulse::DriveOp::create(*builder_, loc, driveTy, toneTy, look(p[0]),
                                   look(p[1]), look(p[2]));
        if (p[5] != -1)
          driveOp->setAttr("start_cycles", builder_->getI64IntegerAttr(p[5]));
        if (p[6] != -1)
          driveOp->setAttr("duration_cycles",
                           builder_->getI64IntegerAttr(p[6]));
        bind(p[3], driveOp.getUpdatedLine());
        bind(p[4], driveOp.getUpdatedTone());
        break;
      }

      case kOpReadout: {
        requirePayload(6);
        auto mode = builder_->getStringAttr("iq");
        auto roOp =
            pulse::ReadoutOp::create(*builder_, loc, readoutTy, toneTy, measTy,
                                     look(p[0]), look(p[1]), look(p[2]), mode);
        bind(p[3], roOp.getUpdatedLine());
        bind(p[4], roOp.getUpdatedTone());
        bind(p[5], roOp.getResult());
        break;
      }

      case kOpSync: {
        if (plen < 1 || p[0] < 0 || 1 + 3 * p[0] != plen)
          throw std::invalid_argument("packed: malformed sync payload");
        int64_t n = p[0];
        llvm::SmallVector<mlir::Value> inVals;
        llvm::SmallVector<mlir::Type> outTypes;
        for (int64_t j = 0; j < n; ++j) {
          inVals.push_back(look(p[1 + 3 * j]));
          int64_t vtype = p[3 + 3 * j];
          outTypes.push_back(vtype == 1 ? mlir::Type(readoutTy)
                                        : mlir::Type(driveTy));
        }
        auto syncOp = pulse::SyncOp::create(*builder_, loc, outTypes, inVals);
        for (int64_t j = 0; j < n; ++j)
          bind(p[2 + 3 * j], syncOp.getResult(j));
        break;
      }

      case kOpWait: {
        requirePayload(3);
        mlir::Value lineVal = look(p[0]);
        mlir::Value durCycles;
        if (pmask & (1 << 2))
          durCycles = look(p[2]);
        else
          durCycles =
              mlir::arith::ConstantIntOp::create(*builder_, loc, p[2], 64)
                  .getResult();
        auto durOp =
            pulse::DurationFromIntOp::create(*builder_, loc, durTy, durCycles);
        auto waitOp = pulse::WaitOp::create(*builder_, loc, lineVal.getType(),
                                            lineVal, durOp.getResult());
        bind(p[1], waitOp.getResult());
        break;
      }

      case kOpVariableWait: {
        // [line, result line, cycles id, lowest, highest (-1: none),
        //  1 if the length is an answer of a classical call]
        requirePayload(6);
        auto dur =
            pulse::DurationFromIntOp::create(*builder_, loc, durTy, look(p[2]));
        mlir::Value line = look(p[0]);
        auto waitOp = pulse::WaitOp::create(*builder_, loc, line.getType(),
                                            line, dur.getResult());
        waitOp->setAttr("pulse.variable_wait", builder_->getUnitAttr());
        waitOp->setAttr("pulse.min_cycles", builder_->getI64IntegerAttr(p[3]));
        if (p[4] >= 0)
          waitOp->setAttr("pulse.max_cycles",
                          builder_->getI64IntegerAttr(p[4]));
        if (p[5])
          waitOp->setAttr("classical.variable", builder_->getUnitAttr());
        bind(p[1], waitOp.getResult());
        break;
      }

      case kOpShiftPhase: {
        requirePayload(3);
        mlir::Value phaseVal =
            (pmask & (1 << 2))
                ? look(p[2])
                : mlir::arith::ConstantFloatOp::create(*builder_, loc, f64Ty,
                                                       llvm::APFloat(i2f(p[2])))
                      .getResult();
        auto spOp = pulse::ShiftPhaseOp::create(*builder_, loc, toneTy,
                                                look(p[0]), phaseVal);
        bind(p[1], spOp.getResult());
        break;
      }

      case kOpSetPhase: {
        requirePayload(3);
        mlir::Value phaseVal =
            (pmask & (1 << 2))
                ? look(p[2])
                : mlir::arith::ConstantFloatOp::create(*builder_, loc, f64Ty,
                                                       llvm::APFloat(i2f(p[2])))
                      .getResult();
        auto spOp = pulse::SetPhaseOp::create(*builder_, loc, toneTy,
                                              look(p[0]), phaseVal);
        bind(p[1], spOp.getResult());
        break;
      }

      case kOpShiftFreq: {
        requirePayload(3);
        mlir::Value fVal =
            (pmask & (1 << 2))
                ? look(p[2])
                : mlir::arith::ConstantFloatOp::create(*builder_, loc, f64Ty,
                                                       llvm::APFloat(i2f(p[2])))
                      .getResult();
        auto sfOp = pulse::ShiftFrequencyOp::create(*builder_, loc, toneTy,
                                                    look(p[0]), fVal);
        bind(p[1], sfOp.getResult());
        break;
      }

      case kOpSetFreq: {
        requirePayload(3);
        mlir::Value fVal =
            (pmask & (1 << 2))
                ? look(p[2])
                : mlir::arith::ConstantFloatOp::create(*builder_, loc, f64Ty,
                                                       llvm::APFloat(i2f(p[2])))
                      .getResult();
        auto sfOp = pulse::SetFrequencyOp::create(*builder_, loc, toneTy,
                                                  look(p[0]), fVal);
        bind(p[1], sfOp.getResult());
        break;
      }

      case kOpClassicalDeclare: {
        // [index, id, link, flags, min, nominal, max, timeout, deadline,
        //  name length, args length, results length, text words...]
        if (plen < 12)
          throw std::invalid_argument("packed: malformed classical function");
        int64_t lengths = p[9] + p[10] + p[11];
        if (p[0] != static_cast<int64_t>(callables.size()) || p[9] <= 0 ||
            p[10] < 0 || p[11] < 0 || plen != 12 + (lengths + 7) / 8)
          throw std::invalid_argument("packed: malformed classical function");
        std::string text(reinterpret_cast<const char *>(p + 12),
                         static_cast<size_t>(lengths));
        std::string symbol = text.substr(0, static_cast<size_t>(p[9]));
        auto split = [&](llvm::StringRef list) {
          llvm::SmallVector<llvm::StringRef> items;
          if (!list.empty())
            list.split(items, ',');
          return builder_->getStrArrayAttr(items);
        };
        llvm::StringRef args = llvm::StringRef(text).substr(
            static_cast<size_t>(p[9]), static_cast<size_t>(p[10]));
        llvm::StringRef results = llvm::StringRef(text).substr(
            static_cast<size_t>(p[9] + p[10]), static_cast<size_t>(p[11]));
        int64_t flags = p[3];
        static const char *modes[] = {"bounded", "timeout", "blocking"};
        if ((flags & 3) > 2)
          throw std::invalid_argument("packed: unknown latency mode");
        mlir::OpBuilder::InsertionGuard guard(*builder_);
        builder_->setInsertionPoint(funcOp);
        auto callable = classical::CallableOp::create(
            *builder_, loc, builder_->getStringAttr(symbol),
            builder_->getI64IntegerAttr(p[1]),
            builder_->getI64IntegerAttr(p[2]), split(args), split(results),
            builder_->getF64FloatAttr(i2f(p[4])),
            builder_->getF64FloatAttr(i2f(p[5])),
            builder_->getF64FloatAttr(i2f(p[6])),
            builder_->getStringAttr(modes[flags & 3]),
            builder_->getStringAttr(flags & 4 ? "lazy" : "eager"),
            flags & 16 ? builder_->getF64FloatAttr(i2f(p[7]))
                       : mlir::FloatAttr(),
            flags & 32 ? builder_->getF64FloatAttr(i2f(p[8]))
                       : mlir::FloatAttr(),
            flags & 8 ? builder_->getUnitAttr() : mlir::UnitAttr());
        callables.push_back(callable);
        break;
      }
      case kOpClassicalSubmit: {
        // [function index, future id, n, argument ids...]
        if (plen < 3 || p[2] < 0 || plen != 3 + p[2])
          throw std::invalid_argument("packed: malformed classical submit");
        auto callable = callableAt(p[0]);
        llvm::SmallVector<mlir::Value> args;
        for (int64_t i = 0; i < p[2]; ++i)
          args.push_back(look(p[3 + i]));
        int64_t device = callable.getDevice();
        auto submit = classical::SubmitOp::create(
            *builder_, loc, classical::LinkType::get(ctx_.get()),
            classical::FutureType::get(ctx_.get()), linkOf(device),
            mlir::FlatSymbolRefAttr::get(callable.getSymNameAttr()), args);
        links[device] = submit.getUpdatedLink();
        bind(p[1], submit.getFuture());
        break;
      }
      case kOpClassicalWait: {
        // [function index, future id, has timeout, timeout, n, result ids...]
        if (plen < 5 || p[4] < 0 || plen != 5 + p[4])
          throw std::invalid_argument("packed: malformed classical wait");
        auto callable = callableAt(p[0]);
        llvm::SmallVector<mlir::Type> types =
            classical::answerTypes(callable, p[2] != 0);
        if (static_cast<int64_t>(types.size()) != p[4])
          throw std::invalid_argument("packed: classical wait expects " +
                                      std::to_string(types.size()) +
                                      " results");
        auto wait = classical::WaitOp::create(
            *builder_, loc, types, look(p[1]),
            mlir::FlatSymbolRefAttr::get(callable.getSymNameAttr()),
            p[2] ? builder_->getF64FloatAttr(i2f(p[3])) : mlir::FloatAttr());
        for (int64_t i = 0; i < p[4]; ++i)
          bind(p[5 + i], wait.getValues()[i]);
        break;
      }
      case kOpClassicalCompare: {
        // [result id, predicate (eq ne lt le gt ge), type (0 i64, 1 f64),
        //  lhs, rhs]
        requirePayload(5);
        if (p[1] < 0 || p[1] > 5 || p[2] < 0 || p[2] > 1)
          throw std::invalid_argument("packed: malformed classical compare");
        mlir::Value lhs = look(p[3]), rhs = look(p[4]);
        mlir::Value result;
        if (p[2] == 1) {
          static const mlir::arith::CmpFPredicate fp[] = {
              mlir::arith::CmpFPredicate::OEQ, mlir::arith::CmpFPredicate::UNE,
              mlir::arith::CmpFPredicate::OLT, mlir::arith::CmpFPredicate::OLE,
              mlir::arith::CmpFPredicate::OGT, mlir::arith::CmpFPredicate::OGE};
          result =
              mlir::arith::CmpFOp::create(*builder_, loc, fp[p[1]], lhs, rhs);
        } else {
          static const mlir::arith::CmpIPredicate ip[] = {
              mlir::arith::CmpIPredicate::eq,  mlir::arith::CmpIPredicate::ne,
              mlir::arith::CmpIPredicate::slt, mlir::arith::CmpIPredicate::sle,
              mlir::arith::CmpIPredicate::sgt, mlir::arith::CmpIPredicate::sge};
          result =
              mlir::arith::CmpIOp::create(*builder_, loc, ip[p[1]], lhs, rhs);
        }
        bind(p[0], result);
        break;
      }

      default:
        throw std::invalid_argument("packed: unknown opcode " +
                                    std::to_string(opcode));
      }

      cur += headerWords + plen;
    }
    if (!regions.empty())
      throw std::invalid_argument("packed: region left open");

    mlir::func::ReturnOp::create(*builder_, loc);
    auto owningModule = mlir::OwningOpRef<mlir::ModuleOp>(module);
    return PulseModule(ctx_, std::move(owningModule));
  }

private:
  std::shared_ptr<mlir::MLIRContext> ctx_;
  std::unique_ptr<mlir::OpBuilder> builder_;
  std::mutex lock_;
};

// -----------------------------------------------------------------------
// JITModule: an LLVM-dialect module compiled in-process by ORC
// -----------------------------------------------------------------------
// Shared libraries a JIT module calls into, opened RTLD_LOCAL. MLIR's
// sharedLibPaths opens them through llvm::sys::DynamicLibrary, which uses
// RTLD_GLOBAL and so would add the GPU runtime and its CUDA dependencies to
// the process's global symbol scope.
class RuntimeLibraries {
public:
  explicit RuntimeLibraries(const std::vector<std::string> &paths) {
    for (const auto &path : paths) {
      void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
      if (!handle) {
        const char *reason = dlerror();
        close();
        throw std::runtime_error("JIT could not load " + path + ": " +
                                 (reason ? reason : "unknown error"));
      }
      handles_.push_back(handle);
    }
  }
  ~RuntimeLibraries() { close(); }
  RuntimeLibraries(const RuntimeLibraries &) = delete;
  RuntimeLibraries &operator=(const RuntimeLibraries &) = delete;

  void *lookup(const std::string &name) const {
    for (void *handle : handles_)
      if (void *symbol = dlsym(handle, name.c_str()))
        return symbol;
    return nullptr;
  }

private:
  void close() {
    for (void *handle : handles_)
      dlclose(handle);
    handles_.clear();
  }

  std::vector<void *> handles_;
};

class JITModule {
public:
  JITModule(std::unique_ptr<RuntimeLibraries> libraries,
            std::unique_ptr<mlir::ExecutionEngine> engine,
            std::vector<std::string> unresolved)
      : libraries_(std::move(libraries)), engine_(std::move(engine)),
        unresolved_(std::move(unresolved)) {}

  uintptr_t lookup(const std::string &name) {
    auto symbol = engine_->lookup(name);
    if (!symbol) {
      std::string message;
      llvm::raw_string_ostream os(message);
      os << symbol.takeError();
      // The missing symbols themselves go to the session's error reporter;
      // name the external functions no given library defines.
      if (!unresolved_.empty()) {
        os << "; external functions in none of the given libraries:";
        for (const std::string &external : unresolved_)
          os << " " << external;
      }
      throw std::runtime_error("JIT symbol lookup failed for '" + name +
                               "': " + message);
    }
    return reinterpret_cast<uintptr_t>(*symbol);
  }

private:
  // Declared first so the engine, whose code points into them, goes first.
  std::unique_ptr<RuntimeLibraries> libraries_;
  std::unique_ptr<mlir::ExecutionEngine> engine_;
  std::vector<std::string> unresolved_;
};

void addPulseLoweringPasses(mlir::PassManager &pm) {
  pm.addPass(pulse::createPulseToQOpPass());
  pm.addPass(qop::createQOpToCuDensityMatPass());
  pm.addPass(cudm::createCuDensityMatToLLVMPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
}

JITModule createJITModule(mlir::ModuleOp module,
                          const std::vector<std::string> &shared_libs) {
  static const bool nativeTargetReady = [] {
    return !llvm::InitializeNativeTarget() &&
           !llvm::InitializeNativeTargetAsmPrinter();
  }();
  if (!nativeTargetReady)
    throw std::runtime_error("LLVM native target initialization failed");

  auto libraries = std::make_unique<RuntimeLibraries>(shared_libs);
  std::vector<std::pair<std::string, void *>> resolved;
  std::vector<std::string> unresolved;
  module.walk([&](mlir::LLVM::LLVMFuncOp function) {
    if (!function.isExternal())
      return;
    std::string name = function.getName().str();
    if (void *address = libraries->lookup(name))
      resolved.emplace_back(std::move(name), address);
    else
      unresolved.push_back(std::move(name));
  });

  // ExecutionEngineOptions::transformer is a function_ref: the
  // std::function it refers to must outlive ExecutionEngine::create.
  auto transformer = mlir::makeOptimizingTransformer(
      /*optLevel=*/2, /*sizeLevel=*/0, /*targetMachine=*/nullptr);
  mlir::ExecutionEngineOptions options;
  options.transformer = transformer;
  auto engine = mlir::ExecutionEngine::create(module, options);
  if (!engine) {
    std::string message;
    llvm::raw_string_ostream os(message);
    os << engine.takeError();
    throw std::runtime_error("JIT compilation failed: " + message);
  }
  (*engine)->registerSymbols([&](llvm::orc::MangleAndInterner interner) {
    llvm::orc::SymbolMap symbols;
    for (const auto &[name, address] : resolved)
      symbols[interner(name)] = {llvm::orc::ExecutorAddr::fromPtr(address),
                                 llvm::JITSymbolFlags::Exported};
    return symbols;
  });
  return JITModule(std::move(libraries), std::move(*engine),
                   std::move(unresolved));
}

// -----------------------------------------------------------------------
// MLIRPipeline: the text-in, text-out lowerings and the JIT entry points of
// the Python runtime
// -----------------------------------------------------------------------
class MLIRPipeline {
public:
  MLIRPipeline() : registry_(makeRegistry()) {}

  JITModule jit_pulse(const std::string &pulse_mlir,
                      const std::vector<std::string> &shared_libs) {
    mlir::MLIRContext ctx;
    ctx.appendDialectRegistry(registry_);
    ctx.loadAllAvailableDialects();
    mlir::registerBuiltinDialectTranslation(ctx);
    mlir::registerLLVMDialectTranslation(ctx);
    auto module =
        parseModule(pulse_mlir, &ctx, "Failed to parse pulse MLIR text");
    mlir::PassManager pm(&ctx);
    addPulseLoweringPasses(pm);
    runPipeline(pm, *module, "MLIR pass pipeline failed");
    return createJITModule(*module, shared_libs);
  }

  JITModule jit_llvm_dialect(const std::string &mlir_text,
                             const std::vector<std::string> &shared_libs) {
    mlir::MLIRContext ctx;
    ctx.appendDialectRegistry(registry_);
    ctx.loadAllAvailableDialects();
    mlir::registerBuiltinDialectTranslation(ctx);
    mlir::registerLLVMDialectTranslation(ctx);
    auto module =
        parseModule(mlir_text, &ctx, "Failed to parse LLVM-dialect MLIR text");
    return createJITModule(*module, shared_libs);
  }

  std::string run_pulse_to_qop(const std::string &pulse_mlir) {
    mlir::MLIRContext ctx;
    ctx.appendDialectRegistry(registry_);
    ctx.loadAllAvailableDialects();
    auto module =
        parseModule(pulse_mlir, &ctx, "Failed to parse pulse MLIR text");
    mlir::PassManager pm(&ctx);
    pm.addPass(pulse::createPulseToQOpPass());
    runPipeline(pm, *module, "pulse-to-qop pass failed");
    std::string result;
    llvm::raw_string_ostream os(result);
    module->print(os);
    return result;
  }

  std::string run_qop_to_cudm(const std::string &qop_mlir) {
    mlir::MLIRContext ctx;
    ctx.appendDialectRegistry(registry_);
    ctx.loadAllAvailableDialects();
    auto module = parseModule(qop_mlir, &ctx, "Failed to parse qop MLIR text");
    mlir::PassManager pm(&ctx);
    pm.addPass(qop::createQOpToCuDensityMatPass());
    runPipeline(pm, *module, "qop-to-cudm pass failed");
    std::string result;
    llvm::raw_string_ostream os(result);
    module->print(os);
    return result;
  }

  PulseModule parse_to_module(const std::string &mlir_text) {
    auto ctx = std::make_shared<mlir::MLIRContext>();
    ctx->appendDialectRegistry(registry_);
    ctx->loadAllAvailableDialects();
    auto module =
        parseModule(mlir_text, ctx.get(), "Failed to parse MLIR text");
    return PulseModule(ctx, std::move(module));
  }

private:
  mlir::DialectRegistry registry_;
};

} // namespace

NB_MODULE(_cudaq_pulse_native, m) {
  m.doc() = "cudaq-pulse native MLIR pipeline bindings";
  SharedContextPool::handleForks();

  nb::class_<PulseModule>(m, "PulseModule")
      .def("print", &PulseModule::print,
           nb::call_guard<nb::gil_scoped_release>(),
           "Print MLIR text representation")
      .def("run_passes", &PulseModule::run_passes,
           nb::call_guard<nb::gil_scoped_release>(),
           "Run named MLIR passes on this module")
      .def("run_full_lowering", &PulseModule::run_full_lowering,
           nb::call_guard<nb::gil_scoped_release>(),
           "Run full lowering: pulse -> qop -> cudm -> llvm")
      .def("schedule", &PulseModule::schedule,
           nb::call_guard<nb::gil_scoped_release>(),
           "Schedule pulse operations", nb::arg("policy"),
           nb::arg("max_drives") = 4, nb::arg("max_readouts") = 2,
           nb::arg("readout_latency") = 0, nb::arg("switch_penalty") = 0)
      .def(
          "events", &PulseModule::events,
          nb::call_guard<nb::gil_scoped_release>(),
          "(kind, line, qubit, start_cycles, duration_cycles) of each timed op")
      .def("controller_events", &PulseModule::controller_events,
           nb::call_guard<nb::gil_scoped_release>(),
           "Each timed op with its iteration, waveform, tone frame, "
           "branches, references, callee and wait, for a controller")
      .def("simulation_module", &PulseModule::simulation_module,
           "This module with loops holding calls unrolled, for simulation")
      .def("next_call", &PulseModule::next_call, nb::arg("path"),
           "The first call on a path whose answer is not bound yet")
      .def("bind_call", &PulseModule::bind_call,
           "This module with one call's answer bound, scheduled again")
      .def("placeholders", &PulseModule::placeholders,
           "This module with unbound answers set to zero")
      .def("epoch_boundaries", &PulseModule::epoch_boundaries,
           "The variable and blocking waits that start new epochs")
      .def("event_epochs", &PulseModule::event_epochs, nb::arg("syncs"),
           "The epoch of each event, in the order the events are listed")
      .def("classical_calls", &PulseModule::classical_calls,
           "Every execution of a classical call, with its timing")
      .def("readout_drives", &PulseModule::readout_drives, nb::arg("path"),
           nb::call_guard<nb::gil_scoped_release>(),
           "Every readout on a path: (index, iteration, frequency Hz, "
           "amplitude)")
      .def("next_measurement", &PulseModule::next_measurement,
           nb::call_guard<nb::gil_scoped_release>(),
           "The next measurement without an outcome on a path", nb::arg("path"))
      .def("operator_program", &PulseModule::operator_program,
           nb::call_guard<nb::gil_scoped_release>(),
           "The lowered Hamiltonian and collapse operators as data, with the "
           "drives evaluated at the given float arguments",
           nb::arg("arguments") = std::vector<double>{})
      .def("set_attributes", &PulseModule::set_attributes,
           nb::call_guard<nb::gil_scoped_release>(),
           "Merge an MLIR dictionary attribute into the module attributes",
           nb::arg("dictionary"))
      .def("param_names", &PulseModule::param_names,
           nb::call_guard<nb::gil_scoped_release>(), "Get parameter names")
      .def("specialize", &PulseModule::specialize,
           nb::call_guard<nb::gil_scoped_release>(),
           "Clone module, substitute parameters with constants, fold, schedule",
           nb::arg("f64_args"), nb::arg("i64_args"),
           nb::arg("schedule") = "alap", nb::arg("max_drives") = 4,
           nb::arg("max_readouts") = 2, nb::arg("readout_latency") = 0,
           nb::arg("switch_penalty") = 0);

  nb::class_<PulseModuleBuilder>(m, "PulseModuleBuilder")
      .def(nb::init<>())
      .def("build_from_packed", &PulseModuleBuilder::build_from_packed,
           nb::call_guard<nb::gil_scoped_release>(),
           "Build an in-memory PulseModule from a packed int64 numpy buffer",
           nb::arg("stream"), nb::arg("clock_ghz"), nb::arg("n_qubits"),
           nb::arg("qubit_freqs"),
           nb::arg("param_names") = std::vector<std::string>{},
           nb::arg("param_types") = std::vector<std::string>{},
           nb::arg("name") = "main");

  m.def(
      "shared_context_budget",
      [] {
        auto limits = SharedContextPool::budget();
        return std::make_tuple(limits.modules, limits.words);
      },
      "(modules, packed words) a shared MLIR context serves before it is "
      "retired, from CUDAQ_PULSE_CONTEXT_MAX_MODULES and "
      "CUDAQ_PULSE_CONTEXT_MAX_WORDS");
  m.def("shared_context_stats", &SharedContextPool::stats,
        "(contexts created, builders on the current context, its packed "
        "words)");

  m.def(
      "evaluate_callback",
      [](int32_t kind, const std::vector<double> &params,
         const std::vector<double> &times) {
        std::vector<double> values;
        values.reserve(times.size());
        for (double time : times) {
          double value = 0.0;
          if (cudaq_pulse::evaluateWaveform(kind, params.data(), params.size(),
                                            time, value))
            throw std::invalid_argument("invalid waveform callback");
          values.push_back(value);
        }
        return values;
      },
      nb::call_guard<nb::gil_scoped_release>(),
      "Evaluate a drive-envelope callback at the given times (ns)",
      nb::arg("kind"), nb::arg("params"), nb::arg("times"));

  nb::class_<JITModule>(m, "JITModule")
      .def("lookup", &JITModule::lookup,
           nb::call_guard<nb::gil_scoped_release>(),
           "Return the address of a JIT-compiled symbol", nb::arg("name"));

  nb::class_<MLIRPipeline>(m, "MLIRPipeline")
      .def(nb::init<>())
      .def("jit_pulse", &MLIRPipeline::jit_pulse,
           nb::call_guard<nb::gil_scoped_release>(),
           "Lower pulse MLIR and JIT-compile it in-process",
           nb::arg("pulse_mlir"), nb::arg("shared_libs"))
      .def("jit_llvm_dialect", &MLIRPipeline::jit_llvm_dialect,
           nb::call_guard<nb::gil_scoped_release>(),
           "JIT-compile an LLVM-dialect MLIR module in-process",
           nb::arg("mlir_text"), nb::arg("shared_libs"))
      .def("run_pulse_to_qop", &MLIRPipeline::run_pulse_to_qop,
           nb::call_guard<nb::gil_scoped_release>(),
           "Run pulse-to-qop pass only")
      .def("run_qop_to_cudm", &MLIRPipeline::run_qop_to_cudm,
           nb::call_guard<nb::gil_scoped_release>(),
           "Run qop-to-cudm pass only")
      .def("parse_to_module", &MLIRPipeline::parse_to_module,
           nb::call_guard<nb::gil_scoped_release>(),
           "Parse MLIR text into a PulseModule");
}
