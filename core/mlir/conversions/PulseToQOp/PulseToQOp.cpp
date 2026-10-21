/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// PulseToQOp conversion pass: lower a scheduled pulse program to the qop
// Hamiltonian/Lindblad construction for one time evolution.
//
// Drives are found by walking the program in execution order
// (Analysis/Execution.h), so retained loops and measurement-conditioned
// branches lower without unrolling the IR. A drive op that plays in every
// iteration of its loops with a fixed period and phase step becomes one
// pulse-train callback. A program that measures is lowered along one
// measurement-outcome path, given as the module attribute `pulse.path`
// (entries [readout index, iteration..., outcome], readouts numbered in
// walk order), over the window [qop.t_start, qop.t_end]: the window must
// end by the first measurement on the path that has no outcome, and the
// caller projects the state at each measurement between windows.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"
#include "cudaq-pulse/Dialect/Pulse/PulseEnums.h.inc"
#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseOps.h.inc"

#include "cudaq-pulse/Dialect/QOp/QOpDialect.h.inc"
#include "cudaq-pulse/Dialect/QOp/QOpEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "cudaq-pulse/Dialect/QOp/QOpAttrs.h.inc"
#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/QOp/QOpTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/QOp/QOpOps.h.inc"

#include "cudaq-pulse/Analysis/Execution.h"
#include "cudaq-pulse/Runtime/WaveformCallback.h"

#include <algorithm>
#include <cmath>
#include <optional>

using namespace mlir;

namespace {

using pulse::constantNumber;

struct WaveformCallbackData {
  int32_t kind;
  SmallVector<double> parameters;
};

/// The phase attributes that virtual-Z folding leaves on a drive.
static double phaseOffsetOf(Operation *drive) {
  double offset = 0.0;
  for (StringRef name : {"phase_offset", "frame_phase_offset"})
    if (auto attribute = drive->getAttrOfType<FloatAttr>(name))
      offset += attribute.getValueAsDouble();
  return offset;
}

/// The parts of a drive occurrence's callback that vary between occurrences
/// of the same op.
struct DriveTiming {
  double startNs, durationNs, phase, detuning;
};

static DriveTiming driveTiming(const pulse::Occurrence &occurrence,
                               double clockGHz, double qubitFrequencyHz,
                               double phaseOffset) {
  const double scale = 1.0 / clockGHz;
  const double startNs = static_cast<double>(occurrence.startVtu) * scale;
  const pulse::ToneState &tone = occurrence.tone;
  return {startNs, static_cast<double>(occurrence.durationVtu) * scale,
          tone.phase +
              2.0 * M_PI * (tone.frequencyHz - tone.frameFrequencyHz) * 1.0e-9 *
                  (startNs - tone.timeNs) +
              2.0 * M_PI * (tone.frameFrequencyHz - qubitFrequencyHz) * 1.0e-9 *
                  startNs +
              phaseOffset,
          2.0 * M_PI * (tone.frequencyHz - qubitFrequencyHz) * 1.0e-9};
}

/// The callback of one drive occurrence for one quadrature.
static std::optional<WaveformCallbackData>
getWaveformCallbackData(const pulse::Occurrence &occurrence, double clockGHz,
                        double qubitFrequencyHz, double driveScale,
                        int32_t quadrature) {
  Operation *drive = occurrence.op;
  auto *waveform = drive->getOperand(1).getDefiningOp();
  if (!waveform)
    return std::nullopt;

  const double scale = 1.0 / clockGHz;
  const DriveTiming timing =
      driveTiming(occurrence, clockGHz, qubitFrequencyHz, phaseOffsetOf(drive));
  // Common layout: start, duration, real amplitude, imaginary amplitude,
  // sigma, beta, rise/fall, phase-at-start, quadrature, detuning (rad/ns),
  // followed by sample data.
  WaveformCallbackData result{
      0,
      {timing.startNs, timing.durationNs, 0.0, 0.0, 0.0, 0.0, 0.0, timing.phase,
       static_cast<double>(quadrature), timing.detuning}};
  auto set = [&](int32_t kind, Value amplitude, Value sigma = {},
                 Value beta = {}) -> bool {
    auto amplitudeValue = constantNumber(amplitude);
    if (!amplitudeValue)
      return false;
    result.kind = kind;
    result.parameters[2] = *amplitudeValue * driveScale;
    if (sigma) {
      auto sigmaValue = constantNumber(sigma);
      if (!sigmaValue)
        return false;
      result.parameters[4] = *sigmaValue * scale;
    }
    if (beta) {
      auto betaValue = constantNumber(beta);
      if (!betaValue)
        return false;
      result.parameters[5] = *betaValue;
    }
    return true;
  };

  if (auto op = dyn_cast<pulse::SquarePulseOp>(waveform)) {
    auto real = constantNumber(op.getAmpReal());
    auto imag = constantNumber(op.getAmpImag());
    if (!real || !imag)
      return std::nullopt;
    result.kind = 1;
    result.parameters[2] = *real * driveScale;
    result.parameters[3] = *imag * driveScale;
  } else if (auto op = dyn_cast<pulse::GaussianPulseOp>(waveform)) {
    if (!set(2, op.getAmplitude(), op.getSigma()))
      return std::nullopt;
  } else if (auto op = dyn_cast<pulse::DRAGPulseOp>(waveform)) {
    if (!set(3, op.getAmplitude(), op.getSigma(), op.getBeta()))
      return std::nullopt;
  } else if (auto op = dyn_cast<pulse::CosinePulseOp>(waveform)) {
    if (!set(4, op.getAmplitude()))
      return std::nullopt;
  } else if (auto op = dyn_cast<pulse::TanhRampOp>(waveform)) {
    if (!set(5, op.getAmplitude(), op.getSigma()))
      return std::nullopt;
  } else if (auto op = dyn_cast<pulse::GaussianSquarePulseOp>(waveform)) {
    if (!set(6, op.getAmplitude(), op.getSigma()))
      return std::nullopt;
    auto riseFall = constantNumber(op.getRisefall());
    if (!riseFall)
      return std::nullopt;
    result.parameters[6] = *riseFall * scale;
  } else if (auto op = dyn_cast<pulse::CustomSamplesOp>(waveform)) {
    result.kind = 7;
    for (Attribute sample : op.getSamples()) {
      auto value = dyn_cast<FloatAttr>(sample);
      if (!value)
        return std::nullopt;
      result.parameters.push_back(value.getValueAsDouble() * driveScale);
    }
  } else {
    return std::nullopt;
  }
  return result;
}

/// Leaf operators on qudit `target`: Pauli spin handlers for two levels,
/// otherwise d x d matrices built from the ladder operator a, with
/// X = a + a^dagger, Y = i (a^dagger - a), Z = 1 - 2 n.
class QuditOperators {
public:
  QuditOperators(OpBuilder &b, Location loc,
                 const DenseMap<int64_t, int64_t> &levels)
      : b(b), loc(loc), levels(levels) {}

  int64_t levelsOf(int64_t qudit) const {
    auto it = levels.find(qudit);
    return it == levels.end() ? 2 : it->second;
  }

  Value leaf(int64_t qudit, StringRef kind) {
    auto handlerTy = qop::HandlerType::get(b.getContext());
    int64_t d = levelsOf(qudit);
    if (d == 2) {
      auto parsed = qop::symbolizeHandlerKind(kind);
      auto target =
          arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(qudit));
      return qop::SpinOp::create(
          b, loc, handlerTy, target,
          qop::HandlerKindAttr::get(b.getContext(), *parsed));
    }
    // Column-major, interleaved real and imaginary parts.
    SmallVector<double> data(2 * d * d, 0.0);
    auto set = [&](int64_t row, int64_t column, double re, double im) {
      data[2 * (column * d + row)] += re;
      data[2 * (column * d + row) + 1] += im;
    };
    for (int64_t n = 1; n < d; ++n) {
      double amplitude = std::sqrt(static_cast<double>(n));
      // a |n> = sqrt(n) |n-1>: entry (n-1, n); a^dagger: entry (n, n-1).
      if (kind == "spin_x") {
        set(n - 1, n, amplitude, 0.0);
        set(n, n - 1, amplitude, 0.0);
      } else if (kind == "spin_y") {
        set(n, n - 1, 0.0, amplitude);
        set(n - 1, n, 0.0, -amplitude);
      } else if (kind == "spin_lowering") {
        set(n - 1, n, amplitude, 0.0);
      } else if (kind == "spin_raising") {
        set(n, n - 1, amplitude, 0.0);
      }
    }
    for (int64_t n = 0; n < d; ++n) {
      double level = static_cast<double>(n);
      if (kind == "spin_z")
        set(n, n, 1.0 - 2.0 * level, 0.0);
      else if (kind == "spin_i")
        set(n, n, 1.0, 0.0);
      else if (kind == "number")
        set(n, n, level, 0.0);
      else if (kind == "anharmonic")
        set(n, n, level * (level - 1.0), 0.0);
    }
    auto type = RankedTensorType::get({d, d, 2}, b.getF64Type());
    auto tensor = arith::ConstantOp::create(
        b, loc, DenseFPElementsAttr::get(type, ArrayRef<double>(data)));
    return qop::MatrixLeafOp::create(b, loc, handlerTy,
                                     b.getDenseI64ArrayAttr({qudit}), tensor,
                                     b.getDenseI64ArrayAttr({d}));
  }

  Value product(ArrayRef<std::pair<int64_t, StringRef>> factors, double re,
                double im = 0.0) {
    SmallVector<Value> leaves;
    for (auto [qudit, kind] : factors)
      leaves.push_back(leaf(qudit, kind));
    auto scalar = qop::ConstScalarOp::create(
        b, loc, qop::ScalarType::get(b.getContext()), b.getF64FloatAttr(re),
        b.getF64FloatAttr(im));
    return qop::MakeProductOp::create(
        b, loc, qop::ProductType::get(b.getContext()), scalar, leaves);
  }

  Value drive(int64_t qudit, StringRef kind, StringRef callbackName,
              const WaveformCallbackData &callback) {
    auto scalar = qop::CallbackScalarOp::create(
        b, loc, qop::ScalarType::get(b.getContext()),
        FlatSymbolRefAttr::get(b.getContext(), callbackName));
    scalar->setAttr("cudm.callback_kind", b.getI32IntegerAttr(callback.kind));
    scalar->setAttr("cudm.callback_params",
                    b.getDenseF64ArrayAttr(callback.parameters));
    return qop::MakeProductOp::create(b, loc,
                                      qop::ProductType::get(b.getContext()),
                                      scalar, ValueRange{leaf(qudit, kind)});
  }

private:
  OpBuilder &b;
  Location loc;
  const DenseMap<int64_t, int64_t> &levels;
};

/// One callback for every occurrence of a drive op, when each occurrence
/// differs from the first only by a fixed start period and phase step per
/// enclosing loop and every iteration plays. `callback` is the first
/// occurrence's; the other occurrences of the same op can differ from it
/// only in their timing.
static std::optional<WaveformCallbackData>
pulseTrain(ArrayRef<pulse::Occurrence> occurrences,
           const WaveformCallbackData &callback, double clockGHz,
           double qubitFrequencyHz) {
  Operation *op = occurrences.front().op;
  auto loops = pulse::enclosingLoops(op);
  if (loops.empty())
    return std::nullopt;
  const size_t depth = loops.size();
  int64_t total = 1;
  SmallVector<int64_t> counts, strides(depth, 1);
  for (auto loop : loops) {
    counts.push_back(pulse::constantTripCount(loop).value_or(0));
    total *= counts.back();
  }
  if (static_cast<int64_t>(occurrences.size()) != total || total < 2)
    return std::nullopt;
  for (size_t level = depth - 1; level > 0; --level)
    strides[level - 1] = strides[level] * counts[level];

  // The occurrences must come in iteration order, one per iteration.
  const double offset = phaseOffsetOf(op);
  SmallVector<DriveTiming> timing;
  timing.reserve(occurrences.size());
  std::vector<int64_t> expected(depth, 0);
  for (const pulse::Occurrence &occurrence : occurrences) {
    if (occurrence.iteration != expected)
      return std::nullopt;
    timing.push_back(
        driveTiming(occurrence, clockGHz, qubitFrequencyHz, offset));
    for (size_t level = depth; level-- > 0;) {
      if (++expected[level] < counts[level])
        break;
      expected[level] = 0;
    }
  }

  const DriveTiming &base = timing.front();
  SmallVector<double> periods(depth, 0.0), steps(depth, 0.0);
  for (size_t level = 0; level < depth; ++level) {
    if (counts[level] < 2)
      continue;
    const DriveTiming &next = timing[strides[level]];
    periods[level] = next.startNs - base.startNs;
    steps[level] = next.phase - base.phase;
    if (periods[level] <= 0.0)
      return std::nullopt;
  }
  for (auto [occurrence, t] : llvm::zip(occurrences, timing)) {
    double start = base.startNs, phase = base.phase;
    for (auto [k, period, step] :
         llvm::zip(occurrence.iteration, periods, steps)) {
      start += k * period;
      phase += k * step;
    }
    if (std::abs(t.startNs - start) > 1e-9 ||
        std::abs(t.phase - phase) > 1e-9 || t.durationNs != base.durationNs ||
        t.detuning != base.detuning)
      return std::nullopt;
  }
  const auto &first = callback.parameters;
  SmallVector<double> header(first.begin(),
                             first.begin() + cudaq_pulse::kWaveformHeader);
  // Square pulses that abut with a continuous carrier phase are one longer
  // square pulse; fold such innermost levels into the duration.
  size_t levels = loops.size();
  auto near = [](double a, double b) {
    return std::abs(a - b) <= 1e-9 * std::max(1.0, std::abs(b));
  };
  while (callback.kind == 1 && levels > 0) {
    size_t level = levels - 1;
    if (counts[level] >= 2 && (!near(periods[level], header[1]) ||
                               !near(steps[level], header[9] * header[1])))
      break;
    header[1] *= static_cast<double>(counts[level]);
    --levels;
  }
  if (levels == 0) {
    WaveformCallbackData square{callback.kind, header};
    square.parameters.append(first.begin() + cudaq_pulse::kWaveformHeader,
                             first.end());
    return square;
  }
  WaveformCallbackData train{callback.kind | cudaq_pulse::kTrainFlag, header};
  train.parameters.push_back(static_cast<double>(levels));
  for (size_t level = 0; level < levels; ++level) {
    train.parameters.push_back(periods[level]);
    train.parameters.push_back(static_cast<double>(counts[level]));
    train.parameters.push_back(steps[level]);
  }
  train.parameters.append(first.begin() + cudaq_pulse::kWaveformHeader,
                          first.end());
  return train;
}

struct PulseToQOpPass
    : public PassWrapper<PulseToQOpPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PulseToQOpPass)

  StringRef getArgument() const final { return "pulse-to-qop"; }
  StringRef getDescription() const final {
    return "Lower pulse dialect ops to qop Hamiltonian/Lindblad construction";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<qop::QOpDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    OpBuilder b(module.getContext());

    func::FuncOp mainFunc;
    module.walk([&](func::FuncOp fn) {
      if (fn.getName() == "main")
        mainFunc = fn;
    });
    if (!mainFunc) {
      module.emitError("no @main function found");
      return signalPassFailure();
    }

    double clockGHz = 1.0;
    if (auto clock = module->getAttrOfType<FloatAttr>("pulse.clock_ghz"))
      clockGHz = clock.getValueAsDouble();
    if (clockGHz <= 0.0) {
      module.emitError("pulse.clock_ghz must be positive");
      return signalPassFailure();
    }

    DenseMap<int64_t, double> qubitFreqHz;
    if (auto frequencies =
            module->getAttrOfType<DenseF64ArrayAttr>("pulse.qubit_freq_hz")) {
      for (auto [index, frequency] : llvm::enumerate(frequencies.asArrayRef()))
        qubitFreqHz[static_cast<int64_t>(index)] = frequency;
    }
    DenseMap<int64_t, int64_t> levels;
    if (auto attr = module->getAttrOfType<DenseI64ArrayAttr>("pulse.levels"))
      for (auto [index, d] : llvm::enumerate(attr.asArrayRef())) {
        if (d < 2) {
          module.emitError("pulse.levels entries must be at least 2");
          return signalPassFailure();
        }
        levels[static_cast<int64_t>(index)] = d;
      }

    // Readouts are numbered in walk order for `pulse.path`.
    SmallVector<Operation *> readouts;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "pulse.get_drive_line") {
        if (auto qubitAttr = op->getAttrOfType<IntegerAttr>("qubit")) {
          int64_t qi = qubitAttr.getInt();
          if (auto freqAttr = op->getAttrOfType<FloatAttr>("frequency_hz"))
            qubitFreqHz[qi] = freqAttr.getValueAsDouble();
        }
      }
      if (isa<pulse::ReadoutOp>(op))
        readouts.push_back(op);
      if (op->getName().getStringRef() == "pulse.iq_acquire")
        readouts.push_back(op);
    });
    pulse::Path path;
    auto pathAttr = module->getAttrOfType<ArrayAttr>("pulse.path");
    if (!readouts.empty() && !pathAttr) {
      readouts.front()->emitError(
          "a program that measures evolves in segments between measurements; "
          "lower it along a measurement-outcome path (pulse.path), as "
          "pulse.evolve() does");
      return signalPassFailure();
    }
    if (pathAttr)
      for (Attribute entry : pathAttr) {
        auto values = dyn_cast<DenseI64ArrayAttr>(entry);
        if (!values || values.size() < 2 || values[0] < 0 ||
            values[0] >= static_cast<int64_t>(readouts.size())) {
          module.emitError("pulse.path entries are [readout, iteration..., "
                           "outcome] with a valid readout index");
          return signalPassFailure();
        }
        auto array = values.asArrayRef();
        path[{readouts[array.front()],
              std::vector<int64_t>(array.begin() + 1, array.end() - 1)}] =
            array.back();
      }

    std::optional<double> windowStart, windowEnd;
    if (auto attr = module->getAttrOfType<FloatAttr>("qop.t_start"))
      windowStart = attr.getValueAsDouble();
    if (auto attr = module->getAttrOfType<FloatAttr>("qop.t_end"))
      windowEnd = attr.getValueAsDouble();

    // Collect the drives played on this path, in execution order.
    int64_t endVtu = 0;
    std::map<pulse::MeasurementKey, double> measurementEndNs;
    SmallVector<Operation *> driveOrder;
    DenseMap<Operation *, SmallVector<pulse::Occurrence>> drives;
    pulse::ExecutionWalker walker(clockGHz, path);
    walker.onTimed = [&](const pulse::Occurrence &occurrence) {
      endVtu = std::max(endVtu, occurrence.startVtu + occurrence.durationVtu);
    };
    walker.onDrive = [&](const pulse::Occurrence &occurrence) {
      double startNs = occurrence.startVtu / clockGHz;
      double endNs = (occurrence.startVtu + occurrence.durationVtu) / clockGHz;
      if ((windowEnd && startNs >= *windowEnd) ||
          (windowStart && endNs <= *windowStart))
        return;
      auto &list = drives[occurrence.op];
      if (list.empty())
        driveOrder.push_back(occurrence.op);
      list.push_back(occurrence);
    };
    walker.onReadout = [&](const pulse::Occurrence &occurrence,
                           const pulse::MeasurementKey &key) {
      measurementEndNs[key] =
          (occurrence.startVtu + occurrence.durationVtu) / clockGHz;
      return false;
    };
    if (!walker.run(mainFunc)) {
      walker.errorOp->emitError(walker.error);
      return signalPassFailure();
    }
    // A branch on a measurement without an outcome is skipped: that is only
    // right if the measurement completes no earlier than the window ends.
    for (const auto &key : walker.undecided) {
      double endNs = measurementEndNs[key];
      if (!windowEnd || endNs < *windowEnd) {
        key.readout->emitError("no outcome on pulse.path for this measurement, "
                               "which completes at ")
            << endNs << " ns, before the evolution window ends";
        return signalPassFailure();
      }
    }

    // Carry the scheduled pulse interval into the evolution lowering. Pulse
    // time is expressed in virtual clock units; cuDensityMat callbacks and
    // integrators use nanoseconds.
    if (endVtu > 0) {
      const double endNs = static_cast<double>(endVtu) / clockGHz;
      if (!module->hasAttr("qop.t_start"))
        module->setAttr("qop.t_start", b.getF64FloatAttr(0.0));
      if (!module->hasAttr("qop.t_end"))
        module->setAttr("qop.t_end", b.getF64FloatAttr(endNs));
      if (!module->hasAttr("qop.num_steps"))
        module->setAttr("qop.num_steps",
                        b.getI64IntegerAttr(std::max<int64_t>(100, endVtu)));
      if (!module->hasAttr("qop.integrator"))
        module->setAttr("qop.integrator", b.getStringAttr("rk4"));
    }

    // The source program is replaced by the operator construction.
    auto &block = mainFunc.getBody().front();
    Operation *returnOp = block.getTerminator();
    SmallVector<Operation *> sourceOps;
    for (Operation &op : block.without_terminator())
      sourceOps.push_back(&op);
    b.setInsertionPoint(returnOp);
    auto loc = returnOp->getLoc();
    QuditOperators ops(b, loc, levels);

    auto opTy = qop::OpType::get(b.getContext());
    auto superOpTy = qop::SuperOpType::get(b.getContext());

    SmallVector<Value> allProducts;

    // Evolution uses one rotating frame per qubit. Bare qubit frequencies are
    // therefore removed from H; tone detuning is encoded in each drive
    // callback below. This avoids a numerically stiff multi-GHz lab-frame ODE.
    auto appendInteractions = [&](StringRef pairAttrName,
                                  StringRef strengthAttrName,
                                  StringRef spinKind) -> LogicalResult {
      auto pairs = module->getAttrOfType<DenseI64ArrayAttr>(pairAttrName);
      auto strengths =
          module->getAttrOfType<DenseF64ArrayAttr>(strengthAttrName);
      if (!pairs && !strengths)
        return success();
      if (!pairs || !strengths || pairs.size() != 2 * strengths.size()) {
        module.emitError("interaction pair and strength metadata disagree for ")
            << pairAttrName;
        return failure();
      }
      for (auto [index, strength] : llvm::enumerate(strengths.asArrayRef())) {
        int64_t first = pairs[2 * index];
        int64_t second = pairs[2 * index + 1];
        if (first < 0 || second < 0 || !qubitFreqHz.count(first) ||
            !qubitFreqHz.count(second)) {
          module.emitError("interaction references an inactive qubit in ")
              << pairAttrName;
          return failure();
        }
        allProducts.push_back(
            ops.product({{first, spinKind}, {second, spinKind}},
                        strength * 2.0 * M_PI * 1.0e-9));
      }
      return success();
    };
    if (failed(appendInteractions("pulse.coupling_pairs",
                                  "pulse.coupling_strength_hz", "spin_x")) ||
        failed(appendInteractions("pulse.crosstalk_pairs",
                                  "pulse.crosstalk_strength_hz", "spin_z")))
      return signalPassFailure();

    // A multilevel transmon's anharmonicity, (alpha / 2) n (n - 1).
    if (auto anharmonicity =
            module->getAttrOfType<DenseF64ArrayAttr>("pulse.anharmonicity_hz"))
      for (auto [index, alpha] : llvm::enumerate(anharmonicity.asArrayRef()))
        if (alpha != 0.0 && ops.levelsOf(index) > 2)
          allProducts.push_back(
              ops.product({{static_cast<int64_t>(index), "anharmonic"}},
                          0.5 * alpha * 2.0 * M_PI * 1.0e-9));

    // 2. Time-dependent drive terms
    int driveIdx = 0;
    for (Operation *op : driveOrder) {
      auto &occurrences = drives[op];
      auto *allocation = pulse::traceAllocation(op->getOperand(0));
      auto qubitAttr = allocation
                           ? allocation->getAttrOfType<IntegerAttr>("qubit")
                           : IntegerAttr();
      if (!qubitAttr) {
        op->emitError("cannot determine the physical qubit for this drive; "
                      "the line must originate from pulse.get_drive_line "
                      "with a 'qubit' attribute");
        return signalPassFailure();
      }
      int64_t qubitIdx = qubitAttr.getInt();

      auto qubitFrequency = qubitFreqHz.find(qubitIdx);
      if (qubitFrequency == qubitFreqHz.end()) {
        op->emitError("cannot determine the physical qubit frequency for "
                      "rotating-frame lowering");
        return signalPassFailure();
      }
      double driveScale = 1.0;
      if (auto scales = module->getAttrOfType<DenseF64ArrayAttr>(
              "pulse.drive_scale_rad_per_ns")) {
        if (qubitIdx < 0 || static_cast<size_t>(qubitIdx) >= scales.size() ||
            scales.asArrayRef()[qubitIdx] <= 0.0) {
          op->emitError("drive amplitude scale is missing or non-positive");
          return signalPassFailure();
        }
        driveScale = scales.asArrayRef()[qubitIdx];
      }

      static constexpr llvm::StringLiteral quadratures[] = {"spin_x", "spin_y"};
      for (auto [quadrature, kind] : llvm::enumerate(quadratures)) {
        SmallVector<WaveformCallbackData> callbacks;
        auto append = [&](const pulse::Occurrence &occurrence) {
          auto callback = getWaveformCallbackData(
              occurrence, clockGHz, qubitFrequency->second, driveScale,
              static_cast<int32_t>(quadrature));
          if (callback)
            callbacks.push_back(std::move(*callback));
          return callback.has_value();
        };
        if (!append(occurrences.front())) {
          op->emitError(
              "waveform cannot be lowered to a cuDensityMat callback; "
              "schedule the program, specialize all numeric values, "
              "and use a built-in or custom_samples waveform");
          return signalPassFailure();
        }
        if (auto train = pulseTrain(occurrences, callbacks.front(), clockGHz,
                                    qubitFrequency->second))
          callbacks.assign({std::move(*train)});
        else
          for (const pulse::Occurrence &occurrence :
               llvm::drop_begin(occurrences))
            append(occurrence);
        for (auto [index, callback] : llvm::enumerate(callbacks)) {
          std::string name = "drive_envelope_" + std::to_string(driveIdx) +
                             (callbacks.size() > 1 ? "_" + std::to_string(index)
                                                   : std::string()) +
                             (quadrature ? "_y" : "_x");
          allProducts.push_back(ops.drive(qubitIdx, kind, name, callback));
        }
      }
      driveIdx++;
    }

    // 3. Assemble Hamiltonian: qop.make_sum
    Value hamiltonian;
    if (!allProducts.empty()) {
      hamiltonian = qop::MakeSumOp::create(b, loc, opTy, allProducts);
    } else {
      // Trivial Hamiltonian: identity
      hamiltonian = qop::MakeSumOp::create(
          b, loc, opTy, ValueRange{ops.product({{0, "spin_i"}}, 0.0)});
    }

    // 4. Dissipators from module attributes (T1, T2)
    SmallVector<Value> collapseOps;

    auto t1AttrRaw = module->getAttrOfType<ArrayAttr>("pulse.t1_times");
    if (!t1AttrRaw)
      t1AttrRaw = module->getAttrOfType<ArrayAttr>("t1_times");
    if (auto t1Attr = t1AttrRaw) {
      for (int64_t qi = 0; qi < (int64_t)t1Attr.size(); qi++) {
        double t1 = cast<FloatAttr>(t1Attr[qi]).getValueAsDouble();
        if (t1 > 0) {
          double gamma = 1.0 / t1;
          auto lowering =
              ops.product({{qi, "spin_lowering"}}, std::sqrt(gamma));
          collapseOps.push_back(
              qop::MakeSumOp::create(b, loc, opTy, ValueRange{lowering}));
        }
      }
    }

    auto t2AttrRaw = module->getAttrOfType<ArrayAttr>("pulse.t2_times");
    if (!t2AttrRaw)
      t2AttrRaw = module->getAttrOfType<ArrayAttr>("t2_times");
    if (auto t2Attr = t2AttrRaw) {
      for (int64_t qi = 0; qi < (int64_t)t2Attr.size(); qi++) {
        double t2 = cast<FloatAttr>(t2Attr[qi]).getValueAsDouble();
        double t1 = 0.0;
        auto t1a = module->getAttrOfType<ArrayAttr>("pulse.t1_times");
        if (!t1a)
          t1a = module->getAttrOfType<ArrayAttr>("t1_times");
        if (t1a)
          if (qi < (int64_t)t1a.size())
            t1 = cast<FloatAttr>(t1a[qi]).getValueAsDouble();
        double gammaPhi = 0.0;
        if (t2 > 0) {
          gammaPhi = 1.0 / t2;
          if (t1 > 0)
            gammaPhi -= 1.0 / (2.0 * t1);
          if (gammaPhi < 0)
            gammaPhi = 0;
        }
        if (gammaPhi > 0) {
          // Off-diagonal elements between levels j and k decay at
          // gammaPhi (j - k)^2; for a qubit sqrt(gammaPhi / 2) Z is the same.
          auto dephase =
              ops.levelsOf(qi) == 2
                  ? ops.product({{qi, "spin_z"}}, std::sqrt(gammaPhi / 2.0))
                  : ops.product({{qi, "number"}}, std::sqrt(2.0 * gammaPhi));
          collapseOps.push_back(
              qop::MakeSumOp::create(b, loc, opTy, ValueRange{dephase}));
        }
      }
    }

    // 5. Construct Lindblad super-operator
    qop::LindbladOp::create(b, loc, superOpTy, hamiltonian, collapseOps);

    if (!levels.empty()) {
      int64_t modes = 0;
      if (auto attr = module->getAttrOfType<IntegerAttr>("qop.n_qubits"))
        modes = attr.getInt();
      else if (auto attr = module->getAttrOfType<IntegerAttr>("pulse.n_qubits"))
        modes = attr.getInt();
      for (auto [qudit, d] : levels)
        modes = std::max(modes, qudit + 1);
      SmallVector<int64_t> extents;
      for (int64_t qudit = 0; qudit < modes; ++qudit)
        extents.push_back(ops.levelsOf(qudit));
      module->setAttr("qop.mode_extents", b.getDenseI64ArrayAttr(extents));
    }

    // The QOp graph above is self-contained. Consume the source Pulse
    // program, regions included, so the next conversion stage receives only
    // QOp plus standard dialect ops.
    for (Operation *op : llvm::reverse(sourceOps)) {
      op->dropAllUses();
      op->dropAllReferences();
    }
    for (Operation *op : llvm::reverse(sourceOps))
      op->erase();
  }
};

} // namespace

namespace pulse {

std::unique_ptr<mlir::Pass> createPulseToQOpPass() {
  return std::make_unique<PulseToQOpPass>();
}

} // namespace pulse
