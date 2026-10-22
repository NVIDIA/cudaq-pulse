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
#include <mutex>
#include <thread>

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
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetSelect.h"

#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseTypes.h.inc"

#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseOps.h.inc"

#include "cudaq-pulse/Analysis/Execution.h"
#include "cudaq-pulse/Analysis/LineTrace.h"
#include "cudaq-pulse/Analysis/Timing.h"
#include "cudaq-pulse/Runtime/WaveformCallback.h"

#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatDialect.h.inc"
#include "cudaq-pulse/Dialect/QOp/QOpDialect.h.inc"

namespace nb = nanobind;

// Conversion passes
namespace pulse {
std::unique_ptr<mlir::Pass> createPulseToQOpPass();
}
namespace qop {
std::unique_ptr<mlir::Pass> createQOpToCuDensityMatPass();
}
namespace cudm {
std::unique_ptr<mlir::Pass> createCuDensityMatToLLVMPass();
}
// Pulse Transforms passes
namespace pulse {
std::unique_ptr<mlir::Pass> createPulseVerifyPass();
std::unique_ptr<mlir::Pass> createPulseCanonicalizePass();
std::unique_ptr<mlir::Pass> createVirtualZPass();
std::unique_ptr<mlir::Pass> createPulseFusionPass();
std::unique_ptr<mlir::Pass> createPulseScheduleAsapPass();
std::unique_ptr<mlir::Pass> createPulseScheduleAlapPass();
std::unique_ptr<mlir::Pass> createPulseScheduleRcpPass(int64_t maxDrives,
                                                       int64_t maxReadouts,
                                                       int64_t readoutLatency,
                                                       int64_t switchPenalty,
                                                       bool isAlap);
} // namespace pulse

namespace {

mlir::DialectRegistry makeRegistry() {
  mlir::DialectRegistry reg;
  reg.insert<pulse::PulseDialect>();
  reg.insert<qop::QOpDialect>();
  reg.insert<cudm::CuDensityMatDialect>();
  reg.insert<mlir::arith::ArithDialect>();
  reg.insert<mlir::func::FuncDialect>();
  reg.insert<mlir::scf::SCFDialect>();
  reg.insert<mlir::ub::UBDialect>();
  reg.insert<mlir::LLVM::LLVMDialect>();
  return reg;
}

// Packed-buffer OpCode enum — must match Python packed_emit.py constants
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

inline double i2f(int64_t bits) {
  double d;
  std::memcpy(&d, &bits, sizeof(double));
  return d;
}

// -----------------------------------------------------------------------
// PulseModule: opaque handle around an in-memory mlir::ModuleOp
// -----------------------------------------------------------------------
// TODO(pulse): this wrapper owns the MLIRContext + ModuleOp and maps pass
// names through the if/else chain in run_passes() below. A cleaner design
// (tracked as a follow-up) is to expose MLIR's own MlirModule/MlirPassManager
// and register the pulse passes so PassManager pipeline strings parse
// directly, removing this bespoke wrapper. Deferred here because it threads
// through the Python frontend (compile.py/jit.py/evolve.py) and warrants its
// own change + validation pass.
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

  void run_passes(const std::vector<std::string> &pipeline) {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    mlir::PassManager pm(ctx_.get());
    for (auto &name : pipeline) {
      if (name == "pulse-to-qop")
        pm.addPass(pulse::createPulseToQOpPass());
      else if (name == "qop-to-cudm")
        pm.addPass(qop::createQOpToCuDensityMatPass());
      else if (name == "cudm-to-llvm")
        pm.addPass(cudm::createCuDensityMatToLLVMPass());
      else if (name == "pulse-verify")
        pm.addPass(pulse::createPulseVerifyPass());
      else if (name == "pulse-canonicalize")
        pm.addNestedPass<mlir::func::FuncOp>(
            pulse::createPulseCanonicalizePass());
      else if (name == "pulse-virtual-z")
        pm.addNestedPass<mlir::func::FuncOp>(pulse::createVirtualZPass());
      else if (name == "pulse-fusion")
        pm.addNestedPass<mlir::func::FuncOp>(pulse::createPulseFusionPass());
      else if (name == "pulse-schedule-asap")
        pm.addNestedPass<mlir::func::FuncOp>(
            pulse::createPulseScheduleAsapPass());
      else if (name == "pulse-schedule-alap")
        pm.addNestedPass<mlir::func::FuncOp>(
            pulse::createPulseScheduleAlapPass());
      else if (name == "loop-invariant-code-motion")
        pm.addPass(mlir::createLoopInvariantCodeMotionPass());
      else
        throw std::runtime_error("Unknown pass: " + name);
    }
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
    runPipeline(pm, *module_, "pulse scheduling failed");
  }

  using Event = std::tuple<std::string, int64_t, int64_t,
                           std::optional<int64_t>, std::optional<int64_t>>;

  /// (kind, line, qubit, start_vtu, duration_vtu) for every drive, readout
  /// and wait, once per loop iteration and in program order. Both arms of a
  /// conditional are listed. Lines are numbered in allocation order.
  std::vector<Event> events() {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    llvm::DenseMap<mlir::Operation *, int64_t> lineIndex;
    module_->walk([&](mlir::Operation *op) {
      if (mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(op))
        lineIndex.try_emplace(op, static_cast<int64_t>(lineIndex.size()));
    });
    std::vector<Event> result;
    llvm::SmallVector<int64_t, 2> iteration;
    std::function<void(mlir::Block &)> collect = [&](mlir::Block &block) {
      for (mlir::Operation &op : block) {
        if (mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp>(op)) {
          mlir::Operation *allocation =
              pulse::traceAllocation(op.getOperand(0));
          int64_t line = allocation ? lineIndex.lookup(allocation) : -1;
          int64_t qubit = -1;
          if (allocation)
            if (auto attr =
                    allocation->getAttrOfType<mlir::IntegerAttr>("qubit"))
              qubit = attr.getInt();
          result.emplace_back(op.getName().stripDialect().str(), line, qubit,
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
    walker.onReadout = [&](const pulse::Occurrence &occurrence,
                           const pulse::MeasurementKey &key) {
      if (chosen.count(key))
        return false;
      double end = (occurrence.startVtu + occurrence.durationVtu) / clockGHz;
      if (next && std::get<4>(*next) <= end)
        return false;
      int64_t index =
          std::distance(readouts.begin(), llvm::find(readouts, occurrence.op));
      int64_t qubit = -1;
      if (auto *allocation =
              pulse::traceAllocation(occurrence.op->getOperand(0)))
        if (auto attr = allocation->getAttrOfType<mlir::IntegerAttr>("qubit"))
          qubit = attr.getInt();
      next = Measurement{index, key.iteration, qubit,
                         occurrence.startVtu / clockGHz, end};
      return false;
    };
    auto function = getFuncOp();
    if (!function)
      throw std::runtime_error("no 'main' func in module");
    if (!walker.run(function))
      throw std::runtime_error(walker.error);
    return next;
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
  /// (mode, dimension, column-major interleaved complex matrix).
  OperatorProgram operator_program() {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    auto lowered = mlir::OwningOpRef<mlir::ModuleOp>(module_->clone());
    mlir::PassManager pm(ctx_.get());
    pm.addPass(pulse::createPulseToQOpPass());
    runPipeline(pm, *lowered, "pulse-to-qop failed");

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
    auto term = [&](mlir::Value productValue) -> Term {
      mlir::Operation *product = productValue.getDefiningOp();
      if (!product || product->getName().getStringRef() != "qop.make_product")
        throw std::runtime_error("qop sum term is not a qop.make_product");
      Term result{1.0, 0.0, 0, {}, {}};
      mlir::Operation *scalar = product->getOperand(0).getDefiningOp();
      if (scalar->getName().getStringRef() == "qop.const_scalar") {
        std::get<0>(result) =
            mlir::cast<mlir::FloatAttr>(scalar->getAttr("real"))
                .getValueAsDouble();
        std::get<1>(result) =
            mlir::cast<mlir::FloatAttr>(scalar->getAttr("imag"))
                .getValueAsDouble();
      } else {
        std::get<2>(result) =
            mlir::cast<mlir::IntegerAttr>(scalar->getAttr("cudm.callback_kind"))
                .getInt();
        auto params = mlir::cast<mlir::DenseF64ArrayAttr>(
            scalar->getAttr("cudm.callback_params"));
        std::get<3>(result).assign(params.asArrayRef().begin(),
                                   params.asArrayRef().end());
      }
      for (mlir::Value factor : product->getOperands().drop_front()) {
        mlir::Operation *leaf = factor.getDefiningOp();
        if (leaf->getName().getStringRef() == "qop.spin") {
          std::string kind;
          llvm::raw_string_ostream os(kind);
          leaf->getAttr("kind").print(os);
          llvm::StringRef name(kind);
          name = name.substr(name.rfind(' ') + 1).rtrim('>');
          std::get<4>(result).emplace_back(constantI64(leaf->getOperand(0)), 2,
                                           spinData(name));
        } else {
          auto targets =
              mlir::cast<mlir::DenseI64ArrayAttr>(leaf->getAttr("targets"));
          auto dims =
              mlir::cast<mlir::DenseI64ArrayAttr>(leaf->getAttr("dimensions"));
          mlir::Attribute data;
          if (!mlir::matchPattern(leaf->getOperand(0), mlir::m_Constant(&data)))
            throw std::runtime_error("qop matrix leaf data is not a constant");
          auto values = mlir::cast<mlir::DenseFPElementsAttr>(data);
          std::get<4>(result).emplace_back(
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
    auto sumTerms = [&](mlir::Value sum) {
      std::vector<Term> terms;
      for (mlir::Value product : sum.getDefiningOp()->getOperands())
        terms.push_back(term(product));
      return terms;
    };
    std::vector<Term> hamiltonian = sumTerms(lindblad->getOperand(0));
    std::vector<Term> collapse;
    for (mlir::Value op : lindblad->getOperands().drop_front())
      for (Term &t : sumTerms(op))
        collapse.push_back(std::move(t));

    mlir::Operation *top = lowered->getOperation();
    std::vector<int64_t> extents;
    if (auto attr =
            top->getAttrOfType<mlir::DenseI64ArrayAttr>("qop.mode_extents"))
      extents.assign(attr.asArrayRef().begin(), attr.asArrayRef().end());
    else {
      int64_t modes = 1;
      if (auto attr = top->getAttrOfType<mlir::IntegerAttr>("qop.n_qubits"))
        modes = attr.getInt();
      else if (auto attr =
                   top->getAttrOfType<mlir::IntegerAttr>("pulse.n_qubits"))
        modes = attr.getInt();
      for (const Term &t : hamiltonian)
        for (const Factor &f : std::get<4>(t))
          modes = std::max(modes, std::get<0>(f) + 1);
      extents.assign(static_cast<size_t>(modes), 2);
    }
    auto number = [&](llvm::StringRef name, double fallback) {
      if (auto attr = top->getAttrOfType<mlir::FloatAttr>(name))
        return attr.getValueAsDouble();
      return fallback;
    };
    int64_t steps = 100;
    if (auto attr = top->getAttrOfType<mlir::IntegerAttr>("qop.num_steps"))
      steps = attr.getInt();
    std::string integrator = "rk4";
    if (auto attr = top->getAttrOfType<mlir::StringAttr>("qop.integrator"))
      integrator = attr.getValue().str();
    return {extents,
            hamiltonian,
            collapse,
            number("qop.t_start", 0.0),
            number("qop.t_end", 0.0),
            steps,
            integrator};
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

  bool is_parametric() {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    auto funcOp = getFuncOp();
    return funcOp && funcOp.getNumArguments() > 0;
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
    auto clonedModule = module_->clone();
    auto funcOp = clonedModule.lookupSymbol<mlir::func::FuncOp>("main");
    if (!funcOp)
      throw std::runtime_error("specialize: no 'main' func in module");

    auto &entryBlock = funcOp.getBody().front();
    auto loc = funcOp.getLoc();
    mlir::OpBuilder builder(ctx_.get());
    builder.setInsertionPointToStart(&entryBlock);

    size_t f64Idx = 0, i64Idx = 0;
    auto f64Ty = builder.getF64Type();

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
      }
      arg.replaceAllUsesWith(replacement);
    }

    // Remove block arguments (make func take no args)
    while (entryBlock.getNumArguments() > 0)
      entryBlock.eraseArgument(entryBlock.getNumArguments() - 1);
    funcOp.setType(builder.getFunctionType({}, {}));

    // Fold parameter expressions before deriving concrete durations.
    mlir::PassManager pm(ctx_.get());
    pm.addPass(mlir::createCanonicalizerPass());
    if (schedule == "asap")
      pm.addNestedPass<mlir::func::FuncOp>(
          pulse::createPulseScheduleAsapPass());
    else if (schedule == "alap")
      pm.addNestedPass<mlir::func::FuncOp>(
          pulse::createPulseScheduleAlapPass());
    else if (schedule == "rcp" || schedule == "alap_rcp")
      pm.addNestedPass<mlir::func::FuncOp>(pulse::createPulseScheduleRcpPass(
          maxDrives, maxReadouts, readoutLatency, switchPenalty,
          schedule == "alap_rcp"));
    else
      throw std::invalid_argument("unsupported specialization schedule: " +
                                  schedule);
    runPipeline(pm, clonedModule, "specialize: pass pipeline failed");

    auto owning = mlir::OwningOpRef<mlir::ModuleOp>(clonedModule);
    return PulseModule(ctx_, std::move(owning));
  }

private:
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

  std::shared_ptr<mlir::MLIRContext> ctx_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
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

    auto bind = [&](int64_t vid, mlir::Value v) { ssa[vid] = v; };
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
    };
    std::vector<Region> regions;

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
        builder_->setInsertionPointToEnd(region.body.get());
        regions.push_back(std::move(region));
        break;
      }
      case kOpElse: {
        requirePayload(0);
        if (regions.empty() || regions.back().isLoop || regions.back().elseBody)
          throw std::invalid_argument("packed: else outside an if");
        regions.back().elseBody = std::make_unique<mlir::Block>();
        builder_->setInsertionPointToEnd(regions.back().elseBody.get());
        break;
      }
      case kOpIfEnd: {
        if (regions.empty() || regions.back().isLoop || plen < 1 || p[0] < 0 ||
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
        auto condition = pulse::IsSetOp::create(
            *builder_, loc, builder_->getI1Type(), region.condition);
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
          driveOp->setAttr("start_vtu", builder_->getI64IntegerAttr(p[5]));
        if (p[6] != -1)
          driveOp->setAttr("duration_vtu", builder_->getI64IntegerAttr(p[6]));
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
class JITModule {
public:
  explicit JITModule(std::unique_ptr<mlir::ExecutionEngine> engine)
      : engine_(std::move(engine)) {}

  uintptr_t lookup(const std::string &name) {
    auto symbol = engine_->lookup(name);
    if (!symbol) {
      std::string message;
      llvm::raw_string_ostream os(message);
      os << symbol.takeError();
      throw std::runtime_error("JIT symbol lookup failed for '" + name +
                               "': " + message);
    }
    return reinterpret_cast<uintptr_t>(*symbol);
  }

private:
  std::unique_ptr<mlir::ExecutionEngine> engine_;
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

  llvm::SmallVector<llvm::StringRef> libPaths(shared_libs.begin(),
                                              shared_libs.end());
  mlir::ExecutionEngineOptions options;
  options.transformer = mlir::makeOptimizingTransformer(
      /*optLevel=*/2, /*sizeLevel=*/0, /*targetMachine=*/nullptr);
  options.sharedLibPaths = libPaths;
  auto engine = mlir::ExecutionEngine::create(module, options);
  if (!engine) {
    std::string message;
    llvm::raw_string_ostream os(message);
    os << engine.takeError();
    throw std::runtime_error("JIT compilation failed: " + message);
  }
  return JITModule(std::move(*engine));
}

// -----------------------------------------------------------------------
// MLIRPipeline: text-in / text-out convenience wrapper (legacy API)
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

  std::string parse_and_print(const std::string &mlir_text) {
    mlir::MLIRContext ctx;
    ctx.appendDialectRegistry(registry_);
    ctx.loadAllAvailableDialects();
    auto module = parseModule(mlir_text, &ctx, "Failed to parse MLIR text");
    std::string result;
    llvm::raw_string_ostream os(result);
    module->print(os);
    return result;
  }

  std::string run_full_pipeline(const std::string &pulse_mlir) {
    mlir::MLIRContext ctx;
    ctx.appendDialectRegistry(registry_);
    ctx.loadAllAvailableDialects();
    auto module =
        parseModule(pulse_mlir, &ctx, "Failed to parse pulse MLIR text");
    mlir::PassManager pm(&ctx);
    addPulseLoweringPasses(pm);
    runPipeline(pm, *module, "MLIR pass pipeline failed");
    std::string result;
    llvm::raw_string_ostream os(result);
    module->print(os);
    return result;
  }

  std::string run_full_pipeline_to_llvm_ir(const std::string &pulse_mlir) {
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

    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);
    if (!llvmModule)
      throw std::runtime_error("Failed to translate LLVM-dialect MLIR");
    std::string result;
    llvm::raw_string_ostream os(result);
    llvmModule->print(os, nullptr);
    return result;
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

  std::string run_cudm_to_llvm(const std::string &cudm_mlir) {
    mlir::MLIRContext ctx;
    ctx.appendDialectRegistry(registry_);
    ctx.loadAllAvailableDialects();
    auto module =
        parseModule(cudm_mlir, &ctx, "Failed to parse cudm MLIR text");
    mlir::PassManager pm(&ctx);
    pm.addPass(cudm::createCuDensityMatToLLVMPass());
    runPipeline(pm, *module, "cudm-to-llvm pass failed");
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
      .def("events", &PulseModule::events,
           nb::call_guard<nb::gil_scoped_release>(),
           "(kind, line, qubit, start_vtu, duration_vtu) of each timed op")
      .def("next_measurement", &PulseModule::next_measurement,
           nb::call_guard<nb::gil_scoped_release>(),
           "The next measurement without an outcome on a path", nb::arg("path"))
      .def("operator_program", &PulseModule::operator_program,
           nb::call_guard<nb::gil_scoped_release>(),
           "The lowered Hamiltonian and collapse operators as data")
      .def("set_attributes", &PulseModule::set_attributes,
           nb::call_guard<nb::gil_scoped_release>(),
           "Merge an MLIR dictionary attribute into the module attributes",
           nb::arg("dictionary"))
      .def("is_parametric", &PulseModule::is_parametric,
           nb::call_guard<nb::gil_scoped_release>(),
           "True if the module has func.func block arguments (parameters)")
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
      .def("parse_and_print", &MLIRPipeline::parse_and_print,
           nb::call_guard<nb::gil_scoped_release>(),
           "Parse MLIR text and print it back (roundtrip)")
      .def("run_full_pipeline", &MLIRPipeline::run_full_pipeline,
           nb::call_guard<nb::gil_scoped_release>(),
           "Run full pipeline: pulse -> qop -> cudm -> llvm")
      .def("run_full_pipeline_to_llvm_ir",
           &MLIRPipeline::run_full_pipeline_to_llvm_ir,
           nb::call_guard<nb::gil_scoped_release>(),
           "Run the full pipeline and translate to textual LLVM IR")
      .def("run_pulse_to_qop", &MLIRPipeline::run_pulse_to_qop,
           nb::call_guard<nb::gil_scoped_release>(),
           "Run pulse-to-qop pass only")
      .def("run_qop_to_cudm", &MLIRPipeline::run_qop_to_cudm,
           nb::call_guard<nb::gil_scoped_release>(),
           "Run qop-to-cudm pass only")
      .def("run_cudm_to_llvm", &MLIRPipeline::run_cudm_to_llvm,
           nb::call_guard<nb::gil_scoped_release>(),
           "Run cudm-to-llvm pass only")
      .def("parse_to_module", &MLIRPipeline::parse_to_module,
           nb::call_guard<nb::gil_scoped_release>(),
           "Parse MLIR text into a PulseModule");
}
