//===- ModuleLoader.cpp - A function of another compiler's module ---------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "ModuleLoader.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include <algorithm>
#include <array>
#include <initializer_list>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

// The dialects' generated classes, in the order their definitions need,
// and after them the header that reads the timed operations.
// clang-format off
#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"
#include "cudaq-pulse/Dialect/Pulse/PulseEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseTypes.h.inc"

#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseAttrs.h"
#include "cudaq-pulse/Dialect/Pulse/PulseOps.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseCallOp.h.inc"

#include "cudaq-pulse/Dialect/Classical/ClassicalDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Classical/ClassicalTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Classical/ClassicalOps.h.inc"

#include "cudaq-pulse/Analysis/LineTrace.h"
#include "cudaq-pulse/Analysis/Timing.h"
#include "cudaq-pulse/Transforms/Passes.h"

namespace pulse {
/// The rules a function a pulse.call runs keeps (PulseCallCheck.cpp).
mlir::LogicalResult checkCalledFunction(mlir::func::FuncOp callee);
} // namespace pulse
// clang-format on

namespace cudaq_pulse::loader {
namespace {

/// Collects the error diagnostics this thread emits while it is alive, so
/// that a refusal carries them; another thread's go on to its own handler.
class DiagnosticCollector {
public:
  explicit DiagnosticCollector(mlir::MLIRContext *ctx)
      : owner_(std::this_thread::get_id()),
        handler_(ctx, [this](mlir::Diagnostic &diagnostic) {
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

class Loader {
public:
  explicit Loader(mlir::MLIRContext *ctx) : ctx_(ctx), builder_(ctx) {}

  /// The scheduled function `entry` of the module `text` (assembly or
  /// bytecode), as a module of its own whose `main` takes the function's
  /// qudits as slots, in argument order: what place_carriers() binds. The
  /// function's results and its attributes other than its schedule state are
  /// dropped.
  /// A module another compiler wrote, parsed and verified by MLIR as it is
  /// given: nothing here bounds the depth of that parse, text or bytecode.
  ///
  /// After parsing, the module's regions are measured without recursion,
  /// the module's own body counting as the first: one nested deeper than
  /// kMostNesting is refused before anything walks or copies it (254
  /// branches or loops nested in a function are the most that load), and
  /// so is any operation with a symbol table of its own below the module,
  /// so every symbol is one of the module's own table.
  static constexpr int kMostNesting = 256;
  static mlir::OwningOpRef<mlir::ModuleOp>
  parseForeign(const std::string &text, mlir::MLIRContext *ctx) {
    auto source = parseModule(text, ctx, "load: cannot read the module");
    // Depth first with an explicit stack: (operation, depth of its regions).
    llvm::SmallVector<std::pair<mlir::Operation *, int>> open;
    for (mlir::Operation &op : *source->getBody())
      open.emplace_back(&op, 1);
    while (!open.empty()) {
      auto [op, depth] = open.pop_back_val();
      if (op->hasTrait<mlir::OpTrait::SymbolTable>())
        throw std::invalid_argument(
            "load: the module holds '" + op->getName().getStringRef().str() +
            "', which has symbols of its own (a module inside the module); "
            "a loaded module is one module of functions");
      if (op->getNumRegions() && depth >= kMostNesting)
        throw std::invalid_argument(
            "load: the module nests regions deeper than " +
            std::to_string(kMostNesting) + "; write it with less nesting");
      for (mlir::Region &region : op->getRegions())
        for (mlir::Block &block : region)
          for (mlir::Operation &inner : block)
            open.emplace_back(&inner, depth + 1);
    }
    return source;
  }
  /// The most operations the calls of a loaded function may write in place,
  /// and the deepest its calls may nest.
  static constexpr int64_t kMostWritten = int64_t(1) << 20;
  static constexpr size_t kMostCallDepth = 256;
  /// What a function of a loaded module may not hold, whether it is the
  /// entry or a function the entry calls: a qudit of its own (the program's
  /// qudits are the entry's arguments), a function taken as a value, a time
  /// outside the range a schedule counts in. A called function also holds
  /// no region but a branch, and selects no value at run time.
  static void refuseUnsupported(mlir::func::FuncOp function, bool called,
                                bool written = false) {
    std::string name = function.getSymName().str();
    if (called && function->hasAttr("pulse.selected_values"))
      throw std::invalid_argument(
          "load: " + name +
          " selects values at run time (pulse.selected_values), which a "
          "loaded function cannot do yet");
    function.walk([&](mlir::Operation *op) {
      if (op == function.getOperation())
        return;
      llvm::StringRef kind = op->getName().getStringRef();
      if (kind == "pulse.qudit_alloc")
        throw std::invalid_argument(
            "load: " + name +
            " allocates a qudit; a loaded function plays on the qudits it "
            "takes");
      if (kind == "func.call")
        throw std::invalid_argument(
            "load: " + name +
            " calls a function with func.call; a loaded function runs a "
            "function scheduled on its own with pulse.call");
      if (kind == "func.constant" || kind == "func.call_indirect")
        throw std::invalid_argument(
            "load: " + name +
            " takes a function as a value; a loaded function calls the "
            "module's functions by name");
      if (called && op->getNumRegions() && !mlir::isa<mlir::scf::IfOp>(op))
        throw std::invalid_argument(
            "load: " + name +
            " has a loop or a schedule block; a called function is written "
            "without them");
      constexpr int64_t most = int64_t(1) << 50;
      std::optional<int64_t> start = pulse::startCycles(op);
      std::optional<int64_t> duration = pulse::durationCycles(op);
      if ((start && (*start < 0 || *start > most)) ||
          (duration && (*duration < 0 || *duration > most)) ||
          llvm::any_of(pulse::periodCycles(op), [](int64_t period) {
            return period < 0 || period > most;
          }))
        throw std::invalid_argument(
            "load: " + name +
            " has an operation whose start, duration or period is outside 0 "
            "to 2^50 cycles" +
            (written ? " once its calls are written out" : ""));
      // In loops an operation starts a period later each trip: its last
      // start is held to the range too. That takes the trips of every loop
      // around it, whatever is between a loop and the operation: an
      // scf.for that counts signed, with bounds and a step that are
      // constants of at most 2^50 either way, so that counting its trips
      // cannot overflow here or in any pass; and no other kind of loop.
      if (!start)
        return;
      auto unbounded = [&](const char *why) {
        return std::invalid_argument(
            "load: " + name + " has a timed operation in " + why +
            (written ? " once its calls are written out" : ""));
      };
      llvm::SmallVector<int64_t, 2> trips;
      for (mlir::Operation *parent = op->getParentOp();
           parent && parent != function.getOperation();
           parent = parent->getParentOp()) {
        // Met before the loop or block that holds it (the walk is
        // post-order): a called function is refused for that, as below.
        if (called && !mlir::isa<mlir::scf::IfOp>(parent))
          throw std::invalid_argument(
              "load: " + name +
              " has a loop or a schedule block; a called function is written "
              "without them");
        // What reads a loaded function takes an operation in a schedule
        // block with every loop around the block, as one in a branch.
        if (mlir::isa<mlir::scf::IfOp, pulse::ScheduleBlockOp>(parent))
          continue;
        auto loop = mlir::dyn_cast<mlir::scf::ForOp>(parent);
        if (!loop)
          throw unbounded("a region that is neither a branch nor an scf.for "
                          "loop; a loaded function repeats with scf.for");
        auto within = [&](mlir::Value bound) {
          // An index: a wider constant would be cut to 64 bits.
          std::optional<int64_t> value = bound.getType().isIndex()
                                             ? pulse::constantIndex(bound)
                                             : std::nullopt;
          return value && *value >= -most && *value <= most;
        };
        // The bounds first: the count of bounds outside them can overflow.
        std::optional<int64_t> count;
        if (!loop.getUnsignedCmp() && within(loop.getLowerBound()) &&
            within(loop.getUpperBound()) && within(loop.getStep()))
          count = pulse::constantTripCount(loop);
        if (!count)
          throw unbounded(
              "a loop whose bounds are not constants; a loaded loop counts "
              "signed and gives its bounds and its step as index constants "
              "of at most 2^50, the step above 0");
        trips.push_back(*count);
      }
      // Outermost first, as the periods are.
      std::reverse(trips.begin(), trips.end());
      llvm::ArrayRef<int64_t> periods = pulse::periodCycles(op);
      int64_t last = *start;
      for (auto [count, period] : llvm::zip(trips, periods)) {
        // Within the range, `last` and `period` cannot overflow the sum.
        if (count > 1 && period > 0 && count - 1 > (most - last) / period)
          throw std::invalid_argument(
              "load: " + name +
              " has an operation in a loop whose last trip starts past 2^50 "
              "cycles" +
              (written ? " once its calls are written out" : ""));
        if (count > 1)
          last += (count - 1) * period;
      }
    });
  }
  /// Whether `attribute` of `op` is one of the two phases a drive is read
  /// with beside its tone's frame: `frame_phase_offset`, a constant phase
  /// shift folded into the drive (pulse-fold-phases writes it), which stays
  /// in the tone's frame, and `phase_offset`, the drive's own, which no
  /// pass sets (fusing two drives carries one over) and the export and the
  /// simulator read. Each is a finite f64 on a pulse.drive; under these
  /// names anything else, or on any other operation, is another compiler's
  /// attribute.
  static bool isDrivePhase(mlir::Operation *op,
                           mlir::NamedAttribute attribute) {
    llvm::StringRef name = attribute.getName().getValue();
    if (name != "frame_phase_offset" && name != "phase_offset")
      return false;
    auto phase = mlir::dyn_cast<mlir::FloatAttr>(attribute.getValue());
    return phase && mlir::isa<pulse::DriveOp>(op) && phase.getType().isF64() &&
           phase.getValue().isFinite();
  }
  /// The operations of `function` that compute from the answer of a
  /// classical call, through operations without regions: what the verifier
  /// takes for a value that depends on an answer.
  static llvm::DenseSet<mlir::Operation *>
  computedFromAnswers(mlir::func::FuncOp function) {
    llvm::DenseSet<mlir::Operation *> computed;
    llvm::SmallVector<mlir::Operation *, 16> open;
    function.walk([&](classical::WaitOp answer) { open.push_back(answer); });
    while (!open.empty())
      for (mlir::Operation *user : open.pop_back_val()->getUsers())
        if (user->getNumRegions() == 0 && computed.insert(user).second)
          open.push_back(user);
    return computed;
  }
  /// What telling such a wait takes: the function, its name for a refusal,
  /// and the operations computed from an answer, found when first needed.
  struct Marked {
    mlir::func::FuncOp function;
    const std::string &entry;
    std::optional<llvm::DenseSet<mlir::Operation *>> computed;
  };
  /// Whether `op`, which carries the mark of a wait whose length is the
  /// answer of a classical call (`classical.variable`), is such a wait: a
  /// pulse.wait whose length is computed from an answer. The verifier does
  /// not hold a marked operation to the length it takes, since the answer
  /// gives it while the program runs, so the mark is kept only where an
  /// answer does: on any other operation it is another compiler's
  /// attribute, and a drive, a readout or a wait of constant length is then
  /// verified for the length it has. A wait that keeps the mark states its
  /// lowest length (`pulse.min_cycles`, a 64-bit integer) and reserves
  /// that, as the scheduler places it.
  [[gnu::cold, gnu::noinline]] static bool isRunTimeWait(mlir::Operation *op,
                                                         Marked &marked) {
    const std::string &entry = marked.entry;
    std::optional<llvm::DenseSet<mlir::Operation *>> &computed =
        marked.computed;
    mlir::func::FuncOp function = marked.function;
    auto wait = mlir::dyn_cast<pulse::WaitOp>(op);
    if (!wait)
      return false;
    if (!computed)
      computed = computedFromAnswers(function);
    mlir::Operation *length = wait.getDuration().getDefiningOp();
    if (!length || !computed->contains(length))
      return false;
    auto lowest = op->getAttrOfType<mlir::IntegerAttr>("pulse.min_cycles");
    std::optional<int64_t> reserved = pulse::durationCycles(op);
    if (!lowest || !lowest.getType().isSignlessInteger(64))
      throw std::invalid_argument(
          "load: " + entry +
          " has a wait whose length is the answer of a classical call "
          "(classical.variable) without a lowest length; such a wait is a "
          "pulse.variable_wait that states it in pulse.min_cycles, a 64-bit "
          "integer");
    if (reserved && *reserved != lowest.getInt())
      throw std::invalid_argument(
          "load: " + entry +
          " has a wait whose length is the answer of a classical call "
          "(classical.variable) with duration_cycles = " +
          std::to_string(*reserved) +
          " and pulse.min_cycles = " + std::to_string(lowest.getInt()) +
          "; such a wait reserves its lowest length, and what follows it "
          "starts after that");
    return true;
  }
  /// The function `caller` calls by `name`, checked to be one a call can be
  /// written in place of: defined, scheduled, one block that ends in a
  /// return, and no loop, schedule block or qudit of its own.
  static mlir::func::FuncOp calledFunction(mlir::ModuleOp source,
                                           const std::string &caller,
                                           const std::string &name) {
    auto callee = source.lookupSymbol<mlir::func::FuncOp>(name);
    if (!callee || callee.isDeclaration())
      throw std::invalid_argument("load: " + caller + " calls " + name +
                                  ", which the module does not define");
    if (pulse::scheduleStateOf(callee) != pulse::ScheduleState::Timed)
      throw std::invalid_argument("load: " + caller + " calls " + name +
                                  ", which is not scheduled");
    if (!llvm::hasSingleElement(callee.getBody()))
      throw std::invalid_argument("load: " + name + " has more than one block");
    if (callee.getBody().front().empty() ||
        !mlir::isa<mlir::func::ReturnOp>(callee.getBody().front().back()))
      throw std::invalid_argument("load: " + name +
                                  " does not end in a return");
    refuseUnsupported(callee, /*called=*/true);
    return callee;
  }
  /// A call states when it starts (the schedule passes place none), and one
  /// period per loop around it, as every timed operation there has. A
  /// function a call runs is `called`: it has no loop, and its calls take the
  /// periods of the call it is written for.
  static void refuseUnplaced(pulse::CallOp call, const std::string &caller,
                             bool called) {
    std::string callee = call.getCallee().str();
    if (!call.getStartCycles())
      throw std::invalid_argument(
          "load: " + caller + " calls " + callee +
          " without stating when it starts; a call in a loaded function "
          "states its start in cycles");
    // The range every timed operation of a loaded function is held to.
    constexpr int64_t most = int64_t(1) << 50;
    auto outside = [&](int64_t cycles) { return cycles < 0 || cycles > most; };
    if (outside(static_cast<int64_t>(*call.getStartCycles())) ||
        outside(static_cast<int64_t>(call.getDurationCycles().value_or(0))) ||
        llvm::any_of(call.getPeriodCycles(), outside))
      throw std::invalid_argument(
          "load: " + caller + " calls " + callee +
          " with a start, duration or period outside 0 to 2^50 cycles");
    size_t periods = call.getPeriodCycles().size();
    if (called && periods)
      throw std::invalid_argument(
          "load: " + caller + " calls " + callee +
          " with a period_cycles; a called function has no loop, and its "
          "calls take the periods of the call it is written for");
    size_t loops = 0;
    for (mlir::Operation *parent = call->getParentOp();
         parent && !mlir::isa<mlir::func::FuncOp>(parent);
         parent = parent->getParentOp())
      loops += mlir::isa<mlir::scf::ForOp>(parent);
    if (periods != loops)
      throw std::invalid_argument(
          "load: " + caller + " calls " + callee + " inside " +
          std::to_string(loops) + " scf.for loop(s) and states " +
          std::to_string(periods) +
          " period(s); give period_cycles one period per scf.for around the "
          "call, outermost first (array<i64: outer, ..., inner>), and none "
          "outside a loop");
  }
  /// What a call of a function puts in its place.
  struct Written {
    /// Its operations and those of the functions it calls, counted up to
    /// just past the most a load writes.
    int64_t operations = 0;
    /// The calls below it: 1 for a function that calls none.
    size_t calls = 1;
    /// The branches around its innermost operation once its own calls are
    /// written out.
    int nesting = 0;
  };
  /// How many branches and loops of its function are around `op`.
  static int regionsAround(mlir::Operation *op) {
    int around = 0;
    for (mlir::Operation *parent = op->getParentOp();
         parent && !mlir::isa<mlir::func::FuncOp>(parent);
         parent = parent->getParentOp())
      ++around;
    return around;
  }
  /// The calls of `first` nest deeper than a load follows.
  [[noreturn]] static void refuseCallDepth(const std::string &first) {
    throw std::invalid_argument("load: the calls of " + first +
                                " nest deeper than " +
                                std::to_string(kMostCallDepth));
  }
  /// Written out, the calls of `first` nest regions deeper than a loaded
  /// module may: the bound parseForeign holds the module to, for the
  /// function its calls make.
  [[noreturn]] static void refuseNesting(const std::string &first) {
    throw std::invalid_argument("load: written in its place, the calls of " +
                                first + " nest regions deeper than " +
                                std::to_string(kMostNesting) +
                                "; write them with less nesting");
  }
  /// What a call of `name` writes in its place. `open` holds the functions
  /// being counted: one met again calls itself. A function counted before
  /// is not counted again, and is held to the same bounds from what was
  /// kept of it: how deep its calls go below it, and how deep they nest.
  static Written writtenBy(mlir::ModuleOp source, const std::string &caller,
                           const std::string &name,
                           llvm::StringMap<Written> &known,
                           llvm::SmallVectorImpl<std::string> &open) {
    auto found = known.find(name);
    if (found != known.end()) {
      if (open.size() + found->second.calls > kMostCallDepth + 1)
        refuseCallDepth(open.front());
      return found->second;
    }
    if (llvm::is_contained(open, name))
      throw std::invalid_argument("load: " + name +
                                  " calls a function that calls itself");
    if (open.size() > kMostCallDepth)
      refuseCallDepth(open.front());
    mlir::func::FuncOp callee = calledFunction(source, caller, name);
    open.push_back(name);
    Written written;
    // The callee's own calls are counted after the walk, not from inside
    // it: the stack then grows with the calls' nesting only.
    llvm::SmallVector<pulse::CallOp> nestedCalls;
    callee.walk([&](mlir::Operation *op) {
      if (op == callee.getOperation())
        return;
      written.operations = std::min(written.operations + 1, kMostWritten + 1);
      if (auto nested = mlir::dyn_cast<pulse::CallOp>(op))
        nestedCalls.push_back(nested);
      else if (op->getNumRegions())
        written.nesting = std::max(written.nesting, regionsAround(op) + 1);
    });
    for (pulse::CallOp nested : nestedCalls) {
      refuseUnplaced(nested, name, /*called=*/true);
      Written below =
          writtenBy(source, name, nested.getCallee().str(), known, open);
      written.operations =
          std::min(written.operations + below.operations, kMostWritten + 1);
      written.calls = std::max(written.calls, below.calls + 1);
      written.nesting =
          std::max(written.nesting, regionsAround(nested) + below.nesting);
      // The module's body and the function's are two of the regions.
      if (written.nesting > kMostNesting - 2)
        refuseNesting(open.front());
    }
    open.pop_back();
    return known[name] = written;
  }
  /// Writes every pulse.call of `main` out (pulse::expandCalls), with the
  /// functions the calls run copied from `source` for it. What the calls
  /// would write is counted before any is written: a function that calls
  /// another twice doubles at every level.
  /// Whether `main` held a call.
  bool writeCalls(mlir::ModuleOp source, mlir::ModuleOp module,
                  mlir::func::FuncOp main, const std::string &entry) {
    llvm::SmallVector<pulse::CallOp> calls;
    main.walk([&](pulse::CallOp call) { calls.push_back(call); });
    if (calls.empty())
      return false;
    llvm::StringMap<Written> counts;
    int64_t written = 0;
    for (pulse::CallOp call : calls) {
      refuseUnplaced(call, entry, /*called=*/false);
      llvm::SmallVector<std::string> open{entry};
      Written below =
          writtenBy(source, entry, call.getCallee().str(), counts, open);
      if (regionsAround(call) + below.nesting > kMostNesting - 2)
        refuseNesting(entry);
      written += below.operations;
      if (written > kMostWritten)
        throw std::invalid_argument(
            "load: the calls of " + entry + " write more than " +
            std::to_string(kMostWritten) + " operations in its place");
    }
    for (const auto &counted : counts) {
      if (counted.getKey() == "main")
        throw std::invalid_argument(
            "load: " + entry +
            " calls a function named main, the name the "
            "loaded function takes");
      module.push_back(
          source.lookupSymbol<mlir::func::FuncOp>(counted.getKey())->clone());
    }
    DiagnosticCollector diagnostics(ctx_);
    // Each called function keeps the rules for one (its results continue
    // its arguments in order, its recorded length is where its last
    // operation ends) before any call is written out by them.
    bool kept = true;
    for (auto function : module.getOps<mlir::func::FuncOp>())
      if (function != main)
        kept &= mlir::succeeded(pulse::checkCalledFunction(function));
    if (!kept || mlir::failed(pulse::expandCalls(module, /*callSites=*/false)))
      diagnostics.fail("load: the calls of " + entry +
                       " cannot be written in its place");
    // The loaded module is the entry alone: a called function that
    // something still names (an attribute of another compiler) goes too.
    for (auto function :
         llvm::make_early_inc_range(module.getOps<mlir::func::FuncOp>()))
      if (function != main)
        function.erase();
    return true;
  }

  /// The name the module gives each link it states (`classical.links`), by
  /// link id; a link without a name is absent.
  std::map<int64_t, std::string> load_link_names(const std::string &text) {
    auto source = parseForeign(text, ctx_);
    std::map<int64_t, std::string> out;
    auto links = source->getOperation()->getAttrOfType<mlir::DictionaryAttr>(
        "classical.links");
    if (!links)
      return out;
    for (mlir::NamedAttribute entry : links) {
      int64_t id = 0;
      auto facts = mlir::dyn_cast<mlir::DictionaryAttr>(entry.getValue());
      if (!facts || entry.getName().getValue().getAsInteger(10, id))
        continue;
      if (auto name = facts.getAs<mlir::StringAttr>("name"))
        if (!name.getValue().empty())
          out[id] = name.getValue().str();
    }
    return out;
  }

  using LoadedSubmission =
      std::tuple<std::string, int64_t, int64_t,
                 std::vector<std::optional<int64_t>>, bool>;
  /// Every submission of a module (assembly or bytecode): the function it
  /// calls, the cycles its schedule gives it, the cycles its link takes to
  /// send it as the classical check wrote them on it (-1 when the check
  /// wrote none), and its leading 64-bit integer arguments, each the
  /// constant it is or nothing; and whether its future goes anywhere but
  /// into a wait (it is yielded from a branch or a loop, or passed on), so
  /// that the wait that takes its answer does not name it directly.
  std::vector<LoadedSubmission> load_submissions(const std::string &text) {
    auto source = parseForeign(text, ctx_);
    std::vector<LoadedSubmission> out;
    source->walk([&](classical::SubmitOp submit) {
      auto needed =
          submit->getAttrOfType<mlir::IntegerAttr>("classical.submit_cycles");
      std::vector<std::optional<int64_t>> words;
      // Its first operand is the link; a message's words come next.
      for (mlir::Value operand : submit->getOperands().drop_front()) {
        if (!operand.getType().isInteger(64) || words.size() == kMostWords)
          break;
        llvm::APInt word;
        words.push_back(mlir::matchPattern(operand, mlir::m_ConstantInt(&word))
                            ? std::optional<int64_t>(word.getSExtValue())
                            : std::nullopt);
      }
      out.emplace_back(
          submit.getCallee().str(), pulse::durationCycles(submit).value_or(-1),
          needed ? needed.getInt() : int64_t(-1), std::move(words),
          !llvm::all_of(submit.getFuture().getUsers(),
                        [](mlir::Operation *user) {
                          return mlir::isa<classical::WaitOp>(user);
                        }));
    });
    return out;
  }
  /// The most leading integer arguments read per submission: a decoder
  /// message has four.
  static constexpr size_t kMostWords = 4;

  using LoadedCallable =
      std::tuple<std::string, int64_t, int64_t, std::vector<std::string>,
                 std::vector<std::string>, std::array<double, 3>, std::string,
                 std::string, std::optional<double>, std::optional<double>,
                 std::optional<double>>;
  /// The classical functions a module declares, in its order: name, dispatch
  /// key, link, argument and result descriptors, (min, nominal, max) latency
  /// in ns, mode, submit policy, timeout, deadline and reaction deadline.
  std::vector<LoadedCallable> load_callables(const std::string &text) {
    auto source = parseForeign(text, ctx_);
    auto strings = [](mlir::ArrayAttr values) {
      std::vector<std::string> out;
      for (mlir::Attribute value : values)
        out.push_back(mlir::cast<mlir::StringAttr>(value).getValue().str());
      return out;
    };
    auto number = [](std::optional<llvm::APFloat> value) {
      return value ? std::optional<double>(value->convertToDouble())
                   : std::nullopt;
    };
    std::vector<LoadedCallable> out;
    for (mlir::Operation &op : *source->getBody())
      if (auto fn = mlir::dyn_cast<classical::CallableOp>(op))
        out.emplace_back(
            fn.getSymName().str(), static_cast<int64_t>(fn.getDispatchKey()),
            static_cast<int64_t>(fn.getDevice()), strings(fn.getArgs()),
            strings(fn.getResults()),
            std::array<double, 3>{fn.getLatencyMinNs().convertToDouble(),
                                  fn.getLatencyNominalNs().convertToDouble(),
                                  fn.getLatencyMaxNs().convertToDouble()},
            fn.getMode().str(), fn.getSubmit().str(), number(fn.getTimeoutNs()),
            number(fn.getDeadlineNs()), number(fn.getReactionDeadlineNs()));
    return out;
  }

  mlir::OwningOpRef<mlir::ModuleOp>
  load_function(const std::string &text, const std::string &entry,
                double clock_ghz, int64_t qudits,
                const std::map<std::string, std::array<double, 3>> &latencies,
                const std::map<std::string, int64_t> &keys,
                const std::map<int64_t, int64_t> &devices) {
    auto controlSystem = pulse::controlSystemAt(ctx_, clock_ghz);
    if (!controlSystem)
      throw std::invalid_argument("load: clock_ghz " +
                                  std::to_string(clock_ghz) +
                                  " has no exact period in femtoseconds");
    auto source = parseForeign(text, ctx_);
    // A module that states its clock must state the quantum system's: its
    // times are counted in that clock's cycles.
    if (pulse::ControlSystemAttr stated = pulse::controlSystemOf(*source))
      if (stated.getClockPeriodFsNumerator() !=
              controlSystem.getClockPeriodFsNumerator() ||
          stated.getClockPeriodFsDenominator() !=
              controlSystem.getClockPeriodFsDenominator())
        throw std::invalid_argument(
            "load: the module counts time in cycles of a " +
            std::to_string(stated.clockGHz()) +
            " GHz clock, and the quantum system's clock is " +
            std::to_string(clock_ghz) + " GHz");
    // The entry is a function of the module's own symbol table, where the
    // functions it calls are found too.
    mlir::Operation *named = mlir::SymbolTable::lookupSymbolIn(
        source->getOperation(), llvm::StringRef(entry));
    if (!named)
      throw std::invalid_argument("load: the module has no function named " +
                                  entry);
    auto found = mlir::dyn_cast<mlir::func::FuncOp>(named);
    if (!found)
      throw std::invalid_argument("load: " + entry + " is not a function");
    if (found.isDeclaration())
      throw std::invalid_argument("load: " + entry + " is only declared");
    for (mlir::Type type : found.getArgumentTypes())
      if (!mlir::isa<pulse::QuditType>(type))
        throw std::invalid_argument(
            "load: " + entry +
            " takes an argument that is not a qudit; a loaded function takes "
            "one !pulse.qudit per qudit it plays on and nothing else");
    if (static_cast<int64_t>(found.getNumArguments()) != qudits)
      throw std::invalid_argument("load: " + entry + " takes " +
                                  std::to_string(found.getNumArguments()) +
                                  " qudit(s), and " + std::to_string(qudits) +
                                  " were given");
    if (!llvm::hasSingleElement(found.getBody()))
      throw std::invalid_argument("load: " + entry +
                                  " has more than one block");
    // A value chosen at run time by a measurement is lowered with the
    // program it was compiled in; a loaded function has none of that.
    if (found->hasAttr("pulse.selected_values"))
      throw std::invalid_argument(
          "load: " + entry +
          " selects values at run time (pulse.selected_values), which a "
          "loaded function cannot do yet");
    refuseUnsupported(found, /*called=*/false);
    if (pulse::scheduleStateOf(found) != pulse::ScheduleState::Timed)
      throw std::invalid_argument(
          "load: " + entry +
          " is not scheduled; load() takes a function whose operations all "
          "have their start and duration");
    auto loc = builder_.getUnknownLoc();
    mlir::OwningOpRef<mlir::ModuleOp> module(
        mlir::ModuleOp::create(loc, mlir::StringRef(entry)));
    module->getOperation()->setAttr(pulse::kControlSystemAttr, controlSystem);
    // The classical functions the program calls are stated by the module,
    // beside its functions; the links they travel on are the control
    // system's, written by the caller.
    for (mlir::Operation &op : *source->getBody())
      if (auto stated = mlir::dyn_cast<classical::CallableOp>(op)) {
        if (stated.getSymName() == "main")
          throw std::invalid_argument(
              "load: the module declares a classical function named main, "
              "the name the loaded function takes");
        mlir::Operation *callable = op.clone();
        // A latency given for the function replaces the one the module
        // states: (min, nominal, max) in ns.
        auto given = latencies.find(stated.getSymName().str());
        if (given != latencies.end()) {
          callable->setAttr("latency_min_ns",
                            builder_.getF64FloatAttr(given->second[0]));
          callable->setAttr("latency_nominal_ns",
                            builder_.getF64FloatAttr(given->second[1]));
          callable->setAttr("latency_max_ns",
                            builder_.getF64FloatAttr(given->second[2]));
        }
        // The control system's key for the function, and its id for the
        // link the function travels on, replace the module's.
        auto key = keys.find(stated.getSymName().str());
        if (key != keys.end() &&
            key->second != static_cast<int64_t>(stated.getDispatchKey())) {
          callable->setAttr("dispatch_key",
                            builder_.getI64IntegerAttr(key->second));
          callable->setAttr("dispatch_key_override", builder_.getUnitAttr());
        }
        auto device = devices.find(static_cast<int64_t>(stated.getDevice()));
        if (device != devices.end())
          callable->setAttr("device",
                            builder_.getI64IntegerAttr(device->second));
        module->push_back(callable);
      }
    auto main = mlir::cast<mlir::func::FuncOp>(found->clone());
    module->push_back(main);
    // What the calls wrote starts a call's start later, each level adding
    // its own: it is held to the range again.
    if (writeCalls(*source, *module, main, entry))
      refuseUnsupported(main, /*called=*/false, /*written=*/true);
    // One walk finds the links the function gets and what its lines, tones,
    // links and qudits go through. An operation of the Pulse dialect makes
    // one or continues the ones it takes, each result the operand in its
    // place; a branch or a loop that gives some is checked once the
    // function is known to be valid; nothing else makes one.
    llvm::SmallVector<mlir::Operation *> joins;
    mlir::Dialect *own = &controlSystem.getDialect();
    main.walk([&](mlir::Operation *op) {
      mlir::Dialect *dialect = op->getDialect();
      if (dialect == own)
        return;
      if (mlir::isa<mlir::scf::IfOp, mlir::scf::ForOp>(op)) {
        if (op->getNumResults())
          joins.push_back(op);
        return;
      }
      // Of the rest, only one that gives a line, tone, link or qudit is
      // looked at further: a classical operation may, and classical.get_link
      // names the link it gets.
      if (llvm::none_of(op->getResultTypes(), isHandle))
        return;
      if (!dialect || dialect->getNamespace() != "classical")
        refuseMade(op, entry);
      if (op->getName().getStringRef() != "classical.get_link")
        return;
      if (auto stated = op->getAttrOfType<mlir::IntegerAttr>("device")) {
        auto device = devices.find(stated.getInt());
        if (device == devices.end())
          throw std::invalid_argument(
              "load: " + entry + " gets link " +
              std::to_string(stated.getInt()) +
              ", which none of the module's classical functions travels on");
        op->setAttr("device", builder_.getI64IntegerAttr(device->second));
      }
    });
    mlir::NamedAttrList kept;
    for (mlir::NamedAttribute attribute : main->getAttrs()) {
      llvm::StringRef name = attribute.getName().getValue();
      if (name == "function_type" || name == pulse::kScheduleStateAttr)
        kept.append(attribute);
    }
    main->setAttrs(kept);
    main.setSymName("main");
    main.setPublic();
    // Attributes of other compilers on the operations are not the program's:
    // only an operation's own attributes, the Pulse dialect's, the mark of
    // a wait whose length is an answer, on such a wait, and the two phases
    // of a drive are kept. What the classical check writes on a call it
    // writes again, and answers bound to a call (`classical.answers`, which
    // a simulation writes on its own copy) are not a module's to state: a
    // loaded program takes every answer while it runs.
    Marked marked{main, entry, std::nullopt};
    main.walk([&](mlir::Operation *op) {
      if (op == main.getOperation())
        return;
      // Read once: an operation that keeps some in its properties builds
      // the list each time. One without attributes has none to drop.
      llvm::ArrayRef<mlir::NamedAttribute> stated = op->getAttrs();
      if (stated.empty())
        return;
      llvm::ArrayRef<mlir::StringAttr> own = op->getName().getAttributeNames();
      mlir::NamedAttrList attributes;
      for (mlir::NamedAttribute attribute : stated)
        if (attribute.getName().getValue().starts_with("pulse.") ||
            (attribute.getName().getValue() == "classical.variable" &&
             isRunTimeWait(op, marked)) ||
            llvm::is_contained(own, attribute.getName()) ||
            isDrivePhase(op, attribute))
          attributes.append(attribute);
      op->setAttrs(attributes);
    });
    // Attributes on the arguments and results are another compiler's too.
    main.removeArgAttrsAttr();
    main.removeResAttrsAttr();
    for (unsigned i = 0; i < main.getNumArguments(); ++i)
      main.setArgAttr(i, "pulse.slot", builder_.getI64IntegerAttr(i));
    main.walk([&](mlir::func::ReturnOp ret) {
      if (ret->getParentOp() == main.getOperation())
        ret->setOperands({});
    });
    main.setType(mlir::FunctionType::get(ctx_, main.getArgumentTypes(), {}));
    // What was written here from another compiler's module is checked as IR
    // before anything reads it; the verifier's reasons go with the refusal.
    {
      DiagnosticCollector diagnostics(ctx_);
      if (mlir::failed(mlir::verify(*module)))
        diagnostics.fail("load: " + entry +
                         " is not valid as a loaded function");
    }
    if (!joins.empty())
      refuseSwitched(joins, entry);
    return module;
  }

private:
  /// Whether a value of `type` stands for a physical line, tone or link, or
  /// for the qudit a line is acquired from.
  static bool isHandle(mlir::Type type) {
    return mlir::isa<pulse::DriveLineType, pulse::ReadoutLineType,
                     pulse::ToneType, pulse::QuditType, classical::LinkType>(
        type);
  }
  static const char *handleName(mlir::Type type) {
    if (mlir::isa<pulse::DriveLineType>(type))
      return "drive line";
    if (mlir::isa<pulse::ReadoutLineType>(type))
      return "readout line";
    if (mlir::isa<pulse::QuditType>(type))
      return "qudit";
    return mlir::isa<pulse::ToneType>(type) ? "tone" : "link";
  }
  /// Refuses `op`, which gives a line, tone, link or qudit and is not one of
  /// the operations that make one or pass one on.
  [[noreturn, gnu::cold, gnu::noinline]] static void
  refuseMade(mlir::Operation *op, const std::string &entry) {
    mlir::Type made = *llvm::find_if(op->getResultTypes(), isHandle);
    throw std::invalid_argument(
        "load: " + entry + " makes a " + handleName(made) + " with '" +
        op->getName().getStringRef().str() +
        "'; a loaded function takes its qudits as arguments, gets a line "
        "and its tone from a qudit (pulse.get_drive_line, "
        "pulse.get_readout_line), a tone of its own with pulse.tone and a "
        "link with classical.get_link, and passes them on through Pulse and "
        "classical operations, scf.if and scf.for");
  }
  /// Where the lines, tones, links and qudits of a function start: each
  /// value followed back, as what reads the function follows it
  /// (pulse::linearPredecessor), to the value nothing continues: a result
  /// of an acquisition, of classical.get_link or of pulse.tone, or a qudit
  /// the function takes. Each value is followed once.
  class Roots {
  public:
    mlir::Value operator()(mlir::Value value) {
      llvm::SmallVector<mlir::Value, 16> chain;
      mlir::Value root;
      while (true) {
        auto known = memo.find(value);
        if (known != memo.end()) {
          root = known->second;
          break;
        }
        chain.push_back(value);
        mlir::Value before = pulse::linearPredecessor(value);
        if (!before) {
          root = value;
          break;
        }
        value = before;
      }
      for (mlir::Value followed : chain)
        memo[followed] = root;
      return root;
    }
    /// Whether two roots are one physical line, tone, link or qudit: one
    /// value, acquisitions of one kind on one qudit (a qudit has one drive
    /// and one readout line), or links of one id.
    bool same(mlir::Value first, mlir::Value second) {
      if (first == second)
        return true;
      mlir::Operation *one = first.getDefiningOp();
      mlir::Operation *other = second.getDefiningOp();
      if (!one || !other || one->getName() != other->getName())
        return false;
      if (mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(one))
        return (*this)(one->getOperand(0)) == (*this)(other->getOperand(0));
      if (one->getName().getStringRef() != "classical.get_link")
        return false;
      auto id = one->getAttrOfType<mlir::IntegerAttr>("device");
      auto otherId = other->getAttrOfType<mlir::IntegerAttr>("device");
      return id && otherId && id.getInt() == otherId.getInt();
    }
    /// The root `root`, of a value of `type`, for a refusal.
    std::string name(mlir::Type type, mlir::Value root) {
      auto qudit = [](mlir::Value value) {
        auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
        return argument && mlir::isa<pulse::QuditType>(value.getType())
                   ? "qudit " + std::to_string(argument.getArgNumber())
                   : std::string();
      };
      mlir::Operation *made = root.getDefiningOp();
      if (!made) {
        std::string taken = qudit(root);
        return taken.empty() ? std::string("a ") + handleName(type) +
                                   " the function does not acquire"
                             : taken;
      }
      if (mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(made)) {
        std::string line =
            mlir::isa<pulse::ToneType>(type) ? "the tone of " : "";
        line += mlir::isa<pulse::GetDriveLineOp>(made) ? "the drive line"
                                                       : "the readout line";
        std::string of = qudit((*this)(made->getOperand(0)));
        return of.empty() ? line : line + " of " + of;
      }
      if (auto id = made->getAttrOfType<mlir::IntegerAttr>("device");
          id && made->getName().getStringRef() == "classical.get_link")
        return "link " + std::to_string(id.getInt());
      return std::string("a ") + handleName(type) + " made by '" +
             made->getName().getStringRef().str() + "'";
    }

  private:
    llvm::DenseMap<mlir::Value, mlir::Value> memo;
  };
  /// Refuses a branch or a loop that gives a qudit, a branch whose arms
  /// give two physical lines, tones or links as one result, and a loop that
  /// takes one in and yields another in its place. What reads the function
  /// takes such a value for the one the first arm gives, or the one the loop
  /// starts with, so it would play on one line what the function plays on
  /// another. `joins` are the scf.if and scf.for operations of the function
  /// that have results.
  static void refuseSwitched(llvm::ArrayRef<mlir::Operation *> joins,
                             const std::string &entry) {
    Roots roots;
    auto refuse = [&](mlir::Type type, mlir::Value first, mlir::Value second,
                      const char *before, const char *between,
                      const char *rule) {
      mlir::Value one = roots(first), other = roots(second);
      if (roots.same(one, other))
        return;
      throw std::invalid_argument("load: " + entry + " has " + before +
                                  roots.name(type, one) + between +
                                  roots.name(type, other) + rule);
    };
    for (mlir::Operation *op : joins) {
      // What reads the function finds a line's qudit at its acquisition:
      // the acquisition takes a qudit the function takes, as it is.
      if (llvm::any_of(op->getResultTypes(), [](mlir::Type type) {
            return mlir::isa<pulse::QuditType>(type);
          }))
        throw std::invalid_argument(
            "load: " + entry + " passes a qudit through an " +
            op->getName().getStringRef().str() +
            "; a loaded function acquires its lines from the qudits it "
            "takes, and a branch or a loop gives the lines");
      if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
        // A branch with results has both arms.
        mlir::Operation *taken = branch.thenBlock()->getTerminator();
        mlir::Operation *otherwise = branch.elseBlock()->getTerminator();
        for (unsigned index = 0; index < op->getNumResults(); ++index)
          if (mlir::Type type = op->getResult(index).getType(); isHandle(type))
            refuse(type, taken->getOperand(index), otherwise->getOperand(index),
                   "an scf.if whose arms give, as one result, ", " and ",
                   "; a line, tone or link that leaves a branch is the same "
                   "one in both arms: write what follows in each arm");
        continue;
      }
      auto loop = mlir::cast<mlir::scf::ForOp>(op);
      mlir::Operation *yield = loop.getBody()->getTerminator();
      for (unsigned index = 0; index < op->getNumResults(); ++index)
        if (mlir::Type type = op->getResult(index).getType(); isHandle(type))
          refuse(type, loop.getInitArgs()[index], yield->getOperand(index),
                 "an scf.for that takes in ", " and yields, in its place, ",
                 "; a line, tone or link a loop carries is the same one in "
                 "every trip");
    }
  }

  mlir::MLIRContext *ctx_;
  mlir::OpBuilder builder_;
};

/// The GIL, given up while a module is read.
struct Released {
  PyThreadState *state = PyEval_SaveThread();
  ~Released() { PyEval_RestoreThread(state); }
};

/// A reference this scope owns.
struct Owned {
  PyObject *object;
  explicit Owned(PyObject *object) : object(object) {}
  ~Owned() { Py_XDECREF(object); }
  Owned(const Owned &) = delete;
  Owned &operator=(const Owned &) = delete;
  PyObject *release() {
    PyObject *given = object;
    object = nullptr;
    return given;
  }
};

[[noreturn]] void wrongArgument(const char *what) {
  PyErr_Clear();
  throw std::invalid_argument(std::string("load: ") + what);
}

std::string bytesOf(PyObject *module) {
  char *data = nullptr;
  Py_ssize_t size = 0;
  if (PyBytes_AsStringAndSize(module, &data, &size))
    wrongArgument("the module is given as bytes");
  return std::string(data, static_cast<size_t>(size));
}

std::string textOf(PyObject *text) {
  Py_ssize_t size = 0;
  const char *data = PyUnicode_AsUTF8AndSize(text, &size);
  if (!data)
    wrongArgument("a name is given as str");
  return std::string(data, static_cast<size_t>(size));
}

int64_t integerOf(PyObject *value) {
  long long number = PyLong_AsLongLong(value);
  if (number == -1 && PyErr_Occurred())
    wrongArgument("a key or a link id is given as int");
  return number;
}

double numberOf(PyObject *value) {
  double number = PyFloat_AsDouble(value);
  if (number == -1.0 && PyErr_Occurred())
    wrongArgument("a latency is given as a number");
  return number;
}

/// Calls `each(key, value)` (borrowed) for the items of the dict `table`.
template <typename Fn>
void forItems(PyObject *table, Fn &&each) {
  if (!PyDict_Check(table))
    wrongArgument("latencies, keys and devices are given as dicts");
  Py_ssize_t position = 0;
  PyObject *key = nullptr, *value = nullptr;
  while (PyDict_Next(table, &position, &key, &value))
    each(key, value);
}

PyObject *textObject(const std::string &text) {
  return PyUnicode_FromStringAndSize(text.data(),
                                     static_cast<Py_ssize_t>(text.size()));
}

PyObject *textList(const std::vector<std::string> &texts) {
  Owned list(PyList_New(static_cast<Py_ssize_t>(texts.size())));
  if (!list.object)
    return nullptr;
  for (size_t i = 0; i < texts.size(); ++i) {
    PyObject *text = textObject(texts[i]);
    if (!text)
      return nullptr;
    PyList_SetItem(list.object, static_cast<Py_ssize_t>(i), text);
  }
  return list.release();
}

PyObject *optionalNumber(const std::optional<double> &value) {
  if (value)
    return PyFloat_FromDouble(*value);
  Py_INCREF(Py_None);
  return Py_None;
}

PyObject *optionalInteger(const std::optional<int64_t> &value) {
  if (value)
    return PyLong_FromLongLong(*value);
  Py_INCREF(Py_None);
  return Py_None;
}

/// A new tuple of `items`, which it takes; null when any is null.
PyObject *tupleOf(std::initializer_list<PyObject *> items) {
  Owned tuple(PyTuple_New(static_cast<Py_ssize_t>(items.size())));
  bool complete = tuple.object != nullptr;
  for (PyObject *item : items)
    complete = complete && item;
  Py_ssize_t index = 0;
  for (PyObject *item : items) {
    if (complete)
      PyTuple_SetItem(tuple.object, index++, item);
    else
      Py_XDECREF(item);
  }
  return complete ? tuple.release() : nullptr;
}

} // namespace

PyObject *callables(mlir::MLIRContext *ctx, PyObject *module) {
  std::string text = bytesOf(module);
  std::vector<Loader::LoadedCallable> rows;
  {
    Released released;
    rows = Loader(ctx).load_callables(text);
  }
  Owned list(PyList_New(static_cast<Py_ssize_t>(rows.size())));
  if (!list.object)
    return nullptr;
  for (size_t i = 0; i < rows.size(); ++i) {
    const auto &[name, key, device, args, results, latency, mode, submit,
                 timeout, deadline, reaction] = rows[i];
    PyObject *row =
        tupleOf({textObject(name), PyLong_FromLongLong(key),
                 PyLong_FromLongLong(device), textList(args), textList(results),
                 tupleOf({PyFloat_FromDouble(latency[0]),
                          PyFloat_FromDouble(latency[1]),
                          PyFloat_FromDouble(latency[2])}),
                 textObject(mode), textObject(submit), optionalNumber(timeout),
                 optionalNumber(deadline), optionalNumber(reaction)});
    if (!row)
      return nullptr;
    PyList_SetItem(list.object, static_cast<Py_ssize_t>(i), row);
  }
  return list.release();
}

PyObject *linkNames(mlir::MLIRContext *ctx, PyObject *module) {
  std::string text = bytesOf(module);
  std::map<int64_t, std::string> names;
  {
    Released released;
    names = Loader(ctx).load_link_names(text);
  }
  Owned table(PyDict_New());
  if (!table.object)
    return nullptr;
  for (const auto &[link, name] : names) {
    Owned key(PyLong_FromLongLong(link)), value(textObject(name));
    if (!key.object || !value.object ||
        PyDict_SetItem(table.object, key.object, value.object))
      return nullptr;
  }
  return table.release();
}

PyObject *submissions(mlir::MLIRContext *ctx, PyObject *module) {
  std::string text = bytesOf(module);
  std::vector<Loader::LoadedSubmission> rows;
  {
    Released released;
    rows = Loader(ctx).load_submissions(text);
  }
  Owned list(PyList_New(static_cast<Py_ssize_t>(rows.size())));
  if (!list.object)
    return nullptr;
  for (size_t i = 0; i < rows.size(); ++i) {
    const auto &[name, written, needed, words, indirect] = rows[i];
    Owned sent(PyList_New(static_cast<Py_ssize_t>(words.size())));
    if (!sent.object)
      return nullptr;
    for (size_t at = 0; at < words.size(); ++at) {
      PyObject *word = optionalInteger(words[at]);
      if (!word)
        return nullptr;
      PyList_SetItem(sent.object, static_cast<Py_ssize_t>(at), word);
    }
    PyObject *row = tupleOf({textObject(name), PyLong_FromLongLong(written),
                             PyLong_FromLongLong(needed), sent.release(),
                             PyBool_FromLong(indirect ? 1 : 0)});
    if (!row)
      return nullptr;
    PyList_SetItem(list.object, static_cast<Py_ssize_t>(i), row);
  }
  return list.release();
}

mlir::OwningOpRef<mlir::ModuleOp> function(mlir::MLIRContext *ctx,
                                           PyObject *module, PyObject *entry,
                                           double clockGHz, int64_t qudits,
                                           PyObject *latencies, PyObject *keys,
                                           PyObject *devices) {
  std::string text = bytesOf(module);
  std::string name = textOf(entry);
  std::map<std::string, std::array<double, 3>> stated;
  forItems(latencies, [&](PyObject *key, PyObject *value) {
    std::array<double, 3> latency{};
    for (Py_ssize_t i = 0; i < 3; ++i) {
      Owned item(PySequence_GetItem(value, i));
      if (!item.object)
        wrongArgument("a latency is given as (min, nominal, max)");
      latency[static_cast<size_t>(i)] = numberOf(item.object);
    }
    stated[textOf(key)] = latency;
  });
  std::map<std::string, int64_t> dispatch;
  forItems(keys, [&](PyObject *key, PyObject *value) {
    dispatch[textOf(key)] = integerOf(value);
  });
  std::map<int64_t, int64_t> links;
  forItems(devices, [&](PyObject *key, PyObject *value) {
    links[integerOf(key)] = integerOf(value);
  });
  Released released;
  return Loader(ctx).load_function(text, name, clockGHz, qudits, stated,
                                   dispatch, links);
}

} // namespace cudaq_pulse::loader
