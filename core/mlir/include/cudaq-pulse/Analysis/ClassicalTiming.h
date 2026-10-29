/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// The timing of classical calls: what a link costs and how long an answer is
// budgeted, in clock cycles.
//
// A link's parameters come from the module attribute `classical.links`, a
// dictionary from the link id (as a string) to a dictionary with any of
// `fixed_ns` (fixed cost of sending a request), `bytes_per_ns` (bandwidth;
// 0 for unbounded), `feedback_ns` (from an answer arriving to a pulse being
// able to use it), `start_uncertainty_ns` (on a link that joins controllers,
// how far apart the starts of the two it joins known least well may be),
// `max_in_flight` (requests a link holds at once) and `slot_bytes` (the
// largest request or answer, header included).
//
// An answer's budget adds the start uncertainty between the controller that
// sends the request and the one that takes the answer: none when one
// controller does both, the link's `start_uncertainty_ns` when the module's
// `pulse.controllers` (see Controllers.h) does not say which they are.
//
// `annotateClassicalCalls` writes on every `classical.submit` the attributes
// scheduling reads: `classical.submit_cycles` (how long the submission holds
// the link), `classical.budget_cycles` (from its end until the answer is due),
// `classical.device`, `classical.max_in_flight` and `classical.eager`; and on
// every `classical.wait` `classical.device` and, in blocking mode,
// `classical.blocking`.
//
// Lines that no run-time wait ties together wait independently: the
// scheduler numbers each such group of lines a timeline, `classical.timeline`
// on their allocations and on the calls made for them. A link used by
// several timelines shares out its slots between them and sends for each in
// turn, so each timeline's requests are timed as if the link were its own.
//
// Include after the Pulse and Classical op headers.

#pragma once

#include "cudaq-pulse/Analysis/Controllers.h"
#include "cudaq-pulse/Dialect/Classical/Schema.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

namespace classical {

struct LinkModel {
  double fixedNs = 0.0;
  double bytesPerNs = 0.0;
  double feedbackNs = 0.0;
  double uncertaintyNs = 0.0;
  int64_t maxInFlight = 64;
  int64_t slotBytes = 1024;
};

inline LinkModel linkModel(mlir::ModuleOp module, int64_t device) {
  LinkModel link;
  auto links = module->getAttrOfType<mlir::DictionaryAttr>("classical.links");
  if (!links)
    return link;
  auto entry = links.getAs<mlir::DictionaryAttr>(std::to_string(device));
  if (!entry)
    return link;
  auto number = [&](llvm::StringRef name, double fallback) {
    mlir::Attribute attr = entry.get(name);
    if (auto f = mlir::dyn_cast_or_null<mlir::FloatAttr>(attr))
      return f.getValueAsDouble();
    if (auto i = mlir::dyn_cast_or_null<mlir::IntegerAttr>(attr))
      return static_cast<double>(i.getInt());
    return fallback;
  };
  link.fixedNs = number("fixed_ns", link.fixedNs);
  link.bytesPerNs = number("bytes_per_ns", link.bytesPerNs);
  link.feedbackNs = number("feedback_ns", link.feedbackNs);
  link.uncertaintyNs = number("start_uncertainty_ns", link.uncertaintyNs);
  link.maxInFlight = static_cast<int64_t>(
      number("max_in_flight", static_cast<double>(link.maxInFlight)));
  link.slotBytes = static_cast<int64_t>(
      number("slot_bytes", static_cast<double>(link.slotBytes)));
  return link;
}

/// Wire bytes of the descriptors in `list`.
inline int64_t payloadBytes(mlir::ArrayAttr list) {
  int64_t bytes = 0;
  for (mlir::Attribute entry : list)
    if (auto desc = parseDescriptor(mlir::cast<mlir::StringAttr>(entry)))
      bytes += desc->wireBytes();
  return bytes;
}

constexpr int64_t kHeaderBytes = sizeof(cudaq_pulse::wire::RequestHeader);

inline int64_t requestBytes(CallableOp callable) {
  return kHeaderBytes + payloadBytes(callable.getArgs());
}

inline int64_t answerBytes(CallableOp callable) {
  return kHeaderBytes + payloadBytes(callable.getResults());
}

/// How long a submission holds the link: at least one cycle.
inline int64_t submitCycles(CallableOp callable, const LinkModel &link,
                            double clockGHz) {
  double ns = link.fixedNs;
  if (link.bytesPerNs > 0.0)
    ns += static_cast<double>(requestBytes(callable)) / link.bytesPerNs;
  return std::max<int64_t>(1, ceilCycles(ns, clockGHz));
}

/// From the end of a submission until its answer is due: the maximum
/// latency (the timeout in timeout mode, or the timeout of the wait that
/// takes the answer), the link's feedback latency and the start uncertainty
/// between sender and taker (`uncertaintyNs`, else the link's).
inline int64_t
budgetCycles(CallableOp callable, const LinkModel &link, double clockGHz,
             std::optional<double> waitTimeoutNs = std::nullopt,
             std::optional<double> uncertaintyNs = std::nullopt) {
  double latency = waitTimeoutNs ? *waitTimeoutNs
                   : callable.getMode() == "timeout"
                       ? callable.getTimeoutNs()->convertToDouble()
                       : callable.getLatencyMaxNs().convertToDouble();
  return ceilCycles(latency, clockGHz) +
         ceilCycles(link.feedbackNs +
                        uncertaintyNs.value_or(link.uncertaintyNs),
                    clockGHz);
}

/// The waits that take `future`, following it into and out of
/// `pulse.schedule_block`s.
inline void waitsOf(mlir::Value future, llvm::SmallVectorImpl<WaitOp> &found) {
  for (mlir::OpOperand &use : future.getUses()) {
    mlir::Operation *user = use.getOwner();
    if (auto wait = mlir::dyn_cast<WaitOp>(user))
      found.push_back(wait);
    else if (auto block = mlir::dyn_cast<pulse::ScheduleBlockOp>(user))
      waitsOf(block.getBody().getArgument(use.getOperandNumber()), found);
    else if (mlir::isa<pulse::YieldOp>(user))
      if (auto block =
              mlir::dyn_cast<pulse::ScheduleBlockOp>(user->getParentOp()))
        waitsOf(block->getResult(use.getOperandNumber()), found);
  }
}

inline llvm::SmallVector<WaitOp, 1> waitsOf(SubmitOp submit) {
  llvm::SmallVector<WaitOp, 1> found;
  waitsOf(submit.getFuture(), found);
  return found;
}

/// Adds to `found` the controllers of the readouts `value` is computed from;
/// false if one is on no known controller.
inline bool readers(mlir::Value value, const pulse::Controllers &controllers,
                    pulse::AllocationTracer &trace,
                    llvm::SmallVectorImpl<int64_t> &found,
                    llvm::DenseSet<mlir::Value> &seen) {
  if (!seen.insert(value).second)
    return true;
  if (auto readout = value.getDefiningOp<pulse::ReadoutOp>()) {
    if (value != readout.getResult())
      return true;
    int64_t reader = controllers.of(readout->getOperand(0), trace);
    found.push_back(reader);
    return reader >= 0;
  }
  if (mlir::isa<pulse::DriveLineType, pulse::ReadoutLineType, pulse::ToneType,
                LinkType>(value.getType()))
    return true;
  mlir::Operation *def = value.getDefiningOp();
  if (!def || mlir::isa<WaitOp, SubmitOp>(def) || def->getNumRegions())
    return true;
  for (mlir::Value operand : def->getOperands())
    if (!readers(operand, controllers, trace, found, seen))
      return false;
  return true;
}

/// Adds to `found` the controllers of the pulses that use `value`, or of
/// the branches on it; false if one of them is unknown, or `value` leaves
/// what can be followed.
inline bool takers(mlir::Value value, const pulse::Controllers &controllers,
                   pulse::AllocationTracer &trace,
                   llvm::SmallVectorImpl<int64_t> &found,
                   llvm::DenseSet<mlir::Value> &seen) {
  if (!seen.insert(value).second)
    return true;
  auto pulse = [&](mlir::Operation *op) {
    int64_t taker = controllers.of(op->getOperand(0), trace);
    found.push_back(taker);
    return taker >= 0;
  };
  for (mlir::OpOperand &use : value.getUses()) {
    mlir::Operation *user = use.getOwner();
    if (mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp>(user)) {
      if (!pulse(user))
        return false;
    } else if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(user)) {
      bool known = true;
      branch->walk([&](mlir::Operation *inner) {
        if (known &&
            mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp>(inner))
          known = pulse(inner);
      });
      if (!known)
        return false;
      for (mlir::Value result : branch.getResults())
        if (!takers(result, controllers, trace, found, seen))
          return false;
    } else if (mlir::isa<mlir::scf::YieldOp>(user)) {
      auto branch = mlir::dyn_cast<mlir::scf::IfOp>(user->getParentOp());
      if (!branch || !takers(branch->getResult(use.getOperandNumber()),
                             controllers, trace, found, seen))
        return false;
    } else if (user->getNumRegions() || user->getNumResults() == 0 ||
               mlir::isa<SubmitOp>(user)) {
      return false;
    } else {
      for (mlir::Value result : user->getResults())
        if (!takers(result, controllers, trace, found, seen))
          return false;
    }
  }
  return true;
}

/// On several controllers, the start uncertainty between the controllers
/// that send the request of `submit` and take its answer, in ns: none when
/// one controller does both. Empty when they are not known.
inline std::optional<double>
callUncertaintyNs(SubmitOp submit, const pulse::Controllers &controllers,
                  pulse::AllocationTracer &trace) {
  if (!controllers.enabled())
    return std::nullopt;
  llvm::SmallVector<int64_t, 2> senders, users;
  llvm::DenseSet<mlir::Value> seen;
  for (mlir::Value operand : submit->getOperands())
    if (!readers(operand, controllers, trace, senders, seen))
      return std::nullopt;
  seen.clear();
  for (WaitOp wait : waitsOf(submit))
    for (mlir::Value value : wait.getValues())
      if (!takers(value, controllers, trace, users, seen))
        return std::nullopt;
  if (senders.empty() || users.empty())
    return std::nullopt;
  double most = 0.0;
  for (int64_t sender : senders)
    for (int64_t user : users)
      if (sender != user)
        most = std::max(most, controllers.uncertainty(sender) +
                                  controllers.uncertainty(user));
  return most;
}

/// The timeout of the wait that takes the answer of `submit`, if it has one.
inline std::optional<double> waitTimeoutNs(SubmitOp submit) {
  for (WaitOp wait : waitsOf(submit))
    if (auto timeout = wait.getTimeoutNs())
      return timeout->convertToDouble();
  return std::nullopt;
}

inline double moduleClockGHz(mlir::ModuleOp module) {
  if (auto clock = module->getAttrOfType<mlir::FloatAttr>("pulse.clock_ghz"))
    return clock.getValueAsDouble();
  return 1.0;
}

/// Whether `module` declares any classical function: only then can it hold
/// classical calls. Looks at top-level ops only.
inline bool declaresClassicalCalls(mlir::ModuleOp module) {
  for (mlir::Operation &op : *module.getBody())
    if (mlir::isa<CallableOp>(op))
      return true;
  return false;
}

inline constexpr const char *kTimelineAttr = "classical.timeline";

/// Write the timing attributes of every call in `function` (see the file
/// comment). Returns false if `function` makes no call.
inline bool annotateClassicalCalls(mlir::func::FuncOp function) {
  auto module = function->getParentOfType<mlir::ModuleOp>();
  if (!module || !declaresClassicalCalls(module))
    return false;
  double clock = moduleClockGHz(module);
  mlir::SymbolTable symbols(module);
  mlir::Builder b(function.getContext());
  pulse::Controllers controllers(module);
  pulse::AllocationTracer trace;
  bool any = false;
  // The timelines on each link, for the submissions that have one.
  std::vector<std::pair<mlir::Operation *, int64_t>> shared;
  std::map<int64_t, std::vector<int64_t>> timelines;
  function.walk([&](mlir::Operation *op) {
    if (auto submit = mlir::dyn_cast<SubmitOp>(op)) {
      auto callable = symbols.lookup<CallableOp>(submit.getCallee());
      if (!callable)
        return;
      any = true;
      LinkModel link = linkModel(module, callable.getDevice());
      mlir::NamedAttrList attrs(op->getAttrDictionary());
      attrs.set("classical.submit_cycles",
                b.getI64IntegerAttr(submitCycles(callable, link, clock)));
      attrs.set("classical.budget_cycles",
                b.getI64IntegerAttr(budgetCycles(
                    callable, link, clock, waitTimeoutNs(submit),
                    callUncertaintyNs(submit, controllers, trace))));
      attrs.set("classical.device", b.getI64IntegerAttr(callable.getDevice()));
      attrs.set("classical.max_in_flight",
                b.getI64IntegerAttr(link.maxInFlight));
      if (callable.getSubmit() == "eager")
        attrs.set("classical.eager", b.getUnitAttr());
      else
        attrs.erase("classical.eager");
      if (auto timeline = attrs.get(kTimelineAttr)) {
        int64_t t = mlir::cast<mlir::IntegerAttr>(timeline).getInt();
        shared.emplace_back(op, t);
        timelines[callable.getDevice()].push_back(t);
      }
      op->setAttrs(attrs.getDictionary(op->getContext()));
    } else if (auto wait = mlir::dyn_cast<WaitOp>(op)) {
      auto callable = symbols.lookup<CallableOp>(wait.getCallee());
      if (!callable)
        return;
      any = true;
      op->setAttr("classical.device",
                  b.getI64IntegerAttr(callable.getDevice()));
      if (callable.getMode() == "blocking")
        op->setAttr("classical.blocking", b.getUnitAttr());
    }
  });
  // A link that T timelines use gives each its share of the slots, the
  // first ones one more when they do not divide evenly. A timeline sends one
  // request at a time, so each is sent at most after one pending send of
  // every other timeline: it holds the link for its own send and T - 1 of
  // the link's longest.
  auto cyclesOf = [](mlir::Operation *op, llvm::StringRef name) {
    return mlir::cast<mlir::IntegerAttr>(op->getAttr(name)).getInt();
  };
  std::map<int64_t, int64_t> longest;
  for (auto [op, t] : shared) {
    int64_t &most = longest[cyclesOf(op, "classical.device")];
    most = std::max(most, cyclesOf(op, "classical.submit_cycles"));
  }
  for (auto &[device, list] : timelines) {
    std::sort(list.begin(), list.end());
    list.erase(std::unique(list.begin(), list.end()), list.end());
  }
  for (auto [op, t] : shared) {
    int64_t device = cyclesOf(op, "classical.device");
    const std::vector<int64_t> &list = timelines[device];
    auto count = static_cast<int64_t>(list.size());
    if (count <= 1)
      continue;
    auto rank = static_cast<int64_t>(
        std::lower_bound(list.begin(), list.end(), t) - list.begin());
    int64_t slots = linkModel(module, device).maxInFlight;
    int64_t share = slots / count + (rank < slots % count ? 1 : 0);
    int64_t send =
        cyclesOf(op, "classical.submit_cycles") + (count - 1) * longest[device];
    mlir::NamedAttrList attrs(op->getAttrDictionary());
    attrs.set("classical.submit_cycles", b.getI64IntegerAttr(send));
    attrs.set("classical.max_in_flight",
              b.getI64IntegerAttr(std::max<int64_t>(1, share)));
    op->setAttrs(attrs.getDictionary(op->getContext()));
  }
  return any;
}

/// Why a wait cannot be in a `pulse.schedule_block`, which is placed whole:
/// the verifier and the scheduler, which may run without it, both say so.
inline constexpr const char *kVariableWaitInBlock =
    "a variable wait cannot be inside a pulse.schedule_block, whose length "
    "is fixed; call the kernel without schedule= or move the wait out of it";
inline constexpr const char *kBlockingWaitInBlock =
    "a blocking wait cannot be inside a pulse.schedule_block, whose length "
    "is fixed; return the future from the kernel and take the answer after "
    "it returns, or call the kernel without schedule=";

inline int64_t intAttr(mlir::Operation *op, llvm::StringRef name,
                       int64_t fallback = 0) {
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>(name))
    return attr.getInt();
  return fallback;
}

} // namespace classical
