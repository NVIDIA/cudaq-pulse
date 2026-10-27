/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// The timing of classical calls: what a link costs and how long an answer is
// budgeted, in virtual time units.
//
// A link's parameters come from the module attribute `classical.links`, a
// dictionary from the link id (as a string) to a dictionary with any of
// `fixed_ns` (fixed cost of sending a request), `bytes_per_ns` (bandwidth;
// 0 for unbounded), `feedback_ns` (from an answer arriving to a pulse being
// able to use it), `max_in_flight` (requests a link holds at once) and
// `slot_bytes` (the largest request or answer, header included).
//
// `annotateClassicalCalls` writes on every `classical.submit` the attributes
// scheduling reads: `classical.submit_vtu` (how long the submission holds the
// link), `classical.budget_vtu` (from its end until the answer is due),
// `classical.device`, `classical.max_in_flight` and `classical.eager`; and on
// every `classical.wait` `classical.device` and, in blocking mode,
// `classical.blocking`.
//
// Include after the Pulse and Classical op headers.

#pragma once

#include "cudaq-pulse/Dialect/Classical/Schema.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

#include <cstdint>
#include <optional>

namespace classical {

struct LinkModel {
  double fixedNs = 0.0;
  double bytesPerNs = 0.0;
  double feedbackNs = 0.0;
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
inline int64_t submitVtu(CallableOp callable, const LinkModel &link,
                         double clockGHz) {
  double ns = link.fixedNs;
  if (link.bytesPerNs > 0.0)
    ns += static_cast<double>(requestBytes(callable)) / link.bytesPerNs;
  return std::max<int64_t>(1, ceilVtu(ns, clockGHz));
}

/// From the end of a submission until its answer is due: the maximum
/// latency (the timeout in timeout mode, or the timeout of the wait that
/// takes the answer) and the link's feedback latency.
inline int64_t budgetVtu(CallableOp callable, const LinkModel &link,
                         double clockGHz,
                         std::optional<double> waitTimeoutNs = std::nullopt) {
  double latency = waitTimeoutNs ? *waitTimeoutNs
                   : callable.getMode() == "timeout"
                       ? callable.getTimeoutNs()->convertToDouble()
                       : callable.getLatencyMaxNs().convertToDouble();
  return ceilVtu(latency, clockGHz) + ceilVtu(link.feedbackNs, clockGHz);
}

/// The timeout of the wait that takes the answer of `submit`, if it has one.
inline std::optional<double> waitTimeoutNs(SubmitOp submit) {
  for (mlir::Operation *user : submit.getFuture().getUsers())
    if (auto wait = mlir::dyn_cast<WaitOp>(user))
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

/// Write the timing attributes of every call in `function` (see the file
/// comment). Returns false if `function` makes no call.
inline bool annotateClassicalCalls(mlir::func::FuncOp function) {
  auto module = function->getParentOfType<mlir::ModuleOp>();
  if (!module || !declaresClassicalCalls(module))
    return false;
  double clock = moduleClockGHz(module);
  mlir::SymbolTable symbols(module);
  mlir::Builder b(function.getContext());
  bool any = false;
  function.walk([&](mlir::Operation *op) {
    if (auto submit = mlir::dyn_cast<SubmitOp>(op)) {
      auto callable = symbols.lookup<CallableOp>(submit.getCallee());
      if (!callable)
        return;
      any = true;
      LinkModel link = linkModel(module, callable.getDevice());
      mlir::NamedAttrList attrs(op->getAttrDictionary());
      attrs.set("classical.submit_vtu",
                b.getI64IntegerAttr(submitVtu(callable, link, clock)));
      attrs.set("classical.budget_vtu",
                b.getI64IntegerAttr(
                    budgetVtu(callable, link, clock, waitTimeoutNs(submit))));
      attrs.set("classical.device", b.getI64IntegerAttr(callable.getDevice()));
      attrs.set("classical.max_in_flight",
                b.getI64IntegerAttr(link.maxInFlight));
      if (callable.getSubmit() == "eager")
        attrs.set("classical.eager", b.getUnitAttr());
      else
        attrs.erase("classical.eager");
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
  return any;
}

inline int64_t intAttr(mlir::Operation *op, llvm::StringRef name,
                       int64_t fallback = 0) {
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>(name))
    return attr.getInt();
  return fallback;
}

} // namespace classical
