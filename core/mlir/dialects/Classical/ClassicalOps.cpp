// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"
#include "cudaq-pulse/Dialect/Pulse/PulseEnums.h.inc"
#include "cudaq-pulse/Dialect/Pulse/PulseTimingFormat.h"
#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseTypes.h.inc"

#include "cudaq-pulse/Dialect/Classical/ClassicalDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Classical/ClassicalTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Classical/ClassicalOps.h.inc"

#include "cudaq-pulse/Dialect/Classical/Answers.h"
#include "cudaq-pulse/Dialect/Classical/Schema.h"

using namespace mlir;

namespace classical {

LogicalResult CallableOp::verify() {
  if (getArgs().size() > cudaq_pulse::wire::kMaxArgs)
    return emitOpError("has ") << getArgs().size() << " arguments; at most "
                               << cudaq_pulse::wire::kMaxArgs << " are allowed";
  if (getResults().size() > cudaq_pulse::wire::kMaxResults)
    return emitOpError("has ")
           << getResults().size() << " results; at most "
           << cudaq_pulse::wire::kMaxResults << " are allowed";
  for (auto [list, what] :
       {std::pair{getArgs(), "argument"}, std::pair{getResults(), "result"}})
    for (Attribute entry : list) {
      StringRef text = cast<StringAttr>(entry).getValue();
      if (!parseDescriptor(text))
        return emitOpError("has a malformed ")
               << what << " type '" << text
               << "'; expected u8, i32, i64, f32, f64, bits:N, levels:N, "
                  "iq:N or an array such as f64[N]";
    }
  double lo = getLatencyMinNs().convertToDouble();
  double mid = getLatencyNominalNs().convertToDouble();
  double hi = getLatencyMaxNs().convertToDouble();
  if (!(lo >= 0.0 && lo <= mid && mid <= hi && std::isfinite(hi)))
    return emitOpError("latency must satisfy 0 <= min <= nominal <= max, got ")
           << lo << ", " << mid << ", " << hi;
  StringRef mode = getMode();
  if (mode != "bounded" && mode != "timeout" && mode != "blocking")
    return emitOpError("mode must be \"bounded\", \"timeout\" or "
                       "\"blocking\", got \"")
           << mode << "\"";
  auto timeout = getTimeoutNs();
  if (mode == "timeout") {
    if (!timeout || !(timeout->convertToDouble() > 0.0))
      return emitOpError("a timeout callable needs a positive timeout_ns");
  } else if (timeout) {
    return emitOpError("timeout_ns is only meaningful with mode \"timeout\"");
  }
  if (getSubmit() != "eager" && getSubmit() != "lazy")
    return emitOpError("submit must be \"eager\" or \"lazy\", got \"")
           << getSubmit() << "\"";
  if (auto deadline = getDeadlineNs())
    if (!(deadline->convertToDouble() > 0.0))
      return emitOpError("deadline_ns must be positive");
  if (auto reaction = getReactionDeadlineNs())
    if (!(reaction->convertToDouble() > 0.0))
      return emitOpError("reaction_deadline_ns must be positive");
  int64_t id = getDispatchKeyAttr().getInt();
  if (id < 0 || id > 0xFFFFFFFFll)
    return emitOpError("dispatch_key must be a 32-bit unsigned value");
  if (!getDispatchKeyOverride()) {
    uint32_t expected = cudaq_pulse::wire::fnv1a32(
        std::string_view(getSymName().data(), getSymName().size()));
    if (static_cast<uint32_t>(id) != expected)
      return emitOpError("dispatch_key ")
             << id << " is not the FNV-1a hash of '" << getSymName() << "' ("
             << expected << "); set dispatch_key_override to use another id";
  }
  if (getDeviceAttr().getInt() < 0)
    return emitOpError("device must not be negative");
  return success();
}

LogicalResult GetLinkOp::verify() {
  if (getDeviceAttr().getInt() < 0)
    return emitOpError("device must not be negative");
  return success();
}

static CallableOp lookupCallable(Operation *op, FlatSymbolRefAttr callee,
                                 SymbolTableCollection &symbols) {
  return symbols.lookupNearestSymbolFrom<CallableOp>(op, callee);
}

LogicalResult SubmitOp::verifySymbolUses(SymbolTableCollection &symbols) {
  CallableOp callable = lookupCallable(*this, getCalleeAttr(), symbols);
  if (!callable)
    return emitOpError("'")
           << getCallee() << "' does not name a classical.callable";
  auto operands = getArgs();
  size_t next = 0;
  for (auto [index, entry] : llvm::enumerate(callable.getArgs())) {
    Descriptor desc = *parseDescriptor(cast<StringAttr>(entry).getValue());
    for (int64_t k = 0; k < desc.values(); ++k, ++next) {
      if (next >= operands.size())
        return emitOpError("passes ")
               << operands.size() << " values; '" << getCallee()
               << "' takes more (argument " << index << " is "
               << cast<StringAttr>(entry).getValue() << ")";
      Type type = operands[next].getType();
      bool ok = desc.takesMeasurements() ? isa<pulse::MeasurementType>(type)
                : desc.isFloat()         ? isa<FloatType>(type)
                                         : type.isSignlessInteger();
      if (!ok)
        return emitOpError("value ")
               << next << " has type " << type << ", but argument " << index
               << " of '" << getCallee() << "' is "
               << cast<StringAttr>(entry).getValue();
    }
  }
  if (next != operands.size())
    return emitOpError("passes ") << operands.size() << " values; '"
                                  << getCallee() << "' takes " << next;
  return success();
}

LogicalResult WaitOp::verifySymbolUses(SymbolTableCollection &symbols) {
  CallableOp callable = lookupCallable(*this, getCalleeAttr(), symbols);
  if (!callable)
    return emitOpError("'")
           << getCallee() << "' does not name a classical.callable";
  if (auto submit = getFuture().getDefiningOp<SubmitOp>())
    if (submit.getCallee() != getCallee())
      return emitOpError("waits for '")
             << getCallee() << "', but the future was submitted to '"
             << submit.getCallee() << "'";
  if (auto timeout = getTimeoutNs())
    if (!(timeout->convertToDouble() > 0.0))
      return emitOpError("timeout_ns must be positive");
  SmallVector<Type> expected =
      answerTypes(callable, getTimeoutNs().has_value());
  if (expected.empty())
    return emitOpError("'")
           << getCallee() << "' returns nothing; there is nothing to wait for";
  if (TypeRange(getValues().getTypes()) != TypeRange(expected)) {
    auto diagnostic = emitOpError("result types do not match '")
                      << getCallee() << "': expected (";
    llvm::interleaveComma(expected, diagnostic);
    return diagnostic << ")";
  }
  return success();
}

using pulse::parseTiming;
using pulse::printTiming;

} // namespace classical

#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Classical/ClassicalOps.cpp.inc"
