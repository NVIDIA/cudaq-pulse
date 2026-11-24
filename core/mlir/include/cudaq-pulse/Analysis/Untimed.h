// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The ops the schedule passes cannot time.
//
// Include after the Pulse dialect, op and type headers and LineTrace.h.

#pragma once

#include "cudaq-pulse/Analysis/Variant.h"

#include "llvm/Support/Compiler.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"

namespace pulse {
PULSE_VARIANT_OPEN

/// Whether the schedule passes cannot time `op`. They time the ops of the
/// pulse and classical dialects, go into the bodies of `scf.for` and
/// `scf.if`, and pass a line, tone, measurement or classical handle through
/// any other op as if it took no time. That is right for the ops of arith,
/// ub and builtin and for a terminator. An op of any other kind that takes
/// or gives such a value, or a qudit whose lines it could acquire (a
/// `func.call` that is handed a line, or other control flow that gives
/// one), would hide what it does to the line. So would an op of a dialect
/// that is not loaded. An op that holds a region and takes or gives none of
/// these is not untimed by this test: the schedule passes refuse it where
/// they come to it.
///
/// The schedule passes and pulse-verify ask this of every op in walks they
/// make anyway, so it is inlined and asks the common dialects first.
LLVM_ATTRIBUTE_ALWAYS_INLINE bool untimed(mlir::Operation *op) {
  if (mlir::Dialect *dialect = op->getDialect()) {
    if (mlir::isa<PulseDialect, mlir::arith::ArithDialect>(*dialect) ||
        mlir::isa<mlir::scf::ForOp, mlir::scf::IfOp>(op))
      return false;
    llvm::StringRef name = dialect->getNamespace();
    if (name == "classical" || name == "ub" || name == "builtin")
      return false;
  }
  if (op->hasTrait<mlir::OpTrait::IsTerminator>())
    return false;
  auto held = [](mlir::Type type) {
    return isLinearPulseType(type) ||
           mlir::isa<MeasurementType, QuditType>(type);
  };
  return llvm::any_of(op->getOperandTypes(), held) ||
         llvm::any_of(op->getResultTypes(), held);
}

/// The error for an op that is `untimed`.
inline mlir::InFlightDiagnostic emitUntimed(mlir::Operation *op) {
  return op->emitError("'")
         << op->getName().getStringRef()
         << "' takes or gives a qudit, a line, a tone, a measurement or a "
            "classical handle, and the schedule passes cannot tell what it "
            "plays or how long it takes; write the ops it stands for in its "
            "place before scheduling";
}

PULSE_VARIANT_CLOSE
} // namespace pulse
