// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Two variants of the analyses and of the passes that use them.
//
// A module that holds a coupling line says so with the unit attribute
// `pulse.coupling_lines`. A pass asks that of the module once
// (`saysCouplingLines`) and hands a module that says so to the variant of
// itself that knows the ops of a coupling line. That variant is the same
// source compiled a second time with PULSE_COUPLED defined to 1: every test
// for a coupling line is written `kCoupled && ...`, which the compiler drops
// from the first variant, so a module without coupling lines runs the code
// it ran before coupling lines existed.
//
// The analysis headers put their contents between PULSE_VARIANT_OPEN and
// PULSE_VARIANT_CLOSE, which are empty in the first variant and the
// namespace `coupled` in the second: the two variants of an inline function
// are two functions. With PULSE_COUPLED 1 the namespace is inline, for a
// source compiled whole as the second variant. With PULSE_COUPLED 2 it is
// not, for a source that holds both and names the second `coupled::`; such
// a source includes a header once for each (the headers that allow it have
// a guard per variant).

// Include after the Pulse dialect type headers.

#include "mlir/IR/BuiltinOps.h"

#ifndef PULSE_COUPLED
#define PULSE_COUPLED 0
#endif

#undef PULSE_VARIANT_OPEN
#undef PULSE_VARIANT_CLOSE
#if PULSE_COUPLED == 1
#define PULSE_VARIANT_OPEN inline namespace coupled {
#define PULSE_VARIANT_CLOSE }
#elif PULSE_COUPLED == 2
#define PULSE_VARIANT_OPEN namespace coupled {
#define PULSE_VARIANT_CLOSE }
#else
#define PULSE_VARIANT_OPEN
#define PULSE_VARIANT_CLOSE
#endif

#ifndef CUDAQ_PULSE_ANALYSIS_VARIANT_H
#define CUDAQ_PULSE_ANALYSIS_VARIANT_H

namespace pulse {

/// The module attribute of a module that holds a coupling line.
inline constexpr llvm::StringLiteral kCouplingLinesAttr =
    "pulse.coupling_lines";

/// Whether the module `op` is in says that it holds a coupling line.
inline bool saysCouplingLines(mlir::Operation *op) {
  auto module = mlir::dyn_cast<mlir::ModuleOp>(op);
  if (!module)
    module = op->getParentOfType<mlir::ModuleOp>();
  return module && module->hasAttr(kCouplingLinesAttr);
}

/// Whether `module` holds a coupling line. The attribute says that one was
/// made; a walk, made only for a module with the attribute, says whether it
/// is still there (a line fetched and never used is dropped as unused): an
/// op that gives one.
inline bool holdsCouplingLine(mlir::ModuleOp module) {
  if (!module->hasAttr(kCouplingLinesAttr)) [[likely]]
    return false;
  return module
      ->walk([](mlir::Operation *op) {
        for (mlir::Type type : op->getResultTypes())
          if (mlir::isa<CouplingLineType>(type))
            return mlir::WalkResult::interrupt();
        return mlir::WalkResult::advance();
      })
      .wasInterrupted();
}

/// What a pass refused of the kernel, for the caller that reports the
/// pass's failure: a code, so that it need not read the diagnostics' text.
/// The pass sets it where it emits the error; the caller takes it and sets
/// it back when it reports a failure.
enum class Refusal { None, ComplexWaveform, NotSimulated };
inline thread_local Refusal lastRefusal = Refusal::None;

} // namespace pulse

#endif // CUDAQ_PULSE_ANALYSIS_VARIANT_H

// Whether this variant knows the ops of a coupling line.
#if PULSE_COUPLED == 0
#ifndef CUDAQ_PULSE_ANALYSIS_VARIANT_PLAIN
#define CUDAQ_PULSE_ANALYSIS_VARIANT_PLAIN
namespace pulse {
inline constexpr bool kCoupled = false;
} // namespace pulse
#endif
#else
#ifndef CUDAQ_PULSE_ANALYSIS_VARIANT_COUPLED
#define CUDAQ_PULSE_ANALYSIS_VARIANT_COUPLED
namespace pulse {
PULSE_VARIANT_OPEN
inline constexpr bool kCoupled = true;
PULSE_VARIANT_CLOSE
} // namespace pulse
#endif
#endif
