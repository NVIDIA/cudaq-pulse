// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The cycles a pulse or wait takes, as the schedule places it: what the
// scheduler writes as `duration_cycles` and pulse-verify checks it against.
//
// Include after the Pulse dialect op and type headers.

#pragma once

#include "cudaq-pulse/Analysis/Variant.h"

#include "cudaq-pulse/Analysis/Timing.h"
#include "mlir/IR/Value.h"

#include <optional>

namespace pulse {
PULSE_VARIANT_OPEN

/// The length of a waveform in cycles, when it is known: none for a
/// waveform that is an argument, or whose duration is computed from one.
/// Waveform arithmetic has the length of its (first) operand.
inline std::optional<int64_t> waveformCycles(mlir::Value waveform) {
  mlir::Operation *definition = waveform.getDefiningOp();
  if (!definition)
    return std::nullopt;
  mlir::Value duration;
  if (auto op = mlir::dyn_cast<GaussianPulseOp>(definition))
    duration = op.getDuration();
  else if (auto op = mlir::dyn_cast<SquarePulseOp>(definition))
    duration = op.getDuration();
  else if (auto op = mlir::dyn_cast<DRAGPulseOp>(definition))
    duration = op.getDuration();
  else if (auto op = mlir::dyn_cast<CosinePulseOp>(definition))
    duration = op.getDuration();
  else if (auto op = mlir::dyn_cast<TanhRampOp>(definition))
    duration = op.getDuration();
  else if (auto op = mlir::dyn_cast<GaussianSquarePulseOp>(definition))
    duration = op.getDuration();
  else if (auto op = mlir::dyn_cast<CustomOp>(definition))
    duration = op.getDuration();
  else if (auto op = mlir::dyn_cast<CustomSamplesOp>(definition))
    return op.getDurationCycles();
  else if (mlir::isa<PulseAddOp, PulseSubOp, PulseMulOp, PulseScaleOp,
                     PulseNegOp>(definition))
    return waveformCycles(definition->getOperand(0));
  else
    return std::nullopt;
  return constantInteger(duration);
}

/// The length of a wait in cycles: its constant length, or a variable
/// wait's lowest one while its length is not bound. None otherwise. A wait
/// a classical call's answer gives the length of (`classical.variable`) is
/// placed by the scheduler from the answer and its bounds.
inline std::optional<int64_t> waitCycles(WaitOp wait) {
  if (auto conversion = wait.getDuration().getDefiningOp<DurationFromIntOp>())
    if (auto cycles = constantInteger(conversion.getCycles()))
      return *cycles;
  if (auto lowest = wait->getAttrOfType<mlir::IntegerAttr>("pulse.min_cycles"))
    return lowest.getInt();
  return std::nullopt;
}

PULSE_VARIANT_CLOSE
} // namespace pulse
