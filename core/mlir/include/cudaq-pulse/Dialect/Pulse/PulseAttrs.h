// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The Pulse dialect's attributes (PulseAttrs.td) and the module and function
// attributes made of them, read by name.

#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Operation.h"

#include "cudaq-pulse/Dialect/Pulse/PulseAttrEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseAttrs.h.inc"

namespace pulse {

/// The function attribute that says how far a function is scheduled.
inline constexpr llvm::StringLiteral kScheduleStateAttr =
    "pulse.schedule_state";

/// How far `function` is scheduled: unscheduled without the attribute.
inline ScheduleState scheduleStateOf(mlir::Operation *function) {
  if (auto attr =
          function->getAttrOfType<ScheduleStateAttr>(kScheduleStateAttr))
    return attr.getValue();
  return ScheduleState::Unscheduled;
}

} // namespace pulse
