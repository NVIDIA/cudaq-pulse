// -*- tablegen -*-
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef CUDAQ_PULSE_DIALECT_PULSE_TIMING
#define CUDAQ_PULSE_DIALECT_PULSE_TIMING

include "mlir/IR/OpBase.td"
include "mlir/IR/Properties.td"

// A number of clock cycles an op may not have yet: written as an i64
// (`40 : i64`), and as `[]` in the generic form when it has none.
def Pulse_CyclesProp : OptionalProp<I64Prop> {
  let summary = "optional number of clock cycles";
  let convertToAttribute = [{
    if (!$_storage)
      return ::mlir::ArrayAttr::get($_ctxt, {});
    return ::mlir::IntegerAttr::get(::mlir::IntegerType::get($_ctxt, 64),
                                    *$_storage);
  }];
  let convertFromAttribute = [{
    if (auto integer = ::llvm::dyn_cast<::mlir::IntegerAttr>($_attr)) {
      $_storage = integer.getInt();
      return ::mlir::success();
    }
    auto none = ::llvm::dyn_cast<::mlir::ArrayAttr>($_attr);
    if (!none || !none.empty())
      return $_diag() << "expected a number of cycles, an i64";
    $_storage = std::nullopt;
    return ::mlir::success();
  }];
}

// One period per enclosing loop, written as `array<i64: ...>`; none outside
// loops.
def Pulse_PeriodsProp : DefaultValuedProp<ArrayProp<I64Prop>,
    "::llvm::ArrayRef<int64_t>{}", "::llvm::SmallVector<int64_t>{}"> {
  let summary = "periods in clock cycles, one per enclosing loop";
  let convertToAttribute = [{
    return ::mlir::DenseI64ArrayAttr::get($_ctxt, $_storage);
  }];
  let convertFromAttribute = [{
    auto periods = ::llvm::dyn_cast<::mlir::DenseI64ArrayAttr>($_attr);
    if (!periods)
      return $_diag() << "expected periods as array<i64: ...>";
    $_storage.assign(periods.asArrayRef().begin(), periods.asArrayRef().end());
    return ::mlir::success();
  }];
}

// The schedule of a timed op, held in the op's properties as integers: its
// start and length in clock cycles and, inside `scf.for` loops, one period per
// enclosing loop, outermost first (see Analysis/Timing.h). Scheduling sets
// them; an op not scheduled has none. Timed ops print them with
// `custom<Timing>` (Dialect/Pulse/PulseTimingFormat.h), as `prop-dict` would:
//   pulse.wait %l, %d <{duration_cycles = 20 : i64, start_cycles = 40 : i64}>
defvar PulseTimingArgs = (ins
  Pulse_CyclesProp:$start_cycles,
  Pulse_CyclesProp:$duration_cycles,
  Pulse_PeriodsProp:$period_cycles
);

#endif // CUDAQ_PULSE_DIALECT_PULSE_TIMING
