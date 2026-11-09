// -*- tablegen -*-
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef CUDAQ_PULSE_DIALECT_PULSE_TIMING
#define CUDAQ_PULSE_DIALECT_PULSE_TIMING

include "mlir/IR/OpBase.td"

// The schedule of a timed op, held in the op's properties: its start and
// length in clock cycles and, inside `scf.for` loops, one period per
// enclosing loop, outermost first (see Analysis/Timing.h). Scheduling sets
// them; an op not scheduled has none.
defvar PulseTimingArgs = (ins
  OptionalAttr<I64Attr>:$start_cycles,
  OptionalAttr<I64Attr>:$duration_cycles,
  OptionalAttr<DenseI64ArrayAttr>:$period_cycles
);

#endif // CUDAQ_PULSE_DIALECT_PULSE_TIMING
