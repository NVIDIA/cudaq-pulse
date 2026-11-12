// -*- tablegen -*-
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef CUDAQ_PULSE_DIALECT_PULSE_ATTRS
#define CUDAQ_PULSE_DIALECT_PULSE_ATTRS

include "cudaq-pulse/Dialect/Pulse/PulseDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Static Hamiltonian terms of a quantum system
//===----------------------------------------------------------------------===//

def StaticTermKind : I32EnumAttr<"StaticTermKind",
  "Operator of a static Hamiltonian term",
  [
    I32EnumAttrCase<"ZZ", 0, "zz">
  ]> {
  let cppNamespace = "::pulse";
  let genSpecializedAttr = 0;
}

def StaticTermAttr : AttrDef<PulseDialect, "StaticTerm"> {
  let mnemonic = "static_term";
  let summary = "A static Hamiltonian term of a quantum system";
  let description = [{
    A term of the system's Hamiltonian that does not depend on time, of the
    operator `kind` on the qudits `qudits` (device indices, as many as the
    kind acts on) at the rate `rate_hz` (Hz, the coefficient over 2 pi).
    A module lists its terms in `pulse.static_terms`; the lowering to QOp
    adds each one between simulated qudits.

    - `zz`: two distinct qudits; `rate_hz` is the ZZ rate zeta / 2 pi,
      zeta = E11 - E10 - E01 + E00, modeled as zeta n_a n_b.

    Example:
    ```mlir
    module attributes {pulse.static_terms = [
        #pulse.static_term<zz, qudits = [3, 4], rate_hz = 4.5e+04>]} {}
    ```
  }];
  let parameters = (ins
    EnumParameter<StaticTermKind>:$kind,
    ArrayRefParameter<"int64_t", "qudits, by device index">:$qudits,
    APFloatParameter<"rate in Hz, an f64">:$rate_hz
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    /// The number of qudits a term of *kind* acts on.
    static unsigned arity(StaticTermKind kind);
  }];
}

//===----------------------------------------------------------------------===//
// How far a function is scheduled
//===----------------------------------------------------------------------===//

def ScheduleState : I32EnumAttr<"ScheduleState",
  "How far a function is scheduled",
  [
    I32EnumAttrCase<"Unscheduled", 0, "unscheduled">,
    I32EnumAttrCase<"Timed", 1, "timed">,
    I32EnumAttrCase<"Placed", 2, "placed">
  ]> {
  let cppNamespace = "::pulse";
  let genSpecializedAttr = 0;
}

def ScheduleStateAttr : EnumAttr<PulseDialect, ScheduleState,
                                 "schedule_state"> {
  let summary = "How far a function is scheduled";
  let description = [{
    The `pulse.schedule_state` of a `func.func`. Without one a function is
    `unscheduled`: its timed ops may carry a start or a length, or not.
    `timed`: every timed op that runs has its start and its length
    (`pulse-verify` checks). `placed`: timed, and each line's calibrated
    delay is applied to its ops' starts. `pulse-schedule-*` writes `placed`
    when it moves some op by a line delay, `timed` otherwise; passes that
    change lengths or the order of ops on a line, such as
    `pulse-merge-waits`, take only an unscheduled function.

    Example:
    ```mlir
    func.func @main() attributes {
        pulse.schedule_state = #pulse.schedule_state<timed>} {
      return
    }
    ```
  }];
  let assemblyFormat = "`<` $value `>`";
}

#endif // CUDAQ_PULSE_DIALECT_PULSE_ATTRS
