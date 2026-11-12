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
    A module's quantum system (`#pulse.quantum_system`) lists its terms in
    `static_terms`; the lowering to QOp adds each one between simulated
    qudits.

    - `zz`: two distinct qudits; `rate_hz` is the ZZ rate zeta / 2 pi,
      zeta = E11 - E10 - E01 + E00, modeled as zeta n_a n_b.

    Example:
    ```mlir
    module attributes {pulse.quantum_system = #pulse.quantum_system<
        static_terms = [
          #pulse.static_term<zz, qudits = [3, 4], rate_hz = 4.5e+04>]>} {}
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
// The quantum system a program plays on
//===----------------------------------------------------------------------===//

def QuantumSystemAttr : AttrDef<PulseDialect, "QuantumSystem"> {
  let mnemonic = "quantum_system";
  let summary = "The quantum system a program plays on";
  let description = [{
    The module attribute `pulse.quantum_system`: what a program needs to know
    of the qudits it plays on, each list indexed by qudit (its index on the
    quantum system). A compiled module gives `qudit_frequency_hz`, the
    frequencies it is compiled for; a module lowered for simulation gives the
    simulated system, whose frequencies may differ from those its tones were
    compiled at. Every field may be left out, and the lists given have one
    entry per qudit.

    - `qudit_frequency_hz`: each qudit's 0-1 transition frequency, in Hz.
    - `level_counts`: the levels each qudit is simulated with, 2 or more.
    - `anharmonicity_hz`: each qudit's anharmonicity, in Hz.
    - `t1_ns`, `t2_ns`: each qudit's relaxation and dephasing times, in ns
      (0: none).
    - `drive_scale_rad_per_ns`: the Rabi rate of a unit drive amplitude.
    - `static_terms`: static Hamiltonian terms between qudits
      (`#pulse.static_term`).

    Example:
    ```mlir
    module attributes {pulse.quantum_system = #pulse.quantum_system<
        qudit_frequency_hz = [5.0e+09, 5.1e+09], level_counts = [3, 3],
        anharmonicity_hz = [-2.0e+08, -2.1e+08],
        static_terms = [#pulse.static_term<zz, qudits = [0, 1],
                                           rate_hz = 4.5e+04>]>} {}
    ```
  }];
  let parameters = (ins
    OptionalParameter<"::mlir::DenseF64ArrayAttr">:$qudit_frequency_hz,
    OptionalParameter<"::mlir::DenseI64ArrayAttr">:$level_counts,
    OptionalParameter<"::mlir::DenseF64ArrayAttr">:$anharmonicity_hz,
    OptionalParameter<"::mlir::DenseF64ArrayAttr">:$t1_ns,
    OptionalParameter<"::mlir::DenseF64ArrayAttr">:$t2_ns,
    OptionalParameter<"::mlir::DenseF64ArrayAttr">:$drive_scale_rad_per_ns,
    OptionalParameter<"::mlir::ArrayAttr">:$static_terms
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// The control system a program plays on
//===----------------------------------------------------------------------===//

def LineKind : I32EnumAttr<"LineKind", "The kind of a line",
  [
    I32EnumAttrCase<"Drive", 0, "drive">,
    I32EnumAttrCase<"Readout", 1, "readout">
  ]> {
  let cppNamespace = "::pulse";
  let genSpecializedAttr = 0;
}

def LineAttr : AttrDef<PulseDialect, "Line"> {
  let mnemonic = "line";
  let summary = "A drive or readout line of a control system";
  let description = [{
    A `drive` or `readout` line: the `qudits` it reaches (by index on the
    quantum system), the most tones it plays at once
    (`max_concurrent_tones`, default 1), its calibrated delay in program
    cycles (`delay_cycles`, default 0: every op on it starts that much later
    than scheduled) and the index of the controller that plays it
    (`controller`, default 0). Fields at their default are not printed.

    Example: `#pulse.line<readout, qudits = [5], delay_cycles = 24,
    controller = 1>`.
  }];
  let parameters = (ins
    EnumParameter<LineKind>:$kind,
    ArrayRefParameter<"int64_t", "qudits it reaches">:$qudits,
    "int64_t":$max_concurrent_tones,
    "int64_t":$delay_cycles,
    "int64_t":$controller
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def ControllerAttr : AttrDef<PulseDialect, "Controller"> {
  let mnemonic = "controller";
  let summary = "A controller of a control system";
  let description = [{
    What scheduling needs to know of one controller, every field optional:

    - `grid_cycles`: the program cycles between the instants it can start an
      operation (default 1); each of its lines moves on that grid.
    - `outcome_latency_cycles`: by controller, how long after a readout on
      this one ends its outcome is usable there, in program cycles.
    - `spread_cycles`: by controller, how far apart in time this one's start
      and that one's may be, in program cycles.
    - `uncertainty_ns`: how well its start is known.
    - `persistent_frames`: its tones keep their frames from one shot to the
      next.
    - `blocks_on_calls`: it plays nothing while one of its classical calls is
      out.
    - `max_duration_cycles`: scheduling fails if the program is longer.

    The two tables are given on every controller of a control system of
    several, or on none.

    Example: `#pulse.controller<grid_cycles = 2, outcome_latency_cycles =
    [100, 140], spread_cycles = [0, 3], uncertainty_ns = 0.5>`.
  }];
  let parameters = (ins
    "int64_t":$grid_cycles,
    OptionalArrayRefParameter<"int64_t">:$outcome_latency_cycles,
    OptionalArrayRefParameter<"int64_t">:$spread_cycles,
    APFloatParameter<"start uncertainty in ns, an f64">:$uncertainty_ns,
    "bool":$persistent_frames,
    "bool":$blocks_on_calls,
    "int64_t":$max_duration_cycles
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def ControlSystemAttr : AttrDef<PulseDialect, "ControlSystem"> {
  let mnemonic = "control_system";
  let summary = "The control system a program plays on";
  let description = [{
    The module attribute `pulse.control_system`: the program clock, as its
    period in femtoseconds, an exact fraction (`clock_period_fs = 1250000/3`
    is a 2.4 GHz clock; a whole period prints without `/1`); its `lines`
    (`#pulse.line`); and its `controllers` (`#pulse.controller`). A line it
    does not list has no delay and, with one controller, is that
    controller's; with several, such as a run-time qudit's line, it may be
    on any of them and moves on every one's grid.

    Example:
    ```mlir
    module attributes {pulse.control_system = #pulse.control_system<
        clock_period_fs = 500000,
        lines = [#pulse.line<drive, qudits = [0], delay_cycles = 3>],
        controllers = [#pulse.controller<grid_cycles = 2>]>} {}
    ```
  }];
  let parameters = (ins
    "int64_t":$clock_period_fs_numerator,
    "int64_t":$clock_period_fs_denominator,
    OptionalArrayRefParameter<"LineAttr">:$lines,
    OptionalArrayRefParameter<"ControllerAttr">:$controllers
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    /// The program clock in GHz: 1e6 over the period in fs.
    double clockGHz() const;
    /// The period in fs of a `clockGHz` clock, as the smallest fraction
    /// `numerator / denominator` whose clock is exactly that double; none
    /// for a clock too fast, too slow or not positive.
    static std::optional<std::pair<int64_t, int64_t>> periodFs(double clockGHz);
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
