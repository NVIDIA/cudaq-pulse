// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A classical call inside a schedule block, its future leaving the block.
// RUN: cudaq-pulse-opt --pulse-schedule-asap %s | FileCheck %s
// RUN: cudaq-pulse-opt --pulse-schedule-alap %s | FileCheck %s
// RUN: cudaq-pulse-opt --pulse-schedule-alap --pulse-classical-check --pulse-verify %s > /dev/null

module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000>} {

classical.callable @decode_wide {dispatch_key = 1, device = 1 : i64,
    args = ["bits:1"], results = ["bits:1"], latency_min_ns = 100.0,
    latency_nominal_ns = 500.0, latency_max_ns = 1000.0, mode = "bounded",
    submit = "eager", dispatch_key_override}

// A schedule block measures and submits, and returns the future without
// waiting: the parent waits after the block, and the block's shape makes the
// future available at its local due time (300 + 1 + 1000 = 1301).
// CHECK-LABEL: func.func @measure_and_submit
// CHECK: classical.submit {{.*}}start_cycles = 300 : i64}
// CHECK: classical.wait {{.*}}start_cycles = 1301 : i64}
func.func @measure_and_submit() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %r0, %rt0 = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %c300 = arith.constant 300 : i64
  %a = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %ro = pulse.square %c300, %a, %z : i64, f64, f64 -> !pulse.waveform
  %l0 = classical.get_link {device = 1 : i64} : !classical.link
  %r1, %rt1, %l1, %f = pulse.schedule_block alap (%r0, %rt0, %l0)
      : (!pulse.readout_line, !pulse.tone, !classical.link)
      -> (!pulse.readout_line, !pulse.tone, !classical.link, !classical.future) {
  ^bb0(%r: !pulse.readout_line, %rt: !pulse.tone, %l: !classical.link):
    %r2, %rt2, %m = pulse.readout %r, %ro, %rt, iq : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    %l2, %g = classical.submit %l, @decode_wide(%m) : (!pulse.measurement) -> (!classical.link, !classical.future)
    pulse.yield %r2, %rt2, %l2, %g : !pulse.readout_line, !pulse.tone, !classical.link, !classical.future
  }
  %fix = classical.wait %f, @decode_wide : i1
  return
}

}
