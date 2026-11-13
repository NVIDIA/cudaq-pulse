// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A schedule block that submits is placed where its requests fit the slots
// the requests around it leave on their link.
// RUN: cudaq-pulse-opt --pulse-schedule-asap %s | FileCheck %s
// RUN: cudaq-pulse-opt --pulse-schedule-alap %s | FileCheck %s
// RUN: cudaq-pulse-opt --pulse-schedule-asap --pulse-classical-check --pulse-verify %s > /dev/null
// RUN: cudaq-pulse-opt --pulse-schedule-alap --pulse-classical-check --pulse-verify %s > /dev/null

module attributes {pulse.clock_ghz = 1.0 : f64, classical.links = {"0" = {
    fixed_ns = 0.0 : f64, bytes_per_ns = 0.0 : f64, feedback_ns = 0.0 : f64,
    slots = 2 : i64, slot_bytes = 1024 : i64}}} {

classical.callable @decode {dispatch_key = 1, device = 0 : i64,
    args = ["bits:1"], results = ["bits:1"], latency_min_ns = 0.0,
    latency_nominal_ns = 100.0, latency_max_ns = 100.0, mode = "bounded",
    submit = "eager", dispatch_key_override}

// The parent's request holds one of the two slots over [10, 111). The block
// sends two requests 1 cycle apart, so it starts once the first slot is free
// again: its second request goes out at 111, its readout at 100.
// CHECK-LABEL: func.func @block_waits_for_a_slot
// CHECK: pulse.readout {{.*}}start_cycles = 0 : i64}
// CHECK: classical.submit {{.*}}start_cycles = 10 : i64}
// CHECK: pulse.readout {{.*}}start_cycles = 100 : i64}
// CHECK: classical.submit {{.*}}start_cycles = 110 : i64}
// CHECK: classical.submit {{.*}}start_cycles = 111 : i64}
func.func @block_waits_for_a_slot() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %r0, %rt0 = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %c10 = arith.constant 10 : i64
  %a = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %ro = pulse.square %c10, %a, %z : i64, f64, f64 -> !pulse.waveform
  %r1, %rt1, %m = pulse.readout %r0, %ro, %rt0, iq : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %l0 = classical.get_link {device = 0 : i64} : !classical.link
  %l1, %f = classical.submit %l0, @decode(%m) : (!pulse.measurement) -> (!classical.link, !classical.future)
  %r2, %rt2, %l2, %g, %h = pulse.schedule_block asap (%r1, %rt1, %l1)
      : (!pulse.readout_line, !pulse.tone, !classical.link)
      -> (!pulse.readout_line, !pulse.tone, !classical.link, !classical.future, !classical.future) {
  ^bb0(%r: !pulse.readout_line, %rt: !pulse.tone, %l: !classical.link):
    %r3, %rt3, %n = pulse.readout %r, %ro, %rt, iq : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    %l3, %g0 = classical.submit %l, @decode(%n) : (!pulse.measurement) -> (!classical.link, !classical.future)
    %l4, %h0 = classical.submit %l3, @decode(%n) : (!pulse.measurement) -> (!classical.link, !classical.future)
    pulse.yield %r3, %rt3, %l4, %g0, %h0 : !pulse.readout_line, !pulse.tone, !classical.link, !classical.future, !classical.future
  }
  %x = classical.wait %f, @decode : i1
  %y = classical.wait %g, @decode : i1
  %w = classical.wait %h, @decode : i1
  return
}

}
