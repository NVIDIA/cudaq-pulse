// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A link with one instance per controller gives each controller its own
// slots: two calls on two controllers over a link of one slot go out at once.
// RUN: cudaq-pulse-opt --pulse-schedule-asap %s | FileCheck %s
// RUN: cudaq-pulse-opt --pulse-schedule-alap %s | FileCheck %s
// RUN: cudaq-pulse-opt --pulse-schedule-asap --pulse-classical-check --pulse-verify %s > /dev/null
// RUN: cudaq-pulse-opt --pulse-schedule-alap --pulse-classical-check --pulse-verify %s > /dev/null

module attributes {pulse.clock_ghz = 1.0 : f64, pulse.n_qubits = 2 : i64,
    pulse.controllers = {drive = array<i64: 0, 1>, readout = array<i64: 0, 1>,
      outcome_latency = array<i64: 0, 0, 0, 0>, spread = array<i64: 0, 0, 0, 0>,
      uncertainty_ns = array<f64: 0.5, 0.5>},
    classical.links = {"0" = {fixed_ns = 0.0 : f64, bytes_per_ns = 0.0 : f64,
      feedback_ns = 0.0 : f64, max_in_flight = 1 : i64, slot_bytes = 1024 : i64,
      per_controller, name = "decoder"}}} {

classical.callable @decode {dispatch_key = 1, device = 0 : i64,
    args = ["bits:1"], results = ["f64"], latency_min_ns = 0.0,
    latency_nominal_ns = 100.0, latency_max_ns = 100.0, mode = "bounded",
    submit = "eager", dispatch_key_override}

// CHECK-LABEL: func.func @each_controller_its_own_slot
// CHECK: classical.submit {{.*}}start_cycles = 10 : i64}> {{.*}}classical.instance = 0 : i64
// CHECK: classical.wait {{.*}}classical.instance = 0 : i64
// CHECK: classical.submit {{.*}}start_cycles = 10 : i64}> {{.*}}classical.instance = 1 : i64
// CHECK: classical.wait {{.*}}classical.instance = 1 : i64
func.func @each_controller_its_own_slot() {
  %c10 = arith.constant 10 : i64
  %c8 = arith.constant 8 : i64
  %a = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %ro = pulse.square %c10, %a, %z : i64, f64, f64 -> !pulse.waveform
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %r0, %rt0 = pulse.get_readout_line %q0 : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %d0, %dt0 = pulse.get_drive_line %q0 : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %r1, %rt1 = pulse.get_readout_line %q1 : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %d1, %dt1 = pulse.get_drive_line %q1 : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %r0a, %rt0a, %m0 = pulse.readout %r0, %ro, %rt0, iq : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %r1a, %rt1a, %m1 = pulse.readout %r1, %ro, %rt1, iq : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %link = classical.get_link {device = 0 : i64} : !classical.link
  %link1, %f0 = classical.submit %link, @decode(%m0) : (!pulse.measurement) -> (!classical.link, !classical.future)
  %x0 = classical.wait %f0, @decode : f64
  %w0 = pulse.square %c8, %x0, %z : i64, f64, f64 -> !pulse.waveform
  %d0a, %dt0a = pulse.drive %d0, %w0, %dt0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %link2, %f1 = classical.submit %link1, @decode(%m1) : (!pulse.measurement) -> (!classical.link, !classical.future)
  %x1 = classical.wait %f1, @decode : f64
  %w1 = pulse.square %c8, %x1, %z : i64, f64, f64 -> !pulse.waveform
  %d1a, %dt1a = pulse.drive %d1, %w1, %dt1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

}
