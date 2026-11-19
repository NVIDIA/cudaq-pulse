// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s --pulse-verify --pulse-schedule-asap | FileCheck %s

// An integrated readout gives its IQ point, !pulse.iq_data, which is
// scheduled as the level of a readout of mode level is: it is available when
// the readout ends, so a call that takes it starts no earlier, and it is
// carried through a loop and out of a schedule block as a measurement is.

module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000>,
    classical.links = {"0" = {bytes_per_ns = 0.0 : f64, feedback_ns = 0.0 : f64,
                              fixed_ns = 0.0 : f64, slots = 64 : i64,
                              slot_bytes = 1024 : i64}}} {

classical.callable @fit {args = ["iq:1"], device = 0 : i64,
    dispatch_key = 2934209580 : i64, latency_max_ns = 100.0 : f64,
    latency_min_ns = 0.0 : f64, latency_nominal_ns = 100.0 : f64,
    mode = "bounded", results = ["f64"], submit = "eager"}

// CHECK-LABEL: func.func @sends_a_level
// CHECK: pulse.readout {{.*}}, level <{duration_cycles = 400 : i64, start_cycles = 0 : i64}>
// CHECK: classical.submit {{.*}}start_cycles = 400 : i64
func.func @sends_a_level() {
  %c400 = arith.constant 400 : i64
  %amp = arith.constant 0.05 : f64
  %zero = arith.constant 0.0 : f64
  %w = pulse.square %c400, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %r0, %t0 = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %r1, %t1, %m = pulse.readout %r0, %w, %t0, level : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %l0 = classical.get_link {device = 0 : i64} : !classical.link
  %l1, %h = classical.submit %l0, @fit(%m) : (!pulse.measurement) -> (!classical.link, !classical.future)
  return
}

// CHECK-LABEL: func.func @sends_a_point
// CHECK: pulse.readout {{.*}}, integrated <{duration_cycles = 400 : i64, start_cycles = 0 : i64}>
// CHECK: classical.submit {{.*}}start_cycles = 400 : i64
func.func @sends_a_point() {
  %c400 = arith.constant 400 : i64
  %amp = arith.constant 0.05 : f64
  %zero = arith.constant 0.0 : f64
  %w = pulse.square %c400, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %r0, %t0 = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %r1, %t1, %p = pulse.readout %r0, %w, %t0, integrated : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.iq_data
  %l0 = classical.get_link {device = 0 : i64} : !classical.link
  %l1, %h = classical.submit %l0, @fit(%p) : (!pulse.iq_data) -> (!classical.link, !classical.future)
  return
}

// The loop's last point is ready when its third readout ends, at 1600; the
// block's when the block's readout ends, at 2000.
// CHECK-LABEL: func.func @carried_and_blocked
// CHECK: classical.submit {{.*}}start_cycles = 1600 : i64
// CHECK: classical.submit {{.*}}start_cycles = 2000 : i64
func.func @carried_and_blocked() {
  %c400 = arith.constant 400 : i64
  %amp = arith.constant 0.05 : f64
  %zero = arith.constant 0.0 : f64
  %w = pulse.square %c400, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %r0, %t0 = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %r1, %t1, %p0 = pulse.readout %r0, %w, %t0, integrated : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.iq_data
  %lb = arith.constant 0 : index
  %ub = arith.constant 3 : index
  %step = arith.constant 1 : index
  %rn, %tn, %pn = scf.for %i = %lb to %ub step %step iter_args(%r = %r1, %t = %t1, %p = %p0) -> (!pulse.readout_line, !pulse.tone, !pulse.iq_data) {
    %r2, %t2, %p2 = pulse.readout %r, %w, %t, integrated : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.iq_data
    scf.yield %r2, %t2, %p2 : !pulse.readout_line, !pulse.tone, !pulse.iq_data
  }
  %rb, %tb, %pb = pulse.schedule_block asap (%rn, %tn) : (!pulse.readout_line, !pulse.tone) -> (!pulse.readout_line, !pulse.tone, !pulse.iq_data) {
  ^bb0(%r: !pulse.readout_line, %t: !pulse.tone):
    %r3, %t3, %p3 = pulse.readout %r, %w, %t, integrated : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.iq_data
    pulse.yield %r3, %t3, %p3 : !pulse.readout_line, !pulse.tone, !pulse.iq_data
  }
  %l0 = classical.get_link {device = 0 : i64} : !classical.link
  %l1, %h1 = classical.submit %l0, @fit(%pn) : (!pulse.iq_data) -> (!classical.link, !classical.future)
  %l2, %h2 = classical.submit %l1, @fit(%pb) : (!pulse.iq_data) -> (!classical.link, !classical.future)
  return
}

}
