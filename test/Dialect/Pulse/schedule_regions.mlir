// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Scheduling through scf.for and measurement-conditioned scf.if.
// RUN: cudaq-pulse-opt --pulse-schedule-asap %s | FileCheck %s --check-prefixes=CHECK,ASAP
// RUN: cudaq-pulse-opt --pulse-schedule-alap %s | FileCheck %s --check-prefixes=CHECK,ALAP
// RUN: cudaq-pulse-opt --pulse-schedule-asap --pulse-verify %s > /dev/null
// RUN: cudaq-pulse-opt --pulse-schedule-alap --pulse-verify %s > /dev/null

// The loop is kept: each iteration starts 240 after the previous one, and
// the wait after it starts once all eight have run.
// CHECK-LABEL: func.func @dd_loop
// CHECK: scf.for
// CHECK: pulse.drive {{.*}}{duration_cycles = 40 : i64, period_cycles = array<i64: 240>, start_cycles = 100 : i64}
// CHECK: pulse.wait {{.*}}{duration_cycles = 200 : i64, period_cycles = array<i64: 240>, start_cycles = 140 : i64}
// CHECK: pulse.wait {{.*}}{duration_cycles = 10 : i64, start_cycles = 2020 : i64}
func.func @dd_loop() {
  %q = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %amp = arith.constant 0.047 : f64
  %zero = arith.constant 0.0 : f64
  %x = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %c100 = arith.constant 100 : i64
  %w100 = pulse.duration_from_int %c100 : (i64) -> !pulse.duration
  %c200 = arith.constant 200 : i64
  %w200 = pulse.duration_from_int %c200 : (i64) -> !pulse.duration
  %c10 = arith.constant 10 : i64
  %w10 = pulse.duration_from_int %c10 : (i64) -> !pulse.duration
  %d1 = pulse.wait %d0, %w100 : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  %lb = arith.constant 0 : index
  %ub = arith.constant 8 : index
  %step = arith.constant 1 : index
  %d2, %t2 = scf.for %i = %lb to %ub step %step iter_args(%d = %d1, %t = %t0) -> (!pulse.drive_line, !pulse.tone) {
    %dd, %tt = pulse.drive %d, %x, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %dw = pulse.wait %dd, %w200 : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
    scf.yield %dw, %tt : !pulse.drive_line, !pulse.tone
  }
  %d3 = pulse.wait %d2, %w10 : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  return
}

// Line b is busy until 500, so the first iteration's drive runs early and
// later ones wait for the sync: ASAP peels the first iteration off the loop.
// CHECK-LABEL: func.func @transient
// ASAP: pulse.drive {{.*}}start_cycles = 0 : i64
// ASAP: scf.for %{{.*}} = %c1{{.*}} to %c6
// ASAP: pulse.drive {{.*}}period_cycles = array<i64: 40>, start_cycles = 500 : i64
// ALAP: scf.for %{{.*}} = %c0 to %c6
// ALAP: pulse.drive {{.*}}period_cycles = array<i64: 40>, start_cycles = 460 : i64
func.func @transient(%x: !pulse.waveform) {
  %qa = pulse.qudit_alloc : !pulse.qref
  %qb = pulse.qudit_alloc : !pulse.qref
  %a0, %ta = pulse.get_drive_line %qa : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %b0, %tb = pulse.get_drive_line %qb : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %amp = arith.constant 0.047 : f64
  %zero = arith.constant 0.0 : f64
  %w = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %c500 = arith.constant 500 : i64
  %w500 = pulse.duration_from_int %c500 : (i64) -> !pulse.duration
  %b1 = pulse.wait %b0, %w500 : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  %lb = arith.constant 0 : index
  %ub = arith.constant 6 : index
  %step = arith.constant 1 : index
  %a2, %b2, %ta2 = scf.for %i = %lb to %ub step %step iter_args(%a = %a0, %b = %b1, %t = %ta) -> (!pulse.drive_line, !pulse.drive_line, !pulse.tone) {
    %ad, %td = pulse.drive %a, %w, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %as, %bs = pulse.sync %ad, %b : !pulse.drive_line, !pulse.drive_line -> !pulse.drive_line, !pulse.drive_line
    scf.yield %as, %bs, %td : !pulse.drive_line, !pulse.drive_line, !pulse.tone
  }
  return
}

// The conditional X waits for the readout; the wait after the if waits for
// the longer arm.
// CHECK-LABEL: func.func @active_reset
// CHECK: pulse.readout {{.*}}{duration_cycles = 300 : i64, start_cycles = 0 : i64}
// CHECK: scf.if
// CHECK: pulse.drive {{.*}}{duration_cycles = 40 : i64, start_cycles = 300 : i64}
// CHECK: pulse.wait {{.*}}{duration_cycles = 10 : i64, start_cycles = 340 : i64}
func.func @active_reset() {
  %q = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %r0, %rt0 = pulse.get_readout_line %q : (!pulse.qref) -> (!pulse.readout_line, !pulse.tone)
  %c300 = arith.constant 300 : i64
  %c40 = arith.constant 40 : i64
  %amp = arith.constant 0.05 : f64
  %zero = arith.constant 0.0 : f64
  %ro = pulse.square %c300, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %x = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %c10 = arith.constant 10 : i64
  %w10 = pulse.duration_from_int %c10 : (i64) -> !pulse.duration
  %r1, %rt1, %m = pulse.readout %r0, %ro, %rt0, "iq" : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %bit = pulse.is_set %m : !pulse.measurement -> i1
  %d1, %t1 = scf.if %bit -> (!pulse.drive_line, !pulse.tone) {
    %d, %t = pulse.drive %d0, %x, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %d, %t : !pulse.drive_line, !pulse.tone
  } else {
    scf.yield %d0, %t0 : !pulse.drive_line, !pulse.tone
  }
  %d2 = pulse.wait %d1, %w10 : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  return
}

// Each conditional drive waits for its own iteration's readout. ASAP plays it
// right after the readout; ALAP packs the drives at the end of the schedule.
// CHECK-LABEL: func.func @feedback_in_loop
// CHECK: pulse.readout {{.*}}{duration_cycles = 300 : i64, period_cycles = array<i64: 300>, start_cycles = 0 : i64}
// ASAP: pulse.drive {{.*}}{duration_cycles = 40 : i64, period_cycles = array<i64: 300>, start_cycles = 300 : i64}
// ALAP: pulse.drive {{.*}}{duration_cycles = 40 : i64, period_cycles = array<i64: 40>, start_cycles = 1340 : i64}
func.func @feedback_in_loop() {
  %q = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %r0, %rt0 = pulse.get_readout_line %q : (!pulse.qref) -> (!pulse.readout_line, !pulse.tone)
  %c300 = arith.constant 300 : i64
  %c40 = arith.constant 40 : i64
  %amp = arith.constant 0.05 : f64
  %zero = arith.constant 0.0 : f64
  %ro = pulse.square %c300, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %x = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %lb = arith.constant 0 : index
  %ub = arith.constant 5 : index
  %step = arith.constant 1 : index
  %d5, %t5, %r5, %rt5 = scf.for %i = %lb to %ub step %step iter_args(%d = %d0, %t = %t0, %r = %r0, %rt = %rt0) -> (!pulse.drive_line, !pulse.tone, !pulse.readout_line, !pulse.tone) {
    %r1, %rt1, %m = pulse.readout %r, %ro, %rt, "iq" : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    %bit = pulse.is_set %m : !pulse.measurement -> i1
    %d1, %t1 = scf.if %bit -> (!pulse.drive_line, !pulse.tone) {
      %dd, %tt = pulse.drive %d, %x, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
      scf.yield %dd, %tt : !pulse.drive_line, !pulse.tone
    } else {
      scf.yield %d, %t : !pulse.drive_line, !pulse.tone
    }
    scf.yield %d1, %t1, %r1, %rt1 : !pulse.drive_line, !pulse.tone, !pulse.readout_line, !pulse.tone
  }
  return
}
