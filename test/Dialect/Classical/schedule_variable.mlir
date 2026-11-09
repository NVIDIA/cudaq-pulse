// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A variable wait (its length from an answer) is a barrier scheduled at its
// lowest length for the lines that depend on it: nothing after it on them
// starts before it ends, and under ALAP nothing before it ends after it
// starts. Lines nothing ties to it keep their time.
// RUN: cudaq-pulse-opt --pulse-schedule-asap %s | FileCheck %s --check-prefixes=CHECK,ASAP
// RUN: cudaq-pulse-opt --pulse-schedule-alap %s | FileCheck %s --check-prefixes=CHECK,ALAP
// RUN: cudaq-pulse-opt --pulse-schedule-rcp %s | FileCheck %s --check-prefixes=CHECK,ASAP
// RUN: cudaq-pulse-opt --pulse-schedule-alap --pulse-classical-check --pulse-verify %s > /dev/null

module attributes {pulse.clock_ghz = 1.0 : f64} {

classical.callable @tau {function_id = 1 : i64, device = 0 : i64,
    args = [], results = ["i64"], latency_min_ns = 0.0,
    latency_nominal_ns = 50.0, latency_max_ns = 100.0, mode = "bounded",
    submit = "eager", function_id_override}

// The answer is due at 101. Line e runs 0..300 and syncs with d after the
// wait, so the barrier waits for it and the wait starts at 300 (length 16),
// then the drive after it at 316. Under ALAP the drive on line d before the
// wait slides up to it (260).
// CHECK-LABEL: func.func @barrier
// CHECK-NOT: classical.timeline
// ASAP: pulse.drive {{.*}}start_cycles = 0 : i64
// ALAP: pulse.drive {{.*}}start_cycles = 260 : i64
// CHECK: pulse.wait {{.*}}classical.variable{{.*}}duration_cycles = 16 : i64, {{.*}}start_cycles = 300 : i64}
// CHECK: pulse.drive {{.*}}start_cycles = 316 : i64
func.func @barrier() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %p = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %e0, %u0 = pulse.get_drive_line %p : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %c300 = arith.constant 300 : i64
  %a = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !pulse.waveform
  %long = pulse.square %c300, %a, %z : i64, f64, f64 -> !pulse.waveform
  %e1, %u1 = pulse.drive %e0, %long, %u0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %d1, %t1 = pulse.drive %d0, %x, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %l = classical.get_link {device = 0 : i64} : !classical.link
  %l1, %h = classical.submit %l, @tau() : () -> (!classical.link, !classical.future)
  %n = classical.wait %h, @tau : i64
  %dur = pulse.duration_from_int %n : (i64) -> !pulse.duration
  %d2 = pulse.wait %d1, %dur {classical.variable, pulse.variable_wait, pulse.min_cycles = 16 : i64, pulse.max_cycles = 1000 : i64} : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  %d3, %t3 = pulse.drive %d2, %x, %t1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %d4, %e2 = pulse.sync %d3, %e1 : !pulse.drive_line, !pulse.drive_line -> !pulse.drive_line, !pulse.drive_line
  return
}

// Without the sync, line e does not depend on the wait: it starts when its
// answer is due (101) while e plays, and e keeps its time. Under ALAP the
// drive before the wait ends where the wait starts (61), and e, which ends
// last, is not moved to the end of d.
// CHECK-LABEL: func.func @independent
// CHECK: pulse.get_drive_line {{.*}}classical.timeline = 0 : i64
// CHECK: pulse.get_drive_line {{.*}}classical.timeline = 1 : i64
// CHECK: pulse.drive {{.*}}start_cycles = 0 : i64
// ASAP: pulse.drive {{.*}}start_cycles = 0 : i64
// ALAP: pulse.drive {{.*}}start_cycles = 61 : i64
// CHECK: pulse.wait {{.*}}classical.variable{{.*}}duration_cycles = 16 : i64, {{.*}}start_cycles = 101 : i64}
// CHECK: pulse.drive {{.*}}start_cycles = 117 : i64
func.func @independent() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %p = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %e0, %u0 = pulse.get_drive_line %p : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %c300 = arith.constant 300 : i64
  %a = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !pulse.waveform
  %long = pulse.square %c300, %a, %z : i64, f64, f64 -> !pulse.waveform
  %e1, %u1 = pulse.drive %e0, %long, %u0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %d1, %t1 = pulse.drive %d0, %x, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %l = classical.get_link {device = 0 : i64} : !classical.link
  %l1, %h = classical.submit %l, @tau() : () -> (!classical.link, !classical.future)
  %n = classical.wait %h, @tau : i64
  %dur = pulse.duration_from_int %n : (i64) -> !pulse.duration
  %d2 = pulse.wait %d1, %dur {classical.variable, pulse.variable_wait, pulse.min_cycles = 16 : i64, pulse.max_cycles = 1000 : i64} : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  %d3, %t3 = pulse.drive %d2, %x, %t1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// One barrier per iteration. A round's wait starts when its answer is due
// (1 + 100) and ends 5 later; nothing after it starts earlier, so the next
// round's submission starts at that end and rounds repeat every 106. The
// drive after the wait overlaps the next submission. Under ALAP the drive
// slides up to the next barrier, and the last round, which has none after it,
// is peeled off the loop.
// CHECK-LABEL: func.func @rounds
// ASAP: classical.submit {{.*}}period_cycles = array<i64: 106>, start_cycles = 0 : i64}
// ASAP: pulse.wait {{.*}}classical.variable{{.*}}period_cycles = array<i64: 106>, {{.*}}start_cycles = 101 : i64}
// ASAP: pulse.drive {{.*}}period_cycles = array<i64: 106>, start_cycles = 106 : i64}
// ALAP: pulse.drive {{.*}}period_cycles = array<i64: 106>, start_cycles = 167 : i64}
// ALAP: pulse.drive {{.*}}start_cycles = 424 : i64}
func.func @rounds() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %a = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !pulse.waveform
  %l0 = classical.get_link {device = 0 : i64} : !classical.link
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 1 : index
  %dn, %tn, %ln = scf.for %i = %lb to %ub step %step
      iter_args(%d = %d0, %t = %t0, %l = %l0)
      -> (!pulse.drive_line, !pulse.tone, !classical.link) {
    %l1, %h = classical.submit %l, @tau() : () -> (!classical.link, !classical.future)
    %n = classical.wait %h, @tau : i64
    %dur = pulse.duration_from_int %n : (i64) -> !pulse.duration
    %d1 = pulse.wait %d, %dur {classical.variable, pulse.variable_wait, pulse.min_cycles = 5 : i64} : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
    %d2, %t2 = pulse.drive %d1, %x, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %d2, %t2, %l1 : !pulse.drive_line, !pulse.tone, !classical.link
  }
  return
}

// A loop that runs carries a measurement its body does not read: what it
// yields is the readout on r in it, so the readout on s before the loop
// stays on a timeline of its own.
// CHECK-LABEL: func.func @carried_unread
// CHECK: pulse.get_drive_line {{.*}}classical.timeline = 0 : i64
// CHECK: pulse.get_readout_line {{.*}}classical.timeline = 0 : i64
// CHECK: pulse.get_readout_line {{.*}}classical.timeline = 1 : i64

func.func @carried_unread() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %p = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %r0, %v0 = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %s0, %w0 = pulse.get_readout_line %p : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %a = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !pulse.waveform
  %s1, %w1, %m0 = pulse.readout %s0, %x, %w0, iq : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %lb = arith.constant 0 : index
  %ub = arith.constant 1 : index
  %step = arith.constant 1 : index
  %r2, %v2, %m = scf.for %i = %lb to %ub step %step iter_args(%r = %r0, %v = %v0, %mi = %m0) -> (!pulse.readout_line, !pulse.tone, !pulse.measurement) {
    %r1, %v1, %m1 = pulse.readout %r, %x, %v, iq : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    scf.yield %r1, %v1, %m1 : !pulse.readout_line, !pulse.tone, !pulse.measurement
  }
  %b = pulse.is_set %m : !pulse.measurement -> i1
  %d1, %t1 = scf.if %b -> (!pulse.drive_line, !pulse.tone) {
    %d5, %t5 = pulse.drive %d0, %x, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %d5, %t5 : !pulse.drive_line, !pulse.tone
  } else {
    scf.yield %d0, %t0 : !pulse.drive_line, !pulse.tone
  }
  %l = classical.get_link {device = 0 : i64} : !classical.link
  %l1, %h = classical.submit %l, @tau() : () -> (!classical.link, !classical.future)
  %n = classical.wait %h, @tau : i64
  %dur = pulse.duration_from_int %n : (i64) -> !pulse.duration
  %d2 = pulse.wait %d1, %dur {classical.variable, pulse.variable_wait, pulse.min_cycles = 16 : i64, pulse.max_cycles = 1000 : i64} : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  return
}

// A loop of no trips yields the measurement it is given: the condition is
// then the readout on s, which joins d's timeline, and every line is in one.
// CHECK-LABEL: func.func @carried_no_trips
// CHECK-NOT: classical.timeline
// CHECK: return

func.func @carried_no_trips() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %p = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %r0, %v0 = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %s0, %w0 = pulse.get_readout_line %p : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %a = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !pulse.waveform
  %s1, %w1, %m0 = pulse.readout %s0, %x, %w0, iq : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %lb = arith.constant 0 : index
  %ub = arith.constant 0 : index
  %step = arith.constant 1 : index
  %r2, %v2, %m = scf.for %i = %lb to %ub step %step iter_args(%r = %r0, %v = %v0, %mi = %m0) -> (!pulse.readout_line, !pulse.tone, !pulse.measurement) {
    %r1, %v1, %m1 = pulse.readout %r, %x, %v, iq : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    scf.yield %r1, %v1, %m1 : !pulse.readout_line, !pulse.tone, !pulse.measurement
  }
  %b = pulse.is_set %m : !pulse.measurement -> i1
  %d1, %t1 = scf.if %b -> (!pulse.drive_line, !pulse.tone) {
    %d5, %t5 = pulse.drive %d0, %x, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %d5, %t5 : !pulse.drive_line, !pulse.tone
  } else {
    scf.yield %d0, %t0 : !pulse.drive_line, !pulse.tone
  }
  %l = classical.get_link {device = 0 : i64} : !classical.link
  %l1, %h = classical.submit %l, @tau() : () -> (!classical.link, !classical.future)
  %n = classical.wait %h, @tau : i64
  %dur = pulse.duration_from_int %n : (i64) -> !pulse.duration
  %d2 = pulse.wait %d1, %dur {classical.variable, pulse.variable_wait, pulse.min_cycles = 16 : i64, pulse.max_cycles = 1000 : i64} : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  return
}

}
