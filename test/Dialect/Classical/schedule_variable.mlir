// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// A variable wait (its length from an answer) is a barrier scheduled at its
// lowest length: nothing after it starts before it ends, and under ALAP
// nothing before it ends after it starts.
// RUN: cudaq-pulse-opt --pulse-schedule-asap %s | FileCheck %s --check-prefixes=CHECK,ASAP
// RUN: cudaq-pulse-opt --pulse-schedule-alap %s | FileCheck %s --check-prefixes=CHECK,ALAP
// RUN: cudaq-pulse-opt --pulse-schedule-rcp %s | FileCheck %s --check-prefixes=CHECK,ASAP
// RUN: cudaq-pulse-opt --pulse-schedule-alap --pulse-classical-check --pulse-verify %s > /dev/null

module attributes {pulse.clock_ghz = 1.0 : f64} {

classical.callable @tau {function_id = 1 : i64, device = 0 : i64,
    args = [], results = ["i64"], latency_min_ns = 0.0,
    latency_nominal_ns = 50.0, latency_max_ns = 100.0, mode = "bounded",
    submit = "eager", function_id_override}

// The answer is due at 101. Line e runs 0..300, so the barrier waits for it
// and the wait starts at 300 (length 16), then the drive after it at 316.
// Under ALAP the drive on line d before the wait slides up to it (260).
// CHECK-LABEL: func.func @barrier
// ASAP: pulse.drive {{.*}}start_vtu = 0 : i64
// ALAP: pulse.drive {{.*}}start_vtu = 260 : i64
// CHECK: pulse.wait {{.*}}classical.variable{{.*}}duration_vtu = 16 : i64, {{.*}}start_vtu = 300 : i64}
// CHECK: pulse.drive {{.*}}start_vtu = 316 : i64
func.func @barrier() {
  %q = pulse.qudit_alloc : !pulse.qref
  %p = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %e0, %u0 = pulse.get_drive_line %p : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
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
  %d2 = pulse.wait %d1, %dur {classical.variable, pulse.variable, pulse.min_vtu = 16 : i64, pulse.max_vtu = 1000 : i64} : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
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
// ASAP: classical.submit {{.*}}period_vtu = array<i64: 106>, start_vtu = 0 : i64}
// ASAP: pulse.wait {{.*}}classical.variable{{.*}}period_vtu = array<i64: 106>, {{.*}}start_vtu = 101 : i64}
// ASAP: pulse.drive {{.*}}period_vtu = array<i64: 106>, start_vtu = 106 : i64}
// ALAP: pulse.drive {{.*}}period_vtu = array<i64: 106>, start_vtu = 167 : i64}
// ALAP: pulse.drive {{.*}}start_vtu = 424 : i64}
func.func @rounds() {
  %q = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
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
    %d1 = pulse.wait %d, %dur {classical.variable, pulse.variable, pulse.min_vtu = 5 : i64} : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
    %d2, %t2 = pulse.drive %d1, %x, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %d2, %t2, %l1 : !pulse.drive_line, !pulse.tone, !classical.link
  }
  return
}

}
