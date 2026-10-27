// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// Scheduling classical calls. At 2 GHz a 1500 ns latency is 3000 cycles; a
// submission without link parameters holds the link for one cycle.
// RUN: cudaq-pulse-opt --pulse-schedule-asap %s | FileCheck %s --check-prefixes=CHECK,ASAP
// RUN: cudaq-pulse-opt --pulse-schedule-alap %s | FileCheck %s --check-prefixes=CHECK,ALAP
// RUN: cudaq-pulse-opt --pulse-schedule-asap --pulse-classical-check --pulse-verify %s > /dev/null
// RUN: cudaq-pulse-opt --pulse-schedule-alap --pulse-classical-check --pulse-verify %s > /dev/null

module attributes {pulse.clock_ghz = 2.0 : f64} {

classical.callable @decode {function_id = 3007678287 : i64, device = 0 : i64,
    args = ["bits:1"], results = ["bits:1"], latency_min_ns = 500.0,
    latency_nominal_ns = 900.0, latency_max_ns = 1500.0, mode = "bounded",
    submit = "eager"}
classical.callable @est {function_id = 2023343271 : i64, device = 0 : i64,
    args = ["iq:1"], results = ["f64"], latency_min_ns = 0.0,
    latency_nominal_ns = 1000.0, latency_max_ns = 1500.0, mode = "bounded",
    submit = "lazy"}
classical.callable @track {function_id = 2067622972 : i64, device = 1 : i64,
    args = ["f64"], results = ["f64"], latency_min_ns = 0.0,
    latency_nominal_ns = 10.0, latency_max_ns = 50.0, mode = "blocking",
    submit = "eager"}

// Readout 0..300, submit 300..301, answer due 3301; the conditional drive
// starts there under both policies, and the eager submission stays at 300.
// CHECK-LABEL: func.func @feedback
// CHECK: pulse.readout {{.*}}start_vtu = 0
// CHECK: classical.submit {{.*}}classical.budget_vtu = 3000 : i64{{.*}}classical.submit_vtu = 1 : i64{{.*}}duration_vtu = 1 : i64, start_vtu = 300 : i64}
// CHECK: classical.wait {{.*}}duration_vtu = 0 : i64, start_vtu = 3301 : i64}
// CHECK: pulse.drive {{.*}}start_vtu = 3301 : i64
func.func @feedback() {
  %q = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %r0, %rt0 = pulse.get_readout_line %q : (!pulse.qref) -> (!pulse.readout_line, !pulse.tone)
  %c300 = arith.constant 300 : i64
  %c40 = arith.constant 40 : i64
  %a = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %ro = pulse.square %c300, %a, %z : i64, f64, f64 -> !pulse.waveform
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !pulse.waveform
  %r1, %rt1, %m = pulse.readout %r0, %ro, %rt0, "iq" : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %l0 = classical.get_link {device = 0 : i64} : !classical.link
  %l1, %f = classical.submit %l0, @decode(%m) : (!pulse.measurement) -> (!classical.link, !classical.future)
  %fix = classical.wait %f, @decode : i1
  %d1, %t1 = scf.if %fix -> (!pulse.drive_line, !pulse.tone) {
    %dd, %tt = pulse.drive %d0, %x, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %dd, %tt : !pulse.drive_line, !pulse.tone
  } else {
    scf.yield %d0, %t0 : !pulse.drive_line, !pulse.tone
  }
  return
}

// The answer sets an amplitude: that drive waits for it, the drive on the
// other qudit does not. The lazy submission floats under ALAP, so its answer is
// due just when the drive needs it, at the 5000-cycle makespan minus 40.
// CHECK-LABEL: func.func @parameter
// CHECK: pulse.drive {{.*}}start_vtu = 0 : i64
// ASAP: classical.submit {{.*}}start_vtu = 300 : i64}
// ALAP: classical.submit {{.*}}start_vtu = 1959 : i64}
// ASAP: classical.wait {{.*}}start_vtu = 3301 : i64}
// ALAP: classical.wait {{.*}}start_vtu = 4960 : i64}
// ASAP: pulse.drive {{.*}}start_vtu = 3301 : i64
// ALAP: pulse.drive {{.*}}start_vtu = 4960 : i64
func.func @parameter() {
  %q = pulse.qudit_alloc : !pulse.qref
  %p = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %e0, %u0 = pulse.get_drive_line %p : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %r0, %rt0 = pulse.get_readout_line %q : (!pulse.qref) -> (!pulse.readout_line, !pulse.tone)
  %c300 = arith.constant 300 : i64
  %c40 = arith.constant 40 : i64
  %c5000 = arith.constant 5000 : i64
  %a = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %long = pulse.square %c5000, %a, %z : i64, f64, f64 -> !pulse.waveform
  %e1, %u1 = pulse.drive %e0, %long, %u0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %ro = pulse.square %c300, %a, %z : i64, f64, f64 -> !pulse.waveform
  %r1, %rt1, %m = pulse.readout %r0, %ro, %rt0, "iq" : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %l0 = classical.get_link {device = 0 : i64} : !classical.link
  %l1, %f = classical.submit %l0, @est(%m) : (!pulse.measurement) -> (!classical.link, !classical.future)
  %amp = classical.wait %f, @est : f64
  %x = pulse.square %c40, %amp, %z : i64, f64, f64 -> !pulse.waveform
  %d1, %t1 = pulse.drive %d0, %x, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// Two submissions on one link go out one after the other, in program order.
// CHECK-LABEL: func.func @link_order
// CHECK: classical.submit {{.*}}start_vtu = 300 : i64}
// CHECK: classical.submit {{.*}}start_vtu = 301 : i64}
func.func @link_order() {
  %q = pulse.qudit_alloc : !pulse.qref
  %r0, %rt0 = pulse.get_readout_line %q : (!pulse.qref) -> (!pulse.readout_line, !pulse.tone)
  %c300 = arith.constant 300 : i64
  %a = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %ro = pulse.square %c300, %a, %z : i64, f64, f64 -> !pulse.waveform
  %r1, %rt1, %m = pulse.readout %r0, %ro, %rt0, "iq" : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %l0 = classical.get_link {device = 0 : i64} : !classical.link
  %l1, %f = classical.submit %l0, @decode(%m) : (!pulse.measurement) -> (!classical.link, !classical.future)
  %l2, %g = classical.submit %l1, @decode(%m) : (!pulse.measurement) -> (!classical.link, !classical.future)
  %a1 = classical.wait %f, @decode : i1
  %a2 = classical.wait %g, @decode : i1
  return
}

// A blocking wait is a barrier: the drive after it starts when the answer
// is in (40 + 1 + 100 = 141). Under ALAP the independent drive before it,
// which would otherwise slide to the makespan, still ends by then.
// CHECK-LABEL: func.func @barrier
// CHECK: pulse.drive {{.*}}start_vtu = 0 : i64
// ASAP: pulse.drive {{.*}}start_vtu = 0 : i64
// ALAP: pulse.drive {{.*}}start_vtu = 101 : i64
// CHECK: classical.wait {{.*}}classical.blocking{{.*}}start_vtu = 141 : i64}
// CHECK: pulse.drive {{.*}}start_vtu = 141 : i64
func.func @barrier() {
  %q = pulse.qudit_alloc : !pulse.qref
  %p = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %e0, %u0 = pulse.get_drive_line %p : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %a = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !pulse.waveform
  %e1, %u1 = pulse.drive %e0, %x, %u0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %o = pulse.qudit_alloc : !pulse.qref
  %g0, %v0 = pulse.get_drive_line %o : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %g1, %v1 = pulse.drive %g0, %x, %v0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %l0 = classical.get_link {device = 1 : i64} : !classical.link
  %t = pulse.get_current_time %e1 : !pulse.drive_line -> !pulse.timepoint
  %c = pulse.timepoint_to_int %t : !pulse.timepoint -> i64
  %cf = arith.sitofp %c : i64 to f64
  %l1, %f = classical.submit %l0, @track(%cf) : (f64) -> (!classical.link, !classical.future)
  %v = classical.wait %f, @track : f64
  %d1, %t1 = pulse.drive %d0, %x, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

}
