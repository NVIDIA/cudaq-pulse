// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// One round of adaptive T1 tracking on the controller: read out, send the
// outcome and the posterior, reset from the outcome, excite, and wait the
// answer's length before the next readout. What the estimate means depends on
// the excitation ending where the variable wait starts: then the qubit decays
// for exactly the answer's length. Under ALAP the excitation slides up to the
// wait; under ASAP it plays right after the reset and the qubit also decays
// through the call's latency.
// RUN: cudaq-pulse-opt --pulse-schedule-alap %s | FileCheck %s --check-prefixes=CHECK,ALAP
// RUN: cudaq-pulse-opt --pulse-schedule-asap %s | FileCheck %s --check-prefixes=CHECK,ASAP
// RUN: cudaq-pulse-opt --pulse-schedule-alap --pulse-classical-check --pulse-verify %s > /dev/null

module attributes {pulse.clock_ghz = 2.0 : f64,
    classical.links = {"0" = {bytes_per_ns = 0.0 : f64, feedback_ns = 8.0 : f64,
                              fixed_ns = 4.0 : f64, max_in_flight = 64 : i64,
                              slot_bytes = 1024 : i64}}} {

classical.callable @update {args = ["bits:1", "f64", "f64", "f64"], device = 0 : i64,
    function_id = 672109684 : i64, latency_max_ns = 500.0 : f64,
    latency_min_ns = 0.0 : f64, latency_nominal_ns = 500.0 : f64,
    mode = "bounded", results = ["f64", "f64", "f64"], submit = "eager"}

// The readout ends at 200 and its outcome goes out (8 cycles on the link).
// The answer is due 500 ns + 2 x 8 ns of feedback = 1016 cycles after the
// submission, at 1224, where the variable wait starts.
// CHECK-LABEL: func.func @round
// CHECK: pulse.readout {{.*}}start_vtu = 0 : i64}
// CHECK: classical.submit {{.*}}duration_vtu = 8 : i64, start_vtu = 200 : i64}
// CHECK: classical.wait {{.*}}start_vtu = 1224 : i64}
// ALAP: pulse.drive {{.*}}start_vtu = 1144 : i64}
// ALAP: pulse.drive {{.*}}duration_vtu = 40 : i64, start_vtu = 1184 : i64}
// ASAP: pulse.drive {{.*}}start_vtu = 200 : i64}
// ASAP: pulse.drive {{.*}}duration_vtu = 40 : i64, start_vtu = 240 : i64}
// CHECK: pulse.wait {{.*}}classical.variable{{.*}}duration_vtu = 32 : i64, {{.*}}start_vtu = 1224 : i64}
// CHECK: pulse.readout {{.*}}start_vtu = 1256 : i64}
func.func @round() {
  %c40 = arith.constant 40 : i64
  %c200 = arith.constant 200 : i64
  %amp = arith.constant 0.1 : f64
  %ro = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %shape = arith.constant 3.0 : f64
  %rate = arith.constant 900.0 : f64
  %w0 = arith.constant 300.0 : f64
  %half = arith.constant 0.5 : f64
  %q = pulse.qudit_alloc {qubit = 0 : i64} : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q {frequency_hz = 5.0e9 : f64, qubit = 0 : i64} : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %r0, %u0 = pulse.get_readout_line %q {frequency_hz = 5.0e9 : f64, qubit = 0 : i64} : (!pulse.qref) -> (!pulse.readout_line, !pulse.tone)
  %pi = pulse.square %c40, %amp, %z : i64, f64, f64 -> !pulse.waveform
  %pr = pulse.square %c200, %ro, %z : i64, f64, f64 -> !pulse.waveform
  %r1, %u1, %m = pulse.readout %r0, %pr, %u0, "iq" : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %l0 = classical.get_link {device = 0 : i64} : !classical.link
  %l1, %h = classical.submit %l0, @update(%m, %shape, %rate, %w0) {classical.eager} : (!pulse.measurement, f64, f64, f64) -> (!classical.link, !classical.future)
  %a:3 = classical.wait %h, @update : f64, f64, f64
  %s:2 = pulse.sync %d0, %r1 : !pulse.drive_line, !pulse.readout_line -> !pulse.drive_line, !pulse.readout_line
  %set = pulse.is_set %m : !pulse.measurement -> i1
  %b:2 = scf.if %set -> (!pulse.drive_line, !pulse.tone) {
    %d, %t = pulse.drive %s#0, %pi, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %d, %t : !pulse.drive_line, !pulse.tone
  } else {
    scf.yield %s#0, %t0 : !pulse.drive_line, !pulse.tone
  }
  %d1, %t1 = pulse.drive %b#0, %pi, %b#1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %rounded = arith.addf %a#2, %half : f64
  %n = arith.fptosi %rounded : f64 to i64
  %len = pulse.duration_from_int %n : (i64) -> !pulse.duration
  %d2 = pulse.wait %d1, %len {classical.variable, pulse.variable, pulse.min_vtu = 32 : i64, pulse.max_vtu = 4000 : i64} : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  %s2:2 = pulse.sync %d2, %s#1 : !pulse.drive_line, !pulse.readout_line -> !pulse.drive_line, !pulse.readout_line
  %r2, %u2, %m2 = pulse.readout %s2#1, %pr, %u1, "iq" : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  return
}

}
