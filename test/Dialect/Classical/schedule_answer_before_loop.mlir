// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// An answer taken before a loop and tested in every iteration is taken before
// the first iteration's branch on it, under every policy.
// RUN: cudaq-pulse-opt --pulse-schedule-alap %s | FileCheck %s --check-prefixes=CHECK,ALAP
// RUN: cudaq-pulse-opt --pulse-schedule-asap %s | FileCheck %s --check-prefixes=CHECK,PEELED
// RUN: cudaq-pulse-opt --pulse-schedule-rcp %s | FileCheck %s --check-prefixes=CHECK,PEELED
// RUN: cudaq-pulse-opt --pulse-schedule-alap --pulse-classical-check --pulse-verify %s > /dev/null

module attributes {pulse.clock_ghz = 1.0 : f64} {

classical.callable @decode {function_id = 3007678287 : i64, device = 0 : i64,
    args = ["bits:1"], results = ["bits:1"], latency_min_ns = 100.0,
    latency_nominal_ns = 200.0, latency_max_ns = 400.0, mode = "bounded",
    submit = "eager"}

// The readout ends at 100 and the answer is due at 501. Each iteration takes
// 270, and its branch on the answer starts at 501 in the first: the ALAP
// pass bounds the answer by the first iteration, not by the last ones it
// simulates. The forward passes peel the first iteration.
// CHECK-LABEL: func.func @tested_in_every_iteration
// CHECK: classical.wait {{.*}}start_cycles = 501 : i64}
// ALAP: scf.for
// ALAP: pulse.drive {{.*}}start_cycles = 461 : i64}
// ALAP: pulse.drive {{.*}}period_cycles = array<i64: 270>, start_cycles = 501 : i64}
// PEELED: pulse.drive {{.*}}start_cycles = 200 : i64}
// PEELED: pulse.drive {{.*}}start_cycles = 501 : i64}
// PEELED: scf.for
func.func @tested_in_every_iteration() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %r0, %rt0 = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %c100 = arith.constant 100 : i64
  %c40 = arith.constant 40 : i64
  %c30 = arith.constant 30 : i64
  %a = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %ro = pulse.square %c100, %a, %z : i64, f64, f64 -> !pulse.waveform
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !pulse.waveform
  %y = pulse.square %c30, %a, %z : i64, f64, f64 -> !pulse.waveform
  %c200 = arith.constant 200 : i64
  %gap = pulse.duration_from_int %c200 : (i64) -> !pulse.duration
  %l0 = classical.get_link {device = 0 : i64} : !classical.link
  %r1, %rt1, %m0 = pulse.readout %r0, %ro, %rt0, "iq" : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %l1, %f = classical.submit %l0, @decode(%m0) : (!pulse.measurement) -> (!classical.link, !classical.future)
  %n = classical.wait %f, @decode : i1
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 1 : index
  %dn, %tn, %rn, %rtn = scf.for %i = %lb to %ub step %step
      iter_args(%d = %d0, %t = %t0, %r = %r1, %rt = %rt1)
      -> (!pulse.drive_line, !pulse.tone, !pulse.readout_line, !pulse.tone) {
    %r2, %rt2, %m = pulse.readout %r, %ro, %rt, "iq" : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    %set = pulse.is_set %m : !pulse.measurement -> i1
    %d1, %t1 = scf.if %set -> (!pulse.drive_line, !pulse.tone) {
      %dd, %tt = pulse.drive %d, %x, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
      scf.yield %dd, %tt : !pulse.drive_line, !pulse.tone
    } else {
      scf.yield %d, %t : !pulse.drive_line, !pulse.tone
    }
    %d2, %t2 = scf.if %n -> (!pulse.drive_line, !pulse.tone) {
      %dd, %tt = pulse.drive %d1, %y, %t1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
      scf.yield %dd, %tt : !pulse.drive_line, !pulse.tone
    } else {
      scf.yield %d1, %t1 : !pulse.drive_line, !pulse.tone
    }
    %d3 = pulse.wait %d2, %gap : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
    scf.yield %d3, %t2, %r2, %rt2 : !pulse.drive_line, !pulse.tone, !pulse.readout_line, !pulse.tone
  }
  return
}

}
