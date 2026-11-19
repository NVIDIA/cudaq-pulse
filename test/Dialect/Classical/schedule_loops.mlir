// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Classical calls in loops: pipelining across rounds and the link's slots.
// RUN: cudaq-pulse-opt --pulse-schedule-asap %s | FileCheck %s --check-prefixes=CHECK,ASAP
// RUN: cudaq-pulse-opt --pulse-schedule-alap %s | FileCheck %s --check-prefixes=CHECK,ALAP
// RUN: cudaq-pulse-opt --pulse-schedule-rcp %s | FileCheck %s --check-prefixes=CHECK,RCP
// RUN: cudaq-pulse-opt --pulse-schedule-asap --pulse-classical-check --pulse-verify %s > /dev/null
// RUN: cudaq-pulse-opt --pulse-schedule-rcp --pulse-classical-check %s > /dev/null
// RUN: cudaq-pulse-opt --pulse-unroll-classical-loops %s | FileCheck %s --check-prefix=UNROLL

// Simulating a module unrolls every loop that calls.
// UNROLL: classical.callable @decode_wide
// UNROLL-NOT: scf.for

module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000>,
                   classical.links = {"0" = {slots = 2 : i64}}} {

classical.callable @decode_wide {dispatch_key = 1, device = 1 : i64,
    args = ["bits:1"], results = ["bits:1"], latency_min_ns = 100.0,
    latency_nominal_ns = 500.0, latency_max_ns = 1000.0, mode = "bounded",
    submit = "eager", dispatch_key_override}
classical.callable @decode {dispatch_key = 3007678287 : i64, device = 0 : i64,
    args = ["bits:1"], results = ["bits:1"], latency_min_ns = 100.0,
    latency_nominal_ns = 500.0, latency_max_ns = 1000.0, mode = "bounded",
    submit = "eager"}

// Each round reads out (300) and submits; answers are due 1001 after the
// submission. Link 0 has 2 slots, so under every policy the third request
// waits until the first answer frees a slot (300 + 1 + 1000 = 1301), and the
// loop is unrolled because its rounds differ.
// CHECK-LABEL: func.func @rounds
// CHECK-NOT: scf.for
// CHECK: classical.submit {{.*}}start_cycles = 300 : i64}
// CHECK: classical.submit {{.*}}start_cycles = 600 : i64}
// CHECK: classical.submit {{.*}}start_cycles = 1301 : i64}
// CHECK: classical.submit {{.*}}start_cycles = 1601 : i64}
func.func @rounds() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %r0, %rt0 = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %c300 = arith.constant 300 : i64
  %c40 = arith.constant 40 : i64
  %a = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %ro = pulse.square %c300, %a, %z : i64, f64, f64 -> !pulse.waveform
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !pulse.waveform
  %l0 = classical.get_link {device = 0 : i64} : !classical.link
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 1 : index
  %dn, %tn, %rn, %rtn, %ln = scf.for %i = %lb to %ub step %step
      iter_args(%d = %d0, %t = %t0, %r = %r0, %rt = %rt0, %l = %l0)
      -> (!pulse.drive_line, !pulse.tone, !pulse.readout_line, !pulse.tone, !classical.link) {
    %r1, %rt1, %m = pulse.readout %r, %ro, %rt, level : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    %l1, %f = classical.submit %l, @decode(%m) : (!pulse.measurement) -> (!classical.link, !classical.future)
    %fix = classical.wait %f, @decode : i1
    %d1, %t1 = scf.if %fix -> (!pulse.drive_line, !pulse.tone) {
      %dd, %tt = pulse.drive %d, %x, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
      scf.yield %dd, %tt : !pulse.drive_line, !pulse.tone
    } else {
      scf.yield %d, %t : !pulse.drive_line, !pulse.tone
    }
    scf.yield %d1, %t1, %r1, %rt1, %l1 : !pulse.drive_line, !pulse.tone, !pulse.readout_line, !pulse.tone, !classical.link
  }
  return
}


// The same rounds on link 1, which has the default 64 slots: calls of
// consecutive rounds overlap, and the loop keeps a period of 300.
// CHECK-LABEL: func.func @pipelined
// CHECK: scf.for
// CHECK: classical.submit {{.*}}period_cycles = array<i64: 300>, start_cycles = 300 : i64}
// ASAP: classical.wait {{.*}}period_cycles = array<i64: 300>, start_cycles = 1301 : i64}
// ASAP: pulse.drive {{.*}}period_cycles = array<i64: 300>, start_cycles = 1301 : i64}
// Under ALAP the waits move to where the drives need them, packed at the end.
// ALAP: classical.wait {{.*}}period_cycles = array<i64: 40>, start_cycles = 2081 : i64}
// RCP: classical.wait {{.*}}period_cycles = array<i64: 300>, start_cycles = 1301 : i64}
func.func @pipelined() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %r0, %rt0 = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %c300 = arith.constant 300 : i64
  %c40 = arith.constant 40 : i64
  %a = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %ro = pulse.square %c300, %a, %z : i64, f64, f64 -> !pulse.waveform
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !pulse.waveform
  %l0 = classical.get_link {device = 1 : i64} : !classical.link
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 1 : index
  %dn, %tn, %rn, %rtn, %ln = scf.for %i = %lb to %ub step %step
      iter_args(%d = %d0, %t = %t0, %r = %r0, %rt = %rt0, %l = %l0)
      -> (!pulse.drive_line, !pulse.tone, !pulse.readout_line, !pulse.tone, !classical.link) {
    %r1, %rt1, %m = pulse.readout %r, %ro, %rt, level : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    %l1, %f = classical.submit %l, @decode_wide(%m) : (!pulse.measurement) -> (!classical.link, !classical.future)
    %fix = classical.wait %f, @decode_wide : i1
    %d1, %t1 = scf.if %fix -> (!pulse.drive_line, !pulse.tone) {
      %dd, %tt = pulse.drive %d, %x, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
      scf.yield %dd, %tt : !pulse.drive_line, !pulse.tone
    } else {
      scf.yield %d, %t : !pulse.drive_line, !pulse.tone
    }
    scf.yield %d1, %t1, %r1, %rt1, %l1 : !pulse.drive_line, !pulse.tone, !pulse.readout_line, !pulse.tone, !classical.link
  }
  return
}

}
