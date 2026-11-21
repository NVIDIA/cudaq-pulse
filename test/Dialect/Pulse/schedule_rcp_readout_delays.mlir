// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Under rcp an outcome is usable a latency after its readout ends on its
// line, so a delayed readout line holds the pulses that branch on it its
// delay longer. Qudit 0's readout line on controllers[0] is delayed 5
// cycles, qudit 2's is not, and qudit 1's drive line is on controllers[1]:
// an outcome is usable 10 cycles after its readout on controllers[0] and 20
// on controllers[1]. The readouts end at 300 as scheduled, 305 once qudit
// 0's is delayed.
// RUN: cudaq-pulse-opt --pulse-schedule-rcp %s | FileCheck %s
// RUN: cudaq-pulse-opt '--pulse-schedule-rcp=alap' %s | FileCheck %s

!d = !pulse.drive_line
!r = !pulse.readout_line
!t = !pulse.tone
!w = !pulse.waveform
!m = !pulse.measurement

module attributes {pulse.qudit_count = 3 : i64,
    pulse.control_system = #pulse.control_system<clock_period_fs = 1000000,
      lines = [#pulse.line<drive, qudits = [0]>, #pulse.line<drive, qudits = [1], controller = 1>,
               #pulse.line<readout, qudits = [0], delay_cycles = 5>,
               #pulse.line<readout, qudits = [2]>],
      controllers = [
        #pulse.controller<feedback_latency_cycles = 10, peer_latency_cycles = 10>,
        #pulse.controller<feedback_latency_cycles = 10>]>} {

// The drive on controllers[1] starts 20 after the delayed readout ends.
// CHECK-LABEL: func.func @across
// CHECK: pulse.readout {{.*}}start_cycles = 5 : i64
// CHECK: pulse.drive {{.*}}start_cycles = 325 : i64
func.func @across() {
  %c300 = arith.constant 300 : i64
  %c40 = arith.constant 40 : i64
  %a = arith.constant 5.000000e-02 : f64
  %z = arith.constant 0.000000e+00 : f64
  %ro = pulse.square %c300, %a, %z : i64, f64, f64 -> !w
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !w
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %r0, %rt0 = pulse.get_readout_line %q0 : (!pulse.qudit) -> (!r, !t)
  %d1, %dt1 = pulse.get_drive_line %q1 : (!pulse.qudit) -> (!d, !t)
  %r1, %rt1, %m = pulse.readout %r0, %ro, %rt0, discriminated : !r, !w, !t -> !r, !t, !m
  %bit = pulse.is_set %m : !m -> i1
  %d2, %dt2 = scf.if %bit -> (!d, !t) {
    %d, %t = pulse.drive %d1, %x, %dt1 : !d, !w, !t -> !d, !t
    scf.yield %d, %t : !d, !t
  } else {
    scf.yield %d1, %dt1 : !d, !t
  }
  return
}

// The drive on the reader's own controller starts 10 after it.
// CHECK-LABEL: func.func @own
// CHECK: pulse.readout {{.*}}start_cycles = 5 : i64
// CHECK: pulse.drive {{.*}}start_cycles = 315 : i64
func.func @own() {
  %c300 = arith.constant 300 : i64
  %c40 = arith.constant 40 : i64
  %a = arith.constant 5.000000e-02 : f64
  %z = arith.constant 0.000000e+00 : f64
  %ro = pulse.square %c300, %a, %z : i64, f64, f64 -> !w
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !w
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %r0, %rt0 = pulse.get_readout_line %q0 : (!pulse.qudit) -> (!r, !t)
  %d0, %dt0 = pulse.get_drive_line %q0 : (!pulse.qudit) -> (!d, !t)
  %r1, %rt1, %m = pulse.readout %r0, %ro, %rt0, discriminated : !r, !w, !t -> !r, !t, !m
  %bit = pulse.is_set %m : !m -> i1
  %d2, %dt2 = scf.if %bit -> (!d, !t) {
    %d, %t = pulse.drive %d0, %x, %dt0 : !d, !w, !t -> !d, !t
    scf.yield %d, %t : !d, !t
  } else {
    scf.yield %d0, %dt0 : !d, !t
  }
  return
}

// An undelayed readout holds the drive on controllers[1] its latency alone.
// CHECK-LABEL: func.func @undelayed
// CHECK: pulse.readout {{.*}}start_cycles = 0 : i64
// CHECK: pulse.drive {{.*}}start_cycles = 320 : i64
func.func @undelayed() {
  %c300 = arith.constant 300 : i64
  %c40 = arith.constant 40 : i64
  %a = arith.constant 5.000000e-02 : f64
  %z = arith.constant 0.000000e+00 : f64
  %ro = pulse.square %c300, %a, %z : i64, f64, f64 -> !w
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !w
  %q2 = pulse.qudit_alloc {qudit = 2 : i64} : !pulse.qudit
  %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %r0, %rt0 = pulse.get_readout_line %q2 : (!pulse.qudit) -> (!r, !t)
  %d1, %dt1 = pulse.get_drive_line %q1 : (!pulse.qudit) -> (!d, !t)
  %r1, %rt1, %m = pulse.readout %r0, %ro, %rt0, discriminated : !r, !w, !t -> !r, !t, !m
  %bit = pulse.is_set %m : !m -> i1
  %d2, %dt2 = scf.if %bit -> (!d, !t) {
    %d, %t = pulse.drive %d1, %x, %dt1 : !d, !w, !t -> !d, !t
    scf.yield %d, %t : !d, !t
  } else {
    scf.yield %d1, %dt1 : !d, !t
  }
  return
}

}
