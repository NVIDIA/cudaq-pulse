// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The pulse-schedule-rcp options schedule_rcp_options.mlir leaves out:
// max-readouts, readout-latency and alap (the extension's alap_rcp).

// RUN: cudaq-pulse-opt --pulse-schedule-rcp %s | FileCheck %s --check-prefix=TWO
// RUN: cudaq-pulse-opt '--pulse-schedule-rcp=max-readouts=1' %s | FileCheck %s --check-prefix=ONE
// RUN: cudaq-pulse-opt '--pulse-schedule-rcp=readout-latency=100' %s | FileCheck %s --check-prefix=LATENCY
// RUN: cudaq-pulse-opt '--pulse-schedule-rcp=alap' %s | FileCheck %s --check-prefix=ALAP

!r = !pulse.readout_line
!t = !pulse.tone
!w = !pulse.waveform
!m = !pulse.measurement

// Two readouts on q0's line and one on q1's. With two readout units the
// lines play together; with one they take turns; a readout latency holds
// q0's line after its first readout; and as late as possible, q1's readout
// ends with q0's second.
// TWO-LABEL: func.func @readouts
// TWO: start_cycles = 0
// TWO: start_cycles = 0
// TWO: start_cycles = 400
// ONE-LABEL: func.func @readouts
// ONE: start_cycles = 0
// ONE: start_cycles = 400
// ONE: start_cycles = 800
// LATENCY-LABEL: func.func @readouts
// LATENCY: start_cycles = 0
// LATENCY: start_cycles = 0
// LATENCY: start_cycles = 500
// ALAP-LABEL: func.func @readouts
// ALAP: start_cycles = 0
// ALAP: start_cycles = 400
// ALAP: start_cycles = 400
func.func @readouts() {
  %c400 = arith.constant 400 : i64
  %a = arith.constant 5.000000e-02 : f64
  %z = arith.constant 0.000000e+00 : f64
  %ro = pulse.square %c400, %a, %z : i64, f64, f64 -> !w
  %q0 = pulse.qudit_alloc : !pulse.qref
  %q1 = pulse.qudit_alloc : !pulse.qref
  %r0, %t0 = pulse.get_readout_line %q0 : (!pulse.qref) -> (!r, !t)
  %r1, %t1 = pulse.get_readout_line %q1 : (!pulse.qref) -> (!r, !t)
  %r0a, %t0a, %m0 = pulse.readout %r0, %ro, %t0, "discriminate" : !r, !w, !t -> !r, !t, !m
  %r1a, %t1a, %m1 = pulse.readout %r1, %ro, %t1, "discriminate" : !r, !w, !t -> !r, !t, !m
  %r0b, %t0b, %m2 = pulse.readout %r0a, %ro, %t0a, "discriminate" : !r, !w, !t -> !r, !t, !m
  return
}

// A short and a long drive before a sync: as soon as possible both start at
// 0; as late as possible the short one ends at the sync.
// TWO-LABEL: func.func @before_sync
// TWO: duration_cycles = 40 : i64, start_cycles = 0
// TWO: duration_cycles = 80 : i64, start_cycles = 0
// TWO: duration_cycles = 40 : i64, start_cycles = 80
// ALAP-LABEL: func.func @before_sync
// ALAP: duration_cycles = 40 : i64, start_cycles = 40
// ALAP: duration_cycles = 80 : i64, start_cycles = 0
// ALAP: duration_cycles = 40 : i64, start_cycles = 80
func.func @before_sync() {
  %c40 = arith.constant 40 : i64
  %c80 = arith.constant 80 : i64
  %a = arith.constant 1.000000e-01 : f64
  %z = arith.constant 0.000000e+00 : f64
  %short = pulse.square %c40, %a, %z : i64, f64, f64 -> !w
  %long = pulse.square %c80, %a, %z : i64, f64, f64 -> !w
  %q0 = pulse.qudit_alloc : !pulse.qref
  %q1 = pulse.qudit_alloc : !pulse.qref
  %d0, %u0 = pulse.get_drive_line %q0 : (!pulse.qref) -> (!pulse.drive_line, !t)
  %d1, %u1 = pulse.get_drive_line %q1 : (!pulse.qref) -> (!pulse.drive_line, !t)
  %e0, %v0 = pulse.drive %d0, %short, %u0 : !pulse.drive_line, !w, !t -> !pulse.drive_line, !t
  %e1, %v1 = pulse.drive %d1, %long, %u1 : !pulse.drive_line, !w, !t -> !pulse.drive_line, !t
  %s0, %s1 = pulse.sync %e0, %e1 : !pulse.drive_line, !pulse.drive_line -> !pulse.drive_line, !pulse.drive_line
  %g0, %y0 = pulse.drive %s0, %short, %v0 : !pulse.drive_line, !w, !t -> !pulse.drive_line, !t
  return
}
