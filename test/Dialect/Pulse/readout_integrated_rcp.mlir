// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt '--pulse-schedule-rcp=alap readout-latency=20' %s | FileCheck %s

// Under rcp a readout holds its line for the readout latency after it ends,
// and an outcome waits for the readout to end on its delayed line (qudit 0's
// readout line is delayed 5 cycles). ALAP places a readout whose level
// nothing reads to end its latency, 20 cycles, before the drive does, at 980,
// and the delay moves it to start at 685. A readout whose IQ point nothing
// reads goes to the same place: an unread point does not pull its readout
// earlier by the latency and the delay.

module attributes {pulse.qudit_count = 1 : i64,
    pulse.control_system = #pulse.control_system<clock_period_fs = 1000000,
      lines = [#pulse.line<drive, qudits = [0]>,
               #pulse.line<readout, qudits = [0], delay_cycles = 5>]>} {

// CHECK-LABEL: func.func @unread_level
// CHECK: pulse.readout {{.*}}, level <{duration_cycles = 300 : i64, start_cycles = 685 : i64}>
func.func @unread_level() {
  %c300 = arith.constant 300 : i64
  %c1000 = arith.constant 1000 : i64
  %a = arith.constant 5.000000e-02 : f64
  %z = arith.constant 0.000000e+00 : f64
  %ro = pulse.square %c300, %a, %z : i64, f64, f64 -> !pulse.waveform
  %x = pulse.square %c1000, %a, %z : i64, f64, f64 -> !pulse.waveform
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %dt0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %r0, %rt0 = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %d1, %dt1 = pulse.drive %d0, %x, %dt0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %r1, %rt1, %m = pulse.readout %r0, %ro, %rt0, level : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  return
}

// CHECK-LABEL: func.func @unread_point
// CHECK: pulse.readout {{.*}}, integrated <{duration_cycles = 300 : i64, start_cycles = 685 : i64}>
func.func @unread_point() {
  %c300 = arith.constant 300 : i64
  %c1000 = arith.constant 1000 : i64
  %a = arith.constant 5.000000e-02 : f64
  %z = arith.constant 0.000000e+00 : f64
  %ro = pulse.square %c300, %a, %z : i64, f64, f64 -> !pulse.waveform
  %x = pulse.square %c1000, %a, %z : i64, f64, f64 -> !pulse.waveform
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %dt0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %r0, %rt0 = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %d1, %dt1 = pulse.drive %d0, %x, %dt0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %r1, %rt1, %p = pulse.readout %r0, %ro, %rt0, integrated : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.iq_data
  return
}

}
