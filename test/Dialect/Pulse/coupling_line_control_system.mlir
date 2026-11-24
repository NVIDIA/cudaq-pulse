// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// What a control system says of a coupling line is on the line's
// acquisition: the controller that drives it among several (`controller`)
// and its calibrated delay in program cycles (`delay_cycles`). The
// schedulers read them as they read a drive line's from the control system.
// RUN: cudaq-pulse-opt --pulse-schedule-asap --pulse-verify %s | FileCheck %s
// RUN: cudaq-pulse-opt --pulse-schedule-alap --pulse-verify %s > /dev/null

// Qudit 0's drive line starts everything 3 cycles later than scheduled and
// its coupling line 5; the pair's coupling line keeps its times.
module attributes {pulse.coupling_lines, pulse.control_system = #pulse.control_system<clock_period_fs = 1000000, lines = [#pulse.line<drive, qudits = [0], delay_cycles = 3>]>} {
// CHECK-LABEL: func.func @delayed
// CHECK-SAME: #pulse.schedule_state<placed>
// CHECK: pulse.drive {{.*}}{duration_cycles = 8 : i64, start_cycles = 3 : i64}
// CHECK: pulse.drive_baseband {{.*}}{duration_cycles = 4 : i64, start_cycles = 5 : i64}
// CHECK: pulse.drive_baseband {{.*}}{duration_cycles = 4 : i64, start_cycles = 9 : i64}
// CHECK: pulse.drive_baseband {{.*}}{duration_cycles = 4 : i64, start_cycles = 0 : i64}
func.func @delayed() {
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q0 : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %c = pulse.get_coupling_line %q0 {delay_cycles = 5 : i64} : (!pulse.qudit) -> !pulse.coupling_line
  %p = pulse.get_coupling_line %q0, %q1 : (!pulse.qudit, !pulse.qudit) -> !pulse.coupling_line
  %c8 = arith.constant 8 : i64
  %c4 = arith.constant 4 : i64
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %w8 = pulse.square %c8, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %w4 = pulse.square %c4, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %d1, %t1 = pulse.drive %d0, %w8, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %c1 = pulse.drive_baseband %c, %w4 : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
  %c2 = pulse.drive_baseband %c1, %w4 : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
  %p1 = pulse.drive_baseband %p, %w4 : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
  return
}
}

// A delayed coupling line where the control system says nothing else is
// in coupling_line_delay_alone.mlir; controllers with their own grids in
// coupling_line_controllers.mlir.
