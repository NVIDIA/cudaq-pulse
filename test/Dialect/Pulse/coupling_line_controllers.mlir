// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt --pulse-schedule-asap --pulse-verify %s | FileCheck %s
// RUN: cudaq-pulse-opt --pulse-schedule-alap --pulse-verify %s > /dev/null

// Qudit 0's coupling line is on controller 0, whose grid is one cycle;
// qudit 1's on controller 1, whose grid is 4: its second 6-cycle pulse
// starts at 8, the first time on its grid after the first.
module attributes {pulse.coupling_lines, pulse.control_system = #pulse.control_system<clock_period_fs = 1000000, lines = [#pulse.line<drive, qudits = [0], controller = 0>, #pulse.line<drive, qudits = [1], controller = 1>], controllers = [#pulse.controller<>, #pulse.controller<grid_cycles = 4>]>} {
// CHECK-LABEL: func.func @on_their_controllers
// CHECK: pulse.drive_baseband {{.*}}{duration_cycles = 6 : i64, start_cycles = 0 : i64}
// CHECK: pulse.drive_baseband {{.*}}{duration_cycles = 6 : i64, start_cycles = 6 : i64}
// CHECK: pulse.drive_baseband {{.*}}{duration_cycles = 6 : i64, start_cycles = 0 : i64}
// CHECK: pulse.drive_baseband {{.*}}{duration_cycles = 6 : i64, start_cycles = 8 : i64}
func.func @on_their_controllers() {
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %a = pulse.get_coupling_line %q0 {controller = 0 : i64} : (!pulse.qudit) -> !pulse.coupling_line
  %b = pulse.get_coupling_line %q1 {controller = 1 : i64} : (!pulse.qudit) -> !pulse.coupling_line
  %c6 = arith.constant 6 : i64
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %w6 = pulse.square %c6, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %a1 = pulse.drive_baseband %a, %w6 : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
  %a2 = pulse.drive_baseband %a1, %w6 : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
  %b1 = pulse.drive_baseband %b, %w6 : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
  %b2 = pulse.drive_baseband %b1, %w6 : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
  return
}
}
