// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt --pulse-schedule-asap --pulse-verify %s | FileCheck %s

// A delayed coupling line in a module with no control system.
module attributes {pulse.coupling_lines} {
// CHECK-LABEL: func.func @delayed_alone
// CHECK: pulse.drive_baseband {{.*}}{duration_cycles = 4 : i64, start_cycles = 7 : i64}
func.func @delayed_alone() {
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %c = pulse.get_coupling_line %q0 {delay_cycles = 7 : i64} : (!pulse.qudit) -> !pulse.coupling_line
  %c4 = arith.constant 4 : i64
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %w4 = pulse.square %c4, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %c1 = pulse.drive_baseband %c, %w4 : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
  return
}
}
