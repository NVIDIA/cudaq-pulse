// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt --pulse-schedule-rcp %s | FileCheck %s --check-prefix=FOUR
// RUN: cudaq-pulse-opt '--pulse-schedule-rcp=max-drives=1' %s | FileCheck %s --check-prefix=ONE
// RUN: cudaq-pulse-opt '--pulse-schedule-rcp=max-drives=1 switch-penalty=5' %s | FileCheck %s --check-prefix=PENALTY

// Three drives on three lines: with four drive units they play together,
// with one they play in turn, and a switch penalty spaces them out.
// FOUR-COUNT-3: start_cycles = 0
// ONE: start_cycles = 0
// ONE: start_cycles = 40
// ONE: start_cycles = 80
// PENALTY: start_cycles = 0
// PENALTY: start_cycles = 45
// PENALTY: start_cycles = 90
func.func @main() {
  %c40 = arith.constant 40 : i64
  %a = arith.constant 1.000000e-01 : f64
  %z = arith.constant 0.000000e+00 : f64
  %w = pulse.square %c40, %a, %z : i64, f64, f64 -> !pulse.waveform
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qref
  %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qref
  %q2 = pulse.qudit_alloc {qudit = 2 : i64} : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q0 : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %d1, %t1 = pulse.get_drive_line %q1 : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %d2, %t2 = pulse.get_drive_line %q2 : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %e0, %u0 = pulse.drive %d0, %w, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %e1, %u1 = pulse.drive %d1, %w, %t1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %e2, %u2 = pulse.drive %d2, %w, %t2 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}
