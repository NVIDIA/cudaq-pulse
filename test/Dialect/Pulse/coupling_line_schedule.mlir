// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt --pulse-schedule-asap --pulse-verify %s | FileCheck %s --check-prefixes=CHECK,ASAP
// RUN: cudaq-pulse-opt --pulse-schedule-alap --pulse-verify %s | FileCheck %s --check-prefixes=CHECK,ALAP
// RUN: cudaq-pulse-opt '--pulse-schedule-rcp=max-drives=1' --pulse-verify %s | FileCheck %s --check-prefixes=CHECK,RCP
// RUN: cudaq-pulse-opt '--pulse-schedule-rcp=alap max-drives=1' --pulse-verify %s | FileCheck %s --check-prefixes=CHECK,ALAPRCP

// A coupling line is a line of its own to every scheduler: its pulses play
// one after another on it, beside those of its qudit's drive line, a sync
// with other lines meets at the latest of them, and a pulse on it takes
// none of the drives a control system plays at once.

// CHECK-LABEL: func @beside_the_drive_line
func.func @beside_the_drive_line() {
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %c = pulse.get_coupling_line %q0 : (!pulse.qudit) -> !pulse.coupling_line
  %p = pulse.get_coupling_line %q0, %q1 : (!pulse.qudit, !pulse.qudit) -> !pulse.coupling_line
  %d0, %t0 = pulse.get_drive_line %q0 {frequency_hz = 5.0e9 : f64}
      : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %d1, %t1 = pulse.get_drive_line %q1 {frequency_hz = 5.1e9 : f64}
      : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %long = arith.constant 60 : i64
  %short = arith.constant 20 : i64
  %amplitude = arith.constant 0.25 : f64
  %zero = arith.constant 0.0 : f64
  %w60 = pulse.square %long, %amplitude, %zero : i64, f64, f64 -> !pulse.waveform
  %w20 = pulse.square %short, %amplitude, %zero : i64, f64, f64 -> !pulse.waveform
  // The two drive lines play one pulse each: with one drive at once, the
  // second waits for the first.
  // ASAP: pulse.drive %{{.*}}, %{{.*}}, %{{.*}} <{duration_cycles = 20 : i64, start_cycles = 0 : i64}>
  // ALAP: pulse.drive %{{.*}}, %{{.*}}, %{{.*}} <{duration_cycles = 20 : i64, start_cycles = 60 : i64}>
  // RCP: pulse.drive %{{.*}}, %{{.*}}, %{{.*}} <{duration_cycles = 20 : i64, start_cycles = 0 : i64}>
  // ALAPRCP: pulse.drive %{{.*}}, %{{.*}}, %{{.*}} <{duration_cycles = 20 : i64, start_cycles = 40 : i64}>
  %d0a, %t0a = pulse.drive %d0, %w20, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  // ASAP: pulse.drive %{{.*}}, %{{.*}}, %{{.*}} <{duration_cycles = 20 : i64, start_cycles = 0 : i64}>
  // ALAP: pulse.drive %{{.*}}, %{{.*}}, %{{.*}} <{duration_cycles = 20 : i64, start_cycles = 60 : i64}>
  // RCP: pulse.drive %{{.*}}, %{{.*}}, %{{.*}} <{duration_cycles = 20 : i64, start_cycles = 20 : i64}>
  // ALAPRCP: pulse.drive %{{.*}}, %{{.*}}, %{{.*}} <{duration_cycles = 20 : i64, start_cycles = 60 : i64}>
  %d1a, %t1a = pulse.drive %d1, %w20, %t1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  // The coupling line of qudit 0 plays while its drive line does, and takes
  // no drive of the control system.
  // ASAP: pulse.drive %{{.*}}, %{{.*}} <{duration_cycles = 60 : i64, start_cycles = 0 : i64}> : !pulse.coupling_line
  // ALAP: pulse.drive %{{.*}}, %{{.*}} <{duration_cycles = 60 : i64, start_cycles = 0 : i64}> : !pulse.coupling_line
  // RCP: pulse.drive %{{.*}}, %{{.*}} <{duration_cycles = 60 : i64, start_cycles = 0 : i64}> : !pulse.coupling_line
  // ALAPRCP: pulse.drive %{{.*}}, %{{.*}} <{duration_cycles = 60 : i64, start_cycles = 0 : i64}> : !pulse.coupling_line
  %c1 = pulse.drive %c, %w60 : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
  // ASAP: pulse.drive %{{.*}}, %{{.*}} <{duration_cycles = 20 : i64, start_cycles = 60 : i64}> : !pulse.coupling_line
  // ALAP: pulse.drive %{{.*}}, %{{.*}} <{duration_cycles = 20 : i64, start_cycles = 60 : i64}> : !pulse.coupling_line
  // RCP: pulse.drive %{{.*}}, %{{.*}} <{duration_cycles = 20 : i64, start_cycles = 60 : i64}> : !pulse.coupling_line
  // ALAPRCP: pulse.drive %{{.*}}, %{{.*}} <{duration_cycles = 20 : i64, start_cycles = 60 : i64}> : !pulse.coupling_line
  %c2 = pulse.drive %c1, %w20 : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
  // Every line meets at 80, and the pair's line plays from there.
  %c3, %p1, %d0b, %d1b = pulse.sync %c2, %p, %d0a, %d1a
      : !pulse.coupling_line, !pulse.coupling_line, !pulse.drive_line, !pulse.drive_line
      -> !pulse.coupling_line, !pulse.coupling_line, !pulse.drive_line, !pulse.drive_line
  // CHECK: pulse.drive %{{.*}}, %{{.*}} <{duration_cycles = 20 : i64, start_cycles = 80 : i64}> : !pulse.coupling_line
  %p2 = pulse.drive %p1, %w20 : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
  // CHECK: pulse.drive %{{.*}}, %{{.*}}, %{{.*}} <{duration_cycles = 20 : i64, start_cycles = 80 : i64}>
  %d0c, %t0c = pulse.drive %d0b, %w20, %t0a : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}
