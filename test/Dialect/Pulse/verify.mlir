// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: not cudaq-pulse-opt --pulse-verify %s 2>&1 | FileCheck %s

func.func @invalid_linearity_and_timing() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q
      {qubit = 0 : i64, frequency_hz = 5.0e9 : f64}
      : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %duration = arith.constant 40 : i64
  %real = arith.constant 0.3 : f64
  %imag = arith.constant 0.0 : f64
  %wf = pulse.square %duration, %real, %imag
      : i64, f64, f64 -> !pulse.waveform
  %d1, %t1 = pulse.drive %d0, %wf, %t0
      <{duration_cycles = 40 : i64, start_cycles = 0 : i64}>
      : !pulse.drive_line, !pulse.waveform, !pulse.tone
      -> !pulse.drive_line, !pulse.tone
  %d2, %t2 = pulse.drive %d0, %wf, %t0
      <{duration_cycles = 40 : i64, start_cycles = 20 : i64}>
      : !pulse.drive_line, !pulse.waveform, !pulse.tone
      -> !pulse.drive_line, !pulse.tone

  %q_wait = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %wait_line, %wait_tone = pulse.get_drive_line %q_wait
      : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %wait_cycles = arith.constant 40 : i64
  %wait_duration = pulse.duration_from_int %wait_cycles
      : (i64) -> !pulse.duration
  %wait_line_1 = pulse.wait %wait_line, %wait_duration
      <{duration_cycles = 40 : i64, start_cycles = 0 : i64}>
      : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  %wait_line_2 = pulse.wait %wait_line_1, %wait_duration
      <{duration_cycles = 40 : i64, start_cycles = 20 : i64}>
      : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  return
}

// CHECK: error: linear pulse value is consumed more than once on one path
// CHECK-COUNT-2: error: operation overlaps or precedes its physical-line predecessor
