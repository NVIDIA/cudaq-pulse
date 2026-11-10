// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt --split-input-file --cse %s | FileCheck %s
// RUN: cudaq-pulse-opt --split-input-file --cse %s | cudaq-pulse-opt --split-input-file --pulse-verify > /dev/null

// Upstream CSE on pulse IR: what it may merge and what it must not.

// Two tones of the same frequency and phase are two frames: each drive
// plays its own.
// CHECK-LABEL: func.func @two_equal_tones
// CHECK-COUNT-2: pulse.tone %
// CHECK-NOT: pulse.tone %
func.func @two_equal_tones(%d0: !pulse.drive_line, %d1: !pulse.drive_line, %w: !pulse.waveform) -> (!pulse.drive_line, !pulse.drive_line) {
  %f = arith.constant 5.0e9 : f64
  %p = arith.constant 0.0 : f64
  %t0 = pulse.tone %f, %p : f64, f64 -> !pulse.tone
  %t1 = pulse.tone %f, %p : f64, f64 -> !pulse.tone
  %a, %ta = pulse.drive %d0, %w, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %b, %tb = pulse.drive %d1, %w, %t1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return %a, %b : !pulse.drive_line, !pulse.drive_line
}

// -----

// Two equal waveforms are one value.
// CHECK-LABEL: func.func @two_equal_waveforms
// CHECK-COUNT-1: pulse.square %
// CHECK-NOT: pulse.square %
func.func @two_equal_waveforms(%d0: !pulse.drive_line, %t0: !pulse.tone) -> (!pulse.drive_line, !pulse.tone) {
  %c40 = arith.constant 40 : i64
  %a = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %w0 = pulse.square %c40, %a, %z : i64, f64, f64 -> !pulse.waveform
  %w1 = pulse.square %c40, %a, %z : i64, f64, f64 -> !pulse.waveform
  %d1, %t1 = pulse.drive %d0, %w0, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %d2, %t2 = pulse.drive %d1, %w1, %t1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return %d2, %t2 : !pulse.drive_line, !pulse.tone
}

// -----

// Two time reads of one version of a line are one value.
// CHECK-LABEL: func.func @two_time_reads
// CHECK-COUNT-1: pulse.get_current_time %
// CHECK-NOT: pulse.get_current_time %
func.func @two_time_reads(%d0: !pulse.drive_line) -> (i64, i64) {
  %t0 = pulse.get_current_time %d0 : !pulse.drive_line -> !pulse.timepoint
  %n0 = pulse.timepoint_to_int %t0 : !pulse.timepoint -> i64
  %t1 = pulse.get_current_time %d0 : !pulse.drive_line -> !pulse.timepoint
  %n1 = pulse.timepoint_to_int %t1 : !pulse.timepoint -> i64
  return %n0, %n1 : i64, i64
}
