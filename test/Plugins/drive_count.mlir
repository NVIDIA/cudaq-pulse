// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// The pass plugin in examples/custom_passes/cpp loads into cudaq-pulse-opt.
// RUN: cudaq-pulse-opt --load-pass-plugin=%pulse_example_plugin \
// RUN:   --pass-pipeline='builtin.module(func.func(pulse-schedule-asap), pulse-example-drive-count)' %s \
// RUN:   | FileCheck %s

// CHECK-LABEL: func.func @two_drives
// CHECK-SAME: pulse_example.drive_cycles = 100 : i64
// CHECK-SAME: pulse_example.drives = 2 : i64
func.func @two_drives() {
  %q = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %c60 = arith.constant 60 : i64
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %x = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %y = pulse.square %c60, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %d1, %t1 = pulse.drive %d0, %x, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %d2, %t2 = pulse.drive %d1, %y, %t1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}
