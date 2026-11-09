// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s | cudaq-pulse-opt | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: cudaq-pulse-opt --canonicalize %s | FileCheck %s --check-prefix=CANON

// ROUNDTRIP-LABEL: func @active_reset
// ROUNDTRIP: %{{.*}}, %{{.*}}, %[[M:[^ ]+]] = pulse.readout
// ROUNDTRIP: %[[B:.*]] = pulse.is_set %[[M]] : !pulse.measurement -> i1
// ROUNDTRIP: scf.if %[[B]] -> (!pulse.drive_line, !pulse.tone)
// ROUNDTRIP: pulse.drive
// ROUNDTRIP: scf.yield
// ROUNDTRIP: } else {
// ROUNDTRIP: scf.yield
func.func @active_reset() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %r0, %rt0 = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %c300 = arith.constant 300 : i64
  %c40 = arith.constant 40 : i64
  %amp = arith.constant 0.05 : f64
  %x_amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %ro = pulse.square %c300, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %x = pulse.square %c40, %x_amp, %zero : i64, f64, f64 -> !pulse.waveform
  %r1, %rt1, %m = pulse.readout %r0, %ro, %rt0, iq : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %bit = pulse.is_set %m : !pulse.measurement -> i1
  %d1, %t1 = scf.if %bit -> (!pulse.drive_line, !pulse.tone) {
    %d, %t = pulse.drive %d0, %x, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %d, %t : !pulse.drive_line, !pulse.tone
  } else {
    scf.yield %d0, %t0 : !pulse.drive_line, !pulse.tone
  }
  return
}

// ROUNDTRIP-LABEL: func @qudit_level
// ROUNDTRIP: pulse.outcome %{{.*}} : !pulse.measurement -> i64
// CANON-LABEL: func @qudit_level
// CANON-NOT: arith.cmpi ne
// CANON: pulse.is_set %{{.*}} : !pulse.measurement -> i1
// CANON: arith.cmpi eq
func.func @qudit_level(%r0: !pulse.readout_line, %rt0: !pulse.tone, %wf: !pulse.waveform) -> (i1, i1) {
  %r1, %rt1, %m = pulse.readout %r0, %wf, %rt0, iq : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %k = pulse.outcome %m : !pulse.measurement -> i64
  %c0 = arith.constant 0 : i64
  %c2 = arith.constant 2 : i64
  %nonzero = arith.cmpi ne, %k, %c0 : i64
  %is2 = arith.cmpi eq, %k, %c2 : i64
  return %nonzero, %is2 : i1, i1
}
