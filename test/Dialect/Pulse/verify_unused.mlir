// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt --pulse-verify %s -o /dev/null 2>&1 | FileCheck %s

// An unused line is a warning that names its qubit, without a dump of the op.
// CHECK: warning: drive line allocated but never used (qubit 3)
// CHECK-NOT: see current operation
// CHECK: warning: readout line allocated but never used (qubit 3)
// CHECK-NOT: see current operation

func.func @unused_lines() {
  %q = pulse.qudit_alloc {qubit = 3 : i64} : !pulse.qref
  %d, %t = pulse.get_drive_line %q {qubit = 3 : i64, frequency_hz = 5.0e9 : f64}
      : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %r, %rt = pulse.get_readout_line %q {qubit = 3 : i64, frequency_hz = 7.0e9 : f64}
      : (!pulse.qref) -> (!pulse.readout_line, !pulse.tone)
  return
}
