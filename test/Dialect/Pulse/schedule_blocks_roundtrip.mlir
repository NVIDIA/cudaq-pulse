// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s | cudaq-pulse-opt | FileCheck %s

// CHECK-LABEL: func.func @roundtrip
// CHECK: pulse.schedule_block alap (%{{.*}}, %{{.*}}, %{{.*}}) {callee = "measure"} : (!pulse.measurement, !pulse.drive_line, !pulse.tone) -> (!pulse.drive_line, !pulse.tone)
// CHECK: ^bb0(%{{.*}}: !pulse.measurement, %{{.*}}: !pulse.drive_line, %{{.*}}: !pulse.tone):
// CHECK: pulse.yield %{{.*}}, %{{.*}} : !pulse.drive_line, !pulse.tone
func.func @roundtrip(%m0: !pulse.measurement, %d0: !pulse.drive_line, %t0: !pulse.tone, %x: !pulse.waveform) {
  %d1, %t1 = pulse.schedule_block alap (%m0, %d0, %t0) {callee = "measure"} : (!pulse.measurement, !pulse.drive_line, !pulse.tone) -> (!pulse.drive_line, !pulse.tone) {
  ^bb0(%m: !pulse.measurement, %d: !pulse.drive_line, %t: !pulse.tone):
    %bit = pulse.is_set %m : !pulse.measurement -> i1
    %d2, %t2 = scf.if %bit -> (!pulse.drive_line, !pulse.tone) {
      %d3, %t3 = pulse.drive %d, %x, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
      scf.yield %d3, %t3 : !pulse.drive_line, !pulse.tone
    } else {
      scf.yield %d, %t : !pulse.drive_line, !pulse.tone
    }
    pulse.yield %d2, %t2 : !pulse.drive_line, !pulse.tone
  }
  return
}
