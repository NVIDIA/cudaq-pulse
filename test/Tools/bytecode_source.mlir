// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// --emit-bytecode states the source of the Pulse and Classical dialects in
// the bytecode's producer string (the LLVM version and a digest of their
// definition files), from assembly and from bytecode, and the module reads
// back as written.
// RUN: cudaq-pulse-opt --emit-bytecode %s | cudaq-pulse-opt | FileCheck %s
// RUN: cudaq-pulse-opt --emit-bytecode %s | FileCheck %s --check-prefix=SOURCE
// RUN: cudaq-pulse-opt --emit-bytecode %s | cudaq-pulse-opt --emit-bytecode | FileCheck %s --check-prefix=SOURCE

// The chunks of a split input are not one module, and are written as the
// driver writes them.
// RUN: cudaq-pulse-opt --emit-bytecode --split-input-file %s | FileCheck %s --check-prefix=SPLIT
// SPLIT: MLIR{{[0-9]}}

// CHECK-LABEL: func.func @waits
// CHECK: pulse.wait %{{.*}}, %{{.*}} <{duration_cycles = 40 : i64, start_cycles = 24 : i64}>
// SOURCE: cudaq-pulse-opt {{[^ ]+}} cudaq-pulse-dialects={{[0-9]+\.[0-9]+\.[0-9]+[^ -]*}}-{{[0-9a-f]+}}
func.func @waits(%q: !pulse.qudit) {
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %c = arith.constant 40 : i64
  %idle = pulse.duration_from_int %c : (i64) -> !pulse.duration
  %d1 = pulse.wait %d, %idle <{duration_cycles = 40 : i64, start_cycles = 24 : i64}> : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  return
}
