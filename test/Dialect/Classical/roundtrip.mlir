// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// RUN: cudaq-pulse-opt %s | cudaq-pulse-opt | FileCheck %s
// RUN: cudaq-pulse-opt %s --mlir-print-op-generic | cudaq-pulse-opt | FileCheck %s

// Every classical op and type, and every argument and result descriptor.

// CHECK: classical.callable @decode {{{.*}}args = ["bits:2"]{{.*}}results = ["bits:1"]
classical.callable @decode {function_id = 3007678287 : i64, device = 0 : i64,
    args = ["bits:2"], results = ["bits:1"], latency_min_ns = 500.0,
    latency_nominal_ns = 900.0, latency_max_ns = 1500.0, mode = "bounded",
    submit = "eager"}
// CHECK: classical.callable @est {{{.*}}mode = "timeout"{{.*}}timeout_ns = 2.000000e+03
classical.callable @est {function_id = 2023343271 : i64, device = 1 : i64,
    args = ["iq:1", "levels:1", "f64", "i64", "u8", "i32", "f32", "f64[2]"],
    results = ["f64", "levels:1", "i32[2]", "iq:1"],
    latency_min_ns = 0.0, latency_nominal_ns = 1000.0, latency_max_ns = 1800.0,
    mode = "timeout", timeout_ns = 2000.0, submit = "lazy", deadline_ns = 5000.0}
// CHECK: classical.callable @f {{{.*}}function_id_override
classical.callable @f {function_id = 7 : i64, device = 0 : i64,
    args = ["u8[3]", "f32[1]"], results = [], latency_min_ns = 0.0,
    latency_nominal_ns = 0.0, latency_max_ns = 0.0, mode = "blocking",
    submit = "eager", function_id_override}

// CHECK-LABEL: func.func @main
func.func @main(%m0: !pulse.measurement, %m1: !pulse.measurement) -> i1 {
  // CHECK: %[[L:.*]] = classical.get_link {device = 0 : i64} : !classical.link
  %l = classical.get_link {device = 0 : i64} : !classical.link
  // CHECK: %[[L1:.*]], %[[H:.*]] = classical.submit %[[L]], @decode(%{{.*}}, %{{.*}}) : (!pulse.measurement, !pulse.measurement) -> (!classical.link, !classical.future)
  %l1, %h = classical.submit %l, @decode(%m0, %m1)
      : (!pulse.measurement, !pulse.measurement) -> (!classical.link, !classical.future)
  // CHECK: classical.wait %[[H]], @decode : i1
  %fix = classical.wait %h, @decode : i1

  %k = classical.get_link {device = 1 : i64} : !classical.link
  %x = arith.constant 1.5 : f64
  %n = arith.constant 3 : i64
  // CHECK: classical.submit %{{.*}}, @est
  %k1, %g = classical.submit %k, @est(%m0, %m1, %x, %n, %n, %n, %x, %x, %x)
      : (!pulse.measurement, !pulse.measurement, f64, i64, i64, i64, f64, f64, f64)
      -> (!classical.link, !classical.future)
  // CHECK: classical.wait %{{.*}}, @est : i1, f64, i64, i64, i64, f64, f64
  %ok, %v, %lvl, %a0, %a1, %i, %q = classical.wait %g, @est
      : i1, f64, i64, i64, i64, f64, f64

  // A call whose answer is not used: no wait.
  // CHECK: classical.submit %{{.*}}, @f
  %y = arith.constant 2 : i64
  %l2, %unused = classical.submit %l1, @f(%y, %y, %y, %x)
      : (i64, i64, i64, f64) -> (!classical.link, !classical.future)
  return %fix : i1
}
