// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s --split-input-file --qop-to-cudm --verify-diagnostics

// The lowering evolves the program's one qop.evolve, of a qop.lindblad.

// expected-error@+1 {{qop-to-cudm requires exactly one qop.evolve op}}
module {
func.func @main() {
  %c0 = arith.constant 0 : i64
  %sz = qop.spin(%c0) {kind = #qop<handler_kind spin_z>} : !qop.handler
  %coeff = qop.const_scalar {real = 1.0 : f64, imag = 0.0 : f64} : !qop.scalar
  %term = qop.make_product(%coeff, %sz) : !qop.product
  %H = qop.make_sum(%term) : !qop.op
  %L = qop.lindblad(%H, %H) : !qop.super_op
  return
}
}

// -----

module {
func.func @main() {
  %c0 = arith.constant 0 : i64
  %sz = qop.spin(%c0) {kind = #qop<handler_kind spin_z>} : !qop.handler
  %coeff = qop.const_scalar {real = 1.0 : f64, imag = 0.0 : f64} : !qop.scalar
  %term = qop.make_product(%coeff, %sz) : !qop.product
  %H = qop.make_sum(%term) : !qop.op
  %left = qop.super_left %H : !qop.super_op
  // expected-error@+1 {{generator must be a qop.lindblad}}
  qop.evolve %left : !qop.super_op
  return
}
}
