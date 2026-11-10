// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s --split-input-file --verify-diagnostics

// One case per diagnostic of the qop op verifiers.

func.func @spin_of_a_boson_kind(%q: i64) {
  // expected-error @+1 {{'qop.spin' op invalid handler kind for spin operator}}
  %h = qop.spin(%q) {kind = #qop<handler_kind boson_create>} : !qop.handler
  return
}

// -----

func.func @boson_of_a_spin_kind(%q: i64) {
  // expected-error @+1 {{'qop.boson' op invalid handler kind for boson operator}}
  %h = qop.boson(%q) {dimension = 3 : i64, kind = #qop<handler_kind spin_x>} : !qop.handler
  return
}

// -----

func.func @boson_of_one_level(%q: i64) {
  // expected-error @+1 {{'qop.boson' op boson dimension must be >= 2, got 1}}
  %h = qop.boson(%q) {dimension = 1 : i64, kind = #qop<handler_kind boson_number>} : !qop.handler
  return
}

// -----

func.func @boson_of_a_negative_dimension(%q: i64) {
  // expected-error @+1 {{'qop.boson' op boson dimension must be >= 2, got -2}}
  %h = qop.boson(%q) {dimension = -2 : i64, kind = #qop<handler_kind boson_number>} : !qop.handler
  return
}

// -----

func.func @fermion_of_a_spin_kind(%q: i64) {
  // expected-error @+1 {{'qop.fermion' op invalid handler kind for fermion operator}}
  %h = qop.fermion(%q) {kind = #qop<handler_kind spin_z>} : !qop.handler
  return
}

// -----

func.func @matrix_leaf_without_targets(%m: tensor<2x2xf64>) {
  // expected-error @+1 {{'qop.matrix_leaf' op matrix leaf must target at least one mode}}
  %h = qop.matrix_leaf(%m) {targets = array<i64>, dimensions = array<i64>} : tensor<2x2xf64> -> !qop.handler
  return
}

// -----

func.func @matrix_leaf_with_fewer_dimensions_than_targets(%m: tensor<4x4xf64>) {
  // expected-error @+1 {{'qop.matrix_leaf' op number of targets (2) must match number of dimensions (1)}}
  %h = qop.matrix_leaf(%m) {targets = array<i64: 0, 1>, dimensions = array<i64: 2>} : tensor<4x4xf64> -> !qop.handler
  return
}

// -----

func.func @callback_scalar_without_a_name() {
  // expected-error @+1 {{'qop.callback_scalar' op callback symbol name must not be empty}}
  %s = qop.callback_scalar @"" : !qop.scalar
  return
}

// -----

func.func @sum_of_no_terms() {
  // expected-error @+1 {{'qop.make_sum' op sum must have at least one product term}}
  %o = qop.make_sum() : !qop.op
  return
}

// -----

func.func @to_matrix_without_dimensions(%o: !qop.op, %t: f64, %p: tensor<0xf64>) {
  // expected-error @+1 {{'qop.to_matrix' op dimensions array must not be empty}}
  %m = qop.to_matrix %o, %t, %p {dimensions = array<i64>} : !qop.op, tensor<0xf64> -> tensor<1x1xcomplex<f64>>
  return
}

// -----

func.func @to_matrix_of_a_one_level_mode(%o: !qop.op, %t: f64, %p: tensor<0xf64>) {
  // expected-error @+1 {{'qop.to_matrix' op each dimension must be >= 2, got 1}}
  %m = qop.to_matrix %o, %t, %p {dimensions = array<i64: 2, 1>} : !qop.op, tensor<0xf64> -> tensor<2x2xcomplex<f64>>
  return
}
