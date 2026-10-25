// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// RUN: cudaq-pulse-opt %s --split-input-file --verify-diagnostics

// One case per diagnostic of the cudm op verifiers.

func.func @state_without_modes(%h: !cudm.handle) {
  // expected-error @+1 {{'cudm.create_state' op mode_extents must have at least one element}}
  %s = cudm.create_state %h {purity = #cudm<purity pure>, data_type = #cudm<compute_type f64>, mode_extents = array<i64>} : (!cudm.handle) -> !cudm.state
  return
}

// -----

func.func @state_of_a_one_level_mode(%h: !cudm.handle) {
  // expected-error @+1 {{'cudm.create_state' op each mode extent must be >= 2, got 1}}
  %s = cudm.create_state %h {purity = #cudm<purity mixed>, data_type = #cudm<compute_type f64>, mode_extents = array<i64: 2, 1>} : (!cudm.handle) -> !cudm.state
  return
}

// -----

func.func @elementary_op_without_modes(%h: !cudm.handle, %t: tensor<1xf64>) {
  // expected-error @+1 {{'cudm.create_elementary_op' op mode_extents must have at least one element}}
  %e = cudm.create_elementary_op %h, %t {sparsity = #cudm<sparsity none>, data_type = #cudm<compute_type f64>, mode_extents = array<i64>} : !cudm.handle, tensor<1xf64> -> !cudm.elementary_op
  return
}

// -----

func.func @elementary_op_of_an_empty_mode(%h: !cudm.handle, %t: tensor<1xf64>) {
  // expected-error @+1 {{'cudm.create_elementary_op' op each mode extent must be >= 2, got 0}}
  %e = cudm.create_elementary_op %h, %t {sparsity = #cudm<sparsity none>, data_type = #cudm<compute_type f64>, mode_extents = array<i64: 0>} : !cudm.handle, tensor<1xf64> -> !cudm.elementary_op
  return
}

// -----

func.func @product_of_no_operators(%h: !cudm.handle, %term: !cudm.op_term) {
  // expected-error @+1 {{'cudm.append_elementary_product' op at least one elementary operator must be provided}}
  "cudm.append_elementary_product"(%h, %term) {modes_acted_on = array<i32>, duality = array<i32>, coeff_real = 1.0 : f64, coeff_imag = 0.0 : f64} : (!cudm.handle, !cudm.op_term) -> ()
  return
}

// -----

func.func @product_with_too_many_modes(%h: !cudm.handle, %term: !cudm.op_term, %e: !cudm.elementary_op) {
  // expected-error @+1 {{'cudm.append_elementary_product' op modes_acted_on size (2) must match number of elementary operators (1)}}
  cudm.append_elementary_product %h, %term, %e {modes_acted_on = array<i32: 0, 1>, duality = array<i32: 0>, coeff_real = 1.0 : f64, coeff_imag = 0.0 : f64} : !cudm.handle, !cudm.op_term, !cudm.elementary_op
  return
}

// -----

func.func @product_with_too_few_dualities(%h: !cudm.handle, %term: !cudm.op_term, %e: !cudm.elementary_op) {
  // expected-error @+1 {{'cudm.append_elementary_product' op duality size (0) must match number of elementary operators (1)}}
  cudm.append_elementary_product %h, %term, %e {modes_acted_on = array<i32: 0>, duality = array<i32>, coeff_real = 1.0 : f64, coeff_imag = 0.0 : f64} : !cudm.handle, !cudm.op_term, !cudm.elementary_op
  return
}

// -----

func.func @term_of_an_unknown_duality(%h: !cudm.handle, %op: !cudm.operator, %term: !cudm.op_term) {
  // expected-error @+1 {{'cudm.operator_append_term' op duality must be 0 (as built) or 1 (every mode action flipped between ket and bra), got 2}}
  cudm.operator_append_term %h, %op, %term {duality = 2 : i32, coeff_real = 1.0 : f64, coeff_imag = 0.0 : f64} : !cudm.handle, !cudm.operator, !cudm.op_term
  return
}

// -----

func.func @prepare_with_a_negative_workspace(%h: !cudm.handle, %op: !cudm.operator, %in: !cudm.state, %out: !cudm.state, %ws: !cudm.workspace) {
  // expected-error @+1 {{'cudm.operator_prepare_action' op workspace_limit must be non-negative, got -1}}
  cudm.operator_prepare_action %h, %op, %in, %out, %ws {compute_type = #cudm<compute_type f64>, workspace_limit = -1 : i64} : !cudm.handle, !cudm.operator, !cudm.state, !cudm.state, !cudm.workspace
  return
}

// -----

func.func @sse_of_no_trajectories(%h: !cudm.handle, %op: !cudm.operator, %s: !cudm.state, %ws: !cudm.workspace) {
  // expected-error @+1 {{'cudm.sse_evolve' op num_trajectories must be >= 1, got 0}}
  %r = "cudm.sse_evolve"(%h, %op, %s, %ws) {integrator = #cudm<integrator rk4>, num_trajectories = 0 : i64, t_start = 0.0 : f64, t_end = 1.0 : f64, num_steps = 10 : i64, seed = 1 : i64} : (!cudm.handle, !cudm.operator, !cudm.state, !cudm.workspace) -> !cudm.trajectory_result
  return
}

// -----

func.func @sse_of_negative_trajectories(%h: !cudm.handle, %op: !cudm.operator, %s: !cudm.state, %ws: !cudm.workspace) {
  // expected-error @+1 {{'cudm.sse_evolve' op num_trajectories must be >= 1, got -1}}
  %r = "cudm.sse_evolve"(%h, %op, %s, %ws) {integrator = #cudm<integrator rk4>, num_trajectories = -1 : i64, t_start = 0.0 : f64, t_end = 1.0 : f64, num_steps = 10 : i64, seed = 1 : i64} : (!cudm.handle, !cudm.operator, !cudm.state, !cudm.workspace) -> !cudm.trajectory_result
  return
}

// -----

func.func @sse_of_no_steps(%h: !cudm.handle, %op: !cudm.operator, %s: !cudm.state, %ws: !cudm.workspace) {
  // expected-error @+1 {{'cudm.sse_evolve' op num_steps must be >= 1, got 0}}
  %r = "cudm.sse_evolve"(%h, %op, %s, %ws) {integrator = #cudm<integrator rk4>, num_trajectories = 8 : i64, t_start = 0.0 : f64, t_end = 1.0 : f64, num_steps = 0 : i64, seed = 1 : i64} : (!cudm.handle, !cudm.operator, !cudm.state, !cudm.workspace) -> !cudm.trajectory_result
  return
}

// -----

func.func @sse_backwards_in_time(%h: !cudm.handle, %op: !cudm.operator, %s: !cudm.state, %ws: !cudm.workspace) {
  // expected-error @+1 {{'cudm.sse_evolve' op t_end (1.000000e+00) must be greater than t_start (2.000000e+00)}}
  %r = "cudm.sse_evolve"(%h, %op, %s, %ws) {integrator = #cudm<integrator rk4>, num_trajectories = 8 : i64, t_start = 2.0 : f64, t_end = 1.0 : f64, num_steps = 10 : i64, seed = 1 : i64} : (!cudm.handle, !cudm.operator, !cudm.state, !cudm.workspace) -> !cudm.trajectory_result
  return
}
