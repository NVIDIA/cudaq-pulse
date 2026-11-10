// -*- tablegen -*-
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef CUDAQ_PULSE_DIALECT_CUDENSITYMAT_OPS
#define CUDAQ_PULSE_DIALECT_CUDENSITYMAT_OPS

include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatDialect.td"
include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatTypes.td"
include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatAttrs.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

class CuDensityMatOp<string mnemonic, list<Trait> traits = []> :
    Op<CuDensityMatDialect, mnemonic, traits>;

// ---- Context Management ----

def CUDM_InitHandleOp : CuDensityMatOp<"init_handle", [MemoryEffects<[MemWrite]>]> {
  let summary = "Create cuDensityMat library context";
  let description = [{
    A cuDensityMat library context (`cudensitymatCreate`). The lowered
    program creates one per run, builds its states and operators on it
    and destroys it last; the run's drive-envelope callbacks belong to
    it.
  }];
  let results = (outs CUDM_HandleType:$handle);
  let assemblyFormat = "attr-dict `:` type($handle)";
}

def CUDM_DestroyHandleOp : CuDensityMatOp<"destroy_handle", [MemoryEffects<[MemWrite]>]> {
  let summary = "Destroy cuDensityMat library context";
  let description = [{
    Destroys a library context (`cudensitymatDestroy`) after everything
    created on it.
  }];
  let arguments = (ins CUDM_HandleType:$handle);
  let assemblyFormat = "$handle attr-dict `:` type($handle)";
}

// ---- Workspace Management ----

def CUDM_CreateWorkspaceOp : CuDensityMatOp<"create_workspace", [MemoryEffects<[MemWrite]>]> {
  let summary = "Create a workspace descriptor";
  let description = [{
    A workspace descriptor (`cudensitymatCreateWorkspace`). The runtime
    allocates its scratch memory when an operator action is prepared.
  }];
  let arguments = (ins CUDM_HandleType:$handle);
  let results = (outs CUDM_WorkspaceType:$workspace);
  let assemblyFormat = "$handle attr-dict `:` functional-type($handle, $workspace)";
}

def CUDM_DestroyWorkspaceOp : CuDensityMatOp<"destroy_workspace", [MemoryEffects<[MemWrite]>]> {
  let summary = "Destroy a workspace descriptor";
  let description = [{
    Destroys a workspace descriptor and frees its memory.
  }];
  let arguments = (ins CUDM_WorkspaceType:$workspace);
  let assemblyFormat = "$workspace attr-dict `:` type($workspace)";
}

// ---- Quantum State ----

def CUDM_CreateStateOp : CuDensityMatOp<"create_state", [MemoryEffects<[MemWrite]>]> {
  let summary = "Define an empty quantum state of given purity and shape";
  let description = [{
    A state over modes of `mode_extents` levels (each at least 2): a pure
    state vector or, with `purity` mixed, a density matrix, in
    complex128. It starts as the ground state, or as the initial state
    the host set for the next allocation on the thread.
  }];
  let arguments = (ins
    CUDM_HandleType:$handle,
    CUDM_StatePurityAttr:$purity,
    CUDM_ComputeTypeAttr:$data_type,
    DenseI64ArrayAttr:$mode_extents,
    DefaultValuedAttr<I64Attr, "0">:$batch_size,
    DefaultValuedAttr<BoolAttr, "false">:$distributed
  );
  let results = (outs CUDM_StateType:$state);
  let assemblyFormat = "$handle attr-dict `:` functional-type($handle, $state)";
  let hasVerifier = 1;
}

def CUDM_DestroyStateOp : CuDensityMatOp<"destroy_state", [MemoryEffects<[MemWrite]>]> {
  let summary = "Destroy a quantum state";
  let description = [{
    Destroys a state and frees its storage.
  }];
  let arguments = (ins CUDM_StateType:$state);
  let assemblyFormat = "$state attr-dict `:` type($state)";
}

// ---- Elementary Operator ----

def CUDM_CreateElementaryOpOp
    : CuDensityMatOp<"create_elementary_op", [MemoryEffects<[MemWrite]>]> {
  let summary = "Create an elementary tensor operator";
  let description = [{
    A dense elementary operator on modes of `mode_extents` levels, from
    `tensor_data` (column-major, interleaved real and imaginary parts, as
    `qop.matrix_leaf` holds it), copied to the device.
  }];
  let arguments = (ins
    CUDM_HandleType:$handle,
    AnyType:$tensor_data,
    CUDM_SparsityAttr:$sparsity,
    CUDM_ComputeTypeAttr:$data_type,
    DenseI64ArrayAttr:$mode_extents,
    OptionalAttr<FlatSymbolRefAttr>:$callback
  );
  let results = (outs CUDM_ElementaryOpType:$elem_op);
  let assemblyFormat = "$handle `,` $tensor_data attr-dict `:` type($handle) `,` type($tensor_data) `->` type($elem_op)";
  let hasVerifier = 1;
}

def CUDM_DestroyElementaryOpOp
    : CuDensityMatOp<"destroy_elementary_op", [MemoryEffects<[MemWrite]>]> {
  let summary = "Destroy an elementary tensor operator";
  let description = [{
    Destroys an elementary operator and frees its device data.
  }];
  let arguments = (ins CUDM_ElementaryOpType:$elem_op);
  let assemblyFormat = "$elem_op attr-dict `:` type($elem_op)";
}

// ---- Operator Term ----

def CUDM_CreateOpTermOp : CuDensityMatOp<"create_op_term", [MemoryEffects<[MemWrite]>]> {
  let summary = "Create an empty operator term";
  let description = [{
    An empty operator term over modes of `mode_extents` levels, to which
    products of elementary operators are appended.
  }];
  let arguments = (ins CUDM_HandleType:$handle, DenseI64ArrayAttr:$mode_extents);
  let results = (outs CUDM_OpTermType:$op_term);
  let assemblyFormat = "$handle attr-dict `:` functional-type($handle, $op_term)";
}

def CUDM_AppendElementaryProductOp
    : CuDensityMatOp<"append_elementary_product", [MemoryEffects<[MemWrite]>]> {
  let summary = "Append a product of elementary operators to a term";
  let description = [{
    Appends to a term the product of `elem_ops` on the modes
    `modes_acted_on`, weighted by `coeff_real + i coeff_imag`. Each
    `duality` entry says whether its operator acts on the ket (0) or the
    bra (1) side of its mode. With `callback`,
    the weight is also multiplied by that drive envelope at each time;
    the lowering carries the envelope's kind and parameters as
    attributes.
  }];
  let arguments = (ins
    CUDM_HandleType:$handle,
    CUDM_OpTermType:$op_term,
    Variadic<CUDM_ElementaryOpType>:$elem_ops,
    DenseI32ArrayAttr:$modes_acted_on,
    DenseI32ArrayAttr:$duality,
    F64Attr:$coeff_real,
    F64Attr:$coeff_imag,
    OptionalAttr<FlatSymbolRefAttr>:$callback
  );
  let assemblyFormat = "$handle `,` $op_term `,` $elem_ops attr-dict `:` type($handle) `,` type($op_term) `,` type($elem_ops)";
  let hasVerifier = 1;
}

def CUDM_DestroyOpTermOp : CuDensityMatOp<"destroy_op_term", [MemoryEffects<[MemWrite]>]> {
  let summary = "Destroy an operator term";
  let description = [{
    Destroys an operator term and releases its envelope callbacks.
  }];
  let arguments = (ins CUDM_OpTermType:$op_term);
  let assemblyFormat = "$op_term attr-dict `:` type($op_term)";
}

// ---- Composite Operator ----

def CUDM_CreateOperatorOp : CuDensityMatOp<"create_operator", [MemoryEffects<[MemWrite]>]> {
  let summary = "Create an empty composite operator (super-operator)";
  let description = [{
    An empty composite operator (a super-operator on the state) over
    modes of `mode_extents` levels.
  }];
  let arguments = (ins CUDM_HandleType:$handle, DenseI64ArrayAttr:$mode_extents);
  let results = (outs CUDM_OperatorType:$op);
  let assemblyFormat = "$handle attr-dict `:` functional-type($handle, $op)";
}

def CUDM_OperatorAppendTermOp
    : CuDensityMatOp<"operator_append_term", [MemoryEffects<[MemWrite]>]> {
  let summary = "Append an operator term to a composite operator";
  let description = [{
    Appends a term to a composite operator, weighted by
    `coeff_real + i coeff_imag`. With `duality` 1, every mode action of
    the term is flipped between the ket and bra sides; the lowering uses
    this for the bra half of `-i [H, rho]`.
  }];
  let arguments = (ins
    CUDM_HandleType:$handle,
    CUDM_OperatorType:$op,
    CUDM_OpTermType:$term,
    I32Attr:$duality,
    F64Attr:$coeff_real,
    F64Attr:$coeff_imag,
    OptionalAttr<FlatSymbolRefAttr>:$callback
  );
  let assemblyFormat = "$handle `,` $op `,` $term attr-dict `:` type($handle) `,` type($op) `,` type($term)";
  let hasVerifier = 1;
}

def CUDM_DestroyOperatorOp : CuDensityMatOp<"destroy_operator", [MemoryEffects<[MemWrite]>]> {
  let summary = "Destroy a composite operator";
  let description = [{
    Destroys a composite operator.
  }];
  let arguments = (ins CUDM_OperatorType:$op);
  let assemblyFormat = "$op attr-dict `:` type($op)";
}

// ---- Operator Action ----

def CUDM_OperatorPrepareActionOp
    : CuDensityMatOp<"operator_prepare_action", [MemoryEffects<[MemWrite]>]> {
  let summary = "Prepare operator for action on quantum state";
  let description = [{
    Prepares the action of an operator on states of the given shape
    within `workspace_limit` bytes of workspace
    (`cudensitymatOperatorPrepareAction`).
  }];
  let arguments = (ins
    CUDM_HandleType:$handle, CUDM_OperatorType:$op,
    CUDM_StateType:$state_in, CUDM_StateType:$state_out,
    CUDM_ComputeTypeAttr:$compute_type,
    I64Attr:$workspace_limit,
    CUDM_WorkspaceType:$workspace
  );
  let assemblyFormat = "$handle `,` $op `,` $state_in `,` $state_out `,` $workspace attr-dict `:` type($handle) `,` type($op) `,` type($state_in) `,` type($state_out) `,` type($workspace)";
  let hasVerifier = 1;
}

def CUDM_OperatorComputeActionOp
    : CuDensityMatOp<"operator_compute_action", [MemoryEffects<[MemWrite]>]> {
  let summary = "Compute operator action: state_out += Op(t, params) * state_in";
  let description = [{
    Accumulates the action of an operator at `time` and parameters
    `params` on `state_in` into `state_out`
    (`cudensitymatOperatorComputeAction`).
  }];
  let arguments = (ins
    CUDM_HandleType:$handle, CUDM_OperatorType:$op,
    F64:$time, I64:$batch_size, I32:$num_params, AnyType:$params,
    CUDM_StateType:$state_in, CUDM_StateType:$state_out,
    CUDM_WorkspaceType:$workspace
  );
  let assemblyFormat = "$handle `,` $op `,` $time `,` $batch_size `,` $num_params `,` $params `,` $state_in `,` $state_out `,` $workspace attr-dict `:` type($handle) `,` type($op) `,` type($params) `,` type($state_in) `,` type($state_out) `,` type($workspace)";
  let hasVerifier = 1;
}

// ---- Expectation ----

def CUDM_CreateExpectationOp
    : CuDensityMatOp<"create_expectation", [MemoryEffects<[MemWrite]>]> {
  let summary = "Create expectation value computation object";
  let description = [{
    An object computing the expectation value of an operator
    (`cudensitymatCreateExpectation`). The execution lowering does not use
    it yet; observables are reserved.
  }];
  let arguments = (ins CUDM_HandleType:$handle, CUDM_OperatorType:$op);
  let results = (outs CUDM_ExpectationType:$expectation);
  let assemblyFormat = "$handle `,` $op attr-dict `:` type($handle) `,` type($op) `->` type($expectation)";
}

def CUDM_ExpectationComputeOp
    : CuDensityMatOp<"expectation_compute", [MemoryEffects<[MemWrite]>]> {
  let summary = "Compute operator expectation value(s)";
  let description = [{
    The expectation value of an operator in `state` at `time`.
  }];
  let arguments = (ins
    CUDM_HandleType:$handle, CUDM_ExpectationType:$expectation,
    F64:$time, I64:$batch_size, I32:$num_params, AnyType:$params,
    CUDM_StateType:$state, CUDM_WorkspaceType:$workspace
  );
  let results = (outs AnyType:$value);
  let assemblyFormat = "$handle `,` $expectation `,` $time `,` $batch_size `,` $num_params `,` $params `,` $state `,` $workspace attr-dict `:` type($handle) `,` type($expectation) `,` type($params) `,` type($state) `,` type($workspace) `->` type($value)";
  let hasVerifier = 1;
}

def CUDM_DestroyExpectationOp
    : CuDensityMatOp<"destroy_expectation", [MemoryEffects<[MemWrite]>]> {
  let summary = "Destroy expectation computation object";
  let description = [{
    Destroys an expectation object.
  }];
  let arguments = (ins CUDM_ExpectationType:$expectation);
  let assemblyFormat = "$expectation attr-dict `:` type($expectation)";
}

// ---- Tier 2: Integration / Solver ops ----

def CUDM_EvolveOp : CuDensityMatOp<"evolve", [MemoryEffects<[MemWrite]>]> {
  let summary = "Time-evolve a quantum state under an operator";
  let description = [{
    Evolves `state_in` under the operator from `t_start` to `t_end` in
    `num_steps` equal steps of the `integrator` (rk1, rk2, rk4, magnus or
    crank_nicolson), writing the final state to `state_out`, which is
    also the result. `cudm-to-llvm` lowers it to the runtime's
    `cudm_evolve` followed by a capture of the result for the host.
  }];
  let arguments = (ins
    CUDM_HandleType:$handle, CUDM_OperatorType:$op,
    CUDM_StateType:$state_in, CUDM_StateType:$state_out,
    CUDM_WorkspaceType:$workspace,
    CUDM_IntegratorKindAttr:$integrator,
    F64Attr:$t_start, F64Attr:$t_end, I64Attr:$num_steps,
    OptionalAttr<CUDM_ComputeTypeAttr>:$compute_type
  );
  let results = (outs CUDM_StateType:$result);
  let assemblyFormat = "$handle `,` $op `,` $state_in `,` $state_out `,` $workspace attr-dict `:` type($handle) `,` type($op) `,` type($state_in) `,` type($state_out) `,` type($workspace) `->` type($result)";
}

def CUDM_ApplyDecoherenceOp : CuDensityMatOp<"apply_decoherence", [MemoryEffects<[MemWrite]>]> {
  let summary = "Apply Lindblad T1/T2 decoherence for one time step";
  let description = [{
    Applies T1 and T2 decoherence of `model` over one step `dt`. The
    execution lowering puts decoherence in the Lindblad operator instead
    and does not use this op yet.
  }];
  let arguments = (ins
    CUDM_HandleType:$handle, CUDM_StateType:$state,
    CUDM_DecoherenceModelType:$model, F64:$dt
  );
  let assemblyFormat = "$handle `,` $state `,` $model `,` $dt attr-dict `:` type($handle) `,` type($state) `,` type($model)";
}

def CUDM_SSEEvolveOp : CuDensityMatOp<"sse_evolve", [MemoryEffects<[MemWrite]>]> {
  let summary = "Stochastic Schrodinger equation ensemble evolution";
  let description = [{
    Evolves `num_trajectories` stochastic Schrodinger trajectories under
    a Hamiltonian and collapse operators from `t_start` to `t_end` in
    `num_steps` steps, seeded by `seed`. The counts must be at least 1
    and `t_end` later than `t_start`. Reserved for trajectory
    simulation; nothing lowers it yet.
  }];
  let arguments = (ins
    CUDM_HandleType:$handle,
    CUDM_OperatorType:$hamiltonian,
    Variadic<CUDM_ElementaryOpType>:$collapse_ops,
    CUDM_StateType:$initial_state,
    CUDM_WorkspaceType:$workspace,
    CUDM_IntegratorKindAttr:$integrator,
    I64Attr:$num_trajectories,
    F64Attr:$t_start, F64Attr:$t_end, I64Attr:$num_steps,
    I64Attr:$seed
  );
  let results = (outs CUDM_TrajectoryResultType:$result);
  let assemblyFormat = "$handle `,` $hamiltonian `,` `(` $collapse_ops `)` `,` $initial_state `,` $workspace attr-dict `:` type($handle) `,` type($hamiltonian) `,` `(` type($collapse_ops) `)` `,` type($initial_state) `,` type($workspace) `->` type($result)";
  let hasVerifier = 1;
}

#endif // CUDAQ_PULSE_DIALECT_CUDENSITYMAT_OPS
