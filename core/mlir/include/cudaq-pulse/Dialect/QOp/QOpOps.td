// -*- tablegen -*-
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef CUDAQ_PULSE_DIALECT_QOP_OPS
#define CUDAQ_PULSE_DIALECT_QOP_OPS

include "cudaq-pulse/Dialect/QOp/QOpDialect.td"
include "cudaq-pulse/Dialect/QOp/QOpTypes.td"
include "cudaq-pulse/Dialect/QOp/QOpAttrs.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

class QOpOp<string mnemonic, list<Trait> traits = []> :
    Op<QOpDialect, mnemonic, traits>;

// ---- Leaf operator construction ----

def QOP_SpinOp : QOpOp<"spin", [Pure]> {
  let summary = "Create a Pauli spin operator on a target qubit";
  let description = [{
    A Pauli or ladder operator on qubit `target`: `kind` is one of
    `spin_i`, `spin_x`, `spin_y`, `spin_z`, `spin_lowering` or
    `spin_raising`. `pulse-to-qop` uses it for two-level qudits.
  }];
  let arguments = (ins I64:$target, QOP_HandlerKindAttr:$kind);
  let results = (outs QOP_HandlerType:$result);
  let assemblyFormat = "`(` $target `)` attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def QOP_BosonOp : QOpOp<"boson", [Pure]> {
  let summary = "Create a bosonic operator on a target mode";
  let description = [{
    A bosonic operator on mode `target`, truncated to `dimension` levels
    (at least 2): `kind` is `boson_i`, `boson_create`, `boson_annihilate`
    or `boson_number`.


    The pulse lowering does not produce it and `qop-to-cudm` does not
    lower it yet; it is part of the operator algebra planned for later
    backends.
  }];
  let arguments = (ins I64:$target, I64Attr:$dimension, QOP_HandlerKindAttr:$kind);
  let results = (outs QOP_HandlerType:$result);
  let assemblyFormat = "`(` $target `)` attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def QOP_FermionOp : QOpOp<"fermion", [Pure]> {
  let summary = "Create a fermionic operator on a target mode";
  let description = [{
    A fermionic operator on mode `target`: `kind` is `fermion_i`,
    `fermion_create`, `fermion_annihilate` or `fermion_number`.


    The pulse lowering does not produce it and `qop-to-cudm` does not
    lower it yet; it is part of the operator algebra planned for later
    backends.
  }];
  let arguments = (ins I64:$target, QOP_HandlerKindAttr:$kind);
  let results = (outs QOP_HandlerType:$result);
  let assemblyFormat = "`(` $target `)` attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def QOP_MatrixLeafOp : QOpOp<"matrix_leaf", [Pure]> {
  let summary = "Create a user-defined matrix leaf operator";
  let description = [{
    A dense operator on the modes `targets`, of `dimensions` levels each.
    `data` holds the matrix column-major with interleaved real and
    imaginary parts, as a `tensor<d x d x 2 x f64>`. `pulse-to-qop` uses
    it for qudits of more than two levels.
  }];
  let arguments = (ins
    DenseI64ArrayAttr:$targets,
    AnyType:$data,
    DenseI64ArrayAttr:$dimensions
  );
  let results = (outs QOP_HandlerType:$result);
  let assemblyFormat = "`(` $data `)` attr-dict `:` type($data) `->` type($result)";
  let hasVerifier = 1;
}

// ---- Scalar coefficient construction ----

def QOP_ConstScalarOp : QOpOp<"const_scalar", [Pure]> {
  let summary = "Create a constant complex scalar coefficient";
  let description = [{
    The constant complex coefficient `real + i imag`.
  }];
  let arguments = (ins F64Attr:$real, F64Attr:$imag);
  let results = (outs QOP_ScalarType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def QOP_CallbackScalarOp : QOpOp<"callback_scalar", [Pure]> {
  let summary = "Create a time/parameter-dependent scalar coefficient";
  let description = [{
    A time-dependent coefficient: the drive envelope named `callback`.
    `pulse-to-qop` attaches the envelope's kind and parameters as
    attributes, which `qop-to-cudm` passes to the cuDensityMat runtime
    for evaluation at each time step.
  }];
  let arguments = (ins FlatSymbolRefAttr:$callback);
  let results = (outs QOP_ScalarType:$result);
  let assemblyFormat = "$callback attr-dict `:` type($result)";
  let hasVerifier = 1;
}

// ---- Operator composition ----

def QOP_MakeProductOp : QOpOp<"make_product", [Pure]> {
  let summary = "Create a tensor product of handlers with a coefficient";
  let description = [{
    The tensor product of the handlers `factors`, weighted by
    `coefficient`.
  }];
  let arguments = (ins QOP_ScalarType:$coefficient, Variadic<QOP_HandlerType>:$factors);
  let results = (outs QOP_ProductType:$result);
  let assemblyFormat = "`(` $coefficient `,` $factors `)` attr-dict `:` type($result)";
}

def QOP_MakeSumOp : QOpOp<"make_sum", [Pure]> {
  let summary = "Create a sum of product terms";
  let description = [{
    The sum of one or more product terms: the general operator, such as a
    Hamiltonian or a collapse operator.
  }];
  let arguments = (ins Variadic<QOP_ProductType>:$terms);
  let results = (outs QOP_OpType:$result);
  let assemblyFormat = "`(` $terms `)` attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def QOP_AddOp : QOpOp<"add", [Pure]> {
  let summary = "Add two operators";
  let description = [{
    The sum `lhs + rhs` of two operators.


    The pulse lowering does not produce it and `qop-to-cudm` does not
    lower it yet; it is part of the operator algebra planned for later
    backends.
  }];
  let arguments = (ins QOP_OpType:$lhs, QOP_OpType:$rhs);
  let results = (outs QOP_OpType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def QOP_ScaleOp : QOpOp<"scale", [Pure]> {
  let summary = "Multiply an operator by a scalar coefficient";
  let description = [{
    The operator `operand` times the scalar `coefficient`.


    The pulse lowering does not produce it and `qop-to-cudm` does not
    lower it yet; it is part of the operator algebra planned for later
    backends.
  }];
  let arguments = (ins QOP_ScalarType:$coefficient, QOP_OpType:$operand);
  let results = (outs QOP_OpType:$result);
  let assemblyFormat = "$coefficient `,` $operand attr-dict `:` type($result)";
}

def QOP_DaggerOp : QOpOp<"dagger", [Pure]> {
  let summary = "Hermitian conjugate of an operator";
  let description = [{
    The Hermitian conjugate of an operator.


    The pulse lowering does not produce it and `qop-to-cudm` does not
    lower it yet; it is part of the operator algebra planned for later
    backends.
  }];
  let arguments = (ins QOP_OpType:$operand);
  let results = (outs QOP_OpType:$result);
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

def QOP_CommutatorOp : QOpOp<"commutator", [Pure]> {
  let summary = "Compute [A, B] = AB - BA";
  let description = [{
    The commutator `lhs rhs - rhs lhs`.


    The pulse lowering does not produce it and `qop-to-cudm` does not
    lower it yet; it is part of the operator algebra planned for later
    backends.
  }];
  let arguments = (ins QOP_OpType:$lhs, QOP_OpType:$rhs);
  let results = (outs QOP_OpType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def QOP_AnticommutatorOp : QOpOp<"anticommutator", [Pure]> {
  let summary = "Compute {A, B} = AB + BA";
  let description = [{
    The anticommutator `lhs rhs + rhs lhs`.


    The pulse lowering does not produce it and `qop-to-cudm` does not
    lower it yet; it is part of the operator algebra planned for later
    backends.
  }];
  let arguments = (ins QOP_OpType:$lhs, QOP_OpType:$rhs);
  let results = (outs QOP_OpType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

// ---- Lindblad / super-operator construction ----

def QOP_LindbladOp : QOpOp<"lindblad", [Pure]> {
  let summary = "Construct Lindblad super-operator from H and collapse ops";
  let description = [{
    The Lindblad generator of a Hamiltonian and collapse operators `L_k`:
    `-i [H, rho] + sum_k (L_k rho L_k^dagger - {L_k^dagger L_k, rho} / 2)`.
    `pulse-to-qop` builds one per program from the drives, couplings and
    the target's T1 and T2, and `qop-to-cudm` lowers it to a cuDensityMat
    operator and `cudm.evolve`.
  }];
  let arguments = (ins QOP_OpType:$hamiltonian, Variadic<QOP_OpType>:$collapse_ops);
  let results = (outs QOP_SuperOpType:$result);
  let assemblyFormat = "`(` $hamiltonian `,` $collapse_ops `)` attr-dict `:` type($result)";
}

def QOP_SuperLeftOp : QOpOp<"super_left", [Pure]> {
  let summary = "Super-operator for left multiplication: S[rho] = A * rho";
  let description = [{
    The super-operator `rho -> A rho` of left multiplication by `operand`.


    The pulse lowering does not produce it and `qop-to-cudm` does not
    lower it yet; it is part of the operator algebra planned for later
    backends.
  }];
  let arguments = (ins QOP_OpType:$operand);
  let results = (outs QOP_SuperOpType:$result);
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

def QOP_SuperRightOp : QOpOp<"super_right", [Pure]> {
  let summary = "Super-operator for right multiplication: S[rho] = rho * A";
  let description = [{
    The super-operator `rho -> rho A` of right multiplication by
    `operand`.


    The pulse lowering does not produce it and `qop-to-cudm` does not
    lower it yet; it is part of the operator algebra planned for later
    backends.
  }];
  let arguments = (ins QOP_OpType:$operand);
  let results = (outs QOP_SuperOpType:$result);
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

def QOP_SuperBothOp : QOpOp<"super_both", [Pure]> {
  let summary = "Super-operator: S[rho] = A * rho * B";
  let description = [{
    The super-operator `rho -> A rho B` for `left` = A and `right` = B.


    The pulse lowering does not produce it and `qop-to-cudm` does not
    lower it yet; it is part of the operator algebra planned for later
    backends.
  }];
  let arguments = (ins QOP_OpType:$left, QOP_OpType:$right);
  let results = (outs QOP_SuperOpType:$result);
  let assemblyFormat = "$left `,` $right attr-dict `:` type($result)";
}

def QOP_SuperAddOp : QOpOp<"super_add", [Pure]> {
  let summary = "Add two super-operators";
  let description = [{
    The sum of two super-operators.


    The pulse lowering does not produce it and `qop-to-cudm` does not
    lower it yet; it is part of the operator algebra planned for later
    backends.
  }];
  let arguments = (ins QOP_SuperOpType:$lhs, QOP_SuperOpType:$rhs);
  let results = (outs QOP_SuperOpType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

// ---- Evaluation / materialization ----

def QOP_ToMatrixOp : QOpOp<"to_matrix", [MemoryEffects<[MemAlloc, MemWrite]>]> {
  let summary = "Materialize operator to dense matrix at given time/params";
  let description = [{
    The dense matrix of `operand` at `time` and parameters `params`, over
    modes of `dimensions` levels each (all at least 2).


    The pulse lowering does not produce it and `qop-to-cudm` does not
    lower it yet; it is part of the operator algebra planned for later
    backends.
  }];
  let arguments = (ins
    QOP_OpType:$operand,
    DenseI64ArrayAttr:$dimensions,
    F64:$time,
    AnyType:$params
  );
  let results = (outs AnyType:$matrix);
  let assemblyFormat = "$operand `,` $time `,` $params attr-dict `:` type($operand) `,` type($params) `->` type($matrix)";
  let hasVerifier = 1;
}

def QOP_DegreesOp : QOpOp<"degrees", [MemoryEffects<[MemAlloc, MemWrite]>]> {
  let summary = "Query which degrees of freedom an operator acts on";
  let description = [{
    The modes an operator acts on.


    The pulse lowering does not produce it and `qop-to-cudm` does not
    lower it yet; it is part of the operator algebra planned for later
    backends.
  }];
  let arguments = (ins QOP_OpType:$operand);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$operand attr-dict `:` type($operand) `->` type($result)";
  let hasVerifier = 1;
}

#endif // CUDAQ_PULSE_DIALECT_QOP_OPS
