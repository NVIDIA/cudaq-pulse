/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 *****************************************************************************/

#ifndef CUDAQ_PULSE_DIALECT_CUDENSITYMAT_ATTRS
#define CUDAQ_PULSE_DIALECT_CUDENSITYMAT_ATTRS

include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatDialect.td"
include "mlir/IR/EnumAttr.td"

def CUDM_StatePurity : I32EnumAttr<"StatePurity",
    "Quantum state purity",
    [
      I32EnumAttrCase<"Pure", 0, "pure">,
      I32EnumAttrCase<"Mixed", 1, "mixed">
    ]> {
  let cppNamespace = "::cudm";
  let genSpecializedAttr = 0;
}
def CUDM_StatePurityAttr : EnumAttr<CuDensityMatDialect, CUDM_StatePurity, "purity">;

def CUDM_ComputeType : I32EnumAttr<"ComputeType",
    "Floating-point compute precision; the runtime computes in complex128 only",
    [
      I32EnumAttrCase<"F64", 16, "f64">
    ]> {
  let cppNamespace = "::cudm";
  let genSpecializedAttr = 0;
}
def CUDM_ComputeTypeAttr : EnumAttr<CuDensityMatDialect, CUDM_ComputeType, "compute_type">;

def CUDM_Sparsity : I32EnumAttr<"Sparsity",
    "Elementary operator sparsity pattern",
    [
      I32EnumAttrCase<"None", 0, "none">,
      I32EnumAttrCase<"Multidiagonal", 1, "multidiagonal">
    ]> {
  let cppNamespace = "::cudm";
  let genSpecializedAttr = 0;
}
def CUDM_SparsityAttr : EnumAttr<CuDensityMatDialect, CUDM_Sparsity, "sparsity">;

def CUDM_IntegratorKind : I32EnumAttr<"IntegratorKind",
    "Time-integration strategy",
    [
      I32EnumAttrCase<"RungeKutta1", 2, "rk1">,
      I32EnumAttrCase<"RungeKutta2", 3, "rk2">,
      I32EnumAttrCase<"RungeKutta4", 4, "rk4">,
      I32EnumAttrCase<"Magnus", 5, "magnus">,
      I32EnumAttrCase<"CrankNicolson", 6, "crank_nicolson">
    ]> {
  let cppNamespace = "::cudm";
  let genSpecializedAttr = 0;
}
def CUDM_IntegratorKindAttr : EnumAttr<CuDensityMatDialect, CUDM_IntegratorKind, "integrator">;

#endif // CUDAQ_PULSE_DIALECT_CUDENSITYMAT_ATTRS
