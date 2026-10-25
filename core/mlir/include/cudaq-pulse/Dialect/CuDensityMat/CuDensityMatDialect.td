/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 *****************************************************************************/

#ifndef CUDAQ_PULSE_DIALECT_CUDENSITYMAT
#define CUDAQ_PULSE_DIALECT_CUDENSITYMAT

include "mlir/IR/OpBase.td"

def CuDensityMatDialect : Dialect {
  let name = "cudm";
  let summary = "Dialect wrapping the NVIDIA cuDensityMat API for quantum dynamics";
  let description = [{
    The `cudm` dialect provides an MLIR representation of the NVIDIA cuDensityMat
    library API for GPU-accelerated quantum dynamics simulation. It serves as
    the lowering target for the `qop` dialect and the source for call
    generation targeting the cuDensityMat runtime API.

    Two tiers of operations:
    - **API ops**: wrappers around `cudensitymat*` C API functions, which
      `cudm-to-llvm` lowers to calls into the cudaq-pulse cuDensityMat
      runtime
    - **Solver ops**: `cudm.evolve`, which the runtime integrates with its own
      rk1, rk2, rk4, Magnus or Crank-Nicolson stepper, and the reserved
      `cudm.sse_evolve` and `cudm.apply_decoherence`, which nothing lowers yet
  }];
  let cppNamespace = "::cudm";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

#endif // CUDAQ_PULSE_DIALECT_CUDENSITYMAT
