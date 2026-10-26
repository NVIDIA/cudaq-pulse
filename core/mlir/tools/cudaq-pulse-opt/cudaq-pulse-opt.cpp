/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"
#include "cudaq-pulse/Dialect/Pulse/PulseEnums.h.inc"
#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseOps.h.inc"

#include "cudaq-pulse/Dialect/Classical/ClassicalDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Classical/ClassicalTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Classical/ClassicalOps.h.inc"

#include "cudaq-pulse/Dialect/QOp/QOpDialect.h.inc"
#include "cudaq-pulse/Dialect/QOp/QOpEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "cudaq-pulse/Dialect/QOp/QOpAttrs.h.inc"
#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/QOp/QOpTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/QOp/QOpOps.h.inc"

#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatDialect.h.inc"
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatAttrs.h.inc"
#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatOps.h.inc"

// Conversion passes
namespace pulse {
std::unique_ptr<mlir::Pass> createPulseToQOpPass();
}
namespace qop {
std::unique_ptr<mlir::Pass> createQOpToCuDensityMatPass();
}
namespace cudm {
std::unique_ptr<mlir::Pass> createCuDensityMatToLLVMPass();
}
// Pulse Transforms passes
namespace pulse {
std::unique_ptr<mlir::Pass> createPulseVerifyPass();
std::unique_ptr<mlir::Pass> createPulseCanonicalizePass();
std::unique_ptr<mlir::Pass> createVirtualZPass();
std::unique_ptr<mlir::Pass> createPulseFusionPass();
std::unique_ptr<mlir::Pass> createPulseScheduleAsapPass();
std::unique_ptr<mlir::Pass> createPulseScheduleAlapPass();
std::unique_ptr<mlir::Pass> createPulseInlineScheduleBlocksPass();
std::unique_ptr<mlir::Pass> createPulseClassicalCheckPass();
std::unique_ptr<mlir::Pass> createPulseScheduleRcpPass(int64_t maxDrives,
                                                       int64_t maxReadouts,
                                                       int64_t readoutLatency,
                                                       int64_t switchPenalty,
                                                       bool isAlap);
} // namespace pulse

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::DialectRegistry registry;
  registry.insert<pulse::PulseDialect>();
  registry.insert<classical::ClassicalDialect>();
  registry.insert<qop::QOpDialect>();
  registry.insert<cudm::CuDensityMatDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::ub::UBDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();

  // Conversion passes
  mlir::registerPass(pulse::createPulseToQOpPass);
  mlir::registerPass(qop::createQOpToCuDensityMatPass);
  mlir::registerPass(cudm::createCuDensityMatToLLVMPass);

  // Pulse Transforms passes
  mlir::registerPass(pulse::createPulseVerifyPass);
  mlir::registerPass(pulse::createPulseCanonicalizePass);
  mlir::registerPass(pulse::createVirtualZPass);
  mlir::registerPass(pulse::createPulseFusionPass);
  mlir::registerPass(pulse::createPulseScheduleAsapPass);
  mlir::registerPass(pulse::createPulseScheduleAlapPass);
  mlir::registerPass(pulse::createPulseInlineScheduleBlocksPass);
  mlir::registerPass(pulse::createPulseClassicalCheckPass);
  // Resource-constrained scheduling with the default machine model (4
  // drives, 2 readouts, no readout latency or switch penalty).
  mlir::registerPass(
      [] { return pulse::createPulseScheduleRcpPass(4, 2, 0, 0, false); });

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "cudaq-pulse MLIR optimizer\n", registry));
}
