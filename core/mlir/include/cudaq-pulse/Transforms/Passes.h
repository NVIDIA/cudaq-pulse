/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// The cudaq-pulse passes and conversions, and their registration under the
// names cudaq-pulse-opt and the Python extension run them by.

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include <cstdint>
#include <memory>

namespace pulse {
std::unique_ptr<mlir::Pass> createPulseToQOpPass();

std::unique_ptr<mlir::Pass> createPulseVerifyPass();
std::unique_ptr<mlir::Pass> createPulseCanonicalizePass();
std::unique_ptr<mlir::Pass> createVirtualZPass();
std::unique_ptr<mlir::Pass> createPulseFusionPass();
std::unique_ptr<mlir::Pass> createPulseScheduleAsapPass();
std::unique_ptr<mlir::Pass> createPulseScheduleAlapPass();
std::unique_ptr<mlir::Pass> createPulseScheduleRcpPass(int64_t maxDrives,
                                                       int64_t maxReadouts,
                                                       int64_t readoutLatency,
                                                       int64_t switchPenalty,
                                                       bool isAlap);
std::unique_ptr<mlir::Pass> createPulseInlineScheduleBlocksPass();
std::unique_ptr<mlir::Pass> createPulseClassicalCheckPass();
std::unique_ptr<mlir::Pass> createPulseUnrollClassicalLoopsPass();
} // namespace pulse

namespace qop {
std::unique_ptr<mlir::Pass> createQOpToCuDensityMatPass();
} // namespace qop

namespace cudm {
std::unique_ptr<mlir::Pass> createCuDensityMatToLLVMPass();
} // namespace cudm

namespace pulse {
/// Register the conversions and the pulse passes. Registration is global and
/// not thread-safe: call it once, before any pipeline is parsed.
inline void registerPulsePasses() {
  mlir::registerPass(createPulseToQOpPass);
  mlir::registerPass(qop::createQOpToCuDensityMatPass);
  mlir::registerPass(cudm::createCuDensityMatToLLVMPass);

  mlir::registerPass(createPulseVerifyPass);
  mlir::registerPass(createPulseCanonicalizePass);
  mlir::registerPass(createVirtualZPass);
  mlir::registerPass(createPulseFusionPass);
  mlir::registerPass(createPulseScheduleAsapPass);
  mlir::registerPass(createPulseScheduleAlapPass);
  mlir::registerPass(createPulseInlineScheduleBlocksPass);
  mlir::registerPass(createPulseClassicalCheckPass);
  // Resource-constrained scheduling with the default machine model (4
  // drives, 2 readouts, no readout latency or switch penalty).
  mlir::registerPass(
      [] { return createPulseScheduleRcpPass(4, 2, 0, 0, false); });
}
} // namespace pulse
