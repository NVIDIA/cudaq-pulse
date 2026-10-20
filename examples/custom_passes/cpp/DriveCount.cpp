/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// A minimal out-of-tree pass plugin for cudaq-pulse-opt.
//
// `pulse-example-drive-count` annotates every function with the number of
// drives it plays and their total scheduled duration:
//
//   cudaq-pulse-opt --load-pass-plugin=PulseExamplePlugin.so \
//       --pass-pipeline='builtin.module(pulse-example-drive-count)' in.mlir
//
// The pass matches ops by name, so it builds against MLIR alone; a plugin can
// equally include the generated Pulse dialect headers and use the op classes.

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

#include "llvm/Config/llvm-config.h"

namespace {

struct DriveCountPass
    : public mlir::PassWrapper<DriveCountPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DriveCountPass)

  llvm::StringRef getArgument() const override {
    return "pulse-example-drive-count";
  }
  llvm::StringRef getDescription() const override {
    return "Annotate each function with its drive count and drive time";
  }

  void runOnOperation() override {
    mlir::Builder builder(&getContext());
    for (mlir::Operation &function : getOperation().getOps()) {
      if (function.getName().getStringRef() != "func.func")
        continue;
      int64_t drives = 0;
      int64_t driveVtu = 0;
      function.walk([&](mlir::Operation *op) {
        if (op->getName().getStringRef() != "pulse.drive")
          return;
        ++drives;
        if (auto duration =
                op->getAttrOfType<mlir::IntegerAttr>("duration_vtu"))
          driveVtu += duration.getInt();
      });
      function.setAttr("pulse_example.drives",
                       builder.getI64IntegerAttr(drives));
      function.setAttr("pulse_example.drive_vtu",
                       builder.getI64IntegerAttr(driveVtu));
    }
  }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "PulseExamplePlugin", LLVM_VERSION_STRING,
          [] { mlir::PassRegistration<DriveCountPass>(); }};
}
