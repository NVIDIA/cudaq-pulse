// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A minimal out-of-tree pass plugin for cudaq-pulse-opt.
//
// `pulse-example-drive-count` annotates every function with the number of
// drives it plays and their total scheduled duration:
//
//   cudaq-pulse-opt --load-pass-plugin=PulseExamplePlugin.so \
//       --pass-pipeline='builtin.module(pulse-example-drive-count)' in.mlir
//
// The pass matches ops by name and reads their properties as attributes, so it
// builds against MLIR alone; a plugin can equally include the generated Pulse
// dialect headers and use the op classes.

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
      int64_t driveCycles = 0;
      function.walk([&](mlir::Operation *op) {
        if (op->getName().getStringRef() != "pulse.drive")
          return;
        ++drives;
        // The schedule is in the op's properties, read here in their
        // attribute form.
        auto properties = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(
            op->getPropertiesAsAttribute());
        if (auto duration = properties ? properties.getAs<mlir::IntegerAttr>(
                                             "duration_cycles")
                                       : mlir::IntegerAttr())
          driveCycles += duration.getInt();
      });
      function.setAttr("pulse_example.drives",
                       builder.getI64IntegerAttr(drives));
      function.setAttr("pulse_example.drive_cycles",
                       builder.getI64IntegerAttr(driveCycles));
    }
  }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "PulseExamplePlugin", LLVM_VERSION_STRING,
          [] { mlir::PassRegistration<DriveCountPass>(); }};
}
