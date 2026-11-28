// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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

#include "cudaq-pulse/Transforms/Passes.h"

#include "CudaqPulseOptVersion.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Block.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

/// The producer string of the bytecode --emit-bytecode writes: it states the
/// source of the Pulse and Classical dialects the bytecode is written with
/// (cmake/write_dialect_source.cmake), which pulse.load holds a bytecode
/// module to. Bytecode holds an operation's properties by position, in the
/// shape of the writer's dialects.
static constexpr char kBytecodeProducer[] =
    "cudaq-pulse-opt " CUDAQ_PULSE_VERSION " cudaq-pulse-dialects="
#include "cudaq-pulse/Dialect/DialectSource.inc"
    ;

/// Writes to `outputFilename` what MLIR's driver wrote to `written`, with
/// this driver's producer string where that is bytecode. The driver writes
/// bytecode under MLIR's own producer string and takes no other, and a
/// string of another length moves every section after it, which a section
/// that is aligned does not allow: the module is read back as it was
/// written, by the dialects that wrote it, and written again in the same
/// version of the format. Anything that is not bytecode is copied.
static mlir::LogicalResult restate(llvm::StringRef written,
                                   llvm::StringRef outputFilename,
                                   const mlir::DialectRegistry &registry,
                                   bool allowUnregisteredDialects,
                                   bool elideResourceData) {
  std::string error;
  auto buffer = mlir::openInputFile(written, &error);
  auto output = buffer ? mlir::openOutputFile(outputFilename, &error) : nullptr;
  if (!output) {
    llvm::errs() << error << "\n";
    return mlir::failure();
  }
  if (!mlir::isBytecode(*buffer)) {
    output->os() << buffer->getBuffer();
    output->keep();
    return mlir::success();
  }
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.allowUnregisteredDialects(allowUnregisteredDialects);
  // The module was verified before it was written.
  mlir::FallbackAsmResourceMap resources;
  mlir::ParserConfig reading(&context, /*verifyAfterParse=*/false, &resources);
  mlir::Block block;
  if (mlir::failed(
          mlir::parseSourceString(buffer->getBuffer(), &block, reading)) ||
      block.empty() || std::next(block.begin()) != block.end()) {
    llvm::errs() << "cudaq-pulse-opt: cannot read back the bytecode to state "
                    "its dialect source\n";
    return mlir::failure();
  }
  mlir::BytecodeWriterConfig writing(resources, kBytecodeProducer);
  writing.setElideResourceDataFlag(elideResourceData);
  // The format's version follows the four magic bytes; up to 127 it is one
  // byte, the version twice plus one.
  unsigned char version = buffer->getBuffer()[4];
  if (version & 1)
    writing.setDesiredBytecodeVersion(version >> 1);
  if (mlir::failed(
          mlir::writeBytecodeToFile(&block.front(), output->os(), writing))) {
    llvm::errs() << "cudaq-pulse-opt: cannot write the bytecode\n";
    return mlir::failure();
  }
  output->keep();
  return mlir::success();
}

int main(int argc, char **argv) {
  llvm::cl::SetVersionPrinter([](llvm::raw_ostream &os) {
    os << "cudaq-pulse-opt " CUDAQ_PULSE_VERSION
          " (LLVM " CUDAQ_PULSE_LLVM_VERSION ")\n";
  });

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

  pulse::registerPulsePasses();

  auto [input, output] = mlir::registerAndParseCLIOptions(
      argc, argv, "cudaq-pulse MLIR optimizer\n", registry);
  // Read here: the driver's own run of the options ends with them gone.
  mlir::MlirOptMainConfig config =
      mlir::MlirOptMainConfig::createFromCLOptions();
  // Output that is not one bytecode module is the driver's as it is: text,
  // a list of the dialects or of the passes, and the chunks of a split
  // input, which are written one after another with the marker between.
  if (!config.shouldEmitBytecode() || config.shouldShowDialects() ||
      config.shouldListPasses() || !config.inputSplitMarker().empty())
    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, input, output, registry));

  // Bytecode: the driver writes to a file of its own, and what it wrote is
  // written to the output with the producer string that states the dialect
  // source.
  llvm::SmallString<128> written;
  if (std::error_code failed = llvm::sys::fs::createTemporaryFile(
          "cudaq-pulse-opt", "mlirbc", written)) {
    llvm::errs() << "cudaq-pulse-opt: cannot make a temporary file: "
                 << failed.message() << "\n";
    return EXIT_FAILURE;
  }
  llvm::FileRemover remover(written);
  if (mlir::failed(mlir::MlirOptMain(argc, argv, input, written, registry)))
    return EXIT_FAILURE;
  // The driver kept the file, which ends its removal on a signal.
  llvm::sys::RemoveFileOnSignal(written);
  return mlir::asMainReturnCode(restate(
      written, output, registry, config.shouldAllowUnregisteredDialects(),
      config.shouldElideResourceDataFromBytecode()));
}
