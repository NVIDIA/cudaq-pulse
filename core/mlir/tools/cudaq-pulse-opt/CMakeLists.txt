# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LLVM_LINK_COMPONENTS Support)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(cudaq-pulse-opt
  cudaq-pulse-opt.cpp
)

target_link_libraries(cudaq-pulse-opt PRIVATE
  CudaqPulsePulse
  CudaqPulsePulseTransforms
  CudaqPulseQOp
  CudaqPulseCuDensityMat
  CudaqPulsePulseToQOp
  CudaqPulseQOpToCuDensityMat
  CudaqPulseCuDensityMatToLLVM
  MLIRFuncDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRUBDialect
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRLLVMDialect
  MLIRReconcileUnrealizedCasts
  MLIRRegisterAllPasses
  MLIROptLib
  MLIRPass
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRTransformUtils
)

llvm_update_compile_flags(cudaq-pulse-opt)
# Pass plugins (--load-pass-plugin) resolve MLIR symbols from this executable.
export_executable_symbols_for_plugins(cudaq-pulse-opt)
mlir_check_all_link_libraries(cudaq-pulse-opt)

if(SKBUILD)
  set(_cudaq_pulse_opt_destination "${SKBUILD_SCRIPTS_DIR}")
else()
  set(_cudaq_pulse_opt_destination "${CMAKE_INSTALL_BINDIR}")
endif()
install(
  TARGETS cudaq-pulse-opt
  RUNTIME DESTINATION "${_cudaq_pulse_opt_destination}"
  COMPONENT CudaqPulse
)
