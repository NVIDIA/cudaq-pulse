# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# STABLE_ABI takes effect only on CPython >= 3.12 with Python::SABIModule,
# which is found when scikit-build-core builds a cp312-abi3 wheel.
if(SKBUILD_SABI_VERSION AND NOT TARGET Python::SABIModule)
  message(FATAL_ERROR
    "scikit-build-core is tagging this wheel abi3 (Python ${SKBUILD_SABI_VERSION}), "
    "but FindPython did not provide Development.SABIModule, so the extension "
    "would not be built against the stable ABI.")
endif()
nanobind_add_module(
  _cudaq_pulse_native
  NB_STATIC
  STABLE_ABI
  bindings.cpp
)

target_link_libraries(_cudaq_pulse_native PRIVATE
  CudaqPulsePulse
  CudaqPulsePulseTransforms
  CudaqPulseQOp
  CudaqPulseCuDensityMat
  CudaqPulsePulseToQOp
  CudaqPulseQOpToCuDensityMat
  CudaqPulseCuDensityMatToLLVM
  MLIRExecutionEngine
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRParser
  MLIRSCFDialect
  MLIRUBDialect
  MLIRTargetLLVMIRExport
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
)

set_target_properties(_cudaq_pulse_native PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/cudaq_pulse/_native"
)

set(_native_dir "${CMAKE_BINARY_DIR}/python/cudaq_pulse/_native")
nanobind_add_stub(
  _cudaq_pulse_native_stub
  MODULE _cudaq_pulse_native
  OUTPUT "${_native_dir}/_cudaq_pulse_native.pyi"
  PYTHON_PATH "${_native_dir}"
  DEPENDS _cudaq_pulse_native
)

install(
  TARGETS _cudaq_pulse_native
  LIBRARY DESTINATION cudaq_pulse/_native
  COMPONENT CudaqPulse
)
install(
  FILES "${_native_dir}/_cudaq_pulse_native.pyi"
  DESTINATION cudaq_pulse/_native
  COMPONENT CudaqPulse
)
