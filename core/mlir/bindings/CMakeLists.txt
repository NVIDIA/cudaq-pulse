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

# Export only PyInit__cudaq_pulse_native. Exporting the embedded LLVM/MLIR
# (about 60k symbols, many weak) made dyld bind and coalesce each of them
# at import, which took over a second on macOS, and let them interpose with
# any other LLVM in the process. HandleLLVMOptions links every module with a
# flat namespace and dynamic lookup of undefined symbols, for plugins; the
# extension uses the two-level namespace, and nanobind's linker response file
# leaves only the Python C API to resolve at load time.
string(REPLACE "-Wl,-flat_namespace -Wl,-undefined -Wl,dynamic_lookup" ""
  CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS}")
if(APPLE)
  target_link_options(_cudaq_pulse_native PRIVATE
    "LINKER:-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/exports.darwin.txt")
else()
  target_link_options(_cudaq_pulse_native PRIVATE
    "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.linux.map"
    "LINKER:--exclude-libs,ALL")
endif()
set_property(TARGET _cudaq_pulse_native APPEND PROPERTY LINK_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/exports.darwin.txt"
  "${CMAKE_CURRENT_SOURCE_DIR}/exports.linux.map")

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
