# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Builds PulseExamplePlugin, a pass plugin for cudaq-pulse-opt.
#
# The cudaq-pulse build includes this directory when tests are enabled. To
# build it on its own, point CMake at the MLIR toolchain cudaq-pulse uses:
#
#   cmake -S examples/custom_passes/cpp -B build/plugin -G Ninja \
#     -DMLIR_DIR=<toolchain>/lib/cmake/mlir
#   cmake --build build/plugin

cmake_minimum_required(VERSION 3.30)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(pulse-example-plugin LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  find_package(MLIR REQUIRED CONFIG)
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
  include(AddLLVM)
  include(AddMLIR)
  include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
endif()

# The plugin resolves MLIR symbols from the executable that loads it, so it
# links no MLIR libraries itself.
add_library(PulseExamplePlugin MODULE DriveCount.cpp)
llvm_update_compile_flags(PulseExamplePlugin)
set_target_properties(PulseExamplePlugin PROPERTIES
  PREFIX ""
  SUFFIX "${LLVM_PLUGIN_EXT}"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
if(APPLE)
  target_link_options(PulseExamplePlugin PRIVATE -undefined dynamic_lookup)
endif()
