# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# cudaq-pulse compiles against a prebuilt MLIR toolchain fetched by
# tools/fetch-mlir-toolchain.py and links MLIR statically into its extension:
#
#   python tools/fetch-mlir-toolchain.py
#   cmake -S . -B build -G Ninja
#
# Configuring runs the fetch itself when the toolchain is not unpacked yet.
#
# See CONTRIBUTING.md for the full build, test, and GPU-runtime instructions.
cmake_minimum_required(VERSION 3.30 FATAL_ERROR)

# Set a default build type / install prefix if none was specified. Must be set
# before project().
set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel")
set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.cudaq_pulse" CACHE STRING
    "Install path prefix, prepended onto install directories")

# C is enabled because LLVM's HandleLLVMOptions probes the C compiler.
project(cudaq-pulse LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

include(GNUInstallDirs)

# The two wheels each build one of these: cudaq-pulse the Python package and
# cudaq-pulse-tools the cudaq-pulse-opt driver (packaging/cudaq-pulse-tools).
option(CUDAQ_PULSE_BUILD_PYTHON "Build the cudaq_pulse Python package and its native extension" ON)
option(CUDAQ_PULSE_BUILD_OPT "Build the cudaq-pulse-opt MLIR driver" ON)
# The perf gate names the native functions in its Callgrind dumps, which a
# stripped extension does not; keeping the symbols leaves its code the same.
option(CUDAQ_PULSE_KEEP_SYMBOLS "Keep the native extension's symbols (no strip)" OFF)
if(NOT CUDAQ_PULSE_BUILD_PYTHON AND NOT CUDAQ_PULSE_BUILD_OPT)
  message(FATAL_ERROR "CUDAQ_PULSE_BUILD_PYTHON and CUDAQ_PULSE_BUILD_OPT are both OFF")
endif()

if(CUDAQ_PULSE_BUILD_PYTHON)
  find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
else()
  find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
endif()

set(CUDAQ_PULSE_LLVM_VERSION "22.1.8")

# tools/fetch-mlir-toolchain.py unpacks the pinned, checksum-verified
# toolchain here. An explicit MLIR_DIR or CMAKE_PREFIX_PATH takes precedence.
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
  set(_cudaq_pulse_host_os macos)
else()
  string(TOLOWER "${CMAKE_HOST_SYSTEM_NAME}" _cudaq_pulse_host_os)
endif()
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64)$")
  set(_cudaq_pulse_host_arch aarch64)
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  set(_cudaq_pulse_host_arch x86_64)
else()
  string(TOLOWER "${CMAKE_HOST_SYSTEM_PROCESSOR}" _cudaq_pulse_host_arch)
endif()
set(_cudaq_pulse_host_platform "${_cudaq_pulse_host_os}-${_cudaq_pulse_host_arch}")
set(CUDAQ_PULSE_FETCHED_MLIR_PREFIX
  "${CMAKE_CURRENT_SOURCE_DIR}/.cache/mlir/mlir-toolchain-${CUDAQ_PULSE_LLVM_VERSION}-${_cudaq_pulse_host_platform}")
set(_cudaq_pulse_fetch_script "${CMAKE_CURRENT_SOURCE_DIR}/tools/fetch-mlir-toolchain.py")

set(_cudaq_pulse_user_mlir_dir "")
if(MLIR_DIR AND NOT MLIR_DIR MATCHES "-NOTFOUND$")
  set(_cudaq_pulse_user_mlir_dir "${MLIR_DIR}")
endif()
if(_cudaq_pulse_user_mlir_dir)
  find_package(MLIR CONFIG QUIET PATHS "${_cudaq_pulse_user_mlir_dir}" NO_DEFAULT_PATH)
  if(NOT MLIR_FOUND)
    message(FATAL_ERROR
      "MLIR_DIR=${_cudaq_pulse_user_mlir_dir} does not contain MLIRConfig.cmake "
      "for the MLIR ${CUDAQ_PULSE_LLVM_VERSION} toolchain.")
  endif()
else()
  # Only the places a user names (MLIR_ROOT, CMAKE_PREFIX_PATH) and the
  # fetched toolchain: an MLIR found on PATH, in the system prefixes or in a
  # package registry is not the pinned toolchain, and would be built against.
  find_package(MLIR CONFIG QUIET
    HINTS "${CUDAQ_PULSE_FETCHED_MLIR_PREFIX}"
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_INSTALL_PREFIX
    NO_CMAKE_PACKAGE_REGISTRY
    NO_CMAKE_SYSTEM_PACKAGE_REGISTRY)
endif()
if(NOT MLIR_FOUND)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${_cudaq_pulse_fetch_script}" --list-platforms
    OUTPUT_VARIABLE _cudaq_pulse_toolchain_platforms
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  string(REPLACE "\n" ";" _cudaq_pulse_toolchain_platforms "${_cudaq_pulse_toolchain_platforms}")
  if(NOT _cudaq_pulse_host_platform IN_LIST _cudaq_pulse_toolchain_platforms)
    list(JOIN _cudaq_pulse_toolchain_platforms ", " _cudaq_pulse_toolchain_platforms)
    message(FATAL_ERROR
      "cudaq-pulse requires the MLIR ${CUDAQ_PULSE_LLVM_VERSION} toolchain, and "
      "no prebuilt toolchain exists for ${_cudaq_pulse_host_platform} (only "
      "${_cudaq_pulse_toolchain_platforms}). Build one with "
      "tools/build-mlir-toolchain.sh and configure with "
      "-DMLIR_DIR=<prefix>/lib/cmake/mlir.")
  endif()
  message(STATUS "Fetching the MLIR ${CUDAQ_PULSE_LLVM_VERSION} toolchain for ${_cudaq_pulse_host_platform}")
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${_cudaq_pulse_fetch_script}"
    OUTPUT_VARIABLE _cudaq_pulse_fetched_prefix
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE _cudaq_pulse_fetch_error
    ERROR_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _cudaq_pulse_fetch_status)
  if(NOT _cudaq_pulse_fetch_status EQUAL 0)
    message(FATAL_ERROR
      "cudaq-pulse requires the MLIR ${CUDAQ_PULSE_LLVM_VERSION} toolchain, "
      "which was not found on MLIR_DIR or CMAKE_PREFIX_PATH, and fetching it "
      "failed:\n${_cudaq_pulse_fetch_error}\n"
      "Alternatively, configure with -DMLIR_DIR=<prefix>/lib/cmake/mlir.")
  endif()
  unset(MLIR_DIR CACHE)
  find_package(MLIR CONFIG REQUIRED
    PATHS "${_cudaq_pulse_fetched_prefix}" NO_DEFAULT_PATH)
endif()
if(NOT LLVM_PACKAGE_VERSION VERSION_EQUAL CUDAQ_PULSE_LLVM_VERSION)
  message(FATAL_ERROR
    "cudaq-pulse requires LLVM/MLIR ${CUDAQ_PULSE_LLVM_VERSION}, but the "
    "toolchain at ${MLIR_DIR} is LLVM ${LLVM_PACKAGE_VERSION}. To use the "
    "pinned toolchain, which configuring fetches, configure a fresh build "
    "directory with neither MLIR_DIR nor CMAKE_PREFIX_PATH naming this one; "
    "or set MLIR_DIR to an MLIR ${CUDAQ_PULSE_LLVM_VERSION} build.")
endif()
if(LLVM_ENABLE_RTTI)
  message(FATAL_ERROR
    "The MLIR toolchain at ${MLIR_DIR} was built with RTTI enabled; "
    "cudaq-pulse requires an LLVM_ENABLE_RTTI=OFF toolchain built by "
    "tools/build-mlir-toolchain.sh. To use the pinned toolchain, which "
    "configuring fetches, configure a fresh build directory with neither "
    "MLIR_DIR nor CMAKE_PREFIX_PATH naming this one.")
endif()
message(STATUS "Using MLIR from ${MLIR_DIR} (LLVM ${LLVM_PACKAGE_VERSION})")

list(APPEND CMAKE_MODULE_PATH
     "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}"
     "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# LLVM's HandleLLVMOptions relies on the pre-CMake-4.3 backslash handling.
if(POLICY CMP0219)
  cmake_policy(SET CMP0219 OLD)
endif()
include(HandleLLVMOptions)
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(CudaqPulseSanitizers)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})

# add_llvm_executable / add_mlir_library place their artifacts here.
set(LLVM_RUNTIME_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/bin")
set(LLVM_LIBRARY_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/lib")

# Keep a single relocatable library tree on distributions where
# GNUInstallDirs would otherwise default to lib64.
set(CMAKE_INSTALL_LIBDIR lib)

set(CUDAQ_PULSE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(CUDAQ_PULSE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(CUDAQ_PULSE_FRONTEND_DIR
    "${CUDAQ_PULSE_SOURCE_DIR}/core/frontend/cudaq_pulse")

include_directories(
  ${CUDAQ_PULSE_SOURCE_DIR}/core/mlir/include
  ${CUDAQ_PULSE_BINARY_DIR}/core/mlir/include)

# Generate the three research-preview dialects.
add_subdirectory(core/mlir/include/cudaq-pulse)

include(CudaqPulseVersion)

# Build the dialects, transforms, conversions, and optimizer driver.
add_subdirectory(core/mlir)
if(CUDAQ_PULSE_BUILD_OPT)
  add_subdirectory(core/mlir/tools/cudaq-pulse-opt)
endif()

if(CUDAQ_PULSE_BUILD_PYTHON)
  # Resolve the optional GPU runtime plus the Python and nanobind dependencies.
  include(CudaqPulseDependencies)
  add_subdirectory(core/mlir/bindings)

  # Stage the Python package and register the aggregate build/install target.
  include(CudaqPulsePython)
else()
  add_custom_target(pulse DEPENDS cudaq-pulse-opt)
endif()

# Keep test and documentation target definitions out of the package entrypoint.
include(CudaqPulseTesting)
include(CudaqPulseDocumentation)
