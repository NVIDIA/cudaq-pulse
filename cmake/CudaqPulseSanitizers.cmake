# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

# Instrument cudaq-pulse's own code (not the prebuilt MLIR toolchain) with
# the listed sanitizers: "address;undefined" or "thread". Include before any
# target is defined.
set(CUDAQ_PULSE_SANITIZERS "" CACHE STRING
  "Sanitizers for cudaq-pulse's code, e.g. address;undefined or thread")

if(NOT CUDAQ_PULSE_SANITIZERS)
  return()
endif()

if("thread" IN_LIST CUDAQ_PULSE_SANITIZERS AND
   ("address" IN_LIST CUDAQ_PULSE_SANITIZERS OR
    "memory" IN_LIST CUDAQ_PULSE_SANITIZERS))
  message(FATAL_ERROR
    "CUDAQ_PULSE_SANITIZERS: thread cannot be combined with address or memory")
endif()

list(JOIN CUDAQ_PULSE_SANITIZERS "," _cudaq_pulse_sanitizers)
add_compile_options(
  -fsanitize=${_cudaq_pulse_sanitizers}
  -fno-sanitize-recover=all
  -fno-omit-frame-pointer
  -g)
# The vptr check needs RTTI, which LLVM's flags turn off.
if("undefined" IN_LIST CUDAQ_PULSE_SANITIZERS)
  add_compile_options(-fno-sanitize=vptr)
endif()
add_link_options(-fsanitize=${_cudaq_pulse_sanitizers})
message(STATUS "cudaq-pulse sanitizers: ${_cudaq_pulse_sanitizers}")
