# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

include_guard(DIRECTORY)

option(CUDAQ_PULSE_BUILD_TESTS "Build the cudaq-pulse test suites" ON)
if(NOT CUDAQ_PULSE_BUILD_TESTS)
  return()
endif()

enable_testing()

set(CUDAQ_PULSE_TOOLS_DIR "${LLVM_RUNTIME_OUTPUT_INTDIR}")

# The pass-plugin example is built and loaded by the lit suite.
add_subdirectory(${CUDAQ_PULSE_SOURCE_DIR}/examples/custom_passes/cpp
                 ${CUDAQ_PULSE_BINARY_DIR}/examples/custom_passes)
set(CUDAQ_PULSE_EXAMPLE_PLUGIN
    "${CUDAQ_PULSE_BINARY_DIR}/lib/PulseExamplePlugin${LLVM_PLUGIN_EXT}")
configure_file(
  ${CUDAQ_PULSE_SOURCE_DIR}/test/lit.site.cfg.py.in
  ${CUDAQ_PULSE_BINARY_DIR}/test/lit.site.cfg.py
  @ONLY)

# lit ships only as a console script (there is no runnable `lit.__main__`).
# Resolve it from the configured interpreter's own environment and run it
# through that interpreter so the launcher matches the installed packages.
get_filename_component(_cudaq_pulse_python_bindir "${Python_EXECUTABLE}" DIRECTORY)
find_program(CUDAQ_PULSE_LIT_EXECUTABLE NAMES lit
  HINTS "${_cudaq_pulse_python_bindir}"
  NO_DEFAULT_PATH)
if(NOT CUDAQ_PULSE_LIT_EXECUTABLE)
  message(FATAL_ERROR
    "lit was not found next to ${Python_EXECUTABLE}. Install the test "
    "dependencies with `uv sync` or `pip install lit`, or configure with "
    "-DCUDAQ_PULSE_BUILD_TESTS=OFF.")
endif()

set(_cudaq_pulse_lit_command
  ${Python_EXECUTABLE} ${CUDAQ_PULSE_LIT_EXECUTABLE} -sv
  ${CUDAQ_PULSE_BINARY_DIR}/test)
set(_cudaq_pulse_pytest_command
  ${CMAKE_COMMAND} -E env "PYTHONPATH=${CUDAQ_PULSE_PYTHONPATH}"
  ${Python_EXECUTABLE} -m pytest ${CUDAQ_PULSE_SOURCE_DIR}/tests -q)

add_custom_target(check-pulse-mlir
  COMMAND ${_cudaq_pulse_lit_command}
  DEPENDS cudaq-pulse-opt PulseExamplePlugin
  COMMENT "Running cudaq-pulse MLIR tests"
  USES_TERMINAL)

add_custom_target(check-pulse-python
  COMMAND ${_cudaq_pulse_pytest_command} -m "not gpu"
  DEPENDS pulse
  COMMENT "Running cudaq-pulse Python tests"
  USES_TERMINAL)

add_custom_target(check-pulse
  DEPENDS check-pulse-mlir check-pulse-python)

add_test(NAME PulseMLIRTests COMMAND ${_cudaq_pulse_lit_command})
set_tests_properties(PulseMLIRTests PROPERTIES LABELS "pulse;mlir")

add_test(NAME PulsePythonUnitTests
  COMMAND ${_cudaq_pulse_pytest_command} -m "not gpu")
set_tests_properties(PulsePythonUnitTests PROPERTIES LABELS "pulse;unit")

if(NOT CUDAQ_PULSE_ENABLE_GPU)
  return()
endif()

# These private drivers are not installed. Without --gpu they verify
# cuDensityMat linkage and version discovery; with --gpu they require a usable
# device and exercise the basic cuDensityMat descriptors used by the runtime.
add_custom_target(check-pulse-cudm-link)
add_dependencies(check-pulse check-pulse-cudm-link)
set(_cudaq_pulse_smoke_targets "")
foreach(_major IN LISTS CUDAQ_PULSE_CUDA_MAJORS)
  set(_smoke cudaq-pulse-cudm-smoke-cu${_major})
  add_executable(${_smoke}
    ${CUDAQ_PULSE_SOURCE_DIR}/tests/runtime/cudm_runtime_smoke.cpp)
  target_link_libraries(${_smoke} PRIVATE
    cudm_runtime_cu${_major}
    cudaq_pulse_cuda${_major})
  set_target_properties(${_smoke} PROPERTIES
    BUILD_RPATH "$<TARGET_FILE_DIR:cudm_runtime_cu${_major}>;${CUDAQ_PULSE_CUDA${_major}_LIBRARY_DIRS}")
  list(APPEND _cudaq_pulse_smoke_targets ${_smoke})

  add_test(NAME PulseCuDensityMatLinkSmokeCu${_major} COMMAND ${_smoke})
  set_tests_properties(PulseCuDensityMatLinkSmokeCu${_major} PROPERTIES
    LABELS "pulse;gpu-build")
  add_custom_target(check-pulse-cudm-link-cu${_major}
    COMMAND ${_smoke}
    DEPENDS ${_smoke}
    COMMENT "Checking cuDensityMat linkage for CUDA ${_major}"
    USES_TERMINAL)
  add_dependencies(check-pulse-cudm-link check-pulse-cudm-link-cu${_major})

  add_test(NAME PulseCuDensityMatGpuSmokeCu${_major} COMMAND ${_smoke} --gpu)
  set_tests_properties(PulseCuDensityMatGpuSmokeCu${_major} PROPERTIES
    LABELS "pulse;gpu")
endforeach()

# The Python GPU suite runs against whichever runtime the loader selects from
# the installed [cuNN] extra.
add_test(NAME PulsePythonGpuTests COMMAND ${_cudaq_pulse_pytest_command} -m gpu)
set_tests_properties(PulsePythonGpuTests PROPERTIES LABELS "pulse;gpu")

add_custom_target(check-pulse-gpu
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L "^gpu$"
  DEPENDS ${_cudaq_pulse_smoke_targets} pulse
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Running cudaq-pulse cuDensityMat and numerical GPU tests"
  USES_TERMINAL)
