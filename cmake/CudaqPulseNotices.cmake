# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

set(CUDAQ_PULSE_LINK_MAP_DIR "${CMAKE_BINARY_DIR}/link-maps")
file(MAKE_DIRECTORY "${CUDAQ_PULSE_LINK_MAP_DIR}")

# cudaq_pulse_third_party_notices(NOTICES <file> COMPONENT <component>
#                                 TARGETS <target>...)
#
# Each target writes a linker map to link-maps/<target>.map, and installing
# COMPONENT fails unless NOTICES is what tools/third-party-notices.py
# generates and lists every third-party component those maps show linked.
function(cudaq_pulse_third_party_notices)
  cmake_parse_arguments(PARSE_ARGV 0 ARG "" "NOTICES;COMPONENT" "TARGETS")
  set(_maps "")
  foreach(_target IN LISTS ARG_TARGETS)
    set(_map "${CUDAQ_PULSE_LINK_MAP_DIR}/${_target}.map")
    if(APPLE)
      target_link_options(${_target} PRIVATE "LINKER:-map,${_map}")
    else()
      target_link_options(${_target} PRIVATE "LINKER:-Map=${_map}")
    endif()
    string(APPEND _maps " --map \"${_map}\"")
  endforeach()
  install(CODE "
    execute_process(
      COMMAND \"${Python3_EXECUTABLE}\"
        \"${CUDAQ_PULSE_SOURCE_DIR}/tools/third-party-notices.py\" check
        --notices \"${ARG_NOTICES}\"${_maps}
      COMMAND_ERROR_IS_FATAL ANY)"
    COMPONENT ${ARG_COMPONENT})
endfunction()
