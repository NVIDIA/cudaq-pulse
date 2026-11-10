# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

# Stage python sources into the build tree, one symlink rule per file.
#
# Modeled on MLIR's add_mlir_python_sources_target minus its install and export
# machinery; these sources are installed through other mechanisms.
function(cudaq_pulse_stage_python_sources name)
    cmake_parse_arguments(ARG "" "ROOT_DIR;OUTPUT_DIRECTORY" "SOURCES" ${ARGN})
    if(ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Unhandled arguments to cudaq_pulse_stage_python_sources(${name}): ${ARG_UNPARSED_ARGUMENTS}")
    endif()

    set(_dest_paths "")
    foreach(_rel_path ${ARG_SOURCES})
        set(_src_path "${ARG_ROOT_DIR}/${_rel_path}")
        set(_dest_path "${ARG_OUTPUT_DIRECTORY}/${_rel_path}")
        get_filename_component(_dest_dir "${_dest_path}" DIRECTORY)
        file(MAKE_DIRECTORY "${_dest_dir}")
        add_custom_command(
            OUTPUT "${_dest_path}"
            COMMENT "Staging python source ${_rel_path}"
            DEPENDS "${_src_path}"
            COMMAND "${CMAKE_COMMAND}" -E create_symlink
                "${_src_path}" "${_dest_path}"
        )
        list(APPEND _dest_paths "${_dest_path}")
    endforeach()

    add_custom_target(${name} DEPENDS ${_dest_paths})
endfunction()
