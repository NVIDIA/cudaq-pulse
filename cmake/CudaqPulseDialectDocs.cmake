# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Script mode (cmake -P): copy the dialect reference pages mlir-tblgen wrote
# under GENERATED_DIR into OUTPUT_DIR as the Sphinx sources
# architecture/dialects/{pulse,qop,cudm}.md. The [TOC] marker is a Hugo
# feature that Sphinx would print as text; the page's own toctree replaces it.

foreach(_variable GENERATED_DIR OUTPUT_DIR)
  if(NOT DEFINED ${_variable})
    message(FATAL_ERROR "CudaqPulseDialectDocs.cmake needs -D${_variable}=...")
  endif()
endforeach()

file(MAKE_DIRECTORY "${OUTPUT_DIR}")
foreach(_pair IN ITEMS "Pulse:pulse" "QOp:qop" "CuDensityMat:cudm")
  string(REPLACE ":" ";" _pair "${_pair}")
  list(GET _pair 0 _dialect)
  list(GET _pair 1 _page)
  set(_input "${GENERATED_DIR}/${_dialect}/${_dialect}Dialect.md")
  if(NOT EXISTS "${_input}")
    message(FATAL_ERROR "missing generated dialect reference ${_input}")
  endif()
  file(READ "${_input}" _text)
  string(REPLACE "[TOC]\n" "" _text "${_text}")
  file(WRITE "${OUTPUT_DIR}/${_page}.md" "${_text}")
endforeach()
