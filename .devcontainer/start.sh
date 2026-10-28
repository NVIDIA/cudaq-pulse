#!/usr/bin/env bash
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Starts the console in the background on port 8765; its log is
# /tmp/nvqconsole.log. The bundle starts from its precomputed runs.

set -euo pipefail

if [[ -f "$HOME/nvqconsole-bundle/run.py" ]]; then
  console="$HOME/nvqconsole-bundle/run.py"
else
  console="$PWD/tools/console/run.py"
fi
setsid nohup python "$console" --no-open > /tmp/nvqconsole.log 2>&1 < /dev/null &
echo "NVQConsole starting on port 8765 (log: /tmp/nvqconsole.log)"
