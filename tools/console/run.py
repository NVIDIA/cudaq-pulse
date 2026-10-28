# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Start the console from a checkout: ``python tools/console/run.py``.

Uses whichever ``cudaq_pulse`` the interpreter imports (an installed wheel,
or a build tree on ``PYTHONPATH``).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nvqconsole.__main__ import main  # noqa: E402

main()
