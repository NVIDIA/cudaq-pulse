# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write ``__builtins__.pyi``: the kernel language as builtins for type checkers.

    python write_builtins_stub.py OPS_INIT_PY OUTPUT_PYI

Pyright, Pylance and basedpyright (and so VS Code-based editors) resolve a
module named ``__builtins__`` like any other import and add its names to the
builtins of the user's files, so the stub at the root of site-packages makes the bare
DSL names inside ``@pulse.kernel`` bodies defined and typed. It re-exports
exactly ``cudaq_pulse.ops.__all__``, read from the source without importing
the package.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

HEADER = '''\
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The cudaq-pulse kernel language, as builtins for type checkers.

Generated from cudaq_pulse.ops.__all__ when cudaq-pulse is built; do not edit.
Pyright, Pylance and basedpyright (and so VS Code-based editors) read it from
site-packages, so the operations used by bare name inside @pulse.kernel
functions are defined and typed with no configuration. Python never imports it.
"""

'''


def ops_names(ops_init: Path) -> list[str]:
    tree = ast.parse(ops_init.read_text(), str(ops_init))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            names = ast.literal_eval(node.value)
            if not names or not all(isinstance(n, str) for n in names):
                break
            return list(names)
    raise SystemExit(f"{ops_init}: no literal __all__ list of names")


def render(names: list[str]) -> str:
    return HEADER + "".join(f"from cudaq_pulse.ops import {n} as {n}\n" for n in names)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        raise SystemExit(__doc__)
    text = render(ops_names(Path(argv[1])))
    out = Path(argv[2])
    if not out.exists() or out.read_text() != text:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
