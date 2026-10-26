# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""mypy plugin: the kernel language for files that import cudaq_pulse.

Enabled in the mypy configuration, which
``python -m cudaq_pulse.doctor --setup-editor`` writes::

    [tool.mypy]
    plugins = ["cudaq_pulse.mypy_plugin"]

mypy reads builtins only from its own typeshed, so the ``__builtins__.pyi``
that pyright uses does not reach it. Instead, when mypy parses a file that
imports ``cudaq_pulse``, the plugin adds ``from cudaq_pulse.ops import ...``
after that import, for the operations the file does not bind itself: the
names used bare in kernels are then defined and typed, and a misspelled one
is still an error. Files that do not import cudaq_pulse are unchanged.
"""

from __future__ import annotations

from mypy.nodes import (
    AssignmentStmt,
    ClassDef,
    Decorator,
    FuncDef,
    Import,
    ImportAll,
    ImportFrom,
    MypyFile,
    NameExpr,
    OverloadedFuncDef,
    Statement,
    TupleExpr,
)
from mypy.plugin import Plugin

from . import ops

_OPS = tuple(ops.__all__)
_PACKAGE = "cudaq_pulse"
_OPS_MODULE = "cudaq_pulse.ops"
# mypy's priority for a module-level import (mypy.build.PRI_MED).
_PRIORITY = 10


def _imports_package(stmt: Statement) -> bool:
    def ours(module: str) -> bool:
        return module == _PACKAGE or module.startswith(_PACKAGE + ".")

    if isinstance(stmt, Import):
        return any(ours(module) for module, _ in stmt.ids)
    if isinstance(stmt, (ImportFrom, ImportAll)):
        return stmt.relative == 0 and ours(stmt.id)
    return False


def _bound(defs: list[Statement]) -> set[str]:
    """Names the module binds at its top level."""
    names: set[str] = set()

    def targets(expr: object) -> None:
        if isinstance(expr, NameExpr):
            names.add(expr.name)
        elif isinstance(expr, TupleExpr):
            for item in expr.items:
                targets(item)

    for stmt in defs:
        if isinstance(stmt, (FuncDef, OverloadedFuncDef, ClassDef)):
            names.add(stmt.name)
        elif isinstance(stmt, Decorator):
            names.add(stmt.func.name)
        elif isinstance(stmt, AssignmentStmt):
            for lvalue in stmt.lvalues:
                targets(lvalue)
        elif isinstance(stmt, Import):
            names.update((alias or module).split(".")[0] for module, alias in stmt.ids)
        elif isinstance(stmt, ImportFrom):
            names.update(alias or name for name, alias in stmt.names)
    return names


class CudaqPulsePlugin(Plugin):
    def get_additional_deps(self, file: MypyFile) -> list[tuple[int, str, int]]:
        if file.fullname == _PACKAGE or file.fullname.startswith(_PACKAGE + "."):
            return []
        anchor = next((i for i, s in enumerate(file.defs) if _imports_package(s)), None)
        if anchor is None:
            return []
        stmt = file.defs[anchor]
        if isinstance(stmt, ImportAll) and stmt.id == _OPS_MODULE:
            return []
        missing = [n for n in _OPS if n not in _bound(file.defs)]
        if missing:
            imp = ImportFrom(_OPS_MODULE, 0, [(n, None) for n in missing])
            imp.set_line(stmt)
            imp.is_top_level = True
            file.defs.insert(anchor + 1, imp)
            file.imports.append(imp)
        return [(_PRIORITY, _OPS_MODULE, -1)]


def plugin(version: str) -> type[Plugin]:
    return CudaqPulsePlugin


__all__ = ["CudaqPulsePlugin", "plugin"]
