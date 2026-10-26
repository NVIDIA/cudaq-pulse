..
   Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.

   This source code and the accompanying materials are made available under
   the terms of the Apache License 2.0 which accompanies this distribution.

Editor Support
==============

Inside a ``@pulse.kernel`` the operations (``drive``, ``gaussian``,
``get_drive_line``, ...) are used by bare name, as extensions of the
language, with only ``import cudaq_pulse as pulse`` at the top of the file.
Editors understand them with no configuration.

What works by default
---------------------

The cudaq-pulse wheel installs ``__builtins__.pyi`` beside the package, at
the root of ``site-packages``. Pyright reads it as additions to the
builtins of your files, so in these editors and checkers the bare names are
defined and typed, with completion, hover documentation and checked
arguments, and nothing to set up:

- Pylance (VS Code), in its default mode and every other mode
- Cursor's Python language server
- pyright
- basedpyright

A misspelled operation is still reported as undefined, and an argument of
the wrong kind (a tone where a line belongs) is reported in pyright's
``basic`` mode and above. Calls of a kernel are checked against its
signature: ``k(q)`` for a kernel of ``(qubit, amplitude: float)`` reports
the missing argument, through ``k.lower(...)`` and ``pulse.compile(k)(...)``
as well.

Two consequences of how the names are provided:

- **Outside kernels too.** The names are builtins for every file of the
  environment, not only inside kernels, so a call of ``drive(...)``
  outside a kernel is not reported; it raises ``RuntimeError`` when it
  runs. A name you define yourself, such as a function called ``square``,
  takes precedence as usual.
- **One owner per environment.** Only one installed distribution can
  provide ``__builtins__.pyi``. If another package also installs one, the
  last one installed wins, and uninstalling either removes it. A
  ``__builtins__.pyi`` in your own project takes precedence over the
  installed one; add ``from cudaq_pulse.ops import *`` to it to keep the
  operations.

Other tools
-----------

Jedi (used by some editors for completion) does not read
``__builtins__.pyi``: it reports nothing about the bare names, and offers
no completion for them.
