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

mypy and ruff
-------------

mypy and ruff take no names from an installed package, so they report the
bare names as undefined until they are told about them, once per project.
In the directory that holds your configuration, run:

.. code-block:: bash

   python -m cudaq_pulse.doctor --setup-editor

It finds the configuration each tool reads (``mypy.ini``, ``.mypy.ini``,
``[tool.mypy]`` in ``pyproject.toml`` or ``[mypy]`` in ``setup.cfg`` for
mypy; ``.ruff.toml``, ``ruff.toml`` or ``[tool.ruff]`` in ``pyproject.toml``
for ruff), shows the change as a diff, and writes it only when you confirm,
or with ``--yes``. A tool without a configuration is left alone, and a
second run reports that both are set up. ``--project DIR`` names another
directory. The change is:

- **mypy:** ``plugins = ["cudaq_pulse.mypy_plugin"]``. For each file that
  imports ``cudaq_pulse``, the plugin makes the operations the file does not
  define itself known to mypy, with their types: misspelled names and wrong
  arguments are still errors, and files that do not import cudaq-pulse are
  unchanged.
- **ruff:** ``builtins = [...]`` with the 23 operations, which ruff then
  treats as defined everywhere in the project.

Nothing is configured when cudaq-pulse is imported or installed.

Other tools
-----------

Jedi (used by some editors for completion) does not read
``__builtins__.pyi``: it reports nothing about the bare names, and offers
no completion for them.
