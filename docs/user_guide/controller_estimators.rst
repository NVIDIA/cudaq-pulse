..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Running estimators on the controller
====================================

A calibration estimator that runs where the qudits are answers in about a
microsecond instead of the milliseconds of a host round trip. In
cudaq-pulse it is a classical function on a link a controller's
``Processor`` serves: compiling costs its body on the processor
(:ref:`what it costs the controller <controller-costs>`), the simulator runs
it, and the export gives a backend its body as a program to run
(:ref:`classical functions <controller-functions>` in the backends guide).
This page puts the three together.

Declaring one
-------------

An estimator takes its state in and gives it back, so its body is a pure
function of its arguments, and it keeps to what a processor runs with
``fixed_cost_only=True``: numbers and fixed-size tuples, ``if`` and
conditional expressions, loops over ``range`` of a constant or a list
argument, calls of plain helper functions, and ``exp``, ``log``,
``sqrt``, ``sin``, ``cos``, ``atan2``, ``**``, ``abs``, ``floor``,
``ceil``, ``min`` and ``max``. The estimators of
``cudaq_pulse.classical.calibration`` written for a processor
(``adaptive_t1_update``, ``binary_search_update``, ``decay_time``,
``sparse_phase`` and the others of its last section) are such helpers:

.. code-block:: python
   :class: fragment

   import math

   import cudaq_pulse as pulse
   from cudaq_pulse.classical import Link
   from cudaq_pulse.classical import calibration as cal

   @pulse.classical(max_latency_ns=2000, link="controller", number_format="fixed:4.28")
   def tracker(bit: pulse.Bits, p: float, wait: float) -> tuple[float, float]:
       """Survival probability after *wait*, smoothed, and the T1 it gives."""
       p = p + 0.02 * (bit[0] - p)
       return p, -wait / math.log(p)

   processor = pulse.Processor(links=["controller"], fixed_cost_only=True)
   system = pulse.ControlSystem(
       controllers=[pulse.Controller(processors=[processor])],
       links={"controller": Link(fixed_ns=4)},
   )

``examples/21_calibration_protocol.py`` runs a whole calibration chain
this way. A function whose body does more (a ``while`` loop, a list, an
exception, ``math.tan``) runs on a link no processor serves, where its
body runs on the host as before.

What the backend gets
---------------------

``scheduled.export_to_controller()`` lists, in ``functions``, each
function a processor computes, by the compiled name its calls give
(``tracker__1`` for one bit). A call keeps its ``submit`` and ``answer``
events; a backend that runs ``functions[name]`` itself computes the
answer from the call's operands and takes it at the answer event. The
body above becomes, in part:

.. code-block:: text

   "let": [
     {"name": "p", "value": {"op": "add", "type": "float", "of": [
       {"argument": 1},
       {"op": "mul", "type": "float", "of": [{"const": 0.02}, {"op": "sub", ...}]}]}}
   ],
   "results": [{"local": "p"}, {"op": "div", "type": "float", ...}]

``controller.evaluate_function(program, arguments)`` computes a program as
the controller must; a backend checks its own evaluation against it. In
``"float64"``, the default, it gives what the Python body gives to the
last bit.

The arithmetic
--------------

``number_format`` declares the arithmetic the processor computes the body
in, and the export and ``evaluate_function`` carry it:

``"float64"``
    IEEE 754 binary64, as the Python body.
``"float32"``
    Every float an argument, a constant or a term gives is rounded to
    binary32 (to nearest, ties to even; an overflow to ±inf). ``+ - * /``
    and ``sqrt`` are exact binary32 results; the other functions are
    computed in binary64 and rounded once.
``"fixed:<i>.<f>"``
    Signed fixed point of ``i + f`` bits (at most 53), ``i`` of them for
    the integer part and the sign: ``fixed:4.28`` holds -8 to 8 - 2\
    :sup:`-28` in steps of 2\ :sup:`-28`. ``+ - *``, ``/``, negation and
    ``abs`` are exact, then rounded to the step (ties to even) and
    saturated to the range; the other functions are computed in binary64,
    then rounded and saturated, an infinity to the end of the range and a
    NaN to 0. A division by zero saturates, and 0 / 0 is 0.

Integers and bools keep their 64-bit arithmetic in every format. A body
in a fixed format has to keep its numbers in range, as the tracker above
does by working in units of 10 us.

Simulating the controller's arithmetic
--------------------------------------

The simulator runs the Python body by default. ``pulse.simulate(...,
controller_arithmetic=True)`` answers each function a processor computes
with its exported program in its ``number_format`` instead, so the
rounding and saturation of a fixed-point estimator show up in the
simulated branches and answers as they would on the controller. A handler
given in ``handlers`` still answers its function.

.. code-block:: python
   :class: fragment

   on_controller = pulse.simulate(scheduled, controller_arithmetic=True)
   for branch in on_controller.branches:
       print(branch.answers("tracker"))
