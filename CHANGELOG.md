# Changelog

All notable changes to cudaq-pulse are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
the [versioning policy](CONTRIBUTING.md#versioning-and-stability): while the
major version is 0, a minor release may break the API.

Each GitHub release takes its notes from the section for its version, so
every change worth telling users about goes in the top section, the next
release's `## [X.Y.Z] - Unreleased`, in the pull request that makes it.

## [0.1.0] - Unreleased

The first public release, a research preview: APIs, dialects and numerical
behavior may change in any 0.x release.

### Added

- **Pickling, for process pools and caches on disk.** Kernels pickle as
  functions do: by reference, or with cloudpickle (Ray, Dask) by value
  when defined in `__main__`, at the prompt or inside a function. Targets, devices, qudit handles and registers
  pickle as what they were built from and unpickle equal, with equal
  hashes; a process builds each device it unpickles once. A device's
  fingerprint covers its calibration: a target with and without one are
  two devices. A shipped calibration pickles as its name, checked against
  the file shipped where it unpickles; any other, as its values. Programs
  (`ck.lower(...)`, bound or not) pickle as MLIR bytecode stamped with the
  build that compiled them (hashes of its native and Python sources, taken
  when it was built) and the Python, and run unpickled (export, simulate, bind)
  without compiling again; another build refuses them with
  `UnsupportedError`, saying to compile again. Errors pickle with what they
  report, so a process pool re-raises them in the parent as they were
  raised. Targets now compare by content.
- **Controllers with their own clocks.** Kernels count cycles of the
  device's program clock, `"clock_ghz"` in the device config (or the
  target's). A controller whose clock is slower starts and times operations
  only every few program cycles: the scheduler puts each line's operations
  on its controller's grid and a `sync` on the grid its lines share, an
  operation off the grid is a compile error naming the line, its controller
  and the nearest valid lengths, and `pulse.cycles_on(line, n)` rounds a
  length onto the grid (`pulse.Rounding`). Controller programs count their
  own cycles, with `"program_clock_ghz"` written when it differs;
  `MachineModel.reference_clock_ghz` is checked against every controller
  clock and exported; the simulator samples generated waveforms at the
  line's controller clock.
- **Runtime qudits.** A parameter annotated `Qudit` or `Register` (or
  `list[Qudit]`, `tuple[Qudit, ...]`) is bound at each call, like a float:
  the kernel compiles once and runs on any qudits of a device with the same
  structure, including a recalibrated one. An unannotated qudit parameter
  stays static, whatever the call passes. `list[float]` parameters are runtime vectors. Their
  controller exports carry a `program_id`, shared by every call of one
  compile, and `symbols`: each symbol's value in the call, and per event the
  symbols its qudit, frequency, phase and waveform depend on.
- **Control systems.** A device can hold the controllers that play it:
  `pulse.Controller` (a controller's limits, the drive and readout lines it
  plays, its peer latency, clock skew and alignment tolerance, its
  `supports`, `backend`, `model` and `version`) and `pulse.ControlSystem`
  (its controllers, its links by name, each `Link` with the `controllers`
  it reaches). `pulse.device({"quantum_system": ..., "control_system":
  ...})` checks the wiring against the quantum system and raises one
  `DeviceDefinitionError` listing every problem; `"target"` still names the
  quantum system. `compile(kernel, device=qpu)` checks every program against
  the system: a line on no controller, each controller's own limits, a
  capability (`pulse.Capability`) a controller lacks, and a call over a link
  that does not reach its controllers fail to compile. A controller that
  supports `loop` holds a rolled loop's body once against `max_events`; one
  that does not plays it unrolled, and its limit error names the loop; unrolling adds pulses,
  not waveform memory, which holds each shape once. `rcp` keeps a branch
  on a readout the `feedback_latency_cycles` of the controllers that read out.
  A control system's `line_delays_ns` and `reference_clock_ghz` are its
  calibrated line delays and shared clock, as on a `MachineModel`; new
  delays alone schedule its programs again without tracing them. Without a
  declared `"clock_ghz"`, a device's program clock is its fastest
  controller clock when every other one divides it; clocks that do not
  divide evenly are a `DeviceDefinitionError` that lists the program clocks
  that work, and every controller clock must be a whole multiple of the
  reference clock. Each controller of a system may run its own clock: every
  line is scheduled on its own controller's grid, a `sync` and a wait from
  an answer on the grid all lines share, and a runtime qudit's lines on that
  shared grid too, so one schedule fits any qudits a call binds. Each
  controller's limits, line delays and `rcp` readout latency count its own
  cycles, and `custom()` samples an envelope at the rate of the line that
  plays it, while samples given at another rate fail to compile.
  `export_controller(controller=k)` and `export_emulation(controller=k)`
  export controller `k`'s part of the program in its own cycles: its
  lines, the events it plays, submits or takes the answer of, its calls and
  its limits, with the program's event and line ids. A program that plays
  the lines of more than one controller needs `controller=`.
  Budgets between controllers add each one's start uncertainty, its
  `clock_skew_cycles` plus one of its own cycles without a
  `reference_clock_ghz`: a pulse that branches on, or computes with,
  another controller's readout waits for the reader's feedback and peer
  latency and both uncertainties (`rcp` waits that long for each pair of
  reader and user, and a pulse on the reader's controller its feedback
  latency alone; starting sooner is a `ControllerLimitError` naming the
  cycle), and an answer taken on another controller than the one that sent
  the request is due that much later (the two controllers' uncertainty; the
  link's worst pair when the scheduler cannot tell which they are). The
  first pulse after a sync of several controllers' lines, on each of them,
  must land within its controller's `alignment_tolerance_cycles`: its start
  uncertainty and its rounded line delay, else a `ControllerLimitError`.
  Under `rcp`, a pulse that follows another controller's on one of the
  `max_concurrent_drives` or `max_concurrent_readouts` lanes starts their
  start uncertainty after it.
- **Processors.** `pulse.Processor` is a classical processor on a
  controller (`Controller(processors=[...])`), which runs the classical
  functions called on its `links`; a link belongs to at most one processor
  of a system. Compiling checks each function there: an operation in its
  `unsupported` list (named with its line), `max_ops_per_function`,
  `max_state_bytes_per_function`, `fixed_cost_only`, and its compute time at
  the processor's `cycles_per_op` and `clock_ghz` against its
  `max_latency_ns`. It checks each processor's `max_ops_total`,
  `max_state_bytes_total`, `max_calls_per_shot`, `max_bytes_per_shot` and
  `max_concurrent_calls` (default 1) across its functions. `"loop"` is a
  processor operation too: its cost is a rolled loop's counter and jump per
  iteration, and a processor without it runs loops unrolled, so their size
  counts unrolled. `dispatch_keys={"name": key}` gives the functions a
  processor runs fixed dispatch keys; a controller that computes with its
  own fields states them as `Controller(dispatch_keys=...)` (a
  `[dispatch_keys]` table in its description). `lowered.controller_usage` reports
  each processor's functions and totals against its limits
  (`.within_limits`; `print` it for a table). A program over them does not
  compile: its `ControllerLimitError` carries the report as `usage`, every
  limit exceeded with by how much (`usage.exceeded`, `LimitExceeded`), and
  ends its message with the table. Descriptions and system files state
  processors as `[[processors]]` tables.
- **Controller descriptions.** `pulse.controller("vendor.model",
  drive_lines=..., readout_lines=...)` loads a controller from a TOML
  description installed with cudaq-pulse (or one given by path), parsed
  once, with the installation's wiring. `reference.controller` ships.
- **System files.** `Device.as_dict()` writes a QPU as one mapping, format
  `cudaq-pulse.system/1`, which `pulse.device()` reads back from a mapping
  or a JSON file, reporting each problem with its line. A file written from
  another version of an installed description is noticed once
  (`UpgradeNotice`). `python -m cudaq_pulse.doctor --system FILE` checks a
  system file and whether its controllers' backends are installed.

### Changed

- **The simulator says which systems it has no model for.** Simulating a
  system that is not a transmon raises `UnsupportedError` (a
  `NotImplementedError`): export and hardware runs work.
- **Calibration on the controller.** `cudaq_pulse.classical.calibration`
  adds the estimators of millisecond on-controller calibration in controller
  form: adaptive T1 tracking with assignment errors (`adaptive_t1_update`,
  `adaptive_t1_wait`), the frequency binary search
  (`binary_search_update`, `binary_search_wait`, `binary_search_phase`),
  the analytical decay estimator (`decay_time`) and the sparse phase
  estimator (`sparse_phase`, `pi_train_amplitude`, `ramsey_detuning`).
  `examples/21_calibration_protocol.py` runs the millisecond calibration
  chain of M. A. Marciniak, R. T. Birke, J. B. Severin, F. Berritta,
  D. Kjær, F. Nilsson, S. N. Themadath, S. Kallatt, J. L. Webb, K. Bentsen,
  T. Madsen, Z. Sun, S. Krøjer, C. W. Warren, J. Hastrup and M. Kjaergaard,
  "Millisecond-Scale Calibration and Benchmarking of Superconducting
  Qubits", arXiv:2602.11912 (2026), as one program with its estimators on
  the controller link, its adaptive T1 step after F. Berritta et al.,
  "Real-time adaptive tracking of fluctuating relaxation rates in
  superconducting qubits", arXiv:2506.09576 (2025);
  `benchmarks/classical/calibration_protocol.py` checks it end to end over
  many seeds against the published numbers.
- **What classical functions cost a controller.** `controller.resources()`
  reports each classical function's operations by class (read from its
  body; an expression of constants folds, and dividing by a constant is a
  multiply), state in bytes, compute time against `max_latency_ns`, and calls,
  slots and buffer bytes per shot, and whether its body is in controller
  form (fixed-size state, bounded loops, no allocation or exceptions).
  `ControllerLimits` gains `op_cycles`, `compute_clock_ghz`, `max_ops`,
  `state_bytes`, `max_calls_per_shot`, `buffer_bytes`, `controller_form`
  and `compute_links`; compiling fails with `ControllerLimitError` when a
  program exceeds them.
- **Faster simulation of long programs.** Idle waits and square pulses in
  their frame are propagated exactly, at a cost independent of their
  length, and driven stretches on registers of up to four levels by RK4
  maps built at once (a 100 us T1 shot: 104 s to under 1 ms).
  `simulate(branches="sampled")` (or `pulse.BranchCoverage.SAMPLED`) follows
  only the branches the shots take, so a program that measures many times in
  one shot costs time linear in its measurements. Finding a program's next measurement and next call is now
  linear in its length.

- **Kernels.** `@pulse.kernel` traces a Python function into Pulse-dialect
  MLIR. Calling a kernel with qudit handles compiles it for their device on
  first use, caches the specialization, and runs it; `kernel.lower(...)`
  stops before running, and `pulse.compile(kernel)` compiles ahead of the
  call. `float` parameters stay symbolic, so a parameter sweep compiles once.
  The module with a call's values substituted, which exports and the LLVM
  stage read, is built once per distinct values and shared by every call
  with them. Kernels call other kernels, loop with `range`, and branch on measurement
  results. A number the arms of an `if` on a measurement set differently
  (`if m: amp = 0.3` / `else: amp = 0.1`) holds the value of the arm that
  ran, as one `arith.select` on the measurement, in `elif` chains, nested
  ifs and rolled loops too. Lowering keeps the select where the target
  takes that amplitude, phase or frequency at run time, so every use reads
  the one value; otherwise, and for durations and waveform shapes, each op
  that reads it runs in both arms of an `if` on the measurement. No target
  takes one at run time yet. Either way the ops wait for the measurement
  and schedule and simulate alike. A name assigned in only one arm, with no
  value before the
  `if`, raises a `CompilationError` naming it and the line of the `if`.
  Kernels take their qudits as parameters; a call without qudit
  handles raises `NoDeviceError` unless the kernel was compiled for a device
  (`pulse.compile(k, device=dev)`).
- **Lines and their tones.** `get_drive_line(q)` and `get_readout_line(q)`
  return the line alone, one value that is not unpacked or iterated;
  `line.tone` is its own tone, and `line.tones[(lower, upper)]`, a
  read-only mapping keyed by levels, the tone of each transition of its
  qudit: the line's own tone is that of the model's `default_transition`
  (`(0, 1)` in the shipped models), and another transition's starts at its
  frequency from the model, with phase 0. `drive(line, waveform)` and
  `readout(line, waveform)` play on the line's own tone and, by keyword
  only, `tone=` on another, such as another line's; the frame operations
  take the tone, as in `shift_phase(line.tone, phase)`.
  A transition the qudit does not have raises a `KeyError`
  (`pulse.NotFoundError`) that lists those it has.
- **Loop unrolling.** A `range` loop whose variable is never read is kept
  as one rolled loop, unless its estimated unrolled size (trip count times
  the calls in its body) is at most the unroll threshold: 8 by default, the
  measured break-even point. The threshold is a compile option only, an
  `int` of 0 or more set with `pulse.compile(k, unroll_threshold=)` or
  `@pulse.kernel(unroll_threshold=)`; 0 means never unroll, so every such
  loop stays rolled, even one of no or one iteration. No environment
  variable sets it.
  Lines and tones kept in a list, tuple or dict, retuned with
  `set_frequency` or not, are carried through a rolled loop. A rolled body
  may iterate over qudits (`for q in qs`) and gather their lines in a list.
- **Runtime wait lengths.** A `wait` whose length is computed from `float`
  parameters with `+`, `-`, `*` and `/` (`wait(d, tau)`, `wait(d, tau / 2)`)
  keeps them runtime parameters: a T1, Ramsey or echo sweep compiles once,
  and a call with a new length costs about as much as a cached call. Each
  call rounds the length to the nearest clock cycle (ties up) and schedules
  exactly as the kernel compiled with that length would. The optional
  `wait(line, length, bounds=(lowest, highest))` bounds it in cycles; a
  length that is not finite, is negative or is out of bounds raises
  `ValueError` at the call. A literal length must be a whole number of
  cycles. A length computed from the answer of a classical call makes the
  wait a variable region: it is rounded the same way, clamped to its
  bounds, and opens a new epoch (below). A length from both, such as
  `wait(d, answer + tau)`, compiles once too: each call checks that its
  float part (the length with its answers at 0) is finite and at most the
  upper bound, and the whole length is clamped to the bounds at run time.
- **Python in kernels.** List, dict and set comprehensions, set displays
  and generator expressions are unrolled while tracing, so
  `[get_drive_line(q) for q in qubits]` builds a list of lines and
  `sum(a for a in amps)` works; a lambda or nested `def` raises a
  `CompilationError` that says what to write instead. Calls with
  `*args` and `**kwargs` are unpacked
  while tracing, so `sync(*lines)` synchronizes a list of lines built in a
  loop. Helpers called from a kernel may take keyword-only, `*args` and
  `**kwargs` parameters; methods of helper objects, `functools.partial`
  objects and decorated helpers that use pulse ops are inlined too. A
  method is inlined like a plain helper, with its object as `self` (its
  class as `cls`) and the globals of the module that defines it, also when
  it reaches a pulse op only through `self`, `cls` or an object `self`
  holds (`self.wf()`, `self.gates.x90(...)`), when it is inherited, a
  `staticmethod` or `classmethod`, stored as a bound method, or an
  object's `__call__`. Lines
  and tones may be kept in any container, such as a list of lines or a
  dict, and lists of them support `pop`, `insert`, `extend`,
  `remove` and `del xs[i]`. `sync()` takes lists and tuples of lines, as
  in `sync(lines)`. A pulse op given the wrong kind of argument, such as a
  tone where its line belongs, raises a `CompilationError` that names the
  op and the argument when it is traced, not an MLIR verifier error.
  Errors raised while tracing carry a note with the file, line and source
  of the kernel line, and of each helper line, that raised them; a
  misspelled op or attribute gets a "did you mean" suggestion, and
  `pulse.drive(...)` says that the kernel language is in `cudaq_pulse.ops`.
  A failing `assert cond, "msg"` reports "assertion failed in kernel: msg".
  Starred assignment, `first, *rest = xs`, works, and so does slice
  assignment on every supported Python. A construct the tracer does not
  support, such as a `match` class pattern or an assignment to a global,
  is named in the error, with what to write instead, rather than reported
  as an unsupported bytecode instruction. A `float` argument that tracing
  makes static, because `abs`, `min`, `math.*`, a comparison, a `range`
  bound or a waveform duration needs its value, warns once per kernel and
  argument, since each new value then compiles the kernel again. The
  warning, a `UserWarning` that is also a `RuntimeWarning`, names the
  argument, the operation and the kernel line (and the helper's line when
  the value is needed in a helper), and suggests passing a host-computed
  value, annotating the argument as `int`, or giving it a fixed value.
  Compiles of one kernel that trace at once warn once between them, and a
  cached call checks nothing new.
- **Operations.** The kernel language lives in `cudaq_pulse.ops` and is
  used by bare name inside kernels. Drive and readout channels per qudit; `drive`, `readout`,
  `wait` and `sync`; phase and frequency control (`shift_phase` follows the
  McKay et al. 2017 virtual-Z convention); Gaussian, square, DRAG (beta in
  clock cycles), cosine, tanh-ramp, Gaussian-square and sampled waveforms;
  waveform sums, differences, scaling and negation (exact) and products
  (sampled once per clock cycle).
- **Editor support with no configuration.** The wheel installs
  `__builtins__.pyi` at the root of site-packages, generated at build time
  from `cudaq_pulse.ops.__all__`, so Pylance, Cursor, pyright and
  basedpyright know the operations used by bare name inside kernels: no
  undefined-name warnings, with completion, hover documentation and checked
  arguments. The names are builtins in the whole environment, outside
  kernels too; see the new Editor Support page. The getting-started advice
  to add a `pyrightconfig.json` is gone.
- `python -m cudaq_pulse.doctor --setup-editor` configures mypy and ruff,
  which take no names from an installed package: it shows the change to the
  project's mypy and ruff configuration and writes it when confirmed or
  with `--yes`. The mypy change enables `cudaq_pulse.mypy_plugin`, which
  types the operations in files that import cudaq_pulse; the ruff change
  lists them as `builtins`.
- `cudaq_pulse.ops.__all__` lists the kernel language.
- `ScheduledEvent.start` and `ScheduledEvent.end`: an event's scheduled
  times in clock cycles as `int`, raising `ValueError` on an event that is
  not scheduled yet.
- `Kernel` and `CompiledKernel` are generic in the decorated function's
  parameters, so type checkers check `k(...)`, `k.lower(...)` and
  `pulse.compile(k)(...)` against the kernel's signature.
- `square()` is typed to take the complex and `[re, im]` amplitudes it
  accepts.
- `custom(duration, f)` samples the Python function `f` once per clock cycle
  while the kernel is traced, so custom envelopes compile, schedule and
  simulate like the built-in ones; it takes a callable, not an envelope
  name, and up to 252 clock cycles, like `custom_samples`.
- **Compilation.** Verification, constant and phase folding and pulse
  fusion by default; ASAP, ALAP and resource-constrained schedules
  (`cudaq_pulse.compiler.MachineModel`); per-kernel schedule policies and
  schedule blocks. `pulse.Lowered.mlir(level)` returns the module at an IR
  level (`cudaq_pulse.compiler.IRLevel`: `"pulse"`, the scheduled Pulse
  MLIR, `"qop"`, `"cudm"` or `"llvm"`), `Lowered.schedule` the timed events
  (`cudaq_pulse.compiler.ScheduledEvent`), `Lowered.duration` how long the
  program runs as a `pulse.Time`, `Lowered.metrics` the compile
  times (`cudaq_pulse.compiler.CompileMetrics`), and `Lowered.timeline()` a
  plot (with the `[viz]` extra). `Device.qudit(label)` finds a qudit by
  its label and `TargetDescription.coupling(a, b)` the coupling of two.
- **Optimization passes.** `passes=` takes members of `pulse.Pass`, a
  string enum, or their names, in any order and combination: `VERIFY`,
  `FOLD_CONSTANTS` (constant arithmetic computed at compile time),
  `REMOVE_UNUSED` (operations whose results nothing uses), `MERGE_WAITS`
  (constant waits in a row on one line joined into one),
  `SIMPLIFY_CONTROL_FLOW` (syncs that order nothing, blocks that take no
  time, loops that run once or never), `FOLD_PHASES` (frame changes folded
  into the next drive), `FUSE_PULSES` (square pulses of the same amplitude
  in a row merged) and `HOIST_LOOP_INVARIANTS`. The default is `VERIFY`,
  `FOLD_CONSTANTS`, `REMOVE_UNUSED`, `FOLD_PHASES` and `FUSE_PULSES`. An
  unknown name is a `ValueError` that names the closest member. Pattern
  passes that follow one another run as one rewrite, so compiling, binding
  a call's values and exporting a new binding are faster. When a call binds
  `float` parameters, `FOLD_PHASES` and `FUSE_PULSES` run again, if the
  kernel compiled with them, where the values reach: a phase bound to a
  number folds into the next drive, and square pulses whose amplitudes are
  bound equal fuse. `cudaq-pulse-opt` runs each pass by name
  (`pulse-fold-constants`, `pulse-remove-unused`, `pulse-merge-waits`,
  `pulse-simplify-control-flow`, `pulse-fold-phases`, `pulse-fuse-pulses`,
  `pulse-hoist-loop-invariants`), as well as `pulse-unroll-classical-loops`
  and `pulse-schedule-rcp` with its machine model as options (`max-drives`,
  `max-readouts`, `readout-latency`, `switch-penalty`, `alap`).
- **Devices and targets.** `pulse.device` is the one device getter: it
  describes a machine as a target plus a qudit layout, from a registered
  target name (`pulse.device("transmon_17q")`), a config mapping
  or file, or, with no argument, the process target. Devices compare by
  content and share compiled kernels. Target descriptions are
  `cudaq_pulse.targets.TargetDescription`; the shipped target,
  `"transmon_17q"`, is a 17-qubit transmon device built from
  Krinner et al., Nature 605, 669 (2022).
  A target's `model` (`cudaq_pulse.targets.TransmonModel` by default, a
  `SimulationModel` class or instance, or `None`) says how the simulator
  treats its qudits; each model class declares its levels (3 for
  transmons), and `TransmonModel(levels={0: 4})` sets a qudit's own. No
  label changes the physics: targets have no `architecture`.
  A `Coupling`'s gate fields (`gate_type`, `gate_duration_ns`,
  `gate_buffer_ns`, `gate_fidelity`) default to `None`, unspecified; the
  `transmon_17q` target gives its published CZ values. A config or file with
  problems raises `pulse.DeviceDefinitionError` (a `ValueError`) listing all
  of them in `problems`, each a `pulse.Problem` with its path in the config
  (`layout[2]`), the offending value and a fix, and for a file its line and
  column; an unknown key or target name suggests the closest one, and the
  reserved keys (`zones`, `sites`, `moves`, `transport`) are problems too.
  `python -m cudaq_pulse.doctor --system FILE [--json]` runs the same checks.
- **Calibrations and the transmon model.** `cudaq_pulse.targets.Calibration`
  loads fitted values, each with its kind, source, uncertainty and cooldown,
  apart from the published target, and a device config takes `"calibration"`
  and `"cooldown"`. `cudaq_pulse.targets.transmon_model` computes
  flux-tunable transmon levels and charge-coupled pairs from circuit values.
  The shipped `transmon_17q` calibration has one device-wide charge coupling,
  fitted to the measured `|11>`-`|20>` coupling of X2-D4, so the
  `transmon_17q` target's exchange (5.67-6.08 MHz) and idle ZZ (-4.5 to -7.3 kHz) depend on
  each pair's frequencies; calibrated qubits convert detunings to and from flux quanta.
  It also holds in-model sudden-net-zero CZ knobs for all 24 pairs (mean
  duration 65.0 +- 4.8 ns against the published 68(7) ns), each with its
  predicted phase and leakage, flags and alternatives.
- **Simulation.** `pulse.simulate` evolves a compiled call with T1, T2 or
  Hahn-echo dephasing, thermal populations, exchange couplings, residual ZZ
  and any number of transmon levels (by default those the target's model
  declares, 3 for a transmon; `levels=` sets them for every qudit or, as a
  dict, per qudit), either with a NumPy reference
  integrator (`backend="cpu"`) or on NVIDIA GPUs through cuDensityMat
  (`backend="gpu"`, Linux, experimental). Programs that measure and branch
  are evolved as an exact ensemble over outcomes. The simulated space holds
  only the qudits the program gets a line on, whatever their device indices
  (a pair at qudits 13 and 3 is a two-mode problem); `SimulationResult.qudits`
  and `Lowered.qudits` give the device qudit of each mode, and the other
  qudits are classical spectators. `result.level_population(level, qudit)`
  and `result.level_populations(qudit)` give a qudit's level probabilities
  from its reduced state, for any number of levels; `qudit` (a device index
  or handle) may be left out when the result models one qudit, and a qudit
  or level the result does not have is an `InvalidValueError` listing those
  it has. `result.branches` holds one `Branch` per measurement path: its
  `levels`, `probability` and classical `calls`, with `answers(function)`
  for the answers of one function.
- **Classical calls at QPU runtime.** `classical`, a classical real-time
  dialect for co-processing at QPU runtime: pulse programs call GPU, host or
  FPGA services (decoders, estimators, calibration updates) while the QPU
  runs and use their answers in the same shot. `@pulse.classical` declares a
  function with typed arguments and results, a latency model and a link;
  `fn.submit(...).result()` calls it in a kernel, with `timeout=` (cycles or
  a time, `350 * pulse.ns`) for a fallback. Type checkers and editors see
  the decorator with its keyword options, and `.submit()` returns a
  `Pending`. Links are named in
  `MachineModel(links={...})`. Every schedule budgets the worst-case
  latency, holds link slots and reports each call's slack and its hidden and
  exposed latency (`Lowered.classical_report()`); `pulse-classical-check`
  verifies slots, deadlines and ordering. A wait whose length is an answer,
  and a blocking wait for an answer, are barriers for the lines that depend
  on them (through a `sync`, a measurement or answer, a branch or a shared
  tone): those lines split into epochs (`Lowered.epochs()`) whose start
  shifts with the run-time lengths before them, and other lines keep their
  time, so qubits waiting on answers of their own run side by side.
- **Simulating and emulating calls.** `pulse.simulate` binds answers per
  measurement path, with timeouts from the latency model, assignment errors
  and relaxation during readout, IQ points from a dispersive readout model,
  and `plant=` to simulate a drifted device without compiling again.
  `Lowered.export_emulation()` writes the schedule, epochs and call manifest
  for a controller; `classical.emulation.emulate` runs a program in virtual
  time against real services, in process or over UDP. Requests and answers
  use the CUDA-Q real-time wire format (`classical.wire`);
  `classical.qec.Decoder` talks to a CUDA-QX real-time decoder. The
  cuDensityMat runtime keeps a classical handler table per run.
- **Calibrating while the QPU runs.** `classical.calibration` (adaptive T1,
  a frequency binary search, three-point decay and phase estimators,
  golden-section and Nelder-Mead steps), `classical.random_index` (an
  on-device random source), `classical.loop` (a parameter store whose values
  programs read at run time, and an estimate log) and `runtime.drift` (a
  drifting device with its truth trace and surrogates). Examples
  `19_classical_calls.py` and `20_adaptive_calibration.py`, and the
  "Classical Calls at QPU Runtime" guide.
- **Controller backends.** `Lowered.export_controller()` exports a
  scheduled kernel for an out-of-tree controller backend, format
  `cudaq-pulse.controller/1` (`cudaq_pulse.controller.FORMAT`): the lines
  with their roles, and every drive, readout, wait, sync and classical call
  with its exact start and end, waveform and parameters (and, with
  `sample_rate_ghz=`, its samples), tone frequency and frame phase (including
  the phase a detuned tone gains between drives), epoch and the
  measurement- or answer-conditioned branches it lies in. What is known only
  at run time is exported as an expression over readouts and answers: a
  waveform parameter (`runtime`, including a value a measurement selects),
  a variable wait's length and bounds, a submission's operands, and, on a
  tone whose frame depends on run time, the frame changes since its
  previous drive (`tone`, `frame_updates`), so a backend can compute them
  where the answer arrives. A program that
  computes nothing at run time exports exactly as before. Within a version,
  keys and event kinds are only added. `MachineModel(controller=
  ControllerLimits(...))` states what a controller can play (clock, issue
  spacing, waveform memory at its `sample_rate_ghz`, tones, the largest
  amplitude, readout acquirers, feedback latency, answer timeouts, program
  size; the compiler itself limits no amplitude) and makes compilation fail with
  `ControllerLimitError` when a program exceeds it, parametric kernels
  included: a limit that only some values of a float parameter exceed, such
  as a tone shift, a waveform width or a wait length, is checked at the call
  with its values, so a call that lowers also exports. `cudaq_pulse.backends` finds backends installed under the
  `cudaq_pulse.backends` entry-point group; a backend that fails to load, is
  not a `Backend` or does not read this export format raises
  `IncompatibleBackendError`, and an unknown name `BackendNotFoundError`,
  both `BackendError`s. `examples/controller_backend/` is a toy backend, and
  the "Writing a Controller Backend" guide walks through one.
- **Public API.** The top level is exactly `kernel`, `Kernel`, `compile`,
  `CompiledKernel`, `Lowered`, `Pass`, `simulate`, `SimulationResult`,
  `BranchCoverage`, `device`, `Device`, `Qudit`, `Register`,
  `PulseError`, `Problem`, `DeviceDefinitionError`, `InvalidValueError`,
  `InvalidTypeError`, `NotFoundError`, `UnsupportedError`,
  `CompilationError`, `TimingError`, `ExecutionError`, `InternalError`,
  `PulseWarning`, `UpgradeNotice`, `NoTargetError`, `NoDeviceError`,
  `DeviceMismatchError`, `GpuRuntimeError`, `viz`, `classical`,
  `controller` and `backends`, plus `__version__`.
  `simulate(program, *args, **options)` is the one way to simulate: it
  takes a kernel and its arguments as `compile` takes a kernel, or a
  program compiled already, such as one a worker process receives pickled,
  without compiling it again. A module with run-time float parameters
  (`lowered.generic`) takes their values as `bind` does; the module of a
  call, from `kernel.lower(*args)` or `bind`, takes none. `bind(values)`
  takes one real number per parameter, in the order of `parameters`; too
  few or too many values are an `InvalidTypeError` naming the parameters.
  A bool, str or other non-number is an `InvalidTypeError`, and a NaN or
  infinity an `InvalidValueError`, naming the parameter where the module
  is used, as for a kernel call's float arguments.
  Every error cudaq-pulse raises on purpose derives from `PulseError` and
  has the built-in base of its kind: `InvalidValueError` is a `ValueError`,
  `InvalidTypeError` a `TypeError`, `NotFoundError` a `KeyError`,
  `UnsupportedError` a `NotImplementedError`, and `CompilationError`,
  `ExecutionError` and `InternalError` are `RuntimeError`s. An
  `InternalError` is a bug in cudaq-pulse, and its message asks for a
  report. `Calibration.load` of a calibration that is neither shipped nor a
  file raises `NotFoundError`, and a malformed request or answer on a
  classical channel an `ExecutionError`. Every warning cudaq-pulse issues
  is a `PulseWarning`.
  `cudaq_pulse.compiler`, `cudaq_pulse.runtime.simulate` and
  `cudaq_pulse.targets.transmon_model` list their public names in `__all__`,
  as the other documented modules do. `cudaq_pulse.targets.UnknownTargetError`
  (an unregistered target name) and
  `cudaq_pulse.compiler.UnsupportedInterpreterError` (a Python whose bytecode
  the kernel tracer does not know) are public, and every public name is in
  the API reference, errors on their own page.
- Target registration, the shipped targets and calibrations, and the kernel
  tracer's caches are safe to use from several threads on free-threaded
  CPython: each target and calibration is built once.
- **Dialects and tools.** The Pulse, Classical, QOp and CuDensityMat MLIR
  dialects with their lowerings to LLVM; `cudaq-pulse-opt`, the MLIR driver, which
  also loads out-of-tree pass plugins; `python -m cudaq_pulse.doctor`, which
  reports what an installation contains and whether GPU execution works.
- **Packaging.** One abi3 wheel per platform for CPython 3.12 and newer:
  Linux x86_64 and aarch64 (manylinux_2_28) with the CUDA 12 and CUDA 13
  GPU runtimes, selected by the `[cu12]` or `[cu13]` extra, and macOS arm64
  (compiler only). The wheels link their own MLIR, so they need neither
  CUDA-Q nor an LLVM installation.

### Fixed

- A tone that an arm of a measurement-conditioned `if` changed with
  `shift_phase`, `set_phase`, `shift_frequency` or `set_frequency` alone,
  with no drive on it in that arm and no `sync` with the readout line after
  the readout, was scheduled as ready before the measurement: the next drive
  on it started while the readout was still playing (ASAP or ALAP), and ALAP
  could move the readout after that drive. In export and simulation the
  drive used a phase or frequency the measurement had not yet chosen. A
  value an arm of an `if` yields is now ready only once its condition is,
  under every schedule, also in kernels with classical calls. Programs
  without such an arm schedule as before.
- Values no op can take are a `CompilationError` that names the op, the
  value and the kernel line that wrote it, rather than compiling silently
  or failing in an MLIR pass: a NaN or infinite amplitude, sigma, beta,
  phase, frequency or `custom_samples` sample; a sigma that is not
  positive; an amplitude or other float that is not a number, such as
  `None`; and a duration outside its cycles. A waveform lasts 1 to 2**53
  cycles, and a wait, a flat top (`gaussian_square` width) or a bound 0 to
  2**53, whether given as an `int`, a time, or an `int` or `float` argument
  the kernel is compiled for. `simulate(shots=...)` and `emulate(shots=...)`
  take a positive `int` (`InvalidTypeError` or `InvalidValueError`
  otherwise; `simulate`'s default `None` still samples nothing), and a
  qudit index past a device or register is an `IndexError` that says how
  many qudits it has. A NaN or infinite float argument is an
  `InvalidValueError` naming the parameter wherever the program is used:
  run, simulated, exported or printed.

[0.1.0]: https://github.com/NVIDIA/cudaq-pulse/releases/tag/v0.1.0
