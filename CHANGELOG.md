# Changelog

All notable changes to cudaq-pulse are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
the [versioning policy](CONTRIBUTING.md#versioning-and-stability): while the
major version is 0, a minor release may break the API.

Each GitHub release takes its notes from the section for its version, so
every change worth telling users about goes in the top section, the next
release's `## [X.Y.Z] - Unreleased`, in the pull request that makes it.

## [0.1.1] - Unreleased

### Changed

- `viz.save_schedule` now accepts `str | os.PathLike[str]` for `path` and
  explicitly names its keyword-only arguments `figsize` and `title` instead of
  `**kwargs`.

### Added

- **Error codes.** Every `CompilationError` has a `code`, a
  `pulse.ErrorCode` (`unknown_name`, `wrong_argument`, `runtime_value`,
  `invalid_duration` and the others listed under Errors), so a program or
  tool can act on the kind of error without reading the message. `fix`
  holds what the message says to write instead, `did_you_mean` the name to
  write in place of a misspelled op or line method (`drve` names `drive`,
  `get_calibraton` names `get_calibration`), and `location`, for an error
  raised while a kernel is traced, the kernel line: a `Problem` with its
  `file`, `line`, `source` (new: the line's text), function and, for a
  misspelling, the misspelled name as `value`. Messages read as before; a
  few that gave no fix now give one (`drive(q, ...)` says to pass
  `get_drive_line(q)`, `shift_phase(line, ...)` to pass `line.tone`).
- **Unit warnings.** `pulse.compile(k, unit_warnings=True)` issues a
  `pulse.UnitWarning` at each kernel line that gives a duration, wait,
  `sigma`, DRAG `beta` or flat-top `width` as a bare number (clock cycles:
  `square(20, a)` is 10 ns at 2 GHz), or a frequency far below the quantum
  system's carriers (`shift_frequency(tone, 5)` is 5 Hz). Times with
  units, `0`, `round_duration` counts and calibrated pulses are not
  warned. It is off by default, does not change the program, and a
  compilation without it checks nothing.
- **Minimum pulse length.** `Controller(min_pulse_cycles=...)` states the
  shortest drive or readout a controller plays, in its own cycles, as some
  instruments require a minimum pulse length. Compiling checks it like the
  other limits: a shorter pulse is a `ControllerLimitError` naming its line,
  start and length, also in rolled loops and on each controller of a
  control system. The default, 0, is no limit, and the export lists the key
  among the limits only when it is set, so existing programs and exports
  are unchanged.
- **Maximum pulse length.** `Controller(max_pulse_cycles=...)` states the
  longest drive or readout a controller plays, in its own cycles, as some
  instruments cap a pulse's length. It is checked wherever the minimum is:
  a longer pulse is a `ControllerLimitError` naming its line, start and
  length, also in rolled loops and on each controller of a control system.
  The default, `None`, is no limit, it may not be below `min_pulse_cycles`,
  and the export lists it among the limits only when it is set, so existing
  programs and exports are unchanged.
- **Reading calibrations.** `PhysicalQudit.calibrations` and
  `Coupling.calibrations` give the calibrations a qudit or coupling
  publishes, read-only and keyed as `calibrations=` takes them (by
  `pulse.gates.Gate`, or `(gate, transition)`), so a calibration step can
  change one and keep the rest: `q.replace(calibrations={**q.calibrations,
  pulse.gates.X: tuned})`. `get_calibration` still derives what is not
  published.
- **Conventions page.** The user guide's Conventions page states the
  units, time bases and signs a program depends on: bare numbers are clock
  cycles and frequencies Hz, the drive axis of a complex amplitude,
  `shift_phase(+phi)` as a virtual `Rz(+phi)` that turns later pulses by
  `-phi`, the frame of a simulated state, which way a virtual detuning
  moves Ramsey fringes (`|D - v|`), and that an exported `phase_rad`
  multiplies the samples by `exp(-i phase_rad)`. Each rule has a snippet
  that shows it; `tests/test_conventions.py` runs the snippets, checks what
  they print, and checks each rule.
- **Update buses.** `Controller(update_buses=[UpdateBus(drive_lines=[0, 1],
  cycles_per_update=8)])` (`cudaq_pulse.controller.UpdateBus`) states that
  a group of lines is retuned over one shared bus, one update at a time.
  An update is a frame update, or a drive or readout whose frequency, or
  whose phase other than as its tone runs on, differs from the one before
  it on its line. Compiling checks that updates on one bus start at least
  `cycles_per_update` apart, naming the bus, the lines and the times, on
  each controller of a control system; on one clock a rolled loop is
  checked at its first three iterations and its last two, which stand for
  every iteration. With `Controller(space_updates=True)` compiling spaces
  the updates instead: it holds the first update that clashes back on its
  line as little as it must and schedules again, until none clashes, and
  the checks, the export and the simulator all read the spaced program.
  The buses are wiring, given when a description is loaded. There are
  none by default and spacing is off, and the export lists both only when
  set, so existing programs and exports are unchanged.
- **Phase gates.** `pulse.gates.Z`, `S` and `T` are virtual Z rotations by
  pi, pi / 2 and pi / 4. `get_calibration` gives each as a `VirtualZ`
  (`cudaq_pulse.quantum_systems.VirtualZ(phase_rad=...)`), derived on every
  transition where a quantum system publishes none, and `drive(line, it)`
  plays it as `shift_phase(line.tone, phase_rad)`: no pulse and no time.
  The operations guide's new "Phase frames and virtual Z gates" section
  gives the sign convention and why a simulated state is in the frame.
- **Post-selection.** `SimulationResult.postselect(keep)` is the result
  over the branches the test `keep` accepts (a function of a `Branch`, as
  `lambda b: b.answers("herald")[-1]`): their state, normalized, their
  probabilities divided by the kept total, which is the new result's
  `acceptance`, and the shots that took them in `records`. Every other
  result has an `acceptance` of 1.0.
- **Validating controllers with long minimum pulses.** `backends.validate`
  makes every pulse and wait of its corpus the fewest whole times as long
  that the controller plays: no shorter than its `min_pulse_cycles`, and
  on its lines' timestep and sample rate. A controller whose shortest pulse
  is 64 cycles had 14 of the 16 programs skipped; it now has them all
  checked, with 16-cycle pulses lasting 64. On a controller that plays the
  corpus as written, the corpus, its exports and every report are as
  before.
- **Validating branches on slow feedback.** `backends.validate` runs the
  corpus's `branch` program on a controller with a feedback latency
  (`feedback_latency_cycles`) instead of skipping it: the pulse on the
  readout's outcome waits for the latency, as every schedule now does.

- **Where exported samples are taken, and what `phase_rad` counts.** The
  controller backends guide states that sample `k` of a shape is its
  envelope at the start of that sample, and a `custom()` envelope's at its
  middle, so a backend whose instrument evaluates shapes at the middle of
  each sample knows to delay them by half a sample. The Conventions page
  states that an event's `phase_rad` is the tone's phase in the frame that
  turns at the line's own frequency from the start of the shot, so it
  includes the phase a detuned tone gains, not only the phase shifts
  before it. Nothing changes in the export.
- **Waveform explorer.** The operations guide has an interactive plot of
  the built-in waveforms next to its waveform table: sliders for each
  constructor's arguments, the `wf_*` combinations, an optional modulated
  drive in the qudit's frame after `shift_frequency` and `shift_phase`, and
  DRAG's suggested `beta = -1 / alpha`. It prints the call that makes the
  plot, in clock cycles with the ns beside them; a test compiles every
  call it prints and checks its samples against `sample_waveform`. Readers
  without JavaScript see a static figure. The new example
  `29_waveform_envelopes.py` prints and plots the samples of every
  built-in envelope.
- **Native line frequencies in the controller export.** Each line of
  `export_to_controller()` now has `native_frequency_hz` (format `/1`) or
  `native_frequencies_hz`, one per qudit (format `/2`): the frequency of
  the line's own tone, at which every shot starts its frame and from which
  a retuned tone's `phase_rad` advances. A backend whose line plays several
  frequencies no longer needs the quantum system to find it. The key is new
  within both formats, so the format names are unchanged; `unroll` carries
  it, and a kernel with qudit parameters gives it as the slot's symbols
  instead, as before.
- **Animated tutorial figures.** T-State Injection plays its tomography
  axes one at a time, its heralded loop once per answer with a playhead,
  and the branches post-selection keeps one readout at a time. Adaptive
  Calibration moves its estimator's distribution from the prior to where
  a readout of 1, or of 0, takes it, and Simulating on a GPU steps through
  its program's lowering one stage at a time. Each plays
  once while it is in view, with Pause and step buttons, and its data
  comes from what the page compiles and simulates. Readers without
  JavaScript, or who ask for reduced motion, see the still frame.

- `UdpChannel` and `InProcess` print what they're set up with:
  `UdpChannel(('127.0.0.1', 50123), slot_bytes=65536, timeout_s=2.0)` and
  `<InProcess serving decode, estimate>`.

### Changed

- **Every schedule waits for readouts.** `asap` and `alap` (the default)
  now start an operation that depends on a readout (an arm of a branch on
  it, a pulse on a tone whose frame a branch on it changed, a value computed
  from it, a measured value carried out of an `if` or a loop, a classical
  request that sends it) only once the readout has ended on its line and
  the controller's feedback latency (and, on another controller, its peer
  latency and both start uncertainties) has passed, as `rcp` and `alap_rcp`
  do. Programs that failed with "... schedule with rcp ..." compile;
  programs in which nothing depends on a readout schedule as before. The
  latency can make a program exceed a budget it declares
  (`max_duration_cycles`, `reaction_deadline_ns`,
  `Controller.max_program_duration`); the error names it. For backend
  authors: `backends.validate` counts change on a controller with a
  feedback latency (`branch` passes instead of being skipped), and its
  `feedback_gap_cycles` argument, new in this release, is gone.
- **A kernel compiled without a quantum system takes its qudits' control
  system.** `lower()` and calls of a kernel compiled without
  `quantum_system=` or `control_system=` now schedule and check the program
  for the control system of the qudits it is lowered on: its feedback
  latency, line delays and controller limits apply. Such a call can now
  raise `ControllerLimitError`, or start its operations later by a line's
  delay, where it compiled before. Qudits of quantum systems that differ
  only in their control system get a program each.
- **Schedule plots look different.** `viz.plot_schedule`, `viz.save_schedule`
  and `timeline()` draw in the style of the documentation's figures: drives
  green, readouts rose, waits hatched grey and classical calls teal (colors
  that stay apart under the common color-vision deficiencies), no box or
  bar outlines, light gridlines, line 0 at the top, and the legend in one
  row above the plot. The rows, bars, labels and axis limits, and the
  functions' arguments, are as before, and matplotlib is still imported only
  when a plot is drawn.

### Fixed

- The schedule passes and `Pass.SIMPLIFY_CONTROL_FLOW` no longer crash on
  two inputs. A schedule block that yields one of its own arguments, passing
  an input through, is inlined with that input as its result; the passes
  used the argument after it was destroyed. And a module with no pulse
  operation or type, such as one with only classical calls, is scheduled
  like any other; the schedule passes wrote the pulse dialect's schedule
  state without loading the dialect.
- On one controller whose clock is not the program's, `rcp` and `alap_rcp`
  waited the feedback latency in the controller's cycles as if they were the
  program's: they failed their own check (100 cycles at 0.5 GHz are 400 of
  a 2 GHz program, not 100), and a request sending a readout's outcome could
  leave before it was there. Simulation and classical emulation schedule
  again with the same latency as `compile()`.
- A schedule longer than `Controller.max_program_duration` compiled when
  waiting for feedback, a line or a link made it longer than its operations
  played one after another (on several controllers nothing was checked at
  all); it is now an error that names the controller, and the feedback
  latency when the program waits one.
- A kernel called with its own `schedule=` could hand out a tone whose phase
  a measured `if` changed (on a bit it read, was passed, or carried from an
  earlier trip of a loop) as if unchanged, so the caller's drive started
  before the readout ended; and a loop in such a kernel that passes tones
  from one iteration argument to another could hand out a tone it was still
  playing on. `alap` and `alap_rcp` could move the readout before a loop past
  the first branch on the measurement the loop carries.
- A program's constants are created in the same order on every platform.
  The Linux x86_64 build created the constants of a waveform's literal
  arguments, and of a loop's bounds, in the reverse order of the Linux
  aarch64 and macOS builds, so `print(scheduled.mlir())` listed them in
  another order under other `%cst` names; and of two rejected values in one
  waveform (`gaussian(40, math.inf, 0.0)`) it named the last where the
  others name the first. The programs were the same. Output on Linux aarch64
  and macOS does not change.
- A quantum system's qudits are checked when it is built, as its couplings
  are: a 0-1 frequency that is not positive, a negative T1, T2* or Hahn-echo
  T2, a thermal population outside [0, 1), or a value that is not a finite
  number raises `QuantumSystemDefinitionError`, which lists every problem
  at its path (`qudits[0].t1_us`) with the value. Such values used to build
  without an error, and a negative T1 or T2 was taken as 0 (no decay) when
  simulating.
- A frame change made after a wait on its tone's line acted where the
  tone's previous drive or readout ended, before the wait:
  `shift_frequency(tone, +d)`, a wait of `T` and `shift_frequency(tone, -d)`
  exported (formats /1 and /2) and simulated with no phase gained, where
  the frame runs `2 pi d T` further on, as it did with a pulse in place of
  the wait; and a frequency set after a wait gained phase during it. The
  change was placed there, and the export and the simulator ran a tone at
  its latest frequency from its latest drive, wherever the change was. A
  change is now made where the program reaches it on the tone's line, after
  the waits, pulses and syncs before it there, and after a readout it
  depends on (in an arm of a branch on the measurement, or with a value
  computed from it), as with a sync after the readout; never after the
  pulse that plays with it. The frame runs at its old frequency until then.
  Phase changes play as before; their frame updates' `start_cycles` move to
  the end of such a wait or readout.
- After a wait whose length a `float` argument sets, `classical_report()`
  and the `"calls"` of `export_to_controller()` (formats /1 and /2, and
  `unroll`) and of `export_to_emulator()` gave each call's times with the
  wait at the lowest length its `bounds` allow, not at the length the
  argument gives: a call the schedule and the export's `"events"` submit at
  cycle 100340 was listed at 540. They now give the times of the program as
  the call's arguments schedule it, the same as its events and `emulate`.
  Waits set by an `int` argument or a literal were not affected.
- `pulse.controller.sample_waveform` is annotated as returning
  `npt.NDArray[np.complex128]`, not a bare `np.ndarray`, so editors and
  type checkers accept `.real` and `.imag` of its samples with NumPy 2.5,
  whose annotations allow them only on arrays of a known dtype.
- `backends.validate` checks the order of a backend's classical calls and
  answers on their link, apart from the pulses of the line with the same
  number. A backend that gives its events back in time order, as a
  sequencer plays them, no longer fails the `call` program because a drive
  on line 0 starts before a call over link 0 that the program makes first.
- `python -m cudaq_pulse.doctor` passes on a machine without GPU support:
  it now simulates a short pulse with the NumPy reference, reports
  `CPU simulation (NumPy reference): ok`, and says how to enable GPU
  simulation (`[cu12]` or `[cu13]` and the NVIDIA driver each needs). It
  still exits non-zero when the compiler, CPU simulation or
  `cudaq-pulse-opt` fails, and when GPU simulation is set up (a CUDA extra
  is installed, or `CUDAQ_PULSE_CUDA` or `CUDM_RUNTIME_LIB` is set) but does
  not work: a driver too old for the extra, both extras installed, a
  library that does not load. `GpuSupportNotBuiltError` no longer says the
  build cannot execute pulse programs: only GPU simulation needs GPU support.
- Under `rcp` and `alap_rcp`, a pulse that branches on, or computes with, a
  readout on a line with a calibrated delay (`ReadoutLine(q, delay_ns=)`)
  waited its feedback latency from where the readout was scheduled to end,
  not from where the delayed line ends it, so the program failed its own
  check (on the reading controller or, across controllers, on the one that
  plays the pulse). The wait now starts where the readout ends on its line,
  under every schedule, the controller's fine delay rounded up to a cycle.
  Programs without a readout delay schedule as before.
- A call over a per-controller link (`Link(per_controller=True)`) whose
  measurements and answer are on two controllers failed with a bare
  `RuntimeError` holding the compiler's IR and the module's name for the
  function (`decode__1`). It is now a `CompilationError` naming the kernel,
  the function as written, the link and both controllers, and saying how to
  fix it, whether or not the kernel has arguments bound at the call.
- `QuantumSystemMismatchError` for qudits of two quantum systems that
  differ only in their control system read "one with layout 0..16, the
  other with layout 0..16". It now names what differs: the control
  systems (one has none, or the first setting that differs, such as
  `control_system.controllers[1].peer_latency_cycles is 30 in one and 20
  in the other`), the layouts, or, with the same layout, the qudits,
  couplings or calibration.
- A program that plays a shared line (a multiplexed readout feedline,
  `ReadoutLine([9, 10, 11, 12], max_concurrent_tones=4)`) exports in format
  `/1`, so it exports on a control system of several controllers, on a
  controller with its own clock, and with line delays, where 0.1.0 refused it
  in both formats. Format `/1` keeps a line for each qudit; a line on a shared
  line also has the line's `qudits` and `max_concurrent_tones`, and each
  event on it a `tone_slot`, the slot format `/2` gives it. Programs on
  dedicated lines export as before. `backends.validate` runs two more
  programs: `shared_line`, and `several_controllers`, each controller's part
  in `/1`.
- `controller.unroll()` of a format `/2` program on a shared line gives each
  event its own qudit, on a line of its own, where it gave the line's first
  qudit.
- `transmon_surface17` played its `X` pulses with a DRAG coefficient of 0,
  had no pulse for the 1-2 transition, and read out without assignment
  errors, so `readout="model"` erred only by relaxing during the readout
  pulse. Its shipped calibration now gives
  each qubit a fitted DRAG `beta_ns` (0.51-0.58 ns; `X` errs by 9.6e-4 in
  simulation, against the published 9e-4, where it erred by 2.7e-3), an `X`
  of its 1-2 transition (`get_calibration(pulse.gates.X, transition=(1, 2))`,
  20 ns, returning `|2>` to `|1>` with probability 0.997-0.999), and readout
  assignment errors with which `readout="model"` reproduces the published
  two-state readout errors (mean assignment fidelity 99.1%, was 99.4%). Each
  value records its kind, source, uncertainty, measurement run and method:
  the file holds 599 values (89 published, 89 digitized, 421 fitted), where
  it held 495. Simulated populations on `transmon_surface17` change in the
  third decimal; `transmon_surface17(calibration=None)` keeps the published
  values alone.
- In the controller export, a branch on an answer (`test` `"answer"`) did
  not say which result of the answer it reads, so for a call taken with
  `result(timeout=...)`, `if ok:` and `if fix[0]:` exported the same
  condition, as did `if fix[0]:` and `if fix[1]:` on any call. Such a
  condition now also has `result`, numbered as in the expression
  `{"answer": i, "result": k}`: for a call with a timeout
  (`result(timeout=...)` or `latency_mode="timeout"`), result 0 is the
  in-time flag and result `k + 1` the function's value `k`. This holds in
  formats `/1` and `/2`, and `controller.unroll()` keeps it. Other
  conditions export as before. `backends.validate` runs one more program,
  `answer_timeout`, which branches on both.
- In the controller export, the answer of a function declared with
  `latency_mode="timeout"` and taken with `result()` had `wait`
  `"bounded"` and a `timeout_ns` of `null`, though its result 0 is the
  in-time flag. It now has `wait` `"timeout"` and the function's
  `timeout_ns`, as a call taken with `result(timeout=...)` does, whose own
  timeout still wins. This holds in formats `/1` and `/2`, and
  `controller.unroll()` keeps it. Other answers export as before.
  `backends.validate` runs one more program, `timeout_function`, which
  branches on such an answer.
- A kernel with more than about 10,000 waits whose length a `float`
  argument sets failed to compile with `RecursionError: maximum recursion
  depth exceeded during compilation` (about 25,000 on Python 3.14 and
  3.15): the check that a call's wait lengths keep the program within the
  most it may last added one term per wait to a single Python expression.
  A rotated surface-code round with a `float` CZ length has about 13,000
  such waits at distance 41. The check now compiles for any number of
  waits and accepts and refuses the same calls as before; waits of one
  length and bounds share one test.

## [0.1.0] - 2026-11-13

The first public release, a research preview: APIs, dialects and numerical
behavior may change in any 0.x release.

### Added

- **Gate calibrations.** `pulse.gates` names the gates a quantum system is
  calibrated for (`X`, `X90`, `Y`, `Y90`, `READOUT`, `CZ`), and
  `get_calibration(gate, transition=(0, 1))` gives each one's pulse: in a
  kernel from a line, `get_drive_line(q).get_calibration(pulse.gates.X)` (a
  `DragPulse`, which `drive(line, pulse)` plays) and
  `get_readout_line(q).get_calibration(pulse.gates.READOUT)` (a
  `ReadoutPulse`); outside one from a handle,
  `dev.qudits[i].get_calibration(...)`, and from a coupling,
  `dev.coupling(a, b).get_calibration(pulse.gates.CZ)` (a `FluxCZ`). A
  qudit's `drive` and `readout` (`MicrowaveDrive`, `DispersiveReadout`)
  hold its physics; its tuned pulses are given as `calibrations=`, by gate,
  and read only through `get_calibration`, as a coupling's are.
  What a quantum system does not publish is derived: an `X` amplitude from the
  pulse's area, `X90` at half of it, `Y` and `Y90` at +90 degrees; an
  unpublished `CZ`, `READOUT` or other transition is an error naming the
  coupling or qudit. A qudit's `metadata` holds the user's own JSON
  values, which the library never reads and which do not make two quantum systems
  differ; system files keep them.
- **Shared lines.** A `DriveLine` or `ReadoutLine` may reach several
  qudits and play several tones at once: a readout feedline is
  `ReadoutLine([0, 1, 2, 3], max_concurrent_tones=4)` (1 by default, at most
  the qudits it reaches). Every line has `qudits`; only a dedicated line has
  `qudit`. A kernel's `get_readout_line(q)` of a qudit on it gives one of
  its tones, for `q`. The compiler checks that a line reaching several
  qudits plays at most `max_concurrent_tones` tones at once (a kernel with
  runtime qudits, when a call binds them); spacing and `max_amplitude`
  apply per tone, waveform memory and `max_frequencies_per_line` per physical
  line. Format `/2` of the controller export lists a shared line once,
  with its `max_concurrent_tones`, and gives each event on it its
  `tone_slot` and `qudits`; format `/1` raises `UnsupportedError` for a
  program on a shared line. System files hold one as `{"qudits": [...],
  "max_concurrent_tones": 4, "delay_ns": 12.0}`. A qudit is reached by at
  most one line of each kind. A program on a delayed shared line does not
  export yet.
- **Pickling, for process pools and caches on disk.** Kernels pickle as
  functions do: by reference, or with cloudpickle (Ray, Dask) by value
  when defined in `__main__`, at the prompt or inside a function. Quantum systems, qudit handles and registers
  pickle as what they were built from and unpickle equal, with equal
  hashes; a process builds each quantum system it unpickles once. A quantum system's
  fingerprint covers its calibration: a quantum system with and without one are
  two quantum systems. A shipped calibration pickles as its name, checked against
  the file shipped where it unpickles; any other, as its values. Programs
  (`ck.lower(...)`, bound or not) pickle as MLIR bytecode stamped with the
  build that compiled them (hashes of its native and Python sources, taken
  when it was built) and the Python, and run unpickled (export, simulate, bind)
  without compiling again; another build refuses them with
  `UnsupportedError`, saying to compile again. Errors pickle with what they
  report, so a process pool re-raises them in the parent as they were
  raised. Quantum systems now compare by content.
- **Controllers with their own clocks.** Kernels count cycles of the
  quantum system's program clock, `pulse.quantum_system(..., clock_ghz=)` (or the
  quantum system's own). A controller whose clock is slower starts and times operations
  only every few program cycles: the scheduler puts each line's operations
  on its controller's grid and a `sync` on the grid its lines share, an
  operation off the grid is a compile error naming the line, its controller
  and the nearest valid lengths, and `pulse.cycles_on(line, n)` rounds a
  length onto the grid (`pulse.Rounding`). Controller programs count their
  own cycles, with `"program_clock_ghz"` written when it differs;
  `ControlSystem.reference_clock_ghz` is checked against every controller
  clock and exported; the simulator samples generated waveforms at the
  line's controller clock.
- **Runtime qudits.** A parameter annotated `Qudit` or `Register` (or
  `list[Qudit]`, `tuple[Qudit, ...]`) is bound at each call, like a float:
  the kernel compiles once and runs on any qudits of a quantum system with the same
  structure, including a recalibrated one. An unannotated qudit parameter
  stays static, whatever the call passes. `list[float]` parameters are runtime vectors. Their
  controller exports carry a `program_id`, shared by every call of one
  compile, and `symbols`: each symbol's value in the call, and per event the
  symbols its qudit, frequency, phase and waveform depend on.
- **Variadic qudits.** A kernel may take `*qudits`:
  `def k(tau: float, *qudits: Qudit)` is called as `k(tau, q0, q1, ...)` or
  `k(tau, *dev.qudits[0:4])`, and from another kernel as
  `k(tau, *qudits)`. In the kernel `qudits` is a register of the call's
  qudits, which indexes, slices, iterates and has a `len()`. Annotated
  `Qudit` it is a runtime register, compiled once per number of qudits;
  unannotated, the kernel is specialized on the qudits. A call may pass
  none. A `*` parameter takes only qudit handles: another annotation is a
  `TypeError` when the kernel is defined, another value one when it is
  called. Kernels still cannot take `**kwargs`. A list or tuple of qudit
  handles is not a register: pass a `Register` (`qs.qudits[0:4]`) or use
  `*qudits`. `kernel.static_parameters` and `kernel.runtime_parameters`
  list parameters in the order of the signature.
- **Control systems.** A quantum system can hold the controllers that play it:
  `pulse.Controller` (a controller's limits, the drive and readout lines it
  plays, its peer latency, clock skew and alignment tolerance, its
  `supports`, `backend`, `model` and `version`) and `pulse.ControlSystem`
  (its controllers, its links by name, each `Link` with the `controllers`
  it reaches). `pulse.quantum_system(quantum_system, control_system=...)` checks
  the wiring against the quantum system and raises one
  `QuantumSystemDefinitionError` listing every problem. `compile(kernel, quantum_system=qpu)` checks every program against
  the system: a line on no controller, each controller's own limits, a
  capability (`pulse.Capability`) a controller lacks, and a call over a link
  that does not reach its controllers fail to compile.
  `compile(kernel, control_system=...)` takes a system, or one `Controller`
  that stands for a system of it alone: what that controller cannot take is
  an error that says to describe the whole system. A system without
  controllers states only its links and lanes. A controller that
  supports `loop` holds a rolled loop's body once against `max_pulses`; one
  that does not plays it unrolled, and its limit error names the loop; unrolling adds pulses,
  not waveform memory, which holds each shape once. `rcp` keeps a branch
  on a readout the `feedback_latency_cycles` of the controllers that read out.
  A line's calibrated delay is part of its declaration,
  `Controller(drive_lines=[DriveLine(5, delay_ns=4.0), 6],
  readout_lines=[ReadoutLine(5, delay_ns=12.0)])` (times or numbers of ns;
  a bare carrier index is an undelayed line), and a control system's
  `reference_clock_ghz` is its shared clock; new delays alone schedule its
  programs again without tracing them. Without a
  declared `clock_ghz`, a quantum system's program clock is its fastest
  controller clock when every other one divides it; clocks that do not
  divide evenly are a `QuantumSystemDefinitionError` that lists the program clocks
  that work, and every controller clock must be a whole multiple of the
  reference clock. Each controller of a system may run its own clock: every
  line is scheduled on its own controller's grid, a `sync` and a wait from
  an answer on the grid all lines share, and a runtime qudit's lines on that
  shared grid too, so one schedule fits any qudits a call binds. Each
  controller's limits, line delays and `rcp` readout latency count its own
  cycles, and `custom()` samples an envelope at the rate of the line that
  plays it, while samples given at another rate fail to compile.
  `export_to_controller(controller=k)` and `export_to_emulator(controller=k)`
  export controller `k`'s part of the program in its own cycles: its
  lines, the events it plays, submits or takes the answer of, its calls and
  its limits, with the program's event and line ids. A program that plays
  the lines of more than one controller needs `controller=`; a controller
  the program does not use exports a valid empty program, and a program
  of calls alone is exported by the controller that makes them.
  `Link(per_controller=True)` (`per_controller = true` in a system file)
  gives each controller its own instance of a link, shared by default:
  its own slots, so its calls do not wait for another controller's, its own
  calls in its export (tagged `"controller"`), its own processor to place
  the link on, and no cross-controller start uncertainty in its budgets. A
  call whose measurements and answer are on different controllers fails to
  compile.
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
  of a system (of each controller, for a per-controller link). Compiling checks each function there: an operation in its
  `unsupported` list (named with its line), `max_ops_per_function`,
  `max_state_bytes_per_function`, `fixed_cost_only`, and its compute time at
  the processor's `cycles_per_op` (`pulse.controller.OpCycles(div=20)`, one
  cycle per class not given) and `clock_ghz` against its
  `max_latency_ns`. It checks each processor's `max_ops_total`,
  `max_state_bytes_total`, `max_calls_per_shot`, `max_bytes_per_shot` and
  `max_concurrent_calls` (default 1) across its functions. Operations come
  in the classes of `pulse.controller.OpClass`; `OpClass.LOOP` is a rolled
  loop's counter and jump per iteration, and a processor with it
  `unsupported` runs loops unrolled, so their size counts unrolled.
  `FunctionCost.ops` is an `OpCounts`, and a built-in's `controller_cost`
  hook returns one. `pulse.controller.ExpressionTerm` names the terms of an
  exported run-time expression. `dispatch_keys={"name": key}` gives the
  functions a processor runs fixed dispatch keys. `scheduled.controller_usage` reports
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
- **System files.** `QuantumSystem.as_dict()` writes a QPU as one mapping, format
  `cudaq-pulse.system/1`, with the keys of `pulse.quantum_system`'s keywords
  (`"name"`, and `"qudits"`, `"couplings"`, ... for one not registered),
  which `pulse.QuantumSystem.from_dict(data)` reads back,
  and `pulse.QuantumSystem.load(path)` from a JSON file, reporting each problem
  with its line. A file written from
  another version of an installed description is noticed once
  (`UpgradeNotice`). `python -m cudaq_pulse.doctor --system FILE` checks a
  system file and whether its controllers' backends are installed.
- **A bound on how long a program runs.** `Controller` takes
  `max_program_duration`, a time or a number of ns, the longest program the
  controller runs. Building the module sums the program op after op, every
  iteration of its loops, and a program longer than it, or than 2**60
  cycles without one, is a `CompilationError` at the op or loop that passes
  it; trip counts multiply without overflow. A wait whose length a float
  sets is counted at its lowest length, and the call checks the length
  again with its value; in an `if` arm, it counts only as far as it makes
  that arm the longer one. A line's `delay_ns` counts too: a delay longer
  than 2**60 cycles, or one that takes the scheduled program past them, is
  a `CompilationError`.

### Changed

- **Long rolled loops are checked without playing every iteration.** The
  controller check walks a rolled loop of more than 64 iterations three
  times when its iterations play alike one period apart, and finds what it
  found with every iteration played, in the same order. Past 10,000
  violations, `ControllerLimitError.violations` and `check()` give a
  sequence that builds each violation when read, with an exact `len()`;
  `str(error)` is unchanged. `ScheduledKernel.schedule` is a list up to 10,000
  events and past that a sequence that builds each event when read, from
  its op and its loops' iterations, so its type is now
  `Sequence[ScheduledEvent]`; `ScheduledKernel.duration` no longer lists the
  events. `classical_report()` takes a long loop without classical calls
  once instead of once per iteration, with the same report.

- **The simulator says which systems it has no model for.** Simulating a
  system that is not a transmon raises `UnsupportedError` (a
  `NotImplementedError`): export and hardware runs work.
- **A simulation step too coarse for its qudits warns.** A `dt_ns` passed to
  `simulate` that takes fewer than 20 steps per period of the fastest
  frequency the simulation holds (a qudit's level spread with three or more
  levels, or the detuning of coupled qudits) issues a `PulseWarning` at the
  caller's line, naming that frequency and a step that resolves it; such a
  step damped the coherence of levels 0 and 2 silently. Examples 03, 14,
  19, 20 and `transmon/transmon_surface17_evolve_gpu.py` take steps that resolve
  their qudits, so two print converged values: example 14's survival at
  depth 20 is 0.952 (was 0.954), and example 20's level-2 branch
  probabilities change (outcomes `(1, 2, 2)` 0.002, was 0.006).
- **The default simulation step resolves many levels.** Without `dt_ns`,
  `simulate` takes 20 steps per period of a qudit's level spread as well as
  of coupled qudits' detuning: 0.025 ns for a transmon with 6 levels and a
  200 MHz anharmonicity, whose levels spread over 2 GHz and which 0.05 ns
  dephased. Wherever 0.05 ns already resolved the simulation (a transmon
  of up to four levels whose anharmonicity is under 333 MHz, as every
  qudit of `transmon_surface17`), the step and the results are unchanged.
- **A simulation that diverges raises instead of returning NaN.** A step
  too coarse for the integrator (4 transmon levels at `dt_ns=1.0`, about
  two steps per period of their 552 MHz spread) made its state overflow,
  and `simulate` returned a density matrix of NaN or of entries near
  1e234. It now raises `ExecutionError`, naming the qudit or coupled pair
  whose frequency the step misses, the step and one that resolves it, with
  a note giving the line that called `simulate`; on the CPU and on the GPU,
  with no cost to a run that does not diverge.
- **Simulation models are read-only.** A model's levels by qudit are fixed
  when it is made, and `SimulationModel` and `TransmonModel` are immutable,
  as built-in types are: setting or deleting one of their attributes raises
  `TypeError`. Subclass them to change what they give a qudit; a subclass
  is not immutable. `simulate` keeps the step it
  resolves for a quantum system with one of them, so a second call on that
  quantum system skips the checks, and drops it when the quantum system is freed.
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
- **What classical functions cost a controller.** `scheduled.controller_usage`
  reports each classical function's operations (read from its body; an
  expression of constants folds, and dividing by a constant is a multiply),
  state in bytes and compute time against `max_latency_ns` on the
  `Processor` that runs it, and each link's calls, slots and bytes per
  shot. `processor.usage(fn)` is one function's cost on a processor on its
  own: its cycles, and its time when the processor has a clock.
- **Faster simulation of long programs.** Idle waits and square pulses in
  their frame are propagated exactly, at a cost independent of their
  length, and driven stretches on registers of up to four levels by RK4
  maps built at once (a 100 us T1 shot: 104 s to under 1 ms). Idle waits of
  coupled qudits are exact too, in the frame that turns with their
  couplings, and on `backend="gpu"` a long idle stretch is propagated
  exactly on the host while the GPU steps the drives around it.
  `simulate(branches="sampled")` (or `pulse.BranchCoverage.SAMPLED`) follows
  only the branches the shots take, so a program that measures many times in
  one shot does not cost time exponential in its measurements. A loop that waits each round on a variable
  or blocking wait costs the same host time per round at any length (example 19's T1 tracker, 30 to 120
  rounds: about 0.55 ms per interval throughout, was 1.5 to 4.7 ms); other programs with calls are scheduled
  again whole after each answer, so their host time per interval grows with the rounds before it. Finding a
  program's next measurement and next call is now linear in its length.

- **Kernels.** `@pulse.kernel` traces a Python function into Pulse-dialect
  MLIR. Calling a kernel with qudit handles compiles it for their quantum system on
  first use, caches the specialization, and runs it; `kernel.lower(...)`
  stops before running, and `pulse.compile(kernel)` compiles ahead of the
  call. `float` parameters stay symbolic, so a parameter sweep compiles once.
  The module with a call's values substituted, which exports and the LLVM
  stage read, is shared by every call with the same values once they
  recur. Kernels call other kernels, loop with `range`, and branch on measurement
  results. A number the arms of an `if` on a measurement set differently
  (`if m: amp = 0.3` / `else: amp = 0.1`) holds the value of the arm that
  ran, as one `arith.select` on the measurement, in `elif` chains, nested
  ifs and rolled loops too. Lowering keeps the select where the quantum system
  takes that amplitude, phase or frequency at run time, so every use reads
  the one value; otherwise, and for durations and waveform shapes, each op
  that reads it runs in both arms of an `if` on the measurement. No quantum system
  takes one at run time yet. Either way the ops wait for the measurement
  and schedule and simulate alike. A name assigned in only one arm, with no
  value before the
  `if`, raises a `CompilationError` naming it and the line of the `if`.
  Kernels take their qudits as parameters; a call without qudit
  handles raises `NoQuantumSystemError` unless the kernel was compiled for a quantum system
  (`pulse.compile(k, quantum_system=dev)`).
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
  bounds, and opens a new segment (below). A length from both, such as
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
  from `cudaq_pulse.ops.__all__`, so Pylance (in VS Code and VS Code-based
  editors), pyright and basedpyright know the operations used by bare name
  inside kernels: no undefined-name warnings, with completion, hover documentation and checked
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
  (`pulse.compile(control_system=...)`); per-kernel schedule policies and
  schedule blocks. `pulse.ScheduledKernel.mlir(level)` returns the module at an IR
  level (`cudaq_pulse.compiler.IRLevel`: `"pulse"`, the scheduled Pulse
  MLIR, `"qop"`, `"cudm"` or `"llvm"`), `ScheduledKernel.schedule` the timed events
  (`cudaq_pulse.compiler.ScheduledEvent`), `ScheduledKernel.duration` how long the
  program runs as a `pulse.Time`, `ScheduledKernel.metrics` the compile
  times (`cudaq_pulse.compiler.CompileMetrics`), and `ScheduledKernel.timeline()` a
  plot (with the `[viz]` extra). `QuantumSystem.qudit(label)` finds a qudit by
  its label and `QuantumSystem.coupling(a, b)` the coupling of two.
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
  and `pulse-schedule-rcp` with the control system's lanes as options (`max-drives`,
  `max-readouts`, `readout-latency`, `switch-penalty`, `alap`).
- **Quantum systems.** `pulse.quantum_system` is the one quantum system getter: it
  describes a machine's physics and a qudit layout, from a registered
  name (`pulse.quantum_system("transmon_surface17")`) or built directly
  (`pulse.quantum_system(name="chip", qudits={...}, couplings=[...],
  model=...)`), with `layout=`, `calibration=`, `measurement_run=`,
  `control_system=` and `clock_ghz=` keywords, or, with no argument, the
  process quantum system; a string is always a name, and
  `QuantumSystem.load` / `QuantumSystem.from_dict` read system files.
  `qs.replace(...)` makes a variant of a quantum system with the same
  keywords, as `dataclasses.replace` does; a `QuantumSystem` passed to
  `pulse.quantum_system` is a `TypeError`. Quantum systems compare by
  content and share compiled kernels, but handles of two quantum systems, even
  equal ones, do not mix in one call or in a kernel compiled for one
  (`QuantumSystemMismatchError`), and an empty `layout` is a
  `QuantumSystemDefinitionError`. The shipped quantum system,
  `"transmon_surface17"`, is a 17-qubit transmon quantum system built from
  Krinner et al., Nature 605, 669 (2022).
  A quantum system's `model` (`cudaq_pulse.quantum_systems.TransmonModel` by default, a
  `SimulationModel` class or instance, or `None`) says how the simulator
  treats its qudits and which operator its couplings are (the exchange
  g (a^dagger b + a b^dagger), in the rotating-wave approximation, for
  `TransmonModel`); each model class declares its levels (3 for
  transmons), and `TransmonModel(levels={0: 4})` sets a qudit's own. No
  label changes the physics: quantum systems have no `architecture`.
  A coupling is `Coupling(qudits=(a, b), strength_hz, calibrations={})`: the
  qudits it couples, in order (two in this release), the strength in Hz of
  the model's operator (g / 2 pi of the exchange), or `None` when it
  is not measured yet, which compiles, schedules and exports while
  `simulate` refuses it by name, and its tuned gates,
  `{pulse.gates.CZ: FluxCZ(duration_ns, buffer_ns=None,
  interaction_frequency_hz=None)}`; the `transmon_surface17` quantum system gives its
  published CZ values, and `cudaq_pulse.quantum_systems.transmon.CZ_LAYERS` the
  (auxiliary, data) pairs of each of its CZ time steps. A quantum system's
  `residual_zz` lists `ResidualZZ(qudits, rate_hz)` entries, a static ZZ
  rate beyond what the couplings produce; the simulator's module carries
  them as typed static terms of its quantum system
  (`static_terms = [#pulse.static_term<zz, qudits = [a, b], rate_hz = ...>]`,
  checked when parsed), and their Hamiltonian term is `static_zz`. Couplings and residual ZZ entries
  are checked when the system is built, all of them at once: each is
  between two distinct qudits of the system (three or more are refused in
  this release), one per pair in either order, with finite strengths and
  rates (zero and negative allowed), a gate's non-empty name, times of 0 ns
  or more and an interaction frequency above 0 Hz. A config or file with
  problems raises `pulse.QuantumSystemDefinitionError` (a `ValueError`) listing all
  of them in `problems`, each a `pulse.Problem` with its path in the config
  (`layout[2]`), the offending value and a fix, and for a file its line and
  column; an unknown key or quantum system name suggests the closest one, and the
  reserved keys (`zones`, `sites`, `moves`, `transport`) are problems too.
  `python -m cudaq_pulse.doctor --system FILE [--json]` runs the same checks.
- **Calibrations and the transmon model.** `cudaq_pulse.quantum_systems.Calibration`
  loads fitted values, each with its kind, source, uncertainty and measurement run,
  apart from the published quantum system, and `pulse.quantum_system` takes `calibration=`
  and `measurement_run=`. `cudaq_pulse.quantum_systems.transmon_model` computes
  flux-tunable transmon levels and charge-coupled pairs from circuit values.
  The shipped `transmon_surface17` calibration has one system-wide charge coupling,
  fitted to the measured `|11>`-`|20>` coupling of X2-D4, so the
  `transmon_surface17` quantum system's exchange (5.67-6.08 MHz) and idle ZZ (-4.5 to -7.3 kHz) depend on
  each pair's frequencies; calibrated qubits convert detunings to and from flux quanta.
  It also holds in-model sudden-net-zero CZ knobs for all 24 pairs (mean
  duration 65.0 +- 4.8 ns against the published 68(7) ns), each with its
  predicted phase and leakage, flags and alternatives.
- **Simulation.** `pulse.simulate` evolves a compiled call with T1, T2 or
  Hahn-echo dephasing, thermal populations, exchange couplings, residual ZZ
  and any number of transmon levels (by default those the quantum system's model
  declares, 3 for a transmon; `levels=` sets them for every qudit or, as a
  dict, per qudit), either with a NumPy reference
  integrator (`backend="cpu"`) or on NVIDIA GPUs through cuDensityMat
  (`backend="gpu"`, Linux, experimental). Programs that measure and branch
  are evolved as an exact ensemble over outcomes. The simulated space holds
  only the qudits the program gets a line on, whatever their quantum system indices
  (a pair at qudits 13 and 3 is a two-mode problem); `SimulationResult.qudits`
  and `ScheduledKernel.qudits` give the quantum system qudit of each mode, and the other
  qudits are classical spectators. `result.level_population(level, qudit)`
  and `result.level_populations(qudit)` give a qudit's level probabilities
  from its reduced state, for any number of levels; `qudit` (a quantum system index
  or handle) may be left out when the result models one qudit, and a qudit
  or level the result does not have is an `InvalidValueError` listing those
  it has. `result.branches` holds one `Branch` per measurement path: its
  `levels`, `probability` and classical `calls`, with `answers(function)`
  for the answers of one function.
- **Classical calls at QPU runtime.** `classical`, a classical real-time
  dialect for co-processing at QPU runtime: pulse programs call GPU, host or
  FPGA services (decoders, estimators, calibration updates) while the QPU
  runs and use their answers in the same shot. `@pulse.classical` declares a
  function from its type hints (`def decode(syndrome: pulse.Bits) -> bool`),
  with a latency model and a link; a list parameter takes its length from
  the call, each length makes a version of the function (`flip__3`), and
  list results take theirs from the body (and the plain Python helpers it
  calls). `pulse.Length` (`typing.Annotated`)
  states a fixed length where one schema is wanted:
  `pulse.Length[pulse.Bits, 8]` (eight measurements read as bits) and
  `pulse.Length[list[int], 8]` (eight numbers). One readout passed where a
  list is taken is a compile error that gives the fix
  (`decode.submit([m])`). A call record or
  report row names the function (`call.function`, `"flip"`), its version
  (`call.compiled_name`, `"flip__3"`) and the key its requests are
  dispatched by (`call.dispatch_key`: the FNV-1a hash of the compiled name,
  or the key a processor's `dispatch_keys` fixes; the call manifest's
  `"dispatch_key"`). Two names that hash to one key on one link are a
  compile error. An external function's stub is a
  declaration whose body is a placeholder for its result (`return float()`,
  `return syndrome[:2]`); its unsized list parameters make versions as for
  `@pulse.classical`. Simulation and emulation check every answer's list
  lengths. `fn.submit(...).result()` calls it in a kernel, with `timeout=`
  (a time such as `350 * pulse.ns`, or a number of nanoseconds, as the
  declaration's latencies take) for a fallback. Type checkers and editors see
  the function's own signature: a wrong argument is an error where it is
  passed, and
  `.submit()` returns a `Pending` of the result's type. Links are named in
  `ControlSystem(links={...})`. Every schedule budgets the worst-case
  latency, holds link slots and reports each call's slack and its hidden and
  exposed latency (`ScheduledKernel.classical_report()`); `pulse-classical-check`
  verifies slots, deadlines and ordering. A wait whose length is an answer,
  and a blocking wait for an answer, are barriers for the lines that depend
  on them (through a `sync`, a measurement or answer, a branch or a shared
  tone): those lines split into segments (`ScheduledKernel.segments()`) whose start
  shifts with the run-time lengths before them, and other lines keep their
  time, so qubits waiting on answers of their own run side by side.
- **Simulating and emulating calls.** `pulse.simulate` binds answers per
  measurement path, with timeouts from the latency model, assignment errors
  and relaxation during readout, IQ points from a dispersive readout model,
  and `quantum_system=` to simulate a drifted quantum system without compiling again.
  `ScheduledKernel.export_to_emulator()` writes the schedule, segments and call manifest
  for a controller; `classical.emulation.emulate` runs a program in virtual
  time against real services, in process or over UDP. Requests and answers
  use the CUDA-Q real-time wire format (`classical.wire`). The
  cuDensityMat runtime keeps a classical handler table per run.
- **CUDA-Q QEC decoders.** `pulse.classical` takes CUDA-Q QEC's real-time
  decoding functions (`cudaq_qec.qec.enqueue_syndromes`, `get_corrections`,
  `reset_decoder`, cudaq-qec 0.6.0 to 0.8.1): a kernel calls a decoder
  configured in CUDA-Q QEC with CUDA-Q QEC's arguments, its requests are the
  decoding server's messages, and simulation and emulation answer with the
  configured decoders through CUDA-Q QEC's simulation library. Example
  `27_realtime_decoding.py`.
- **Calibrating while the QPU runs.** `classical.calibration` (adaptive T1,
  a frequency binary search, three-point decay and phase estimators,
  golden-section and Nelder-Mead steps), `classical.random_index` (an
  on-device random source), `classical.loop` (a parameter store whose values
  programs read at run time, and an estimate log), and the outcome
  probabilities of those experiments with ideal pulses
  (`survival_probability`, `ramsey_probability`, `rabi_probability`), for
  drawing shots without simulating them. Every time an
  estimator takes is a `pulse.Time` or a number of ns
  (`ramsey_detuning(mean_hz, theta, 2 * pulse.us)`); converting it costs a
  controller nothing. Examples
  `19_classical_calls.py` and `20_adaptive_calibration.py`, and the
  "Classical Calls at QPU Runtime" guide.
- **Parameter drift.** `pulse.quantum_system(..., parameter_drift=True,
  parameter_drift_seed=5)` (or `qs.replace(parameter_drift=...)`) builds a
  simulated quantum system whose qudit frequencies, T1 and drive amplitudes
  drift: `True` for the default `ParameterDrift()` on every qudit, a model
  for every qudit, or `{qudit: model}`. Compiling reads only its calibrated
  values (`qs.physical_qudits`), so it has its non-drifting twin's
  fingerprint, compiled programs and exports; `pulse.simulate` plays its
  true values (`qs.true_qudits`, the same `PhysicalQudit` type), which start
  at the calibration. Drift time follows the run clock: each `simulate`
  advances it by `shots_count` times the program's duration plus
  `reset_time` (a new `simulate` keyword, by default five times the longest
  calibrated T1 of the program's qudits), holding the true values for the
  call and splitting a call longer than the drift's correlation time into
  chunks of shots; `qs.advance(20 * pulse.ms)` moves time on between
  experiments and `qs.parameter_drift_time` reads it. The defaults
  (`cudaq_pulse.quantum_systems.drift`: `OrnsteinUhlenbeck` frequency and
  amplitude, `TelegraphT1`) are slow, transmon-sized drift; a model of your
  own is any object with `start(calibrated, rng)` and
  `advance(true, calibrated, seconds, rng)`. Example
  `26_drift_following.py` follows a drifting qudit by swapping programs.
- **Controller backends.** `ScheduledKernel.export_to_controller()` exports a
  scheduled kernel for an out-of-tree controller backend, format
  `cudaq-pulse.controller/1` (`cudaq_pulse.controller.FORMAT`): the lines
  with their roles, and every drive, readout, wait, sync and classical call
  with its exact start and end, waveform and parameters (and, with
  `sample_rate_ghz=`, its samples), tone frequency and frame phase (including
  the phase a detuned tone gains between drives), segment and the
  measurement- or answer-conditioned branches it lies in. What is known only
  at run time is exported as an expression over readouts and answers: a
  waveform parameter (`runtime`, including a value a measurement selects),
  a variable wait's length and bounds, a submission's operands, and, on a
  tone whose frame depends on run time, the frame changes since its
  previous drive (`tone`, `frame_updates`), each with where it is made
  (`start_cycles`), so a backend can compute them where the answer arrives
  and the phase the frame gains from there, and each such tone's line and
  the frame every shot starts it in (`runtime_tones`). A program that
  computes nothing at run time exports exactly as before. Within a version,
  keys and event kinds are only added. A `Controller(...)`, given to
  `pulse.compile(control_system=...)` alone or in its `ControlSystem`,
  states what a controller can play (clock, issue
  spacing, waveform memory at its `sample_rate_ghz`, tones, the largest
  amplitude, readout detectors, feedback latency, answer timeouts, program
  size; the compiler itself limits no amplitude) and makes compilation fail with
  `ControllerLimitError` when a program exceeds it, parametric kernels
  included: a limit that only some values of a float parameter exceed, such
  as a tone shift, a waveform width or a wait length, is checked at the call
  with its values, so a call that lowers also exports. `cudaq_pulse.backends` finds backends installed under the
  `cudaq_pulse.backends` entry-point group; a backend that fails to load, is
  not a `Backend` or reads none of the export formats raises
  `IncompatibleBackendError`, and an unknown name `BackendNotFoundError`,
  both `BackendError`s. `examples/controller_backend/` is a toy backend, and
  the "Writing a Controller Backend" guide walks through one.
- **Controller programs that keep their loops.**
  `export_to_controller(version=2)` exports format `cudaq-pulse.controller/2`,
  which grows with a program's operations rather than its executions: a
  loop whose iterations repeat is one `loop` event holding its body, with
  its trip count, period and each tone's phase step per iteration; each
  distinct waveform is listed once; and what follows from the rest is left
  out. The program writes the controller's `sample_rate_ghz` beside
  `clock_ghz`, so it states its own timing (a waveform of n samples lasts
  n × clock_ghz / sample_rate_ghz cycles); the two may be in any ratio, and
  the export raises rather than round a waveform that is not a whole number
  of samples. A line lists its qudits and an event its tones, and `programs` and
  `bindings` are reserved for programs shared by several controllers.
  `controller.unroll()` turns it into format /1. The export and the
  controller limit checks are native, and the checks reason about a
  repeating loop from its first two iterations and its last, so checking a
  rolled loop costs the same whatever its trip count. On a 320,000-event
  rolled program, compiling with limits takes 25 ms instead of 6 s,
  and the /2 export 2 ms and 14 KB instead of 117 MB; /1 is built on the
  same walk, 3 to 4 times faster than before. Values known only at run
  time are in /2 as in /1, with references as places in the tree and a
  waveform's run-time parameters as expressions in its entry, so a loop of
  calls whose answers set its amplitudes stays one loop.
- **Frames on the timeline.** Every shot starts from the reset frame. The
  four frame changes (`shift_phase`, `set_phase`, `shift_frequency`,
  `set_frequency`) are points on the timeline, with a start and no
  duration in the scheduled IR; the next drive or readout on the tone takes
  each, in a loop the next iteration does. The compiler keeps every change:
  where nothing takes one on some path, the export gives it as a
  `frame_update` event of its own (`Event.FRAME_UPDATE`), with its change
  and the tone's frame after it. A change nothing takes is one where it is
  made, also in a rolled loop's body; the change a loop carries out of its
  last iteration follows the loop; one made before a branch that only one
  arm takes starts the other arm. A loop kept rolled stays rolled, and no
  event, loop or program starts or lasts otherwise for a change.
  `Controller(persistent_frames=True)` declares a controller whose
  frames persist from shot to shot: such changes then have no event, and
  the export's `frame_state` gives the frame each tone ends the shot in.
  `Controller(blocks_on_calls=True)` declares one that plays nothing while
  a classical call is out: each call holds every line from its submission
  until its answer is due, in every schedule, and its exports list the
  limit. In a control system of several controllers, a call holds those
  that block and issue it, read a measurement it sends or play with its
  answer.
  Programs without such changes export as before.
- **Timing budgets.** `@pulse.kernel(max_duration_cycles=n)` fails
  compilation when an execution of the kernel can take more than `n`
  cycles: each call of a kernel with its own `schedule=` (a loop body that
  must fit its cycle, checked once whatever the trip count), or the whole
  program for the kernel compiled. `@pulse.classical(reaction_deadline_ns=...)`
  fails it when a call's answer is due later than that after the end of
  the readouts its request carries, in any iteration. Waits whose length
  is known only at run time count at their highest length.
- **Waveform memory.** `Controller(waveform_memory_samples=65536,
  waveform_memory="shared")` describes a controller's waveform memory in
  plain values, the same in Python and in a TOML description: its size in
  samples, and whether each line has one (`"per_line"`, the default) or one
  serves every line (`"shared"`, holding a waveform the lines share once);
  `WaveformMemoryScope` names the two. In format /2,
  each waveform has an `id` from its content and a `shape_id` equal for
  waveforms that differ only in amplitude and phase, and `parameters` says
  what each `float` parameter reaches, so a planner can tell which values
  change only amplitudes. The links are written in /2 under their new
  names (`send_ns`, `slots`, ...), what the controller computes as a list
  of `processors`, and the
  quantum system's name as `quantum_system`. Two waveforms of
  one shape at `float` amplitudes share memory, and a limit that only some
  `float` values exceed is checked when a call supplies them.
- **Event kinds.** `pulse.Event` names the kinds of event in format /2 for
  what a controller does, whatever the modality: `play`, `readout`,
  `wait`, `sync`, `loop`, `call` and `answer` (/1 keeps `drive`, `readout`
  and `submit`). The export holds the plain strings.
- **Backend validation.** `backends.validate(backend, controller=...)` runs
  the export test corpus, a program per event kind, through a backend: each
  program must be emitted or refused with `IncompatibleBackendError`, and
  unknown event kinds and newer formats refused. A backend that gives its
  emitted events back (`events()`) is checked for timing, order, program
  count and limits, and one with an emulator (`emulate()`) against the
  simulator. The report's `ok` and `str()` say what failed.
- **Public API.** The top level is exactly:
  - kernels and programs: `kernel`, `Kernel`, `compile`, `CompiledKernel`,
    `ScheduledKernel`, `Pass`, `simulate`, `SimulationResult`,
    `BranchCoverage`;
  - quantum and control systems: `quantum_system`, `QuantumSystem`, `Qudit`,
    `Register`, `Controller`, `ControlSystem`, `Processor`, `Capability`,
    `Link`, and `gates`;
  - time: `Time`, `ns`, `us`, `ms`, `s`, `round_duration`, `RoundDirection`;
  - classical calls and exports: `classical`, `Bits`, `Length`, `Event`;
  - errors and warnings: `PulseError`, `Problem`, `InvalidValueError`,
    `InvalidTypeError`, `NotFoundError`, `IndexOutOfRangeError`,
    `UnsupportedError`, `CompilationError`, `TimingError`, `ExecutionError`,
    `InternalError`, `QuantumSystemDefinitionError`, `NoQuantumSystemError`,
    `NoProcessQuantumSystemError`, `QuantumSystemMismatchError`,
    `GpuRuntimeError`, `PulseWarning`, `UpgradeNotice`;
  - modules: `viz`, `controller` and `backends`;

  plus `__version__`.

  `simulate(program, *args, **options)` is the one way to simulate: it
  takes a kernel and its arguments as `compile` takes a kernel, or a
  program compiled already, such as one a worker process receives pickled,
  without compiling it again. A module with run-time float parameters
  (`scheduled.generic`) takes their values as `bind` does; the module of a
  call, from `kernel.lower(*args)` or `bind`, takes none. `bind(values)`
  takes one real number per parameter, in the order of `parameters`; too
  few or too many values are an `InvalidTypeError` naming the parameters.
  A bool, str or other non-number is an `InvalidTypeError`, and a NaN or
  infinity an `InvalidValueError`, naming the parameter where the module
  is used, as for a kernel call's float arguments.
  Every error cudaq-pulse raises on purpose derives from `PulseError` and
  has the built-in base of its kind: `InvalidValueError` is a `ValueError`,
  `InvalidTypeError` a `TypeError`, `NotFoundError` a `KeyError`,
  `IndexOutOfRangeError` an `IndexError`, `UnsupportedError` a
  `NotImplementedError`, and `CompilationError`, `ExecutionError` and
  `InternalError` are `RuntimeError`s. An
  `InternalError` is a bug in cudaq-pulse, and its message asks for a
  report. `Calibration.load` of a calibration that is neither shipped nor a
  file raises `NotFoundError`, and a malformed request or answer on a
  classical channel an `ExecutionError`. Every warning cudaq-pulse issues
  is a `PulseWarning`.
  `cudaq_pulse.compiler`, `cudaq_pulse.runtime.simulate` and
  `cudaq_pulse.quantum_systems.transmon_model` list their public names in
  `__all__`, as the other documented modules do.
  `cudaq_pulse.quantum_systems.UnknownQuantumSystemError` (an unregistered
  quantum system name) and
  `cudaq_pulse.compiler.UnsupportedInterpreterError` (a Python whose bytecode
  the kernel tracer does not know) are public, and every public name is in
  the API reference, errors on their own page.
- Quantum system registration, the shipped quantum systems and calibrations, and the kernel
  tracer's caches are safe to use from several threads on free-threaded
  CPython: each quantum system and calibration is built once.
- **Dialects and tools.** The Pulse, Classical, QOp and CuDensityMat MLIR
  dialects with their lowerings to LLVM. A module describes the quantum
  system it plays on in `pulse.quantum_system`
  (`#pulse.quantum_system<qudit_frequency_hz = [..], level_counts = [..],
  anharmonicity_hz = [..], t1_ns = [..], t2_ns = [..],
  drive_scale_rad_per_ns = [..], static_terms = [..]>`, lists by qudit,
  each field optional): a compiled module gives the frequencies it is
  compiled for, a module lowered for simulation the simulated system,
  whose frequencies may differ from its tones'. It describes the control
  system in `pulse.control_system` (`#pulse.control_system<clock_period_fs
  = 1250000/3, lines = [#pulse.line<drive, qudits = [0], delay_cycles =
  3>], controllers = [#pulse.controller<grid_cycles = 2, ...>]>`): the
  program clock's period in femtoseconds as an exact fraction, each line's
  qudits, tones at once, delay and controller, and each controller's grid,
  outcome latencies, start spread and uncertainty, persistent frames,
  blocking on calls and `max_duration_cycles`. A function's
  `pulse.schedule_state` (`#pulse.schedule_state<unscheduled|timed|placed>`,
  unscheduled when absent) says how far it is scheduled: scheduling marks it
  `timed`, or `placed` when a line delay moved some op; `pulse-verify`
  requires a start and duration on every timed op of a timed or placed
  function, and `pulse-merge-waits` takes only an unscheduled one. The
  tools: `cudaq-pulse-opt`, the MLIR driver, which
  also loads out-of-tree pass plugins and whose `--version` prints the
  cudaq-pulse version and the LLVM it was built with; `python -m cudaq_pulse.doctor`, which
  reports what an installation contains and whether GPU execution works.
  A program in the dialects keeps its meaning through MLIR's standard
  passes, such as `cse`, `canonicalize` and `inline`: an operator program
  ends in `qop.evolve`, which evolves the state, so the passes keep it.
- **Packaging.** One abi3 wheel per platform for CPython 3.12 and newer:
  Linux x86_64 and aarch64 (manylinux_2_28) with the CUDA 12 and CUDA 13
  GPU runtimes, selected by the `[cu12]` or `[cu13]` extra, and macOS arm64
  (compiler only). The wheels link their own MLIR, so they need neither
  CUDA-Q nor an LLVM installation. Free-threaded CPython 3.14 (3.14t) has
  its own wheel per platform, whose extension runs without the GIL.
  `cudaq-pulse-opt` comes in a distribution of its own, `cudaq-pulse-tools`,
  one `py3-none` wheel per platform for every Python, which
  `pip install cudaq-pulse` installs automatically (`cudaq-pulse` requires
  it at the same version). Each
  wheel carries its license files and `THIRD_PARTY_NOTICES`, generated from
  what its binaries link, with each third-party license's text. The wheels
  are on PyPI (`pip install cudaq-pulse`, with `[cu12]`, `[cu13]` or `[gpu]`
  for GPU simulation); each GitHub release also carries them and the sdist.

### Fixed

- Only the first rolled `range()` loop over a name stayed rolled: every
  later loop over the same name (`for _ in range(n)` twice in a row, or an
  inner loop in each iteration of an unrolled outer one) was unrolled, even
  with `unroll_threshold=0`, because the value the name was left holding
  counted as Python state the body changed. Each such loop now stays
  rolled; the program, its schedule and its simulation are as before.
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
  `None` or a string, even one `float()` would parse (`"0.5"`); and a
  duration outside its cycles. A waveform lasts 1 to 2**53
  cycles, and a wait, a flat top (`gaussian_square` width) or a bound 0 to
  2**53, whether given as an `int`, a time, or an `int` or `float` argument
  the kernel is compiled for. A `range()` loop runs at most 2**63 - 1
  times; a longer one is a `CompilationError` at its line, not an
  `OverflowError`. A controller holds at most
  `Controller.max_instructions` operations (unset: no limit); a loop
  that unrolls past them stops tracing as soon as the program does, with a
  `CompilationError` at the loop's line naming the loop, about how many
  operations it unrolls to and the limit. `simulate(shots_count=...)` and `emulate(shots_count=...)`
  take a positive `int` (`InvalidTypeError` or `InvalidValueError`
  otherwise; `simulate`'s default `None` still samples nothing), and a
  qudit index past a quantum system or register is an `IndexOutOfRangeError`, an
  `IndexError`, that says how many qudits it has. A NaN or infinite float argument is an
  `InvalidValueError` naming the parameter wherever the program is used:
  run, simulated, exported or printed. Finding the line of a value the
  native builder rejects traces the kernel again only up to it, and never
  runs its host code again (Python functions it calls, `custom()`
  envelopes, `print`, `next`, impure functions): when host code comes
  first, the error names that call as the line the value is at or after.

[0.1.0]: https://github.com/NVIDIA/cudaq-pulse/releases/tag/v0.1.0
