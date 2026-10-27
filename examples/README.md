# cudaq-pulse examples

Run any example with the package installed:

```bash
python examples/01_single_qubit_rabi.py
```

Every example except `transmon/krinner_evolve_gpu.py` runs on a CPU-only
machine, Linux or macOS, and the test suite runs each of them. **Simulates**
means the example also evolves the program with the NumPy reference
(`pulse.simulate`); the others compile kernels and print their MLIR,
schedules or metrics. `cudaq-pulse-opt` is installed with the package.

| Example | What it shows | Runs on |
| --- | --- | --- |
| [`01_single_qubit_rabi.py`](01_single_qubit_rabi.py) | Rabi oscillation: an amplitude sweep that compiles once | CPU, simulates |
| [`02_two_qubit_cross_resonance.py`](02_two_qubit_cross_resonance.py) | CNOT by echoed cross-resonance | CPU |
| [`03_t1_t2_dissipator.py`](03_t1_t2_dissipator.py) | T1 decay against its prediction; Ramsey and echo kernels | CPU, simulates |
| [`04_hahn_echo.py`](04_hahn_echo.py) | Hahn and CPMG echo sequences | CPU |
| [`05_waveform_gallery.py`](05_waveform_gallery.py) | Every waveform constructor and the waveform algebra | CPU |
| [`06_phase_frequency_control.py`](06_phase_frequency_control.py) | Phase and frequency control of tones | CPU |
| [`07_multi_qubit_sync.py`](07_multi_qubit_sync.py) | `sync` barriers and registers of qudit handles | CPU |
| [`08_readout_and_branching.py`](08_readout_and_branching.py) | Readout and branching on a measurement result | CPU, simulates |
| [`09_compilation_pipeline.py`](09_compilation_pipeline.py) | `pulse.compile`, `lower()` and the compiler stages | CPU |
| [`10_scheduling_comparison.py`](10_scheduling_comparison.py) | ASAP, ALAP and resource-constrained schedules side by side | CPU |
| [`12_error_detection.py`](12_error_detection.py) | `pulse-verify` diagnostics for malformed Pulse MLIR | CPU |
| [`13_dynamical_decoupling.py`](13_dynamical_decoupling.py) | XY4, Uhrig and CPMG dynamical decoupling | CPU |
| [`14_randomized_benchmarking.py`](14_randomized_benchmarking.py) | Randomized benchmarking: random Cliffords and their inverse | CPU, simulates |
| [`16_visualization.py`](16_visualization.py) | Schedule timelines; plots with the `[viz]` extra | CPU |
| [`17_ghz_state_prep.py`](17_ghz_state_prep.py) | GHZ state preparation with cross-resonance CNOTs | CPU |
| [`18_per_kernel_schedules.py`](18_per_kernel_schedules.py) | Per-kernel ASAP and ALAP policies in a stabilizer cycle | CPU |
| [`19_classical_calls.py`](19_classical_calls.py) | Classical calls at QPU runtime: decoding with feed-forward, slack and latency reports | CPU, simulates |
| [`20_adaptive_calibration.py`](20_adaptive_calibration.py) | Adaptive calibration with variable waits against a drifting device | CPU, simulates |
| [`21_calibration_protocol.py`](21_calibration_protocol.py) | A millisecond calibration chain on the controller, after Marciniak et al., arXiv:2602.11912 (2026), with adaptive T1 estimation after Berritta et al., arXiv:2506.09576 (2025) | CPU, simulates |
| [`custom_passes/python_pass.py`](custom_passes/python_pass.py) | A schedule check written in Python | CPU |
| [`custom_passes/cpp/`](custom_passes/cpp) | An MLIR pass plugin for `cudaq-pulse-opt --load-pass-plugin` | CPU (builds with CMake) |
| [`controller_backend/toy_controller/__init__.py`](controller_backend/toy_controller/__init__.py) | A toy controller backend found by its entry point: limits, `emit` and `run` | CPU |
| [`transmon/krinner_rabi.py`](transmon/krinner_rabi.py) | Rabi drive on the 17-qubit transmon device | CPU |
| [`transmon/krinner_full_pipeline.py`](transmon/krinner_full_pipeline.py) | Each stage of the lowering: Pulse, QOp and LLVM IR | CPU |
| [`transmon/krinner_surface_code_cycle.py`](transmon/krinner_surface_code_cycle.py) | The timeline of one surface-code cycle | CPU |
| [`transmon/krinner_evolve_gpu.py`](transmon/krinner_evolve_gpu.py) | `pulse.simulate(..., backend="gpu")` | GPU: Linux, an NVIDIA GPU and the `[cu12]` or `[cu13]` extra |
| [`neutral_atom/rydberg_blockade_demo.py`](neutral_atom/rydberg_blockade_demo.py) | Rydberg blockade between two atoms | CPU |
| [`neutral_atom/rydberg_chain_adiabatic.py`](neutral_atom/rydberg_chain_adiabatic.py) | A detuning sweep on a 1D Rydberg chain | CPU |

The neutral-atom targets compile only; simulating them is not supported yet.
Numbers 11 and 15 were removed and are not reused. The
[examples page](https://nvidia.github.io/cudaq-pulse/examples.html) of the
documentation has the same index.
