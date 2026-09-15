# Reproducibility Protocol & Environment Specification

**Project Track:** Track 2 — Computational Performance Characterization of Neural ODEs  
**Status:** Operational Protocol (Pre-Experiment)

---

## 1. Experimental Scope & Boundary
* **Hardware Execution:** Strictly **CPU-ONLY** throughout all experimental stages.
  * No GPU, CUDA, or mixed-precision (FP16/BF16) experiments.
  * Hardware is recorded solely as an environment descriptor for scientific reproducibility, not as an experimental variable.
* **Target Platforms:** x86_64 CPU workstation environment running PyTorch on Windows / Linux.

---

## 2. Software Stack & Dependencies
* **Core Runtime:**
  * Python: `3.11.x`
  * PyTorch: `2.x` (CPU build with MKL / OpenBLAS backend)
  * ODE Solvers: `torchdiffeq` (Chen et al.), `scipy.integrate` (classical reference)
  * Numerical & Scientific: `numpy`, `scipy`, `matplotlib`
* **Dependency Freezing:**
  * Project dependencies will be version-pinned in `requirements.txt` upon Stage 1 deployment.

---

## 3. Concurrency & Profiling Policy
* **Thread Affinity:**
  * Primary timing runs: `torch.set_num_threads(1)` to eliminate multi-threading context switching and thread pool synchronization jitter.
  * Reference environment runs: default multi-thread configuration (`torch.get_num_threads()`) logged.
* **Timing Instrumentation:**
  * High-resolution wall-clock timer: `time.perf_counter_ns()`.
  * Warmup iterations: Minimum 20–50 unrecorded forward passes prior to timing loops to eliminate cold-start cache misses and JIT trace overheads.
  * Statistical repetitions: Minimum 10 independent runs per experimental configuration with mean, standard deviation, and median recorded.

---

## 4. Metadata Logging Standard (Supervisor Rule Compliant)
Every experimental run must output structured metadata logging:
1. `experiment_id`: Unique hierarchical ID (e.g., `EXP1-LV-DOPRI5-RTOL1E-5`).
2. `timestamp`: ISO 8601 UTC timestamp.
3. `system_name`: Dynamical system evaluated (LV, FHN, VdP, ROBER).
4. `solver`: ODE solver algorithm and library (`dopri5`, `rk4`, `radau`, etc.).
5. `tolerances`: Explicit `rtol` and `atol`.
6. `architecture`: Topology string, depth $L$, width $W$, activation function.
7. `num_params`: Exact parameter count.
8. `thread_count`: Threads allocated to PyTorch.
9. `nfe`: Forward and backward function evaluations reported by solver.
10. `wall_clock_ms`: Total elapsed integration time.
11. `t_f_ms`: Isolated network forward-pass evaluation time.
12. `t_solver_ms`: Calculated solver overhead ($T_{\text{total}} - \text{NFE} \times T_f$).
13. `status`: Outcome code (`SUCCESS`, `STEP_LIMIT_EXCEEDED`, `VALIDATION_TIMEOUT`, `FAILED`).

---

## 5. Random Seed & Determinism Protocol
* Seed initialization: `torch.manual_seed(42)`, `np.random.seed(42)`.
* Deterministic flags: `torch.use_deterministic_algorithms(True)` where applicable.
* Boundary note: While thread count and seeds are fixed, absolute wall-clock runtime contains small hardware jitter due to OS background interrupts and dynamic CPU power states.
