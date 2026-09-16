# Stage 1: Controlled Experiment Protocol Design

**Project Track:** Track 2 — Computational Performance Characterization of Neural ODEs  
**Document Purpose:** Formal Experimental Protocol Specification for Stage 1 Controlled Experiments  
**Status:** Audited Protocol Design Specification (Pre-Execution — Zero experiments executed, zero models trained)  
**File Location:** `docs/stage1_experiment_protocol.md`  
**Parent Frozen Specification:** `docs/benchmark_system_spec.md`  
**Problem Statement Reference:** `docs/problem_statement.pdf` / `docs/problem_statement.tex`  
**Dataset Generation Reference:** `data/generate_data.py`  
**Surrogate Validation Reference:** `experiments/logs/surrogate_validation_report.json`  

---

## 1. Epistemic Demarcation & Stage Position

This document formalizes the **Controlled Experiment Protocol Design** stage within the project's research pipeline:

$$\text{Literature} \longrightarrow \text{Research Gap} \longrightarrow \text{Research Question / Hypothesis} \longrightarrow \text{Experimental Foundation (Frozen)} \longrightarrow \mathbf{Stage\ 1\ Protocol\ Design} \longrightarrow \text{Execution} \longrightarrow \text{Hypothesis Evaluation}$$

### Strict Epistemic Boundaries
* **What this document DOES:** Formulates an immutable, reproducible experimental protocol defining primary and follow-up experiments, runtime decomposition, solver statistics, CPU timing controls, accuracy metrics, decision rules, and machine-readable logging schemas.
* **What this document DOES NOT DO:**
  1. This document does **NOT** report experimental findings.
  2. This document does **NOT** execute any benchmark, profiling run, or training loop.
  3. This document does **NOT** generate synthetic, sample, or fabricated results.
  4. This document does **NOT** claim that the research hypothesis is proven, rejected, or partially confirmed.
  5. This document does **NOT** modify or contradict the frozen benchmark specification (`docs/benchmark_system_spec.md`).

---

## 2. Research Gap, Central Research Question, and Hypothesis

### 2.1 The Literature Gap
In foundational Neural ODE literature (Chen et al., 2018; Dupont et al., 2019; Finlay et al., 2020), the **Number of Function Evaluations (NFE)** is routinely cited as the primary or sole scalar proxy for computational complexity. However, reporting aggregate NFE alone does not fully represent practical computational cost because:
1. **Network Evaluation Latency ($T_f$):** A single evaluation of the vector field $f_\theta(z, t)$ has variable computational cost depending on model width, depth, and parameter count.
2. **Solver Housekeeping Overhead ($T_{\text{solver}}$):** Numerical ODE solvers incur non-trivial internal overheads—including error estimation, Butcher tableau intermediate stage evaluations, step-size adaptation logic, and rejected trial steps (Lienen & Günnemann, 2022).
3. **Stiffness Dynamics:** Under stiff dynamics, explicit solvers experience step-size collapse, while implicit solvers require costly Newton-Raphson iterations, Jacobian evaluations, and LU linear solves that NFE counters omit (Kim et al., 2021; Caldana & Hesthaven, 2025).
4. **Numerical Settings:** Tolerances ($\text{rtol}, \text{atol}$) and horizon lengths alter the balance between step adaptation and raw function evaluation throughput.

### 2.2 Central Research Question (Frozen)
> **"To what extent does NFE characterize the computational cost of Neural ODEs, and under what conditions do model, solver, numerical, and hardware factors cause deviations between NFE and actual computational cost?"**  
> `[VERIFIED SOURCE: docs/problem_statement.pdf]`

### 2.3 Central Research Hypothesis (Frozen)
> **"NFE is a useful proxy for Neural ODE computational cost, but its relationship with actual wall-clock runtime systematically varies with model complexity, solver characteristics, stiffness, numerical settings, and hardware/implementation."**  
> `[VERIFIED SOURCE: docs/problem_statement.pdf]`

---

## 3. Experimental Units & Checkpoint Epistemic Status

The experimental units are the four dynamical systems frozen in `docs/benchmark_system_spec.md`. Their classifications, dimensions, integration horizons, and surrogate validation statuses are cataloged below:

| Benchmark System | State Dimension | Integration Horizon | Classification & Epistemic Status | Frozen Experimental Role | Surrogate Checkpoint & Validation Status |
|:---|:---:|:---:|:---|:---|:---|
| **Lotka-Volterra (LV)** | $D=2$ | $t \in [0.0, 15.0]$, $y_0 = [1.0, 0.5]^T$ | **Non-Stiff Reference System**<br>`[VERIFIED SUITABLE]` | Pure oscillatory baseline; provides uncontaminated baseline measurement of isolated network evaluation latency ($T_f$) and explicit solver overhead ($T_{\text{solver}}$) on CPU. | `models/checkpoints/lv_h64.pt`<br>**`[VALIDATED SURROGATE]`**<br>(Passed all 4 tiers in `surrogate_validation_report.json`; NRMSE = 0.0414). |
| **FitzHugh-Nagumo (FHN)** | $D=2$ | $t \in [0.0, 50.0]$, $y_0 = [-1.0, 1.0]^T$ | **Candidate Non-Stiff / Mildly Multiscale**<br>`[CONDITIONALLY SUITABLE]` | Tests whether localized slow-fast excursion dynamics induce adaptive step-size refinement overhead on CPU without classical solver breakdown. *(Explicit boundary: NOT classically stiff; prior 35× claim retracted).* | `models/checkpoints/fhn_h64.pt`<br>**`[VALIDATED SURROGATE]`**<br>(Passed all 4 tiers in `surrogate_validation_report.json`; NRMSE = 0.0020). |
| **Van der Pol (VdP)** | $D=2$ | $t \in [0.0, 30.0]$, $y_0 = [2.0, 0.0]^T$ | **Stiff Benchmark with Tunable $\mu$**<br>`[CONDITIONALLY SUITABLE]` | Continuous parameter axis ($\mu \in [1, 100]$) to test whether physical stiffness induces explicit solver step collapse on Neural ODEs. *(Explicit boundary: prior 14.4× speedup was classical ODE, not Neural ODE; Neural ODE behavior is `[NOT YET MEASURED]`)*. | `models/checkpoints/vdp_mu{1,5,10,25,50,100}_h64.pt`<br>**`[REGIME-SPECIFIC SURROGATES]`**<br>(Validation outcomes vary by $\mu$ in `surrogate_validation_report.json`; see Section 11, Exp 5). |
| **Robertson (ROBER)** | $D=3$ | $t \in [10^{-5}, 10^5]$, $y_0 = [1.0, 0.0, 0.0]^T$ | **Extreme Stiff / Multiscale Benchmark**<br>`[CONDITIONALLY SUITABLE]` | Asymptotic extreme where explicit solvers fail/timeout, isolating CPU implicit linear-algebra costs ($T_{\text{solver}}$, Jacobians, LU factorizations). *(Explicit boundary: specific eigenvalue envelopes are `[INSUFFICIENT VERIFIED INFORMATION — CANNOT CONCLUDE]`)*. | `models/checkpoints/robertson_h64.pt`<br>**`[UNVALIDATED CHECKPOINT]`**<br>(Failed Phase 2 validation: `VALIDATION_TIMEOUT` at 5.031s, `NRMSE_THRESHOLD_EXCEEDED`; see Section 11, Exp 6). |

---

## 4. Canonical Baseline Neural Network Architecture

The vector field $f_\theta(z)$ is parameterized by the single canonical architecture frozen in `docs/benchmark_system_spec.md` (formally aligned with the verified Phase-2 implementation per `docs/architecture_decision_record.md`):

```
State Vector z(t) ∈ R^D
           │
           ▼
   Linear(D ──► 64)
           │
           ▼
        Softplus
           │
           ▼
   Linear(64 ──► 64)
           │
           ▼
        Softplus
           │
           ▼
   Linear(64 ──► D)
           │
           ▼
   Output dz/dt ∈ R^D
```

### Architectural Specifications
1. **Network Topology:** Autonomous Multilayer Perceptron (MLP) vector field $f_\theta(z)$.
2. **Input Representation:** Direct state vector $z(t) \in \mathbb{R}^D$ ($D_{\text{in}} = 2$ for $D=2$; $D_{\text{in}} = 3$ for $D=3$). Augmentation dimension is zero (`augment_dim = 0`). The forward pass computes $f_\theta(z)$ without explicit time conditioning.
3. **Hidden Structure:** Exactly 2 hidden layers ($L=2$), each with width $W=64$ hidden units.
4. **Activation Function:** **Softplus** (`nn.Softplus`, $\beta=1$) applied after hidden layer 1 and hidden layer 2.
   * *Rationale:* Smooth $C^\infty$ continuity provides continuously differentiable dynamics, avoiding gradient and Jacobian step discontinuities; directly aligns with the verified Phase 2 training and surrogate validation implementation as documented in `docs/architecture_decision_record.md`.
5. **Output Layer:** Linear projection $\mathbb{R}^{64} \to \mathbb{R}^D$ without output activation.
6. **Frozen Parameter Counts:**
   $$\text{Params}(D) = [D \times 64 + 64] + [64 \times 64 + 64] + [64 \times D + D] = 129 D + 4,224$$
   * **$D=2$ (LV, FHN, VdP):** Exactly **4,482 parameters** `[VERIFIED MATHEMATICAL ARITHMETIC & PHASE-2 IMPLEMENTATION]`.
   * **$D=3$ (Robertson Baseline Reference):** Exactly **4,611 parameters** `[VERIFIED MATHEMATICAL ARITHMETIC & IMPLEMENTATION]`.

*(Historical Provenance Note: An earlier draft documented a theoretical $[z; t] \in \mathbb{R}^{D+1}$ model with GELU yielding 4,546 parameters for $D=2$. Pre-flight audit revealed the actual Phase 2 checkpoints were trained and validated as autonomous Softplus models; the specification has been formally aligned to match per `docs/architecture_decision_record.md`).*

---

## 5. Primary Experiment 1: Baseline NFE-vs-Runtime Characterization

### 5.1 Objective
Experiment 1 establishes the empirical baseline relationship between NFE and wall-clock runtime on CPU under uncontaminated, non-stiff conditions. It tests whether NFE linearly predicts wall-clock runtime when model complexity, solver algorithm, hardware, and tolerances are held strictly invariant.

### 5.2 Controlled Multi-Point NFE Variation Protocol
To provide a statistically rigorous evaluation of NFE-vs-runtime (rather than comparing only two isolated system points), Experiment 1 introduces **controlled NFE variation** through two complementary, non-confounding mechanisms:

#### A. Adaptive Solver Horizon Scaling (`dopri5`)
For each non-stiff benchmark system, all model weights, tolerances, solver settings, and initial conditions remain strictly invariant. The integration horizon $T_{\text{span}} = [0, T_k]$ is stepped across 10 evenly spaced sub-intervals:
* **Lotka-Volterra Horizon Ladder (10 points):**
  $$T_k \in \{1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0\}$$
* **FitzHugh-Nagumo Horizon Ladder (10 points):**
  $$T_k \in \{5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0\}$$

*Scientific Control Rationale:* In non-stiff oscillatory systems, the average step size chosen by an adaptive solver is determined by the local dynamics and tolerances. Therefore, extending the integration horizon systematically scales the required number of integration steps and total NFE across a wide, continuous dynamic range while keeping $f_\theta$, tolerances, and hardware identical.

#### B. Deterministic Fixed-Step NFE Ladder (`rk4`)
To provide an absolute mathematical reference where NFE is known *a priori* and completely decoupled from adaptive step controller logic:
* Using the classical 4th-order Runge-Kutta solver (`rk4`) on the full horizon ($t \in [0, 15]$ for LV), the number of uniform steps is scaled across 6 discrete levels:
  $$N_{\text{steps}} \in \{25, 50, 100, 200, 400, 800\}$$
* Since `rk4` evaluates exactly 4 stages per step, this establishes a deterministic integer NFE ladder:
  $$\text{NFE} \in \{100, 200, 400, 800, 1600, 3200\}$$

**Total Multi-Point Sample Size:** $10\text{ (LV adaptive)} + 10\text{ (FHN adaptive)} + 6\text{ (LV deterministic fixed)} = \mathbf{26\text{ distinct controlled points}}$.

#### C. Disaggregated Regression & Evaluation Policy (Strict Anti-Pooling Rule)
* **Strict Prohibition of Pooled Regression:** The 26 observations must **NOT** be pooled into a single homogeneous regression model. Combining observations across disparate dynamical systems (Lotka-Volterra vs. FitzHugh-Nagumo) or across different solver mechanics (adaptive `dopri5` vs. fixed `rk4`) would introduce severe confounding from unmodeled system-specific properties and violate independent, identically distributed (i.i.d.) assumptions.
* **Separated Regression Series:** Statistical characterization of the NFE-vs-runtime relationship is conducted independently for each ladder:
  1. **Series 1A (Lotka-Volterra Adaptive `dopri5`):** Independently fits the linear model:
     $$T_{\text{total, LV}} = \beta_1^{\text{LV}} \cdot \text{NFE} + \beta_0^{\text{LV}}$$
     across the 10 horizon points $T_k \in [1.5, 15.0]$. Yields system-specific slope $\beta_1^{\text{LV}}$ (marginal runtime cost per function evaluation), intercept $\beta_0^{\text{LV}}$ (fixed solver initialization overhead), and coefficient of determination $R_{\text{LV}}^2$.
  2. **Series 1B (FitzHugh-Nagumo Adaptive `dopri5`):** Independently fits:
     $$T_{\text{total, FHN}} = \beta_1^{\text{FHN}} \cdot \text{NFE} + \beta_0^{\text{FHN}}$$
     across the 10 horizon points $T_k \in [5.0, 50.0]$. Yields $\beta_1^{\text{FHN}}$, $\beta_0^{\text{FHN}}$, and $R_{\text{FHN}}^2$.
  3. **Series 1C (Lotka-Volterra Deterministic `rk4`):** Independently fits:
     $$T_{\text{total, RK4}} = \beta_1^{\text{RK4}} \cdot \text{NFE} + \beta_0^{\text{RK4}}$$
     across the 6 step-count points as an exact zero-adaptation mathematical reference baseline.
* **Cross-Series Comparative Analysis:** Compare slopes ($\beta_1^{\text{LV}}$ vs. $\beta_1^{\text{FHN}}$ vs. $\beta_1^{\text{RK4}}$) and intercepts. If $\beta_1^{\text{LV}} \ne \beta_1^{\text{FHN}}$, this provides empirical evidence relevant to the hypothesis regarding whether the relationship between NFE and wall-clock runtime varies across dynamical systems even when model architecture ($W=64, L=2$), tolerances, and CPU hardware are held strictly identical; it does not by itself prove or fully evaluate the central hypothesis across all dimensions.

### 5.3 Controlled Parameter Matrix (Experiment 1)

| Parameter | Specification | Scientific Control Justification |
|:---|:---|:---|
| **Dynamical Systems** | Lotka-Volterra ($D=2$), FitzHugh-Nagumo ($D=2$) | Evaluates standard oscillatory vs. mildly multiscale dynamics without classical stiffness. |
| **Model Architecture** | Canonical Baseline ($W=64, L=2$, Softplus, autonomous $z$) | Holds vector-field computational complexity ($T_f$) constant. |
| **Checkpoints** | `models/checkpoints/lv_h64.pt`, `models/checkpoints/fhn_h64.pt` | Uses verified, validated surrogate weights representing learned dynamics. |
| **Solvers** | `dopri5` (canonical adaptive explicit Runge-Kutta 5(4)) and `rk4` (fixed-step reference) | Chen et al. (2018) reference solver vs. deterministic stage baseline. |
| **Numerical Tolerances** | $\text{rtol} = 1.0 \times 10^{-5}$, $\text{atol} = 1.0 \times 10^{-7}$ (adaptive runs) | Standard literature baseline default. |
| **Evaluation Points ($N_{\text{eval}}$)** | LV: 150 points ($\Delta t = 0.10$); FHN: 200 points ($\Delta t = 0.25$) | Matches ground-truth trajectory sampling. |
| **Initial Conditions ($y_0$)** | LV: $[1.0, 0.5]^T$; FHN: $[-1.0, 1.0]^T$ | Frozen reference initial conditions. |
| **Hardware Execution** | **CPU ONLY** | Strict mentor constraint; zero GPU/CUDA execution. |
| **PyTorch Threading** | `torch.set_num_threads(1)` | Restricts intra-op parallelism to 1 thread, eliminating OpenMP context switching and thread pool jitter. |
| **Random Seed** | 42 (`torch.manual_seed(42)`, `np.random.seed(42)`) | Ensures deterministic initialization. |
| **Warm-Up Executions** | 20 unrecorded integration cycles | Warms CPU instruction/data caches; eliminates dynamic linker and bytecode overhead. |
| **Measured Repetitions** | 15 independent recorded integration cycles per horizon point | Sufficient sample size for robust non-parametric summary statistics (median, IQR). |
| **High-Precision Timer** | `time.perf_counter_ns()` | Monotonic nanosecond timer avoiding system clock adjustments. |
| **Watchdog Timeout** | 30.0 seconds per integration run | Hard safety limit. Timed-out runs are marked as right-censored/terminated observations ($T > 30.0\text{ s}$) and strictly excluded from standard OLS regressions. |

---

## 6. Runtime Decomposition Methodology

### 6.1 Mathematical Formulation & Terminology
Total Neural ODE integration wall-clock time on CPU decomposes into:

$$T_{\text{total}} = \widehat{T}_{\text{isolated\_net}} + \widehat{T}_{\text{solver}} = (\text{NFE} \times T_f) + \widehat{T}_{\text{solver}}$$

Where:
* $T_{\text{total}}$: **Direct measured wall-clock time** required to integrate the trajectory over $[t_0, t_1]$ (milliseconds).
* $\text{NFE}$: **Direct empirical integer count** of forward passes through $f_\theta$ executed by the solver during integration.
* $T_f$: **Direct isolated latency measurement** of a single forward evaluation of the neural network vector field $f_\theta(z, t)$ (microseconds).
* $\widehat{T}_{\text{isolated\_net}} = \text{NFE} \times T_f$: **Isolated-latency-based estimate** of cumulative neural-network evaluation time (milliseconds).
  * *Epistemic Note:* This is a derived synthetic estimate based on multiplying the integer NFE by the standalone bench-tested latency $T_f$; it is **not** an in-situ measured neural-network execution time during the solve.
* $\widehat{T}_{\text{solver}} = T_{\text{total}} - \widehat{T}_{\text{isolated\_net}}$: **Estimated solver overhead** (milliseconds), capturing internal solver housekeeping:
  * Butcher tableau stage multiplications and intermediate buffer allocations.
  * Local truncation error norm computation ($\text{err} = \|z_{\text{embedded}} - z_{\text{higher}}\|$).
  * Adaptive step-size controller calculations ($h_{\text{new}} = h \cdot \min(\text{facmax}, \max(\text{facmin}, \text{fac} \cdot (\text{tol} / \text{err})^\alpha))$).
  * Step acceptance/rejection branch logic.
  * Continuous output polynomial interpolation (dense output at $t_{\text{eval}}$).

### 6.2 Measurement Protocol for Isolated Latency ($T_f$)

#### A. Strict Separation by Checkpoint & System (No Cross-System Pooling)
* **Strict Independence Rule:** $T_f$ must be measured **separately and independently for each trained checkpoint and system** ($T_{f, \text{LV}}$ for `lv_h64.pt`, $T_{f, \text{FHN}}$ for `fhn_h64.pt`, and $T_{f, \text{VdP},\mu}$ for each `vdp_mu{X}_h64.pt`).
* **Scientific Justification:** Although the canonical baseline topology ($W=64, L=2, D=2$) is structurally identical between Lotka-Volterra and FitzHugh-Nagumo, $T_f$ cannot be assumed identical. Differences in trained weight parameter distributions, activation function operating ranges (Softplus transition vs. linear regimes), and floating-point denormal handling on CPU can introduce statistically significant execution latency differences.
* **System-Specific Decomposition:** Each system's runtime decomposition is strictly anchored to its own measured $T_{f, \text{system}}$:
  $$\widehat{T}_{\text{solver, LV}} = T_{\text{total, LV}} - (\text{NFE} \times T_{f, \text{LV}})$$
  $$\widehat{T}_{\text{solver, FHN}} = T_{\text{total, FHN}} - (\text{NFE} \times T_{f, \text{FHN}})$$
  Assuming a single pooled or universal $T_f$ across distinct systems is strictly prohibited.

#### B. Measurement Execution Protocol
For each checkpoint/system under evaluation:
1. Sample 50 representative state coordinates $z \in \mathbb{R}^D$ and timestamps $t \in \mathbb{R}$ from that specific system's pre-generated ground-truth trajectory.
2. Execute 50 warm-up forward evaluations on that specific checkpoint: `_ = model.func(t, z)`.
3. Measure 1,000 independent forward evaluations using `time.perf_counter_ns()`.
4. Compute median $T_{f, \text{system}}$ ($\mu\text{s}$) and Interquartile Range ($\text{IQR}_{T_f}$).
5. **Dimensionally Valid Stability Criterion:** Verify that the relative dispersion satisfies:
   $$\frac{\text{IQR}_{T_f}}{T_{f, \text{median}}} \le 0.05 \quad (5\%)$$
   *(Dimensionless ratio: both numerator and denominator have units of microseconds).*

### 6.3 Estimated Solver Overhead Metrics
1. **Estimated Absolute Solver Overhead:**
   $$\widehat{T}_{\text{solver}} = T_{\text{total}} - (\text{NFE} \times T_{f, \text{system}})$$
2. **Estimated Solver Overhead Ratio:**
   $$\widehat{\Omega}_{\text{solver}} = \frac{\widehat{T}_{\text{solver}}}{T_{\text{total}}} = 1 - \frac{\text{NFE} \times T_{f, \text{system}}}{T_{\text{total}}}$$

### 6.4 Epistemic Guardrail: Measured vs. Estimated
* $T_{\text{total}}$ is a **DIRECT HIGH-RESOLUTION MEASUREMENT**.
* $T_f$ is an **ISOLATED DIRECT MEASUREMENT**.
* $\text{NFE}$ is an **EXACT EMPIRICAL INTEGER COUNT**.
* $\widehat{T}_{\text{isolated\_net}}$ and $\widehat{T}_{\text{solver}}$ are **DERIVED ESTIMATES**, not independent in-situ measurements.
* Actual in-situ neural execution time inside the solver loop cannot be measured without inserting high-resolution timer calls (`time.perf_counter`) around every call to $f_\theta$, which would introduce substantial instrumentation overhead, disrupt CPU instruction cache lines, and distort solver timing.
* If measurement noise causes $T_{\text{total}} < \text{NFE} \times T_f$ (e.g., due to CPU instruction cache locality during tight loop unrolling), $\widehat{T}_{\text{solver}}$ is reported as $0.0\text{ ms}$ with the exact signed residual recorded in `notes`. **Synthetic extrapolation or fabricated scaling is strictly prohibited.**

---

## 7. Solver Statistics: Implementation Capabilities & Backend Boundaries

Different solver implementations expose different internal counters. To prevent ungrounded or hallucinated metrics, available capabilities are formally cataloged:

| Metric | `torchdiffeq` Backend (Chen et al.) | `scipy.integrate.solve_ivp` Backend | Epistemic Status & Handling |
|:---|:---:|:---:|:---|
| **Total NFE (`nfe`)** | Exists (via wrapper call counter) | Exists (`res.nfev`) | **`[DIRECT MEASUREMENT]`** |
| **Wall-Clock Time ($T_{\text{total}}$)** | Measured via `perf_counter` | Measured via `perf_counter` | **`[DIRECT MEASUREMENT]`** |
| **Accepted Steps Count ($N_{\text{acc}}$)** | **NOT EXPOSED** | Exists (`len(res.t) - 1` for internal steps) | In `torchdiffeq`: Mark **`[UNAVAILABLE IN BACKEND]`**. Do NOT fabricate. |
| **Rejected Steps Count ($N_{\text{rej}}$)** | **NOT EXPOSED** | Exists in some solvers | In `torchdiffeq`: Mark **`[UNAVAILABLE IN BACKEND]`**. Do NOT estimate. |
| **Step-size Sequence ($h_n$)** | **NOT EXPOSED** | Exists (`np.diff(res.t)`) | In `torchdiffeq`: Mark **`[UNAVAILABLE IN BACKEND]`**. |
| **Jacobian Evaluations (`njev`)** | **N/A** (Explicit solvers only) | Exists (`res.njev` in Radau/BDF) | Applicable only to implicit solvers in SciPy. |
| **LU Decompositions (`nlu`)** | **N/A** (Explicit solvers only) | Exists (`res.nlu` in Radau/BDF) | Applicable only to implicit solvers in SciPy. |

---

## 8. CPU Execution Boundary & Reproducibility Protocol

Per mentor decision, hardware is strictly constrained:

### 8.1 Hardware Constraints
1. **Mandatory Scope:** Strictly **CPU-ONLY**.
2. **Prohibited Technologies:** CUDA, ROCm, GPU acceleration, Tensor Cores, Apple Metal, and mixed-precision execution (FP16/BF16).
3. **Execution Role:** Hardware is treated as a recorded environment descriptor for scientific reproducibility, **never as an independent experimental variable**.

### 8.2 Concurrency & Execution Control
1. **PyTorch Concurrency:**
   ```python
   torch.set_num_threads(1)
   torch.set_num_interop_threads(1)
   ```
   *Rationale:* Eliminates multi-threading race conditions, core migration, OpenMP barrier synchronization latency, and false sharing in L1/L2 caches.
2. **Determinism Control:**
   ```python
   torch.manual_seed(42)
   np.random.seed(42)
   ```

### 8.3 Environment Metadata Capture
Every execution session must programmatically capture and log:
* Host OS platform, kernel release, and build number (`platform.platform()`).
* CPU processor architecture, model name, physical core count, and logical thread count (`psutil`, `platform.processor()`).
* Python executable path and exact version (`sys.version`).
* PyTorch version (`torch.__version__`).
* `torchdiffeq` version (`torchdiffeq.__version__`).
* NumPy and SciPy versions (`np.__version__`, `scipy.__version__`).
* PyTorch BLAS backend (`torch.__config__.show()`, MKL / OpenBLAS availability).

---

## 9. Repetitions, Summary Statistics, and Outlier Policy

Because CPU execution is subject to operating system background tasks, thread scheduling interrupts, and hardware dynamic power states, single-run measurements are scientifically invalid.

### 9.1 Measurement Protocol
1. **Warm-up:** Exactly 20 unrecorded integration cycles to establish steady-state CPU cache locality and warm PyTorch JIT pathways.
2. **Repetitions:** Exactly 15 recorded integration cycles per experimental condition.
3. **Primary Representative Metric:** **Median** ($T_{\text{median}}$).
   * *Rationale:* The median is robust against positive skew introduced by infrequent OS context-switch interruptions.
4. **Primary Dispersion Metric:** **Interquartile Range** ($\text{IQR} = Q_3 - Q_1$).
5. **Secondary Metrics:** Arithmetic mean ($T_{\text{mean}}$) and standard deviation ($T_{\text{std}}$) recorded for completeness.

### 9.2 Anti-Cherry-Picking Rule
* **Strict Prohibition:** Reporting $\min(T_{\text{total}})$ as the primary result is **strictly forbidden**. Minimum-latency reporting selectively ignores solver overhead variance and systematically distorts computational cost comparisons.
* **Outlier Identification:** An individual measurement $T_i$ is flagged as an invalid outlier if:
  $$T_i > Q_3 + 3.0 \times \text{IQR}$$
  Flagged runs are documented in the raw execution log with the OS interrupt note, but excluded from reported median/IQR figures.

---

## 10. Numerical Accuracy & Ground-Truth Verification

A Neural ODE integration that executes with low NFE or rapid runtime is scientifically meaningless if the resulting trajectory is numerically inaccurate or unstable.

### 10.1 Ground-Truth Trajectories & Verified Generation Tolerances
Trajectories are compared against the pre-generated reference solutions stored in `data/trajectories/*_ground_truth.pt`. As verified in `data/generate_data.py` (lines 47–56), these ground-truth trajectories were computed using:
* **Solver Algorithm:** SciPy `solve_ivp` with method **`Radau`** (5th-order implicit Runge-Kutta, L-stable).
* **Analytical Jacobian:** Supplied directly (`jac=sys_obj.jacobian`).
* **Verified Tolerances:** $\mathbf{\text{rtol} = 1.0 \times 10^{-10}, \quad \text{atol} = 1.0 \times 10^{-12}}$ `[VERIFIED IN data/generate_data.py]`.

### 10.2 Accuracy Metrics
For evaluated trajectory $z(t) \in \mathbb{R}^{N \times D}$ and ground truth $y_{\text{GT}}(t) \in \mathbb{R}^{N \times D}$:
1. **Mean Squared Error (MSE):**
   $$\text{MSE} = \frac{1}{N \cdot D} \sum_{i=1}^N \sum_{j=1}^D \left( z_{i,j} - y_{\text{GT}, i, j} \right)^2$$
2. **Relative $L_2$ Error:**
   $$\text{Rel-}L_2 = \frac{\|z - y_{\text{GT}}\|_F}{\|y_{\text{GT}}\|_F}$$
3. **Maximum Absolute Coordinate Error ($L_\infty$):**
   $$L_\infty = \max_{1 \le i \le N} \max_{1 \le j \le D} |z_{i,j} - y_{\text{GT}, i, j}|$$

### 10.3 Distinction: Solver Tolerance vs. Model Fit Error
* **Solver Local Truncation Error:** Controlled by numerical solver tolerances ($\text{rtol}, \text{atol}$) relative to the vector field $f_\theta$.
* **Surrogate Generalization Error:** The discrepancy between the learned neural dynamics $f_\theta$ and the true analytical ODE.
* *Control Policy:* Accuracy control checks that the solver faithfully integrates the learned surrogate $f_\theta$ without numerical divergence, and verifies that the surrogate maintains acceptable fidelity to ground truth ($\text{MSE} \le 1.0 \times 10^{-2}$ for LV/FHN; $\text{MSE} \le 5.0 \times 10^{-2}$ for VdP $\mu \le 10$).

---

## 11. Controlled Follow-Up Experiment Matrix

Following the completion of Experiment 1, follow-up experiments systematically isolate one independent variable at a time. All other parameters remain anchored to the Canonical Baseline.

```
                                 STAGE 1 EXPERIMENTAL SEQUENCE
                                 
      ┌────────────────────────────────────────────────────────────────────────┐
      │ Experiment 1: Baseline NFE vs. Runtime (LV, FHN, Canonical Baseline)   │
      │ 26 Controlled Points: 10 Horizon LV + 10 Horizon FHN + 6 RK4 Steps    │
      └───────────────────────────────────┬────────────────────────────────────┘
                                          │
                  ┌───────────────────────┼───────────────────────┐
                  ▼                       ▼                       ▼
      ┌───────────────────────┐ ┌───────────────────┐ ┌────────────────────────┐
      │ Exp 2: Solver Effect  │ │ Exp 3: Complexity │ │ Exp 4: Tolerance Sweep │
      │ 2A: Algorithms (diffeq)│ │ 3A: Width (16-256)│ │ rtol (1e-2 to 1e-9)    │
      │ 2B: Backend (torch/sci)│ │ 3B: Depth (1-5)   │ │ atol (1e-4 to 1e-11)   │
      │ 2C: Implicit (scipy)  │ └───────────────────┘ └────────────────────────┘
      └───────────────────────┘           │
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
      ┌───────────────────────────────────────┐ ┌──────────────────────────────┐
      │ Exp 5: Stiffness Scaling (VdP μ Axis) │ │ Exp 6: Extreme Stiff (ROBER) │
      │ 5A: Analytical ODE Baseline (μ 1-100) │ │ Stress-Test on Unvalidated   │
      │ 5B: Neural Surrogate Regime (μ 1-100) │ │ robertson_h64 + ODE Control  │
      └───────────────────────────────────────┘ └──────────────────────────────┘
```

### Experiment 2: Decoupled Solver Algorithm vs. Implementation Backend
To prevent conflating mathematical solver algorithms with language runtime or data structure implementations:

* **Experiment 2A: Pure Solver Algorithm Effect (Fixed Backend: `torchdiffeq`)**
  * **Independent Variable:** Algorithm $\in \{\text{`dopri5'}, \text{`tsit5'}, \text{`rk4'}, \text{`midpoint'}, \text{`euler'}\}$.
  * **Fixed Invariants:** Software backend (`torchdiffeq`), Canonical baseline model (`lv_h64.pt`), system (LV, $t \in [0, 15]$), tolerances ($10^{-5}, 10^{-7}$ for adaptive), thread count (1).
  * **Target:** Isolates Butcher tableau size and step-adaptation logic strictly within identical PyTorch tensor execution pathways.
* **Experiment 2B: Implementation & Backend Overhead (Cross-Framework Benchmark)**
  * **Independent Variable:** Software backend $\in \{\text{`torchdiffeq'}, \text{`scipy'}\}$.
  * **Algorithm Invariant:** Dormand-Prince 5(4) (`dopri5` in `torchdiffeq` vs. `RK45` in `scipy`).
  * **Fixed Invariants:** Identical system (LV), tolerances ($10^{-5}, 10^{-7}$), initial condition, thread count (1).
  * **Target:** Directly quantifies the framework/runtime overhead (Python tensor dispatch in PyTorch vs. NumPy/C array dispatch in SciPy).
* **Experiment 2C: Stiff Implicit Solvers (Fixed Backend: `scipy`)**
  * **Independent Variable:** Implicit algorithm $\in \{\text{`Radau'}, \text{`BDF'}\}$.
  * **Target:** Quantifies the computational cost of implicit linear algebra (Newton iterations, Jacobian evaluations, LU factorizations) within the compiled SciPy backend.

### Experiment 3: Model Complexity Scaling (Single Axis Variation)
Evaluates whether increasing neural network evaluation latency ($T_f$) alters the relative dominance of NFE vs. solver overhead:
* **Experiment 3A (Width Scaling — Primary Axis):**
  * **Independent Variable:** Hidden layer width $W \in \{16, 32, 64, 128, 256\}$.
  * **Fixed Invariants:** Depth fixed at $L=2$, Softplus activation, `dopri5`, $\text{rtol}=10^{-5}, \text{atol}=10^{-7}$, system (LV), thread count (1).
  * **Theoretical Expectation:** $T_f$ scales as $O(W^2)$, progressively increasing the $\widehat{T}_{\text{isolated\_net}} = \text{NFE} \times T_f$ fraction of total runtime while NFE remains relatively stable.
* **Experiment 3B (Depth Scaling — Secondary Axis):**
  * **Independent Variable:** Number of hidden layers $L \in \{1, 2, 3, 4, 5\}$.
  * **Fixed Invariants:** Width fixed at $W=64$, Softplus activation, `dopri5`, tolerances, system (LV), thread count (1).
  * **Theoretical Expectation:** $T_f$ scales as $O(L)$, evaluating sequential layer overhead without expanding layer width.

### Experiment 4: Numerical Tolerance Sweep
* **Independent Variable:** Tolerance pairs:
  $$\{(\text{rtol}, \text{atol})\} = \{(10^{-2}, 10^{-4}), (10^{-3}, 10^{-5}), (10^{-5}, 10^{-7}), (10^{-7}, 10^{-9}), (10^{-9}, 10^{-11})\}$$
* **Fixed Invariants:** Canonical baseline architecture ($W=64, L=2$, Softplus), systems (LV, FHN), solver (`dopri5`), thread count (1).
* **Research Target:** Tests whether NFE scales according to theoretical asymptotic convergence orders ($NFE \propto \text{tol}^{-1/p}$) and whether tighter tolerances alter the estimated ratio $\widehat{\Omega}_{\text{solver}} = \widehat{T}_{\text{solver}} / T_{\text{total}}$.

### Experiment 5: Stiffness Scaling along Van der Pol $\mu$ Progression (Confound-Free Protocol)
*Confound Identification:* Loading different checkpoint weights $\theta_\mu$ across $\mu$ changes both the physical dynamics and the neural network parameter values, creating a dual-confound. To isolate variables cleanly:
* **Experiment 5A: Pure Physical Stiffness Baseline (Analytical VdP ODE)**
  * **Independent Variable:** Parameter $\mu \in \{1, 2, 5, 10, 25, 50, 100\}$.
  * **Fixed Invariants:** Analytical ODE formulation (`VanDerPol(mu=mu)`), solvers (`dopri5` vs. `Radau`), tolerances ($10^{-5}, 10^{-7}$), thread count (1).
  * **Target:** Isolates the pure mathematical effect of stiffness $O(\mu^2)$ on NFE and runtime collapse with zero neural network weight confounding.
* **Experiment 5B: Neural Surrogate Regime Evaluation**
  * **Independent Variable:** Dynamical regime $\mu \in \{1, 2, 5, 10, 25, 50, 100\}$ using corresponding checkpoints `vdp_mu{X}_h64.pt`.
  * **Control Protocol:** $T_{f, \mu}$ is bench-tested individually for each checkpoint $\theta_\mu$ prior to integration runs.
  * **Target:** Evaluates whether the learned Neural ODE vector field exhibits the same step collapse and NFE explosion observed in the analytical system, while factoring out individual checkpoint latency differences via $\widehat{T}_{\text{solver}} = T_{\text{total}} - \text{NFE} \times T_{f, \mu}$.

### Experiment 6: Robertson Extreme Multiscale Kinetics & Failure Stress-Testing
*Epistemic Reality:* The historical checkpoint `robertson_h64.pt` is **unvalidated** (`VALIDATION_TIMEOUT` at 5.031s, `NRMSE_THRESHOLD_EXCEEDED`). It is strictly evaluated as an unvalidated stress test:
* **Part 6A (Explicit Solver Failure Characterization):**
  * Evaluate `robertson_h64.pt` under `dopri5` with watchdog timer ($T_{\text{max}} = 30.0\text{ s}$) and step limit ($100,000$).
  * Target: Measure whether the unvalidated checkpoint triggers `STEP_LIMIT_EXCEEDED` or `VALIDATION_TIMEOUT`, empirically confirming the explicit solver breakdown boundary on extreme multiscale problems.
* **Part 6B (Implicit Reference Baseline on Analytical Robertson):**
  * Evaluate the classical Robertson system (`Robertson()` in `benchmarks/systems.py`, $D=3$, $t \in [10^{-5}, 10^5]$) using implicit solvers (`Radau`, `BDF` in SciPy).
  * Target: Directly measure the computational cost of implicit linear algebra ($T_{\text{solver}}$, Jacobian evaluations `njev`, LU decompositions `nlu`) on CPU over an extreme multiscale horizon.

---

## 12. Experimental Decision Rules & Failure Handling

To guarantee scientific objectivity, all outcome classifications are governed by unambiguous, quantitative decision criteria:

| Outcome Classification | Exact Quantitative Criterion | Logging Action & Statistical Handling |
|:---|:---|:---|
| **`SUCCESS`** | Integration terminates within $T_{\text{max}} = 30.0\text{ s}$, $\text{NFE} \le 100,000$, zero NaN/Inf, and $\text{MSE} \le \text{Threshold}$. | Record all performance metrics; valid uncensored run included in linear regressions and median/IQR calculations. |
| **`VALIDATION_TIMEOUT`** | Integration elapsed time exceeds $T_{\text{max}} = 30.0\text{ s}$ via watchdog timer. | Terminate process immediately; record `status = VALIDATION_TIMEOUT`, `is_right_censored = true`, `censoring_threshold_s = 30.0`.<br>**Strict Statistical Rule:** Must be treated as a **right-censored survival / terminated observation** ($T > 30.0\text{ s}$, $\text{NFE} > \text{recorded}$). **STRICTLY EXCLUDED from ordinary least squares (OLS) regressions, correlation coefficients, and uncensored mean/median runtime calculations.** Treating a timed-out run as an ordinary 30-second measurement artificially truncates the distribution and severely biases fitted slopes downward. |
| **`STEP_LIMIT_EXCEEDED`** | Integration reaches $\text{NFE} > 100,000$ without reaching $t_1$. | Abort solve; record `status = STEP_LIMIT_EXCEEDED`; marks explicit solver collapse under stiffness. Treated as right-censored observation. |
| **`NUMERICAL_DIVERGENCE`** | Any state component contains `NaN`, `Inf`, or absolute value $|z_i(t)| > 10^6$. | Abort solve; record `status = NUMERICAL_DIVERGENCE`; log last valid time step $t_{\text{div}}$. |
| **`INSUFFICIENT_ACCURACY`** | Solve completes successfully, but trajectory $\text{MSE} > \text{Threshold}$ against ground truth. | Record run with `status = INSUFFICIENT_ACCURACY`; invalid for runtime comparison. |
| **`MEASUREMENT_ANOMALY`** | Individual run runtime exceeds $Q_3 + 3.0 \times \text{IQR}$ due to external OS interrupt. | Flag as anomaly in raw log; exclude from summary statistics; repeat cycle. |

---

## 13. Machine-Readable Experiment Logging Schema

All experimental outputs must be recorded in structured JSON (per experiment series) and CSV (tabular row per run). The unified schema requires the following fields:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "NeuralODE_Experiment_Log_Schema",
  "type": "object",
  "required": [
    "experiment_id",
    "timestamp_iso",
    "research_stage",
    "system_name",
    "state_dimension",
    "model_architecture",
    "parameter_count",
    "solver_name",
    "solver_backend",
    "rtol",
    "atol",
    "thread_count",
    "nfe_median",
    "runtime_ms_median",
    "tf_us_median",
    "estimated_solver_overhead_ms",
    "estimated_solver_overhead_ratio",
    "trajectory_mse",
    "status"
  ],
  "properties": {
    "experiment_id": { "type": "string", "example": "EXP1-LV-DOPRI5-H15.0" },
    "timestamp_iso": { "type": "string", "format": "date-time" },
    "research_stage": { "type": "string", "enum": ["STAGE_1_BASELINE", "STAGE_1_FOLLOWUP"] },
    "system_name": { "type": "string", "enum": ["Lotka-Volterra", "FitzHugh-Nagumo", "VanDerPol", "Robertson"] },
    "system_dimension": { "type": "integer", "enum": [2, 3] },
    "system_parameters": { "type": "object" },
    "horizon_start": { "type": "number" },
    "horizon_end": { "type": "number" },
    "num_eval_points": { "type": "integer" },
    "model_architecture": { "type": "string", "example": "MLP[D->64->64->D]_Softplus" },
    "hidden_width": { "type": "integer", "example": 64 },
    "hidden_depth": { "type": "integer", "example": 2 },
    "activation": { "type": "string", "example": "Softplus" },
    "parameter_count": { "type": "integer", "example": 4482 },
    "checkpoint_path": { "type": "string" },
    "checkpoint_validation_status": { "type": "string", "enum": ["VALIDATED", "UNVALIDATED", "ANALYTICAL_ODE"] },
    "solver_name": { "type": "string", "example": "dopri5" },
    "solver_backend": { "type": "string", "enum": ["torchdiffeq", "scipy"] },
    "solver_type": { "type": "string", "enum": ["explicit_adaptive", "explicit_fixed", "implicit_adaptive"] },
    "rtol": { "type": "number", "example": 1e-5 },
    "atol": { "type": "number", "example": 1e-7 },
    "random_seed": { "type": "integer", "example": 42 },
    "hardware_device": { "type": "string", "const": "CPU" },
    "cpu_model": { "type": "string" },
    "cpu_cores_physical": { "type": "integer" },
    "cpu_cores_logical": { "type": "integer" },
    "os_platform": { "type": "string" },
    "python_version": { "type": "string" },
    "pytorch_version": { "type": "string" },
    "torchdiffeq_version": { "type": "string" },
    "thread_count": { "type": "integer", "const": 1 },
    "warmup_runs": { "type": "integer", "example": 20 },
    "measured_runs": { "type": "integer", "example": 15 },
    "nfe_median": { "type": "integer" },
    "nfe_mean": { "type": "number" },
    "nfe_std": { "type": "number" },
    "runtime_ms_median": { "type": "number" },
    "runtime_ms_iqr": { "type": "number" },
    "runtime_ms_mean": { "type": "number" },
    "runtime_ms_std": { "type": "number" },
    "raw_runtimes_ms": { "type": "array", "items": { "type": "number" } },
    "tf_us_median": { "type": "number" },
    "tf_us_iqr": { "type": "number" },
    "isolated_net_eval_est_ms": { "type": "number" },
    "estimated_solver_overhead_ms": { "type": "number" },
    "estimated_solver_overhead_ratio": { "type": "number" },
    "trajectory_mse": { "type": "number" },
    "trajectory_linf": { "type": "number" },
    "status": { "type": "string", "enum": ["SUCCESS", "VALIDATION_TIMEOUT", "STEP_LIMIT_EXCEEDED", "NUMERICAL_DIVERGENCE", "INSUFFICIENT_ACCURACY"] },
    "is_right_censored": { "type": "boolean", "default": false },
    "censoring_threshold_s": { "type": "number", "example": 30.0 },
    "failure_reason": { "type": "string" },
    "notes": { "type": "string" }
  }
}
```

---

## 14. Resource Budget & Computational Efficiency

Because this investigation is strictly CPU-based, the experimental protocol is optimized to minimize execution time while preserving rigorous statistical validity:

1. **Zero Retraining Overhead:** All models use pre-trained surrogate checkpoints (`models/checkpoints/*.pt`). No backpropagation, adjoint gradient passes, or parameter updates occur during benchmarking.
2. **Lean Sample Sizing:** Exactly 20 warm-up runs and 15 measured repetitions provide stable median and IQR estimates without wasteful computation.
3. **Execution Time Budget:**
   * **Isolated $T_f$ Measurement (1,000 passes):** $\approx 0.05$–$0.15\text{ s}$ CPU time per architecture.
   * **Experiment 1 (LV + FHN Horizon Ladders + RK4 Step Ladder = 26 points $\times$ 15 runs):** $\approx 390$ solves $\approx 30$–$60\text{ seconds}$ total CPU time.
   * **Experiment 2 (Solver Comparison, 8 solvers $\times$ 15 runs):** $\approx 1$–$2\text{ minutes}$ total CPU time.
   * **Experiment 3 (Width/Depth Sweeps, 10 configurations $\times$ 15 runs):** $\approx 2$–$4\text{ minutes}$ total CPU time.
   * **Experiment 4 (Tolerance Sweeps, 5 pairs $\times$ 15 runs):** $\approx 1$–$3\text{ minutes}$ total CPU time.
   * **Experiment 5 & 6 (Stiffness & Robertson with 30s timeout cap):** Max budget $\le 10\text{ minutes}$ total CPU time.
   * **Total Stage 1 Protocol Execution Budget:** $\le 20\text{ minutes}$ on a standard multi-core x86_64 workstation.

---

## 15. Explicit Mapping of Experiments to Research Gap and Questions

Every planned experiment is explicitly mapped to the mentor-approved Problem Statement and relevant literature:

| Planned Experiment | Specific Factor Isolated | Frozen Literature Gap Addressed | Relevant Literature Reference |
|:---|:---|:---|:---|
| **Experiment 1** (Baseline NFE vs. Runtime Multi-Point Ladder) | Empirical scalar proportionality across 26 controlled NFE levels in non-stiff systems (adaptive horizons + fixed steps). | Evaluates whether NFE linearly tracks wall-clock time under baseline conditions across a wide dynamic range. | Chen et al. (2018), Finlay et al. (2020) |
| **Experiment 2A** (Solver Algorithm Effect) | Numerical solver algorithm housekeeping, Butcher tableau size, and step adaptation overhead within a fixed backend (`torchdiffeq`). | Directly measures $\widehat{T}_{\text{solver}}$ differences between explicit solvers (`dopri5` vs. `tsit5` vs. `rk4`) at identical model weights. | Lienen & Günnemann (2022) |
| **Experiment 2B** (Backend / Implementation Effect) | Software implementation and memory layout overhead (`torchdiffeq` Tensor dispatch vs. `scipy` NumPy/C dispatch). | Decouples implementation-level overheads from numerical algorithm differences under identical mathematical solver formulations (`dopri5` vs. `RK45`). | Lienen & Günnemann (2022) |
| **Experiment 2C** (Implicit Stiff Solvers) | Computational cost of implicit linear algebra (Newton steps, Jacobians, LU factorizations) in SciPy. | Measures implicit solver overhead under stiff conditions. | Kim et al. (2021) |
| **Experiment 3A** (Model Width Scaling) | Neural vector-field complexity ($T_f$) across parameter counts ($W \in [16, 256]$). | Quantifies the shift from solver-dominated runtime ($\widehat{\Omega}_{\text{solver}} \gg 0$) to network-dominated runtime ($\widehat{T}_{\text{isolated\_net}} \gg \widehat{T}_{\text{solver}}$) as $T_f$ scales. | Finlay et al. (2020), Dupont et al. (2019) |
| **Experiment 3B** (Model Depth Scaling) | Sequential layer latency across depth ($L \in [1, 5]$). | Measures sequential layer evaluation latency impact on $T_f$ and overall runtime proportionality. | Finlay et al. (2020) |
| **Experiment 4** (Tolerance Scaling) | Numerical tolerances ($\text{rtol} \in [10^{-2}, 10^{-9}]$, $\text{atol} \in [10^{-4}, 10^{-11}]$). | Evaluates whether step-adaptation overhead fraction increases as tolerances tighten, and tests theoretical step scaling. | Chen et al. (2018), Lienen & Günnemann (2022) |
| **Experiment 5A** (Analytical VdP Stiffness) | Pure physical stiffness parameter $\mu \in [1, 100]$ on the analytical ODE. | Establishes the unconfounded ground-truth baseline of how stiffness forces explicit step collapse and creates implicit speedup. | Hairer & Wanner (1996) |
| **Experiment 5B** (Neural VdP Stiffness) | Dynamical regime $\mu \in [1, 100]$ across corresponding neural surrogates with isolated $T_{f, \mu}$ compensation. | Tests whether physical stiffness induces explicit solver step collapse on Neural ODEs without confounding checkpoint latency. | Kim et al. (2021), Caldana & Hesthaven (2025) |
| **Experiment 6A** (Robertson Explicit Failure Stress-Test) | Explicit solver breakdown under extreme multiscale reaction kinetics using unvalidated `robertson_h64.pt`. | Characterizes explicit solver failure modes (`STEP_LIMIT_EXCEEDED`, `VALIDATION_TIMEOUT`) on extreme chemical kinetics. | Kim et al. (2021) |
| **Experiment 6B** (Robertson Implicit Baseline) | Extreme 9-orders-of-magnitude rate disparity integrated via implicit solvers (`Radau`, `BDF`) on analytical Robertson ODE. | Isolates implicit linear-algebra costs ($T_{\text{solver}}$, Jacobians, LU factorizations) on CPU. | Kim et al. (2021), Hairer & Wanner (1996) |

---

## 16. Protocol Approval & Next Operational Step

### Protocol Status
This audited protocol completes the revised design requirements for **Stage 1: Controlled Experiment Protocol Design**.

### Next Operational Step (Upon Approval)
Once this protocol specification is reviewed and approved:
1. Construct the Stage 1 execution harness (`experiments/stage1_baseline_experiment.py`) implementing the exact logging schema, timing controls, and watchdog timeouts specified herein.
2. Execute **Experiment 1 only** (the 26-point baseline LV, FHN, and RK4 characterization).
3. Record measured JSON and CSV results, update evidence indices, and compare observed NFE-runtime proportionality against the baseline hypothesis.
4. **Execution Boundary:** No code implementation or benchmark execution will proceed without explicit authorization.
