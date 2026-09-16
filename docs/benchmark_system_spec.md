# Benchmark Portfolio & Canonical Baseline Architecture Specification

**Project Track:** Track 2 — Computational Performance Characterization of Neural ODEs  
**Document Purpose:** Final Freeze Specification for Experimental Foundation  
**Status:** Frozen Specification (No code implemented, no experiments executed)  
**File Location:** `docs/benchmark_system_spec.md`

---

## A. Research-Stage Position

This specification document formalizes the **Experimental Foundation** stage of the project research pipeline:

$$\text{Literature} \longrightarrow \text{Research Gap} \longrightarrow \text{Research Question / Hypothesis} \longrightarrow \mathbf{Experimental\ Foundation} \longrightarrow \text{Controlled Experiment} \longrightarrow \text{Evidence} \longrightarrow \text{Hypothesis Evaluation}$$

* **What this stage accomplishes:** Establishes immutable, controlled experimental parameters—specifically the benchmark dynamical systems and the canonical baseline neural network architecture—so that subsequent experiments isolate numerical and architectural variables without confounding factors.
* **What this stage does NOT do:** This document does NOT present experimental findings, does NOT claim the hypothesis is proven, does NOT develop a novel architecture, and does NOT execute any training, benchmarking, or profiling runs.

---

## B. Frozen Research Question and Hypothesis

The research question and hypothesis formalize the scope defined in the mentor-approved Problem Statement (`docs/problem_statement.pdf`):

### 1. Central Research Question
> **"To what extent does NFE characterize the computational cost of Neural ODEs, and under what conditions do model, solver, numerical, and hardware factors cause deviations between NFE and actual computational cost?"**  
> `[VERIFIED SOURCE: docs/problem_statement.pdf]`

### 2. Central Research Hypothesis
> **"NFE is a useful proxy for Neural ODE computational cost, but its relationship with actual wall-clock runtime systematically varies with model complexity, solver characteristics, stiffness, numerical settings, and hardware/implementation."**  
> `[VERIFIED SOURCE: docs/problem_statement.pdf]`

---

## C. Frozen Benchmark Portfolio

The benchmark portfolio consists of four dynamical systems spanning controlled dynamical and numerical regimes. Ground-truth trajectory files are pre-generated and located in `data/trajectories/`.

```
                      BENCHMARK PORTFOLIO PROGRESSION
┌────────────────────────────────────────┐  ┌────────────────────────────────────────┐
│     NON-STIFF / CANDIDATE NON-STIFF    │  │           STIFF / MULTISCALE           │
├───────────────────┬────────────────────┤  ├───────────────────┬────────────────────┤
│  Lotka-Volterra   │  FitzHugh-Nagumo   │  │   Van der Pol     │    Robertson       │
│      (2D)         │      (2D)          │  │      (2D)         │      (3D)          │
│ Pure Oscillatory  │ Mildly Multiscale  │  │ Tunable Stiffness │ Extreme Multiscale │
│   Reference       │     Dynamics       │  │ (μ ∈ [1, 100])    │ Chemical Kinetics  │
└───────────────────┴────────────────────┘  └───────────────────┴────────────────────┘
```

### 1. Lotka-Volterra (LV) — Non-Stiff Reference System
* **Mathematical Formulation:**
  $$\begin{aligned}
  \frac{dy_1}{dt} &= \alpha y_1 - \beta y_1 y_2 \\
  \frac{dy_2}{dt} &= \delta y_1 y_2 - \gamma y_2
  \end{aligned}$$
  with default parameters: $\alpha = 1.5, \beta = 1.0, \gamma = 3.0, \delta = 1.0$.
* **State Dimension & Horizon:** $D = 2$, $t \in [0.0, 15.0]$, initial condition $y_0 = [1.0, 0.5]^T$.
* **Classification & Status:** **Non-Stiff Reference System** `[VERIFIED SUITABLE]`.
* **Literature Grounding:** Rackauckas et al. (2020, Paper #14) `[VERIFIED SOURCE]`. Standard non-dissipative oscillatory limit cycle. *(Note: Chen et al., 2018 did not evaluate LV).*
* **Measured Project Diagnostics (`phase1_spectral_diagnostics.json`):**
  * Real eigenvalue envelope: $[-2.287, +2.572]$, mean real mode: $+0.021$ `[PROJECT MEASUREMENT]`.
  * Imaginary envelope: $[-4.096, +4.096]$, mean absolute imaginary: $1.054$ `[PROJECT MEASUREMENT]`.
  * Spectral radius range: $[0.663, 4.136]$, mean spectral radius: $2.121$ `[PROJECT MEASUREMENT]`.
  * Fastest spectral timescale: $\tau = 0.242\text{ s}$ `[PROJECT MEASUREMENT]`.
* **Experimental Role:** Provides a zero-stiffness baseline to measure uncontaminated vector-field evaluation time ($T_f$) and baseline solver overhead ($T_{\text{solver}}$) on CPU.

### 2. FitzHugh-Nagumo (FHN) — Candidate Non-Stiff / Mildly Multiscale System
* **Mathematical Formulation:**
  $$\begin{aligned}
  \frac{dv}{dt} &= v - \frac{v^3}{3} - w + I \\
  \frac{dw}{dt} &= \frac{1}{\tau}(v + a - bw)
  \end{aligned}$$
  with default parameters: $a = 0.7, b = 0.8, \tau = 12.5, I = 0.5$.
* **State Dimension & Horizon:** $D = 2$, $t \in [0.0, 50.0]$, initial condition $y_0 = [-1.0, 1.0]^T$.
* **Classification & Status:** **Candidate Non-Stiff / Mildly Multiscale System** `[CONDITIONALLY SUITABLE]`.
* **Literature Grounding:** FitzHugh (1961), Nagumo et al. (1962), Hairer & Wanner (1996) `[VERIFIED SOURCE]`. Excitable membrane model displaying localized slow-fast action potential dynamics.
* **Measured Project Diagnostics (`phase1_spectral_diagnostics.json`):**
  * Real eigenvalue envelope: $[-2.852, +0.916]$, mean real mode: $-0.566$ `[PROJECT MEASUREMENT]`.
  * Imaginary envelope: $[-0.283, +0.283]$, mean absolute imaginary: $0.054$ `[PROJECT MEASUREMENT]`.
  * Spectral radius range: $[0.222, 2.852]$, mean spectral radius: $1.222$ `[PROJECT MEASUREMENT]`.
  * Fastest spectral timescale: $\tau_{\text{fast}} = 0.351\text{ s}$ `[PROJECT MEASUREMENT]`.
* **Explicit Boundary Constraints:** FHN is **NOT** classified as classically stiff. The prior heuristic "$35\times$ stiffness ratio" is ungrounded and retracted. The Phase-1 diagnostics do not by themselves establish classical numerical stiffness or explicit-solver collapse. Solver behavior remains to be empirically measured.
* **Experimental Role:** Tests whether localized slow-fast dynamics induce adaptive step-size refinement overhead on CPU without causing numerical instability.

### 3. Van der Pol (VdP) — Stiff Benchmark with Tunable $\mu$
* **Mathematical Formulation:**
  $$\begin{aligned}
  \frac{dy_1}{dt} &= y_2 \\
  \frac{dy_2}{dt} &= \mu(1 - y_1^2)y_2 - y_1
  \end{aligned}$$
  with parameter $\mu \ge 1.0$.
* **State Dimension & Horizon:** $D = 2$, $t \in [0.0, 30.0]$, initial condition $y_0 = [2.0, 0.0]^T$.
* **Classification & Status:** **Stiff Benchmark with Tunable Stiffness** `[CONDITIONALLY SUITABLE]`.
* **Literature Grounding:** Kim et al. (2021, Paper #4), Caldana & Hesthaven (2025, Paper #15), Hairer & Wanner (1996) `[VERIFIED SOURCE]`. Standard textbook benchmark where increasing $\mu$ increases stiffness as $O(\mu^2)$, forcing explicit solvers into small step sizes.
* **Explicit Boundary Constraints:**
  * Prior crossover measurements ($\mu \approx 25$, 14.4× speedup) in project archives were measured on **classical SciPy ODE integration**, NOT on Neural ODEs `[PROJECT MEASUREMENT — CLASSICAL SCIPY ODE, NOT NEURAL ODE]`.
  * Neural ODE stiffness behavior across $\mu \in [1, 100]$ is `[NOT YET MEASURED]`.
  * Trajectory spectral diagnostics for VdP are `[NOT YET MEASURED]`.
* **Experimental Role:** Provides the primary continuous parameter axis ($\mu$) to test the hypothesis that increasing stiffness causes explicit solver step collapse and decouples NFE from wall-clock runtime on Neural ODEs.

### 4. Robertson (ROBER) — Extreme Stiff / Multiscale Benchmark
* **Mathematical Formulation:**
  $$\begin{aligned}
  \frac{dy_1}{dt} &= -k_1 y_1 + k_3 y_2 y_3 \\
  \frac{dy_2}{dt} &= k_1 y_1 - k_2 y_2^2 - k_3 y_2 y_3 \\
  \frac{dy_3}{dt} &= k_2 y_2^2
  \end{aligned}$$
  with rate constants: $k_1 = 0.04, k_2 = 3 \times 10^7, k_3 = 10^4$.
* **State Dimension & Horizon:** $D = 3$, $t \in [10^{-5}, 10^5]$, initial condition $y_0 = [1.0, 0.0, 0.0]^T$.
* **Classification & Status:** **Extreme Stiff / Multiscale Benchmark** `[CONDITIONALLY SUITABLE]`.
* **Literature Grounding:** Kim et al. (2021, Paper #4), Robertson (1966), Hairer & Wanner (1996) `[VERIFIED SOURCE]`. Autocatalytic chemical reaction kinetics with 9 orders of magnitude rate disparity. Literature reports severe difficulty/failure of explicit adaptive solvers over the long Robertson integration horizon, motivating implicit methods.
* **Explicit Boundary Constraints:** Specific numerical eigenvalue envelopes ($\lambda \approx -10^7$) and "$> 10^9$" stiffness ratios are `[INSUFFICIENT VERIFIED INFORMATION — CANNOT CONCLUDE]` (not computed in project logs). Grounding rests strictly on literature proof (Kim et al., 2021) of explicit solver breakdown and implicit solver necessity.
* **Experimental Role:** Represents the asymptotic extreme where explicit solver NFE is invalid/unbounded, isolating the CPU cost of implicit linear algebra ($T_{\text{solver}}$, Jacobians, LU factorizations).

---

## D. Canonical Baseline Architecture

The Canonical Baseline Neural Network architecture is frozen to serve as the invariant reference across all benchmark systems (formally aligned with the verified Phase-2 implementation per `docs/architecture_decision_record.md`):

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

### Architecture Specifications
1. **Network Type:** Autonomous Multilayer Perceptron (MLP) vector field $f_\theta(z)$.
2. **Input Representation:** Direct state vector $z(t) \in \mathbb{R}^D$ ($D_{\text{in}} = 2$ for $D=2$; $D_{\text{in}} = 3$ for $D=3$). Augmentation dimension is zero (`augment_dim = 0`). The forward pass computes $f_\theta(z)$ without explicit time conditioning.
3. **Hidden Depth:** Exactly 2 hidden layers (3 linear transformations total).
4. **Hidden Width:** Exactly 64 units per hidden layer.
5. **Activation Function:** **Softplus** (`nn.Softplus`, $\beta=1$) applied after Layer 1 and Layer 2.
   * *Rationale:* Smooth $C^\infty$ continuity provides continuously differentiable dynamics; directly matches the verified Phase 2 training and surrogate validation implementation as documented in `docs/architecture_decision_record.md`.
   * *Constraint:* Softplus CPU execution speed relative to other activations is `[NOT YET MEASURED]`.
6. **Output Layer:** Linear transformation $\mathbb{R}^{64} \to \mathbb{R}^D$ representing $\frac{dz}{dt}$. No activation function on the final layer.
7. **Status:** **`[OUR CANONICAL BASELINE]`**. Aligned with the verified Phase 2 implementation per `docs/architecture_decision_record.md`. Synthesizes structural ranges from literature (Kim et al., 2021; Finlay et al., 2020; Caldana & Hesthaven, 2025; Dupont et al., 2019) into a standardized reference model for experimental control; it is not claimed to come identically from any single paper.

---

## E. Parameter-Count Calculation

The parameter count is derived via verified mathematical arithmetic for an autonomous MLP with input dimension $D_{\text{in}} = D$, hidden width $W = 64$, depth $L = 2$ hidden layers, output dimension $D_{\text{out}} = D$, and `augment_dim = 0`:

$$\text{Params}(D) = \underbrace{[D \times 64 + 64]}_{\text{Layer 1}} + \underbrace{[64 \times 64 + 64]}_{\text{Layer 2}} + \underbrace{[64 \times D + D]}_{\text{Layer 3}} = 129 D + 4,224$$

### 1. Two-Dimensional Systems ($D = 2$: Lotka-Volterra, FitzHugh-Nagumo, Van der Pol)
* **Layer 1** ($\text{Linear}(2 \to 64)$):
  * Weights: $2 \times 64 = 128$
  * Biases: $64$
  * Subtotal: **192**
* **Layer 2** ($\text{Linear}(64 \to 64)$):
  * Weights: $64 \times 64 = 4,096$
  * Biases: $64$
  * Subtotal: **4,160**
* **Layer 3** ($\text{Linear}(64 \to 2)$):
  * Weights: $64 \times 2 = 128$
  * Biases: $2$
  * Subtotal: **130**
* **Total Baseline Parameters ($D=2$):** $192 + 4,160 + 130 = \mathbf{4,482\text{ parameters}}$ `[VERIFIED MATHEMATICAL ARITHMETIC & PHASE-2 IMPLEMENTATION]`.

### 2. Three-Dimensional System ($D = 3$: Robertson Baseline Reference)
* **Layer 1** ($\text{Linear}(3 \to 64)$):
  * Weights: $3 \times 64 = 192$
  * Biases: $64$
  * Subtotal: **256**
* **Layer 2** ($\text{Linear}(64 \to 64)$):
  * Weights: $64 \times 64 = 4,096$
  * Biases: $64$
  * Subtotal: **4,160**
* **Layer 3** ($\text{Linear}(64 \to 3)$):
  * Weights: $64 \times 3 = 192$
  * Biases: $3$
  * Subtotal: **195**
* **Total Baseline Parameters ($D=3$):** $256 + 4,160 + 195 = \mathbf{4,611\text{ parameters}}$ `[VERIFIED MATHEMATICAL ARITHMETIC & IMPLEMENTATION]`.

*(Historical Provenance Note: An earlier draft specified a non-autonomous $[z; t] \in \mathbb{R}^{D+1}$ model with GELU yielding 4,546 parameters for $D=2$ and 4,675 for $D=3$. As established in `docs/architecture_decision_record.md`, the actual Phase 2 checkpoints were trained and validated as autonomous Softplus models; the specification has been formally aligned to match).*

---

## F. Controlled Complexity Axis

To address the model-complexity aspect of the central research question without compromising baseline experimental control:

1. **Baseline Invariance:** The canonical baseline architecture ($W=64, L=2$) remains the frozen standard for all cross-system and solver comparisons.
2. **Single-Factor Variation:** Model complexity will later be evaluated by varying **exactly one architectural factor at a time** while holding all other variables constant:
   * **Width Variation (Primary Axis):** Varying hidden width $W$ while holding depth fixed at $L=2$.
   * **Depth Variation (Secondary Axis):** Varying layer count $L$ while holding width fixed at $W=64$.
3. **No Premature Benchmarking:** Specific width or depth values are **NOT** selected or evaluated at this stage. They will be defined and justified in the subsequent experiment-design stage.

---

## G. CPU Experimental Boundary

Hardware execution boundaries are strictly enforced per mentor decision:

1. **Mandatory Scope Constraint:** **HARDWARE = CPU ONLY** throughout the project `[VERIFIED SOURCE: Mentor Decision]`.
   * **Prohibited:** GPU execution, CUDA profiling, mixed-precision (FP16/BF16), CPU-vs-GPU performance comparisons, and GPU kernel analysis.
   * **Rationale:** Preserves feasibility, eliminates confounding hardware factors, and keeps technical focus squarely on numerical solver overhead and stiffness.
2. **Environment Logging Protocol:**
   * Hardware specifications (CPU processor model, physical/logical core counts, cache sizes, dynamic frequency behavior) and software stack details (OS build, PyTorch version, BLAS/LAPACK backend) are `[NOT YET MEASURED]`.
   * These fields will be queried and logged directly from the host machine during environment setup, rather than assumed.
3. **Concurrency Protocol:**
   * **Proposed Profiling Setting:** `torch.set_num_threads(1)` `[OUR PROPOSED DESIGN]`.
   * **Technical Justification:** Restricting PyTorch's intra-op thread pool to 1 thread removes multi-threading context switching and thread-pool synchronization overhead, providing cleaner per-step timing measurements.
   * **Boundary:** Does not guarantee absolute wall-clock determinism due to OS background scheduling and hardware power states.

---

## H. Evidence & Epistemic Status Matrix

| Element | Description | Epistemic Status |
|---|---|---|
| **Research Question & Hypothesis** | NFE characterization across model, solver, and numerical factors | **[VERIFIED SOURCE: `docs/problem_statement.pdf`]** |
| **CPU-Only Hardware Scope** | Mentor-mandated elimination of GPU/CUDA/mixed-precision | **[VERIFIED SOURCE: Mentor Decision]** |
| **Lotka-Volterra Formulation** | 2D predator-prey equations and parameters | **[VERIFIED SOURCE: Rackauckas et al., 2020]** |
| **Lotka-Volterra Diagnostics** | Spectral radius $\rho \in [0.66, 4.14]$, $\tau = 0.242\text{ s}$ | **[PROJECT MEASUREMENT: `phase1_spectral_diagnostics.json`]** |
| **FitzHugh-Nagumo Formulation** | 2D neuronal excitation equations ($\tau = 12.5$) | **[VERIFIED SOURCE: FitzHugh, 1961; Nagumo et al., 1962]** |
| **FitzHugh-Nagumo Diagnostics** | Spectral radius $\rho \in [0.22, 2.85]$, $\tau_{\text{fast}} = 0.351\text{ s}$ | **[PROJECT MEASUREMENT: `phase1_spectral_diagnostics.json`]** |
| **Van der Pol Formulation** | 2D relaxation oscillator with tunable $\mu$ | **[VERIFIED SOURCE: Kim et al., 2021; Caldana & Hesthaven, 2025]** |
| **Van der Pol SciPy Speedup** | 14.4× implicit speedup at $\mu=100$ on analytical ODE | **[PROJECT MEASUREMENT — CLASSICAL SCIPY ODE, NOT NEURAL ODE]** |
| **Van der Pol Spectral Diagnostics** | Analytical Jacobian eigenvalues along trajectory | **[NOT YET MEASURED]** |
| **Robertson Formulation** | 3D chemical kinetics ($k_1=0.04, k_2=3\times 10^7, k_3=10^4$) | **[VERIFIED SOURCE: Kim et al., 2021; Hairer & Wanner, 1996]** |
| **Robertson Eigenvalue Envelope** | Numerical eigenvalues along trajectory ($\lambda \approx -10^7$) | **[INSUFFICIENT VERIFIED INFORMATION — CANNOT CONCLUDE]** |
| **Baseline Architecture Topology** | Autonomous MLP, 2 hidden layers, width 64, Softplus, $z$ input, `augment_dim=0` | **[OUR CANONICAL BASELINE — PHASE-2 ALIGNED (ADR)]** |
| **Baseline Parameter Arithmetic** | $D=2 \to 4,482\text{ params}$; $D=3 \to 4,611\text{ params}$ | **[VERIFIED MATHEMATICAL ARITHMETIC & IMPLEMENTATION]** |
| **Softplus CPU Throughput** | Execution latency of Softplus relative to other activations | **[NOT YET MEASURED]** |
| **Host Machine Hardware Profile** | CPU model, cache hierarchy, OS build, BLAS backend | **[NOT YET MEASURED]** |
| **Hypothesis Testing on Baseline** | NFE vs. runtime characterization on frozen Neural ODE | **[NOT YET MEASURED]** |

---

## I. Explicit "What This Freeze Does NOT Claim"

To prevent premature conclusions or scientific misrepresentation, this specification explicitly declares:

1. **No Claim of Proven Hypothesis:** This document does not claim that NFE has been proven insufficient for Neural ODEs. The hypothesis remains an open research proposition awaiting controlled empirical testing.
2. **No Claim of Neural ODE Stiffness Cliff:** Prior project observations of explicit solver collapse and implicit speedup on Van der Pol ($\mu=100$) were obtained via classical SciPy `solve_ivp` on analytical equations. They do **NOT** constitute proof of Neural ODE behavior.
3. **No Claim of Literature-Standard Architecture:** The canonical baseline ($W=64, L=2$, Softplus) is our synthesized design for experimental control; it is not claimed to be an established standard taken from a single paper.
4. **No Claim of Activation Superiority:** Softplus is chosen for its continuous derivative properties and direct alignment with validated Phase 2 checkpoints; it is not claimed to be faster on CPU than alternative activations.
5. **No Claim of Classical Stiffness for FHN:** FitzHugh-Nagumo is recognized as a mildly multiscale / candidate non-stiff system, not a classically stiff system.
6. **No Claim of Robertson Trajectory Eigenvalues:** Specific numerical eigenvalue figures ($\lambda \approx -10^7$) and "$> 10^9$" stiffness ratios are unverified in project logs and are not treated as factual measurements.
7. **No Claim of Absolute Determinism:** Single-thread CPU execution reduces scheduling contention but does not guarantee absolute timing determinism.

---

## J. Next Experimental Stage

With the experimental foundation frozen, the project advances to:

* **Stage 1: Controlled Experiment Protocol Design**
  * Define exact explicit solvers (e.g., `dopri5`, `tsit5`) and implicit solvers (e.g., `radau`, `bdf`) to evaluate on the frozen baseline.
  * Define tolerance sweep ranges ($\text{rtol}, \text{atol}$) and batch size protocols.
  * Construct the isolated high-resolution timing harness for CPU ($T_{\text{total}}$, $T_f$, $T_{\text{solver}}$).
* **Execution Boundary:** No code implementation, training loops, or benchmark profiling will begin until the Stage 1 Protocol is formally designed, reviewed, and approved.
