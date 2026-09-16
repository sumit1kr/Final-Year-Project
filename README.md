# Computational Performance Characterization of Neural ODEs

**Academic Context:** Final Year Project (FYP) — Track 2: Computational Characterization  
**Department:** Department of Computer Science and Engineering  
**Institution:** Indian Institute of Information Technology, Design and Manufacturing (IIITDM), Kancheepuram  
**Guide:** Dr. Noor Mahammad Shaik  
**Candidate:** Sumit Kumar (Roll No.: CS23B2008)  
**Date:** September 2026  

---

## 1. Project Overview

Neural Ordinary Differential Equations (Neural ODEs) model continuous-time dynamics by parameterizing the instantaneous rate of change with a neural network vector field $\frac{dy(t)}{dt} = f_\theta(y(t), t)$. While scientific machine learning literature almost universally reports the computational cost of Neural ODEs via the **Number of Function Evaluations (NFE)**, an aggregate NFE count treats every function evaluation and numerical step as a uniform scalar unit.

In real-world computer systems, computational cost is governed by the interaction of neural network parameter capacity ($T_f$), numerical solver mechanics ($T_{\text{solver}}$), dynamical stiffness, error tolerances, and hardware architecture. This project delivers a rigorous, empirical performance characterization of Neural ODEs on modern CPU hardware, determining the precise conditions under which NFE serves as a faithful proxy for execution cost and where solver overhead and numerical stiffness cause significant deviations.

---

## 2. Problem Statement

*(Transcribed verbatim from the mentor-approved project specification `docs/problem_statement.tex`):*

> Neural Ordinary Differential Equations (Neural ODEs) represent continuous-time dynamical processes by parameterizing the time derivative of a state using a neural network vector field,
> $$\frac{dy(t)}{dt} = f_\theta(y(t), t), \quad y(t_1) = y(t_0) + \int_{t_0}^{t_1} f_\theta(y(t), t) \, dt,$$
> where the continuous trajectory is evaluated numerically via an adaptive or fixed-step ODE solver. In machine learning literature, the computational cost of a Neural ODE is commonly summarized using the *Number of Function Evaluations (NFE)*—the count of times the neural network $f_\theta$ is invoked during numerical integration.
> 
> However, an aggregate NFE count does not directly represent the actual wall-clock time and hardware resources consumed by each evaluation or solver step. A single function evaluation varies in latency depending on network depth and width, while the numerical solver itself incurs step-adaptation, interpolation, and tableau overheads. Furthermore, under stiff dynamics, implicit solvers introduce substantial linear-algebra operations (such as Jacobian evaluations and LU decompositions) that NFE counting entirely overlooks.
> 
> This project investigates how closely NFE corresponds to actual wall-clock runtime and practical computational cost, and systematically identifies under what conditions model, solver, numerical, and hardware factors create deviations between NFE and observed computational effort. The primary research question is formulated neutrally. The objective of this work is rigorous computational performance characterization, not the development of a new Neural ODE architecture.

---

## 3. Central Research Question

> **"To what extent does NFE characterize the computational cost of Neural ODEs, and under what conditions do model, solver, numerical, and hardware factors cause deviations between NFE and actual computational cost?"**  
> `[VERIFIED SOURCE: docs/problem_statement.pdf]`

---

## 4. Research Hypothesis

> **"Hypothesis: NFE is a useful proxy for Neural ODE computational cost, but its relationship with actual wall-clock runtime systematically varies with model complexity, solver characteristics, stiffness, numerical settings, and hardware/implementation."**  
> `[VERIFIED SOURCE: docs/problem_statement.pdf]`

---

## 5. Research Objective

The objective is to establish an empirical, reproducible map of Neural ODE computational performance by:
1. Decomposing wall-clock execution time on CPU into neural network evaluation latency ($T_f$) and solver loop overhead ($T_{\text{solver}}$).
2. Quantifying the impact of network parameter capacity ($W, L$) on single-evaluation arithmetic cost.
3. Identifying the crossover boundary where dynamical stiffness causes explicit solver step collapse, rendering NFE an invalid metric compared to implicit linear-algebra workloads.

---

## 6. Experimental Scope & Boundaries

* **Hardware Scope:** Strictly **CPU-ONLY** throughout all experimental stages per mentor instruction.
  * *Prohibited:* GPU execution, CUDA profiling, mixed-precision (FP16/BF16), and CPU-vs-GPU comparisons.
  * *Rationale:* Eliminates GPU driver/kernel launch confounders and keeps technical focus strictly on numerical solver dynamics and stiffness.
* **Core Metrics:**
  * Number of Function Evaluations (Forward NFE, Backward NFE).
  * High-resolution CPU wall-clock runtime (total integration time, $T_f$ latency, $T_{\text{solver}}$ overhead).
  * Solver step statistics (accepted steps, rejected steps, step sizes).
  * Implicit solver overhead (Jacobian evaluations $N_{\text{jev}}$, LU factorizations $N_{\text{lu}}$).
* **Controlled Variables:**
  * Dynamical stiffness ($\mu \in [1, 100]$ on Van der Pol; reaction rates on Robertson).
  * Solver family (Explicit Runge-Kutta: `dopri5`, `tsit5`; Implicit Stiff: `radau`, `bdf`).
  * Numerical tolerances ($\text{rtol}, \text{atol} \in [10^{-3}, 10^{-9}]$).
  * Model capacity (width $W$, depth $L$).

---

## 7. Benchmark Dynamical Systems

The project evaluates four canonical benchmark systems spanning non-stiff to extreme multiscale regimes:

| System | State Dim ($D$) | Horizon | Classification | Role in Computational Study |
|:---|:---:|:---:|:---|:---|
| **Lotka-Volterra (LV)** | 2 | $t \in [0, 15]$ | Non-Stiff Reference | Baseline zero-stiffness reference; isolates pure network evaluation cost ($T_f$) and baseline solver overhead without numerical stiffness. |
| **FitzHugh-Nagumo (FHN)** | 2 | $t \in [0, 50]$ | Candidate Non-Stiff / Mildly Multiscale | Tests adaptive step controller responsiveness during localized fast action-potential transitions without solver instability. |
| **Van der Pol (VdP)** | 2 | $t \in [0, 30]$ | Stiff Benchmark (Tunable $\mu$) | Continuous parametric testbed ($\mu \in [1, 100]$) to map the breakdown of explicit solvers and observe the "Stiffness Cliff" on Neural ODEs. |
| **Robertson (ROBER)** | 3 | $t \in [10^{-5}, 10^5]$ | Extreme Multiscale Stiff | Asymptotic extreme where explicit solvers fail over long horizons, isolating the CPU cost of implicit linear algebra ($T_{\text{solver}}$, LU solves). |

---

## 8. Canonical Neural ODE Baseline

To eliminate architectural confounding factors across benchmark comparisons, all baseline evaluations use a standardized reference model (formally aligned with the verified Phase-2 implementation per `docs/architecture_decision_record.md`):

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

* **Status:** **`[OUR CANONICAL BASELINE — PHASE-2 ALIGNED]`** (synthesizes structural parameters for experimental control, aligned with verified Phase 2 implementation per `docs/architecture_decision_record.md`).
* **Topology:** Autonomous MLP, 2 hidden layers, width 64, Softplus activations, direct state input $z(t) \in \mathbb{R}^D$ (`augment_dim=0`), linear readout.
* **Verified Parameter Counts:**
  * $D=2$ (LV, FHN, VdP): **4,482 parameters** `[VERIFIED MATHEMATICAL ARITHMETIC & PHASE-2 IMPLEMENTATION]`.
  * $D=3$ (ROBER baseline architecture): **4,611 parameters** `[VERIFIED MATHEMATICAL ARITHMETIC & IMPLEMENTATION]`.

*(Historical Provenance Note: An earlier draft documented a theoretical $[z; t] \in \mathbb{R}^{D+1}$ model with GELU yielding 4,546 parameters for $D=2$. Pre-flight audit revealed the actual Phase 2 checkpoints were trained and validated as autonomous Softplus models; the specification has been formally aligned to match per `docs/architecture_decision_record.md`).*

---

## 9. Research Workflow: Completed vs. Planned

```
[COMPLETED] 1. Literature Review & 15-Paper Categorization
[COMPLETED] 2. Research Gap Identification & Neutral RQ Formulation
[COMPLETED] 3. Problem Statement Document Approved by Mentor (docs/problem_statement.pdf)
[COMPLETED] 4. Ground-Truth Reference Datasets Generated & Verified (data/trajectories/)
[COMPLETED] 5. Legacy Code & Flawed Experiments Quarantined (Phase 0)
[COMPLETED] 6. Trajectory Analytical Spectral Diagnostics for LV & FHN (Phase 1)
[COMPLETED] 7. Four-Tier Surrogate Validation & Watchdog Integration (Phase 2)
[COMPLETED] 8. Forensic Evidence Audit & Claim Retraction (Phase 3)
[COMPLETED] 9. Benchmark Portfolio & Canonical Architecture Freeze (Phase 4)
-------------------------------- CURRENT POSITION --------------------------------
[NEXT]      10. Stage 1 Controlled Experiment Protocol Design
[FUTURE]    11. Stage 1 Baseline Benchmarks (NFE vs. CPU Runtime Across Solvers)
[FUTURE]    12. Controlled Model Complexity Sweeps (Width & Depth Axes)
[FUTURE]    13. Stiffness Scaling Sweeps (Van der Pol μ Progression)
[FUTURE]    14. Tolerance Scaling & Step Rejection Analysis
[FUTURE]    15. Hypothesis Evaluation & Final FYP Thesis Submission
```

---

## 10. Repository Organization & Branching Structure

The repository maintains an auditable, phase-by-phase Git branching architecture:

* **`main`:** Clean operational root containing this README, frozen specifications, master evidence index, phase walkthroughs, and verified core code.
* **`phase-0-audit`:** Preserves legacy experiment quarantine, audit script, and reference datasets.
* **`phase-1-spectral`:** Preserves analytical Jacobian spectral analysis and measured JSON results.
* **`phase-2-surrogate`:** Preserves surrogate validation reports, checkpoints, and process watchdog tests.
* **`phase-3-evidence-audit`:** Preserves the formal claim audit and retraction records.
* **`phase-4-benchmark-freeze`:** Preserves the immutable benchmark and architecture specification.
* **`literature-corpus`:** Dedicated branch archiving the 15 research paper PDFs and bibliographic index.

---

## 11. Evidence & Epistemic Policy

To guarantee academic rigor and prevent unsubstantiated claims:
1. **Design $\neq$ Execution:** A plan or specification proves only that a design was made.
2. **Script $\neq$ Execution:** A script proves implementation exists, not that it was executed.
3. **Execution Log $\neq$ Measured Result:** A console log proves execution; a structured CSV/JSON file proves recorded numerical measurements.
4. **Paper $\neq$ Empirical Finding:** A published paper provides literature context, not our measured evidence.
5. **No Fabricated Evidence:** If an execution log was not captured historically, it is explicitly classified as `"NO DIRECT EXECUTION ARTIFACT FOUND"` rather than reconstructed retrospectively.

---

## 12. Current Project Status

* **Current Stage:** **Experimental Foundation / Stage 1 Protocol Design.**
* **CRITICAL MILESTONE CLARIFICATION:**
  * **Stage 1 controlled benchmark experiments have NOT yet been executed.**
  * NFE vs. runtime correspondence on the frozen baseline is `[NOT YET MEASURED]`.
  * Neural ODE stiffness scaling behavior is `[NOT YET MEASURED]`.
  * Model-complexity scaling on CPU runtime is `[NOT YET MEASURED]`.

---

## 13. Known Limitations & Missing Evidence

In accordance with strict scientific transparency, the following project gaps are documented:
1. **Phase 0 Execution Log:** `phase0_audit.py` executed file moves and dataset checks, but no stdout console log (`phase0_audit.log`) was captured to disk.
2. **Standalone Watchdog Unit Log:** `test_watchdog_mechanism.py` has no standalone console log; its operational firing is verified via `surrogate_validation_report.json`.
3. **Van der Pol & Robertson Spectra:** Analytical trajectory Jacobian spectra for VdP and ROBER have not yet been evaluated in `phase1_spectral_diagnostics.json`.
4. **Hardware Specifications:** Exact host CPU model, cache hierarchy, and BLAS backend will be queried and logged directly from the host machine when Stage 1 profiling begins.
