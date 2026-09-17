# Computational Performance Characterization of Neural ODEs

**Academic Context:** Final Year Project (FYP) — Track 2: Computational Characterization  
**Department:** Department of Computer Science and Engineering  
**Institution:** Indian Institute of Information Technology, Design and Manufacturing (IIITDM), Kancheepuram  
**Guide:** Dr. Noor Mahammad Shaik  
**Candidate:** Sumit Kumar (Roll No.: CS23B2008)  
**Date:** September 2026  
**Active Branch:** `exp1-analysis-evidence` (Preserves Stage 1 Experiment 1 evidence and analysis)

---

## 1. Executive Summary: The 2-Minute Mentor Guide

### 1.1 The Research Problem
Neural Ordinary Differential Equations (Neural ODEs) parameterize continuous-time state derivatives with a neural network vector field $\frac{dy(t)}{dt} = f_\theta(y(t), t)$. In scientific machine learning literature, the computational cost of Neural ODEs is almost universally summarized by a single scalar metric: the **Number of Function Evaluations (NFE)**.

However, aggregate NFE treats every vector-field evaluation and numerical step as a uniform unit of computational effort. In practical computing environments, wall-clock execution time is governed by an interaction of:
1. **Network Evaluation Latency ($T_f$):** Cost per forward pass as a function of network width $W$ and depth $L$.
2. **Solver Housekeeping Overhead ($T_{\text{solver}}$):** Error estimation, Butcher tableau intermediate stages, step-size adaptation, and dense polynomial interpolation.
3. **Dynamical Stiffness:** Step-size collapse in explicit solvers and costly Jacobian/LU solves in implicit solvers.
4. **Numerical Settings:** User tolerances ($\text{rtol}, \text{atol}$) altering adaptation frequency.
5. **Hardware & Execution Environment:** Thread pinning, memory bandwidth, cache locality, and framework dispatch overhead.

This project delivers a rigorous, empirical performance characterization on CPU hardware, establishing the precise regimes where NFE serves as a faithful cost proxy and where solver mechanics and stiffness cause systematic divergence.

---

### 1.2 Central Research Question (Frozen)
*(Transcribed verbatim from the mentor-approved project specification `docs/problem_statement.pdf`):*

> **"To what extent does NFE characterize the computational cost of Neural ODEs, and under what conditions do model, solver, numerical, and hardware factors cause deviations between NFE and actual computational cost?"**  
> `[VERIFIED SOURCE: docs/problem_statement.pdf]`

---

### 1.3 Central Research Hypothesis (Frozen)
*(Transcribed verbatim from the mentor-approved project specification `docs/problem_statement.pdf`):*

> **"Hypothesis: NFE is a useful proxy for Neural ODE computational cost, but its relationship with actual wall-clock runtime systematically varies with model complexity, solver characteristics, stiffness, numerical settings, and hardware/implementation."**  
> `[VERIFIED SOURCE: docs/problem_statement.pdf]`

---

### 1.4 Experimental Strategy
The project decomposes total wall-clock integration time on CPU into neural network latency and solver overhead:

$$T_{\text{total}} = \widehat{T}_{\text{net}} + \widehat{T}_{\text{solver}} = (\text{NFE} \times T_f) + \widehat{T}_{\text{solver}}$$

By holding specific computational factors invariant while systematically varying others across controlled multi-point ladders, we empirically measure the sensitivity of wall-clock runtime to NFE across distinct solvers, architectures, tolerances, and dynamical regimes.

All evaluations are executed under strict CPU-only execution, single-thread pinning (`torch.set_num_threads(1)`), nanosecond monotonic timing (`time.perf_counter_ns()`), and isolated process watchdog monitoring.

---

### 1.5 Experiment Roadmap (Branch Scope)

| Experiment | Purpose | Independent Variable | Status on this Branch |
|:---|:---|:---|:---:|
| **Stage 1 Experiment 1** | Baseline NFE vs. runtime characterization under controlled workload | Integration horizon / step count | **COMPLETED & AUDITED** |
| *Stage 1 Experiment 2A* | Solver algorithm & overhead decoupling across 7 explicit solvers | Solver algorithm | *Planned / Future Work — Not on this branch* |
| *Stage 1 Experiment 3* | Model complexity scaling across width axis ($W \in [16, 256]$) | Network parameter width ($W$) | *Planned / Future Work — Not on this branch* |
| *Stage 1 Experiment 4* | Tolerance sensitivity & step adaptation efficiency | Numerical tolerances ($\text{rtol}, \text{atol}$) | *Planned / Future Work — Not on this branch* |
| *Stage 1 Experiment 5* | Stiffness scaling & explicit breakdown on Van der Pol ($\mu \in [1, 100]$) | Stiffness parameter ($\mu$) | *Planned / Future Work — Not on this branch* |
| *Stage 1 Experiment 6* | Extreme multiscale breakdown & implicit linear algebra on Robertson | Reaction rates / timescale ratio | *Planned / Future Work — Not on this branch* |

> [!NOTE]
> This branch (`exp1-analysis-evidence`) preserves the complete evidence chain for **Stage 1 Experiment 1**. Future experiments (Exp 2A–6) represent planned research stages and are not part of this branch's empirical claims.

---

### 1.6 Current Status: Stage 1 Experiment 1
**STATUS: COMPLETED, FORENSICALLY AUDITED, AND PERMANENTLY PRESERVED**

* **WHAT WE ASKED:** Does NFE linearly predict wall-clock runtime when model architecture, weights, solver algorithm, tolerances, and hardware are held strictly invariant?
* **WHAT WE CHANGED:**
  * **Series 1A (Lotka-Volterra Adaptive `dopri5`):** Scaled horizon $T_k \in \{1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0\}$ (10 points).
  * **Series 1B (FitzHugh-Nagumo Adaptive `dopri5`):** Scaled horizon $T_k \in \{5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0\}$ (10 points).
  * **Series 1C (Lotka-Volterra Deterministic `rk4`):** Scaled uniform steps $N_{\text{steps}} \in \{25, 50, 100, 200, 400, 800\}$ yielding deterministic $\text{NFE} \in \{100, 200, 400, 800, 1600, 3200\}$ (6 points).
* **WHAT WE KEPT FIXED:**
  * Canonical baseline architecture: Autonomous Softplus MLP ($W=64, L=2$, 4,482 parameters).
  * Validated surrogate checkpoints: `models/checkpoints/lv_h64.pt` and `models/checkpoints/fhn_h64.pt`.
  * Numerical tolerances: $\text{rtol} = 1.0 \times 10^{-5}, \text{atol} = 1.0 \times 10^{-7}$.
  * Execution environment: CPU-only, 1 thread (`torch.set_num_threads(1)`), 20 warm-up solves, 15 recorded repetitions per point (390 total measured trajectories).
* **WHAT WE MEASURED:**
  * Exact integer NFE via forward wrapper counter (`node.nfe`).
  * Total integration runtime ($T_{\text{total}}$) via monotonic `time.perf_counter_ns()`.
  * Isolated forward latency ($T_f = 24.6000\ \mu\text{s}$ across 1,000 passes).
  * Numerical trajectory accuracy (MSE, $L_\infty$) via `CubicSpline` interpolation against high-precision Radau ground truth.
* **WHAT WE FOUND:**
  1. **Strong Within-Series Linear Predictability:** NFE linearly predicts wall-clock runtime within any single unconfounded ladder ($R^2 = 0.994051$ on LV `dopri5`, $R^2 = 0.997757$ on FHN `dopri5`, $R^2 = 0.999534$ on LV `rk4`).
  2. **Solver-Dependent Marginal Cost Disparity ($2.42\times$):** Under identical model weights and hardware, each function evaluation in adaptive `dopri5` incurred **$2.42\times$ higher marginal wall-clock cost** than in fixed-step `rk4` ($127.44\ \mu\text{s/NFE}$ vs. $52.63\ \mu\text{s/NFE}$).
  3. **Dominance of Solver Overhead in Adaptive Solvers:** In adaptive `dopri5`, internal solver housekeeping constituted **$\approx 81.5\%$ to $82.0\%$** of total wall-clock runtime; raw neural network evaluation represented only $\approx 18\%$.
  4. **System-Dependent Marginal Cost Disparity ($8.87\%$):** Under identical architecture, solver, and tolerances, FitzHugh-Nagumo exhibited an **$8.87\%$ higher marginal cost per NFE** than Lotka-Volterra ($138.74\ \mu\text{s/NFE}$ vs. $127.44\ \mu\text{s/NFE}$) due to trajectory geometry and step adaptation dynamics.
  5. **Surrogate Error vs. Solver Breakdown:** In Series 1C, as step size $h \to 0$ ($\text{NFE} \to 3200$), trajectory MSE converged to $1.9213$, matching the adaptive `dopri5` MSE of $1.9219$. This proves residual error reflects surrogate generalization drift over multiple periods, not numerical solver instability.

---

### 1.7 Direct Navigation to Experiment 1 Evidence

* 📖 **[Dedicated Experiment 1 Guide](docs/experiments/exp1/README.md):** Comprehensive mentor guide answering all 22 experimental protocol questions.
* 📊 **[Experiment 1 Full Scientific Analysis](docs/experiments/phase1_exp1_analysis.md):** 239-line disaggregated regression analysis, statistical diagnostics, and hypothesis evaluation.
* 📜 **[Experiment 1 Evidence Manifest](docs/experiments/exp1_evidence_manifest.md):** Cryptographic verification log, SHA-256 checksums, and forensic parity audit.
* 💻 **[Experiment 1 Runner Script](experiments/stage1_exp1_runner.py):** Executable Python runner with threading locks, warmups, timer harness, and watchdog.
* 📦 **[Raw JSON Results](experiments/logs/stage1_exp1_results.json):** Full-fidelity archive containing 390 individual raw timing measurements and host environment metadata (`0c4c7ee0c095ba117984121b8f5ea48fa1937067ce63ba68c352f87f3b12288c`).
* 📋 **[Raw CSV Results](experiments/logs/stage1_exp1_results.csv):** Tabular archive containing 45 schema columns across all 26 points (`ac1dab806af15f89ba9106390b3be2dc4bac0d3595679503ca7b0b7ddf9241bd`).
* 🗺️ **[Experiment Directory Map](docs/experiments/README.md):** Explains directory roles and separates raw evidence, derived metrics, and interpretation.

---

## 2. Experimental Scope & Rigorous Controls

* **Hardware Scope:** Strictly **CPU-ONLY** throughout all experimental stages per mentor instruction.
  * *Prohibited:* GPU execution, CUDA profiling, mixed-precision (FP16/BF16), and CPU-vs-GPU comparisons.
  * *Rationale:* Eliminates GPU kernel dispatch overhead, device memory transfers, and driver jitter, isolating numerical solver mechanics.
* **Threading Isolation:** Intra-op threads pinned to 1 (`torch.set_num_threads(1)`), inter-op threads pinned to 1 (`torch.set_num_interop_threads(1)`), and BLAS environment locks enforced (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`).
* **Process Watchdog:** Multi-tier Windows-safe process watchdog (`multiprocessing.get_context("spawn")`) with hard right-censoring timeout cap (30.0 seconds) to terminate non-converging or runaway solves.
* **Warm-Up Protocol:** 20 unrecorded complete integration solves per point prior to measurement to eliminate instruction-cache cold misses and dynamic linker latency.
* **Non-Parametric Reporting:** Primary representative metric is the **median** across 15 recorded repetitions, paired with the **Interquartile Range (IQR)** to ensure complete robustness against operating system background interrupts.

---

## 3. Canonical Benchmark Systems

The project evaluates four canonical benchmark systems spanning non-stiff to extreme multiscale regimes:

| System | State Dim ($D$) | Horizon | Classification | Role in Computational Study | Checkpoint Status |
|:---|:---:|:---:|:---|:---|:---|
| **Lotka-Volterra (LV)** | 2 | $t \in [0, 15]$ | Non-Stiff Reference | Baseline oscillatory reference; isolates pure network evaluation cost ($T_f$) and explicit solver overhead without stiffness. | `models/checkpoints/lv_h64.pt`<br>**[Validated Surrogate]** |
| **FitzHugh-Nagumo (FHN)** | 2 | $t \in [0, 50]$ | Candidate Non-Stiff / Mildly Multiscale | Evaluates adaptive step controller responsiveness during localized fast action-potential transitions without solver instability. | `models/checkpoints/fhn_h64.pt`<br>**[Validated Surrogate]** |
| **Van der Pol (VdP)** | 2 | $t \in [0, 30]$ | Stiff Benchmark (Tunable $\mu$) | Continuous parametric testbed ($\mu \in [1, 100]$) to map explicit solver step collapse and the "Stiffness Cliff" on Neural ODEs. | `models/checkpoints/vdp_mu100_stiff_h128.pt`<br>**[Validated Surrogate]** |
| **Robertson (ROBER)** | 3 | $t \in [10^{-5}, 10^5]$ | Extreme Multiscale Stiff | Asymptotic extreme where explicit solvers fail over long horizons, isolating the CPU cost of implicit linear algebra ($T_{\text{solver}}$, LU factorizations). | `models/checkpoints/robertson_h64.pt`<br>**[Unvalidated / Failed Tier 4]** |

---

## 4. Canonical Baseline Neural ODE Architecture

To eliminate architectural confounding factors, all baseline evaluations utilize a standardized, verified reference architecture (formally aligned with Phase-2 implementation per `docs/architecture_decision_record.md`):

```
State Vector z(t) ∈ R^D
           │
           ▼
   Linear(D ──► 64)
           │
           ▼
        Softplus (β = 1.0)
           │
           ▼
   Linear(64 ──► 64)
           │
           ▼
        Softplus (β = 1.0)
           │
           ▼
   Linear(64 ──► D)
           │
           ▼
   Output dz/dt ∈ R^D
```

* **Topology:** Autonomous Multilayer Perceptron (MLP), direct state input $z(t) \in \mathbb{R}^D$ (`augment_dim = 0`), no explicit temporal conditioning $[z; t]$.
* **Hidden Layers:** Exactly 2 hidden layers ($L=2$), width $W=64$ hidden units.
* **Activation Function:** Continuous smooth $\text{Softplus}(\beta=1.0)$, providing $C^\infty$ differentiability.
* **Verified Parameter Counts:**
  $$\text{Params}(D) = [D \times 64 + 64] + [64 \times 64 + 64] + [64 \times D + D] = 129 D + 4,224$$
  * $D=2$ (LV, FHN, VdP): **Exactly 4,482 parameters** `[VERIFIED MATHEMATICAL ARITHMETIC & PHASE-2 IMPLEMENTATION]`.
  * $D=3$ (ROBER reference): **Exactly 4,611 parameters** `[VERIFIED MATHEMATICAL ARITHMETIC & IMPLEMENTATION]`.

---

## 5. Repository Structure & Directory Map

```
FYP-computational/
├── README.md                                 # Root project documentation & mentor executive summary
├── LICENSE                                   # Repository license
│
├── docs/                                     # Core scientific specifications & protocols
│   ├── problem_statement.pdf                 # Mentor-approved problem statement & frozen RQ/hypothesis
│   ├── problem_statement.tex                 # LaTeX source for problem statement
│   ├── stage1_experiment_protocol.md         # Master experimental protocol for Stage 1
│   ├── benchmark_system_spec.md              # Canonical benchmark and solver specifications
│   ├── architecture_decision_record.md       # ADR aligning canonical architecture with Phase-2 models
│   ├── EVIDENCE_INDEX.md                     # Master index of proof-of-work across all project phases
│   ├── reproducibility.md                    # Environment requirements and reproduction guide
│   ├── experiments/                          # Experiment analysis & evidence documentation
│   │   ├── README.md                         # Experiments directory guide & epistemic demarcation
│   │   ├── exp1/                             # Dedicated Stage 1 Experiment 1 documentation
│   │   │   └── README.md                     # Comprehensive 22-question Experiment 1 mentor guide
│   │   ├── phase1_exp1_analysis.md           # Exp1 full scientific analysis & disaggregated regressions
│   │   └── exp1_evidence_manifest.md         # Exp1 cryptographic manifest & forensic audit verification
│   ├── phases/                               # Historical phase completion records (Phases 0–4)
│   └── legacy_reports/                       # Archived prior literature reports and student theses
│
├── experiments/                              # Executable experiment code & logs
│   ├── stage1_exp1_runner.py                 # Executable Stage 1 Experiment 1 runner harness
│   ├── phase1_spectral_diagnostics.py        # Analytical Jacobian spectral analysis script
│   ├── phase2_train_surrogates.py            # Phase 2 surrogate training script
│   ├── logs/                                 # Raw experimental outputs & forensic logs
│   │   ├── stage1_exp1_results.json          # Exp1 full-fidelity raw timing & environment archive
│   │   ├── stage1_exp1_results.csv           # Exp1 tabular 45-column dataset (26 points)
│   │   ├── phase1_spectral_diagnostics.json  # Phase 1 trajectory spectral diagnostics
│   │   ├── surrogate_validation_report.json  # Phase 2 4-tier surrogate validation report
│   │   └── archive/                          # Quarantined legacy execution logs
│   ├── plots/                                # Diagnostic trajectory and loss plots
│   └── archive/                              # Quarantined legacy experiment scripts
│
├── models/                                   # Neural network architectures & validated weights
│   ├── neural_ode.py                         # Neural ODE wrapper class with NFE counter
│   ├── vector_fields.py                      # Autonomous MLP vector field definitions
│   ├── scaled_node.py                        # Scaled Neural ODE variant
│   └── checkpoints/                          # Verified PyTorch model checkpoints
│       ├── lv_h64.pt                         # Validated Lotka-Volterra surrogate (4,482 params)
│       ├── fhn_h64.pt                        # Validated FitzHugh-Nagumo surrogate (4,482 params)
│       └── vdp_mu100_stiff_h128.pt           # Validated Van der Pol stiff surrogate
│
├── data/                                     # Ground truth reference datasets & generators
│   ├── generate_data.py                      # High-precision Radau ODE integrator script
│   ├── trajectories/                         # High-precision reference trajectories (.pt and .csv)
│   └── plots/                                # Reference phase-portrait plots
│
├── profiling/                                # Profiling harness and logging utilities
│   ├── harness.py                            # Execution harness with CPU thread pinning
│   └── logger.py                             # Structured experiment logging utilities
│
└── tests/                                    # Unit tests and regression suites
    ├── test_pipeline.py                      # End-to-end pipeline test
    ├── test_robertson_validation.py          # Robertson validation test
    └── test_watchdog_mechanism.py            # Windows subprocess watchdog unit test
```

---

## 6. Repository Branching Architecture

The repository maintains an auditable, phase-by-phase Git branching architecture:

* **`main`:** Clean operational root containing frozen specifications, master evidence index, phase walkthroughs, and verified core code.
* **`exp1-analysis-evidence`:** Dedicated working branch preserving Stage 1 Experiment 1 empirical results, raw CSV/JSON logs, analysis, and evidence manifests.
* **`phase-4-benchmark-freeze`:** Preserves the immutable benchmark and architecture specification.
* **`phase-3-evidence-audit`:** Preserves the formal claim audit and retraction records.
* **`phase-2-surrogate`:** Preserves surrogate validation reports, checkpoints, and process watchdog tests.
* **`phase-1-spectral`:** Preserves analytical Jacobian spectral analysis and measured JSON results.
* **`phase-0-audit`:** Preserves legacy experiment quarantine, audit script, and reference datasets.
* **`literature-corpus`:** Dedicated branch archiving the 15 research paper PDFs and bibliographic index.

---

## 7. Strict Epistemic & Evidence Policy

To guarantee academic rigor and prevent unsubstantiated claims:
1. **Design $\neq$ Execution:** A plan or protocol proves only that an experimental design was formalized.
2. **Script $\neq$ Execution:** A script proves implementation exists, not that it was executed.
3. **Execution Log $\neq$ Measured Result:** A console log proves execution occurred; a structured CSV/JSON file proves recorded numerical measurements.
4. **Paper $\neq$ Empirical Finding:** A published paper provides literature context, not our measured evidence.
5. **Separation of Evidence:** Direct measurements (Category A) are strictly distinguished from derived metrics (Category B) and scientific interpretations (Category C).
6. **No Fabricated Evidence:** If an execution log was not captured historically, it is explicitly classified as `"NO DIRECT EXECUTION ARTIFACT FOUND"` rather than reconstructed retrospectively.

---

## 8. Reproducing Stage 1 Experiment 1

To reproduce the Experiment 1 empirical measurements from scratch under the verified host environment:

```bash
# Ensure single-thread CPU execution environment
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Run the verified Experiment 1 execution harness
python experiments/stage1_exp1_runner.py
```

*Estimated execution time:* $\approx 73$ seconds on an Intel CPU (26 points $\times$ [20 warmups + 15 measured repetitions] = 910 total ODE solves).  
*Output location:* `experiments/logs/stage1_exp1_results.json` and `experiments/logs/stage1_exp1_results.csv`.
