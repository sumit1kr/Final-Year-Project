# Stage 1 Experiment 1: Baseline NFE-vs-Runtime Characterization

**Academic Context:** Final Year Project (FYP) — Track 2: Computational Characterization  
**Department:** Department of Computer Science and Engineering  
**Institution:** Indian Institute of Information Technology, Design and Manufacturing (IIITDM), Kancheepuram  
**Guide:** Dr. Noor Mahammad Shaik  
**Candidate:** Sumit Kumar (Roll No.: CS23B2008)  
**Experiment Identifier:** `STAGE_1_EXP1`  
**Status:** **COMPLETED, FORENSICALLY AUDITED, AND PRESERVED**  
**Execution Timestamp:** 2026-09-16T08:05:49Z to 2026-09-16T08:07:12Z (UTC)  
**Execution Commit Hash:** `09e9c1a`  

---

## Executive Overview

This document serves as the dedicated mentor landing page and scientific guide for **Stage 1 Experiment 1**. Experiment 1 establishes the foundational empirical baseline of the Final Year Project, evaluating the direct relationship between the scalar **Number of Function Evaluations (NFE)** and actual **CPU wall-clock execution time** under strict experimental control.

A mentor can review this document to understand the full scientific justification, methodology, results, and evidence chain without inspecting Python source code.

---

## 1. The 22 Core Experimental Questions

### Q1: What is Experiment 1?
Experiment 1 is the baseline controlled benchmark of Track 2. It characterizes the empirical relationship between scalar NFE and total wall-clock integration time on CPU across 26 distinct, systematically varied integration workloads across two non-stiff benchmark dynamical systems.

### Q2: Why was it performed?
Foundational Neural ODE literature routinely reports NFE as the sole indicator of computational cost. Before evaluating complex confounding factors (such as numerical stiffness, variable network capacities, and solver algorithms), we must first establish whether NFE reliably predicts wall-clock execution time under clean, uncontaminated baseline conditions where architecture, weights, solver, tolerances, and hardware are held strictly invariant.

### Q3: What research question does it address?
Experiment 1 addresses the baseline foundation of our frozen Central Research Question:
> *"To what extent does NFE characterize the computational cost of Neural ODEs, and under what conditions do model, solver, numerical, and hardware factors cause deviations between NFE and actual computational cost?"*

Specifically: **Under uncontaminated, non-stiff conditions with fixed model architecture and fixed tolerances, does NFE linearly predict wall-clock runtime, and what fraction of runtime is consumed by internal solver mechanics vs. neural evaluations?**

### Q4: What is the independent variable?
The primary independent variable is the **integration workload**, systematically varied to drive NFE across a wide dynamic range through two complementary ladders:
1. **Horizon Scaling in Adaptive `dopri5`:** Integrating over 10 sub-horizons ($T_k \in [1.5, 15.0]$ for Lotka-Volterra; $T_k \in [5.0, 50.0]$ for FitzHugh-Nagumo).
2. **Step Count Scaling in Deterministic `rk4`:** Scaling uniform steps $N_{\text{steps}} \in \{25, 50, 100, 200, 400, 800\}$ on Lotka-Volterra, forcing integer evaluations $\text{NFE} \in \{100, 200, 400, 800, 1600, 3200\}$.

### Q5: What was kept fixed?
Strict experimental controls held all potential confounders invariant:
* **Neural Architecture:** Identical autonomous Softplus MLP ($W=64, L=2$, 4,482 parameters).
* **Model Checkpoints:** Identical pre-trained Phase-2 weights (`lv_h64.pt`, `fhn_h64.pt`).
* **Numerical Tolerances:** Strictly fixed at $\text{rtol} = 1.0 \times 10^{-5}, \text{atol} = 1.0 \times 10^{-7}$ for adaptive runs.
* **Hardware Device:** Single-thread CPU execution (`torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`).
* **Random Seed:** Set to 42.
* **Process Environment:** BLAS thread locks (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`).

### Q6: Which Neural ODE architecture was used?
The canonical baseline architecture formally adopted in `docs/architecture_decision_record.md`:
* **Input:** State vector $z(t) \in \mathbb{R}^D$ ($D=2$, `augment_dim = 0`), no temporal concatenation.
* **Hidden Layers:** 2 layers, width $W=64$.
* **Activation Function:** Continuous smooth $\text{Softplus}(\beta=1.0)$, providing $C^\infty$ differentiability.
* **Output:** $\mathbb{R}^{64} \to \mathbb{R}^D$ linear derivative projection $\frac{dz}{dt}$.
* **Total Parameters:** Exactly **4,482 parameters** ($\text{Params}(2) = 129(2) + 4,224 = 4,482$).

### Q7: Which ODE systems were used?
Two non-stiff, well-characterized 2D dynamical systems from `docs/benchmark_system_spec.md`:
1. **Lotka-Volterra (LV):** Non-stiff predator-prey oscillatory system ($y_0 = [1.0, 0.5]^\top$, full horizon $T=15.0$).
2. **FitzHugh-Nagumo (FHN):** Candidate non-stiff / mildly multiscale excitable membrane system ($y_0 = [-1.0, 1.0]^\top$, full horizon $T=50.0$).

### Q8: Which solvers were used?
1. **`dopri5` (Dormand-Prince 5(4)):** The canonical adaptive explicit Runge-Kutta solver utilized throughout Neural ODE literature (Chen et al., 2018).
2. **`rk4` (Classical 4th-order Runge-Kutta):** A fixed-step reference solver evaluated as a zero-adaptation control baseline where NFE is known *a priori* ($\text{NFE} = 4 \times N_{\text{steps}}$).

### Q9: What integration horizons/workloads were tested?
A total of **26 distinct points**:
* **Series 1A (LV `dopri5`):** 10 points ($T_k \in \{1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0\}$).
* **Series 1B (FHN `dopri5`):** 10 points ($T_k \in \{5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0\}$).
* **Series 1C (LV `rk4`):** 6 points ($N_{\text{steps}} \in \{25, 50, 100, 200, 400, 800\}$ on $T=15.0$).

### Q10: What is NFE?
**Number of Function Evaluations (NFE)** is the cumulative integer count of times the neural network vector field $f_\theta(z, t)$ is evaluated by the numerical solver during trajectory integration. For adaptive solvers, NFE reflects accepted step stages, rejected trial steps, and interpolation evaluations.

### Q11: How was NFE measured?
NFE was measured directly using a transparent forward wrapper counter attached to the neural network model:
```python
class CountWrapper(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
        self.nfe = 0
    def forward(self, t, y):
        self.nfe += 1
        return self.func(t, y)
```
Before each solve, `wrapper.nfe` was reset to zero. After integration completed, the integer counter was recorded directly into the output record.

### Q12: What is $T_f$?
$T_f$ is the **isolated forward evaluation latency** of the neural network vector field. It represents the raw CPU time required to execute one forward pass $f_\theta(z)$ on a single state coordinate tensor outside the numerical ODE solver.

### Q13: How was $T_f$ measured?
$T_f$ was benchmarked per checkpoint prior to integration:
1. 50 representative state coordinates sampled along the pre-computed high-precision reference trajectory.
2. 50 unrecorded warm-up evaluations.
3. 1,000 independent timed forward evaluations using monotonic `time.perf_counter_ns()`.
4. Result: Both checkpoints yielded a median latency of **$T_f = 24.6000\ \mu\text{s}$** ($0.02460\text{ ms}$) with $\text{IQR} \le 1.00\ \mu\text{s}$ ($4.07\%$ dispersion), satisfying the protocol stability criterion ($\le 5.0\%$).

### Q14: How was wall-clock runtime measured?
Wall-clock integration runtime ($T_{\text{total}}$) was measured in-process using monotonic nanosecond timers:
```python
t_start = time.perf_counter_ns()
solution = odeint(model, y0, t_eval, method=solver, rtol=rtol, atol=atol)
t_end = time.perf_counter_ns()
runtime_ms = (t_end - t_start) / 1e6
```
This isolates the solve call from setup, validation, disk I/O, and garbage collection.

### Q15: How many warmups?
Exactly **20 unrecorded, complete ODE integration solves** were executed for each configuration before taking any recorded measurements. This fully primes CPU L1/L2/L3 instruction and data caches, branch predictors, and eliminates dynamic library linking jitter.

### Q16: How many recorded repetitions?
Exactly **15 independent recorded solves** were captured per point ($15 \times 26 = 390$ total recorded solves). All 15 individual measurement timings are preserved in the raw JSON archive array `raw_runtimes_ms`.

### Q17: What accuracy metrics were measured?
Numerical solution trajectories were evaluated against high-precision reference trajectories generated with SciPy Radau ($\text{rtol}=10^{-10}, \text{atol}=10^{-12}$):
* **Mean Squared Error (MSE):** Evaluated at exact solution timestamps using `scipy.interpolate.CubicSpline`.
* **Relative $L_2$ Error (Rel-$L_2$):** Normalized Euclidean trajectory error.
* **Chebyshev Error ($L_\infty$):** Maximum absolute coordinate discrepancy across the trajectory.

### Q18: What derived metrics were calculated?
From direct measurements, four derived metrics were computed (with strict epistemic demarcation):
1. **Estimated Network Runtime:** $\widehat{T}_{\text{net}} = \frac{\text{NFE} \times T_f}{1000.0}$ (ms).
2. **Estimated Solver Overhead:** $\widehat{T}_{\text{solver}} = T_{\text{total}} - \widehat{T}_{\text{net}}$ (ms).
3. **Solver Overhead Ratio:** $\widehat{\Omega}_{\text{solver}} = \frac{\widehat{T}_{\text{solver}}}{T_{\text{total}}}$.
4. **Disaggregated OLS Regressions:** Slope $\beta_1$ (marginal cost in $\mu\text{s/NFE}$), intercept $\beta_0$, and coefficient of determination $R^2$ fitted independently per ladder.

### Q19: What were the actual results?
* **Linearity:** Within each series, NFE linearly predicts wall-clock runtime with extreme fidelity ($R^2 \ge 0.994$).
* **Solver Overhead:** In adaptive `dopri5`, solver housekeeping consumes **$81.5\%$ to $82.0\%$** of total wall-clock runtime. Raw vector-field evaluation represents only $\approx 18\%$.
* **Marginal Cost Disparity:** Under identical model and CPU, adaptive `dopri5` incurred **$2.42\times$ greater marginal cost per NFE** than fixed-step `rk4` ($127.44\ \mu\text{s}$ vs. $52.63\ \mu\text{s}$).
* **System Disparity:** FitzHugh-Nagumo incurred **$8.87\%$ higher marginal cost per NFE** than Lotka-Volterra ($138.74\ \mu\text{s}$ vs. $127.44\ \mu\text{s}$) under identical architecture and tolerances.
* **Trajectory Accuracy:** Trajectory MSE converged to $1.9213$ as $h \to 0$ in `rk4`, exactly matching the adaptive `dopri5` MSE ($1.9219$), confirming that residual error reflects surrogate generalization drift, not solver breakdown.

### Q20: What does the result mean?
NFE is a reliable linear proxy for runtime **only within a single, fixed solver-system ladder**. However, NFE cannot be directly compared across different solvers: an evaluation under adaptive `dopri5` is over twice as expensive in wall-clock time as an evaluation under fixed `rk4`. Furthermore, for lightweight Neural ODEs ($W=64$), algorithmic solver housekeeping dominates runtime over raw neural arithmetic.

### Q21: What are the limitations?
1. **Workstation-Specific Timings:** Marginal latencies ($\mu\text{s/NFE}$) reflect the single-threaded CPU architecture of the test workstation (Intel Core 14th Gen, Windows 10).
2. **Fixed Architecture Capacity:** Evaluated only $W=64, L=2$ ($T_f = 24.6\ \mu\text{s}$). Whether $T_f$ overtakes solver overhead for wider networks ($W=256$) is evaluated in Stage 1 Experiment 3.
3. **Non-Stiff Regime:** Both LV and FHN are non-stiff systems; explicit solver step collapse under severe stiffness is evaluated in Stage 1 Experiment 5.
4. **Derived Residual:** Solver overhead $\widehat{T}_{\text{solver}}$ is a synthetic residual based on isolated $T_f$, not an in-situ kernel timer.

### Q22: Why does Experiment 2A follow from Experiment 1?
Experiment 1 revealed a massive $2.42\times$ marginal cost difference between `dopri5` and `rk4`, and demonstrated that solver housekeeping accounts for $>80\%$ of runtime in `dopri5`. However, Exp1 compared only two solvers. **Experiment 2A (Solver Algorithm Characterization)** directly follows to decouple:
* Does this overhead disparity persist across 7 explicit Runge-Kutta orders (`euler`, `midpoint`, `rk4`, `adaptive_heun`, `bosh3`, `dopri5`, `dopri8`)?
* What is the exact algorithmic stage cost per order?

---

## 2. Concise Experimental Result Summaries

> [!IMPORTANT]
> **Scientific Guardrail:** NFE showed strong within-series runtime predictability under the controlled Experiment-1 setup. We do NOT claim that NFE universally predicts runtime across disparate models, solvers, or hardware platforms.

### Series 1A: Lotka-Volterra Adaptive `dopri5` (Horizon Ladder)
* **Model:** `models/checkpoints/lv_h64.pt` (4,482 parameters)
* **Solver:** `dopri5` ($\text{rtol}=1.0 \times 10^{-5}, \text{atol}=1.0 \times 10^{-7}$)
* **Sample Size:** 10 points $\times$ 15 repetitions = 150 recorded solves

| Point ID | Horizon $T_k$ | Eval Grid ($N_{\text{eval}}$) | NFE (med $\pm$ IQR) | $T_{\text{total}}$ ms (med $\pm$ IQR) | $\widehat{T}_{\text{net}}$ ms | $\widehat{T}_{\text{solver}}$ ms | Solver Overhead $\widehat{\Omega}$ | Trajectory MSE | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `EXP1-S1A-LV-H1.5` | 1.5 | 16 | $74 \pm 0$ | $9.74 \pm 0.97$ | 1.82 | 7.92 | 81.3% | $2.89 \times 10^{-2}$ | `SUCCESS` |
| `EXP1-S1A-LV-H3.0` | 3.0 | 31 | $164 \pm 0$ | $20.93 \pm 0.93$ | 4.03 | 16.90 | 80.7% | $1.17 \times 10^{-1}$ | `SUCCESS` |
| `EXP1-S1A-LV-H4.5` | 4.5 | 46 | $206 \pm 0$ | $27.82 \pm 2.75$ | 5.07 | 22.75 | 81.8% | $1.04 \times 10^{-1}$ | `SUCCESS` |
| `EXP1-S1A-LV-H6.0` | 6.0 | 61 | $308 \pm 0$ | $42.94 \pm 2.62$ | 7.58 | 35.36 | 82.4% | $5.14 \times 10^{-1}$ | `SUCCESS` |
| `EXP1-S1A-LV-H7.5` | 7.5 | 76 | $356 \pm 0$ | $50.13 \pm 2.50$ | 8.76 | 41.37 | 82.5% | $4.61 \times 10^{-1}$ | `SUCCESS` |
| `EXP1-S1A-LV-H9.0` | 9.0 | 91 | $434 \pm 0$ | $61.02 \pm 6.01$ | 10.68 | 50.34 | 82.5% | $7.40 \times 10^{-1}$ | `SUCCESS` |
| `EXP1-S1A-LV-H10.5` | 10.5 | 106 | $512 \pm 0$ | $67.70 \pm 1.99$ | 12.60 | 55.10 | 81.4% | $1.13 \times 10^{0}$ | `SUCCESS` |
| `EXP1-S1A-LV-H12.0` | 12.0 | 121 | $554 \pm 0$ | $71.20 \pm 2.84$ | 13.63 | 57.57 | 80.9% | $1.18 \times 10^{0}$ | `SUCCESS` |
| `EXP1-S1A-LV-H13.5` | 13.5 | 136 | $650 \pm 0$ | $84.41 \pm 8.41$ | 15.99 | 68.42 | 81.1% | $2.03 \times 10^{0}$ | `SUCCESS` |
| `EXP1-S1A-LV-H15.0` | 15.0 | 151 | $698 \pm 0$ | $88.67 \pm 1.79$ | 17.17 | 71.50 | 80.6% | $1.92 \times 10^{0}$ | `SUCCESS` |

* **Regression Fit:** $T_{\text{total}} = 0.127435 \cdot \text{NFE} + 2.0414\text{ ms}$ ($\beta_1 = 127.44 \pm 3.49\ \mu\text{s/NFE}$, $R^2 = 0.994051$).

---

### Series 1B: FitzHugh-Nagumo Adaptive `dopri5` (Horizon Ladder)
* **Model:** `models/checkpoints/fhn_h64.pt` (4,482 parameters)
* **Solver:** `dopri5` ($\text{rtol}=1.0 \times 10^{-5}, \text{atol}=1.0 \times 10^{-7}$)
* **Sample Size:** 10 points $\times$ 15 repetitions = 150 recorded solves

| Point ID | Horizon $T_k$ | Eval Grid ($N_{\text{eval}}$) | NFE (med $\pm$ IQR) | $T_{\text{total}}$ ms (med $\pm$ IQR) | $\widehat{T}_{\text{net}}$ ms | $\widehat{T}_{\text{solver}}$ ms | Solver Overhead $\widehat{\Omega}$ | Trajectory MSE | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `EXP1-S1B-FHN-H5.0` | 5.0 | 21 | $86 \pm 0$ | $11.44 \pm 1.75$ | 2.12 | 9.32 | 81.5% | $2.41 \times 10^{-5}$ | `SUCCESS` |
| `EXP1-S1B-FHN-H10.0` | 10.0 | 41 | $122 \pm 0$ | $16.43 \pm 0.23$ | 3.00 | 13.43 | 81.7% | $3.52 \times 10^{-5}$ | `SUCCESS` |
| `EXP1-S1B-FHN-H15.0` | 15.0 | 61 | $146 \pm 0$ | $20.37 \pm 0.63$ | 3.59 | 16.78 | 82.4% | $3.76 \times 10^{-5}$ | `SUCCESS` |
| `EXP1-S1B-FHN-H20.0` | 20.0 | 81 | $170 \pm 0$ | $23.74 \pm 0.72$ | 4.18 | 19.56 | 82.4% | $2.80 \times 10^{-4}$ | `SUCCESS` |
| `EXP1-S1B-FHN-H25.0` | 25.0 | 101 | $254 \pm 0$ | $34.70 \pm 2.81$ | 6.25 | 28.45 | 82.0% | $3.47 \times 10^{-2}$ | `SUCCESS` |
| `EXP1-S1B-FHN-H30.0` | 30.0 | 121 | $326 \pm 0$ | $43.34 \pm 0.52$ | 8.02 | 35.32 | 81.5% | $3.08 \times 10^{-2}$ | `SUCCESS` |
| `EXP1-S1B-FHN-H35.0` | 35.0 | 141 | $362 \pm 0$ | $48.91 \pm 0.57$ | 8.91 | 40.00 | 81.8% | $2.76 \times 10^{-2}$ | `SUCCESS` |
| `EXP1-S1B-FHN-H40.0` | 40.0 | 161 | $458 \pm 0$ | $62.40 \pm 1.54$ | 11.27 | 51.13 | 81.9% | $4.04 \times 10^{-2}$ | `SUCCESS` |
| `EXP1-S1B-FHN-H45.0` | 45.0 | 181 | $512 \pm 0$ | $73.11 \pm 5.67$ | 12.60 | 60.51 | 82.8% | $3.64 \times 10^{-2}$ | `SUCCESS` |
| `EXP1-S1B-FHN-H50.0` | 50.0 | 201 | $542 \pm 0$ | $73.96 \pm 0.75$ | 13.33 | 60.63 | 82.0% | $3.30 \times 10^{-2}$ | `SUCCESS` |

* **Regression Fit:** $T_{\text{total}} = 0.138736 \cdot \text{NFE} - 0.4761\text{ ms}$ ($\beta_1 = 138.74 \pm 2.33\ \mu\text{s/NFE}$, $R^2 = 0.997757$).

---

### Series 1C: Lotka-Volterra Deterministic `rk4` (Step Count Ladder)
* **Model:** `models/checkpoints/lv_h64.pt` (4,482 parameters)
* **Solver:** Classical fixed-step `rk4` on $T \in [0, 15.0]$
* **Sample Size:** 6 points $\times$ 15 repetitions = 90 recorded solves

| Point ID | Steps ($N_{\text{steps}}$) | Step Size $h$ | NFE (Deterministic) | $T_{\text{total}}$ ms (med $\pm$ IQR) | $\widehat{T}_{\text{net}}$ ms | $\widehat{T}_{\text{solver}}$ ms | Solver Overhead $\widehat{\Omega}$ | Trajectory MSE | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `EXP1-S1C-LV-RK4-N25` | 25 | 0.6000 | 100 | $8.31 \pm 0.27$ | 2.46 | 5.85 | 70.4% | 1.7400 | `SUCCESS` |
| `EXP1-S1C-LV-RK4-N50` | 50 | 0.3000 | 200 | $14.05 \pm 0.65$ | 4.92 | 9.13 | 65.0% | 1.4983 | `SUCCESS` |
| `EXP1-S1C-LV-RK4-N100` | 100 | 0.1500 | 400 | $27.63 \pm 1.29$ | 9.84 | 17.79 | 64.4% | 1.8759 | `SUCCESS` |
| `EXP1-S1C-LV-RK4-N200` | 200 | 0.0750 | 800 | $46.33 \pm 4.52$ | 19.68 | 26.65 | 57.5% | 1.9104 | `SUCCESS` |
| `EXP1-S1C-LV-RK4-N400` | 400 | 0.0375 | 1600 | $89.77 \pm 2.23$ | 39.36 | 50.41 | 56.2% | 1.9191 | `SUCCESS` |
| `EXP1-S1C-LV-RK4-N800` | 800 | 0.01875 | 3200 | $172.21 \pm 5.82$ | 78.72 | 93.49 | 54.3% | 1.9213 | `SUCCESS` |

* **Regression Fit:** $T_{\text{total}} = 0.052629 \cdot \text{NFE} + 4.4575\text{ ms}$ ($\beta_1 = 52.63 \pm 0.57\ \mu\text{s/NFE}$, $R^2 = 0.999534$).

---

## 3. Cross-Series Synthesis & Key Findings

| Comparison Dimension | Series 1A: LV `dopri5` | Series 1B: FHN `dopri5` | Series 1C: LV `rk4` | Key Empirical Finding |
|:---|:---:|:---:|:---:|:---|
| **Marginal Cost ($eta_1$)** | **$127.44\ \mu\text{s/NFE}$** | **$138.74\ \mu\text{s/NFE}$** | **$52.63\ \mu\text{s/NFE}$** | Adaptive `dopri5` is **$2.42\times$ more expensive per NFE** than fixed `rk4`. |
| **Linear Fit ($R^2$)** | **0.994051** | **0.997757** | **0.999534** | Exceptionally high within-series linearity across all ladders. |
| **Mean Solver Overhead** | **81.5%** | **82.0%** | **61.3%** | Solver housekeeping dominates total runtime for lightweight models. |
| **Fixed Intercept ($eta_0$)** | $+2.04\text{ ms}$ | $-0.48\text{ ms}$ | $+4.46\text{ ms}$ | Minimal initialization constant; runtime scales directly with evaluations. |

---

## 4. Experiment-1 Evidence Map

Every artifact listed below is a verified physical file in this repository:

* 📋 **[Experiment 1 — Experimental Protocol](../../stage1_experiment_protocol.md):** Formal pre-execution specification governing Series 1A, 1B, and 1C (Section 5).
* 💻 **[Experiment 1 — Experiment Runner](../../../experiments/stage1_exp1_runner.py):** Executable Python runner script with thread pinning, monotonic timer harness, and spawn watchdog.
* 📦 **[Experiment 1 — Raw JSON Results](../../../experiments/logs/stage1_exp1_results.json):** Full-fidelity archive containing 390 individual raw timing measurements and host environment metadata (SHA-256: `0c4c7ee0c095ba117984121b8f5ea48fa1937067ce63ba68c352f87f3b12288c`).
* 📊 **[Experiment 1 — Raw CSV Results](../../../experiments/logs/stage1_exp1_results.csv):** Tabular archive containing 45 schema columns across all 26 points (SHA-256: `ac1dab806af15f89ba9106390b3be2dc4bac0d3595679503ca7b0b7ddf9241bd`).
* 📑 **[Experiment 1 — Full Scientific Analysis](../phase1_exp1_analysis.md):** Formal 239-line analysis report containing statistical diagnostics, OLS regressions, and hypothesis evaluations.
* 🔒 **[Experiment 1 — Evidence Manifest](../exp1_evidence_manifest.md):** Cryptographic verification log, failure boundary checks, and forensic audit verification.
* 🎯 **[Lotka-Volterra Ground Truth Trajectory](../../../data/trajectories/lotka-volterra_ground_truth.pt):** High-precision Radau ground truth reference.
* 🎯 **[FitzHugh-Nagumo Ground Truth Trajectory](../../../data/trajectories/fitzhugh-nagumo_ground_truth.pt):** High-precision Radau ground truth reference.
* 🧠 **[Lotka-Volterra Baseline Checkpoint](../../../models/checkpoints/lv_h64.pt):** Verified Phase-2 surrogate checkpoint (4,482 parameters).
* 🧠 **[FitzHugh-Nagumo Baseline Checkpoint](../../../models/checkpoints/fhn_h64.pt):** Verified Phase-2 surrogate checkpoint (4,482 parameters).
* 🏛️ **[Architecture Decision Record](../../architecture_decision_record.md):** Formal ADR adopting autonomous Softplus MLP as canonical baseline.
* 📐 **[Benchmark System Specification](../../benchmark_system_spec.md):** Canonical benchmark systems and mathematical formulations.
* 📜 **[Master Evidence Index](../../EVIDENCE_INDEX.md):** Master proof-of-work index tracking all research stages.
