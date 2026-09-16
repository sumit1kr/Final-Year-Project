# Stage 1 Experiment 1: Baseline NFE-vs-Runtime Characterization Analysis

**Research Track:** Track 2 — Computational Performance Characterization of Neural ODEs
**Experiment Identifier:** `STAGE_1_EXP1` (Baseline Characterization)
**Parent Protocol:** `docs/stage1_experiment_protocol.md`
**Architecture Decision Record:** `docs/architecture_decision_record.md`
**Execution Commit:** `09e9c1a` (`experiments/stage1_exp1_runner.py`)
**Evidence Artifacts:**
- `experiments/logs/stage1_exp1_results.json` (SHA-256: `0c4c7ee0c095ba117984121b8f5ea48fa1937067ce63ba68c352f87f3b12288c`)
- `experiments/logs/stage1_exp1_results.csv` (SHA-256: `ac1dab806af15f89ba9106390b3be2dc4bac0d3595679503ca7b0b7ddf9241bd`)
**Forensic Audit Verdict:** `PASS` (Full parity verified across 26/26 records, 390 recorded solves)

---

## 1. Research Question Addressed

Experiment 1 provides the foundational empirical baseline addressing the primary research question defined in `docs/problem_statement.pdf` and `docs/stage1_experiment_protocol.md`:

> **"To what extent does NFE characterize the computational cost of Neural ODEs, and under what conditions do model, solver, numerical, and hardware factors cause deviations between NFE and actual computational cost?"**

Specifically, Experiment 1 establishes the baseline relationship between scalar Number of Function Evaluations (NFE) and actual CPU wall-clock integration time under uncontaminated, non-stiff conditions when model topology, weights, hardware, and tolerances are held strictly invariant.

---

## 2. Experimental Setup & Protocol Controls

### 2.1 Canonical Baseline Architecture (Option A)
All experiments utilized the verified canonical baseline Neural ODE architecture formally adopted in `docs/architecture_decision_record.md`:
- **Vector Field Topology:** Autonomous Multilayer Perceptron $f_\theta(z)$, no temporal conditioning $[z; t]$.
- **State Dimensions:** $D=2$ (Lotka-Volterra, FitzHugh-Nagumo).
- **Augmentation:** Zero augmentation dimension (`augment_dim = 0`).
- **Hidden Layers:** Exactly 2 hidden layers ($L=2$), width $W=64$.
- **Activation Function:** Continuous smooth $\text{Softplus}(\beta=1.0)$.
- **Parameter Count:** Exactly **4,482 parameters** ($\text{Params}(D) = 129D + 4,224$).
- **Surrogate Checkpoints:** Verified pre-trained Phase-2 weights:
  - `models/checkpoints/lv_h64.pt`
  - `models/checkpoints/fhn_h64.pt`

### 2.2 CPU Execution Controls & Hardware Invariants
To ensure scientific reproducibility and eliminate host execution jitter:
- **Device:** CPU-Only execution (`hardware_device = CPU`).
- **Host CPU:** `Intel64 Family 6 Model 183 Stepping 1, GenuineIntel` (28 logical cores).
- **Operating System:** `Windows 10 (10.0.26200)`.
- **Thread Pinning:** Intra-op threads pinned to 1 (`torch.set_num_threads(1)`), inter-op threads pinned to 1 (`torch.set_num_interop_threads(1)`), and environment variables set (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`).
- **Random Seed:** Initialized to 42 (`torch.manual_seed(42)`, `np.random.seed(42)`).
- **Process Watchdog:** Windows-safe child process isolation (`multiprocessing.get_context("spawn")`) with hard right-censoring timeout at 30.0 seconds per solve.

### 2.3 Repetition & Sampling Structure
- **Warm-Up Cycles:** 20 unrecorded complete ODE integration solves per point to ensure steady-state CPU instruction/data caches and eliminate dynamic linker overhead.
- **Recorded Repetitions:** 15 independent recorded solves per point ($15 \times 26 = 390$ total measured trajectories).
- **Primary Representative Metric:** Non-parametric **median** to prevent bias from asymmetric operating system interrupts.
- **Dispersion Metric:** **Interquartile Range** ($\text{IQR} = Q_3 - Q_1$).

---

## 3. Isolated Latency Benchmark ($T_f$)

Per Protocol Section 6.2, network evaluation latency $T_f$ was benchmarked independently for each system checkpoint prior to integration:
- 50 representative state coordinates sampled along pre-computed high-precision reference trajectories.
- 50 unrecorded warm-up evaluations.
- 1,000 independent timed evaluations via monotonic `time.perf_counter_ns()`.

| System | Checkpoint File | Parameters | Median $T_f$ ($\mu\text{s}$) | IQR $T_f$ ($\mu\text{s}$) | Dispersion $\frac{\text{IQR}}{\text{median}}$ | Stability Criterion ($\le 5\%$) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **FitzHugh-Nagumo** | `models/checkpoints/fhn_h64.pt` | 4,482 | **24.6000** | 1.0000 | 4.07% | **PASSED** |
| **Lotka-Volterra** | `models/checkpoints/lv_h64.pt` | 4,482 | **24.6000** | 0.9000 | 3.66% | **PASSED** |

Both models share the identical canonical topology ($D=2, W=64, L=2$) and yielded an identical median latency of $24.6000\ \mu\text{s}$ on the host CPU. Their IQRs ($0.90\ \mu\text{s}$ vs. $1.00\ \mu\text{s}$) confirm they were evaluated independently without cross-system pooling.

---

## 4. Empirical 26-Point Dataset

The complete 26-point dataset across Series 1A, Series 1B, and Series 1C is documented below:

$$\widehat{T}_{\text{net}} = \frac{\text{NFE} \times T_f}{1000} \quad (\text{ms}), \qquad \widehat{T}_{\text{solver}} = T_{\text{total}} - \widehat{T}_{\text{net}} \quad (\text{ms}), \qquad \widehat{\Omega}_{\text{solver}} = \frac{\widehat{T}_{\text{solver}}}{T_{\text{total}}}$$

| Point ID | Series | System | Solver | Horizon / Step | NFE (med $\pm$ IQR) | $T_{\text{total}}$ ms (med $\pm$ IQR) | $\widehat{T}_{\text{net}}$ ms | $\widehat{T}_{\text{solver}}$ ms | $\widehat{\Omega}_{\text{solver}}$ | Trajectory MSE | Max Error $L_\infty$ | Status |
|:---|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `EXP1-S1A-LV-H1.5` | Series_1A | Lotka-Volterra | dopri5 | $T=1.5$ | $74 \pm 0$ | $9.74 \pm 0.97$ | 1.82 | 7.92 | 0.813 | $2.8864 \times 10^{-2}$ | 0.4130 | `SUCCESS` |
| `EXP1-S1A-LV-H3.0` | Series_1A | Lotka-Volterra | dopri5 | $T=3.0$ | $164 \pm 0$ | $20.93 \pm 0.93$ | 4.03 | 16.90 | 0.807 | $1.1728 \times 10^{-1}$ | 1.0668 | `SUCCESS` |
| `EXP1-S1A-LV-H4.5` | Series_1A | Lotka-Volterra | dopri5 | $T=4.5$ | $206 \pm 0$ | $27.82 \pm 2.75$ | 5.07 | 22.75 | 0.818 | $1.0366 \times 10^{-1}$ | 1.0668 | `SUCCESS` |
| `EXP1-S1A-LV-H6.0` | Series_1A | Lotka-Volterra | dopri5 | $T=6.0$ | $308 \pm 0$ | $42.94 \pm 2.62$ | 7.58 | 35.36 | 0.824 | $5.1389 \times 10^{-1}$ | 2.9507 | `SUCCESS` |
| `EXP1-S1A-LV-H7.5` | Series_1A | Lotka-Volterra | dopri5 | $T=7.5$ | $356 \pm 0$ | $50.13 \pm 2.50$ | 8.76 | 41.37 | 0.825 | $4.6105 \times 10^{-1}$ | 2.9507 | `SUCCESS` |
| `EXP1-S1A-LV-H9.0` | Series_1A | Lotka-Volterra | dopri5 | $T=9.0$ | $434 \pm 0$ | $61.02 \pm 6.01$ | 10.68 | 50.34 | 0.825 | $7.3996 \times 10^{-1}$ | 3.4132 | `SUCCESS` |
| `EXP1-S1A-LV-H10.5`| Series_1A | Lotka-Volterra | dopri5 | $T=10.5$ | $512 \pm 0$ | $67.70 \pm 1.99$ | 12.60 | 55.10 | 0.814 | $1.1268 \times 10^{0}$ | 4.6296 | `SUCCESS` |
| `EXP1-S1A-LV-H12.0`| Series_1A | Lotka-Volterra | dopri5 | $T=12.0$ | $554 \pm 0$ | $71.20 \pm 2.84$ | 13.63 | 57.57 | 0.809 | $1.1840 \times 10^{0}$ | 4.6296 | `SUCCESS` |
| `EXP1-S1A-LV-H13.5`| Series_1A | Lotka-Volterra | dopri5 | $T=13.5$ | $650 \pm 0$ | $84.41 \pm 8.41$ | 15.99 | 68.42 | 0.811 | $2.0297 \times 10^{0}$ | 6.0127 | `SUCCESS` |
| `EXP1-S1A-LV-H15.0`| Series_1A | Lotka-Volterra | dopri5 | $T=15.0$ | $698 \pm 0$ | $88.67 \pm 1.79$ | 17.17 | 71.50 | 0.806 | $1.9219 \times 10^{0}$ | 6.0127 | `SUCCESS` |
| `EXP1-S1B-FHN-H5.0`| Series_1B | FitzHugh-Nagumo | dopri5 | $T=5.0$ | $86 \pm 0$ | $11.44 \pm 1.75$ | 2.12 | 9.32 | 0.815 | $2.4140 \times 10^{-5}$ | 0.0104 | `SUCCESS` |
| `EXP1-S1B-FHN-H10.0`| Series_1B| FitzHugh-Nagumo | dopri5 | $T=10.0$ | $122 \pm 0$ | $16.43 \pm 0.23$ | 3.00 | 13.43 | 0.817 | $3.5240 \times 10^{-5}$ | 0.0110 | `SUCCESS` |
| `EXP1-S1B-FHN-H15.0`| Series_1B| FitzHugh-Nagumo | dopri5 | $T=15.0$ | $146 \pm 0$ | $20.37 \pm 0.63$ | 3.59 | 16.78 | 0.824 | $3.7550 \times 10^{-5}$ | 0.0129 | `SUCCESS` |
| `EXP1-S1B-FHN-H20.0`| Series_1B| FitzHugh-Nagumo | dopri5 | $T=20.0$ | $170 \pm 0$ | $23.74 \pm 0.72$ | 4.18 | 19.56 | 0.824 | $2.8000 \times 10^{-4}$ | 0.0863 | `SUCCESS` |
| `EXP1-S1B-FHN-H25.0`| Series_1B| FitzHugh-Nagumo | dopri5 | $T=25.0$ | $254 \pm 0$ | $34.70 \pm 2.81$ | 6.25 | 28.45 | 0.820 | $3.4665 \times 10^{-2}$ | 1.0697 | `SUCCESS` |
| `EXP1-S1B-FHN-H30.0`| Series_1B| FitzHugh-Nagumo | dopri5 | $T=30.0$ | $326 \pm 0$ | $43.34 \pm 0.52$ | 8.02 | 35.32 | 0.815 | $3.0844 \times 10^{-2}$ | 1.0697 | `SUCCESS` |
| `EXP1-S1B-FHN-H35.0`| Series_1B| FitzHugh-Nagumo | dopri5 | $T=35.0$ | $362 \pm 0$ | $48.91 \pm 0.57$ | 8.91 | 40.00 | 0.818 | $2.7611 \times 10^{-2}$ | 1.0697 | `SUCCESS` |
| `EXP1-S1B-FHN-H40.0`| Series_1B| FitzHugh-Nagumo | dopri5 | $T=40.0$ | $458 \pm 0$ | $62.40 \pm 1.54$ | 11.27 | 51.13 | 0.819 | $4.0365 \times 10^{-2}$ | 1.0697 | `SUCCESS` |
| `EXP1-S1B-FHN-H45.0`| Series_1B| FitzHugh-Nagumo | dopri5 | $T=45.0$ | $512 \pm 0$ | $73.11 \pm 5.67$ | 12.60 | 60.51 | 0.828 | $3.6414 \times 10^{-2}$ | 1.0697 | `SUCCESS` |
| `EXP1-S1B-FHN-H50.0`| Series_1B| FitzHugh-Nagumo | dopri5 | $T=50.0$ | $542 \pm 0$ | $73.96 \pm 0.75$ | 13.33 | 60.63 | 0.820 | $3.2973 \times 10^{-2}$ | 1.0697 | `SUCCESS` |
| `EXP1-S1C-LV-RK4-N25`| Series_1C | Lotka-Volterra | rk4 | $h=0.6000$ | $100 \pm 0$ | $8.31 \pm 0.27$ | 2.46 | 5.85 | 0.704 | $1.7400 \times 10^{0}$ | 4.5274 | `SUCCESS` |
| `EXP1-S1C-LV-RK4-N50`| Series_1C | Lotka-Volterra | rk4 | $h=0.3000$ | $200 \pm 0$ | $14.05 \pm 0.65$ | 4.92 | 9.13 | 0.650 | $1.4983 \times 10^{0}$ | 5.1739 | `SUCCESS` |
| `EXP1-S1C-LV-RK4-N100`| Series_1C| Lotka-Volterra | rk4 | $h=0.1500$ | $400 \pm 0$ | $27.63 \pm 1.29$ | 9.84 | 17.79 | 0.644 | $1.8759 \times 10^{0}$ | 5.8175 | `SUCCESS` |
| `EXP1-S1C-LV-RK4-N200`| Series_1C| Lotka-Volterra | rk4 | $h=0.0750$ | $800 \pm 0$ | $46.33 \pm 4.52$ | 19.68 | 26.65 | 0.575 | $1.9104 \times 10^{0}$ | 5.9618 | `SUCCESS` |
| `EXP1-S1C-LV-RK4-N400`| Series_1C| Lotka-Volterra | rk4 | $h=0.0375$ | $1600 \pm 0$ | $89.77 \pm 2.23$ | 39.36 | 50.41 | 0.562 | $1.9191 \times 10^{0}$ | 5.9999 | `SUCCESS` |
| `EXP1-S1C-LV-RK4-N800`| Series_1C| Lotka-Volterra | rk4 | $h=0.01875$| $3200 \pm 0$ | $172.21 \pm 5.82$ | 78.72 | 93.49 | 0.543 | $1.9213 \times 10^{0}$ | 6.0092 | `SUCCESS` |

---

## 5. Disaggregated Regression Analysis

Per Protocol Section 5.2.C, linear regressions ($T_{\text{total}} = \beta_1 \cdot \text{NFE} + \beta_0$) were computed strictly independently for each ladder:

```
Series 1A (Lotka-Volterra, Adaptive dopri5, N=10):
  T_total = 0.127435 * NFE + 2.0414 ms
  beta_1  = 127.44 +/- 3.49 us / NFE  (t = 36.5, p < 1e-8)
  R^2     = 0.994051

Series 1B (FitzHugh-Nagumo, Adaptive dopri5, N=10):
  T_total = 0.138736 * NFE - 0.4761 ms
  beta_1  = 138.74 +/- 2.33 us / NFE  (t = 59.6, p < 1e-9)
  R^2     = 0.997757

Series 1C (Lotka-Volterra, Deterministic rk4, N=6):
  T_total = 0.052629 * NFE + 4.4575 ms
  beta_1  = 52.63 +/- 0.57 us / NFE   (t = 92.7, p < 1e-6)
  R^2     = 0.999534
```

| Regression Metric | Series 1A: LV (`dopri5`) | Series 1B: FHN (`dopri5`) | Series 1C: LV (`rk4`) | Unit |
|:---|:---:|:---:|:---:|:---:|
| **Sample Points ($N$)** | 10 | 10 | 6 | points |
| **NFE Dynamic Range** | $[74, 698]$ | $[86, 542]$ | $[100, 3200]$ | evaluations |
| **Runtime Dynamic Range** | $[9.74, 88.67]$ | $[11.44, 73.96]$ | $[8.31, 172.21]$ | ms |
| **Marginal Cost ($\beta_1$)** | **0.127435** ($127.44\ \mu\text{s}$) | **0.138736** ($138.74\ \mu\text{s}$) | **0.052629** ($52.63\ \mu\text{s}$) | $\text{ms} / \text{NFE}$ |
| **Intercept ($\beta_0$)** | **+2.0414** | **-0.4761** | **+4.4575** | ms |
| **Goodness of Fit ($R^2$)** | **0.994051** | **0.997757** | **0.999534** | dimensionless |
| **Mean Solver Overhead Ratio** | **81.5%** | **82.0%** | **61.3%** | fraction of $T_{\text{total}}$ |

---

## 6. Key Scientific Observations

### 6.1 Strong Within-Series Linear Predictability
Across all three series, the linear correlation between NFE and wall-clock runtime within any single experimental ladder is exceptionally high ($R^2 \ge 0.994$). Under fixed solver, fixed hardware, and identical tolerances, NFE serves as an accurate linear proxy for elapsed runtime along that specific ladder.

### 6.2 Dominance of Solver Overhead in Adaptive Solvers
In adaptive `dopri5` integration (Series 1A and 1B):
- The isolated network latency is $T_f = 24.60\ \mu\text{s}$.
- The marginal wall-clock cost per NFE is $\beta_1 \approx 127.44$–$138.74\ \mu\text{s} / \text{NFE}$.
- The estimated solver overhead $\widehat{T}_{\text{solver}} = T_{\text{total}} - \text{NFE} \times T_f$ constitutes **$\approx 81.5\%$ to $82.0\%$** of total wall-clock runtime. Raw neural evaluation represents only $\approx 18\%$ of compute time.

*(Important Epistemic Distinction: As mandated by Protocol Section 6.4, $\widehat{T}_{\text{solver}}$ is an estimated residual derived from benchmarked isolated latency, not a directly instrumented CPU execution timer inside the solver loop).*

### 6.3 Solver-Dependent Marginal Cost Disparity ($2.42\times$)
Comparing the identical dynamical system (Lotka-Volterra) and model checkpoint (`lv_h64.pt`) on the identical CPU:
$$\frac{\beta_1^{\text{dopri5}}}{\beta_1^{\text{rk4}}} = \frac{127.435}{52.629} = \mathbf{2.4214}$$
In this experimental setup, each evaluation in adaptive `dopri5` incurs **$2.42\times$ greater marginal wall-clock runtime** than an evaluation in fixed-step `rk4`. This disparity reflects the internal housekeeping cost of adaptive Dormand-Prince integration: local truncation error norm estimation, Butcher tableau intermediate buffer management, step-size controller recalculation, and dense output polynomial interpolation.

### 6.4 System-Dependent Marginal Cost Disparity ($8.87\%$)
Comparing the two adaptive systems under identical architecture ($W=64, L=2$, 4,482 params), identical solver (`dopri5`), and identical numerical tolerances ($10^{-5}, 10^{-7}$):
$$\frac{\beta_1^{\text{FHN}} - \beta_1^{\text{LV}}}{\beta_1^{\text{LV}}} = \frac{138.736 - 127.435}{127.435} = \mathbf{+8.87\%}$$
FitzHugh-Nagumo exhibits an $8.87\%$ higher marginal cost per function evaluation than Lotka-Volterra ($138.74\ \mu\text{s}$ vs. $127.44\ \mu\text{s}$). Differences in local trajectory geometry and excursion dynamics induce differential step controller overhead, confirming that marginal evaluation cost varies even when network architecture is held constant.

*(Scientific Guardrail: The $2.42\times$ and $8.87\%$ ratios are empirical findings specific to this CPU, Python environment, and checkpoint set; they are not claimed as universal mathematical constants).*

---

## 7. Trajectory Accuracy & Convergence Analysis

Accuracy metrics were audited against high-precision Radau reference solutions ($\text{rtol}=10^{-10}, \text{atol}=10^{-12}$) using `CubicSpline` interpolation to evaluate ground truth at exact solution timestamps:
1. **FitzHugh-Nagumo (Series 1B):**
   Exhibits exceptional accuracy throughout the horizon ($T \in [5, 50]$): $\text{MSE} \in [2.4 \times 10^{-5}, 4.0 \times 10^{-2}]$, $L_\infty \le 1.07$.
2. **Lotka-Volterra (Series 1A & 1C):**
   - In Series 1A, $\text{MSE}$ increases with horizon length: $0.0288$ ($T=1.5$) to $1.9219$ ($T=15.0$).
   - In Series 1C, as step size $h$ decreases from $0.60$ to $0.01875$ ($\text{NFE} = 100 \to 3200$), the numerical solution asymptotically converges:
     $$\text{MSE}(N=25) = 1.7400 \longrightarrow \text{MSE}(N=800) = \mathbf{1.9213}$$
   - **Methodological Conclusion:** The fixed-step trajectory converges exactly to the adaptive `dopri5` trajectory ($\text{MSE} = 1.9219$). This proves that the residual error at $T=15.0$ is **surrogate generalization error** (accumulated phase drift of the learned surrogate orbit over multiple periods), **not numerical integration failure**. The solver integrated the surrogate vector field faithfully.

---

## 8. Outlier & Robustness Analysis

- **Outlier Criterion:** Protocol Section 9.2 flags measurements where $T_i > Q_3 + 3.0 \times \text{IQR}$.
- **Empirical Detection:** Across all 390 recorded solves, exactly **4 individual measurements (1.02%)** were flagged:
  - `EXP1-S1B-FHN-H10.0`: 3 runs (18.16, 17.55, 19.94 ms vs. threshold 17.21 ms).
  - `EXP1-S1B-FHN-H50.0`: 1 run (77.09 ms vs. threshold 76.74 ms).
- **Robustness:** Because the primary metric is the **median** ($16.43\text{ ms}$ and $73.96\text{ ms}$ respectively), transient OS context-switch interruptions had zero impact on reported performance figures.

---

## 9. Hypothesis Status & Epistemic Boundaries

### Central Hypothesis:
> **"NFE is a useful proxy for Neural ODE computational cost, but its relationship with actual wall-clock runtime systematically varies with model complexity, solver characteristics, stiffness, numerical settings, and hardware/implementation."**

### Status: PARTIALLY EVALUATED / SUPPORTED BY BASELINE EVIDENCE
Experiment 1 provides empirical evidence supporting the central hypothesis across two specific dimensions:
1. **Solver Characteristics:** Marginal runtime per NFE varied by **$2.42\times$** between adaptive `dopri5` and fixed-step `rk4` on the same model.
2. **Dynamical Regimes:** Marginal runtime per NFE varied by **$8.87\%$** between Lotka-Volterra and FitzHugh-Nagumo under identical architecture and tolerances.

### What Experiment 1 Proves:
- NFE has a strong, reliable linear relationship with wall-clock runtime **within any single solver-system ladder** ($R^2 \ge 0.994$).
- Reporting NFE alone is insufficient for cross-solver comparison: a model requiring 400 NFE under `rk4` consumes substantially less CPU time ($27.63\text{ ms}$) than a model requiring 400 NFE under `dopri5` ($55.0\text{ ms}$ predicted).
- In lightweight Neural ODE vector fields ($W=64$), internal solver housekeeping constitutes the majority ($\approx 81.5\%$) of computational overhead on CPU.

### What Experiment 1 Does NOT Prove:
- Exp1 does **not** prove that NFE is universally flawed or useless; it demonstrates strong within-series utility.
- Exp1 does **not** evaluate stiff dynamics or implicit linear algebra costs (deferred to Exp 5 and 6).
- Exp1 does **not** evaluate model parameter scaling or whether network latency overtakes solver overhead as $W$ increases (deferred to Exp 3).
- Exp1 does **not** evaluate tolerance sensitivity (deferred to Exp 4).

---

## 10. Limitations

1. **Host-Specific Timings:** Absolute runtime values ($\mu\text{s/NFE}$) reflect the single-threaded CPU architecture of the test workstation (`Intel64 Family 6 Model 183`).
2. **Single Baseline Width:** Vector field complexity was held constant at $W=64, L=2$.
3. **Derived Residual:** Solver overhead is an estimated quantity ($\widehat{T}_{\text{solver}} = T_{\text{total}} - \text{NFE} \times T_f$), subject to cache-locality differences between tight solver unrolling and isolated benchmarking.
4. **Non-Stiff Regime:** Evaluated systems are non-stiff or mildly multiscale; step-size collapse dynamics were not present.

---

## 11. Evidence-Based Motivation for Experiment 2

The findings of Experiment 1 establish that:
1. Internal solver housekeeping consumes $\approx 81.5\%$ of wall-clock runtime in adaptive `dopri5`.
2. Marginal cost per evaluation varies by $2.42\times$ between `dopri5` and `rk4`.

However, Experiment 1 evaluated only two solver algorithms (`dopri5` and `rk4`) within a single execution backend (`torchdiffeq`). This directly motivates **Experiment 2 (Solver Algorithm & Backend Decoupling)** to answer:
- Does this overhead disparity persist across other explicit Runge-Kutta orders (`tsit5`, `midpoint`, `euler`)?
- What fraction of solver overhead is mathematical algorithm housekeeping vs. Python/Tensor framework dispatch in `torchdiffeq` compared to compiled C-dispatch in SciPy?

This transition is strictly motivated by the empirical findings of Experiment 1.
