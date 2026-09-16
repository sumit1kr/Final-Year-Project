# Master Evidence & Proof-of-Work Index

**Academic Context:** Final Year Project (FYP) — Track 2: Computational Characterization  
**Department:** Department of Computer Science and Engineering  
**Institution:** Indian Institute of Information Technology, Design and Manufacturing (IIITDM), Kancheepuram  
**Guide:** Dr. Noor Mahammad Shaik  
**Candidate:** Sumit Kumar (Roll No.: CS23B2008)  
**Date:** September 2026  
**Document Purpose:** Permanent, forensic index of verified research artifacts, execution logs, measured results, and literature groundings.  
**Auditing Rule:** A prompt, plan, or specification proves only that a design was made; an execution log proves execution occurred; a CSV/JSON file proves recorded measurements; a published paper proves literature claims. Agent claims without corresponding project artifacts are classified as `[AGENT CLAIM ONLY]` or `[NOT VERIFIED]`.

---

## 1. Master Evidence Matrix

| Phase / Stage | Claimed Research Action | Artifact Path | Evidence Type | Status | Forensic Verification Details & Limitations |
|:---|:---|:---|:---:|:---:|:---|
| **Problem Statement** | Formalize Problem Statement, central RQ, hypothesis, and CPU-only scope | • `docs/problem_statement.tex`<br>• `docs/problem_statement.pdf` | `[DIRECT ARTIFACT]` | **VERIFIED** | Compiled 2-page brief approved by mentor (72,371 B, compiled Sept 10, 2026). Frozen RQ and hypothesis recorded. |
| **Literature Corpus** | Catalog 15 papers across Groups A/B/C, rank Top-5, download all physical PDFs | • `literature/README.md`<br>• `literature/Group_{A,B,C}/*.pdf`<br>• `literature/Top_5_Papers/*.pdf` | `[DIRECT ARTIFACT]`<br>`[VERIFIED SOURCE]` | **VERIFIED** | 15 local PDFs verified in Group folders + 5 duplicates in `Top_5_Papers/`. Bibliographic metadata, DOIs, and venues documented. |
| **Phase 0 (Audit)** | Quarantine flawed legacy scripts and preserve historical CSV logs | • `experiments/archive/*.py`<br>• `experiments/logs/archive/*.csv`<br>• `RESEARCH_VALIDATION_REPORT.md` | `[DIRECT ARTIFACT]`<br>`[MEASURED RESULT (LEGACY)]` | **VERIFIED** | 5 legacy scripts and 6 CSVs physically quarantined. Confirmed VdP 14.4× speedup was classical SciPy ODE, not Neural ODE. |
| **Phase 0 (Hygiene)** | Run workspace cleanup script and verify 4 reference datasets | • `phase0_audit.py`<br>• `data/trajectories/*.pt` | `[DIRECT ARTIFACT]` | **PARTIALLY VERIFIED** | Script exists and file reorganization is physically present. **NO DIRECT EXECUTION ARTIFACT FOUND** (no `phase0_audit.log` was saved). |
| **Phase 1 (Spectral)** | Analytical Jacobian eigenvalue analysis on LV and FHN trajectories | • `experiments/phase1_spectral_diagnostics.py`<br>• `experiments/logs/phase1_spectral_diagnostics.json` | `[DIRECT ARTIFACT]`<br>`[MEASURED RESULT]` | **VERIFIED (LV, FHN)** | Measured spectral radius, real/imag envelopes, and timescales recorded for LV and FHN. **VdP and ROBER are NOT YET MEASURED.** |
| **Phase 2 (Surrogates)** | Multi-tier surrogate training & validation with process watchdog | • `experiments/phase2_train_surrogates.py`<br>• `experiments/logs/surrogate_validation_report.json`<br>• `models/checkpoints/*.pt` | `[DIRECT ARTIFACT]`<br>`[MEASURED RESULT]` | **VERIFIED** | Measured metrics recorded for 12 models. Confirmed `robertson_h64` failed validation; confirmed `vdp_fallback` passed and was admitted. |
| **Phase 2 (Watchdog)** | Subprocess watchdog termination on Windows under stalled/runaway workloads | • `tests/test_watchdog_mechanism.py`<br>• `tests/test_robertson_validation.py`<br>• `experiments/logs/surrogate_validation_report.json` | `[DIRECT ARTIFACT]`<br>`[MEASURED RESULT]` | **PARTIALLY VERIFIED** | Watchdog code exists; operational timeout logged in surrogate report (`VALIDATION_TIMEOUT` at 5.031s). **No standalone test log saved.** |
| **Phase 3 (Claim Audit)** | Retract unsupported claims, unverified ratios, and invalid citations | • `docs/phases/phase_3_evidence_audit.md`<br>• Diff in `docs/benchmark_system_spec.md` | `[DOCUMENTED DECISION]`<br>`[DIRECT ARTIFACT]` | **VERIFIED** | Retracted Chen et al. for LV, FHN 35× ratio, Robertson $\lambda \approx -10^7$, and unmeasured GELU speed claims. |
| **Phase 4 (System Freeze)**| Freeze 4 benchmark systems, canonical baseline MLP, parameter math, CPU boundary | • `docs/benchmark_system_spec.md` (274 lines, 20,298 B) | `[DIRECT ARTIFACT]`<br>`[DOCUMENTED DECISION]` | **VERIFIED** | Baseline ($W=64, L=2$, GELU, $[z; t]$ input) frozen as proposed reference model. Parameter counts verified ($4,546$ and $4,675$). |
| **Stage 1 (Controlled Exp)**| Stage 1 benchmark experiments (NFE vs. runtime characterization on CPU) | N/A (Not yet implemented) | `[DOCUMENTED DESIGN ONLY]` | **NOT YET EXECUTED** | Clean slate: zero Stage 1 code executed. NFE-runtime correlation on frozen baseline is **NOT YET MEASURED**. |

---

## 2. Granular Breakdown of Measured Evidence

### Phase 1: Spectral Diagnostics (`experiments/logs/phase1_spectral_diagnostics.json`)
* **Execution Timestamp Traceability:** Measured elapsed diagnostic time: `0.0375` seconds on CPU.
* **Measured Systems:**
  1. **Lotka-Volterra (2D, Non-stiff reference):**
     * Trajectory: 150 sample points over $t \in [0.0, 15.0]$.
     * Real eigenvalue envelope: $[-2.287497, +2.571851]$, mean real mode: $+0.021151$.
     * Imaginary eigenvalue envelope: $[-4.095858, +4.095858]$, mean absolute imaginary: $1.053810$.
     * Fraction of points with complex eigenvalues: $56.67\%$.
     * Spectral radius range: $[0.663238, 4.136061]$, mean spectral radius: $2.121030$.
     * Fastest local spectral timescale: $\tau = 0.241776\text{ s}$.
  2. **FitzHugh-Nagumo (2D, Candidate non-stiff / mildly multiscale):**
     * Trajectory: 200 sample points over $t \in [0.0, 50.0]$.
     * Real eigenvalue envelope: $[-2.852136, +0.916185]$, mean real mode: $-0.565620$.
     * Imaginary eigenvalue envelope: $[-0.282825, +0.282825]$, mean absolute imaginary: $0.053717$.
     * Fraction of points with complex eigenvalues: $23.50\%$.
     * Spectral radius range: $[0.222329, 2.852136]$, mean spectral radius: $1.221950$.
     * Fastest local spectral timescale: $\tau_{\text{fast}} = 0.350614\text{ s}$.
* **Unmeasured Elements:**
  * Van der Pol spectral envelope along trajectory: `NOT YET MEASURED`.
  * Robertson spectral envelope along trajectory: `NOT YET MEASURED`.

---

### Phase 2: Surrogate Validation Report (`experiments/logs/surrogate_validation_report.json`)
* **Validation Standards (4 Tiers):**
  * **Tier 1:** Derivative field alignment ($R^2 \ge 0.95$ per component).
  * **Tier 2:** Trajectory integration accuracy ($\text{NRMSE} \le 0.05$ under `dopri5`).
  * **Tier 3:** Physical invariant preservation (energy bounds, limit cycle bounds).
  * **Tier 4:** Numerical convergence stability under tight tolerances ($\text{atol}=10^{-7}, \text{rtol}=10^{-5}$) with process watchdog.
* **Key Measured Findings:**
  1. **`lv_h64` (Lotka-Volterra Baseline):**
     * Tier 1 $R^2 = 0.999$, Tier 2 $\text{NRMSE} = 0.0414$, Tier 4 = `CONVERGED` (1.734s), $\rho_{\max} = 4.118$.
     * **Result:** `all_tiers_passed: true` (Admitted as non-stiff baseline).
  2. **`fhn_h64` (FitzHugh-Nagumo Baseline):**
     * Tier 1 $R^2 = 0.999$, Tier 2 $\text{NRMSE} = 0.00196$, Tier 4 = `CONVERGED` (1.694s), $\rho_{\max} = 2.517$.
     * **Result:** `all_tiers_passed: true` (Admitted as mildly multiscale baseline).
  3. **`robertson_h64` (Robertson Stiff Candidate — Failed Validation):**
     * Training: 700 epochs, 7,425 ms wall time, validation loss = $38,582.8$.
     * Tier 1: $R^2 = 0.985$ (Passed).
     * Tier 2: $\text{NRMSE} = 0.0965$ (Failed; threshold is $0.05$).
     * Tier 4: **`VALIDATION_TIMEOUT`** (Watchdog forcibly killed child process after 5.031s).
     * **Result:** `all_tiers_passed: false` (**Failed validation; excluded from stiff baseline**).
  4. **`vdp_mu100_stiff_h128_fallback` (Van der Pol $\mu=100$ Stiff Surrogate — Passed):**
     * Training: 800 epochs, 9,907 ms wall time, validation loss = $0.0024$.
     * Tier 1: $R^2 = 0.994$ (Passed).
     * Tier 2: $\text{NRMSE} = 0.00091$, `CONVERGED` (2.051s) (Passed).
     * Tier 3: Limit cycle amplitude = $2.0$ (Passed).
     * Tier 4: `CONVERGED` (2.819s), $\rho_{\max} = 271.406$ (Passed).
     * **Result:** `all_tiers_passed: true` (**Admitted as verified stiff Neural ODE benchmark**).

---

### Legacy Measurements Quarantined in `experiments/logs/archive/`
* **Historical CSV Logs:**
  * `exp1_solver_sweep.csv` (16 configurations: solver latency across euler, rk4, dopri5).
  * `exp2_critical_workload.csv` (4 configurations: Workload A [354 params, 62 NFE, 13.6 ms] vs. Workload B [68k params, 40 NFE, 15.6 ms]).
  * `exp3_batch_scaling.csv` (9 batch sizes: batch throughput amortizing from 13.5 ms down to 0.054 ms/sample).
  * `exp3_stiffness_breakdown.csv` (12 evaluations across $\mu \in [1, 100]$: 14.4× speedup of implicit Radau over explicit RK45 at $\mu=100$).
  * `exp3_tolerance_scaling.csv` (14 tolerance levels: tolerance tightening from $10^{-3}$ to $10^{-9}$).
* **Critical Distinction:** The 14.4× speedup in `exp3_stiffness_breakdown.csv` was measured on analytical equations (`vdp.rhs`) via `scipy.integrate.solve_ivp`. It provides **classical ODE evidence**, NOT Neural ODE evidence.

---

## 3. Retracted & Corrected Claims Log

| Claim / Topic | Original Unverified Statement | Forensic Source Audit | Corrected Status |
|:---|:---|:---|:---|
| **Lotka-Volterra Citation** | "Chen et al. (2018) established LV Neural ODE" | Chen et al. (2018) studied 2D spiral, 1D function, CNFs; did not study LV. | **CITED RACKAUCKAS ET AL. (2020)** |
| **FitzHugh-Nagumo Timescale** | "FHN has a $35\times$ stiffness ratio" | Dividing ODE recovery parameter $\tau=12.5$ by local spectral radius is not a rigorous stiffness ratio. | **RETRACTED AS UNGROUNDED HEURISTIC** |
| **FHN Stiffness Classification** | "FHN is a stiff dynamical system" | Measured spectral radius $\le 2.85$; solves cleanly with explicit solvers without step collapse. | **RECLASSIFIED: CANDIDATE NON-STIFF / MILDLY MULTISCALE** |
| **Van der Pol Speedup** | "Neural ODEs show 14.4× implicit speedup at $\mu=100$" | Code audit of `exp3_stiffness_breakdown.py` revealed solver evaluated analytical ODE `vdp.rhs`, not neural net. | **RECLASSIFIED: CLASSICAL SCIPY ODE RESULT ONLY** |
| **Robertson Eigenvalues** | "Robertson eigenvalues $\lambda \approx -10^7$; stiffness ratio $> 10^9$" | No script or log file in the repository ever computed or logged this spectral decomposition. | **MARKED: INSUFFICIENT VERIFIED INFORMATION** |
| **GELU CPU Latency** | "GELU was selected because it is faster on CPU" | CPU throughput across activations was never profiled or logged in project benchmarks. | **MARKED: NOT YET MEASURED** |
| **Baseline Architecture** | "Canonical architecture is a literature standard" | Synthesizes parameter ranges across 5 papers; no single paper used this exact configuration. | **LABELED: OUR PROPOSED CANONICAL BASELINE** |

---

## 4. Known Missing Evidence & Verification Gaps

1. **Phase 0 Execution Log:** `phase0_audit.py` was executed to organize the workspace, but no stdout text file (`phase0_audit.log`) was captured. Status remains `PARTIALLY VERIFIED`.
2. **Watchdog Unit Test Log:** `tests/test_watchdog_mechanism.py` has no standalone console log. Firing is verified solely through `surrogate_validation_report.json`. Status remains `PARTIALLY VERIFIED`.
3. **Van der Pol & Robertson Analytical Spectra:** Trajectory Jacobian spectra have not yet been evaluated in `phase1_spectral_diagnostics.json`.
4. **Host Hardware Metadata:** Specific CPU model string, microarchitecture, cache hierarchy, and linked BLAS backend will be formally queried and recorded when Stage 1 experiments begin.
