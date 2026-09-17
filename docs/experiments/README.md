# Experiment Documentation & Repository Map

**Academic Context:** Final Year Project (FYP) — Track 2: Computational Characterization  
**Department:** Department of Computer Science and Engineering  
**Institution:** Indian Institute of Information Technology, Design and Manufacturing (IIITDM), Kancheepuram  
**Guide:** Dr. Noor Mahammad Shaik  
**Candidate:** Sumit Kumar (Roll No.: CS23B2008)  
**Document Purpose:** Structural Guide to Experimental Artifacts and Epistemic Demarcation  

---

## 1. Directory Structure & Roles

This repository is organized to ensure complete transparency, cryptographic traceability, and strict separation between code, raw data, scientific analysis, and archived legacy experiments:

| Directory | Primary Role | Contents & Epistemic Status |
|:---|:---|:---|
| **`experiments/`** | **Executable Experiment Runners** | Active, production Python runners (`stage1_exp1_runner.py`) instrumented with thread pinning, monotonic timers, and watchdog subprocess monitoring. |
| **`experiments/logs/`** | **Raw Empirical Outputs** | Direct, unmanipulated machine-generated outputs (`stage1_exp1_results.json`, `stage1_exp1_results.csv`) and diagnostic JSON records. Cryptographically locked. |
| **`docs/experiments/`** | **Analyses & Evidence Manifests** | Human-readable scientific interpretations, disaggregated regression tables, hypothesis evaluations, and cryptographic audit manifests. |
| **`models/`** | **Architectures & Validated Weights** | Autonomous MLP vector-field definitions (`models/vector_fields.py`) and pre-trained Phase-2 model checkpoints (`models/checkpoints/*.pt`). |
| **`data/`** | **Ground-Truth Datasets** | High-precision numerical reference trajectories generated via SciPy Radau (`rtol=1e-10, atol=1e-12`) used for validation and trajectory error benchmarking. |
| **`docs/`** | **Specifications & Governance** | Master research protocols (`stage1_experiment_protocol.md`), architecture decision records (`architecture_decision_record.md`), benchmark specifications (`benchmark_system_spec.md`), and master evidence indexes. |
| **`experiments/archive/`** | **Quarantined Historical Code** | Flawed legacy experiment scripts preserved strictly for forensic auditing; **NOT** part of the active evidence chain. |
| **`experiments/logs/archive/`**| **Quarantined Historical CSVs** | Historical logs from initial exploratory runs (e.g., legacy classical SciPy ODE benchmarks); **NOT** Neural ODE evidence. |

---

## 2. Epistemic Demarcation: The Four Tiers of Information

To prevent unsubstantiated claims and maintain rigorous academic integrity, all information in this repository is strictly segregated into four epistemic categories:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. RAW EMPIRICAL EVIDENCE (Direct Machine Measurements)     │
│    • High-precision nanosecond timer intervals              │
│    • Integer wrapper forward pass counters (NFE)            │
│    • Host system metadata and OS environment captures       │
│    Location: experiments/logs/*.json, *.csv                 │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Mathematical transformation)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. DERIVED ANALYTICAL METRICS (Computed Statistics)         │
│    • Non-parametric medians and IQRs across repetitions     │
│    • Synthetic network latency: T_net = NFE * T_f           │
│    • Synthetic solver overhead: T_solver = T_total - T_net  │
│    • Disaggregated OLS linear regressions (beta_1, R^2)     │
│    Location: Derived columns in CSV / Analysis reports       │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Scientific reasoning & deduction)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. SCIENTIFIC INTERPRETATION (Deductions & Hypotheses)      │
│    • Linearity within fixed ladders                         │
│    • 2.42x marginal cost disparity between solvers          │
│    • Hypothesis evaluation and epistemic limitations        │
│    Location: docs/experiments/phase1_exp1_analysis.md       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 4. HISTORICAL & QUARANTINED MATERIAL (Separated/Archived)   │
│    • Legacy exploratory scripts and logs                    │
│    • Classical ODE benchmarks (SciPy solve_ivp)             │
│    Location: experiments/archive/, experiments/logs/archive/│
└─────────────────────────────────────────────────────────────┘
```

### Tier 1: Raw Empirical Evidence
Direct observations recorded by instrumented hardware and operating system interfaces without human intervention:
- Total wall-clock time ($T_{\text{total}}$) captured via monotonic `time.perf_counter_ns()`.
- Wrapper forward evaluation count ($\text{NFE}$) captured via `CountWrapper.nfe`.
- Raw individual measurement timings preserved in array `raw_runtimes_ms`.
- Host CPU model, logical cores, PyTorch/torchdiffeq versions, and OS build embedded in JSON.

### Tier 2: Derived Analytical & Statistical Metrics
Deterministic mathematical transformations computed directly from Tier 1 raw evidence:
- **Non-Parametric Medians and IQRs:** Used to guard against asymmetric operating system interrupts.
- **Estimated Network Runtime ($\widehat{T}_{\text{net}}$):** Calculated as $\frac{\text{NFE} \times T_f}{1000.0}\text{ ms}$. *(Epistemic note: This is an analytical estimation based on isolated benchmarking, not an in-situ timer measurement).*
- **Estimated Solver Overhead ($\widehat{T}_{\text{solver}}$):** Residual difference $T_{\text{total}} - \widehat{T}_{\text{net}}$. *(Epistemic note: This represents aggregate non-network elapsed time, not instrumented CPU cycles inside the solver loop).*
- **Disaggregated OLS Linear Regressions:** Independently fitted slope $\beta_1$, intercept $\beta_0$, and $R^2$ per series.

### Tier 3: Scientific Interpretations & Inferences
Empirical deductions and conclusions drawn from Tier 1 and Tier 2:
- Within-series linear predictability ($R^2 \ge 0.994$).
- Marginal cost disparity between adaptive and fixed-step algorithms ($2.42\times$).
- Evaluation of Central Hypothesis boundaries and limitations.

### Tier 4: Historical & Quarantined Material
Artifacts preserved strictly for forensic transparency and historical provenance:
- Legacy scripts in `experiments/archive/` (e.g., `exp1_solver_sweep.py`, `exp3_stiffness_breakdown.py`).
- Legacy logs in `experiments/logs/archive/` (e.g., `exp3_stiffness_breakdown.csv` where the 14.4× speedup was classical SciPy ODE, not Neural ODE).
- These files are completely quarantined and are **never** cited as evidence for Neural ODE claims.

---

## 3. Experiment Portfolio Status on this Branch

| Experiment | Focus Area | Status | Documentation & Evidence Links |
|:---|:---|:---:|:---|
| **Stage 1 Exp 1** | Baseline NFE vs. Runtime Characterization | **COMPLETED & AUDITED** | • [Dedicated Exp1 Guide](exp1/README.md)<br>• [Full Scientific Analysis](phase1_exp1_analysis.md)<br>• [Evidence Manifest](exp1_evidence_manifest.md)<br>• [Raw JSON](../../experiments/logs/stage1_exp1_results.json)<br>• [Raw CSV](../../experiments/logs/stage1_exp1_results.csv)<br>• [Runner](../../experiments/stage1_exp1_runner.py) |
| *Stage 1 Exp 2A* | Solver Algorithm Decoupling (7 solvers) | *Planned Future Work* | Outlined in `docs/stage1_experiment_protocol.md` (Section 11.2). |
| *Stage 1 Exp 3* | Model Complexity Scaling ($W \in [16, 256]$) | *Planned Future Work* | Outlined in `docs/stage1_experiment_protocol.md` (Section 11.3). |
| *Stage 1 Exp 4* | Tolerance Sweeps ($\text{rtol}, \text{atol}$) | *Planned Future Work* | Outlined in `docs/stage1_experiment_protocol.md` (Section 11.4). |
| *Stage 1 Exp 5* | Stiffness Breakdown on Van der Pol ($\mu$) | *Planned Future Work* | Outlined in `docs/stage1_experiment_protocol.md` (Section 11.5). |
| *Stage 1 Exp 6* | Extreme Multiscale Breakdown on Robertson | *Planned Future Work* | Outlined in `docs/stage1_experiment_protocol.md` (Section 11.6). |
