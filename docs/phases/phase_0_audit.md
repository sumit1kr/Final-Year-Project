# Phase 0: Workspace & Legacy Audit

**Status:** Completed & Quarantined  
**Verification Level:** `[PARTIALLY VERIFIED]`  
**Context:** Workspace hygiene, legacy script isolation, reference dataset validation.

---

## 1. Objective
Establish clean repository hygiene by:
1. Auditing all legacy scripts and isolating flawed preliminary experiments into dedicated archive folders.
2. Verifying the physical integrity and finiteness of ground-truth reference datasets.
3. Ensuring core infrastructure modules exist without syntax or import errors.

---

## 2. Actions Implemented (`phase0_audit.py`)
* **Archival Quarantine:**
  * Created `experiments/archive/` and `experiments/logs/archive/`.
  * Moved 5 legacy scripts: `exp1_solver_sweep.py`, `exp2_critical_workload.py`, `exp3_stiffness_breakdown.py`, `exp3_tolerance_scaling.py`, and `exp3_batch_scaling.py`.
  * Moved 6 historical CSV logs into `experiments/logs/archive/`.
* **Ground-Truth Dataset Integrity Check:**
  * Verified 4 trajectory `.pt` files in `data/trajectories/` (Lotka-Volterra, FitzHugh-Nagumo, Van der Pol, Robertson).
  * Checked tensor shapes, float dtypes, and asserted absence of `NaN` or non-finite values.

---

## 3. Quarantined Artifacts & Legacy Findings

| Archived Script | Quarantined CSV Log | Rows | Key Measured Observation | Methodological Limitation / Flaw |
|:---|:---|:---:|:---|:---|
| `exp1_solver_sweep.py` | `exp1_solver_sweep.csv` | 16 | $T_{\text{solver}}$ accounts for 73.8%–83.3% of total time in adaptive Dopri5. | Uncontrolled parameter capacity variations across systems. |
| `exp2_critical_workload.py` | `exp2_critical_workload.csv` | 4 | 354-param net (62 NFE) ran in 13.6 ms; 68k-param net (40 NFE) ran in 15.6 ms. | Demonstrates NFE inversion, but tested synthetic MLP width extremes. |
| `exp3_batch_scaling.py` | `exp3_batch_scaling.csv` | 9 | Batching amortized per-sample latency from 13.46 ms down to 0.054 ms. | Batch interactions conflated with solver step control. |
| `exp3_stiffness_breakdown.py` | `exp3_stiffness_breakdown.csv` | 12 | 14.4× wall-clock speedup for implicit Radau over explicit RK45 at $\mu=100$. | **CRITICAL AUDIT FINDING:** Evaluated classical SciPy ODE (`vdp.rhs`), NOT Neural ODE! |
| `exp3_tolerance_scaling.py` | `exp3_tolerance_scaling.csv` | 14 | Tightening tolerance caused 10× NFE spike on stiff VdP vs 2.5× on LV. | Confounded network capacity with solver tolerance. |

---

## 4. Verification Limitations & Missing Evidence
* **Missing Execution Log:** While `phase0_audit.py` exists and its filesystem modifications (file moves, archive folder creation) are physically verified, **NO DIRECT stdout text log (`phase0_audit.log`) was preserved** on disk.
* **Epistemic Classification:** Classified as **`[PARTIALLY VERIFIED]`** in accordance with the project's strict evidence rules.
