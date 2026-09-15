# Phase 3: Evidence Verification & Claim Corrections

**Status:** Completed & Audited  
**Verification Level:** `[VERIFIED]`  
**Context:** Rigorous evidence audit, claim retractions, and epistemic demarcation across all project documentation.

---

## 1. Objective
Conduct an exhaustive forensic audit of all claims, citations, heuristic ratios, and performance metrics across the project codebase and reports to ensure zero ungrounded or hallucinated assertions exist before experimental freezing.

---

## 2. Core Audit Findings & Retractions

### A. Lotka-Volterra Citation
* **Prior Claim:** Stated that Chen et al. (2018) established the Lotka-Volterra Neural ODE.
* **Audit Finding:** Chen et al. (2018) benchmarked on a 2D spiral (decaying linear oscillator), a 1D function, and continuous normalizing flows; they never evaluated Lotka-Volterra.
* **Correction:** Replaced with Rackauckas et al. (2020, Paper #14) and standard dynamical systems literature.

### B. FitzHugh-Nagumo Stiffness & Timescale Ratio
* **Prior Claim:** Claimed FHN had a measured "$35\times$ stiffness ratio" and was classically stiff.
* **Audit Finding:** Dividing the recovery parameter $\tau = 12.5$ by the local spectral radius $\tau_{\text{fast}} \approx 0.35$ produces $35.6$, but this is an informal heuristic, not a mathematically rigorous stiffness ratio. Spectral radius $\le 2.85$ does not indicate classical stiffness.
* **Correction:** Retracted the $35\times$ claim. Reclassified FHN as **Candidate Non-Stiff / Mildly Multiscale System**. Explicitly noted that Phase-1 diagnostics do not establish solver collapse.

### C. Van der Pol Legacy Speedup
* **Prior Claim:** Stated that Neural ODEs on Van der Pol exhibited a 14.4× speedup for implicit Radau over explicit RK45 at $\mu=100$.
* **Audit Finding:** Inspection of `experiments/archive/exp3_stiffness_breakdown.py` revealed that lines 55 and 65 passed `vdp.rhs` to `solve_ivp`. The solver integrated the classical analytical ODE; the neural network was never evaluated inside the integration loop.
* **Correction:** Reclassified all prior 14.4× speedup results as **`[CLASSICAL ODE MEASUREMENT]`**, not Neural ODE evidence. Neural ODE stiffness behavior on VdP is classified as **`[NOT YET MEASURED]`**.

### D. Robertson Eigenvalue Claims
* **Prior Claim:** Stated Robertson had trajectory eigenvalues $\lambda_1 \approx -10^7, \lambda_2 \approx -10^{-2}$ and a stiffness ratio $> 10^9$.
* **Audit Finding:** No script or log file in the repository ever computed or recorded this spectral decomposition along the trajectory.
* **Correction:** Marked as **`[INSUFFICIENT VERIFIED INFORMATION — CANNOT CONCLUDE]`**. Robertson's stiffness is grounded strictly in literature proof (Kim et al., 2021) of explicit solver breakdown.

### E. Activation Function & CPU Throughput
* **Prior Claim:** Claimed GELU was selected because it is faster on CPU and that $C^1$ smoothness is "mandatory."
* **Audit Finding:** GELU latency on CPU relative to Tanh/Softplus was never profiled. Smoothness is preferred for implicit Newton stability, but not mathematically mandatory.
* **Correction:** Retracted CPU speed superiority claim (`NOT YET MEASURED`); softened smoothness to literature-supported preference.

### F. Canonical Baseline Status
* **Prior Claim:** Implied the canonical baseline architecture was directly taken from literature.
* **Audit Finding:** Synthesizes structural ranges from 5 papers (Kim, Finlay, Caldana, Dupont, Lienen); no single paper used this exact combination.
* **Correction:** Formally labeled as **`[OUR PROPOSED CANONICAL BASELINE]`**.
