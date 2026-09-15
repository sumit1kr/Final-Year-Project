# Phase 2: Surrogate / Neural Vector-Field Validation

**Status:** Completed & Validated  
**Verification Level:** `[VERIFIED]`  
**Context:** Multi-tier surrogate validation, process watchdog integration, checkpoint archiving.

---

## 1. Objective
Validate candidate neural vector fields to ensure that fitted Neural ODE surrogates satisfy rigorous dynamical and numerical criteria before admission into benchmarking. Prevent unstable, diverging, or invalid models from entering the computational profiling testbed.

---

## 2. Four-Tier Validation Architecture
1. **Tier 1 (Derivative Alignment):** Derivative $R^2 \ge 0.95$ per state component against ground-truth vector fields.
2. **Tier 2 (Trajectory Integration):** Normalized Root-Mean-Square Error ($\text{NRMSE} \le 0.05$) under adaptive `dopri5` integration.
3. **Tier 3 (Invariant Preservation):** Physical domain bounds, positivity, or limit cycle amplitudes within physical tolerance.
4. **Tier 4 (Numerical Stability & Watchdog):** Convergence under tight solver tolerances ($\text{atol}=10^{-7}, \text{rtol}=10^{-5}$) within a bounded execution window monitored by a dedicated subprocess watchdog.

---

## 3. Measured Results (`experiments/logs/surrogate_validation_report.json`)

| Model Checkpoint | System | $W$ | Params | Tier 1 ($R^2$) | Tier 2 (NRMSE) | Tier 4 Outcome | All Tiers Passed? | Admitted Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `lv_h64.pt` | Lotka-Volterra | 64 | 4,482 | 0.999 | 0.0414 | `CONVERGED` (1.73s) | **TRUE** | **Admitted (Non-Stiff Reference)** |
| `fhn_h64.pt` | FitzHugh-Nagumo | 64 | 4,482 | 0.999 | 0.00196 | `CONVERGED` (1.69s) | **TRUE** | **Admitted (Mildly Multiscale)** |
| `robertson_h64.pt` | Robertson | 64 | 8,771 | 0.985 | 0.0965 (FAIL) | `VALIDATION_TIMEOUT` (5.03s) | **FALSE** | **REJECTED (Failed Validation)** |
| `vdp_mu100_stiff_h128.pt` | Van der Pol ($\mu=100$) | 128 | 17,154 | 0.994 | 0.00091 | `CONVERGED` (2.82s) | **TRUE** | **Admitted (Stiff Benchmark)** |
| `vdp_mu2_h8.pt` | Van der Pol ($\mu=2$) | 8 | 114 | 0.333 (FAIL) | 0.156 (FAIL) | `CONVERGED` | **FALSE** | Rejected (Capacity insufficient) |
| `vdp_mu2_h32.pt` | Van der Pol ($\mu=2$) | 32 | 1,218 | 0.929 (FAIL) | 0.0517 (FAIL)| `CONVERGED` | **FALSE** | Rejected (Capacity insufficient) |
| `vdp_mu2_h128.pt` | Van der Pol ($\mu=2$) | 128 | 17,154 | 0.992 | 0.00758 | `CONVERGED` | **TRUE** | Validated candidate |

---

## 4. Preservation of Failed Results & Watchdog Evidence
* **Robertson Failure Preserved:** `robertson_h64.pt` failed Tier 2 trajectory error ($\text{NRMSE} = 0.0965 > 0.05$) and triggered a hard `VALIDATION_TIMEOUT` at 5.031s under the process watchdog. This failed result is explicitly preserved as scientific evidence of stiff Neural ODE failure modes.
* **Fallback Activation:** In accordance with the Phase 2 pipeline rules, the failure of `robertson_h64.pt` triggered the training and validation of a dedicated stiff fallback surrogate (`vdp_mu100_stiff_h128.pt`), which passed all 4 tiers and was formally admitted.
* **Watchdog Execution Trace:** While no standalone unit-test log file was saved for `tests/test_watchdog_mechanism.py`, the watchdog's real-world execution is empirically logged by the timeout event in `surrogate_validation_report.json`.
