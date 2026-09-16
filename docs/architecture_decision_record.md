# Architecture Decision Record

## 1. Decision Status

* **Status:** Adopted (Pre-Execution Methodological Decision).
* **Execution Boundary:** This document records a methodological alignment decision made prior to the execution of Stage 1 benchmarking. **No new experiment has been executed** as part of this decision, no benchmark sweeps have been run, and no models have been trained or modified.

---

## 2. Problem

During the pre-flight fact verification conducted prior to Stage 1 execution, a verified architectural discrepancy was identified between the frozen specification documents and the actual Phase 2 implementation and checkpoints:

1. **Frozen Specification Documents** (`docs/benchmark_system_spec.md`, `docs/stage1_experiment_protocol.md`, `docs/phases/phase_4_benchmark_freeze.md`):
   * Specified a time-conditioned neural vector field: $f_\theta(z(t), t)$ taking concatenation $[z; t] \in \mathbb{R}^{D+1}$ ($D_{\text{in}} = 3$ for $D=2$).
   * Specified **GELU** activation after hidden layers 1 and 2.
   * Specified parameter count for $D=2$: $\text{Params} = (3 \times 64 + 64) + (64 \times 64 + 64) + (64 \times 2 + 2) = 256 + 4,160 + 130 = \mathbf{4,546\text{ parameters}}$.

2. **Actual Phase 2 Codebase & Checkpoints** (`models/vector_fields.py`, `experiments/phase2_train_surrogates.py`, `models/checkpoints/*.pt`, `experiments/logs/surrogate_validation_report.json`):
   * Implemented an autonomous neural vector field: $f_\theta(z)$ taking state vector $z \in \mathbb{R}^D$ ($D_{\text{in}} = 2$ for $D=2$). The forward pass ignores time $t$.
   * Implemented and trained with **Softplus** activation (`nn.Softplus`).
   * Checkpoints and validation reports record parameter count for $D=2$: $\text{Params} = (2 \times 64 + 64) + (64 \times 64 + 64) + (64 \times 2 + 2) = 192 + 4,160 + 130 = \mathbf{4,482\text{ parameters}}$.

Executing Stage 1 under this condition without formal resolution would produce a fundamental invalidity: either the experimental harness would run the actual 4,482-parameter Softplus checkpoints while documentation claimed a 4,546-parameter GELU model, or attempting to enforce the specification would fail against the saved checkpoint weights.

---

## 3. Verified Evidence

The following table summarizes the verified architectural properties established from repository code, logs, and checkpoints:

| Property | Frozen Specification | Actual Phase-2 Implementation | Evidence Status | Exact Repository Location |
|:---|:---|:---|:---:|:---|
| **Input Representation** | $[z; t] \in \mathbb{R}^{D+1}$ ($D_{\text{in}} = 3$) | $z \in \mathbb{R}^D$ ($D_{\text{in}} = 2$) | **Verified** | Spec: `docs/benchmark_system_spec.md#L150`<br>Code: `models/vector_fields.py#L21` |
| **Time Conditioning** | Yes (concatenation) | No ($t$ ignored in `forward`) | **Verified** | Spec: `docs/benchmark_system_spec.md#L127`<br>Code: `models/vector_fields.py#L46-49` |
| **Hidden Width ($W$)** | 64 | 64 | **Verified** | Spec: `docs/benchmark_system_spec.md#L152`<br>Code: `models/checkpoints/lv_h64.pt` (`net.0.weight` shape `[64, 2]`) |
| **Hidden Depth ($L$)** | 2 hidden layers (3 linear layers) | 2 hidden layers (3 linear layers) | **Verified** | Spec: `docs/benchmark_system_spec.md#L151`<br>Code: `models/checkpoints/lv_h64.pt` (layers `net.0`, `net.2`, `net.4`) |
| **Activation Function** | **GELU** | **Softplus** | **Verified** | Spec: `docs/benchmark_system_spec.md#L153`<br>Code: `experiments/phase2_train_surrogates.py#L455,L483` |
| **Output Dimension ($D_{\text{out}}$)** | $D$ ($2$ for $D=2$) | $D$ ($2$ for $D=2$) | **Verified** | Spec: `docs/benchmark_system_spec.md#L156`<br>Code: `models/checkpoints/lv_h64.pt` (`net.4.weight` shape `[2, 64]`) |
| **Augmentation Dim** | 0 | 0 | **Verified** | Code: `models/vector_fields.py#L16` (`augment_dim=0`) |
| **Total Parameters ($D=2$)** | **4,546** | **4,482** | **Verified** | Spec: `docs/benchmark_system_spec.md#L180`<br>Report: `experiments/logs/surrogate_validation_report.json#L338` |

---

## 4. Existing Empirical Evidence

The repository contains extensive, completed Phase 2 experimental artifacts that are anchored strictly to the autonomous Softplus architecture:

1. **Lotka-Volterra Baseline Checkpoint (`models/checkpoints/lv_h64.pt`):**
   * File size: 20,681 bytes; tensor shapes: `net.0.weight` is `[64, 2]`, `net.2.weight` is `[64, 64]`, `net.4.weight` is `[2, 64]`; total parameter count: 4,482.
   * Multi-tier validation record (`experiments/logs/surrogate_validation_report.json` lines 334–366): Tier 1 derivative $R^2 = 0.9991$, Tier 2 trajectory $\text{NRMSE} = 0.0414$, Tier 4 stability: `CONVERGED` (1.734s), status: `all_tiers_passed: true`.
2. **FitzHugh-Nagumo Baseline Checkpoint (`models/checkpoints/fhn_h64.pt`):**
   * File size: 20,693 bytes; tensor shapes: `net.0.weight` is `[64, 2]`, `net.2.weight` is `[64, 64]`, `net.4.weight` is `[2, 64]`; total parameter count: 4,482.
   * Multi-tier validation record (`experiments/logs/surrogate_validation_report.json` lines 368–400): Tier 1 derivative $R^2 = 0.9995$, Tier 2 trajectory $\text{NRMSE} = 0.00196$, Tier 4 stability: `CONVERGED` (1.694s), status: `all_tiers_passed: true`.
3. **Van der Pol Model Ladder (`models/checkpoints/vdp_mu{1,5,10,25,50,100}_h64.pt`):**
   * All 6 checkpoints have identical tensor structure ($D=2$, 4,482 parameters) and passed Phase 2 validation under Softplus.
4. **Empirical Activation Incompatibility:**
   * Numerical evaluation confirms that loading existing checkpoint weights into an `MLPVectorField` with Softplus replicates reported validation metrics down to floating-point precision ($0.0414$ for LV, $0.00196$ for FHN).
   * In contrast, loading the identical checkpoint weights into `MLPVectorField` with GELU causes catastrophic failure: Lotka-Volterra trajectory NRMSE explodes to $55,058.3$ ($R^2 = 0.1282$), and FitzHugh-Nagumo derivative $R^2$ collapses to $-5.864$ ($\text{NRMSE} = 0.2334$).
   * These empirical results establish that existing checkpoints strictly represent Softplus autonomous models and cannot be treated as validated under GELU or time-conditioning.

---

## 5. Scientific Relevance

The repository evidence establishes the following scientific considerations regarding the architecture:

1. **Mathematical Nature of Benchmark ODEs:**
   * In [`benchmarks/systems.py`](../benchmarks/systems.py), all four benchmark dynamical systems are mathematically autonomous ODEs:
     * Lotka-Volterra (lines 40–44): $\frac{dy_1}{dt} = \alpha y_1 - \beta y_1 y_2$, $\frac{dy_2}{dt} = \delta y_1 y_2 - \gamma y_2$.
     * FitzHugh-Nagumo (lines 73–77): $\frac{dv}{dt} = v - \frac{v^3}{3} - w + I$, $\frac{dw}{dt} = \frac{1}{\tau}(v + a - bw)$.
     * Van der Pol (lines 107–111): $\frac{dy_1}{dt} = y_2$, $\frac{dy_2}{dt} = \mu(1 - y_1^2)y_2 - y_1$.
     * Robertson (lines 143–148): $\frac{dy_1}{dt} = -k_1 y_1 + k_3 y_2 y_3$, etc.
   * None of these systems contain explicit time dependence ($\frac{\partial f}{\partial t} = 0$). An autonomous neural vector field $f_\theta(z)$ is therefore mathematically exact and natural for these systems.
2. **Alignment with Research Question and Hypothesis:**
   * The project's central research hypothesis ([`docs/problem_statement.tex#L60-L64`](problem_statement.tex#L60-L64)) investigates whether NFE corresponds to wall-clock computational cost across solver characteristics, stiffness, tolerances, and model complexity.
   * The research question evaluates the relationship between NFE and runtime; it does not mandate time conditioning, nor is non-autonomous parameterization a variable under test.
3. **Implications of Enforcing Time-Conditioning + GELU:**
   * Enforcing $[z; t] + \text{GELU}$ would require inventing synthetic time distributions during training on autonomous systems, retraining all baseline models from scratch, and executing a complete re-validation cycle.
   * This document does not claim that Softplus is universally superior to GELU, nor that autonomous parameterization is universally preferable to non-autonomous parameterization for general ODEs. It establishes only that for the autonomous benchmark systems in this study, the autonomous Softplus architecture is a fully validated, empirically verified surrogate representation.

---

## 6. Options Considered

### Option A: Align Canonical Specification with Verified Phase-2 Implementation
* **Description:** Update the canonical baseline specification across documentation to match the actual, verified Phase 2 implementation: autonomous state input $z \in \mathbb{R}^D$, Softplus activation, $W=64$, $L=2$ hidden layers ($4,482$ parameters for $D=2$).
* **What Changes:** Specification documents (`docs/benchmark_system_spec.md`, `docs/stage1_experiment_protocol.md`, `docs/phases/phase_4_benchmark_freeze.md`, `README.md`) are amended to document autonomous $z$ input and Softplus activation.
* **What Remains Valid:** 100% of Phase 2 checkpoints (`lv_h64.pt`, `fhn_h64.pt`, `vdp_mu*_h64.pt`), the training script (`experiments/phase2_train_surrogates.py`), and all validation metrics in `experiments/logs/surrogate_validation_report.json`.
* **Empirical Evidence Preserved:** All multi-tier surrogate validation records, derivative fits, trajectory errors, and watchdog execution logs are preserved without invalidation.
* **Transparency Requirement:** The historical divergence between Phase 4 documentation and Phase 2 implementation must be explicitly disclosed in the updated documents.

### Option B: Preserve Frozen Specification via Code Modification and Retraining
* **Description:** Retain the $[z; t] + \text{GELU}$ specification and modify the codebase and checkpoints to match it.
* **Required Code Changes:** Modify `models/vector_fields.py` to concatenate $[z; t]$ and default to GELU; modify `experiments/phase2_train_surrogates.py` to sample space-time collocation points and pass time $t$ during derivative fitting.
* **Retraining Requirements:** Retrain `lv_h64`, `fhn_h64`, and all Van der Pol models from scratch to convergence using GELU and time conditioning.
* **Validation Requirements:** Re-run the full 4-tier validation screening under the process watchdog for every new model to determine if it meets Tier 1 ($R^2 \ge 0.95$) and Tier 2 ($\text{NRMSE} \le 0.05$) standards.
* **Empirical Artifacts Invalidated:** All existing checkpoints and all entries in `experiments/logs/surrogate_validation_report.json` for 64-width models would become obsolete and would no longer represent the canonical baseline.

---

## 7. Decision

**The canonical Stage-1 baseline will be aligned with the verified Phase-2 implementation: an autonomous Softplus MLP vector field with $W=64$ and $L=2$ hidden layers.**

### Rationale:
1. **Implementation Provenance:** The codebase, training loops, process watchdog routines, and checkpoint binaries were created, debugged, and validated exclusively on the autonomous Softplus architecture.
2. **Preservation of Verified Empirical Evidence:** Adopting the verified implementation preserves 100% of the Phase 2 multi-tier surrogate validation records, avoiding the invalidation of verified research evidence.
3. **Dynamical Consistency:** All benchmark target equations are mathematically autonomous ODEs ($\frac{dy}{dt} = f(y)$); providing time $t$ as an explicit input introduces an extraneous parameter dimension that is mathematically invariant.
4. **Research Question Integrity:** The central hypothesis concerns the relationship between NFE and wall-clock execution time across solvers, tolerances, and stiffness regimes. Holding an autonomous Softplus architecture constant across systems fully satisfies all experimental control requirements.

*Disclaimer:* This decision is methodological. It does **not** assert that Softplus or autonomous parameterization is universally superior, nor does it constitute proof or support of the project's central research hypothesis.

---

## 8. Transparency Statement

To maintain complete scientific and forensic integrity:
1. It is explicitly acknowledged that the Phase 4 freeze document (`docs/phases/phase_4_benchmark_freeze.md`), the benchmark specification (`docs/benchmark_system_spec.md`), and the Stage 1 experiment protocol (`docs/stage1_experiment_protocol.md`) incorrectly documented the canonical architecture as time-conditioned $[z; t]$ with GELU activation ($4,546$ parameters).
2. This discrepancy was discovered during pre-flight fact verification before any Stage 1 code was implemented or executed.
3. Prior Phase 2 evidence was **not silently relabeled or overwritten**.
4. All subsequent modifications to specification documents will explicitly cite this Architecture Decision Record as the audit trail for the alignment.

---

## 9. Required Follow-up Changes

The following documentation files must be updated in a subsequent step to reflect this decision:
1. [`docs/benchmark_system_spec.md`](benchmark_system_spec.md):
   * Update Section D (architecture diagram and text) to autonomous $z \in \mathbb{R}^D$ and Softplus activation.
   * Update Section E (parameter arithmetic) to $\text{Params}(D) = 129D + 4,224$ ($D=2 \to 4,482$; $D=3 \to 4,611$).
   * Update Section H (summary table rows 243–244).
2. [`docs/stage1_experiment_protocol.md`](stage1_experiment_protocol.md):
   * Update Section 4 (architecture diagram, text, and parameter counts).
   * Update Section 5.3 (Table row 152).
   * Update Section 9.3 (JSON schema example).
3. [`docs/phases/phase_4_benchmark_freeze.md`](phases/phase_4_benchmark_freeze.md):
   * Update Section 3 (architecture description and parameter arithmetic).
4. [`README.md`](../README.md):
   * Update Section 8 (canonical baseline topology and parameter counts).

*(No files outside documentation will be modified).*

---

## 10. Pre-Execution Boundary

* **Zero Stage-1 Experiments Executed:** Experiment 1 has not been run. No benchmark timing measurements have been taken.
* **Zero New Models Trained:** No neural network training has taken place.
* **Zero Results Generated:** No synthetic or empirical results have been fabricated or updated.
* **Boundary:** This Architecture Decision Record documents the decision only. Updating the four specification documents listed in Section 9 is a separate subsequent task.

---

## 11. Evidence References

1. `models/vector_fields.py`: Lines 21, 26, 32, 41, 46–49.
2. `experiments/phase2_train_surrogates.py`: Lines 41–88, 375, 420, 455, 463, 483, 491, 551, 591.
3. `models/checkpoints/lv_h64.pt`: 20,681 bytes, state dict shapes `[64, 2]`, `[64, 64]`, `[2, 64]`, 4,482 params.
4. `models/checkpoints/fhn_h64.pt`: 20,693 bytes, state dict shapes `[64, 2]`, `[64, 64]`, `[2, 64]`, 4,482 params.
5. `experiments/logs/surrogate_validation_report.json`: Lines 334–366 (`lv_h64`), Lines 368–400 (`fhn_h64`).
6. `docs/phases/phase_2_surrogate.md`: Lines 24–33.
7. `docs/phases/phase_3_evidence_audit.md`: Lines 36–40.
8. `docs/phases/phase_4_benchmark_freeze.md`: Lines 56–65.
9. `docs/benchmark_system_spec.md`: Lines 123–195, 243–244.
10. `docs/stage1_experiment_protocol.md`: Lines 66–102, 152, 464.
11. `benchmarks/systems.py`: Lines 40–44, 73–77, 107–111, 143–148.
12. `docs/problem_statement.tex`: Lines 60–64.
