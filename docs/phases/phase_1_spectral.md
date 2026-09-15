# Phase 1: Spectral Diagnostics

**Status:** Completed (Partial Portfolio Coverage)  
**Verification Level:** `[VERIFIED]` for LV & FHN; `[NOT YET MEASURED]` for VdP & ROBER  
**Context:** Analytical Jacobian eigenvalue decomposition along ground-truth trajectories.

---

## 1. Objective
Establish an empirical, analytical spectral characterization of candidate benchmark dynamical systems along verified ground-truth trajectories. Measure:
1. Real and imaginary eigenvalue envelopes.
2. Spectral radius range $\rho(\mathbf{J}(t, y)) = \max_i |\lambda_i|$.
3. Fastest local spectral timescale $\tau = 1 / \max |\lambda|$.
4. Fraction of trajectory points exhibiting non-zero imaginary components (oscillatory mode indicator).

---

## 2. Methodology (`experiments/phase1_spectral_diagnostics.py`)
* **Trajectory Source:** Loads verified `.pt` ground-truth trajectories from `data/trajectories/`.
* **Jacobian Evaluation:** Evaluates exact analytical Jacobians implemented in `benchmarks/systems.py`.
* **Zero Solver Interference:** Purely diagnostic; invokes no ODE solvers, no neural networks, and no optimization loops.

---

## 3. Measured Results (`experiments/logs/phase1_spectral_diagnostics.json`)

### A. Lotka-Volterra (2D, Non-Stiff Reference)
* **Sample Points:** 150 points over $t \in [0.0, 15.0]$.
* **Real Envelope:** $[-2.287497, +2.571851]$, mean real mode: $+0.021151$.
* **Imaginary Envelope:** $[-4.095858, +4.095858]$, mean absolute imaginary: $1.053810$.
* **Fraction Complex Points:** $56.67\%$.
* **Spectral Radius:** Range $[0.663238, 4.136061]$, mean: $2.121030$.
* **Fastest Spectral Timescale:** $\tau = 0.241776\text{ s}$.
* **Conclusion:** $O(1)$ spectral radius, bounded real modes, no rapid dissipative decay modes. Bounded limit cycle confirmed non-stiff.

### B. FitzHugh-Nagumo (2D, Candidate Non-Stiff / Mildly Multiscale)
* **Sample Points:** 200 points over $t \in [0.0, 50.0]$.
* **Real Envelope:** $[-2.852136, +0.916185]$, mean real mode: $-0.565620$.
* **Imaginary Envelope:** $[-0.282825, +0.282825]$, mean absolute imaginary: $0.053717$.
* **Fraction Complex Points:** $23.50\%$.
* **Spectral Radius:** Range $[0.222329, 2.852136]$, mean: $1.221950$.
* **Fastest Spectral Timescale:** $\tau_{\text{fast}} = 0.350614\text{ s}$.
* **Conclusion:** Modest spectral radius ($\le 2.85$). Does not exhibit high-frequency dissipative modes typical of classical numerical stiffness. Confirmed as candidate non-stiff / mildly multiscale system.

---

## 4. Verification Limitations & Missing Evidence
* **Van der Pol Trajectory Spectrum:** `NOT YET MEASURED` in existing project logs.
* **Robertson Trajectory Spectrum:** `NOT YET MEASURED` in existing project logs.
* **Retracted Heuristics:** Prior informal claim of a "$35\times$ stiffness ratio" for FHN was retracted as mathematically ungrounded.
