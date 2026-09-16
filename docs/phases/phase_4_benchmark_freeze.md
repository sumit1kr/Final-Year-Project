# Phase 4: Benchmark System & Architecture Freeze

**Status:** Completed & Frozen  
**Verification Level:** `[VERIFIED]`  
**Context:** Specification freezing 4 benchmark systems, canonical baseline MLP architecture, parameter arithmetic, and CPU execution boundary.

---

## 1. Objective
Establish an immutable experimental foundation by locking:
1. Four benchmark dynamical systems across controlled stiffness regimes.
2. A single canonical baseline Neural ODE vector-field architecture.
3. Strict CPU-only execution boundaries.
This ensures all subsequent Stage 1 experiments isolate numerical and model variables without confounding factors.

---

## 2. Frozen Benchmark Portfolio

| System | Dimension | Horizon | Classification | Grounding | Experimental Role |
|:---|:---:|:---:|:---:|:---|:---|
| **Lotka-Volterra** | 2D | $t \in [0, 15]$ | Non-Stiff Reference | Rackauckas et al. (2020); measured $\rho \in [0.66, 4.14]$ | Zero-stiffness baseline to measure pure network latency ($T_f$) and solver overhead ($T_{\text{solver}}$). |
| **FitzHugh-Nagumo**| 2D | $t \in [0, 50]$ | Candidate Non-Stiff / Mildly Multiscale | FitzHugh (1961), Nagumo et al. (1962); measured $\rho \le 2.85$ | Test localized adaptive step-size refinement during sharp excursions without solver breakdown. |
| **Van der Pol** | 2D | $t \in [0, 30]$ | Stiff Benchmark (Tunable $\mu \in [1, 100]$) | Kim et al. (2021), Caldana & Hesthaven (2025) | Primary continuous axis to test whether physical stiffness induces explicit solver step collapse on Neural ODEs. |
| **Robertson** | 3D | $t \in [10^{-5}, 10^5]$ | Extreme Stiff / Multiscale | Kim et al. (2021), Hairer & Wanner (1996) | Asymptotic extreme to isolate implicit linear-algebra costs ($T_{\text{solver}}$, Jacobians, LU factorizations). |

---

## 3. Canonical Baseline Architecture (`[OUR CANONICAL BASELINE — PHASE-2 ALIGNED]`)

The canonical baseline architecture is formally aligned with the verified Phase-2 implementation per `docs/architecture_decision_record.md`:

```
State Vector z(t) ∈ R^D
           │
           ▼
   Linear(D ──► 64)
           │
           ▼
        Softplus
           │
           ▼
   Linear(64 ──► 64)
           │
           ▼
        Softplus
           │
           ▼
   Linear(64 ──► D)
           │
           ▼
   Output dz/dt ∈ R^D
```

* **Network:** Autonomous Multilayer Perceptron (MLP) vector field $f_\theta(z)$.
* **Input Representation:** Direct state vector $z(t) \in \mathbb{R}^D$ ($D_{\text{in}} = 2$ for $D=2$; $D_{\text{in}} = 3$ for $D=3$). Augmentation dimension is zero (`augment_dim = 0`). The forward pass computes $f_\theta(z)$ without explicit time conditioning.
* **Depth:** Exactly 2 hidden layers (3 linear transformations total).
* **Width:** Exactly 64 hidden units per layer.
* **Activation:** **Softplus** (`nn.Softplus`, $\beta=1$), providing continuously differentiable $C^\infty$ dynamics matching trained Phase 2 checkpoints.
* **Readout:** Direct linear projection to $\mathbb{R}^D$ (no output activation).

### Verified Parameter Arithmetic
$$\text{Params}(D) = [D \times 64 + 64] + [64 \times 64 + 64] + [64 \times D + D] = 129 D + 4,224$$
* **$D=2$ (LV, FHN, VdP):** $129(2) + 4,224 = \mathbf{4,482\text{ parameters}}$ `[VERIFIED MATHEMATICAL ARITHMETIC & PHASE-2 IMPLEMENTATION]`.
* **$D=3$ (Robertson Baseline Reference):** $129(3) + 4,224 = \mathbf{4,611\text{ parameters}}$ `[VERIFIED MATHEMATICAL ARITHMETIC & IMPLEMENTATION]`.

*(Historical Provenance Note: An earlier draft documented a theoretical $[z; t] \in \mathbb{R}^{D+1}$ model with GELU yielding 4,546 parameters for $D=2$. Pre-flight audit revealed the actual Phase 2 checkpoints were trained and validated as autonomous Softplus models; the specification has been formally aligned to match per `docs/architecture_decision_record.md`).*

---

## 4. Controlled Complexity Axis (Future Experiments)
* **Single-Factor Rule:** The baseline ($W=64, L=2$) remains immutable. Model complexity will later be evaluated by varying **one factor at a time**:
  * Width Scaling: vary $W$ while holding depth fixed at $L=2$.
  * Depth Scaling: vary $L$ while holding width fixed at $W=64$.
* *Boundary:* Complexity variants are **NOT** benchmarked at this stage.

---

## 5. CPU Experimental Boundary
* **Hardware Scope:** Mandatory **CPU-ONLY** per mentor decision. Zero GPU, CUDA, or mixed-precision.
* **Concurrency:** `torch.set_num_threads(1)` proposed for isolated per-step profiling without thread contention.
* **Environment Logging:** Host CPU model, cache hierarchy, OS build, and BLAS backend will be queried and logged during Stage 1 setup.
