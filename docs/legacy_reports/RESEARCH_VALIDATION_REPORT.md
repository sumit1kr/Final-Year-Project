# Research Validation Report: Computational Performance Characterization of Neural Ordinary Differential Equations

**Track 2: Computational Realization**  
**Author:** Sumit Kumar (Roll No.: CS23B2008)  
**Department of Computer Science and Engineering**  
**Indian Institute of Information Technology, Design and Manufacturing (IIITDM), Kancheepuram**  
**Guide:** Dr. Noor Mahammad Shaik  
**Date:** September 2026  

---

## Executive Summary
This report presents the complete findings of the **Research Validation Phase** for Track 2 (*Computational Performance Characterization of Neural ODEs*). The primary objective was to investigate whether **Number of Function Evaluations (NFE)** is sufficient to characterize the real-world computational cost of Neural ODEs on modern computing hardware, particularly across varying stiffness regimes, network capacities, and error tolerances. 

Through controlled empirical benchmarking across 4 canonical dynamical systems, this study provides quantitative proof that **NFE alone is fundamentally incomplete and often misleading as a standalone complexity metric**. We demonstrate scenarios where workloads with 1.55× higher NFE execute in less wall-clock time than low-NFE configurations, expose an internal solver overhead ($T_{\text{solver}}$) accounting for up to 83% of total runtime, and empirically identify the **"Stiffness Cliff"** at which explicit numerical integration suffers step-size collapse, allowing implicit methods to achieve a **14.4× wall-clock speedup**.

---

## 1. Problem Definition
Neural Ordinary Differential Equations (Neural ODEs) represent a continuous-depth machine learning formulation where state transitions are governed by a parameterized vector field:
$$\frac{dz(t)}{dt} = f_\theta(z(t), t), \quad z(t_1) = z(t_0) + \int_{t_0}^{t_1} f_\theta(z(t), t) \, dt$$

In scientific machine learning literature, the computational cost of integrating this continuous system is almost universally reported via the **Number of Function Evaluations (NFE)**—the count of times the ODE solver evaluates the neural network $f_\theta$. 

However, the practical computational pipeline consists of multiple abstraction layers:
$$\text{Continuous Dynamics} \longrightarrow \text{Neural Approximation} \longrightarrow \text{Numerical Solver} \longrightarrow \text{Computer Hardware}$$

Because NFE treats every function evaluation as a uniform scalar unit, it completely abstracts away:
1. **Network Parameter Capacity ($m$):** Evaluating an MLP with 300 parameters vs. 70,000 parameters requires vastly different arithmetic intensity and memory bandwidth.
2. **Internal Solver Mechanics ($T_{\text{solver}}$):** Adaptive step-size controllers, error norm computations, and rejection logic introduce substantial non-evaluation latency.
3. **Stiffness and Numerical Regimes:** When systems exhibit multiple separated time scales (stiffness), explicit solvers take millions of tiny steps, whereas implicit solvers require $O(k^3)$ linear system factorizations per step.
4. **Hardware Realities:** Sequential solver step dependencies prevent GPU instruction pipelining and induce kernel launch latency.

Thus, a rigorous characterization of Neural ODE computational cost on modern hardware is urgently required.

---

## 2. Literature Review
The standard evaluation paradigm was established by Chen et al. (2018), which posited that NFE directly measures computational efficiency. Subsequent works noted that NFE escalates uncontrollably during training, prompting regularization strategies such as kinetic energy penalties and Jacobian regularization (Finlay et al., 2020; Kelly et al., 2020). 

In parallel, numerical analysts demonstrated that reversing trajectories during the continuous adjoint method leads to catastrophic gradient instability on stiff systems (Gholami et al., 2019). Kim et al. (2021) demonstrated that explicit solvers fail completely on stiff chemical systems like Robertson's problem, and proved that naive implicit solvers scale cubically ($O((k+m)^3)$) with network parameters. While Poli et al. (2020) and Pal et al. (2022) studied solver heuristics and learned discrete solvers (Hypersolvers), the direct empirical relationship between NFE, stiffness, and hardware execution cost remained unmapped.

---

## 3. Closest 5 Papers (In-Depth Analysis)

### 1. Kim et al. (2021) — *Stiff Neural Ordinary Differential Equations*
* **Core Contribution:** Identified the failure modes of standard Neural ODEs on stiff chemical kinetics (ROBER, POLLU). Showed that standard adjoints blow up exponentially and implicit solvers scale cubically $O((k+m)^3)$. Proposed `QuadratureAdjoint` to decouple state and parameter solves to $O(k^3+m)$, combined with equation and loss scaling.
* **Limitation:** Benchmarked exclusively in Julia on a single CPU core; did not evaluate GPU hardware scaling or systematically analyze NFE validity.
* **Role in our work:** Serves as our base anchor paper for stiff benchmark systems and theoretical complexity modeling.

### 2. Chen et al. (NeurIPS 2018) — *Neural Ordinary Differential Equations*
* **Core Contribution:** Introduced continuous-depth architectures, backpropagation via the Pontryagin adjoint state method, and established NFE as the universal metric for computational cost.
* **Limitation:** Relied solely on explicit adaptive solvers (`dopri5`) on non-stiff image and toy time-series data.
* **Role in our work:** The foundational baseline against which our empirical critique of NFE is directed.

### 3. Finlay et al. (ICML 2020) — *How to Train Your Neural ODE: the World of Jacobian and Kinetic Energy Regularizations*
* **Core Contribution:** Documented that as training progresses, learned dynamics become increasingly stiff and tortuous, causing NFE to explode. Proposed Frobenius norm regularization of the Jacobian and kinetic energy regularization to reduce NFE.
* **Limitation:** Did not benchmark on physically stiff differential equations or measure hardware-level solver overheads.
* **Role in our work:** Validates that NFE explosion is an inherent failure mode during continuous optimization.

### 4. Pal et al. (ICLR 2022) — *Opening the Black Box: Accelerating Neural Differential Equations by Regularizing Internal Solver Heuristics*
* **Core Contribution:** Analyzed internal solver step rejections and showed that standard NFE metrics hide immense computational waste from solver step rejections.
* **Limitation:** Focused on regularizing solver controllers rather than establishing an overarching hardware performance model.
* **Role in our work:** Directly supports our inclusion of the $T_{\text{solver}}$ component in the runtime decomposition equation.

### 5. Poli et al. (NeurIPS 2020) — *Hypersolvers: Toward Fast Continuous-Depth Models*
* **Core Contribution:** Demonstrated that classical adaptive ODE solvers underutilize modern GPU hardware due to sequential step dependencies and branch divergence. Proposed training auxiliary neural networks to predict truncation errors.
* **Limitation:** Approximated discrete solver steps rather than resolving exact stiffness scaling.
* **Role in our work:** Provides foundational motivation for GPU kernel latency and hardware-level benchmarking.

---

## 4. Existing Limitations in Prior Literature
1. **The "Scalar NFE" Assumption:** Prior works equate a solver step evaluating an 8-neuron network with one evaluating a 512-neuron network, despite orders of magnitude difference in FLOPs and memory access.
2. **Neglect of Internal Solver Overhead:** Scientific ML literature assumes network evaluation ($NFE \times T_f$) dominates total time, overlooking the cost of step-size controllers, error norms, and interpolations.
3. **Stiffness Avoidance:** The vast majority of Neural ODE papers avoid stiff dynamical systems entirely, relying on explicit Runge-Kutta methods that fail when eigenvalues diverge.

---

## 5. Identified Research Gap
There is no comprehensive, empirical performance characterization that decomposes wall-clock execution time across:
$$T_{\text{NODE}} = \text{NFE} \times T_f + T_{\text{solver}} + T_{\text{memory}} + T_{\text{launch}} + T_{\text{sync}}$$
Specifically, the scientific machine learning community lacks a quantified map of the **crossover boundary (The Stiffness Cliff)** where explicit solvers collapse, and an empirical proof of the conditions under which **NFE fails to predict wall-clock latency**.

---

## 6. Research Question
> **How do the numerical characteristics of a Neural ODE translate into actual computational cost on modern hardware, and under what conditions is NFE insufficient to characterize that cost?**

---

## 7. Hypothesis
> **Initial Hypothesis:** NFE alone is insufficient to characterize the computational cost of Neural ODEs across different numerical regimes, model sizes, and hardware configurations. Specifically:
> 1. High-NFE configurations with compact networks can execute faster than low-NFE configurations with wide/deep networks.
> 2. Internal solver overhead ($T_{\text{solver}}$) constitutes a major fraction of total runtime in adaptive solvers.
> 3. As stiffness increases, explicit solvers suffer step-size collapse, creating a distinct cliff where implicit solvers become orders of magnitude faster in wall-clock time despite higher per-step linear algebra complexity.

---

## 8. Experimental Design & Methodology

### 8.1 Controlled Benchmark Suite
We implemented 4 canonical dynamical systems spanning a rigorous progression of stiffness:
1. **Lotka-Volterra (2D, Non-stiff):** Predator-prey harmonic interactions ($t \in [0, 15]$).
2. **FitzHugh-Nagumo (2D, Mildly stiff):** Neuronal activation dynamics ($t \in [0, 50]$).
3. **Van der Pol Oscillator (2D, Tunable stiffness):** Relaxation oscillator with variable parameter $\mu \in [1.0, 100.0]$ ($t \in [0, 30]$).
4. **Robertson Chemical Reaction (3D, Highly stiff):** Stiff kinetics ($k_1=0.04, k_2=3\times 10^7, k_3=10^4$) across logarithmic time ($t \in [10^{-5}, 10^5]$).

### 8.2 High-Resolution Profiling Instrumentation
* **Timing Harness:** Isolated network forward pass latency ($T_f$) measured across 50 warmup iterations using high-precision CPU `time.perf_counter()` and CUDA event timing.
* **Runtime Decomposition Engine:**
  $$T_{\text{net\_eval}} = \text{NFE} \times T_f, \quad T_{\text{solver}} = T_{\text{total}} - T_{\text{net\_eval}}$$
* **Metadata Logger:** Fully compliant with Supervisor Rule 2 (logging experiment ID, timestamp, ODE, stiffness, solver, tolerances, architecture, parameter count, device, NFE, total runtime, $T_f$, $T_{\text{solver}}$, and observations).

---

## 9. Preliminary Results & Quantitative Evidence

### 9.1 Experiment 1: Solver Overhead Decomposition
Testing explicit solvers (`euler`, `rk4`, `dopri5`) across benchmark systems revealed that adaptive step-size control introduces immense non-evaluation latency:
* On `dopri5`, **73.8% to 83.3% of the total integration time is consumed by internal solver overhead ($T_{\text{solver}}$)**, leaving only 17–26% of time spent executing neural network layers.
* *Artifact:* `experiments/plots/solver_stiffness_sweep.png`.

### 9.2 Experiment 2: The Critical Workload Experiment (NFE Inversion)
We tested Workload A (High NFE, small net: 354 params) against Workload B (Low NFE, large net: 68,746 params):

| Workload | Network Parameters | Reported NFE | Wall-Clock Runtime | Solver Overhead |
| :--- | :---: | :---: | :---: | :---: |
| **Workload A (Small Net)** | 354 | **62** | **13.60 ms** | 42.4% |
| **Workload B (Large Net)** | 68,746 | **40** | **15.57 ms** | 54.5% |

* **Empirical Confirmation:** Workload A had **1.55× higher NFE**, yet Workload B required **longer wall-clock time (15.57 ms)**. NFE failed completely to predict true execution cost.
* *Artifact:* `experiments/plots/critical_workload_nfe_vs_runtime.png`.

### 9.3 Experiment 3.1: Tolerance Scaling Across Stiffness Regimes
Sweeping tolerances ($\text{rtol} \in [10^{-3}, 10^{-9}]$):
* On non-stiff Lotka-Volterra: Tightening tolerance from $10^{-3}$ to $10^{-9}$ increased runtime by only **2.5×** (5.13 ms $\rightarrow$ 13.24 ms; NFE: 20 $\rightarrow$ 74).
* On stiff Van der Pol ($\mu=10$): The same tolerance tightening triggered a **9.2× runtime explosion** (5.68 ms $\rightarrow$ 52.06 ms) and a **10× NFE spike** (20 $\rightarrow$ 200).
* *Artifact:* `experiments/plots/tolerance_vs_nfe_runtime.png`.

### 9.4 Experiment 3.2: The Stiffness Cliff
Sweeping Van der Pol stiffness $\mu \in [1, 100]$ comparing explicit (`RK45/Dopri5`) vs. implicit (`Radau`):

| Stiffness ($\mu$) | Explicit (RK45) NFE | Explicit Time | Implicit (Radau) NFE | Implicit Time | Speedup Factor |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $\mu = 1.0$ | 926 | **3.95 ms** | 2,317 | 30.60 ms | Explicit is 7.7× faster |
| $\mu = 25.0$ | 2,102 | 7.50 ms | 462 | **6.27 ms** | **Crossover Boundary** |
| $\mu = 100.0$ | **11,498** | **40.04 ms** | **191** | **2.77 ms** | **Implicit is 14.4× faster!** |

* **Empirical Confirmation:** Demonstrates the exact boundary ($\mu \approx 25$) where explicit solvers face step-size collapse, proving that implicit solvers are up to **14.4× faster in wall-clock time** despite their higher per-step complexity.
* *Artifact:* `experiments/plots/stiffness_scaling_cliff.png`.

### 9.5 Experiment 3.3: Batch Scaling Dynamics
Testing batch sizes $B \in [1, 256]$ demonstrated that batching amortizes dispatch overhead, driving per-sample latency from **13.46 ms down to 0.054 ms** and boosting throughput from **74.3 traj/s to 18,380.4 traj/s (a 247× boost)**.
* *Artifact:* `experiments/plots/batch_scaling_throughput.png`.

---

## 10. Hypothesis Evaluation
| Hypothesis Sub-Claim | Status | Evidence |
| :--- | :---: | :--- |
| **Claim 1:** NFE alone does not predict wall-clock time across network scales. | **SUPPORTED** | Experiment 2 showed an inversion where a 62-NFE workload was faster than a 40-NFE workload. |
| **Claim 2:** Internal solver overhead ($T_{\text{solver}}$) constitutes a major fraction of total runtime. | **SUPPORTED** | Experiment 1 & 3.1 demonstrated that $T_{\text{solver}}$ accounts for 54% to 83% of total solve time on adaptive solvers. |
| **Claim 3:** Explicit solvers suffer step-size collapse on stiff systems, giving implicit methods a decisive speedup. | **SUPPORTED** | Experiment 3.2 proved that at $\mu=100$, implicit Radau runs 14.4× faster with 60× fewer evaluations than explicit RK45. |

---

## 11. Proposed Research Contribution
For the final conference/journal submission, this project delivers:
1. **The First Unified Neural ODE Cost Decomposition Model:**
   $$T_{\text{NODE}} = \text{NFE} \times T_f + T_{\text{solver}} + T_{\text{memory}} + T_{\text{launch}} + T_{\text{sync}}$$
   Quantifying the exact boundary conditions under which NFE breaks down.
2. **Empirical Characterization of the Stiffness Cliff:** A concrete taxonomy mapping dynamical stiffness ($\mu$) to solver efficiency transitions.
3. **Practical Guidelines for Practitioners:** Rigorous heuristic rules advising when deep learning researchers should abandon explicit Runge-Kutta methods in favor of implicit/decoupled sensitivity architectures.

---

## 12. Plan for Full Experimental Study (Next 8 Weeks)
* **Weeks 1–2:** Multi-trajectory training runs comparing loss convergence on stiff chemical kinetics (ROBER) using equation scaling.
* **Weeks 3–4:** GPU vs. CPU kernel execution profiling on Intel Arc dedicated hardware to quantify $T_{\text{launch}}$ and $T_{\text{sync}}$.
* **Weeks 5–6:** Draft full 8-page research manuscript in IEEE/ACM conference format.
* **Weeks 7–8:** Guide review, internal rehearsal, and final paper submission.

---

## 13. References
1. **Chen, R. T., Rubanova, Y., Bettencourt, J., & Duvenaud, D.** (2018). *Neural ordinary differential equations.* Advances in Neural Information Processing Systems (NeurIPS), 31.
2. **Kim, S., Ji, W., Deng, S., Ma, Y., & Rackauckas, C.** (2021). *Stiff neural ordinary differential equations.* arXiv preprint arXiv:2103.15341.
3. **Finlay, C., Jacobsen, J. H., Nurbekyan, L., & Oberman, A.** (2020). *How to train your neural ODE: the world of Jacobian and kinetic energy regularizations.* International Conference on Machine Learning (ICML), PMLR, 3154-3164.
4. **Pal, A., et al.** (2022). *Opening the black box: Accelerating neural differential equations by regularizing internal solver heuristics.* International Conference on Learning Representations (ICLR).
5. **Poli, M., Massaroli, S., Yamashita, A., Asama, H., & Park, J.** (2020). *Hypersolvers: Toward fast continuous-depth models.* Advances in Neural Information Processing Systems (NeurIPS), 33.
6. **Gholami, A., Keutzer, K., & Biros, G.** (2019). *ANODE: Unconditionally accurate memory-efficient gradients for neural ODEs.* arXiv preprint arXiv:1902.10298.
7. **Hairer, E., & Wanner, G.** (1996). *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems.* Springer-Verlag.
8. **Shampine, L. F., & Gear, C. W.** (1979). *A user's view of solving stiff ordinary differential equations.* SIAM Review, 21(1), 1-17.
