# Literature Archive — Neural ODE Computational Performance Characterization

## Research Question

> To what extent does NFE characterize the computational cost of Neural ODEs, and under what conditions do model, solver, numerical, and hardware factors cause deviations between NFE and actual computational cost?

---

## Folder Structure

```
literature/
├── Group_A_MLP_Neural_ODE/           (6 papers)
├── Group_B_Modified_Neural_ODE/      (3 papers)
├── Group_C_Computational_Methodology/ (6 papers)
├── Top_5_Papers/                     (5 copies)
└── README.md                         (this file)
```

---

## Paper Index

| # | Paper | Year | Group | Top 5? | Filename | Source | Type |
|---|-------|------|-------|--------|----------|--------|------|
| 1 | Neural Ordinary Differential Equations — Chen, Rubanova, Bettencourt, Duvenaud | 2018 | A | No | `Group_A_MLP_Neural_ODE/01_Neural_Ordinary_Differential_Equations_2018.pdf` | [arXiv:1806.07366](https://arxiv.org/abs/1806.07366) | Peer-reviewed conference paper (NeurIPS 2018, Best Paper) |
| 2 | How to train your neural ODE: the world of Jacobian and kinetic regularization — Finlay, Jacobsen, Nurbekyan, Oberman | 2020 | A | **Yes (#2)** | `Group_A_MLP_Neural_ODE/02_How_to_Train_Your_Neural_ODE_2020.pdf` | [arXiv:2002.02798](https://arxiv.org/abs/2002.02798) | Peer-reviewed conference paper (ICML 2020) |
| 3 | STEER: Simple Temporal Regularization for Neural ODEs — Ghosh, Behl, Dupont, Torr, Namboodiri | 2020 | A | No | `Group_A_MLP_Neural_ODE/03_STEER_Simple_Temporal_Regularization_2020.pdf` | [arXiv:2006.10711](https://arxiv.org/abs/2006.10711) | Peer-reviewed conference paper (NeurIPS 2020) |
| 4 | Stiff neural ordinary differential equations — Kim, Ji, Deng, Ma, Rackauckas | 2021 | A | **Yes (#1)** | `Group_A_MLP_Neural_ODE/04_Stiff_Neural_Ordinary_Differential_Equations_2021.pdf` | [DOI:10.1063/5.0060697](https://doi.org/10.1063/5.0060697) · [arXiv:2103.15341](https://arxiv.org/abs/2103.15341) | Peer-reviewed journal article (Chaos, Vol. 31, Issue 9) |
| 5 | Discretize-Optimize vs. Optimize-Discretize for Time-Series Regression and Continuous Normalizing Flows — Onken, Ruthotto | 2020 | A | No | `Group_A_MLP_Neural_ODE/05_Discretize_Optimize_vs_Optimize_Discretize_2020.pdf` | [arXiv:2005.13420](https://arxiv.org/abs/2005.13420) | arXiv preprint (presented at SIAM MDS 2020) |
| 6 | Augmented Neural ODEs — Dupont, Doucet, Teh | 2019 | B | **Yes (#5)** | `Group_B_Modified_Neural_ODE/06_Augmented_Neural_ODEs_2019.pdf` | [arXiv:1904.01681](https://arxiv.org/abs/1904.01681) | Peer-reviewed conference paper (NeurIPS 2019) |
| 7 | Heavy Ball Neural Ordinary Differential Equations — Xia, Suliafu, Ji, Nguyen, Bertozzi, Osher, Wang | 2021 | B | No | `Group_B_Modified_Neural_ODE/07_Heavy_Ball_Neural_ODEs_2021.pdf` | [arXiv:2110.04840](https://arxiv.org/abs/2110.04840) | Peer-reviewed conference paper (NeurIPS 2021) |
| 8 | Hypersolvers: Toward Fast Continuous-Depth Models — Poli, Massaroli, Yamashita, Asama, Park | 2020 | B | No | `Group_B_Modified_Neural_ODE/08_Hypersolvers_Fast_Continuous_Depth_Models_2020.pdf` | [arXiv:2007.09601](https://arxiv.org/abs/2007.09601) | Peer-reviewed conference paper (NeurIPS 2020) |
| 9 | DiffEqFlux.jl – A Julia Library for Neural Differential Equations — Rackauckas, Innes, Ma, Bettencourt, White, Dixit | 2019 | C | No | `Group_C_Computational_Methodology/09_DiffEqFlux_Julia_Library_Neural_DiffEq_2019.pdf` | [arXiv:1902.02376](https://arxiv.org/abs/1902.02376) | arXiv preprint |
| 10 | torchode: A Parallel ODE Solver for PyTorch — Lienen, Günnemann | 2022 | C | **Yes (#3)** | `Group_C_Computational_Methodology/10_torchode_Parallel_ODE_Solver_PyTorch_2022.pdf` | [arXiv:2210.12375](https://arxiv.org/abs/2210.12375) | Workshop paper (NeurIPS 2022 Workshop) |
| 11 | On Neural Differential Equations — Kidger | 2022 | C | No | `Group_C_Computational_Methodology/11_On_Neural_Differential_Equations_Thesis_2022.pdf` | [arXiv:2202.02435](https://arxiv.org/abs/2202.02435) | **PhD Thesis** (University of Oxford) |
| 12 | Adaptive Checkpoint Adjoint Method for Gradient Estimation in Neural ODE — Zhuang, Dvornek, Li, Tatikonda, Papademetris, Duncan | 2020 | C | No | `Group_C_Computational_Methodology/12_Adaptive_Checkpoint_Adjoint_2020.pdf` | [arXiv:2006.02493](https://arxiv.org/abs/2006.02493) | Peer-reviewed conference paper (ICML 2020) |
| 13 | MALI: A memory efficient and reverse accurate integrator for Neural ODEs — Zhuang, Dvornek, Tatikonda, Duncan | 2021 | C | No | `Group_C_Computational_Methodology/13_MALI_Memory_Efficient_Reverse_Accurate_2021.pdf` | [arXiv:2102.04668](https://arxiv.org/abs/2102.04668) | Peer-reviewed conference paper (ICLR 2021) |
| 14 | Universal Differential Equations for Scientific Machine Learning — Rackauckas, Ma, Martensen, Warner, Zubov, Supekar, Skinner, Ramadhan, Edelman | 2020 | C | No | `Group_C_Computational_Methodology/14_Universal_Differential_Equations_SciML_2020.pdf` | [arXiv:2001.04385](https://arxiv.org/abs/2001.04385) | arXiv preprint |
| 15 | Neural Ordinary Differential Equations for Model Order Reduction of Stiff Systems — Caldana, Hesthaven | 2024/2025 | A | **Yes (#4)** | `Group_A_MLP_Neural_ODE/15_Neural_ODE_Model_Order_Reduction_Stiff_Systems_2024.pdf` | [arXiv:2408.06073](https://arxiv.org/abs/2408.06073) · Published: *Int. J. Numer. Meth. Engng.* (2025) | Peer-reviewed journal article (arXiv version downloaded) |

---

## Top 5 Papers (Ranked)

| Rank | # | Paper | Group | Top_5_Papers/ filename |
|------|---|-------|-------|------------------------|
| 1 | 4 | Stiff neural ordinary differential equations | A | `04_Stiff_Neural_Ordinary_Differential_Equations_2021.pdf` |
| 2 | 2 | How to train your neural ODE | A | `02_How_to_Train_Your_Neural_ODE_2020.pdf` |
| 3 | 10 | torchode: A Parallel ODE Solver for PyTorch | C | `10_torchode_Parallel_ODE_Solver_PyTorch_2022.pdf` |
| 4 | 15 | Neural ODE for Model Order Reduction of Stiff Systems | A | `15_Neural_ODE_Model_Order_Reduction_Stiff_Systems_2024.pdf` |
| 5 | 6 | Augmented Neural ODEs | B | `06_Augmented_Neural_ODEs_2019.pdf` |

Top 5 PDFs in `Top_5_Papers/` are **copies** — originals remain in their respective group folders.

---

## Notes

- Paper #11 (Kidger) is a **PhD thesis**, not a peer-reviewed conference/journal paper. Retained for its comprehensive technical reference value.
- Paper #15 was published in *International Journal for Numerical Methods in Engineering* (2025). The arXiv 2024 preprint version was downloaded as the open-access version.
- Paper #5 (Onken & Ruthotto) and #9 (DiffEqFlux.jl) and #14 (Universal DEs) are arXiv preprints not published as journal articles under these exact titles.
- All PDFs were downloaded from official arXiv sources.
