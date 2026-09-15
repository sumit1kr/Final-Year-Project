"""
Experiment 3.2: The Stiffness Cliff & Solver Breakdown.
Sweeps the Van der Pol stiffness parameter mu from 1 (non-stiff) to 100 (highly stiff).
Compares:
- Explicit adaptive solver: dopri5 (Runge-Kutta 4/5)
- Stiff-capable implicit solver: Radau (SciPy implicit Runge-Kutta)

Identifies the exact stiffness threshold where explicit methods collapse computationally.
Outputs:
- experiments/logs/exp3_stiffness_breakdown.csv
- experiments/plots/stiffness_scaling_cliff.png
"""

import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from benchmarks.systems import VanDerPol
from models.vector_fields import MLPVectorField
from models.neural_ode import NeuralODE
from profiling.logger import ExperimentLogger


def run_stiffness_breakdown():
    print("=== Running Experiment 3.2: The Stiffness Cliff ===")
    logger = ExperimentLogger(filename="exp3_stiffness_breakdown.csv")

    mu_values = [1.0, 5.0, 10.0, 25.0, 50.0, 100.0]
    t_span = (0.0, 20.0)
    t_eval = torch.linspace(t_span[0], t_span[1], 100)
    t_eval_np = t_eval.numpy()
    z0 = torch.tensor([2.0, 0.0])
    z0_np = z0.numpy()

    # Fixed neural vector field for consistent comparisons
    vf = MLPVectorField(state_dim=2, hidden_dim=32, num_layers=3, activation="gelu")
    node = NeuralODE(vf, solver="dopri5", rtol=1e-5, atol=1e-7)

    dopri_nfes = []
    dopri_times = []
    radau_nfes = []
    radau_times = []

    for mu in mu_values:
        print(f"\nEvaluating stiffness mu = {mu}...")
        vdp = VanDerPol(mu=mu)

        # 1. Evaluate explicit solver (dopri5) on actual system trajectory fitting
        t0 = time.perf_counter()
        sol_explicit = solve_ivp(vdp.rhs, t_span, z0_np, method="RK45", rtol=1e-5, atol=1e-7)
        t1 = time.perf_counter()
        rk45_nfe = sol_explicit.nfev
        rk45_time_ms = (t1 - t0) * 1000.0

        dopri_nfes.append(rk45_nfe)
        dopri_times.append(rk45_time_ms)

        # 2. Evaluate stiff implicit solver (Radau)
        t0 = time.perf_counter()
        sol_implicit = solve_ivp(vdp.rhs, t_span, z0_np, method="Radau", jac=vdp.jacobian, rtol=1e-5, atol=1e-7)
        t1 = time.perf_counter()
        radau_nfe = sol_implicit.nfev
        radau_time_ms = (t1 - t0) * 1000.0

        radau_nfes.append(radau_nfe)
        radau_times.append(radau_time_ms)

        # Log both
        logger.log({
            "experiment_id": f"EXP3-STIFF-MU{int(mu)}-EXPLICIT",
            "ode_name": "VanDerPol",
            "stiffness": f"mu={mu}",
            "solver": "RK45/Dopri5 (Explicit)",
            "rtol": 1e-5,
            "atol": 1e-7,
            "nfe": rk45_nfe,
            "total_runtime_ms": rk45_time_ms,
            "status": "SUCCESS" if sol_explicit.success else "FAILED",
            "observation": f"Explicit solver at mu={mu}"
        })
        logger.log({
            "experiment_id": f"EXP3-STIFF-MU{int(mu)}-IMPLICIT",
            "ode_name": "VanDerPol",
            "stiffness": f"mu={mu}",
            "solver": "Radau (Implicit Stiff)",
            "rtol": 1e-5,
            "atol": 1e-7,
            "nfe": radau_nfe,
            "total_runtime_ms": radau_time_ms,
            "status": "SUCCESS" if sol_implicit.success else "FAILED",
            "observation": f"Implicit Radau solver at mu={mu}"
        })

        print(f"  Explicit (RK45): NFE = {rk45_nfe:6d} | Time = {rk45_time_ms:8.2f} ms")
        print(f"  Implicit (Radau): NFE = {radau_nfe:6d} | Time = {radau_time_ms:8.2f} ms")

    # Plot Figure: NFE Explosion and Runtime Crossover
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # NFE comparison
    ax1.plot(mu_values, dopri_nfes, marker="o", lw=2.5, color="#C44E52", label="Explicit (RK45/Dopri5)")
    ax1.plot(mu_values, radau_nfes, marker="s", lw=2.5, color="#4C72B0", label="Implicit (Radau)")
    ax1.set_yscale("log")
    ax1.set_title("NFE Explosion vs. Stiffness (Log Scale)", fontweight="bold")
    ax1.set_xlabel("Stiffness Parameter mu")
    ax1.set_ylabel("Number of Function Evaluations (NFE)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Runtime comparison
    ax2.plot(mu_values, dopri_times, marker="o", lw=2.5, color="#C44E52", label="Explicit (RK45/Dopri5)")
    ax2.plot(mu_values, radau_times, marker="s", lw=2.5, color="#4C72B0", label="Implicit (Radau)")
    ax2.set_yscale("log")
    ax2.set_title("Wall-Clock Time Crossover (Log Scale)", fontweight="bold")
    ax2.set_xlabel("Stiffness Parameter mu")
    ax2.set_ylabel("Runtime (ms)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle("The Stiffness Cliff: Computational Breakdown of Explicit Solvers", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_path = "experiments/plots/stiffness_scaling_cliff.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\n[DONE] Experiment 3.2 completed! Plot saved to: {plot_path}")


if __name__ == "__main__":
    run_stiffness_breakdown()
