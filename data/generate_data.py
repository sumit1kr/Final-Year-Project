"""
High-Precision Ground Truth Data Generation for Neural ODE Benchmarks.
Uses SciPy's Radau and BDF stiff integrators with tight error tolerances.
Saves PyTorch tensors, CSV files, and verification trajectory plots.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from benchmarks.systems import LotkaVolterra, FitzHughNagumo, VanDerPol, Robertson


def generate_and_save_data():
    os.makedirs("data/trajectories", exist_ok=True)
    os.makedirs("data/plots", exist_ok=True)

    systems = [
        LotkaVolterra(),
        FitzHughNagumo(),
        VanDerPol(mu=10.0),
        Robertson()
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, sys_obj in enumerate(systems):
        print(f"Generating ground truth for: {sys_obj.name} ({sys_obj.stiffness_regime})...")

        t_span = sys_obj.default_t_span
        y0 = sys_obj.default_y0
        num_pts = sys_obj.default_num_points

        if sys_obj.name == "Robertson":
            # Logarithmic time grid for Robertson chemical kinetics (1e-5 to 1e5)
            t_eval = np.logspace(np.log10(t_span[0]), np.log10(t_span[1]), num_pts)
        else:
            t_eval = np.linspace(t_span[0], t_span[1], num_pts)

        # Solve with high-precision Radau (L-stable, 5th order implicit Runge-Kutta)
        sol = solve_ivp(
            fun=sys_obj.rhs,
            t_span=(t_eval[0], t_eval[-1]),
            y0=y0,
            t_eval=t_eval,
            method="Radau",
            jac=sys_obj.jacobian,
            rtol=1e-10,
            atol=1e-12
        )

        if not sol.success:
            print(f"Warning: Integration failed for {sys_obj.name}: {sol.message}")
            continue

        t_data = sol.t
        y_data = sol.y.T  # Shape: [num_pts, dim]

        # 1. Save PyTorch tensors
        torch_data = {
            "name": sys_obj.name,
            "stiffness": sys_obj.stiffness_regime,
            "t": torch.tensor(t_data, dtype=torch.float32),
            "y": torch.tensor(y_data, dtype=torch.float32),
            "y0": torch.tensor(y0, dtype=torch.float32)
        }
        pt_path = f"data/trajectories/{sys_obj.name.lower()}_ground_truth.pt"
        torch.save(torch_data, pt_path)

        # 2. Save CSV format (friendly for Track 1 sharing)
        csv_header = "t," + ",".join([f"y{i+1}" for i in range(sys_obj.dim)])
        csv_matrix = np.hstack([t_data.reshape(-1, 1), y_data])
        csv_path = f"data/trajectories/{sys_obj.name.lower()}_ground_truth.csv"
        np.savetxt(csv_path, csv_matrix, delimiter=",", header=csv_header, comments="")

        print(f"  -> Saved {pt_path} and {csv_path}")

        # 3. Plot individual trajectory
        ax = axes[idx]
        if sys_obj.name == "Robertson":
            ax.set_xscale("log")
            ax.plot(t_data, y_data[:, 0], label="y1 (A)", color="navy", lw=2)
            ax.plot(t_data, y_data[:, 1] * 1e4, label="y2 (B) [x 10^4]", color="crimson", lw=2, linestyle="--")
            ax.plot(t_data, y_data[:, 2], label="y3 (C)", color="forestgreen", lw=2)
            ax.set_title(f"{sys_obj.name} (Stiff Chemical Kinetics)", fontsize=11, fontweight="bold")
            ax.set_xlabel("Time (s) [Log scale]")
            ax.set_ylabel("Concentration")
        else:
            for d in range(sys_obj.dim):
                ax.plot(t_data, y_data[:, d], label=f"y{d+1}", lw=2)
            ax.set_title(f"{sys_obj.name} ({sys_obj.stiffness_regime})", fontsize=11, fontweight="bold")
            ax.set_xlabel("Time t")
            ax.set_ylabel("State Values")

        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    plt.tight_layout()
    overview_plot = "data/plots/all_benchmarks_overview.png"
    plt.savefig(overview_plot, dpi=200)
    plt.close()
    print(f"\n[DONE] Overview verification plot saved to: {overview_plot}")


if __name__ == "__main__":
    generate_and_save_data()
