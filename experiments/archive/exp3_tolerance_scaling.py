"""
Experiment 3.1: Tolerance and Step-Size Scaling Across Systems.
Sweeps relative and absolute tolerances:
rtol in [1e-3, 1e-5, 1e-7, 1e-9]
Evaluates on:
- Non-stiff: Lotka-Volterra
- Moderately Stiff: Van der Pol (mu=10)

Quantifies:
- NFE scaling vs. Wall-clock runtime
- Growth of internal solver overhead (T_solver) vs. network evaluation (NFE * T_f)
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.vector_fields import get_vector_field
from models.neural_ode import NeuralODE
from profiling.harness import ProfilingHarness
from profiling.logger import ExperimentLogger


def run_tolerance_scaling():
    print("=== Running Experiment 3.1: Tolerance Scaling ===")
    logger = ExperimentLogger(filename="exp3_tolerance_scaling.csv")
    harness = ProfilingHarness(device="cpu")

    tolerances = [
        {"name": "1e-3", "rtol": 1e-3, "atol": 1e-5},
        {"name": "1e-5", "rtol": 1e-5, "atol": 1e-7},
        {"name": "1e-7", "rtol": 1e-7, "atol": 1e-9},
        {"name": "1e-9", "rtol": 1e-9, "atol": 1e-11}
    ]

    systems = [
        {"name": "Lotka-Volterra", "file": "lotka-volterra_ground_truth.pt", "stiffness": "non-stiff"},
        {"name": "VanDerPol-10", "file": "vanderpol_ground_truth.pt", "stiffness": "moderate-stiff"}
    ]

    all_records = {s["name"]: {"tols": [], "nfe": [], "runtime": [], "solver_overhead": [], "net_eval": []} for s in systems}

    for sys_cfg in systems:
        data_path = os.path.join("data/trajectories", sys_cfg["file"])
        data = torch.load(data_path)
        t_eval = data["t"][:40]
        z0 = data["y0"]
        dim = z0.shape[0]

        vf = get_vector_field("small", state_dim=dim)

        for tol in tolerances:
            exp_id = f"EXP3-TOL-{sys_cfg['name'][:4].upper()}-{tol['name']}"
            node = NeuralODE(vf, solver="dopri5", rtol=tol["rtol"], atol=tol["atol"])

            res = harness.profile_integration(node, z0, t_eval, num_repeats=6)

            logger.log({
                "experiment_id": exp_id,
                "ode_name": sys_cfg["name"],
                "stiffness": sys_cfg["stiffness"],
                "solver": "dopri5",
                "rtol": tol["rtol"],
                "atol": tol["atol"],
                "net_preset": "small",
                "num_params": vf.num_params,
                "device": "cpu",
                "nfe": res["nfe"],
                "total_runtime_ms": res["total_time_ms"],
                "tf_mean_ms": res["tf_mean_ms"],
                "net_eval_ms": res["total_net_eval_ms"],
                "solver_overhead_ms": res["solver_overhead_ms"],
                "solver_overhead_ratio": res["solver_overhead_ratio"],
                "observation": f"Tol sweep {tol['name']} on {sys_cfg['name']}"
            })

            rec = all_records[sys_cfg["name"]]
            rec["tols"].append(tol["name"])
            rec["nfe"].append(res["nfe"])
            rec["runtime"].append(res["total_time_ms"])
            rec["solver_overhead"].append(res["solver_overhead_ms"])
            rec["net_eval"].append(res["total_net_eval_ms"])

    # Plotting Figure: 2x2 Grid
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # 1. NFE vs Tolerance
    for s_name, rec in all_records.items():
        axes[0, 0].plot(rec["tols"], rec["nfe"], marker="o", lw=2, label=s_name)
    axes[0, 0].set_title("NFE vs. Solver Tolerance", fontweight="bold")
    axes[0, 0].set_ylabel("NFE")
    axes[0, 0].set_xlabel("Relative Tolerance (rtol)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # 2. Total Runtime vs Tolerance
    for s_name, rec in all_records.items():
        axes[0, 1].plot(rec["tols"], rec["runtime"], marker="s", lw=2, label=s_name)
    axes[0, 1].set_title("Wall-Clock Runtime (ms) vs. Tolerance", fontweight="bold")
    axes[0, 1].set_ylabel("Runtime (ms)")
    axes[0, 1].set_xlabel("Relative Tolerance (rtol)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # 3. Stacked Decomposition for Lotka-Volterra
    lv = all_records["Lotka-Volterra"]
    x = np.arange(len(lv["tols"]))
    axes[1, 0].bar(x, lv["net_eval"], label="Network Eval (NFE * T_f)", color="#4C72B0", alpha=0.85)
    axes[1, 0].bar(x, lv["solver_overhead"], bottom=lv["net_eval"], label="Solver Overhead (T_solver)", color="#C44E52", alpha=0.85)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(lv["tols"])
    axes[1, 0].set_title("Runtime Breakdown: Lotka-Volterra (Non-Stiff)", fontweight="bold")
    axes[1, 0].set_ylabel("Time (ms)")
    axes[1, 0].set_xlabel("Tolerance (rtol)")
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.3)

    # 4. Stacked Decomposition for Van der Pol
    vdp = all_records["VanDerPol-10"]
    axes[1, 1].bar(x, vdp["net_eval"], label="Network Eval (NFE * T_f)", color="#4C72B0", alpha=0.85)
    axes[1, 1].bar(x, vdp["solver_overhead"], bottom=vdp["net_eval"], label="Solver Overhead (T_solver)", color="#C44E52", alpha=0.85)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(vdp["tols"])
    axes[1, 1].set_title("Runtime Breakdown: Van der Pol (Moderate Stiff)", fontweight="bold")
    axes[1, 1].set_ylabel("Time (ms)")
    axes[1, 1].set_xlabel("Tolerance (rtol)")
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", alpha=0.3)

    plt.suptitle("Impact of Tolerance on Computational Cost and Internal Solver Overheads", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_path = "experiments/plots/tolerance_vs_nfe_runtime.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\n[DONE] Experiment 3.1 completed! Plot saved to: {plot_path}")


if __name__ == "__main__":
    run_tolerance_scaling()
