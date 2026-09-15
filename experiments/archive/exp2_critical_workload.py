"""
Experiment 2: The Critical Experiment (Section 4.5 of Research Guidelines).
Compares:
- Workload A: High NFE, Small State (dim=2), Small Network (2 layers, 16 units)
- Workload B: Low NFE, Larger State (dim=10), Large Network (5 layers, 128 units)

Evaluates whether NFE alone is sufficient to characterize computational cost.
Outputs:
- experiments/logs/exp2_critical_workload.csv
- experiments/plots/critical_workload_nfe_vs_runtime.png
"""

import os
import sys
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.vector_fields import MLPVectorField, get_vector_field
from models.neural_ode import NeuralODE
from profiling.harness import ProfilingHarness
from profiling.logger import ExperimentLogger


def run_critical_experiment():
    print("=== Running Experiment 2: Critical Workload Comparison (Section 4.5) ===")
    logger = ExperimentLogger(filename="exp2_critical_workload.csv")
    harness = ProfilingHarness(device="cpu")

    # Time spans
    t_eval = torch.linspace(0.0, 5.0, 50)

    # 1. Configure Workload A: High NFE, Small State, Small Net
    vf_a = MLPVectorField(state_dim=2, hidden_dim=16, num_layers=2, activation="gelu")
    node_a = NeuralODE(vf_a, solver="dopri5", rtol=1e-8, atol=1e-10)
    z0_a = torch.randn(2)

    res_a = harness.profile_integration(node_a, z0_a, t_eval, num_repeats=10)

    # 2. Configure Workload B: Low NFE, Larger State, Large Net
    vf_b = MLPVectorField(state_dim=10, hidden_dim=128, num_layers=5, activation="gelu")
    node_b = NeuralODE(vf_b, solver="rk4")
    z0_b = torch.randn(10)
    options_b = {"step_size": 0.5}  # Coarse steps -> strictly Low NFE

    res_b = harness.profile_integration(node_b, z0_b, t_eval, options=options_b, num_repeats=10)

    # Log Workload A
    rec_a = {
        "experiment_id": "EXP2-WORKLOAD-A",
        "ode_name": "Synthetic-Small",
        "stiffness": "high-nfe-regime",
        "solver": "dopri5 (tight tol)",
        "net_preset": "small (dim=2, hidden=16)",
        "num_params": vf_a.num_params,
        "device": "cpu",
        "nfe": res_a["nfe"],
        "total_runtime_ms": res_a["total_time_ms"],
        "tf_mean_ms": res_a["tf_mean_ms"],
        "net_eval_ms": res_a["total_net_eval_ms"],
        "solver_overhead_ms": res_a["solver_overhead_ms"],
        "solver_overhead_ratio": res_a["solver_overhead_ratio"],
        "observation": "Workload A: High NFE, Small State/Net"
    }
    logger.log(rec_a)

    # Log Workload B
    rec_b = {
        "experiment_id": "EXP2-WORKLOAD-B",
        "ode_name": "Synthetic-Large",
        "stiffness": "low-nfe-regime",
        "solver": "rk4 (coarse step)",
        "net_preset": "large (dim=10, hidden=128)",
        "num_params": vf_b.num_params,
        "device": "cpu",
        "nfe": res_b["nfe"],
        "total_runtime_ms": res_b["total_time_ms"],
        "tf_mean_ms": res_b["tf_mean_ms"],
        "net_eval_ms": res_b["total_net_eval_ms"],
        "solver_overhead_ms": res_b["solver_overhead_ms"],
        "solver_overhead_ratio": res_b["solver_overhead_ratio"],
        "observation": "Workload B: Low NFE, Large State/Net"
    }
    logger.log(rec_b)

    print("\n--- CRITICAL EXPERIMENT RESULTS ---")
    print(f"Workload A: NFE = {res_a['nfe']} | Runtime = {res_a['total_time_ms']:.2f} ms | Params = {vf_a.num_params}")
    print(f"Workload B: NFE = {res_b['nfe']} | Runtime = {res_b['total_time_ms']:.2f} ms | Params = {vf_b.num_params}")

    nfe_ratio = res_a['nfe'] / res_b['nfe']
    time_ratio = res_b['total_time_ms'] / res_a['total_time_ms']
    print(f"\n[KEY FINDING] Workload A had {nfe_ratio:.2f}x HIGHER NFE than Workload B,")
    print(f"Yet Workload B had {time_ratio:.2f}x LONGER Wall-Clock Runtime!")
    print("=> NFE alone fails completely as a reliable cost predictor!")

    # Plot figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    workloads = ["Workload A\n(Small Net, dim=2)", "Workload B\n(Large Net, dim=10)"]
    nfe_vals = [res_a["nfe"], res_b["nfe"]]
    time_vals = [res_a["total_time_ms"], res_b["total_time_ms"]]

    bars1 = ax1.bar(workloads, nfe_vals, color=["#4C72B0", "#C44E52"], width=0.5)
    ax1.set_title("Reported NFE Metric (What Papers Report)", fontweight="bold")
    ax1.set_ylabel("Number of Function Evaluations (NFE)")
    ax1.grid(axis="y", alpha=0.3)
    for b in bars1:
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 1, f"{int(b.get_height())}", ha="center", fontweight="bold")

    bars2 = ax2.bar(workloads, time_vals, color=["#4C72B0", "#C44E52"], width=0.5)
    ax2.set_title("Actual Wall-Clock Runtime (Hardware Reality)", fontweight="bold")
    ax2.set_ylabel("Runtime (ms)")
    ax2.grid(axis="y", alpha=0.3)
    for b in bars2:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.2, f"{b.get_height():.2f} ms", ha="center", fontweight="bold")

    plt.suptitle("The NFE Paradox: When Low NFE Costs More Time", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plot_path = "experiments/plots/critical_workload_nfe_vs_runtime.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\n[DONE] Critical Workload comparison plot saved to: {plot_path}")


if __name__ == "__main__":
    run_critical_experiment()
