"""
Experiment 1: Solver Behavior Across the Stiffness Spectrum.
Evaluates how explicit solvers (Euler, RK4, Dopri5) behave across:
- Non-stiff: Lotka-Volterra
- Mildly stiff: FitzHugh-Nagumo
- Moderately stiff: Van der Pol (mu=10)
- Stiff: Van der Pol (mu=50)

Measures: NFE, Runtime (ms), Solver Overhead (%).
Outputs CSV logs and visualization plots.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.vector_fields import get_vector_field
from models.neural_ode import NeuralODE
from profiling.harness import ProfilingHarness
from profiling.logger import ExperimentLogger


def run_solver_sweep():
    print("=== Running Experiment 1: Solver Sweep Across Stiffness ===")
    logger = ExperimentLogger(filename="exp1_solver_sweep.csv")
    harness = ProfilingHarness(device="cpu")

    test_configs = [
        {"name": "Lotka-Volterra", "file": "lotka-volterra_ground_truth.pt", "stiffness": "non-stiff"},
        {"name": "FitzHugh-Nagumo", "file": "fitzhugh-nagumo_ground_truth.pt", "stiffness": "non-stiff"},
        {"name": "VanDerPol-10", "file": "vanderpol_ground_truth.pt", "stiffness": "moderate-stiff"}
    ]

    solvers = ["euler", "rk4", "dopri5"]
    records = []

    for cfg in test_configs:
        data_path = os.path.join("data/trajectories", cfg["file"])
        data = torch.load(data_path)
        t_eval = data["t"][:50]  # First 50 points
        z0 = data["y0"]
        dim = z0.shape[0]

        vf = get_vector_field("small", state_dim=dim)

        for solver in solvers:
            exp_id = f"EXP1-{cfg['name'][:4].upper()}-{solver.upper()}"
            options = {"step_size": float(t_eval[1] - t_eval[0])} if solver in ["euler", "rk4"] else None

            node = NeuralODE(vf, solver=solver, rtol=1e-5, atol=1e-7)
            res = harness.profile_integration(node, z0, t_eval, options=options, num_repeats=5)

            rec = {
                "experiment_id": exp_id,
                "ode_name": cfg["name"],
                "stiffness": cfg["stiffness"],
                "solver": solver,
                "net_preset": "small",
                "num_params": vf.num_params,
                "device": "cpu",
                "nfe": res["nfe"],
                "total_runtime_ms": res["total_time_ms"],
                "tf_mean_ms": res["tf_mean_ms"],
                "net_eval_ms": res["total_net_eval_ms"],
                "solver_overhead_ms": res["solver_overhead_ms"],
                "solver_overhead_ratio": res["solver_overhead_ratio"],
                "observation": f"{solver} on {cfg['name']} ({cfg['stiffness']})"
            }
            logger.log(rec)
            records.append(rec)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    names = [f"{r['ode_name'][:4]}_{r['solver']}" for r in records]
    nfes = [r["nfe"] for r in records]
    times = [r["total_runtime_ms"] for r in records]
    overheads = [r["solver_overhead_ratio"] * 100 for r in records]

    colors = ["#4C72B0" if "dopri" in n else "#55A868" if "rk4" in n else "#C44E52" for n in names]

    ax1.bar(names, times, color=colors, alpha=0.85)
    ax1.set_title("Runtime (ms) by Solver & System", fontweight="bold")
    ax1.set_ylabel("Runtime (ms)")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(names, overheads, color="purple", alpha=0.7)
    ax2.set_title("Internal Solver Overhead (% of Total Time)", fontweight="bold")
    ax2.set_ylabel("Overhead (%)")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = "experiments/plots/solver_stiffness_sweep.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\n[DONE] Experiment 1 finished! Plot saved to: {plot_path}")


if __name__ == "__main__":
    run_solver_sweep()
