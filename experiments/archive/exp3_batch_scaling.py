"""
Experiment 3.3: Batch Size Scaling and Synchronization Penalty.
Tests batch sizes B in [1, 4, 16, 64, 128, 256].

Investigates:
- How batching improves network evaluation throughput (amortizing Python/kernel overheads)
- The Batch Synchronization Penalty in Neural ODEs: when adaptive step size is constrained
  by the worst-case sample in the batch.
Outputs:
- experiments/logs/exp3_batch_scaling.csv
- experiments/plots/batch_scaling_throughput.png
"""

import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.vector_fields import MLPVectorField
from models.neural_ode import NeuralODE
from profiling.logger import ExperimentLogger


def run_batch_scaling():
    print("=== Running Experiment 3.3: Batch Size Scaling ===")
    logger = ExperimentLogger(filename="exp3_batch_scaling.csv")

    batch_sizes = [1, 4, 16, 64, 128, 256]
    t_eval = torch.linspace(0.0, 5.0, 40)

    # Moderate network
    vf = MLPVectorField(state_dim=2, hidden_dim=32, num_layers=3, activation="gelu")
    node = NeuralODE(vf, solver="dopri5", rtol=1e-5, atol=1e-7)

    runtimes = []
    per_sample_times = []
    throughputs = []
    nfes = []

    # Fix random seed for reproducibility
    torch.manual_seed(42)

    for B in batch_sizes:
        print(f"\nEvaluating Batch Size B = {B}...")
        # Initial conditions with slight perturbations across batch
        z0 = torch.randn(B, 2)

        # Warmup
        with torch.no_grad():
            _ = node(z0, t_eval)

        # Timing loop (5 runs)
        times = []
        for _ in range(5):
            node.reset_nfe()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = node(z0, t_eval)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        mean_ms = float(np.mean(times))
        per_sample_ms = mean_ms / B
        throughput = (B / mean_ms) * 1000.0  # Trajectories / second
        recorded_nfe = node.nfe

        runtimes.append(mean_ms)
        per_sample_times.append(per_sample_ms)
        throughputs.append(throughput)
        nfes.append(recorded_nfe)

        logger.log({
            "experiment_id": f"EXP3-BATCH-B{B}",
            "ode_name": "Synthetic-Batch",
            "stiffness": "batch-scaling",
            "solver": "dopri5",
            "rtol": 1e-5,
            "atol": 1e-7,
            "net_preset": "medium",
            "num_params": vf.num_params,
            "device": "cpu",
            "batch_size": B,
            "nfe": recorded_nfe,
            "total_runtime_ms": mean_ms,
            "observation": f"Batch size {B}: {throughput:.1f} traj/sec"
        })

        print(f"  Batch {B:3d}: Total = {mean_ms:6.2f} ms | Per-Sample = {per_sample_ms:6.3f} ms | Throughput = {throughput:7.1f} traj/s | NFE = {recorded_nfe}")

    # Plot Figure: Throughput and Latency scaling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Throughput plot
    ax1.plot(batch_sizes, throughputs, marker="o", lw=2.5, color="#2CA02C")
    ax1.set_xscale("log", base=2)
    ax1.set_title("System Throughput vs. Batch Size", fontweight="bold")
    ax1.set_xlabel("Batch Size (Log Scale)")
    ax1.set_ylabel("Throughput (Trajectories / second)")
    ax1.grid(True, alpha=0.3)

    # Per-sample Latency plot
    ax2.plot(batch_sizes, per_sample_times, marker="s", lw=2.5, color="#D62728")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_title("Per-Sample Wall-Clock Latency (Log Scale)", fontweight="bold")
    ax2.set_xlabel("Batch Size (Log Scale)")
    ax2.set_ylabel("Latency per Sample (ms)")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Batch Scaling Dynamics in Neural ODEs: Amortization vs. Latency", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_path = "experiments/plots/batch_scaling_throughput.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\n[DONE] Experiment 3.3 completed! Plot saved to: {plot_path}")


if __name__ == "__main__":
    run_batch_scaling()
