"""
Pipeline verification test.
Tests ground-truth loading, vector field instantiation, profiling harness, and logging.
"""

import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.vector_fields import get_vector_field
from models.neural_ode import NeuralODE
from profiling.harness import ProfilingHarness
from profiling.logger import ExperimentLogger


def test_pipeline():
    print("Testing pipeline components...")

    # 1. Test ground truth datasets exist
    expected_files = [
        "data/trajectories/lotka-volterra_ground_truth.pt",
        "data/trajectories/fitzhugh-nagumo_ground_truth.pt",
        "data/trajectories/vanderpol_ground_truth.pt",
        "data/trajectories/robertson_ground_truth.pt"
    ]
    for ef in expected_files:
        assert os.path.exists(ef), f"Missing dataset: {ef}"
        data = torch.load(ef)
        print(f"  [OK] Loaded {ef} | Shape: {data['y'].shape}")

    # 2. Test Model & Profiling
    vf = get_vector_field("small", state_dim=2)
    node = NeuralODE(vf, solver="rk4")
    harness = ProfilingHarness(device="cpu")

    z0 = torch.tensor([1.0, 0.5])
    t_eval = torch.linspace(0.0, 2.0, 20)
    options = {"step_size": 0.1}

    results = harness.profile_integration(node, z0, t_eval, options=options, num_repeats=2)
    print(f"  [OK] Profiling output: NFE={results['nfe']}, Time={results['total_time_ms']:.2f}ms")

    # 3. Test Logger
    logger = ExperimentLogger(filename="test_log.csv")
    logger.log({
        "experiment_id": "TEST-001",
        "ode_name": "Lotka-Volterra",
        "stiffness": "non-stiff",
        "solver": "rk4",
        "rtol": 1e-5,
        "atol": 1e-7,
        "step_size": 0.1,
        "net_preset": "small",
        "num_params": vf.num_params,
        "device": "cpu",
        "nfe": results["nfe"],
        "total_runtime_ms": results["total_time_ms"],
        "tf_mean_ms": results["tf_mean_ms"],
        "net_eval_ms": results["total_net_eval_ms"],
        "solver_overhead_ms": results["solver_overhead_ms"],
        "solver_overhead_ratio": results["solver_overhead_ratio"],
        "observation": "Unit test verification log"
    })
    print("  [OK] Logger verification complete!")
    print("\nALL PIPELINE TESTS PASSED!")


if __name__ == "__main__":
    test_pipeline()
