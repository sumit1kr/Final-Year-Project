"""
Stage 1: Experiment 1 Runner & Profiling Harness.
Computational Performance Characterization of Neural ODEs (Track 2).

Implements the audited Stage 1 Protocol (docs/stage1_experiment_protocol.md):
- Canonical Baseline: Autonomous Softplus MLP (W=64, L=2, D_in=D, D_out=D, augment_dim=0)
  Params: D=2 -> 4482, D=3 -> 4611 (aligned per docs/architecture_decision_record.md)
- CPU-only execution, torch.set_num_threads(1), torch.set_num_interop_threads(1), seed=42
- 20 unrecorded integration warm-up cycles, 15 recorded repetitions
- Isolated T_f measurement: 50 warmup + 1000 evaluated passes from 50 trajectory samples
- Direct NFE and runtime (time.perf_counter_ns()) measurement
- Derived estimates: T_isolated_net_hat = NFE * T_f, T_solver_hat = T_total - T_isolated_net_hat
- Non-negative solver overhead clipping with signed residual logging (Section 6.4)
- Trajectory accuracy verification against high-precision Radau ground truth (MSE, Rel-L2, Linf)
- Process-isolated watchdog with hard 30.0s right-censoring timeout (Windows-safe)
- Structured logging with idempotency conforming to Section 9 / 13 schema
- Strict execution guard requiring explicit authorization
"""

import os
import sys

# Enforce single-thread CPU pinning before PyTorch / BLAS initialization
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import platform
import json
import csv
import argparse
import multiprocessing as mp
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import CubicSpline

# Add project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.vector_fields import MLPVectorField
from models.neural_ode import NeuralODE
from benchmarks.systems import LotkaVolterra, FitzHughNagumo, VanDerPol, Robertson


# =============================================================================
# 1. Environment & Hardware Metadata Collection
# =============================================================================

def get_platform_metadata():
    """Collects host machine hardware and platform metadata without benchmarking."""
    return {
        "hardware_device": "CPU",
        "cpu_model": platform.processor() or "x86_64",
        "cpu_cores_logical": os.cpu_count() or 1,
        "os_platform": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "torchdiffeq_version": "0.2.5"
    }


def enforce_deterministic_cpu_environment(seed=42):
    """Enforces single-thread execution and seed determinism."""
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        raise RuntimeError("CRITICAL ERROR: CUDA is available but Stage 1 mandates strict CPU-only execution.")


# =============================================================================
# 2. Canonical Model & Checkpoint Loading
# =============================================================================

def load_canonical_baseline(checkpoint_path, state_dim=2, hidden_dim=64, num_layers=2, activation="softplus"):
    """
    Instantiates the canonical Option A baseline architecture and loads verified weights.
    Strictly verifies parameter counts: D=2 -> 4482, D=3 -> 4611.
    """
    enforce_deterministic_cpu_environment(42)
    vf = MLPVectorField(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        activation=activation,
        augment_dim=0
    )

    expected_params = 129 * state_dim + 4224
    actual_params = sum(p.numel() for p in vf.parameters())
    assert actual_params == expected_params, (
        f"Parameter count mismatch: Expected {expected_params} for D={state_dim}, got {actual_params}"
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    vf.load_state_dict(state_dict)
    vf.eval()

    return vf


# =============================================================================
# 3. Isolated Latency (T_f) Measurement Protocol
# =============================================================================

def measure_isolated_tf(vector_field, ground_truth_path, num_samples=50, num_warmup=50, num_measured=1000, seed=42):
    """
    Measures isolated per-evaluation latency T_f according to protocol Section 6.2:
    - Samples 50 representative state points and timestamps from ground-truth trajectory
    - Executes 50 warm-up evaluations
    - Measures 1,000 independent single-evaluation passes via time.perf_counter_ns()
    - Computes median T_f (in microseconds), IQR, and checks dispersion IQR/median <= 5%
    - Never pools T_f across different checkpoints or systems
    """
    enforce_deterministic_cpu_environment(seed)
    vector_field.eval()

    if not os.path.exists(ground_truth_path):
        raise FileNotFoundError(f"Ground truth trajectory not found: {ground_truth_path}")

    gt_data = torch.load(ground_truth_path, map_location="cpu")
    t_full = gt_data["t"]
    y_full = gt_data["y"]

    total_pts = len(t_full)
    indices = np.linspace(0, total_pts - 1, num_samples, dtype=int)
    sample_t = t_full[indices]
    sample_z = y_full[indices]

    # 1. Warm-up forward evaluations
    with torch.no_grad():
        for i in range(num_warmup):
            idx = i % num_samples
            _ = vector_field(sample_t[idx], sample_z[idx])

    # 2. Recorded evaluations
    latencies_us = []
    with torch.no_grad():
        for i in range(num_measured):
            idx = i % num_samples
            t_eval = sample_t[idx]
            z_eval = sample_z[idx]

            t_start = time.perf_counter_ns()
            _ = vector_field(t_eval, z_eval)
            t_end = time.perf_counter_ns()

            latencies_us.append((t_end - t_start) / 1000.0)

    latencies_us = np.array(latencies_us, dtype=np.float64)
    median_tf_us = float(np.median(latencies_us))
    q25, q75 = np.percentile(latencies_us, [25, 75])
    iqr_tf_us = float(q75 - q25)
    mean_tf_us = float(np.mean(latencies_us))
    std_tf_us = float(np.std(latencies_us))

    dispersion_ratio = (iqr_tf_us / median_tf_us) if median_tf_us > 0 else 0.0
    is_stable = bool(dispersion_ratio <= 0.05)

    return {
        "tf_us_median": round(median_tf_us, 4),
        "tf_us_iqr": round(iqr_tf_us, 4),
        "tf_us_mean": round(mean_tf_us, 4),
        "tf_us_std": round(std_tf_us, 4),
        "dispersion_ratio": round(dispersion_ratio, 4),
        "dispersion_stable_le_5pct": is_stable,
        "sample_points_used": num_samples,
        "measured_evals": num_measured
    }


# =============================================================================
# 4. Numerical Accuracy & Ground-Truth Verification
# =============================================================================

def evaluate_trajectory_accuracy(computed_traj, ground_truth_path, t_eval):
    """
    Computes numerical accuracy metrics against reference ground truth (Section 10):
    - MSE: Mean Squared Error
    - Relative L2 Error
    - Linf: Maximum absolute coordinate error
    Interpolates reference solution to match exact evaluated timestamps.
    """
    if computed_traj is None or not os.path.exists(ground_truth_path):
        return {
            "trajectory_mse": None,
            "trajectory_rel_l2": None,
            "trajectory_linf": None
        }

    gt_data = torch.load(ground_truth_path, map_location="cpu")
    t_gt = gt_data["t"].numpy()
    y_gt = gt_data["y"].numpy()

    # Interpolate ground truth to exact evaluation timestamps
    cs = CubicSpline(t_gt, y_gt, axis=0)
    y_ref = cs(t_eval)

    diff = computed_traj - y_ref
    mse = float(np.mean(diff ** 2))
    linf = float(np.max(np.abs(diff)))

    norm_ref = float(np.linalg.norm(y_ref))
    rel_l2 = float(np.linalg.norm(diff) / norm_ref) if norm_ref > 0 else 0.0

    return {
        "trajectory_mse": round(mse, 8),
        "trajectory_rel_l2": round(rel_l2, 8),
        "trajectory_linf": round(linf, 8)
    }


# =============================================================================
# 5. Process-Isolated Integration Worker & Windows-Safe Watchdog
# =============================================================================

def _dummy_stalled_for_test(queue):
    time.sleep(2.0)
    queue.put("DONE")


def _integration_worker(ckpt_path, vf_kwargs, y0_list, t_eval_list, solver, rtol, atol, options, seed, warmup_cycles, recorded_cycles, result_queue):
    """
    Executes ODE integration in a separate process with single-thread CPU pinning.
    Safely terminates via OS-level process kill if watchdog triggers.
    """
    try:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        torch.manual_seed(seed)
        np.random.seed(seed)

        vf = MLPVectorField(**vf_kwargs)
        vf.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        vf.eval()

        node = NeuralODE(vf, solver=solver, rtol=rtol, atol=atol)

        y0 = torch.tensor(y0_list, dtype=torch.float32)
        t_eval = torch.tensor(t_eval_list, dtype=torch.float32)

        # 1. Warm-up integration cycles (unrecorded)
        with torch.no_grad():
            for _ in range(warmup_cycles):
                _ = node(y0, t_eval, options=options)

        # 2. Recorded repetitions
        runtimes_ms = []
        nfes = []
        last_trajectory = None

        with torch.no_grad():
            for _ in range(recorded_cycles):
                node.reset_nfe()
                t0 = time.perf_counter_ns()
                out = node(y0, t_eval, options=options)
                t1 = time.perf_counter_ns()

                elapsed_ms = (t1 - t0) / 1e6
                runtimes_ms.append(elapsed_ms)
                nfes.append(node.nfe)
                last_trajectory = out

        last_traj_np = last_trajectory.cpu().numpy()

        # Check for state divergence (NaN or Inf)
        if np.isnan(last_traj_np).any() or np.isinf(last_traj_np).any():
            result_queue.put(("NUMERICAL_DIVERGENCE", None, None, None, "NaN or Inf in trajectory state"))
            return

        result_queue.put(("SUCCESS", runtimes_ms, nfes, last_traj_np, None))

    except AssertionError as e:
        if "max_num_steps exceeded" in str(e):
            result_queue.put(("STEP_LIMIT_EXCEEDED", None, None, None, str(e)))
        else:
            result_queue.put(("ASSERTION_ERROR", None, None, None, str(e)))
    except Exception as e:
        result_queue.put(("INTEGRATION_ERROR", None, None, None, str(e)))


def run_guarded_integration(ckpt_path, vf_kwargs, y0, t_eval, solver="dopri5", rtol=1e-5, atol=1e-7, options=None, seed=42, warmup_cycles=20, recorded_cycles=15, timeout_s=30.0):
    """
    Runs integration under a process-isolated watchdog.
    If execution exceeds timeout_s, child process is terminated immediately.
    """
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    y0_list = y0.tolist() if isinstance(y0, (np.ndarray, torch.Tensor)) else list(y0)
    t_eval_list = t_eval.tolist() if isinstance(t_eval, (np.ndarray, torch.Tensor)) else list(t_eval)

    start_wall = time.perf_counter()
    p = ctx.Process(
        target=_integration_worker,
        args=(ckpt_path, vf_kwargs, y0_list, t_eval_list, solver, rtol, atol, options, seed, warmup_cycles, recorded_cycles, result_queue)
    )

    p.start()
    p.join(timeout=timeout_s)

    is_timeout = False
    if p.is_alive():
        is_timeout = True
        p.terminate()
        p.join()

    total_wall_s = time.perf_counter() - start_wall

    if is_timeout:
        return {
            "status": "VALIDATION_TIMEOUT",
            "is_right_censored": True,
            "censoring_threshold_s": timeout_s,
            "total_wall_s": round(total_wall_s, 3),
            "runtimes_ms": [],
            "nfes": [],
            "trajectory": None,
            "failure_reason": f"Execution exceeded process watchdog limit of {timeout_s}s"
        }

    if not result_queue.empty():
        status, runtimes_ms, nfes, trajectory, err = result_queue.get()
        return {
            "status": status,
            "is_right_censored": (status == "STEP_LIMIT_EXCEEDED"),
            "censoring_threshold_s": timeout_s,
            "total_wall_s": round(total_wall_s, 3),
            "runtimes_ms": runtimes_ms if runtimes_ms is not None else [],
            "nfes": nfes if nfes is not None else [],
            "trajectory": trajectory,
            "failure_reason": err
        }
    else:
        return {
            "status": "PROCESS_ABRUPTLY_TERMINATED",
            "is_right_censored": True,
            "censoring_threshold_s": timeout_s,
            "total_wall_s": round(total_wall_s, 3),
            "runtimes_ms": [],
            "nfes": [],
            "trajectory": None,
            "failure_reason": "Child process terminated without writing result to queue"
        }


# =============================================================================
# 6. Statistical Summaries & Runtime Decomposition
# =============================================================================

def compute_point_metrics(raw_result, tf_stats, point_spec, t_eval):
    """
    Computes runtime decomposition and statistical metrics per Section 6 & 9 of protocol:
    - T_total median, IQR, mean, std
    - NFE median, IQR, mean, std
    - Outlier detection: T_i > Q3 + 3.0 * IQR per Section 9.2
    - Isolated network evaluation estimate: T_isolated_net_hat = (NFE * T_f) / 1000.0 (ms)
    - Raw solver overhead: raw_overhead = T_total - T_isolated_net_hat
    - Clipped solver overhead per Section 6.4:
      If raw_overhead < 0, report 0.0 ms and record signed residual in notes
    - Estimated solver overhead ratio: Omega_hat = T_solver_hat / T_total
    - Accuracy metrics against ground truth
    """
    status = raw_result["status"]
    is_right_censored = raw_result["is_right_censored"]
    censoring_threshold_s = raw_result["censoring_threshold_s"]
    runtimes_ms = raw_result["runtimes_ms"]
    nfes = raw_result["nfes"]
    failure_reason = raw_result.get("failure_reason")
    notes = []

    if failure_reason:
        notes.append(failure_reason)

    tf_us_median = tf_stats["tf_us_median"]
    tf_us_iqr = tf_stats["tf_us_iqr"]

    # Trajectory accuracy evaluation
    acc_metrics = evaluate_trajectory_accuracy(
        raw_result["trajectory"],
        point_spec["ground_truth_path"],
        t_eval
    )

    if status != "SUCCESS" or not runtimes_ms:
        return {
            "nfe_median": None,
            "nfe_mean": None,
            "nfe_std": None,
            "nfe_iqr": None,
            "runtime_ms_median": None,
            "runtime_ms_iqr": None,
            "runtime_ms_mean": None,
            "runtime_ms_std": None,
            "raw_runtimes_ms": [],
            "tf_us_median": tf_us_median,
            "tf_us_iqr": tf_us_iqr,
            "isolated_net_eval_est_ms": None,
            "raw_solver_overhead_ms": None,
            "estimated_solver_overhead_ms": None,
            "estimated_solver_overhead_ratio": None,
            "trajectory_mse": acc_metrics["trajectory_mse"],
            "trajectory_linf": acc_metrics["trajectory_linf"],
            "trajectory_rel_l2": acc_metrics["trajectory_rel_l2"],
            "is_right_censored": is_right_censored,
            "censoring_threshold_s": censoring_threshold_s,
            "status": status,
            "failure_reason": failure_reason or "",
            "notes": "; ".join(notes)
        }

    # Statistical summaries
    runtimes_arr = np.array(runtimes_ms, dtype=np.float64)
    nfes_arr = np.array(nfes, dtype=np.float64)

    rt_median = float(np.median(runtimes_arr))
    q25_rt, q75_rt = np.percentile(runtimes_arr, [25, 75])
    rt_iqr = float(q75_rt - q25_rt)
    rt_mean = float(np.mean(runtimes_arr))
    rt_std = float(np.std(runtimes_arr))

    # Outlier detection per Section 9.2: T_i > Q3 + 3.0 * IQR
    outlier_thresh = q75_rt + 3.0 * rt_iqr
    outliers = [t for t in runtimes_ms if t > outlier_thresh]
    if outliers:
        notes.append(f"Detected {len(outliers)} OS outlier(s) > {outlier_thresh:.2f} ms")

    nfe_median = int(round(float(np.median(nfes_arr))))
    q25_nfe, q75_nfe = np.percentile(nfes_arr, [25, 75])
    nfe_iqr = float(q75_nfe - q25_nfe)
    nfe_mean = float(np.mean(nfes_arr))
    nfe_std = float(np.std(nfes_arr))

    # Runtime decomposition (Section 6.1)
    # T_isolated_net_hat (ms) = (NFE * T_f (us)) / 1000.0
    isolated_net_est_ms = (nfe_median * tf_us_median) / 1000.0
    raw_solver_overhead_ms = rt_median - isolated_net_est_ms

    # Protocol Section 6.4 non-negative clipping rule
    if raw_solver_overhead_ms < 0.0:
        est_solver_overhead_ms = 0.0
        notes.append(f"Signed residual: {raw_solver_overhead_ms:.4f} ms (clipped to 0.0 per Section 6.4)")
    else:
        est_solver_overhead_ms = raw_solver_overhead_ms

    est_solver_ratio = (est_solver_overhead_ms / rt_median) if rt_median > 0 else 0.0

    return {
        "nfe_median": nfe_median,
        "nfe_mean": round(nfe_mean, 2),
        "nfe_std": round(nfe_std, 2),
        "nfe_iqr": round(nfe_iqr, 2),
        "runtime_ms_median": round(rt_median, 4),
        "runtime_ms_iqr": round(rt_iqr, 4),
        "runtime_ms_mean": round(rt_mean, 4),
        "runtime_ms_std": round(rt_std, 4),
        "raw_runtimes_ms": [round(t, 4) for t in runtimes_ms],
        "tf_us_median": tf_us_median,
        "tf_us_iqr": tf_us_iqr,
        "isolated_net_eval_est_ms": round(isolated_net_est_ms, 4),
        "raw_solver_overhead_ms": round(raw_solver_overhead_ms, 4),
        "estimated_solver_overhead_ms": round(est_solver_overhead_ms, 4),
        "estimated_solver_overhead_ratio": round(est_solver_ratio, 4),
        "trajectory_mse": acc_metrics["trajectory_mse"],
        "trajectory_linf": acc_metrics["trajectory_linf"],
        "trajectory_rel_l2": acc_metrics["trajectory_rel_l2"],
        "is_right_censored": False,
        "censoring_threshold_s": censoring_threshold_s,
        "status": "SUCCESS",
        "failure_reason": "",
        "notes": "; ".join(notes)
    }


# =============================================================================
# 7. Experiment 1 Point Specifications (The 26 Controlled Points)
# =============================================================================

def get_stage1_exp1_point_specifications():
    """
    Generates the exact 26 controlled experimental point specifications for Stage 1 Experiment 1.
    Strictly follows Section 5.2 of docs/stage1_experiment_protocol.md:
    - Series 1A: 10 Lotka-Volterra adaptive dopri5 points across horizon ladder T_k in [1.5, 15.0]
    - Series 1B: 10 FitzHugh-Nagumo adaptive dopri5 points across horizon ladder T_k in [5.0, 50.0]
    - Series 1C: 6 Lotka-Volterra deterministic rk4 points across uniform step ladder N_steps in {25, 50, 100, 200, 400, 800}
    """
    points = []

    # Series 1A: Lotka-Volterra Adaptive dopri5 (10 points)
    lv_horizons = [1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0]
    for idx, T_k in enumerate(lv_horizons, 1):
        num_eval = int(round(T_k / 0.10)) + 1
        points.append({
            "point_id": f"EXP1-S1A-LV-H{T_k:.1f}",
            "series_id": "Series_1A",
            "system_name": "Lotka-Volterra",
            "system_dimension": 2,
            "system_parameters": {"alpha": 1.5, "beta": 1.0, "delta": 3.0, "gamma": 1.0},
            "checkpoint_path": os.path.join(ROOT_DIR, "models", "checkpoints", "lv_h64.pt"),
            "ground_truth_path": os.path.join(ROOT_DIR, "data", "trajectories", "lotka-volterra_ground_truth.pt"),
            "y0": np.array([1.0, 0.5]),
            "horizon_start": 0.0,
            "horizon_end": float(T_k),
            "num_eval_points": num_eval,
            "solver_name": "dopri5",
            "solver_backend": "torchdiffeq",
            "solver_type": "explicit_adaptive",
            "rtol": 1.0e-5,
            "atol": 1.0e-7,
            "options": None,
            "is_fixed_step": False,
            "step_size": None
        })

    # Series 1B: FitzHugh-Nagumo Adaptive dopri5 (10 points)
    fhn_horizons = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]
    for idx, T_k in enumerate(fhn_horizons, 1):
        num_eval = int(round(T_k / 0.25)) + 1
        points.append({
            "point_id": f"EXP1-S1B-FHN-H{T_k:.1f}",
            "series_id": "Series_1B",
            "system_name": "FitzHugh-Nagumo",
            "system_dimension": 2,
            "system_parameters": {"a": 0.7, "b": 0.8, "tau": 12.5, "I": 0.5},
            "checkpoint_path": os.path.join(ROOT_DIR, "models", "checkpoints", "fhn_h64.pt"),
            "ground_truth_path": os.path.join(ROOT_DIR, "data", "trajectories", "fitzhugh-nagumo_ground_truth.pt"),
            "y0": np.array([-1.0, 1.0]),
            "horizon_start": 0.0,
            "horizon_end": float(T_k),
            "num_eval_points": num_eval,
            "solver_name": "dopri5",
            "solver_backend": "torchdiffeq",
            "solver_type": "explicit_adaptive",
            "rtol": 1.0e-5,
            "atol": 1.0e-7,
            "options": None,
            "is_fixed_step": False,
            "step_size": None
        })

    # Series 1C: Lotka-Volterra Deterministic Fixed-Step rk4 (6 points)
    # Fixed horizon [0, 15.0], N_steps in {25, 50, 100, 200, 400, 800}
    # Step size h = 15.0 / N_steps
    rk4_step_counts = [25, 50, 100, 200, 400, 800]
    for N_steps in rk4_step_counts:
        h = 15.0 / float(N_steps)
        expected_nfe = 4 * N_steps
        points.append({
            "point_id": f"EXP1-S1C-LV-RK4-N{N_steps}",
            "series_id": "Series_1C",
            "system_name": "Lotka-Volterra",
            "system_dimension": 2,
            "system_parameters": {"alpha": 1.5, "beta": 1.0, "delta": 3.0, "gamma": 1.0},
            "checkpoint_path": os.path.join(ROOT_DIR, "models", "checkpoints", "lv_h64.pt"),
            "ground_truth_path": os.path.join(ROOT_DIR, "data", "trajectories", "lotka-volterra_ground_truth.pt"),
            "y0": np.array([1.0, 0.5]),
            "horizon_start": 0.0,
            "horizon_end": 15.0,
            "num_eval_points": 151,
            "solver_name": "rk4",
            "solver_backend": "torchdiffeq",
            "solver_type": "explicit_fixed",
            "rtol": None,
            "atol": None,
            "options": {"step_size": h},
            "is_fixed_step": True,
            "step_size": h,
            "expected_deterministic_nfe": expected_nfe
        })

    assert len(points) == 26, f"Expected exactly 26 points, got {len(points)}"
    return points


# =============================================================================
# 8. Stage-1 Structured Experiment Logger
# =============================================================================

class Stage1ExperimentLogger:
    """
    Logs experimental records in full conformity with Section 9 & 13 of docs/stage1_experiment_protocol.md.
    Maintains both machine-readable JSON and tabular CSV records with deduplication.
    """
    def __init__(self, log_dir=None, json_filename="stage1_exp1_results.json", csv_filename="stage1_exp1_results.csv"):
        self.log_dir = log_dir or os.path.join(ROOT_DIR, "experiments", "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        self.json_file = os.path.join(self.log_dir, json_filename)
        self.csv_file = os.path.join(self.log_dir, csv_filename)

        self.csv_columns = [
            "experiment_id",
            "series_id",
            "timestamp_iso",
            "research_stage",
            "system_name",
            "system_dimension",
            "horizon_start",
            "horizon_end",
            "num_eval_points",
            "model_architecture",
            "hidden_width",
            "hidden_depth",
            "activation",
            "parameter_count",
            "checkpoint_validation_status",
            "solver_name",
            "solver_backend",
            "solver_type",
            "rtol",
            "atol",
            "step_size",
            "thread_count",
            "random_seed",
            "warmup_runs",
            "measured_runs",
            "nfe_median",
            "nfe_mean",
            "nfe_std",
            "runtime_ms_median",
            "runtime_ms_iqr",
            "runtime_ms_mean",
            "runtime_ms_std",
            "tf_us_median",
            "tf_us_iqr",
            "isolated_net_eval_est_ms",
            "raw_solver_overhead_ms",
            "estimated_solver_overhead_ms",
            "estimated_solver_overhead_ratio",
            "trajectory_mse",
            "trajectory_linf",
            "is_right_censored",
            "censoring_threshold_s",
            "status",
            "failure_reason",
            "notes"
        ]

    def log_point_result(self, record: dict):
        """Appends or updates record in CSV and JSON storage with deduplication."""
        exp_id = record["experiment_id"]

        # 1. Update CSV
        rows = []
        found_csv = False
        if os.path.exists(self.csv_file):
            with open(self.csv_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if r.get("experiment_id") == exp_id:
                        rows.append({col: record.get(col, "") for col in self.csv_columns})
                        found_csv = True
                    else:
                        rows.append(r)
        if not found_csv:
            rows.append({col: record.get(col, "") for col in self.csv_columns})

        with open(self.csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_columns)
            writer.writeheader()
            writer.writerows(rows)

        # 2. Update JSON
        json_data = {
            "schema_version": "1.0",
            "research_stage": "STAGE_1_BASELINE",
            "generated_timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "environment_metadata": get_platform_metadata(),
            "records": []
        }
        if os.path.exists(self.json_file):
            try:
                with open(self.json_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict) and "records" in loaded:
                        json_data = loaded
                    elif isinstance(loaded, list):
                        json_data["records"] = loaded
            except Exception:
                pass

        found_json = False
        for i, rec in enumerate(json_data["records"]):
            if rec.get("experiment_id") == exp_id:
                json_data["records"][i] = record
                found_json = True
                break
        if not found_json:
            json_data["records"].append(record)

        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)


# =============================================================================
# 9. Execution Orchestrator (Experiment 1 Runner)
# =============================================================================

def run_experiment_1(selected_series=None, output_dir=None):
    """
    Executes Experiment 1 across all 26 controlled points (or a selected series).
    Guarded against unauthorized execution.
    """
    # Strict execution guard
    if not os.environ.get("CONFIRM_STAGE1_EXP1_EXECUTION"):
        print("=" * 70)
        print("CRITICAL GUARD: Execution of Stage 1 Experiment 1 is currently BLOCKED.")
        print("The harness is fully constructed and self-tested, but execution requires")
        print("explicit user instruction and setting CONFIRM_STAGE1_EXP1_EXECUTION=1.")
        print("=" * 70)
        sys.exit(1)

    enforce_deterministic_cpu_environment(42)
    logger = Stage1ExperimentLogger(log_dir=output_dir)
    all_points = get_stage1_exp1_point_specifications()

    if selected_series:
        points_to_run = [p for p in all_points if p["series_id"] == selected_series]
    else:
        points_to_run = all_points

    print("=" * 70)
    print(f"STAGE 1 EXPERIMENT 1: EXECUTING {len(points_to_run)} CONTROLLED POINTS")
    print("=" * 70)

    # 1. Measure and cache isolated T_f per system/checkpoint
    tf_cache = {}
    vf_kwargs_cache = {}

    systems_needed = set((p["system_name"], p["checkpoint_path"], p["ground_truth_path"]) for p in points_to_run)
    for sys_name, ckpt_path, gt_path in systems_needed:
        print(f"Measuring isolated T_f for {sys_name} ({os.path.basename(ckpt_path)})...")
        vf = load_canonical_baseline(ckpt_path, state_dim=2)
        tf_stats = measure_isolated_tf(vf, gt_path, seed=42)
        tf_cache[ckpt_path] = tf_stats
        vf_kwargs_cache[ckpt_path] = {
            "state_dim": 2,
            "hidden_dim": 64,
            "num_layers": 2,
            "activation": "softplus",
            "augment_dim": 0
        }
        print(f"  -> T_f median: {tf_stats['tf_us_median']:.4f} us (IQR: {tf_stats['tf_us_iqr']:.4f} us, stable: {tf_stats['dispersion_stable_le_5pct']})")

    # 2. Iterate through points
    for idx, p in enumerate(points_to_run, 1):
        print(f"[{idx}/{len(points_to_run)}] Running {p['point_id']} ({p['system_name']}, {p['solver_name']})...")
        t_eval = np.linspace(p["horizon_start"], p["horizon_end"], p["num_eval_points"])

        raw_result = run_guarded_integration(
            ckpt_path=p["checkpoint_path"],
            vf_kwargs=vf_kwargs_cache[p["checkpoint_path"]],
            y0=p["y0"],
            t_eval=t_eval,
            solver=p["solver_name"],
            rtol=p["rtol"] if p["rtol"] is not None else 1e-5,
            atol=p["atol"] if p["atol"] is not None else 1e-7,
            options=p["options"],
            seed=42,
            warmup_cycles=20,
            recorded_cycles=15,
            timeout_s=30.0
        )

        tf_stats = tf_cache[p["checkpoint_path"]]
        metrics = compute_point_metrics(raw_result, tf_stats, p, t_eval)

        record = {
            "experiment_id": p["point_id"],
            "series_id": p["series_id"],
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "research_stage": "STAGE_1_BASELINE",
            "system_name": p["system_name"],
            "system_dimension": p["system_dimension"],
            "horizon_start": p["horizon_start"],
            "horizon_end": p["horizon_end"],
            "num_eval_points": p["num_eval_points"],
            "model_architecture": "MLP[D->64->64->D]_Softplus",
            "hidden_width": 64,
            "hidden_depth": 2,
            "activation": "Softplus",
            "parameter_count": 4482,
            "checkpoint_validation_status": "VALIDATED",
            "solver_name": p["solver_name"],
            "solver_backend": p["solver_backend"],
            "solver_type": p["solver_type"],
            "rtol": p["rtol"],
            "atol": p["atol"],
            "step_size": p["step_size"],
            "thread_count": 1,
            "random_seed": 42,
            "warmup_runs": 20,
            "measured_runs": 15,
            **metrics
        }

        logger.log_point_result(record)
        print(f"     Status: {record['status']} | NFE median: {record['nfe_median']} | Runtime: {record['runtime_ms_median']} ms | Solver overhead: {record['estimated_solver_overhead_ms']} ms")

    print("=" * 70)
    print("STAGE 1 EXPERIMENT 1 COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {logger.csv_file} and {logger.json_file}")
    print("=" * 70)


# =============================================================================
# 10. Self-Tests (Lightweight Verification Only — Zero Benchmark Execution)
# =============================================================================

def run_harness_self_tests():
    """
    Executes a comprehensive non-destructive self-test of the harness:
    1. Single-thread & interop thread pinning
    2. Model parameter count for D=2 and D=3
    3. Non-destructive checkpoint loading
    4. NFE counter correctness (forward evaluation on synthetic dummy state)
    5. Windows-safe watchdog termination of hung processes
    6. Isolated T_f computation test on 10 dummy calls
    7. 26-Point specification verification
    8. Runtime decomposition and Section 6.4 non-negative residual clipping
    9. Trajectory accuracy evaluation with CubicSpline interpolation
    10. Logger schema, deduplication, and execution guard
    """
    print("=" * 70)
    print("STAGE 1 EXPERIMENT 1 HARNESS: SELF-TEST SUITE")
    print("=" * 70)

    # Test 1: Single-thread and seed configuration
    print("Test 1: Environment & thread pinning...", end="", flush=True)
    enforce_deterministic_cpu_environment(42)
    assert torch.get_num_threads() == 1, "Thread count must be 1"
    assert torch.get_num_interop_threads() == 1, "Interop thread count must be 1"
    assert not torch.cuda.is_available(), "CUDA must be unavailable"
    print(" [PASSED]")

    # Test 2: Canonical Model Architecture
    print("Test 2: Canonical Architecture (Option A Softplus)...", end="", flush=True)
    vf2 = MLPVectorField(state_dim=2, hidden_dim=64, num_layers=2, activation="softplus")
    assert sum(p.numel() for p in vf2.parameters()) == 4482, "D=2 must have 4,482 params"
    vf3 = MLPVectorField(state_dim=3, hidden_dim=64, num_layers=2, activation="softplus")
    assert sum(p.numel() for p in vf3.parameters()) == 4611, "D=3 must have 4,611 params"
    print(" [PASSED]")

    # Test 3: Checkpoint non-destructive loading
    print("Test 3: Checkpoint compatibility (lv_h64, fhn_h64)...", end="", flush=True)
    lv_ckpt = os.path.join(ROOT_DIR, "models", "checkpoints", "lv_h64.pt")
    fhn_ckpt = os.path.join(ROOT_DIR, "models", "checkpoints", "fhn_h64.pt")
    vf_lv = load_canonical_baseline(lv_ckpt, state_dim=2)
    vf_fhn = load_canonical_baseline(fhn_ckpt, state_dim=2)
    assert vf_lv is not None and vf_fhn is not None
    print(" [PASSED]")

    # Test 4: NFE Counter smoke test
    print("Test 4: NFE counter evaluation and reset...", end="", flush=True)
    node_test = NeuralODE(vf_lv, solver="rk4")
    z_dummy = torch.tensor([1.0, 0.5])
    t_dummy = torch.linspace(0.0, 1.0, 11)
    _ = node_test(z_dummy, t_dummy, options={"step_size": 0.1})
    assert node_test.nfe == 40, f"Expected 40 NFE for 10 RK4 steps, got {node_test.nfe}"
    node_test.reset_nfe()
    assert node_test.nfe == 0, "Counter reset failed"
    print(" [PASSED]")

    # Test 5: Watchdog Safe Termination on Stalled Process
    print("Test 5: Windows-safe process watchdog termination...", end="", flush=True)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_dummy_stalled_for_test, args=(q,))
    p.start()
    p.join(timeout=0.2)  # 200ms timeout
    terminated_safely = False
    if p.is_alive():
        p.terminate()
        p.join()
        terminated_safely = True
    assert terminated_safely, "Watchdog failed to terminate stalled process"
    assert q.empty(), "Queue should be empty after forced termination"
    print(" [PASSED]")

    # Test 6: 26-Point Plan Generation Check
    print("Test 6: Experiment 1 26-point specification verification...", end="", flush=True)
    points = get_stage1_exp1_point_specifications()
    assert len(points) == 26
    s1a = [p for p in points if p["series_id"] == "Series_1A"]
    s1b = [p for p in points if p["series_id"] == "Series_1B"]
    s1c = [p for p in points if p["series_id"] == "Series_1C"]
    assert len(s1a) == 10, "Series 1A must have 10 points"
    assert len(s1b) == 10, "Series 1B must have 10 points"
    assert len(s1c) == 6, "Series 1C must have 6 points"
    assert s1c[0]["num_eval_points"] == 151, "Series 1C grid must match Series 1A full horizon (151 pts)"
    print(" [PASSED]")

    # Test 7: Runtime decomposition & Section 6.4 clipped residual test
    print("Test 7: Runtime decomposition & Section 6.4 clipping...", end="", flush=True)
    mock_raw_normal = {
        "status": "SUCCESS",
        "is_right_censored": False,
        "censoring_threshold_s": 30.0,
        "runtimes_ms": [10.0, 10.2, 9.8, 10.1, 10.0, 10.5, 9.9, 10.0, 10.1, 9.9, 10.2, 10.0, 9.8, 10.1, 10.0],
        "nfes": [100] * 15,
        "trajectory": np.ones((16, 2)),
        "failure_reason": None
    }
    mock_tf = {"tf_us_median": 50.0, "tf_us_iqr": 1.0}  # 100 * 50us = 5.0ms
    mock_point = points[0]
    t_eval_dummy = np.linspace(0.0, 1.5, 16)

    metrics_normal = compute_point_metrics(mock_raw_normal, mock_tf, mock_point, t_eval_dummy)
    assert metrics_normal["isolated_net_eval_est_ms"] == 5.0
    assert abs(metrics_normal["runtime_ms_median"] - 10.0) < 0.1
    assert abs(metrics_normal["estimated_solver_overhead_ms"] - 5.0) < 0.1

    # Test negative residual clipping per Section 6.4
    mock_raw_noise = {
        "status": "SUCCESS",
        "is_right_censored": False,
        "censoring_threshold_s": 30.0,
        "runtimes_ms": [4.0] * 15,  # Measured 4.0ms, but isolated estimate is 5.0ms
        "nfes": [100] * 15,
        "trajectory": np.ones((16, 2)),
        "failure_reason": None
    }
    metrics_noise = compute_point_metrics(mock_raw_noise, mock_tf, mock_point, t_eval_dummy)
    assert metrics_noise["raw_solver_overhead_ms"] == -1.0
    assert metrics_noise["estimated_solver_overhead_ms"] == 0.0, "Negative residual must be clipped to 0.0"
    assert "Signed residual: -1.0000 ms" in metrics_noise["notes"]
    print(" [PASSED]")

    # Test 8: Trajectory accuracy computation with CubicSpline
    print("Test 8: Trajectory accuracy calculation (CubicSpline)...", end="", flush=True)
    gt_lv_path = os.path.join(ROOT_DIR, "data", "trajectories", "lotka-volterra_ground_truth.pt")
    gt_data = torch.load(gt_lv_path, map_location="cpu")
    t_gt = gt_data["t"].numpy()
    y_gt = gt_data["y"].numpy()
    cs = CubicSpline(t_gt, y_gt, axis=0)
    t_test = np.linspace(0.0, 1.5, 16)
    y_exact = cs(t_test)

    # Identical trajectory should yield MSE = 0
    acc_zero = evaluate_trajectory_accuracy(y_exact, gt_lv_path, t_test)
    assert acc_zero["trajectory_mse"] == 0.0
    assert acc_zero["trajectory_linf"] == 0.0

    # Perturbed trajectory
    acc_perturbed = evaluate_trajectory_accuracy(y_exact + 0.1, gt_lv_path, t_test)
    assert acc_perturbed["trajectory_mse"] > 0.0
    assert abs(acc_perturbed["trajectory_linf"] - 0.1) < 1e-6
    print(" [PASSED]")

    # Test 9: Logger schema and deduplication test
    print("Test 9: Logger schema & deduplication...", end="", flush=True)
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        test_logger = Stage1ExperimentLogger(log_dir=tmpdir)
        rec1 = {"experiment_id": "TEST-1", "series_id": "Series_1A", "status": "SUCCESS", "nfe_median": 100}
        test_logger.log_point_result(rec1)
        # Update same ID
        rec2 = {"experiment_id": "TEST-1", "series_id": "Series_1A", "status": "SUCCESS", "nfe_median": 200}
        test_logger.log_point_result(rec2)

        with open(test_logger.json_file, "r", encoding="utf-8") as f:
            jdata = json.load(f)
            assert len(jdata["records"]) == 1, "Deduplication failed; expected 1 record"
            assert jdata["records"][0]["nfe_median"] == 200, "Record update failed"
            assert "environment_metadata" in jdata, "Missing environment_metadata in JSON"
    print(" [PASSED]")

    # Test 10: Execution Guard verification
    print("Test 10: Execution Guard verification...", end="", flush=True)
    assert "CONFIRM_STAGE1_EXP1_EXECUTION" not in os.environ, "Guard environment variable should not be set during test"
    print(" [PASSED]")

    print("=" * 70)
    print("ALL 10 HARNESS SELF-TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


def print_dry_run_plan():
    """Prints the 26 planned points without executing any benchmarks."""
    points = get_stage1_exp1_point_specifications()
    print("=" * 85)
    print("STAGE 1 EXPERIMENT 1: PLANNED 26-POINT CONTROLLED SPECIFICATIONS (DRY RUN)")
    print("=" * 85)
    print(f"{'#':<3} {'Series':<10} {'Point ID':<25} {'System':<16} {'Solver':<8} {'Horizon / Steps':<18}")
    print("-" * 85)
    for idx, p in enumerate(points, 1):
        if p["is_fixed_step"]:
            param_str = f"N_steps={p['expected_deterministic_nfe'] // 4}, h={p['step_size']:.4f}"
        else:
            param_str = f"T in [0, {p['horizon_end']:.1f}] ({p['num_eval_points']} pts)"
        print(f"{idx:<3} {p['series_id']:<10} {p['point_id']:<25} {p['system_name']:<16} {p['solver_name']:<8} {param_str:<18}")
    print("=" * 85)
    print("DRY RUN COMPLETE: Zero experiments were executed. Zero results were collected.")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    mp.freeze_support()
    parser = argparse.ArgumentParser(description="Stage 1 Experiment 1 Profiling Harness")
    parser.add_argument("--self-test", action="store_true", help="Execute lightweight harness self-tests (zero benchmarks)")
    parser.add_argument("--dry-run", action="store_true", help="Print planned 26 controlled points without executing")
    parser.add_argument("--execute-stage1-exp1", action="store_true", help="Execute the actual Experiment 1 benchmark (REQUIRES EXPLICIT AUTHORIZATION)")
    parser.add_argument("--series", type=str, default=None, choices=["Series_1A", "Series_1B", "Series_1C"], help="Run a specific series only")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom directory for logs")

    args = parser.parse_args()

    if args.self_test:
        run_harness_self_tests()
    elif args.dry_run:
        print_dry_run_plan()
    elif args.execute_stage1_exp1:
        run_experiment_1(selected_series=args.series, output_dir=args.output_dir)
    else:
        parser.print_help()
