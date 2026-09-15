"""
Phase 1: Diagnostic Spectral Characterization of Candidate Benchmark Systems.
Computes analytical Jacobian eigenvalues along verified ground-truth trajectories
for Lotka-Volterra and FitzHugh-Nagumo.

Design Principles:
- Root-relative path resolution via ROOT_DIR (independent of current working directory).
- Explicit trajectory integrity assertions (shape, finiteness, monotonicity, dimension).
- Evaluates analytical Jacobians from benchmarks/systems.py along verified trajectories.
- Reports empirical spectral envelopes: real parts, imaginary parts, spectral radius.
- Measures strongest negative real component as a dissipative-mode diagnostic.
- Measures fraction of trajectory exhibiting complex eigenvalues as a descriptive diagnostic.
- Measures fastest local spectral timescale tau_spectral = 1 / max |lambda|.
- REMOVED: Generic explicit stability assertions (stability regions depend on solver tableau).
- REMOVED: Predetermined stiffness classifications or arbitrary ratio thresholds.
- Purely diagnostic reporting; wall-clock timing is for diagnostic traceability, not benchmark timing.
- Zero neural network training, zero ODE solvers invoked, zero benchmark timing.
"""

import os
import sys
import time
import json
import torch
import numpy as np

# Root-relative path resolution
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from benchmarks.systems import LotkaVolterra, FitzHughNagumo

# Descriptive threshold for imaginary part presence
COMPLEX_IMAG_THRESHOLD = 1e-7


def verify_and_load_trajectory(trajectory_path):
    """
    Loads and validates ground-truth trajectory with strict integrity checks.
    
    Parameters
    ----------
    trajectory_path : str
        Absolute root-relative path to .pt file containing 't' and 'y'.
        
    Returns
    -------
    t_vals : np.ndarray (1D)
    y_vals : np.ndarray (2D, shape: [N, 2])
    """
    assert os.path.exists(trajectory_path), f"Trajectory file not found: {trajectory_path}"
    
    data = torch.load(trajectory_path, map_location="cpu")
    assert "t" in data and "y" in data, f"Missing 't' or 'y' in dataset: {trajectory_path}"
    
    t_vals = data["t"].detach().cpu().numpy().astype(np.float64)
    y_vals = data["y"].detach().cpu().numpy().astype(np.float64)

    # Lightweight trajectory integrity assertions
    assert len(t_vals) == len(y_vals), (
        f"Length mismatch: len(t)={len(t_vals)} vs len(y)={len(y_vals)}"
    )
    assert y_vals.ndim == 2, f"Expected 2D trajectory array, got {y_vals.ndim}D"
    assert y_vals.shape[1] == 2, f"Expected state dimension 2, got {y_vals.shape[1]}"
    assert np.all(np.isfinite(t_vals)), "t_vals contains non-finite (NaN or Inf) values"
    assert np.all(np.isfinite(y_vals)), "y_vals contains non-finite (NaN or Inf) values"
    assert np.all(np.diff(t_vals) > 0), "t_vals is not strictly monotonically increasing"

    return t_vals, y_vals


def analyze_system_spectrum(system_name, system, trajectory_path):
    """
    Computes spectral statistics of the analytical Jacobian along a verified trajectory.
    
    Parameters
    ----------
    system_name : str
        Explicit name of the system (does not rely on unverified instance attributes).
    system : BenchmarkODE
        Dynamical system instance providing jacobian(t, y).
    trajectory_path : str
        Path to verified .pt ground-truth file.
        
    Returns
    -------
    dict
        Measured spectral statistics across all trajectory sample points.
    """
    t_vals, y_vals = verify_and_load_trajectory(trajectory_path)
    num_points = len(t_vals)
    state_dim = y_vals.shape[1]

    all_eigenvalues = []
    has_complex = []
    spectral_radii = []

    for k in range(num_points):
        t_k = float(t_vals[k])
        y_k = y_vals[k]
        
        # Analytical Jacobian evaluation
        J_k = system.jacobian(t_k, y_k)
        assert J_k.shape == (state_dim, state_dim), f"Unexpected Jacobian shape: {J_k.shape}"
        assert np.all(np.isfinite(J_k)), f"Jacobian contains non-finite values at t={t_k}"
        
        # Eigenvalue decomposition
        eigvals = np.linalg.eigvals(J_k)
        all_eigenvalues.append(eigvals)
        spectral_radii.append(np.max(np.abs(eigvals)))
        
        # Check for non-negligible imaginary components (oscillatory mode indicator)
        has_complex.append(bool(np.any(np.abs(np.imag(eigvals)) > COMPLEX_IMAG_THRESHOLD)))

    all_eigvals_flat = np.concatenate(all_eigenvalues)
    real_parts = np.real(all_eigvals_flat)
    imag_parts = np.imag(all_eigvals_flat)

    # Spectral measurements
    strongest_negative_real = float(np.min(real_parts))
    max_positive_real = float(np.max(real_parts))
    mean_real = float(np.mean(real_parts))
    
    max_abs_imag = float(np.max(np.abs(imag_parts)))
    mean_abs_imag = float(np.mean(np.abs(imag_parts)))
    
    max_spectral_radius = float(np.max(spectral_radii))
    min_spectral_radius = float(np.min(spectral_radii))
    mean_spectral_radius = float(np.mean(spectral_radii))
    
    fraction_complex = float(np.mean(has_complex))
    fastest_timescale = float(1.0 / max_spectral_radius) if max_spectral_radius > 0 else float("inf")

    return {
        "system_name": system_name,
        "trajectory_file": os.path.basename(trajectory_path),
        "trajectory_points": num_points,
        "t_span": [float(t_vals[0]), float(t_vals[-1])],
        "state_dimension": state_dim,
        "spectral_diagnostics": {
            "real_envelope": [float(np.min(real_parts)), float(np.max(real_parts))],
            "strongest_negative_real_mode": strongest_negative_real,
            "max_positive_real_mode": max_positive_real,
            "mean_real_mode": mean_real,
            "imag_envelope": [float(np.min(imag_parts)), float(np.max(imag_parts))],
            "max_abs_imag": max_abs_imag,
            "mean_abs_imag": mean_abs_imag,
            "fraction_complex_points": fraction_complex,
            "spectral_radius_range": [min_spectral_radius, max_spectral_radius],
            "mean_spectral_radius": mean_spectral_radius,
            "fastest_spectral_timescale_tau": fastest_timescale
        }
    }


def main():
    print("=" * 74)
    print("Phase 1: Diagnostic Spectral Characterization of Candidate Systems")
    print("=" * 74)
    print("Note: Wall-clock timing is for diagnostic traceability, not benchmark timing.")

    start_wall_time = time.perf_counter()

    systems_to_evaluate = [
        (
            "Lotka-Volterra",
            LotkaVolterra(alpha=1.5, beta=1.0, gamma=3.0, delta=1.0),
            os.path.join(ROOT_DIR, "data", "trajectories", "lotka-volterra_ground_truth.pt")
        ),
        (
            "FitzHugh-Nagumo",
            FitzHughNagumo(a=0.7, b=0.8, tau=12.5, I=0.5),
            os.path.join(ROOT_DIR, "data", "trajectories", "fitzhugh-nagumo_ground_truth.pt")
        )
    ]

    all_results = {}

    for sys_name, system, traj_path in systems_to_evaluate:
        print(f"\nEvaluating: {sys_name}")
        print(f"  Trajectory source: {traj_path}")
        
        diag = analyze_system_spectrum(sys_name, system, traj_path)
        all_results[sys_name] = diag
        
        sd = diag["spectral_diagnostics"]
        print(f"  Points analyzed: {diag['trajectory_points']} over t in {diag['t_span']}")
        print(f"  Real part range [min, max]:       [{sd['real_envelope'][0]:.4f}, {sd['real_envelope'][1]:.4f}]")
        print(f"  Strongest negative real mode:     {sd['strongest_negative_real_mode']:.4f}")
        print(f"  Max positive real mode:           {sd['max_positive_real_mode']:.4f}")
        print(f"  Imaginary part range [min, max]:  [{sd['imag_envelope'][0]:.4f}, {sd['imag_envelope'][1]:.4f}]")
        print(f"  Fraction of complex timepoints:   {sd['fraction_complex_points'] * 100:.1f}%")
        print(f"  Spectral radius range [min, max]: [{sd['spectral_radius_range'][0]:.4f}, {sd['spectral_radius_range'][1]:.4f}]")
        print(f"  Fastest spectral timescale tau:   {sd['fastest_spectral_timescale_tau']:.4f}")

    total_wall_time = time.perf_counter() - start_wall_time

    # Output directory and file resolution relative to ROOT_DIR
    out_dir = os.path.join(ROOT_DIR, "experiments", "logs")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "phase1_spectral_diagnostics.json")

    summary_payload = {
        "metadata": {
            "description": "Phase 1 Diagnostic Spectral Characterization",
            "diagnostic_traceability_wall_clock_seconds": round(total_wall_time, 4),
            "timing_purpose": "diagnostic traceability, not benchmark timing",
            "complex_imaginary_threshold": COMPLEX_IMAG_THRESHOLD,
            "device": "CPU"
        },
        "results": all_results
    }

    with open(out_file, "w") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"\n[OK] Diagnostic spectral report saved to: {out_file}")
    print(f"[OK] Diagnostic traceability elapsed time: {total_wall_time:.4f} seconds")
    print("=" * 74)


if __name__ == "__main__":
    main()
