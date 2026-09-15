"""
Phase 2 Stage A: Surrogate Training and 4-Tier Validation Pipeline.

Integrates a verified process-based watchdog for all trajectory validations.
- Preserves all 12 existing checkpoints without retraining.
- Dual safety guards: solver max_num_steps=1000 + process watchdog timeout_sec=5.0s.
- Evaluates Scaled Robertson; logs genuine outcome (CONVERGED, STEP_LIMIT_EXCEEDED, VALIDATION_TIMEOUT).
- If Robertson fails, trains and evaluates a fresh dedicated VdP mu=100 surrogate.
- Never reuses the failed vdp_mu100_h64 checkpoint as a valid benchmark.
- If fresh VdP mu=100 also fails, reports second stiff Neural ODE benchmark as NOT ADMITTED.
"""

import os
import sys
import time
import json
import multiprocessing as mp
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import qmc
from scipy.integrate import solve_ivp
from torchdiffeq import odeint

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from benchmarks.systems import LotkaVolterra, FitzHughNagumo, VanDerPol, Robertson
from models.vector_fields import MLPVectorField

CHECKPOINT_DIR = os.path.join(ROOT_DIR, "models", "checkpoints")
LOG_DIR = os.path.join(ROOT_DIR, "experiments", "logs")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# =========================================================================
# Multi-process Watchdog Worker Functions
# =========================================================================
def _integration_worker(vf_state, vf_kwargs, y0_t, t_eval_t, rtol, atol, max_num_steps, result_queue):
    """Executes trajectory integration in an isolated child process."""
    try:
        from models.vector_fields import MLPVectorField
        from torchdiffeq import odeint
        import torch
        import numpy as np
        
        vf = MLPVectorField(**vf_kwargs)
        vf.load_state_dict(vf_state)
        vf.eval()
        
        options = {"max_num_steps": max_num_steps}
        with torch.no_grad():
            sol = odeint(
                vf,
                y0_t,
                t_eval_t,
                method="dopri5",
                rtol=rtol,
                atol=atol,
                options=options
            )
            sol_np = sol.cpu().numpy()
            
        result_queue.put(("SUCCESS", sol_np, "CONVERGED"))
    except AssertionError as e:
        if "max_num_steps exceeded" in str(e):
            result_queue.put(("FAILED", None, "STEP_LIMIT_EXCEEDED"))
        else:
            result_queue.put(("FAILED", None, f"ASSERTION_ERROR: {e}"))
    except Exception as e:
        result_queue.put(("FAILED", None, f"INTEGRATION_ERROR: {e}"))


def run_integration_with_process_watchdog(model, y0, t_eval, rtol=1e-5, atol=1e-7, max_num_steps=1000, timeout_sec=5.0):
    """Executes ODE integration with both an internal step limit and hard process termination."""
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    vf_state = {k: v.cpu() for k, v in model.state_dict().items()}
    linear_layers = [m for m in model.net if isinstance(m, torch.nn.Linear)]
    vf_kwargs = {
        "state_dim": model.state_dim,
        "hidden_dim": getattr(model, "hidden_dim", linear_layers[0].out_features),
        "num_layers": getattr(model, "num_layers", len(linear_layers) - 1),
        "activation": getattr(model, "activation", "softplus")
    }

    start_time = time.perf_counter()
    p = ctx.Process(
        target=_integration_worker,
        args=(vf_state, vf_kwargs, y0.cpu(), t_eval.cpu(), rtol, atol, max_num_steps, result_queue)
    )
    p.start()
    p.join(timeout=timeout_sec)

    is_timeout = False
    if p.is_alive():
        is_timeout = True
        p.terminate()  # Forcibly terminate child process
        p.join()

    elapsed = time.perf_counter() - start_time

    if is_timeout:
        return False, None, "VALIDATION_TIMEOUT", round(elapsed, 3)

    if not result_queue.empty():
        status, sol, reason = result_queue.get()
        return (status == "SUCCESS"), sol, reason, round(elapsed, 3)
    else:
        return False, None, "PROCESS_TERMINATED_ABRUPTLY", round(elapsed, 3)


# =========================================================================
# Collocation Dataset Generation & Training
# =========================================================================
def generate_collocation_dataset(system, t_span, y0, num_traj_pts=200, num_lhs=350, seed=42):
    """Generates collocation dataset combining reference trajectory with Latin Hypercube Samples."""
    np.random.seed(seed)
    
    sol = solve_ivp(
        system.rhs,
        t_span,
        y0,
        method="Radau" if getattr(system, "stiffness_regime", "") in ["stiff", "highly-stiff"] else "RK45",
        rtol=1e-10,
        atol=1e-12,
        dense_output=True
    )
    t_traj = np.linspace(t_span[0], t_span[1], num_traj_pts)
    y_traj = sol.sol(t_traj).T
    
    dim = y_traj.shape[1]
    y_min = np.min(y_traj, axis=0)
    y_max = np.max(y_traj, axis=0)
    span = np.maximum(y_max - y_min, 1e-3)
    delta = 0.25 * span
    l_bounds = y_min - delta
    u_bounds = y_max + delta
    
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    y_lhs = qmc.scale(sampler.random(n=num_lhs), l_bounds, u_bounds)
    
    y_all = np.vstack([y_traj, y_lhs])
    dy_all = np.array([system.rhs(0.0, y_pt) for y_pt in y_all], dtype=np.float64)
    
    num_total = len(y_all)
    indices = np.random.permutation(num_total)
    split = int(0.8 * num_total)
    
    train_idx, test_idx = indices[:split], indices[split:]
    
    return {
        "train_y": torch.tensor(y_all[train_idx], dtype=torch.float32),
        "train_dy": torch.tensor(dy_all[train_idx], dtype=torch.float32),
        "test_y": torch.tensor(y_all[test_idx], dtype=torch.float32),
        "test_dy": torch.tensor(dy_all[test_idx], dtype=torch.float32),
        "ref_t": torch.tensor(t_traj, dtype=torch.float32),
        "ref_y": torch.tensor(y_traj, dtype=torch.float32),
        "y0": torch.tensor(y0, dtype=torch.float32),
        "t_span": t_span
    }


def train_surrogate(model, data, max_epochs=600, lr=1e-3, batch_size=64, timeout_sec=20.0, seed=42):
    """Trains autonomous neural vector field via supervised derivative fitting."""
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
    criterion = nn.MSELoss()
    
    train_y = data["train_y"]
    train_dy = data["train_dy"]
    num_samples = len(train_y)
    
    start_time = time.perf_counter()
    best_loss = float("inf")
    best_state = None
    epochs_run = 0
    
    for epoch in range(1, max_epochs + 1):
        if (time.perf_counter() - start_time) > timeout_sec:
            break
        
        model.train()
        perm = torch.randperm(num_samples)
        epoch_loss = 0.0
        num_batches = 0
        
        for i in range(0, num_samples, batch_size):
            batch_idx = perm[i : i + batch_size]
            y_b = train_y[batch_idx]
            dy_b = train_dy[batch_idx]
            
            optimizer.zero_grad()
            pred_dy = model(y_b)
            loss = criterion(pred_dy, dy_b)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
        scheduler.step()
        epochs_run = epoch
        avg_loss = epoch_loss / num_batches
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    wall_ms = (time.perf_counter() - start_time) * 1000.0
    
    if best_state is not None:
        model.load_state_dict(best_state)
        
    return {
        "training_epochs": epochs_run,
        "training_wall_ms": round(wall_ms, 2),
        "best_val_loss": float(best_loss)
    }


def validate_surrogate(model, data, system_name, mu=None):
    """Performs 4-Tier Validation Screening with the process-based watchdog."""
    model.eval()
    
    # Tier 1: Derivative R2
    with torch.no_grad():
        pred_dy = model(data["test_y"]).detach().numpy()
    true_dy = data["test_dy"].numpy()
    
    ss_res = np.sum((true_dy - pred_dy) ** 2)
    ss_tot = np.sum((true_dy - np.mean(true_dy, axis=0)) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-12)))
    tier1_pass = bool(r2 >= 0.95)
    
    per_comp_r2 = []
    for c in range(true_dy.shape[1]):
        c_res = np.sum((true_dy[:, c] - pred_dy[:, c]) ** 2)
        c_tot = np.sum((true_dy[:, c] - np.mean(true_dy[:, c])) ** 2)
        c_r2 = 1.0 - (c_res / (c_tot + 1e-12))
        per_comp_r2.append(float(c_r2))

    # Tier 2: Short-Horizon Trajectory NRMSE under Process Watchdog
    ref_t = data["ref_t"]
    ref_y = data["ref_y"].numpy()
    num_short = max(5, int(0.25 * len(ref_t)))
    t_short = ref_t[:num_short]
    ref_y_short = ref_y[:num_short]
    
    success_short, pred_y_short, reason_short, elapsed_short = run_integration_with_process_watchdog(
        model, data["y0"], t_short, rtol=1e-5, atol=1e-7, max_num_steps=1000, timeout_sec=5.0
    )
    
    tier2_pass = False
    nrmse = float("inf")
    if success_short and pred_y_short is not None and np.all(np.isfinite(pred_y_short)):
        mse = np.mean((pred_y_short - ref_y_short) ** 2)
        y_range = np.maximum(np.max(ref_y_short) - np.min(ref_y_short), 1e-3)
        nrmse = float(np.sqrt(mse) / y_range)
        tier2_pass = bool(nrmse <= 0.05)
        tier2_outcome = "CONVERGED" if tier2_pass else "NRMSE_THRESHOLD_EXCEEDED"
    else:
        tier2_outcome = reason_short

    # Tier 3: System-Appropriate Dynamic Invariants
    tier3_pass = True
    invariant_details = {}
    if "VanDerPol" in system_name:
        if tier2_pass:
            max_amp = float(np.max(np.abs(pred_y_short[:, 0])))
            invariant_details["max_amplitude"] = max_amp
            tier3_pass = bool(max_amp <= 4.0)
        else:
            tier3_pass = False
    elif "Lotka-Volterra" in system_name:
        if tier2_pass:
            min_pop = float(np.min(pred_y_short))
            invariant_details["min_population"] = min_pop
            tier3_pass = bool(min_pop > -0.05)
        else:
            tier3_pass = False
    elif "FitzHugh-Nagumo" in system_name:
        if tier2_pass:
            max_val = float(np.max(np.abs(pred_y_short)))
            invariant_details["max_state_val"] = max_val
            tier3_pass = bool(max_val <= 3.5)
        else:
            tier3_pass = False
    elif "Robertson" in system_name:
        if tier2_pass:
            min_val = float(np.min(pred_y_short))
            mass = np.sum(pred_y_short, axis=1)
            mass_err = float(np.max(np.abs(mass - 1.0)))
            invariant_details["min_concentration"] = min_val
            invariant_details["max_mass_conservation_error"] = mass_err
            tier3_pass = bool(min_val >= -0.05 and mass_err <= 0.10)
        else:
            tier3_pass = False

    # Tier 4: Full-Horizon Stability under Process Watchdog
    success_full, pred_y_full, reason_full, elapsed_full = run_integration_with_process_watchdog(
        model, data["y0"], ref_t, rtol=1e-5, atol=1e-7, max_num_steps=1000, timeout_sec=5.0
    )
    tier4_pass = bool(success_full and pred_y_full is not None and np.all(np.isfinite(pred_y_full)) and np.max(np.abs(pred_y_full)) < 1e4)
    tier4_outcome = "CONVERGED" if tier4_pass else reason_full

    # Pre-Benchmark Jacobian Spectral Diagnostics: rho_max
    rho_max = 0.0
    try:
        sample_y = data["ref_y"][:min(30, len(ref_t))]
        radii = []
        for pt in sample_y:
            pt_t = pt.clone().detach().requires_grad_(True)
            J = torch.autograd.functional.jacobian(lambda z: model(z), pt_t)
            eigs = np.linalg.eigvals(J.detach().cpu().numpy())
            radii.append(float(np.max(np.abs(eigs))))
        rho_max = float(np.max(radii))
    except Exception:
        rho_max = float("nan")

    all_passed = bool(tier1_pass and tier2_pass and tier3_pass and tier4_pass)
    
    return {
        "tier1_derivative_r2": r2,
        "per_component_r2": per_comp_r2,
        "tier1_pass": tier1_pass,
        "tier2_nrmse": nrmse,
        "tier2_outcome": tier2_outcome,
        "tier2_elapsed_s": elapsed_short,
        "tier2_pass": tier2_pass,
        "tier3_invariants": invariant_details,
        "tier3_pass": tier3_pass,
        "tier4_outcome": tier4_outcome,
        "tier4_elapsed_s": elapsed_full,
        "tier4_pass": tier4_pass,
        "rho_max": rho_max,
        "all_tiers_passed": all_passed
    }


def main():
    print("=" * 78)
    print("Phase 2 Stage A: Surrogate Training & 4-Tier Validation (Watchdog Guarded)")
    print("=" * 78)
    print("Autonomous vector fields: f_theta(y) -> dy/dt")
    print("Guards: max_num_steps=1000, timeout_sec=5.0s (Process Watchdog)")
    print(f"Checkpoints directory: {CHECKPOINT_DIR}")
    print(f"Logs directory:        {LOG_DIR}")
    print("-" * 78)

    all_models_summary = {}

    # =========================================================================
    # 1. LAYER 1: Van der Pol mu=2.0 Capacity Sweep (H in {8, 32, 128, 256})
    # =========================================================================
    print("\n[Layer 1] Van der Pol mu=2.0 Capacity Sweep Surrogates...")
    vdp2_system = VanDerPol(mu=2.0)
    vdp2_data = generate_collocation_dataset(
        vdp2_system,
        t_span=(0.0, 5.0),
        y0=np.array([2.0, 0.0]),
        num_traj_pts=150,
        num_lhs=350,
        seed=42
    )
    
    layer1_widths = [8, 32, 128, 256]
    for H in layer1_widths:
        model_name = f"vdp_mu2_h{H}"
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_name}.pt")
        vf = MLPVectorField(state_dim=2, hidden_dim=H, num_layers=2, activation="softplus")
        
        if os.path.exists(ckpt_path):
            print(f"  [CACHED] Validating {model_name} (H={H})...", end="", flush=True)
            vf.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            train_stats = {"training_epochs": "cached", "training_wall_ms": "cached", "best_val_loss": "cached"}
        else:
            print(f"  Training {model_name} (H={H})...", end="", flush=True)
            train_stats = train_surrogate(vf, vdp2_data, max_epochs=600, lr=1e-3, timeout_sec=20.0, seed=42)
            torch.save(vf.state_dict(), ckpt_path)
            
        val_stats = validate_surrogate(vf, vdp2_data, system_name="VanDerPol", mu=2.0)
        status_str = "PASSED" if val_stats["all_tiers_passed"] else f"FAILED ({val_stats['tier2_outcome']})"
        print(f" [{status_str}] R2={val_stats['tier1_derivative_r2']:.4f}, NRMSE={val_stats['tier2_nrmse']:.4f}, rho_max={val_stats['rho_max']:.2f}")
        
        all_models_summary[model_name] = {
            "layer": "Layer1",
            "system_name": "VanDerPol",
            "mu": 2.0,
            "hidden_dim": H,
            "num_params": vf.num_params,
            "training_stats": train_stats,
            "validation_stats": val_stats,
            "checkpoint_file": f"{model_name}.pt"
        }

    # =========================================================================
    # 2. LAYER 2A: Van der Pol Controlled mu Sweep (mu in {1, 5, 10, 25, 50, 100})
    # =========================================================================
    print("\n[Layer 2A] Van der Pol Controlled mu Sweep Surrogates (H=64)...")
    mu_sweep = [1.0, 5.0, 10.0, 25.0, 50.0, 100.0]
    for mu in mu_sweep:
        model_name = f"vdp_mu{int(mu)}_h64"
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_name}.pt")
        
        vdp_sys = VanDerPol(mu=mu)
        vdp_data = generate_collocation_dataset(
            vdp_sys,
            t_span=(0.0, 20.0),
            y0=np.array([2.0, 0.0]),
            num_traj_pts=200,
            num_lhs=400,
            seed=42 + int(mu)
        )
        
        vf = MLPVectorField(state_dim=2, hidden_dim=64, num_layers=2, activation="softplus")
        if os.path.exists(ckpt_path):
            print(f"  [CACHED] Validating {model_name} (mu={mu})...", end="", flush=True)
            vf.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            train_stats = {"training_epochs": "cached", "training_wall_ms": "cached", "best_val_loss": "cached"}
        else:
            print(f"  Training {model_name} (mu={mu})...", end="", flush=True)
            train_stats = train_surrogate(vf, vdp_data, max_epochs=700, lr=1e-3, timeout_sec=20.0, seed=42)
            torch.save(vf.state_dict(), ckpt_path)
            
        val_stats = validate_surrogate(vf, vdp_data, system_name="VanDerPol", mu=mu)
        status_str = "PASSED" if val_stats["all_tiers_passed"] else f"FAILED ({val_stats['tier2_outcome']})"
        print(f" [{status_str}] R2={val_stats['tier1_derivative_r2']:.4f}, NRMSE={val_stats['tier2_nrmse']:.4f}, rho_max={val_stats['rho_max']:.2f}")
        
        all_models_summary[model_name] = {
            "layer": "Layer2A",
            "system_name": "VanDerPol",
            "mu": mu,
            "hidden_dim": 64,
            "num_params": vf.num_params,
            "training_stats": train_stats,
            "validation_stats": val_stats,
            "checkpoint_file": f"{model_name}.pt"
        }

    # =========================================================================
    # 3. LAYER 2B: Supervisor Portfolio Baseline Surrogates
    # =========================================================================
    print("\n[Layer 2B] Supervisor Portfolio Baseline Surrogates...")
    
    # Lotka-Volterra
    lv_name = "lv_h64"
    lv_ckpt = os.path.join(CHECKPOINT_DIR, f"{lv_name}.pt")
    lv_sys = LotkaVolterra()
    lv_data = generate_collocation_dataset(lv_sys, (0.0, 15.0), np.array([1.0, 0.5]), 150, 350, 101)
    lv_vf = MLPVectorField(state_dim=2, hidden_dim=64, num_layers=2, activation="softplus")
    if os.path.exists(lv_ckpt):
        print("  [CACHED] Validating lv_h64...", end="", flush=True)
        lv_vf.load_state_dict(torch.load(lv_ckpt, map_location="cpu"))
        lv_train = {"training_epochs": "cached", "training_wall_ms": "cached", "best_val_loss": "cached"}
    else:
        print("  Training lv_h64...", end="", flush=True)
        lv_train = train_surrogate(lv_vf, lv_data, 600, 1e-3, timeout_sec=20.0, seed=101)
        torch.save(lv_vf.state_dict(), lv_ckpt)
    lv_val = validate_surrogate(lv_vf, lv_data, "Lotka-Volterra")
    status_str = "PASSED" if lv_val["all_tiers_passed"] else f"FAILED ({lv_val['tier2_outcome']})"
    print(f" [{status_str}] R2={lv_val['tier1_derivative_r2']:.4f}, NRMSE={lv_val['tier2_nrmse']:.4f}, rho_max={lv_val['rho_max']:.2f}")
    all_models_summary[lv_name] = {
        "layer": "Layer2B",
        "system_name": "Lotka-Volterra",
        "system_role": "candidate-non-stiff-baseline",
        "hidden_dim": 64,
        "num_params": lv_vf.num_params,
        "training_stats": lv_train,
        "validation_stats": lv_val,
        "checkpoint_file": f"{lv_name}.pt"
    }

    # FitzHugh-Nagumo
    fhn_name = "fhn_h64"
    fhn_ckpt = os.path.join(CHECKPOINT_DIR, f"{fhn_name}.pt")
    fhn_sys = FitzHughNagumo()
    fhn_data = generate_collocation_dataset(fhn_sys, (0.0, 50.0), np.array([-1.0, 1.0]), 200, 400, 202)
    fhn_vf = MLPVectorField(state_dim=2, hidden_dim=64, num_layers=2, activation="softplus")
    if os.path.exists(fhn_ckpt):
        print("  [CACHED] Validating fhn_h64...", end="", flush=True)
        fhn_vf.load_state_dict(torch.load(fhn_ckpt, map_location="cpu"))
        fhn_train = {"training_epochs": "cached", "training_wall_ms": "cached", "best_val_loss": "cached"}
    else:
        print("  Training fhn_h64...", end="", flush=True)
        fhn_train = train_surrogate(fhn_vf, fhn_data, 600, 1e-3, timeout_sec=20.0, seed=202)
        torch.save(fhn_vf.state_dict(), fhn_ckpt)
    fhn_val = validate_surrogate(fhn_vf, fhn_data, "FitzHugh-Nagumo")
    status_str = "PASSED" if fhn_val["all_tiers_passed"] else fhn_val['tier2_outcome']
    print(f" [{status_str}] R2={fhn_val['tier1_derivative_r2']:.4f}, NRMSE={fhn_val['tier2_nrmse']:.4f}, rho_max={fhn_val['rho_max']:.2f}")
    all_models_summary[fhn_name] = {
        "layer": "Layer2B",
        "system_name": "FitzHugh-Nagumo",
        "system_role": "candidate-non-stiff-baseline",
        "hidden_dim": 64,
        "num_params": fhn_vf.num_params,
        "training_stats": fhn_train,
        "validation_stats": fhn_val,
        "checkpoint_file": f"{fhn_name}.pt"
    }

    # Scaled Robertson (Conditional Screening under Watchdog)
    print("  Training & Validating robertson_h64 (Scaled Robertson, H=64, 3D)...", end="", flush=True)
    rob_sys = Robertson()
    y_scale = np.array([1.0, 1e-4, 1.0], dtype=np.float64)
    np.random.seed(303)
    sol_rob = solve_ivp(
        rob_sys.rhs,
        (1e-5, 1e5),
        np.array([1.0, 0.0, 0.0]),
        method="Radau",
        rtol=1e-10,
        atol=1e-12
    )
    t_log = np.logspace(-5, 5, 200)
    from scipy.interpolate import interp1d
    interp = interp1d(sol_rob.t, sol_rob.y, axis=1, fill_value="extrapolate")
    y_rob_raw = interp(t_log).T
    y_rob_scaled = y_rob_raw / y_scale
    
    sampler = qmc.LatinHypercube(d=3, seed=303)
    y_lhs_scaled = sampler.random(n=300)
    y_all_scaled = np.vstack([y_rob_scaled, y_lhs_scaled])
    
    dy_all_scaled = []
    for y_s in y_all_scaled:
        y_unscaled = y_s * y_scale
        dy_unscaled = rob_sys.rhs(0.0, y_unscaled)
        dy_scaled = dy_unscaled / y_scale
        dy_all_scaled.append(dy_scaled)
    dy_all_scaled = np.array(dy_all_scaled, dtype=np.float64)
    
    indices = np.random.permutation(len(y_all_scaled))
    split = int(0.8 * len(y_all_scaled))
    
    rob_data = {
        "train_y": torch.tensor(y_all_scaled[indices[:split]], dtype=torch.float32),
        "train_dy": torch.tensor(dy_all_scaled[indices[:split]], dtype=torch.float32),
        "test_y": torch.tensor(y_all_scaled[indices[split:]], dtype=torch.float32),
        "test_dy": torch.tensor(dy_all_scaled[indices[split:]], dtype=torch.float32),
        "ref_t": torch.tensor(t_log, dtype=torch.float32),
        "ref_y": torch.tensor(y_rob_scaled, dtype=torch.float32),
        "y0": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
        "t_span": (1e-5, 1e5)
    }
    
    rob_vf = MLPVectorField(state_dim=3, hidden_dim=64, num_layers=3, activation="softplus")
    rob_train = train_surrogate(rob_vf, rob_data, max_epochs=700, lr=1e-3, timeout_sec=20.0, seed=303)
    rob_val = validate_surrogate(rob_vf, rob_data, system_name="Robertson")
    torch.save(rob_vf.state_dict(), os.path.join(CHECKPOINT_DIR, "robertson_h64.pt"))
    
    rob_passed = rob_val["all_tiers_passed"]
    rob_outcome = rob_val["tier2_outcome"]
    print(f" [{rob_outcome}] R2={rob_val['tier1_derivative_r2']:.4f}, Tier2={rob_outcome} ({rob_val['tier2_elapsed_s']}s), rho_max={rob_val['rho_max']:.2f}")
    
    all_models_summary["robertson_h64"] = {
        "layer": "Layer2B",
        "system_name": "Robertson",
        "system_role": "highly-stiff-candidate-conditional",
        "hidden_dim": 64,
        "num_params": rob_vf.num_params,
        "training_stats": rob_train,
        "validation_stats": rob_val,
        "tier2_outcome": rob_outcome,
        "all_tiers_passed": rob_passed,
        "checkpoint_file": "robertson_h64.pt"
    }

    # =========================================================================
    # 4. Fallback Handling: Fresh Dedicated VdP mu=100 Candidate
    # =========================================================================
    if not rob_passed:
        print("\n[Layer 2B Fallback] Robertson failed validation. Evaluating fresh dedicated VdP mu=100 candidate...")
        print("  Notice: Previous vdp_mu100_h64 had R2=0.2911 (failed); training fresh dedicated surrogate...")
        
        vdp100_sys = VanDerPol(mu=100.0)
        vdp100_fresh_data = generate_collocation_dataset(
            vdp100_sys,
            t_span=(0.0, 20.0),
            y0=np.array([2.0, 0.0]),
            num_traj_pts=300,
            num_lhs=500,
            seed=999
        )
        
        # Dedicated higher-capacity candidate for stiff boundary layer
        vdp100_fresh_vf = MLPVectorField(state_dim=2, hidden_dim=128, num_layers=2, activation="softplus")
        print("  Training fresh vdp_mu100_stiff_h128 surrogate...", end="", flush=True)
        vdp100_fresh_train = train_surrogate(vdp100_fresh_vf, vdp100_fresh_data, max_epochs=800, lr=1e-3, timeout_sec=20.0, seed=999)
        vdp100_fresh_val = validate_surrogate(vdp100_fresh_vf, vdp100_fresh_data, system_name="VanDerPol", mu=100.0)
        
        vdp100_fresh_ckpt = os.path.join(CHECKPOINT_DIR, "vdp_mu100_stiff_h128.pt")
        torch.save(vdp100_fresh_vf.state_dict(), vdp100_fresh_ckpt)
        
        fresh_passed = vdp100_fresh_val["all_tiers_passed"]
        fresh_outcome = vdp100_fresh_val["tier2_outcome"]
        print(f" [{fresh_outcome}] R2={vdp100_fresh_val['tier1_derivative_r2']:.4f}, Tier2={fresh_outcome}, rho_max={vdp100_fresh_val['rho_max']:.2f}")
        
        all_models_summary["vdp_mu100_stiff_h128_fallback"] = {
            "layer": "Layer2B_Fallback",
            "system_name": "VanDerPol",
            "mu": 100.0,
            "hidden_dim": 128,
            "num_params": vdp100_fresh_vf.num_params,
            "training_stats": vdp100_fresh_train,
            "validation_stats": vdp100_fresh_val,
            "tier2_outcome": fresh_outcome,
            "all_tiers_passed": fresh_passed,
            "admitted_as_stiff_neural_benchmark": fresh_passed,
            "checkpoint_file": "vdp_mu100_stiff_h128.pt"
        }
        
        if fresh_passed:
            print("  [ADMITTED] Fresh VdP mu=100 passed all 4 tiers -> Admitted as 2nd Stiff Neural ODE Benchmark.")
        else:
            print("  [NOT ADMITTED] Fresh VdP mu=100 failed 4 tiers -> Second Stiff Neural ODE Benchmark was NOT admitted.")
            print("  (Classical reference solve required for any baseline reporting; excluded from Neural ODE benchmark).")
    else:
        print("\n[Layer 2B] Scaled Robertson passed all 4 tiers -> Admitted as 2nd Stiff Neural ODE Benchmark.")

    # =========================================================================
    # 5. Save Comprehensive Validation Report
    # =========================================================================
    report_file = os.path.join(LOG_DIR, "surrogate_validation_report.json")
    with open(report_file, "w") as f:
        json.dump(all_models_summary, f, indent=2)

    print("\n" + "=" * 78)
    print(f"[OK] Stage A Surrogate training and validation complete.")
    print(f"[OK] Comprehensive validation report saved to: {report_file}")
    print("=" * 78)


if __name__ == "__main__":
    mp.freeze_support()
    main()
