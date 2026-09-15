"""
Phase 0: Workspace Audit and File Hygiene Script.
Performs read-only verification of ground truth datasets, archives flawed legacy scripts,
and checks baseline infrastructure integrity.
"""

import os
import shutil
import glob
import torch

def run_phase_0():
    print("=== PHASE 0: AUDIT & FILE HYGIENE ===")

    # 1. Create Archive Folders
    os.makedirs("experiments/archive", exist_ok=True)
    os.makedirs("experiments/logs/archive", exist_ok=True)
    print("[OK] Created archive directories: experiments/archive/ and experiments/logs/archive/")

    # 2. Archive legacy scripts
    legacy_scripts = [
        "experiments/exp1_solver_sweep.py",
        "experiments/exp2_critical_workload.py",
        "experiments/exp3_stiffness_breakdown.py",
        "experiments/exp3_tolerance_scaling.py",
        "experiments/exp3_batch_scaling.py"
    ]
    archived_count = 0
    for s in legacy_scripts:
        if os.path.exists(s):
            shutil.move(s, os.path.join("experiments/archive", os.path.basename(s)))
            print(f"  Archived legacy script: {s}")
            archived_count += 1
    print(f"[OK] Total legacy scripts archived: {archived_count}")

    # 3. Archive old CSVs
    csv_count = 0
    for csv_file in glob.glob("experiments/logs/*.csv"):
        shutil.move(csv_file, os.path.join("experiments/logs/archive", os.path.basename(csv_file)))
        print(f"  Archived legacy CSV: {csv_file}")
        csv_count += 1
    print(f"[OK] Total legacy CSV logs archived: {csv_count}")

    # 4. Verify Ground-Truth Reference Datasets
    print("\n[AUDIT 1/2] Verifying Ground-Truth Reference Datasets:")
    gt_files = [
        "data/trajectories/lotka-volterra_ground_truth.pt",
        "data/trajectories/fitzhugh-nagumo_ground_truth.pt",
        "data/trajectories/vanderpol_ground_truth.pt",
        "data/trajectories/robertson_ground_truth.pt"
    ]
    all_valid = True
    for f in gt_files:
        if not os.path.exists(f):
            print(f"  [FAIL] Missing reference dataset: {f}")
            all_valid = False
            continue
        data = torch.load(f)
        t = data["t"]
        y = data["y"]
        has_nan = torch.isnan(y).any().item()
        status = "FAIL (Contains NaN)" if has_nan else "PASS"
        if has_nan:
            all_valid = False
        print(f"  [{status}] {os.path.basename(f):35s} | shape: {str(y.shape):18s} | dtype: {str(y.dtype):14s} | NaN: {has_nan}")

    # 5. Verify Core Modules
    print("\n[AUDIT 2/2] Verifying Core Infrastructure Modules:")
    core_modules = [
        "benchmarks/systems.py",
        "models/vector_fields.py",
        "models/neural_ode.py",
        "models/scaled_node.py",
        "profiling/logger.py"
    ]
    for m in core_modules:
        exists = os.path.exists(m)
        status = "PASS" if exists else "FAIL"
        if not exists:
            all_valid = False
        print(f"  [{status}] {m}")

    print("\n==========================================")
    if all_valid:
        print("PHASE 0 AUDIT STATUS: SUCCESS - READY FOR PHASE 1")
    else:
        print("PHASE 0 AUDIT STATUS: FAILED (See errors above)")
    print("==========================================")

if __name__ == "__main__":
    run_phase_0()
