"""
Test B: Scaled Robertson Validation Safety Smoke Test.

Verifies that trajectory validation on Scaled Robertson terminates safely
within the configured safety boundary under dual guards:
- Solver internal guard: max_num_steps = 1000
- Process-based watchdog: timeout_sec = 5.0

Accepts and reports the genuine outcome:
CONVERGED, STEP_LIMIT_EXCEEDED, VALIDATION_TIMEOUT, or a documented error.
"""

import os
import sys
import time
import multiprocessing as mp
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.vector_fields import MLPVectorField
from benchmarks.systems import Robertson


def _robertson_worker(vf_state, vf_kwargs, y0_t, t_eval_t, rtol, atol, max_num_steps, result_queue):
    """Executes Robertson integration in an isolated child process."""
    try:
        from models.vector_fields import MLPVectorField
        from torchdiffeq import odeint
        
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
            
        result_queue.put(("CONVERGED", sol_np, None))
    except AssertionError as e:
        if "max_num_steps exceeded" in str(e):
            result_queue.put(("STEP_LIMIT_EXCEEDED", None, str(e)))
        else:
            result_queue.put(("ASSERTION_ERROR", None, str(e)))
    except Exception as e:
        result_queue.put(("INTEGRATION_ERROR", None, str(e)))


def validate_robertson_with_watchdog(model, y0, t_eval, rtol=1e-5, atol=1e-7, max_num_steps=1000, timeout_sec=5.0):
    """Executes validation integration with both a step limit and process watchdog."""
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    vf_state = {k: v.cpu() for k, v in model.state_dict().items()}
    vf_kwargs = {
        "state_dim": 3,
        "hidden_dim": 64,
        "num_layers": 3,
        "activation": "softplus"
    }

    start_time = time.perf_counter()
    p = ctx.Process(
        target=_robertson_worker,
        args=(vf_state, vf_kwargs, y0.cpu(), t_eval.cpu(), rtol, atol, max_num_steps, result_queue)
    )
    p.start()
    p.join(timeout=timeout_sec)

    is_timeout = False
    if p.is_alive():
        is_timeout = True
        p.terminate()
        p.join()

    elapsed = time.perf_counter() - start_time

    if is_timeout:
        return {
            "status": "VALIDATION_TIMEOUT",
            "elapsed_seconds": round(elapsed, 3),
            "solution": None,
            "error_msg": "Exceeded process-level timeout watchdog"
        }

    if not result_queue.empty():
        status, sol, err = result_queue.get()
        return {
            "status": status,
            "elapsed_seconds": round(elapsed, 3),
            "solution": sol,
            "error_msg": err
        }
    else:
        return {
            "status": "PROCESS_TERMINATED_ABRUPTLY",
            "elapsed_seconds": round(elapsed, 3),
            "solution": None,
            "error_msg": "Child process exited without writing to queue"
        }


def main():
    print("=" * 70)
    print("Test B: Scaled Robertson Validation Safety Smoke Test")
    print("=" * 70)

    # 1. Instantiate or load Robertson vector field
    ckpt_path = os.path.join(ROOT_DIR, "models", "checkpoints", "robertson_h64.pt")
    vf = MLPVectorField(state_dim=3, hidden_dim=64, num_layers=3, activation="softplus")
    if os.path.exists(ckpt_path):
        print(f"Loading existing checkpoint: {ckpt_path}")
        vf.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    else:
        print("Using freshly initialized Scaled Robertson MLPVectorField (H=64, 3D)...")

    # 2. Setup short-horizon evaluation grid (where previous dopri5 solve stalled)
    t_log = np.logspace(-5, 5, 200)
    num_short = int(0.25 * len(t_log))
    t_short = torch.tensor(t_log[:num_short], dtype=torch.float32)
    y0_scaled = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

    timeout_cfg = 5.0
    step_limit_cfg = 1000
    print(f"Integration interval: t in [{t_short[0]:.2e}, {t_short[-1]:.2e}] ({len(t_short)} points)")
    print(f"Configured guards:    max_num_steps={step_limit_cfg}, timeout_sec={timeout_cfg}s")
    print("Executing integration with process watchdog...", flush=True)

    # 3. Execute guarded integration
    outcome = validate_robertson_with_watchdog(
        vf,
        y0_scaled,
        t_short,
        rtol=1e-5,
        atol=1e-7,
        max_num_steps=step_limit_cfg,
        timeout_sec=timeout_cfg
    )

    print("\n--- Measured Smoke Test Outcome ---")
    print(f"Status:          {outcome['status']}")
    print(f"Elapsed Time:    {outcome['elapsed_seconds']} seconds")
    if outcome["error_msg"]:
        print(f"Message/Detail:  {outcome['error_msg']}")

    # 4. Verification of safe termination
    valid_statuses = ["CONVERGED", "STEP_LIMIT_EXCEEDED", "VALIDATION_TIMEOUT", "INTEGRATION_ERROR"]
    assert outcome["status"] in valid_statuses, f"Unexpected outcome status: {outcome['status']}"

    # Verify termination occurred within configured safety boundary (timeout + margin for process overhead)
    max_safe_elapsed = timeout_cfg + 4.0
    assert outcome["elapsed_seconds"] <= max_safe_elapsed, (
        f"Execution took {outcome['elapsed_seconds']}s, exceeding safety limit of {max_safe_elapsed}s"
    )

    print("\n[OK] Safe termination verified: Process terminated within safety boundary.")
    print("=" * 70)


if __name__ == "__main__":
    mp.freeze_support()
    main()
