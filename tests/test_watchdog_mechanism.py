"""
Test A: Process-Based Watchdog Mechanism Verification.

Tests that an isolated child process executing a stalled or runaway task is
forcibly terminated when timeout_sec is exceeded, allowing the parent process
to recover safely within a bounded safety window on Windows.
Also verifies that a normal fast task returns immediately without waiting for timeout.
"""

import time
import multiprocessing as mp
import sys


def _stalled_worker(result_queue):
    """Simulates a runaway or blocked computation."""
    try:
        # Long CPU-bound or sleep stall
        time.sleep(10.0)
        result_queue.put(("SUCCESS", "Finished stall (unexpected)"))
    except Exception as e:
        result_queue.put(("ERROR", str(e)))


def _fast_worker(x, result_queue):
    """Simulates a fast successful computation."""
    try:
        res = x * 2
        result_queue.put(("SUCCESS", res))
    except Exception as e:
        result_queue.put(("ERROR", str(e)))


def run_with_watchdog(target_fn, args=(), timeout_sec=2.0):
    """Executes target_fn in a spawned subprocess with hard termination on timeout."""
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    all_args = args + (result_queue,)

    start_time = time.perf_counter()
    p = ctx.Process(target=target_fn, args=all_args)
    p.start()
    p.join(timeout=timeout_sec)

    is_timeout = False
    if p.is_alive():
        is_timeout = True
        p.terminate()  # Forcibly terminate child process
        p.join()

    elapsed = time.perf_counter() - start_time

    if is_timeout:
        return {
            "status": "VALIDATION_TIMEOUT",
            "elapsed_seconds": round(elapsed, 3),
            "result": None,
            "child_alive_after_terminate": p.is_alive()
        }

    if not result_queue.empty():
        status, val = result_queue.get()
        return {
            "status": status,
            "elapsed_seconds": round(elapsed, 3),
            "result": val,
            "child_alive_after_terminate": False
        }
    else:
        return {
            "status": "PROCESS_TERMINATED_ABRUPTLY",
            "elapsed_seconds": round(elapsed, 3),
            "result": None,
            "child_alive_after_terminate": False
        }


def main():
    print("=" * 70)
    print("Test A: Process-Based Watchdog Mechanism Verification")
    print("=" * 70)

    # 1. Test fast successful execution
    print("\n1. Testing fast task (timeout=2.0s)...")
    fast_out = run_with_watchdog(_fast_worker, args=(21,), timeout_sec=2.0)
    print(f"   Fast Task Result:  {fast_out['status']} | Output={fast_out['result']} | Elapsed={fast_out['elapsed_seconds']}s")
    assert fast_out["status"] == "SUCCESS", f"Expected SUCCESS, got {fast_out['status']}"
    assert fast_out["result"] == 42, f"Expected 42, got {fast_out['result']}"
    assert fast_out["elapsed_seconds"] < 2.0, "Fast task exceeded timeout window"
    print("   [OK] Fast task completed without delay.")

    # 2. Test stalled task termination
    timeout_limit = 2.5
    print(f"\n2. Testing stalled task (stall=10.0s, timeout={timeout_limit}s)...")
    stall_out = run_with_watchdog(_stalled_worker, timeout_sec=timeout_limit)
    print(f"   Stalled Task Status:  {stall_out['status']}")
    print(f"   Elapsed Time:         {stall_out['elapsed_seconds']}s")
    print(f"   Child Still Alive:    {stall_out['child_alive_after_terminate']}")

    # Verification assertions
    assert stall_out["status"] == "VALIDATION_TIMEOUT", f"Expected VALIDATION_TIMEOUT, got {stall_out['status']}"
    assert not stall_out["child_alive_after_terminate"], "Child process was not terminated"
    # Allow reasonable margin for Windows process startup and termination overhead (up to timeout + 3.0s)
    max_allowed_elapsed = timeout_limit + 3.0
    assert timeout_limit <= stall_out["elapsed_seconds"] <= max_allowed_elapsed, (
        f"Elapsed time {stall_out['elapsed_seconds']}s outside allowable safety window [{timeout_limit}, {max_allowed_elapsed}]"
    )
    print("   [OK] Runaway child process was forcibly terminated within safety boundary.")

    print("\n" + "=" * 70)
    print("[PASS] Test A completed successfully: Process-based watchdog verified.")
    print("=" * 70)


if __name__ == "__main__":
    mp.freeze_support()
    main()
