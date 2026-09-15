"""
Hardware Profiling and Instrumentation Harness for Track 2.
Decomposes Neural ODE execution time:
T_NODE = NFE * T_f + T_solver + T_launch + T_sync

Measures:
- High-precision wall-clock time (CPU perf_counter, CUDA Events)
- Isolated vector field evaluation cost (T_f)
- Number of Function Evaluations (NFE)
- Solver internal overhead (T_solver)
- Memory consumption (RAM / VRAM)
"""

import time
import os
import torch
import numpy as np


class ProfilingHarness:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)

    def measure_tf(self, vector_field, sample_z, sample_t, num_repeats=50, num_warmup=10):
        """
        Measures the isolated execution time of a single vector field evaluation T_f.
        """
        vector_field.eval()
        z = sample_z.to(self.device)
        t = sample_t.to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = vector_field(t, z)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            with torch.no_grad():
                for _ in range(num_repeats):
                    _ = vector_field(t, z)
            end_event.record()
            torch.cuda.synchronize()
            total_time_ms = start_event.elapsed_time(end_event)
            mean_tf_ms = total_time_ms / num_repeats
        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(num_repeats):
                    _ = vector_field(t, z)
            t1 = time.perf_counter()
            mean_tf_ms = ((t1 - t0) * 1000.0) / num_repeats

        return mean_tf_ms

    def profile_integration(self, node_model, z0, t_eval, options=None, num_repeats=5):
        """
        Profiles a full continuous trajectory integration.
        Returns:
            metrics dict:
                - total_time_ms
                - nfe
                - tf_mean_ms
                - total_net_eval_ms (NFE * T_f)
                - solver_overhead_ms (total_time_ms - total_net_eval_ms)
                - solver_overhead_ratio
                - trajectory (output tensor)
        """
        node_model.to(self.device)
        z0 = z0.to(self.device)
        t_eval = t_eval.to(self.device)

        # First measure isolated T_f
        dummy_t = t_eval[0]
        tf_mean_ms = self.measure_tf(node_model.func, z0, dummy_t)

        # Warmup solve
        with torch.no_grad():
            _ = node_model(z0, t_eval, options=options)

        times = []
        nfe_recorded = 0
        last_out = None

        for _ in range(num_repeats):
            node_model.reset_nfe()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                s_evt = torch.cuda.Event(enable_timing=True)
                e_evt = torch.cuda.Event(enable_timing=True)
                s_evt.record()
                with torch.no_grad():
                    out = node_model(z0, t_eval, options=options)
                e_evt.record()
                torch.cuda.synchronize()
                elapsed_ms = s_evt.elapsed_time(e_evt)
            else:
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = node_model(z0, t_eval, options=options)
                t1 = time.perf_counter()
                elapsed_ms = (t1 - t0) * 1000.0

            times.append(elapsed_ms)
            nfe_recorded = node_model.nfe
            last_out = out

        mean_total_ms = float(np.mean(times))
        std_total_ms = float(np.std(times))
        total_net_eval_ms = nfe_recorded * tf_mean_ms
        solver_overhead_ms = max(0.0, mean_total_ms - total_net_eval_ms)
        solver_overhead_ratio = solver_overhead_ms / mean_total_ms if mean_total_ms > 0 else 0.0

        # Memory tracking
        if self.device.type == "cuda":
            peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            peak_mem_mb = 0.0

        return {
            "total_time_ms": mean_total_ms,
            "std_time_ms": std_total_ms,
            "nfe": nfe_recorded,
            "tf_mean_ms": tf_mean_ms,
            "total_net_eval_ms": total_net_eval_ms,
            "solver_overhead_ms": solver_overhead_ms,
            "solver_overhead_ratio": solver_overhead_ratio,
            "peak_mem_mb": peak_mem_mb,
            "trajectory": last_out.cpu()
        }
