"""
Automated Experiment Metadata Logger.
Implements Rule 2 (Reproducibility) from the supervisor's guidelines.
Logs experiment configuration, hardware specs, NFE, timings, and observations into CSV.
"""

import os
import csv
from datetime import datetime


class ExperimentLogger:
    def __init__(self, log_dir="experiments/logs", filename="benchmark_results.csv"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, filename)

        self.columns = [
            "experiment_id",
            "timestamp",
            "ode_name",
            "stiffness",
            "solver",
            "rtol",
            "atol",
            "step_size",
            "net_preset",
            "num_params",
            "device",
            "nfe",
            "total_runtime_ms",
            "tf_mean_ms",
            "net_eval_ms",
            "solver_overhead_ms",
            "solver_overhead_ratio",
            "peak_mem_mb",
            "trajectory_mse",
            "status",
            "observation"
        ]

        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.columns)

    def log(self, entry: dict):
        """
        Appends a record to the CSV logger.
        """
        row = [
            entry.get("experiment_id", ""),
            entry.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            entry.get("ode_name", ""),
            entry.get("stiffness", ""),
            entry.get("solver", ""),
            entry.get("rtol", ""),
            entry.get("atol", ""),
            entry.get("step_size", "adaptive"),
            entry.get("net_preset", ""),
            entry.get("num_params", 0),
            entry.get("device", "cpu"),
            entry.get("nfe", 0),
            round(entry.get("total_runtime_ms", 0.0), 3),
            round(entry.get("tf_mean_ms", 0.0), 5),
            round(entry.get("net_eval_ms", 0.0), 3),
            round(entry.get("solver_overhead_ms", 0.0), 3),
            round(entry.get("solver_overhead_ratio", 0.0), 4),
            round(entry.get("peak_mem_mb", 0.0), 2),
            entry.get("trajectory_mse", "N/A"),
            entry.get("status", "SUCCESS"),
            entry.get("observation", "")
        ]

        with open(self.log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        print(f"[LOGGED] Exp: {entry.get('experiment_id')} | NFE: {entry.get('nfe')} | Time: {round(entry.get('total_runtime_ms', 0), 2)}ms | Overhead: {round(entry.get('solver_overhead_ratio', 0)*100, 1)}%")
