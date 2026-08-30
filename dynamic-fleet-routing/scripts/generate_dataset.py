"""Generate simulation datasets for offline analysis.

Usage:
    python scripts/generate_dataset.py
"""

from __future__ import annotations

import sys
import os
import csv
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.environment.fleet_env import DynamicFleetEnv
from src.environment.request import RequestStatus
from src.utils.config import load_base_config
from src.utils.seed import set_global_seed


def main():
    """Generate a simulation dataset."""
    set_global_seed(42)

    config = load_base_config()
    env = DynamicFleetEnv(config)

    output_dir = Path("artifacts/datasets")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating simulation dataset...")

    obs, info = env.reset(seed=42)

    # Run a full episode with greedy dispatch
    from src.baselines.greedy_dispatch import GreedyDispatcher
    dispatcher = GreedyDispatcher()

    step_data = []
    request_data = []
    step_count = 0

    while True:
        action = dispatcher.select_action(env)
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

        step_data.append({
            "step": step_count,
            "time": env.current_time,
            "action": action,
            "reward": reward,
            "pending_requests": info.get("pending_requests", 0),
            "total_deliveries": info.get("total_deliveries", 0),
            "traffic_state": info.get("traffic_state", ""),
            "fleet_utilization": info.get("fleet_utilization", 0.0),
        })

        if terminated or truncated:
            break

    # Collect request data
    for req in env.requests.values():
        request_data.append({
            "request_id": req.request_id,
            "pickup_location": req.pickup_location,
            "dropoff_location": req.dropoff_location,
            "request_time": req.request_time,
            "deadline": req.deadline,
            "priority": req.priority.value,
            "package_size": req.package_size,
            "status": req.status.value,
            "delivery_time": req.delivery_time,
            "turnaround_time": req.turnaround_time,
        })

    # Save step data
    with open(output_dir / "simulation_steps.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=step_data[0].keys())
        writer.writeheader()
        writer.writerows(step_data)

    # Save request data
    with open(output_dir / "requests.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=request_data[0].keys())
        writer.writeheader()
        writer.writerows(request_data)

    total = len(request_data)
    delivered = sum(1 for r in request_data if r["status"] == "DELIVERED")
    expired = sum(1 for r in request_data if r["status"] == "EXPIRED")

    print(f"Generated {step_count} simulation steps")
    print(f"Total requests: {total}")
    print(f"Delivered: {delivered} ({delivered/total*100:.1f}%)")
    print(f"Expired: {expired} ({expired/total*100:.1f}%)")
    print(f"Data saved to {output_dir}")


if __name__ == "__main__":
    main()
