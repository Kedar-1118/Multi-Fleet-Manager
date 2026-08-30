"""Run ablation experiments comparing all dispatch methods.

Usage:
    python scripts/run_experiment.py
"""

from __future__ import annotations

import sys
import os
import csv
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.fleet_env import DynamicFleetEnv
from src.baselines.nearest_vehicle import NearestVehicleDispatcher
from src.baselines.greedy_dispatch import GreedyDispatcher
from src.baselines.ortools_vrp import ORToolsVRPDispatcher
from src.planning.mcts import MCTSPlanner
from src.training.evaluate import evaluate_dispatcher
from src.utils.config import load_base_config
from src.utils.seed import set_global_seed
from src.utils.logger import setup_logger


def run_experiment(
    name: str,
    config: dict,
    dispatchers: list[tuple[str, object]],
    n_episodes: int = 3,
    seed: int = 42,
) -> dict[str, dict]:
    """Run an experiment with given config and dispatchers.

    Args:
        name: Experiment name.
        config: Environment configuration.
        dispatchers: List of (name, dispatcher) tuples.
        n_episodes: Episodes per dispatcher.
        seed: Random seed.

    Returns:
        Results dict mapping dispatcher name to metrics summary.
    """
    logger = setup_logger("experiment")
    logger.info(f"\n{'='*60}")
    logger.info(f"Experiment: {name}")
    logger.info(f"{'='*60}")

    results = {}
    for disp_name, dispatcher in dispatchers:
        logger.info(f"\n--- {disp_name} ---")
        env = DynamicFleetEnv(config)
        collector = evaluate_dispatcher(env, dispatcher, n_episodes, seed)
        results[disp_name] = collector.get_summary()

    return results


def main():
    """Run all ablation experiments."""
    logger = setup_logger("experiment")
    set_global_seed(42)

    output_dir = Path("artifacts/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_base_config()

    # Base dispatchers
    base_dispatchers = [
        ("nearest_vehicle", NearestVehicleDispatcher()),
        ("greedy", GreedyDispatcher()),
        ("ortools", ORToolsVRPDispatcher(time_limit_ms=500)),
        ("mcts_10", MCTSPlanner(num_simulations=10, rollout_horizon=3)),
        ("mcts_50", MCTSPlanner(num_simulations=50, rollout_horizon=5)),
    ]

    all_experiment_results = {}

    # Experiment 1: Base comparison
    results = run_experiment(
        "baseline_comparison", base_config, base_dispatchers, n_episodes=3
    )
    all_experiment_results["baseline_comparison"] = results

    # Experiment 5: Different traffic volatility
    for traffic_name, congestion_prob in [("low_traffic", 0.0), ("high_traffic", 0.2)]:
        traffic_config = load_base_config({
            "traffic": {"congestion_probability": congestion_prob},
        })
        results = run_experiment(
            f"traffic_{traffic_name}", traffic_config,
            base_dispatchers[:3], n_episodes=2
        )
        all_experiment_results[f"traffic_{traffic_name}"] = results

    # Experiment 6: Different request arrival rates
    for rate_name, lambda_rate in [("low_demand", 2), ("high_demand", 10)]:
        rate_config = load_base_config({
            "requests": {"lambda_rate": lambda_rate},
        })
        results = run_experiment(
            f"demand_{rate_name}", rate_config,
            base_dispatchers[:3], n_episodes=2
        )
        all_experiment_results[f"demand_{rate_name}"] = results

    # Experiment 7: Different fleet sizes
    for fleet_name, n_vehicles in [("small_fleet", 3), ("large_fleet", 10)]:
        fleet_config = load_base_config({
            "simulation": {"num_vehicles": n_vehicles},
            "observation": {"max_vehicles": n_vehicles},
        })
        results = run_experiment(
            f"fleet_{fleet_name}", fleet_config,
            base_dispatchers[:3], n_episodes=2
        )
        all_experiment_results[f"fleet_{fleet_name}"] = results

    # Save all results
    _save_experiment_results(all_experiment_results, output_dir)

    logger.info(f"\nAll experiments complete. Results in {output_dir}")


def _save_experiment_results(
    all_results: dict[str, dict[str, dict]],
    output_dir: Path,
) -> None:
    """Save experiment results to CSV files."""
    # Combined summary
    rows = []
    for exp_name, dispatchers in all_results.items():
        for disp_name, metrics in dispatchers.items():
            row = {
                "experiment": exp_name,
                "method": disp_name,
            }
            row.update(metrics)
            rows.append(row)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_dir / "experiment_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
