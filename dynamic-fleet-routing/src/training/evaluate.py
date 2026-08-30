"""Evaluation framework for all dispatch methods.

Runs episodes with a given dispatcher and collects standardized metrics.

Usage:
    python -m src.training.evaluate --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.environment.fleet_env import DynamicFleetEnv
from src.environment.request import RequestStatus
from src.utils.config import load_config, load_base_config
from src.utils.logger import setup_logger, get_logger
from src.utils.metrics import EpisodeMetrics, MetricsCollector, LatencyTracker
from src.utils.seed import set_global_seed


def evaluate_dispatcher(
    env: DynamicFleetEnv,
    dispatcher: Any,
    n_episodes: int = 5,
    seed: int = 42,
) -> MetricsCollector:
    """Evaluate a dispatcher over multiple episodes.

    Args:
        env: The fleet environment.
        dispatcher: Object with select_action(env) -> int method.
        n_episodes: Number of evaluation episodes.
        seed: Base random seed.

    Returns:
        MetricsCollector with all episode results.
    """
    logger = get_logger("evaluate")
    collector = MetricsCollector()
    latency_tracker = LatencyTracker()

    for ep in range(n_episodes):
        ep_seed = seed + ep
        obs, info = env.reset(seed=ep_seed)
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            latency_tracker.start()
            action = dispatcher.select_action(env)
            lat_ms = latency_tracker.stop()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            steps += 1

            collector.record_step(
                reward=reward,
                reward_components=info.get("reward_breakdown"),
                latency_ms=lat_ms,
            )

        # Collect episode-level metrics
        total_requests = len(env.requests)
        delivered = sum(
            1 for r in env.requests.values()
            if r.status == RequestStatus.DELIVERED
        )
        expired = sum(
            1 for r in env.requests.values()
            if r.status == RequestStatus.EXPIRED
        )

        # Turnaround times and SLA
        for r in env.requests.values():
            if r.status == RequestStatus.DELIVERED:
                tt = r.turnaround_time
                on_time = r.was_delivered_on_time()
                if tt is not None and on_time is not None:
                    collector.record_delivery(tt, on_time)

        total_distance = sum(v.total_distance for v in env.vehicles)
        total_fuel = sum(v.fuel_consumed for v in env.vehicles)
        avg_util = np.mean([v.utilization for v in env.vehicles])

        ep_metrics = EpisodeMetrics(
            total_requests=total_requests,
            delivered_requests=delivered,
            expired_requests=expired,
            total_distance=total_distance,
            total_fuel_consumed=total_fuel,
            fleet_utilization=float(avg_util),
            total_reward=ep_reward,
        )

        # Throughput
        ep_duration = env.current_time / 60.0  # hours
        if ep_duration > 0:
            ep_metrics.requests_per_second = total_requests / (ep_duration * 3600)

        collector.finish_episode(ep_metrics)

        logger.info(
            f"Episode {ep+1}/{n_episodes} | "
            f"Delivered: {delivered}/{total_requests} | "
            f"Reward: {ep_reward:.1f} | "
            f"Steps: {steps}"
        )

    return collector


def run_all_baselines(
    config: dict[str, Any],
    n_episodes: int = 3,
    seed: int = 42,
    save_dir: str = "artifacts/metrics",
) -> dict[str, dict[str, Any]]:
    """Run evaluation for all baseline dispatchers.

    Args:
        config: Environment configuration.
        n_episodes: Number of evaluation episodes per method.
        seed: Random seed.
        save_dir: Directory for saving results.

    Returns:
        Dictionary mapping method name to summary metrics.
    """
    logger = setup_logger("evaluate")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    from src.baselines.nearest_vehicle import NearestVehicleDispatcher
    from src.baselines.greedy_dispatch import GreedyDispatcher
    from src.baselines.ortools_vrp import ORToolsVRPDispatcher

    dispatchers = [
        NearestVehicleDispatcher(),
        GreedyDispatcher(),
        ORToolsVRPDispatcher(time_limit_ms=500),
    ]

    results: dict[str, dict[str, Any]] = {}

    for dispatcher in dispatchers:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {dispatcher.name}")
        logger.info(f"{'='*60}")

        env = DynamicFleetEnv(config)
        collector = evaluate_dispatcher(env, dispatcher, n_episodes, seed)

        summary = collector.get_summary()
        results[dispatcher.name] = summary

        # Save per-method CSV
        csv_path = os.path.join(save_dir, f"{dispatcher.name}_results.csv")
        collector.export_csv(csv_path)
        logger.info(f"Results saved to {csv_path}")

    # Print comparison table
    _print_comparison_table(results)

    return results


def _print_comparison_table(results: dict[str, dict[str, Any]]) -> None:
    """Print a formatted comparison table."""
    logger = get_logger("evaluate")

    metrics_to_show = [
        ("completion_rate_mean", "Completion Rate"),
        ("sla_compliance_rate_mean", "SLA Compliance"),
        ("avg_turnaround_time_mean", "Avg Turnaround (min)"),
        ("total_distance_mean", "Total Distance (km)"),
        ("total_fuel_consumed_mean", "Total Fuel"),
        ("total_reward_mean", "Total Reward"),
        ("avg_inference_latency_ms_mean", "Avg Latency (ms)"),
    ]

    header = f"{'Metric':<25}"
    for name in results:
        header += f" | {name:<15}"
    logger.info(f"\n{header}")
    logger.info("-" * len(header))

    for metric_key, metric_name in metrics_to_show:
        row = f"{metric_name:<25}"
        for name, summary in results.items():
            val = summary.get(metric_key, 0.0)
            row += f" | {val:<15.4f}"
        logger.info(row)


def main() -> None:
    """Run evaluation from command line."""
    parser = argparse.ArgumentParser(description="Evaluate fleet routing methods")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained PPO model (.zip)")
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="artifacts/metrics")
    args = parser.parse_args()

    config = load_config(args.config) if os.path.exists(args.config) else load_base_config()
    set_global_seed(args.seed)

    results = run_all_baselines(config, args.n_episodes, args.seed, args.save_dir)

    # If model provided, also evaluate PPO
    if args.model and os.path.exists(args.model):
        from src.agents.ppo_agent import PPOAgent
        logger = get_logger("evaluate")
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: PPO (from {args.model})")
        logger.info(f"{'='*60}")

        env = DynamicFleetEnv(config)
        agent = PPOAgent(config=config)
        agent.load(args.model)

        collector = evaluate_dispatcher(env, agent, args.n_episodes, args.seed)
        results["ppo"] = collector.get_summary()
        collector.export_csv(os.path.join(args.save_dir, "ppo_results.csv"))

    # Save combined results
    import csv
    combined_path = os.path.join(args.save_dir, "comparison_results.csv")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    if results:
        all_keys = set()
        for summary in results.values():
            all_keys.update(summary.keys())

        with open(combined_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["method"] + sorted(all_keys))
            writer.writeheader()
            for method, summary in results.items():
                row = {"method": method}
                row.update(summary)
                writer.writerow(row)


if __name__ == "__main__":
    main()
