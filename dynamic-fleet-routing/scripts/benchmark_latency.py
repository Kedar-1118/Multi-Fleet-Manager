"""Latency benchmarking script.

Measures inference latency for PPO, MCTS, and Hybrid PPO+MCTS
dispatch methods with percentile statistics.

Usage:
    python scripts/benchmark_latency.py
"""

from __future__ import annotations

import sys
import os
import time

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.fleet_env import DynamicFleetEnv
from src.baselines.nearest_vehicle import NearestVehicleDispatcher
from src.baselines.greedy_dispatch import GreedyDispatcher
from src.planning.mcts import MCTSPlanner
from src.utils.config import load_base_config
from src.utils.seed import set_global_seed
from src.utils.metrics import LatencyTracker


def benchmark_dispatcher(
    env: DynamicFleetEnv,
    dispatcher,
    name: str,
    warmup_iters: int = 100,
    benchmark_iters: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Benchmark a dispatcher's inference latency.

    Args:
        env: Fleet environment.
        dispatcher: Object with select_action(env) method.
        name: Name of the dispatcher.
        warmup_iters: Number of warmup iterations.
        benchmark_iters: Number of benchmark iterations.
        seed: Random seed.

    Returns:
        Dictionary with latency statistics.
    """
    tracker = LatencyTracker()

    # Warmup
    print(f"\n[{name}] Warming up ({warmup_iters} iterations)...")
    for i in range(warmup_iters):
        env.reset(seed=seed + (i % 10))
        # Take a few random steps to get to interesting states
        mask = env.get_action_mask()
        valid = np.where(mask)[0]
        if len(valid) > 0:
            env.step(int(np.random.choice(valid)))

        tracker.start()
        dispatcher.select_action(env)
        tracker.stop()

    tracker.reset()  # Clear warmup data

    # Benchmark
    print(f"[{name}] Benchmarking ({benchmark_iters} iterations)...")
    for i in range(benchmark_iters):
        env.reset(seed=seed + (i % 50))
        # Advance to a decision-relevant state
        for _ in range(min(3, 10)):
            mask = env.get_action_mask()
            valid = np.where(mask)[0]
            if len(valid) > 1:
                env.step(int(np.random.choice(valid)))

        tracker.start()
        dispatcher.select_action(env)
        tracker.stop()

    stats = tracker.get_stats()
    stats["name"] = name

    print(f"[{name}] Results:")
    print(f"  Mean:  {stats['mean']:.4f} ms")
    print(f"  P50:   {stats['p50']:.4f} ms")
    print(f"  P95:   {stats['p95']:.4f} ms")
    print(f"  P99:   {stats['p99']:.4f} ms")
    print(f"  Count: {stats['count']}")

    return stats


def main():
    """Run all latency benchmarks."""
    print("=" * 60)
    print("Fleet Routing Latency Benchmark")
    print("=" * 60)

    set_global_seed(42)

    config = load_base_config({
        "simulation": {"num_nodes": 50, "num_vehicles": 5,
                       "simulation_duration": 300, "seed": 42},
        "requests": {"lambda_rate": 5, "arrival_interval": 20},
        "reward": {"normalize": False},
    })

    env = DynamicFleetEnv(config)

    # Dispatchers to benchmark
    dispatchers = [
        ("Nearest Vehicle", NearestVehicleDispatcher()),
        ("Greedy", GreedyDispatcher()),
        ("MCTS (10 sims)", MCTSPlanner(num_simulations=10, rollout_horizon=3)),
        ("MCTS (50 sims)", MCTSPlanner(num_simulations=50, rollout_horizon=5)),
    ]

    all_results = []

    for name, dispatcher in dispatchers:
        stats = benchmark_dispatcher(
            env, dispatcher, name,
            warmup_iters=100,
            benchmark_iters=1000,
        )
        all_results.append(stats)

    # Print summary table
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Method':<25} {'Mean':>8} {'P50':>8} {'P95':>8} {'P99':>8}")
    print("-" * 60)
    for r in all_results:
        print(
            f"{r['name']:<25} "
            f"{r['mean']:>7.3f}ms "
            f"{r['p50']:>7.3f}ms "
            f"{r['p95']:>7.3f}ms "
            f"{r['p99']:>7.3f}ms"
        )

    # Save results
    os.makedirs("artifacts/metrics", exist_ok=True)
    import csv
    with open("artifacts/metrics/latency_benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "mean", "p50", "p95", "p99", "count"])
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved to artifacts/metrics/latency_benchmark.csv")


if __name__ == "__main__":
    main()
