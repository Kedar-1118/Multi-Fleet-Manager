"""Carbon Emissions Analysis: EV vs ICE Fleet Comparison.

Runs a full 24-hour simulation for each dispatch method and compares
CO₂ emissions between an all-electric fleet and an equivalent ICE baseline.

Usage:
    python scripts/carbon_analysis.py
    python scripts/carbon_analysis.py --episodes 5 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.environment.fleet_env import DynamicFleetEnv
from src.environment.request import RequestStatus
from src.utils.config import load_base_config
from src.utils.emissions import EmissionsCalculator
from src.utils.seed import set_global_seed


def run_episode(
    env: DynamicFleetEnv, dispatcher: Any, seed: int
) -> dict[str, float]:
    """Run a single simulation episode and collect metrics.

    Args:
        env: Fleet environment.
        dispatcher: Dispatcher with select_action(env) method.
        seed: Random seed for this episode.

    Returns:
        Dictionary of episode metrics.
    """
    obs, info = env.reset(seed=seed)
    done = False
    steps = 0

    while not done:
        action = dispatcher.select_action(env)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

    # Collect metrics
    total_requests = len(env.requests)
    delivered = sum(
        1 for r in env.requests.values()
        if r.status == RequestStatus.DELIVERED
    )

    total_distance = sum(v.total_distance for v in env.vehicles)
    total_fuel = sum(v.fuel_consumed for v in env.vehicles)
    total_energy = sum(v.total_energy_consumed_kwh for v in env.vehicles)
    avg_battery_soc = float(np.mean([v.battery_soc for v in env.vehicles]))

    # SLA compliance
    on_time = 0
    total_delivered = 0
    turnaround_times = []
    for r in env.requests.values():
        if r.status == RequestStatus.DELIVERED:
            total_delivered += 1
            if r.was_delivered_on_time():
                on_time += 1
            tt = r.turnaround_time
            if tt is not None:
                turnaround_times.append(tt)

    sla_rate = on_time / max(total_delivered, 1)
    completion_rate = delivered / max(total_requests, 1)
    avg_turnaround = float(np.mean(turnaround_times)) if turnaround_times else 0.0

    return {
        "total_requests": total_requests,
        "delivered": delivered,
        "completion_rate": completion_rate,
        "sla_compliance": sla_rate,
        "avg_turnaround_min": avg_turnaround,
        "total_distance_km": total_distance,
        "total_fuel": total_fuel,
        "total_energy_kwh": total_energy,
        "avg_final_battery_soc": avg_battery_soc,
        "steps": steps,
    }


def main() -> None:
    """Run carbon emissions analysis across all dispatch methods."""
    parser = argparse.ArgumentParser(
        description="Carbon Emissions Analysis: EV vs ICE Fleet Comparison"
    )
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per method")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    set_global_seed(args.seed)
    config = load_base_config()
    emissions_calc = EmissionsCalculator()

    # Import dispatchers
    from src.baselines.nearest_vehicle import NearestVehicleDispatcher
    from src.baselines.greedy_dispatch import GreedyDispatcher

    dispatchers = {
        "Nearest Vehicle": NearestVehicleDispatcher(),
        "Greedy Dispatch": GreedyDispatcher(),
    }

    # Try to import OR-Tools (may fail on some setups)
    try:
        from src.baselines.ortools_vrp import ORToolsVRPDispatcher
        dispatchers["OR-Tools CP-SAT"] = ORToolsVRPDispatcher(time_limit_ms=500)
    except Exception:
        print("[WARN] OR-Tools not available, skipping CP-SAT baseline.")

    print("=" * 80)
    print("  CARBON EMISSIONS ANALYSIS: Electric Fleet vs ICE Baseline")
    print("=" * 80)
    print(f"  Episodes per method: {args.episodes}")
    print(f"  Simulation duration: 24 hours (1440 min)")
    print(f"  EV emission factor:  {emissions_calc.ev_co2_per_kwh} kg CO₂/kWh")
    print(f"  ICE emission factor: {emissions_calc.ice_co2_per_km} kg CO₂/km")
    print("=" * 80)

    results: list[dict[str, Any]] = []

    for method_name, dispatcher in dispatchers.items():
        print(f"\n{'─' * 60}")
        print(f"  Evaluating: {method_name}")
        print(f"{'─' * 60}")

        ep_results = []
        for ep in range(args.episodes):
            ep_seed = args.seed + ep
            env = DynamicFleetEnv(config)
            metrics = run_episode(env, dispatcher, ep_seed)
            ep_results.append(metrics)
            print(f"  Episode {ep+1}/{args.episodes} | "
                  f"Delivered: {metrics['delivered']}/{metrics['total_requests']} | "
                  f"Energy: {metrics['total_energy_kwh']:.1f} kWh | "
                  f"Distance: {metrics['total_distance_km']:.1f} km")

        # Aggregate across episodes
        avg_distance = float(np.mean([r["total_distance_km"] for r in ep_results]))
        avg_energy = float(np.mean([r["total_energy_kwh"] for r in ep_results]))
        avg_completion = float(np.mean([r["completion_rate"] for r in ep_results]))
        avg_sla = float(np.mean([r["sla_compliance"] for r in ep_results]))
        avg_turnaround = float(np.mean([r["avg_turnaround_min"] for r in ep_results]))
        avg_fuel = float(np.mean([r["total_fuel"] for r in ep_results]))
        avg_soc = float(np.mean([r["avg_final_battery_soc"] for r in ep_results]))

        # Calculate emissions
        emissions = emissions_calc.calculate_savings(avg_distance, avg_energy)

        result_row = {
            "method": method_name,
            "completion_rate": round(avg_completion * 100, 1),
            "sla_compliance": round(avg_sla * 100, 1),
            "avg_turnaround_min": round(avg_turnaround, 1),
            "total_distance_km": round(avg_distance, 1),
            "total_fuel": round(avg_fuel, 1),
            "total_energy_kwh": round(avg_energy, 1),
            "avg_final_battery_soc": round(avg_soc * 100, 1),
            "ev_co2_kg": round(emissions.ev_co2_kg, 2),
            "ice_co2_kg": round(emissions.ice_co2_kg, 2),
            "co2_saved_kg": round(emissions.co2_saved_kg, 2),
            "co2_reduction_pct": round(emissions.co2_reduction_pct, 1),
        }
        results.append(result_row)

    # Print comparison table
    print("\n\n")
    print("=" * 100)
    print("  CARBON EMISSIONS COMPARISON TABLE")
    print("=" * 100)

    header = (
        f"{'Method':<20} | {'Completion':>10} | {'SLA':>8} | {'Distance':>10} | "
        f"{'Energy':>10} | {'EV CO₂':>9} | {'ICE CO₂':>9} | {'Saved':>9} | {'Reduction':>10}"
    )
    print(header)
    print("─" * 100)

    for r in results:
        row = (
            f"{r['method']:<20} | "
            f"{r['completion_rate']:>9.1f}% | "
            f"{r['sla_compliance']:>6.1f}% | "
            f"{r['total_distance_km']:>8.1f}km | "
            f"{r['total_energy_kwh']:>8.1f}kWh | "
            f"{r['ev_co2_kg']:>7.2f}kg | "
            f"{r['ice_co2_kg']:>7.2f}kg | "
            f"{r['co2_saved_kg']:>7.2f}kg | "
            f"{r['co2_reduction_pct']:>8.1f}%"
        )
        print(row)

    # Save to CSV
    output_dir = Path("artifacts/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "carbon_comparison.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n  Results saved to: {csv_path}")

    # Generate bar chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plots_dir = Path("artifacts/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)

        methods = [r["method"] for r in results]
        ev_co2 = [r["ev_co2_kg"] for r in results]
        ice_co2 = [r["ice_co2_kg"] for r in results]

        x = np.arange(len(methods))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        bars_ice = ax.bar(x - width/2, ice_co2, width, label="ICE Fleet (Diesel/Petrol)",
                          color="#e74c3c", alpha=0.85, edgecolor="white")
        bars_ev = ax.bar(x + width/2, ev_co2, width, label="EV Fleet (Electric)",
                         color="#2ecc71", alpha=0.85, edgecolor="white")

        ax.set_xlabel("Dispatch Method", fontsize=12, fontweight="bold")
        ax.set_ylabel("CO₂ Emissions (kg)", fontsize=12, fontweight="bold")
        ax.set_title("Carbon Emissions: EV Fleet vs ICE Baseline\n(24-Hour Urban Logistics Simulation)",
                     fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=10)
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)

        # Add savings annotation
        for i, r in enumerate(results):
            savings_text = f"-{r['co2_reduction_pct']:.0f}%"
            ax.annotate(
                savings_text,
                xy=(x[i] + width/2, ev_co2[i]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=10,
                fontweight="bold",
                color="#27ae60",
            )

        plt.tight_layout()
        plot_path = plots_dir / "carbon_savings.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Chart saved to: {plot_path}")

    except ImportError:
        print("  [WARN] matplotlib not available, skipping chart generation.")

    print("\n" + "=" * 80)
    print("  Analysis complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
