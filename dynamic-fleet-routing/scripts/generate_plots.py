"""Generate publication-ready comparison charts and visual plots.

Reads generated benchmark CSVs and dataset logs from artifacts/
and saves styled figures in artifacts/plots/.

Usage:
    python scripts/generate_plots.py
"""

from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Styling defaults
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})

OUTPUT_DIR = Path("artifacts/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR = Path("artifacts/metrics")
DATASETS_DIR = Path("artifacts/datasets")


def plot_latency_benchmark():
    """Plot latency comparison with SLA line."""
    csv_path = METRICS_DIR / "latency_benchmark.csv"
    if not csv_path.exists():
        print(f"Skipping latency plot: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(df))
    width = 0.2

    ax.bar(x - 1.5 * width, df["p50"], width, label="P50 (Median)", color="#2b5c8f")
    ax.bar(x - 0.5 * width, df["mean"], width, label="Mean", color="#4682b4")
    ax.bar(x + 0.5 * width, df["p95"], width, label="P95", color="#e67e22")
    ax.bar(x + 1.5 * width, df["p99"], width, label="P99", color="#c0392b")

    # 45ms SLA threshold line
    ax.axhline(45.0, color="#d35400", linestyle="--", linewidth=1.5, label="45ms SLA Budget")

    ax.set_xticks(x)
    ax.set_xticklabels(df["name"], rotation=15, ha="right")
    ax.set_ylabel("Latency (ms, log scale)")
    ax.set_yscale("log")
    ax.set_title("Decision Inference Latency by Dispatch Method (Log Scale)")
    ax.legend(frameon=True, facecolor="white")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_file = OUTPUT_DIR / "latency_comparison.png"
    plt.savefig(plot_file)
    plt.close()
    print(f"Saved {plot_file}")


def plot_turnaround_distribution():
    """Plot request turnaround time distribution."""
    csv_path = DATASETS_DIR / "requests.csv"
    if not csv_path.exists():
        print(f"Skipping turnaround plot: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    delivered = df[df["status"] == "DELIVERED"].copy()
    if delivered.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(delivered["turnaround_time"], kde=True, bins=25, ax=ax, color="#16a085")
    ax.set_xlabel("Turnaround Time (minutes)")
    ax.set_ylabel("Number of Completed Requests")
    ax.set_title("Dynamic Request Turnaround Time Distribution")
    mean_tt = delivered["turnaround_time"].mean()
    median_tt = delivered["turnaround_time"].median()
    ax.axvline(mean_tt, color="#c0392b", linestyle="--", label=f"Mean: {mean_tt:.1f} min")
    ax.axvline(median_tt, color="#2980b9", linestyle=":", label=f"Median: {median_tt:.1f} min")
    ax.legend()

    plt.tight_layout()
    plot_file = OUTPUT_DIR / "turnaround_distribution.png"
    plt.savefig(plot_file)
    plt.close()
    print(f"Saved {plot_file}")


def plot_simulation_dynamics():
    """Plot step-by-step pending requests and fleet utilization."""
    csv_path = DATASETS_DIR / "simulation_steps.csv"
    if not csv_path.exists():
        print(f"Skipping dynamics plot: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(df["time"], df["pending_requests"], color="#8e44ad", linewidth=1.5)
    ax1.set_ylabel("Pending Requests")
    ax1.set_title("Simulation Dynamics: Request Queue and Fleet Utilization Over Time")
    ax1.grid(True, alpha=0.5)

    ax2.plot(df["time"], df["fleet_utilization"] * 100, color="#27ae60", linewidth=1.5)
    ax2.set_ylabel("Fleet Utilization (%)")
    ax2.set_xlabel("Simulation Time (minutes)")
    ax2.grid(True, alpha=0.5)

    plt.tight_layout()
    plot_file = OUTPUT_DIR / "simulation_dynamics.png"
    plt.savefig(plot_file)
    plt.close()
    print(f"Saved {plot_file}")


def plot_experiment_ablation():
    """Plot ablation comparison if results are available."""
    csv_path = METRICS_DIR / "experiment_results.csv"
    if not csv_path.exists():
        print(f"Skipping experiment plot: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    base_df = df[df["experiment"] == "baseline_comparison"].copy()

    if not base_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        methods = base_df["method"]
        completion = base_df["completion_rate_mean"] * 100
        sla = base_df["sla_compliance_rate_mean"] * 100

        x = np.arange(len(methods))
        width = 0.35

        ax.bar(x - width/2, completion, width, label="Completion Rate (%)", color="#3498db")
        ax.bar(x + width/2, sla, width, label="SLA Compliance Rate (%)", color="#2ecc71")

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha="right")
        ax.set_ylabel("Percentage (%)")
        ax.set_title("Baseline Comparison: Completion Rate and SLA Compliance")
        ax.set_ylim(0, 105)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.4)

        plt.tight_layout()
        plot_file = OUTPUT_DIR / "baseline_comparison.png"
        plt.savefig(plot_file)
        plt.close()
        print(f"Saved {plot_file}")


def main():
    print("Generating figures in artifacts/plots/...")
    plot_latency_benchmark()
    plot_turnaround_distribution()
    plot_simulation_dynamics()
    plot_experiment_ablation()
    print("All plots generated successfully.")


if __name__ == "__main__":
    main()
