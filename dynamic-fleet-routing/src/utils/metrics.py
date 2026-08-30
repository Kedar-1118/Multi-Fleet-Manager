"""Metrics collection and reporting for the fleet routing system.

Provides a MetricsCollector class for accumulating simulation metrics
and exporting results to CSV and summary formats.
"""

from __future__ import annotations

import csv
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class EpisodeMetrics:
    """Metrics collected from a single simulation episode.

    Attributes:
        total_requests: Total requests generated.
        delivered_requests: Number of successfully delivered requests.
        expired_requests: Number of expired requests.
        total_distance: Total fleet distance traveled (km).
        total_fuel_consumed: Total fuel consumed by fleet.
        avg_turnaround_time: Average time from request to delivery (minutes).
        sla_compliance_rate: Fraction of deliveries within deadline.
        fleet_utilization: Average vehicle utilization fraction.
        total_reward: Cumulative reward for the episode.
        avg_inference_latency_ms: Average decision latency in milliseconds.
        requests_per_second: Throughput of request processing.
        completion_rate: Fraction of requests delivered.
    """
    total_requests: int = 0
    delivered_requests: int = 0
    expired_requests: int = 0
    total_distance: float = 0.0
    total_fuel_consumed: float = 0.0
    avg_turnaround_time: float = 0.0
    sla_compliance_rate: float = 0.0
    fleet_utilization: float = 0.0
    total_reward: float = 0.0
    avg_inference_latency_ms: float = 0.0
    requests_per_second: float = 0.0
    completion_rate: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to a flat dictionary."""
        return {
            "total_requests": self.total_requests,
            "delivered_requests": self.delivered_requests,
            "expired_requests": self.expired_requests,
            "total_distance": round(self.total_distance, 3),
            "total_fuel_consumed": round(self.total_fuel_consumed, 3),
            "avg_turnaround_time": round(self.avg_turnaround_time, 3),
            "sla_compliance_rate": round(self.sla_compliance_rate, 4),
            "fleet_utilization": round(self.fleet_utilization, 4),
            "total_reward": round(self.total_reward, 3),
            "avg_inference_latency_ms": round(self.avg_inference_latency_ms, 4),
            "requests_per_second": round(self.requests_per_second, 2),
            "completion_rate": round(self.completion_rate, 4),
        }


class MetricsCollector:
    """Collects and aggregates metrics across simulation episodes.

    Tracks per-step and per-episode metrics, supports CSV export,
    and provides summary statistics.
    """

    def __init__(self) -> None:
        """Initialize the metrics collector."""
        self._episode_metrics: list[EpisodeMetrics] = []
        self._current_episode: dict[str, list[float]] = defaultdict(list)
        self._reward_components: dict[str, list[float]] = defaultdict(list)
        self._latencies: list[float] = []
        self._turnaround_times: list[float] = []
        self._sla_results: list[bool] = []

    def record_step(
        self,
        reward: float,
        reward_components: Optional[dict[str, float]] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        """Record metrics from a single simulation step.

        Args:
            reward: Total reward for this step.
            reward_components: Breakdown of reward by component.
            latency_ms: Decision latency in milliseconds.
        """
        self._current_episode["rewards"].append(reward)

        if reward_components:
            for key, value in reward_components.items():
                self._reward_components[key].append(value)

        if latency_ms is not None:
            self._latencies.append(latency_ms)

    def record_delivery(self, turnaround_time: float, on_time: bool) -> None:
        """Record a completed delivery.

        Args:
            turnaround_time: Time from request to delivery.
            on_time: Whether delivery met the SLA deadline.
        """
        self._turnaround_times.append(turnaround_time)
        self._sla_results.append(on_time)

    def finish_episode(self, episode_metrics: EpisodeMetrics) -> None:
        """Finalize and store the current episode's metrics.

        Args:
            episode_metrics: Completed episode metrics.
        """
        # Populate latency stats from collected data
        if self._latencies:
            episode_metrics.avg_inference_latency_ms = float(np.mean(self._latencies))
        if self._turnaround_times:
            episode_metrics.avg_turnaround_time = float(np.mean(self._turnaround_times))
        if self._sla_results:
            episode_metrics.sla_compliance_rate = float(np.mean(self._sla_results))
        if episode_metrics.total_requests > 0:
            episode_metrics.completion_rate = (
                episode_metrics.delivered_requests / episode_metrics.total_requests
            )

        self._episode_metrics.append(episode_metrics)
        self._reset_episode()

    def _reset_episode(self) -> None:
        """Reset per-episode accumulators."""
        self._current_episode.clear()
        self._reward_components.clear()
        self._latencies.clear()
        self._turnaround_times.clear()
        self._sla_results.clear()

    def get_summary(self) -> dict[str, Any]:
        """Get aggregate summary statistics across all episodes.

        Returns:
            Dictionary with mean and std for each metric.
        """
        if not self._episode_metrics:
            return {}

        metrics_dicts = [m.to_dict() for m in self._episode_metrics]
        summary: dict[str, Any] = {}

        for key in metrics_dicts[0]:
            values = [m[key] for m in metrics_dicts]
            summary[f"{key}_mean"] = round(float(np.mean(values)), 4)
            summary[f"{key}_std"] = round(float(np.std(values)), 4)

        summary["num_episodes"] = len(self._episode_metrics)
        return summary

    def export_csv(self, filepath: str) -> None:
        """Export all episode metrics to a CSV file.

        Args:
            filepath: Output CSV file path.
        """
        if not self._episode_metrics:
            return

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        metrics_dicts = [m.to_dict() for m in self._episode_metrics]
        fieldnames = list(metrics_dicts[0].keys())

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics_dicts)

    def get_latest(self) -> Optional[EpisodeMetrics]:
        """Get the most recently completed episode metrics."""
        if self._episode_metrics:
            return self._episode_metrics[-1]
        return None

    @property
    def num_episodes(self) -> int:
        """Return the number of completed episodes."""
        return len(self._episode_metrics)


class LatencyTracker:
    """Tracks inference latency with high-resolution timing.

    Provides percentile statistics for benchmarking.
    """

    def __init__(self) -> None:
        """Initialize the latency tracker."""
        self._latencies: list[float] = []
        self._start_time: Optional[float] = None

    def start(self) -> None:
        """Start timing a decision."""
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop timing and record the latency.

        Returns:
            Latency in milliseconds.

        Raises:
            RuntimeError: If start() was not called.
        """
        if self._start_time is None:
            raise RuntimeError("LatencyTracker.start() was not called")
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        self._latencies.append(elapsed_ms)
        self._start_time = None
        return elapsed_ms

    def get_stats(self) -> dict[str, float]:
        """Get latency statistics.

        Returns:
            Dictionary with mean, p50, p95, p99 latencies in ms.
        """
        if not self._latencies:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}

        arr = np.array(self._latencies)
        return {
            "mean": round(float(np.mean(arr)), 4),
            "p50": round(float(np.percentile(arr, 50)), 4),
            "p95": round(float(np.percentile(arr, 95)), 4),
            "p99": round(float(np.percentile(arr, 99)), 4),
            "count": len(self._latencies),
        }

    def reset(self) -> None:
        """Clear all recorded latencies."""
        self._latencies.clear()
        self._start_time = None
