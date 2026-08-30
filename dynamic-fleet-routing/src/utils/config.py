"""Configuration loader for the fleet routing system.

Provides YAML configuration loading with defaults, overrides,
and nested access support.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml


def load_config(config_path: str, overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Load a YAML configuration file with optional overrides.

    Args:
        config_path: Path to the YAML configuration file.
        overrides: Dictionary of override values to merge.

    Returns:
        Merged configuration dictionary.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f) or {}

    if overrides:
        config = deep_merge(config, overrides)

    return config


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence.

    Args:
        base: Base dictionary.
        override: Override dictionary (values take precedence).

    Returns:
        Merged dictionary.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_base_config(overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Load the base configuration from the default location.

    Searches for configs/base.yaml relative to the project root.

    Args:
        overrides: Optional override values.

    Returns:
        Base configuration dictionary.
    """
    # Find project root by looking for configs directory
    search_dirs = [
        Path.cwd(),
        Path(__file__).parent.parent.parent,  # src/utils -> project root
    ]

    for search_dir in search_dirs:
        config_path = search_dir / "configs" / "base.yaml"
        if config_path.exists():
            return load_config(str(config_path), overrides)

    # Return defaults if no config file found
    return _default_config(overrides)


def _default_config(overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Return default configuration values.

    Args:
        overrides: Optional override values.

    Returns:
        Default configuration dictionary.
    """
    defaults: dict[str, Any] = {
        "simulation": {
            "num_nodes": 50,
            "num_vehicles": 5,
            "requests_per_day": 100,
            "simulation_duration": 1440,
            "time_step_minutes": 1,
            "seed": 42,
        },
        "city": {
            "grid_size": 10.0,
            "edge_density": 0.15,
            "min_edge_weight": 1.0,
            "max_edge_weight": 5.0,
            "base_speed_kmh": 30.0,
        },
        "traffic": {
            "rush_hour_windows": [
                {"start": 480, "end": 600, "multiplier": 2.0},
                {"start": 1020, "end": 1200, "multiplier": 2.5},
            ],
            "congestion_probability": 0.05,
            "noise_std": 0.1,
            "traffic_update_interval": 30,
        },
        "vehicles": {
            "capacity": 10,
            "fuel_consumption_per_km": 0.08,
            "service_time_minutes": 5,
            "initial_fuel": 100.0,
        },
        "requests": {
            "lambda_rate": 5,
            "arrival_interval": 30,
            "min_deadline_minutes": 30,
            "max_deadline_minutes": 120,
            "min_package_size": 1,
            "max_package_size": 3,
            "priority_weights": [0.6, 0.3, 0.1],
        },
        "observation": {
            "top_k_requests": 20,
            "max_vehicles": 10,
        },
        "reward": {
            "delivery_reward": 10.0,
            "travel_penalty": 0.1,
            "fuel_penalty": 0.5,
            "sla_violation_penalty": 20.0,
            "idle_penalty": 0.05,
            "utilization_reward": 1.0,
            "expiry_penalty": 15.0,
            "normalize": True,
            "normalization_window": 100,
        },
    }

    if overrides:
        defaults = deep_merge(defaults, overrides)

    return defaults


def get_nested(config: dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Get a nested value from a config dict using dot notation.

    Args:
        config: Configuration dictionary.
        key_path: Dot-separated key path (e.g., 'simulation.num_vehicles').
        default: Default value if key not found.

    Returns:
        The value at the key path, or default if not found.
    """
    keys = key_path.split(".")
    current = config
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current
