"""Pydantic schemas for the FastAPI serving layer."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TrafficStateEnum(str, Enum):
    """Traffic state enum for API requests."""
    LOW_TRAFFIC = "LOW_TRAFFIC"
    NORMAL_TRAFFIC = "NORMAL_TRAFFIC"
    HEAVY_TRAFFIC = "HEAVY_TRAFFIC"
    CONGESTION = "CONGESTION"


class VehicleSchema(BaseModel):
    """Vehicle state in an API request."""
    vehicle_id: int
    current_location: int
    capacity: int = 10
    current_load: int = 0
    status: str = "IDLE"
    fuel_remaining: float = 100.0


class RequestSchema(BaseModel):
    """Delivery request in an API request."""
    request_id: int
    pickup_location: int
    dropoff_location: int
    deadline_minutes: float = 60.0
    priority: int = Field(0, ge=0, le=2, description="0=LOW, 1=MEDIUM, 2=HIGH")
    package_size: int = Field(1, ge=1)


class DispatchRequest(BaseModel):
    """Request body for the /dispatch endpoint."""
    vehicles: list[VehicleSchema]
    pending_requests: list[RequestSchema]
    traffic_state: TrafficStateEnum = TrafficStateEnum.NORMAL_TRAFFIC
    current_time: float = Field(0.0, description="Current simulation time in minutes")


class DispatchResponse(BaseModel):
    """Response body for the /dispatch endpoint."""
    selected_request_id: Optional[int] = None
    selected_vehicle_id: Optional[int] = None
    decision_source: str = "PPO"
    latency_ms: float = 0.0
    action: int = 0
    is_noop: bool = False


class SimulateRequest(BaseModel):
    """Request body for the /simulate endpoint."""
    method: str = Field("greedy", description="Dispatch method: nearest, greedy, ortools, ppo, ppo_mcts")
    n_episodes: int = Field(1, ge=1, le=10)
    seed: int = 42
    config_overrides: Optional[dict] = None


class SimulateResponse(BaseModel):
    """Response body for the /simulate endpoint."""
    method: str
    episodes_completed: int
    metrics: dict


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""
    status: str = "healthy"
    model_loaded: bool = False
    version: str = "1.0.0"


class MetricsResponse(BaseModel):
    """Response body for the /metrics endpoint."""
    total_dispatches: int = 0
    avg_latency_ms: float = 0.0
    model_loaded: bool = False
    uptime_seconds: float = 0.0
