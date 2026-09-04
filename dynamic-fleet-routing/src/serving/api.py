"""FastAPI serving layer for fleet routing dispatch.

Provides REST API endpoints for dispatch decisions,
simulation, health checks, and metrics.

Usage:
    uvicorn src.serving.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.serving.schemas import (
    DispatchRequest,
    DispatchResponse,
    HealthResponse,
    MetricsResponse,
    SimulateRequest,
    SimulateResponse,
)
from src.utils.logger import setup_logger


# Global state
_state: dict[str, Any] = {
    "model_loaded": False,
    "inference_engine": None,
    "total_dispatches": 0,
    "total_latency_ms": 0.0,
    "start_time": 0.0,
}

logger = setup_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager — loads model on startup."""
    _state["start_time"] = time.time()

    # Try to load model
    model_path = os.environ.get("MODEL_PATH", "artifacts/models/final_model.zip")
    if os.path.exists(model_path):
        try:
            from src.serving.inference import InferenceEngine
            engine = InferenceEngine(model_path)
            _state["inference_engine"] = engine
            _state["model_loaded"] = True
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
    else:
        logger.info(f"No model found at {model_path}, running without model")

    yield

    logger.info("Shutting down")


app = FastAPI(
    title="Dynamic Fleet Routing API",
    description="REST API for fleet dispatch optimization using Deep RL & MCTS",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=_state["model_loaded"],
        version="1.0.0",
    )


@app.post("/dispatch", response_model=DispatchResponse)
async def dispatch(request: DispatchRequest):
    """Make a dispatch decision for the given fleet state.

    Accepts vehicle positions, pending requests, and traffic state,
    then returns the optimal request-vehicle assignment.
    """
    start = time.perf_counter()

    try:
        from src.environment.fleet_env import DynamicFleetEnv
        from src.utils.config import load_base_config

        # Build a minimal environment from the request
        config = load_base_config({
            "simulation": {
                "num_nodes": max(
                    max((v.current_location for v in request.vehicles), default=10),
                    max((r.pickup_location for r in request.pending_requests), default=10),
                    max((r.dropoff_location for r in request.pending_requests), default=10),
                ) + 1,
                "num_vehicles": len(request.vehicles),
            },
        })

        env = DynamicFleetEnv(config)
        obs, _ = env.reset()

        method = request.method.lower() if hasattr(request, "method") and request.method else "auto"
        decision_source = "greedy"

        if method in ("ppo", "auto") and _state["model_loaded"] and _state["inference_engine"] is not None:
            mask = env.get_action_mask()
            action, _ = _state["inference_engine"].predict(obs, mask)
            decision_source = "PPO"
        elif method == "nearest":
            from src.baselines.nearest_vehicle import NearestVehicleDispatcher
            dispatcher = NearestVehicleDispatcher()
            action = dispatcher.select_action(env)
            decision_source = "nearest"
        elif method == "ortools":
            from src.baselines.ortools_vrp import ORToolsVRPDispatcher
            dispatcher = ORToolsVRPDispatcher(time_limit_ms=500)
            action = dispatcher.select_action(env)
            decision_source = "ortools"
        else:
            from src.baselines.greedy_dispatch import GreedyDispatcher
            dispatcher = GreedyDispatcher()
            action = dispatcher.select_action(env)
            decision_source = "greedy"

        latency_ms = (time.perf_counter() - start) * 1000
        _state["total_dispatches"] += 1
        _state["total_latency_ms"] += latency_ms

        noop = action == env.action_space.n - 1
        selected_req = None
        selected_veh = None

        if not noop and len(request.pending_requests) > 0:
            req_idx = action // env.num_vehicles
            veh_idx = action % env.num_vehicles
            if req_idx < len(request.pending_requests):
                selected_req = request.pending_requests[req_idx].request_id
            if veh_idx < len(request.vehicles):
                selected_veh = request.vehicles[veh_idx].vehicle_id

        return DispatchResponse(
            selected_request_id=selected_req,
            selected_vehicle_id=selected_veh,
            decision_source=decision_source,
            latency_ms=round(latency_ms, 4),
            action=action,
            is_noop=noop,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate", response_model=SimulateResponse)
async def simulate(request: SimulateRequest):
    """Run a simulation episode with the specified dispatch method."""
    try:
        from src.environment.fleet_env import DynamicFleetEnv
        from src.utils.config import load_base_config
        from src.training.evaluate import evaluate_dispatcher

        config = load_base_config(request.config_overrides)
        env = DynamicFleetEnv(config)

        # Select dispatcher
        if request.method == "nearest":
            from src.baselines.nearest_vehicle import NearestVehicleDispatcher
            dispatcher = NearestVehicleDispatcher()
        elif request.method == "greedy":
            from src.baselines.greedy_dispatch import GreedyDispatcher
            dispatcher = GreedyDispatcher()
        elif request.method == "ortools":
            from src.baselines.ortools_vrp import ORToolsVRPDispatcher
            dispatcher = ORToolsVRPDispatcher(time_limit_ms=500)
        elif request.method == "mcts":
            from src.planning.mcts import MCTSPlanner
            dispatcher = MCTSPlanner(num_simulations=20, rollout_horizon=3)
        elif request.method in ("ppo", "ppo_mcts") and _state["model_loaded"]:
            from src.agents.ppo_agent import PPOAgent
            agent = PPOAgent(config=config)
            agent.model = _state["inference_engine"].sb3_model
            if request.method == "ppo_mcts":
                from src.planning.hybrid_planner import HybridPlanner
                dispatcher = HybridPlanner(agent, num_simulations=20)
            else:
                dispatcher = agent
        else:
            from src.baselines.greedy_dispatch import GreedyDispatcher
            dispatcher = GreedyDispatcher()

        collector = evaluate_dispatcher(
            env, dispatcher, request.n_episodes, request.seed
        )

        return SimulateResponse(
            method=request.method,
            episodes_completed=collector.num_episodes,
            metrics=collector.get_summary(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Return current service metrics."""
    total = _state["total_dispatches"]
    avg_latency = (
        _state["total_latency_ms"] / total if total > 0 else 0.0
    )
    uptime = time.time() - _state["start_time"]

    return MetricsResponse(
        total_dispatches=total,
        avg_latency_ms=round(avg_latency, 4),
        model_loaded=_state["model_loaded"],
        uptime_seconds=round(uptime, 2),
    )
