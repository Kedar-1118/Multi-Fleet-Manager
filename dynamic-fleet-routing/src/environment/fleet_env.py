"""Dynamic Fleet Routing Gymnasium Environment.

Implements an event-driven fleet simulation where multiple vehicles
serve pickup/delivery requests under stochastic traffic conditions.
Supports action masking, state cloning for MCTS, and fixed-size
observations compatible with Stable-Baselines3.
"""

from __future__ import annotations

import heapq
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.environment.city_graph import CityGraph, CityGraphConfig
from src.environment.request import Priority, Request, RequestStatus
from src.environment.request_generator import RequestGenerator, RequestGeneratorConfig
from src.environment.reward import RewardBreakdown, RewardCalculator, RewardConfig
from src.environment.traffic_model import TrafficModel, TrafficConfig, TrafficState
from src.environment.vehicle import Vehicle, VehicleStatus


class EventType(Enum):
    """Types of simulation events."""
    NEW_REQUEST = "NEW_REQUEST"
    VEHICLE_ARRIVAL = "VEHICLE_ARRIVAL"
    PICKUP_COMPLETE = "PICKUP_COMPLETE"
    DROPOFF_COMPLETE = "DROPOFF_COMPLETE"
    TRAFFIC_CHANGE = "TRAFFIC_CHANGE"
    REQUEST_EXPIRATION = "REQUEST_EXPIRATION"


@dataclass
class SimEvent:
    """A simulation event in the priority queue.

    Attributes:
        time: Simulation time when event occurs.
        event_type: Type of the event.
        data: Additional event data (vehicle_id, request_id, etc.).
    """
    time: float
    event_type: EventType
    data: dict[str, Any]

    def __lt__(self, other: SimEvent) -> bool:
        return self.time < other.time

    def __le__(self, other: SimEvent) -> bool:
        return self.time <= other.time


class DynamicFleetEnv(gym.Env):
    """Custom Gymnasium environment for dynamic fleet dispatch.

    An event-driven simulation where an RL agent assigns delivery
    requests to vehicles. The environment models stochastic traffic,
    dynamic request arrivals, SLA constraints, and fuel consumption.

    Action Space:
        Discrete: action = request_index * num_vehicles + vehicle_index
        Plus a NO-OP action (wait) at the last index.

    Observation Space:
        Box: Fixed-size vector containing global features, vehicle
        features, and top-K request features (normalized).

    Args:
        config: Configuration dictionary (from configs/base.yaml).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        """Initialize the fleet routing environment.

        Args:
            config: Configuration dictionary. Uses defaults if None.
        """
        super().__init__()

        if config is None:
            from src.utils.config import _default_config
            config = _default_config()

        self._config = config
        sim_cfg = config.get("simulation", {})
        veh_cfg = config.get("vehicles", {})
        obs_cfg = config.get("observation", {})
        req_cfg = config.get("requests", {})
        city_cfg = config.get("city", {})
        traffic_cfg = config.get("traffic", {})
        reward_cfg = config.get("reward", {})

        # Core parameters
        self._num_vehicles: int = sim_cfg.get("num_vehicles", 5)
        self._num_nodes: int = sim_cfg.get("num_nodes", 50)
        self._sim_duration: float = float(sim_cfg.get("simulation_duration", 1440))
        self._seed: int = sim_cfg.get("seed", 42)
        self._top_k_requests: int = obs_cfg.get("top_k_requests", 20)
        self._max_vehicles_obs: int = min(
            obs_cfg.get("max_vehicles", 10), self._num_vehicles
        )
        self._vehicle_capacity: int = veh_cfg.get("capacity", 10)
        self._fuel_rate: float = veh_cfg.get("fuel_consumption_per_km", 0.08)
        self._service_time: float = veh_cfg.get("service_time_minutes", 5.0)
        self._initial_fuel: float = veh_cfg.get("initial_fuel", 100.0)

        # Build city graph
        graph_config = CityGraphConfig(
            num_nodes=self._num_nodes,
            grid_size=city_cfg.get("grid_size", 10.0),
            edge_density=city_cfg.get("edge_density", 0.15),
            base_speed_kmh=city_cfg.get("base_speed_kmh", 30.0),
            seed=self._seed,
        )
        self.city_graph = CityGraph(graph_config)

        # Build traffic model
        rush_windows_raw = traffic_cfg.get("rush_hour_windows", [])
        from src.environment.traffic_model import RushHourWindow
        rush_windows = [
            RushHourWindow(
                start=w.get("start", 480),
                end=w.get("end", 600),
                multiplier=w.get("multiplier", 2.0),
            )
            for w in rush_windows_raw
        ] if rush_windows_raw else None

        t_config = TrafficConfig(
            congestion_probability=traffic_cfg.get("congestion_probability", 0.05),
            noise_std=traffic_cfg.get("noise_std", 0.1),
            traffic_update_interval=traffic_cfg.get("traffic_update_interval", 30.0),
            seed=self._seed,
        )
        if rush_windows:
            t_config.rush_hour_windows = rush_windows
        self.traffic_model = TrafficModel(t_config)

        # Build request generator
        rg_config = RequestGeneratorConfig(
            lambda_rate=req_cfg.get("lambda_rate", 5),
            arrival_interval=req_cfg.get("arrival_interval", 30.0),
            min_deadline_minutes=req_cfg.get("min_deadline_minutes", 30.0),
            max_deadline_minutes=req_cfg.get("max_deadline_minutes", 120.0),
            min_package_size=req_cfg.get("min_package_size", 1),
            max_package_size=req_cfg.get("max_package_size", 3),
            priority_weights=req_cfg.get("priority_weights", [0.6, 0.3, 0.1]),
            seed=self._seed,
        )
        self.request_generator = RequestGenerator(self.city_graph, rg_config)

        # Build reward calculator
        r_config = RewardConfig(
            delivery_reward=reward_cfg.get("delivery_reward", 10.0),
            travel_penalty=reward_cfg.get("travel_penalty", 0.1),
            fuel_penalty=reward_cfg.get("fuel_penalty", 0.5),
            sla_violation_penalty=reward_cfg.get("sla_violation_penalty", 20.0),
            idle_penalty=reward_cfg.get("idle_penalty", 0.05),
            utilization_reward=reward_cfg.get("utilization_reward", 1.0),
            expiry_penalty=reward_cfg.get("expiry_penalty", 15.0),
            normalize=reward_cfg.get("normalize", True),
            normalization_window=reward_cfg.get("normalization_window", 100),
        )
        self.reward_calculator = RewardCalculator(r_config)

        # Random number generator
        self._rng = np.random.RandomState(self._seed)

        # Simulation state — initialized in reset()
        self.vehicles: list[Vehicle] = []
        self.requests: dict[int, Request] = {}
        self.pending_requests: list[int] = []
        self.current_time: float = 0.0
        self._event_queue: list[SimEvent] = []
        self._step_count: int = 0
        self._total_deliveries: int = 0
        self._total_expired: int = 0
        self._total_sla_violations: int = 0
        self._total_on_time: int = 0
        self._cumulative_reward: float = 0.0
        self._last_reward_breakdown: Optional[RewardBreakdown] = None

        # Define action and observation spaces
        # Action = request_idx * num_vehicles + vehicle_idx, plus NO-OP
        self._action_size = self._top_k_requests * self._num_vehicles + 1
        self.action_space = spaces.Discrete(self._action_size)

        # Observation space dimensions
        self._global_feat_size = 6
        self._vehicle_feat_size = 6  # per vehicle
        self._request_feat_size = 5  # per request
        obs_size = (
            self._global_feat_size
            + self._max_vehicles_obs * self._vehicle_feat_size
            + self._top_k_requests * self._request_feat_size
        )
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to initial state.

        Args:
            seed: Optional random seed.
            options: Optional reset options.

        Returns:
            Tuple of (observation, info dictionary).
        """
        super().reset(seed=seed)

        if seed is not None:
            self._rng = np.random.RandomState(seed)
            self._seed = seed

        # Reset simulation state
        self.current_time = 0.0
        self._event_queue = []
        self._step_count = 0
        self._total_deliveries = 0
        self._total_expired = 0
        self._total_sla_violations = 0
        self._total_on_time = 0
        self._cumulative_reward = 0.0
        self.requests = {}
        self.pending_requests = []

        # Reset subsystems
        self.traffic_model.reset(seed=self._seed)
        self.request_generator.reset(seed=self._seed)
        self.reward_calculator.reset()

        # Initialize vehicles at random locations
        self.vehicles = []
        for i in range(self._num_vehicles):
            start_loc = self.city_graph.get_random_node(self._rng)
            v = Vehicle(
                vehicle_id=i,
                current_location=start_loc,
                capacity=self._vehicle_capacity,
                fuel_remaining=self._initial_fuel,
            )
            self.vehicles.append(v)

        # Schedule initial events
        self._schedule_event(0.0, EventType.NEW_REQUEST, {})
        self._schedule_event(
            self.traffic_model.config.traffic_update_interval,
            EventType.TRAFFIC_CHANGE, {},
        )

        # Process initial request generation
        self._process_events_until_decision()

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one dispatch decision step.

        Args:
            action: Encoded dispatch action (request_idx * num_vehicles + vehicle_idx),
                    or action_size - 1 for NO-OP (wait).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        self._step_count += 1
        step_deliveries = 0
        step_distance = 0.0
        step_fuel = 0.0
        step_sla_violations = 0
        step_expired = 0
        step_on_time = 0

        # Decode and validate action
        is_noop = (action == self._action_size - 1)

        if not is_noop:
            request_idx = action // self._num_vehicles
            vehicle_idx = action % self._num_vehicles

            # Validate action
            valid, reason = self._validate_action(request_idx, vehicle_idx)
            if not valid:
                # Invalid action: apply penalty and skip assignment
                breakdown = self.reward_calculator.calculate(
                    sla_violations=1,  # Penalize invalid action
                )
                self._last_reward_breakdown = breakdown
                obs = self._get_observation()
                info = self._get_info()
                info["invalid_action"] = True
                info["invalid_reason"] = reason
                return obs, breakdown.total_reward, False, False, info

            # Execute assignment
            request_id = self.pending_requests[request_idx]
            request = self.requests[request_id]
            vehicle = self.vehicles[vehicle_idx]

            # Assign request to vehicle
            request.assign_to(vehicle.vehicle_id, self.current_time)
            vehicle.set_status(VehicleStatus.MOVING_TO_PICKUP)
            vehicle.current_request_id = request_id

            # Calculate travel to pickup
            pickup_dist = self.city_graph.get_shortest_distance(
                vehicle.current_location, request.pickup_location
            )
            traffic_mult = self.traffic_model.get_travel_time_multiplier(
                self.current_time, vehicle.current_location, request.pickup_location
            )
            pickup_travel_time = self.city_graph.get_travel_time(
                vehicle.current_location, request.pickup_location, traffic_mult
            )

            # Consume fuel for pickup travel
            vehicle.consume_fuel(pickup_dist, self._fuel_rate)
            step_distance += pickup_dist
            step_fuel += pickup_dist * self._fuel_rate

            # Schedule pickup arrival event
            arrival_time = self.current_time + pickup_travel_time
            self._schedule_event(
                arrival_time,
                EventType.VEHICLE_ARRIVAL,
                {"vehicle_id": vehicle.vehicle_id, "request_id": request_id,
                 "destination": request.pickup_location, "phase": "pickup"},
            )

            # Remove from pending
            self.pending_requests.remove(request_id)

        # Process events until next decision point
        events_result = self._process_events_until_decision()
        step_deliveries += events_result.get("deliveries", 0)
        step_distance += events_result.get("distance", 0.0)
        step_fuel += events_result.get("fuel", 0.0)
        step_expired += events_result.get("expired", 0)
        step_on_time += events_result.get("on_time", 0)
        step_sla_violations += events_result.get("sla_violations", 0)

        # Update totals
        self._total_deliveries += step_deliveries
        self._total_expired += step_expired
        self._total_sla_violations += step_sla_violations
        self._total_on_time += step_on_time

        # Calculate fleet utilization
        active_vehicles = sum(
            1 for v in self.vehicles if v.status != VehicleStatus.IDLE
        )
        fleet_utilization = active_vehicles / max(self._num_vehicles, 1)

        # Calculate idle time
        idle_vehicles = self._num_vehicles - active_vehicles
        idle_time = idle_vehicles * 1.0  # 1 minute per idle vehicle per step

        # Calculate reward
        breakdown = self.reward_calculator.calculate(
            deliveries_completed=step_deliveries,
            distance_traveled=step_distance,
            fuel_consumed=step_fuel,
            sla_violations=step_sla_violations,
            idle_time=idle_time,
            fleet_utilization=fleet_utilization,
            requests_expired=step_expired,
            on_time_deliveries=step_on_time,
        )
        self._last_reward_breakdown = breakdown
        self._cumulative_reward += breakdown.total_reward

        # Check termination
        terminated = self.current_time >= self._sim_duration
        truncated = self._step_count >= 10000  # Safety limit

        obs = self._get_observation()
        info = self._get_info()
        info["reward_breakdown"] = breakdown.to_dict()

        return obs, breakdown.total_reward, terminated, truncated, info

    def _validate_action(self, request_idx: int, vehicle_idx: int) -> tuple[bool, str]:
        """Validate a dispatch action.

        Args:
            request_idx: Index into pending_requests list.
            vehicle_idx: Index into vehicles list.

        Returns:
            Tuple of (is_valid, reason_if_invalid).
        """
        if request_idx >= len(self.pending_requests):
            return False, f"Request index {request_idx} out of range (have {len(self.pending_requests)})"

        if vehicle_idx >= self._num_vehicles:
            return False, f"Vehicle index {vehicle_idx} out of range"

        request_id = self.pending_requests[request_idx]
        request = self.requests[request_id]
        vehicle = self.vehicles[vehicle_idx]

        if request.status != RequestStatus.PENDING:
            return False, f"Request {request_id} is {request.status.value}, not PENDING"

        if request.is_expired_at(self.current_time):
            return False, f"Request {request_id} has expired"

        if not vehicle.can_accept_package(request.package_size):
            return False, (
                f"Vehicle {vehicle_idx} capacity insufficient "
                f"(remaining: {vehicle.capacity_remaining}, need: {request.package_size})"
            )

        if vehicle.status != VehicleStatus.IDLE:
            return False, f"Vehicle {vehicle_idx} is {vehicle.status.value}, not IDLE"

        return True, ""

    def get_action_mask(self) -> np.ndarray:
        """Return a binary mask of valid actions.

        Returns:
            Boolean array of shape (action_size,) where True = valid action.
        """
        mask = np.zeros(self._action_size, dtype=bool)

        # NO-OP is always valid
        mask[-1] = True

        for req_idx, req_id in enumerate(self.pending_requests[:self._top_k_requests]):
            request = self.requests[req_id]
            if request.status != RequestStatus.PENDING:
                continue
            if request.is_expired_at(self.current_time):
                continue

            for veh_idx, vehicle in enumerate(self.vehicles):
                if vehicle.status != VehicleStatus.IDLE:
                    continue
                if not vehicle.can_accept_package(request.package_size):
                    continue
                action = req_idx * self._num_vehicles + veh_idx
                if action < self._action_size - 1:
                    mask[action] = True

        return mask

    def _schedule_event(self, time: float, event_type: EventType, data: dict) -> None:
        """Add an event to the priority queue.

        Args:
            time: Simulation time for the event.
            event_type: Type of event.
            data: Event-specific data.
        """
        event = SimEvent(time=time, event_type=event_type, data=data)
        heapq.heappush(self._event_queue, event)

    def _process_events_until_decision(self) -> dict[str, Any]:
        """Process simulation events until a decision point is reached.

        A decision point occurs when there are pending requests and
        idle vehicles, or when simulation ends.

        Returns:
            Dictionary of accumulated step results.
        """
        result: dict[str, Any] = {
            "deliveries": 0, "distance": 0.0, "fuel": 0.0,
            "expired": 0, "on_time": 0, "sla_violations": 0,
        }

        max_events = 500  # Safety limit per step
        events_processed = 0

        while self._event_queue and events_processed < max_events:
            # Peek at next event
            next_event = self._event_queue[0]

            # Don't process future events beyond simulation end
            if next_event.time > self._sim_duration:
                self.current_time = self._sim_duration
                break

            # Process the event
            event = heapq.heappop(self._event_queue)
            self.current_time = max(self.current_time, event.time)
            events_processed += 1

            event_result = self._handle_event(event)

            # Accumulate results
            result["deliveries"] += event_result.get("deliveries", 0)
            result["distance"] += event_result.get("distance", 0.0)
            result["fuel"] += event_result.get("fuel", 0.0)
            result["expired"] += event_result.get("expired", 0)
            result["on_time"] += event_result.get("on_time", 0)
            result["sla_violations"] += event_result.get("sla_violations", 0)

            # Check if we have a decision point
            has_pending = len(self.pending_requests) > 0
            has_idle = any(v.status == VehicleStatus.IDLE for v in self.vehicles)
            if has_pending and has_idle:
                break

        return result

    def _handle_event(self, event: SimEvent) -> dict[str, Any]:
        """Handle a single simulation event.

        Args:
            event: The event to process.

        Returns:
            Dictionary of event results.
        """
        result: dict[str, Any] = {}

        if event.event_type == EventType.NEW_REQUEST:
            result = self._handle_new_requests(event)

        elif event.event_type == EventType.VEHICLE_ARRIVAL:
            result = self._handle_vehicle_arrival(event)

        elif event.event_type == EventType.PICKUP_COMPLETE:
            result = self._handle_pickup_complete(event)

        elif event.event_type == EventType.DROPOFF_COMPLETE:
            result = self._handle_dropoff_complete(event)

        elif event.event_type == EventType.TRAFFIC_CHANGE:
            self._handle_traffic_change(event)

        elif event.event_type == EventType.REQUEST_EXPIRATION:
            result = self._handle_request_expiration(event)

        return result

    def _handle_new_requests(self, event: SimEvent) -> dict[str, Any]:
        """Generate and add new requests to the simulation."""
        batch = self.request_generator.generate_batch(self.current_time)

        for request in batch:
            self.requests[request.request_id] = request
            self.pending_requests.append(request.request_id)

            # Schedule expiration event
            self._schedule_event(
                request.deadline,
                EventType.REQUEST_EXPIRATION,
                {"request_id": request.request_id},
            )

        # Schedule next request batch
        self.request_generator.advance_arrival_time()
        next_time = self.request_generator.get_next_arrival_time()
        if next_time <= self._sim_duration:
            self._schedule_event(next_time, EventType.NEW_REQUEST, {})

        return {}

    def _handle_vehicle_arrival(self, event: SimEvent) -> dict[str, Any]:
        """Handle a vehicle arriving at a destination."""
        vehicle_id = event.data["vehicle_id"]
        request_id = event.data["request_id"]
        phase = event.data.get("phase", "pickup")
        vehicle = self.vehicles[vehicle_id]

        if phase == "pickup":
            request = self.requests[request_id]
            destination = request.pickup_location
            vehicle.current_location = destination

            # Start servicing (pickup)
            vehicle.set_status(VehicleStatus.SERVICING)
            self._schedule_event(
                self.current_time + self._service_time,
                EventType.PICKUP_COMPLETE,
                {"vehicle_id": vehicle_id, "request_id": request_id},
            )

        elif phase == "dropoff":
            request = self.requests[request_id]
            destination = request.dropoff_location
            vehicle.current_location = destination

            # Start servicing (dropoff)
            vehicle.set_status(VehicleStatus.SERVICING)
            self._schedule_event(
                self.current_time + self._service_time,
                EventType.DROPOFF_COMPLETE,
                {"vehicle_id": vehicle_id, "request_id": request_id},
            )

        return {}

    def _handle_pickup_complete(self, event: SimEvent) -> dict[str, Any]:
        """Handle completion of pickup servicing."""
        vehicle_id = event.data["vehicle_id"]
        request_id = event.data["request_id"]
        vehicle = self.vehicles[vehicle_id]
        request = self.requests[request_id]

        # Load package
        request.mark_picked_up(self.current_time)
        vehicle.load_package(request.package_size, request_id)

        # Start moving to dropoff
        vehicle.set_status(VehicleStatus.MOVING_TO_DROPOFF)

        dropoff_dist = self.city_graph.get_shortest_distance(
            vehicle.current_location, request.dropoff_location
        )
        traffic_mult = self.traffic_model.get_travel_time_multiplier(
            self.current_time, vehicle.current_location, request.dropoff_location
        )
        dropoff_travel_time = self.city_graph.get_travel_time(
            vehicle.current_location, request.dropoff_location, traffic_mult
        )

        vehicle.consume_fuel(dropoff_dist, self._fuel_rate)

        self._schedule_event(
            self.current_time + dropoff_travel_time,
            EventType.VEHICLE_ARRIVAL,
            {"vehicle_id": vehicle_id, "request_id": request_id,
             "destination": request.dropoff_location, "phase": "dropoff"},
        )

        return {"distance": dropoff_dist, "fuel": dropoff_dist * self._fuel_rate}

    def _handle_dropoff_complete(self, event: SimEvent) -> dict[str, Any]:
        """Handle completion of dropoff servicing."""
        vehicle_id = event.data["vehicle_id"]
        request_id = event.data["request_id"]
        vehicle = self.vehicles[vehicle_id]
        request = self.requests[request_id]

        # Unload package
        request.mark_delivered(self.current_time)
        vehicle.unload_package(request.package_size, request_id)
        vehicle.set_status(VehicleStatus.IDLE)
        vehicle.current_request_id = None

        # Track SLA
        on_time = request.was_delivered_on_time()
        sla_violations = 0
        on_time_count = 0
        if on_time:
            on_time_count = 1
        else:
            sla_violations = 1

        return {
            "deliveries": 1,
            "on_time": on_time_count,
            "sla_violations": sla_violations,
        }

    def _handle_traffic_change(self, event: SimEvent) -> None:
        """Handle periodic traffic state update."""
        self.traffic_model.update(self.current_time)

        # Schedule next traffic update
        next_time = self.current_time + self.traffic_model.config.traffic_update_interval
        if next_time <= self._sim_duration:
            self._schedule_event(next_time, EventType.TRAFFIC_CHANGE, {})

    def _handle_request_expiration(self, event: SimEvent) -> dict[str, Any]:
        """Handle a request reaching its deadline."""
        request_id = event.data["request_id"]
        if request_id not in self.requests:
            return {}

        request = self.requests[request_id]

        # Only expire if still pending or assigned (not picked up/delivered)
        if request.status in {RequestStatus.PENDING, RequestStatus.ASSIGNED}:
            # If assigned, free the vehicle
            if request.status == RequestStatus.ASSIGNED and request.assigned_vehicle_id is not None:
                vehicle = self.vehicles[request.assigned_vehicle_id]
                if vehicle.current_request_id == request_id:
                    vehicle.set_status(VehicleStatus.IDLE)
                    vehicle.current_request_id = None

            request.mark_expired()

            if request_id in self.pending_requests:
                self.pending_requests.remove(request_id)

            return {"expired": 1}

        return {}

    def _get_observation(self) -> np.ndarray:
        """Build the fixed-size observation vector.

        Structure:
            [global_features | vehicle_features | request_features]

        Global features (6):
            - Normalized simulation time
            - Number of pending requests (normalized)
            - Number of active vehicles (normalized)
            - Average fleet utilization
            - Traffic level (normalized)
            - SLA violation rate

        Vehicle features (6 per vehicle):
            - Normalized x position
            - Normalized y position
            - Current load / capacity
            - Status encoding (0-1)
            - Estimated remaining route time (normalized)
            - Fuel remaining (normalized)

        Request features (5 per request):
            - Normalized pickup x
            - Normalized pickup y
            - Normalized dropoff x, y combined as distance
            - Remaining SLA time (normalized)
            - Priority (normalized)

        Returns:
            Observation array of shape (obs_size,).
        """
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        idx = 0

        # === Global features ===
        obs[idx] = self.current_time / self._sim_duration  # normalized time
        idx += 1
        obs[idx] = min(len(self.pending_requests) / self._top_k_requests, 1.0)
        idx += 1
        active = sum(1 for v in self.vehicles if v.status != VehicleStatus.IDLE)
        obs[idx] = active / max(self._num_vehicles, 1)
        idx += 1
        obs[idx] = np.mean([v.utilization for v in self.vehicles]) if self.vehicles else 0.0
        idx += 1
        traffic_level = self.traffic_model.get_current_multiplier() / 4.0  # normalize
        obs[idx] = min(traffic_level, 1.0)
        idx += 1
        total_completed = self._total_deliveries + self._total_sla_violations
        sla_rate = self._total_sla_violations / max(total_completed, 1)
        obs[idx] = sla_rate
        idx += 1

        # === Vehicle features ===
        for v_idx in range(self._max_vehicles_obs):
            if v_idx < len(self.vehicles):
                v = self.vehicles[v_idx]
                nx, ny = self.city_graph.get_normalized_position(v.current_location)
                obs[idx] = nx
                obs[idx + 1] = ny
                obs[idx + 2] = v.utilization
                obs[idx + 3] = float(v.status != VehicleStatus.IDLE)
                obs[idx + 4] = 0.0  # placeholder for route time estimate
                obs[idx + 5] = v.fuel_remaining / self._initial_fuel
            idx += self._vehicle_feat_size

        # === Request features ===
        for r_idx in range(self._top_k_requests):
            if r_idx < len(self.pending_requests):
                req_id = self.pending_requests[r_idx]
                req = self.requests[req_id]
                px, py = self.city_graph.get_normalized_position(req.pickup_location)
                obs[idx] = px
                obs[idx + 1] = py
                # Dropoff info encoded as distance from pickup
                dist = self.city_graph.get_shortest_distance(
                    req.pickup_location, req.dropoff_location
                )
                obs[idx + 2] = min(dist / self.city_graph.config.grid_size, 1.0)
                # SLA remaining normalized
                sla_remain = req.sla_remaining_at(self.current_time)
                obs[idx + 3] = np.clip(sla_remain / 120.0, -1.0, 1.0)
                obs[idx + 4] = float(req.priority.value) / 2.0
            idx += self._request_feat_size

        return obs

    def _get_info(self) -> dict[str, Any]:
        """Build the info dictionary for this step.

        Returns:
            Dictionary with simulation state information.
        """
        return {
            "current_time": self.current_time,
            "step": self._step_count,
            "pending_requests": len(self.pending_requests),
            "total_requests": len(self.requests),
            "total_deliveries": self._total_deliveries,
            "total_expired": self._total_expired,
            "total_sla_violations": self._total_sla_violations,
            "total_on_time": self._total_on_time,
            "cumulative_reward": self._cumulative_reward,
            "traffic_state": self.traffic_model.get_state_name(),
            "fleet_utilization": sum(
                1 for v in self.vehicles if v.status != VehicleStatus.IDLE
            ) / max(self._num_vehicles, 1),
            "action_mask": self.get_action_mask(),
        }

    # === State cloning for MCTS ===

    def clone_state(self) -> dict[str, Any]:
        """Clone the complete environment state for MCTS planning.

        Returns a serializable state dict that can be restored without
        mutating the original environment.

        Returns:
            Dictionary containing all mutable state.
        """
        return {
            "vehicles": [v.clone() for v in self.vehicles],
            "requests": {rid: r.clone() for rid, r in self.requests.items()},
            "pending_requests": list(self.pending_requests),
            "current_time": self.current_time,
            "event_queue": [
                SimEvent(time=e.time, event_type=e.event_type, data=dict(e.data))
                for e in self._event_queue
            ],
            "step_count": self._step_count,
            "total_deliveries": self._total_deliveries,
            "total_expired": self._total_expired,
            "total_sla_violations": self._total_sla_violations,
            "total_on_time": self._total_on_time,
            "cumulative_reward": self._cumulative_reward,
            "traffic_model": self.traffic_model.clone(),
            "rng_state": self._rng.get_state(),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore environment state from a clone.

        Args:
            state: State dictionary from clone_state().
        """
        self.vehicles = [v.clone() for v in state["vehicles"]]
        self.requests = {rid: r.clone() for rid, r in state["requests"].items()}
        self.pending_requests = list(state["pending_requests"])
        self.current_time = state["current_time"]
        self._event_queue = [
            SimEvent(time=e.time, event_type=e.event_type, data=dict(e.data))
            for e in state["event_queue"]
        ]
        self._step_count = state["step_count"]
        self._total_deliveries = state["total_deliveries"]
        self._total_expired = state["total_expired"]
        self._total_sla_violations = state["total_sla_violations"]
        self._total_on_time = state["total_on_time"]
        self._cumulative_reward = state["cumulative_reward"]
        self.traffic_model = state["traffic_model"].clone()
        self._rng.set_state(state["rng_state"])

    @property
    def num_vehicles(self) -> int:
        """Return the number of vehicles."""
        return self._num_vehicles

    @property
    def top_k_requests(self) -> int:
        """Return the top-K request slots."""
        return self._top_k_requests
