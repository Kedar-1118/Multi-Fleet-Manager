"""Tests for the FastAPI serving layer.

Uses the FastAPI TestClient (synchronous) to validate all four endpoints:
  GET  /health
  POST /dispatch
  POST /simulate
  GET  /metrics

Tests run without a trained model loaded, which exercises the fallback
dispatcher paths in the API.
"""

import pytest
from fastapi.testclient import TestClient

from src.serving.api import app


@pytest.fixture(scope="module")
def client():
    """Create a test client for the FastAPI app."""
    with TestClient(app) as c:
        yield c


# =========================================================================
# GET /health
# =========================================================================

class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_returns_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_body(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert data["version"] == "1.0.0"


# =========================================================================
# GET /metrics
# =========================================================================

class TestMetricsEndpoint:
    """Tests for the service metrics endpoint."""

    def test_metrics_returns_200(self, client: TestClient) -> None:
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_fields(self, client: TestClient) -> None:
        data = client.get("/metrics").json()
        assert "total_dispatches" in data
        assert "avg_latency_ms" in data
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0.0


# =========================================================================
# POST /dispatch
# =========================================================================

class TestDispatchEndpoint:
    """Tests for the dispatch endpoint."""

    def _make_payload(self, method: str = "greedy") -> dict:
        return {
            "vehicles": [
                {"vehicle_id": 0, "current_location": 0, "capacity": 10,
                 "current_load": 0, "status": "IDLE"},
                {"vehicle_id": 1, "current_location": 5, "capacity": 10,
                 "current_load": 0, "status": "IDLE"},
            ],
            "pending_requests": [
                {"request_id": 100, "pickup_location": 1,
                 "dropoff_location": 3, "deadline_minutes": 60,
                 "priority": 0, "package_size": 1},
            ],
            "traffic_state": "NORMAL_TRAFFIC",
            "method": method,
        }

    def test_dispatch_greedy_returns_200(self, client: TestClient) -> None:
        resp = client.post("/dispatch", json=self._make_payload("greedy"))
        assert resp.status_code == 200

    def test_dispatch_nearest_returns_200(self, client: TestClient) -> None:
        resp = client.post("/dispatch", json=self._make_payload("nearest"))
        assert resp.status_code == 200

    def test_dispatch_response_fields(self, client: TestClient) -> None:
        data = client.post("/dispatch", json=self._make_payload("greedy")).json()
        assert "decision_source" in data
        assert "latency_ms" in data
        assert isinstance(data["latency_ms"], (int, float))
        assert data["latency_ms"] > 0.0

    def test_dispatch_no_vehicles(self, client: TestClient) -> None:
        """Should handle a payload with no vehicles gracefully."""
        payload = {
            "vehicles": [],
            "pending_requests": [
                {"request_id": 1, "pickup_location": 0,
                 "dropoff_location": 1, "deadline_minutes": 60,
                 "priority": 0, "package_size": 1},
            ],
            "traffic_state": "NORMAL_TRAFFIC",
            "method": "greedy",
        }
        resp = client.post("/dispatch", json=payload)
        # Should either return 200 with a noop or 500 — not crash unhandled
        assert resp.status_code in (200, 500)

    def test_dispatch_no_requests(self, client: TestClient) -> None:
        """Should return a noop when there are no pending requests."""
        payload = {
            "vehicles": [
                {"vehicle_id": 0, "current_location": 0, "capacity": 10,
                 "current_load": 0, "status": "IDLE"},
            ],
            "pending_requests": [],
            "traffic_state": "NORMAL_TRAFFIC",
            "method": "greedy",
        }
        resp = client.post("/dispatch", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_noop"] is True

    def test_dispatch_invalid_traffic_state(self, client: TestClient) -> None:
        """Invalid traffic state enum should return 422."""
        payload = self._make_payload()
        payload["traffic_state"] = "INVALID_STATE"
        resp = client.post("/dispatch", json=payload)
        assert resp.status_code == 422


# =========================================================================
# POST /simulate
# =========================================================================

class TestSimulateEndpoint:
    """Tests for the simulate endpoint."""

    def test_simulate_greedy_returns_200(self, client: TestClient) -> None:
        payload = {"method": "greedy", "n_episodes": 1, "seed": 42}
        resp = client.post("/simulate", json=payload)
        assert resp.status_code == 200

    def test_simulate_response_fields(self, client: TestClient) -> None:
        payload = {"method": "nearest", "n_episodes": 1, "seed": 42}
        data = client.post("/simulate", json=payload).json()
        assert data["method"] == "nearest"
        assert data["episodes_completed"] >= 1
        assert "metrics" in data

    def test_simulate_invalid_episodes(self, client: TestClient) -> None:
        """n_episodes out of valid range should return 422."""
        payload = {"method": "greedy", "n_episodes": 0, "seed": 42}
        resp = client.post("/simulate", json=payload)
        assert resp.status_code == 422
