"""Tests for merged server FastAPI endpoints.

Uses FastAPI's TestClient with mocked downstream service clients.
The lifespan creates real clients, so we overwrite app.state after
entering the TestClient context.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from dr_ai_palletizer.config import Settings
from dr_ai_palletizer.server import app
from fastapi.testclient import TestClient


def _mock_sim() -> MagicMock:
    mock = MagicMock()
    mock.health = AsyncMock(return_value={"status": "ok", "sim_time": 1.0})
    mock.close = AsyncMock()
    return mock


def _mock_inference() -> MagicMock:
    mock = MagicMock()
    mock.health = AsyncMock(return_value=True)
    mock.get_plan = AsyncMock(return_value="SINGLE_PICK box A to pallet 1")
    mock.close = AsyncMock()
    return mock


@pytest.fixture()
def client() -> Iterator[TestClient]:
    with TestClient(app) as tc:
        tc.app.state.settings = Settings(
            sim_server_url="http://mock-sim:8100",
            inference_server_url="http://mock-inference:8200/v1",
            inference_model="test-model",
        )
        tc.app.state.sim_client = _mock_sim()
        tc.app.state.inference_client = _mock_inference()
        yield tc


class TestHealth:
    def test_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestStatus:
    def test_all_healthy(self, client: TestClient) -> None:
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert len(data["services"]) == 2
        assert all(s["healthy"] for s in data["services"])

    def test_degraded_when_sim_down(self, client: TestClient) -> None:
        client.app.state.sim_client.health = AsyncMock(side_effect=Exception("timeout"))
        resp = client.get("/api/status")
        data = resp.json()
        assert data["status"] == "degraded"
        sim_svc = next(s for s in data["services"] if s["name"] == "sim-server")
        assert sim_svc["healthy"] is False

    def test_degraded_when_inference_down(self, client: TestClient) -> None:
        client.app.state.inference_client.health = AsyncMock(return_value=False)
        resp = client.get("/api/status")
        data = resp.json()
        assert data["status"] == "degraded"


class TestPlan:
    def test_returns_plan(self, client: TestClient) -> None:
        resp = client.post("/api/plan", json={"scenario_text": "boxes on conveyor"})
        assert resp.status_code == 200
        data = resp.json()
        assert "SINGLE_PICK" in data["plan"]
        assert data["model"] == "test-model"

    def test_rejects_empty_scenario(self, client: TestClient) -> None:
        resp = client.post("/api/plan", json={"scenario_text": ""})
        assert resp.status_code == 422

    def test_returns_502_when_inference_fails(self, client: TestClient) -> None:
        client.app.state.inference_client.get_plan = AsyncMock(
            side_effect=Exception("connection refused"),
        )
        resp = client.post("/api/plan", json={"scenario_text": "boxes on conveyor"})
        assert resp.status_code == 502
        assert "Inference server error" in resp.json()["detail"]


class TestPalletize:
    def test_returns_stub(self, client: TestClient) -> None:
        resp = client.post("/api/palletize", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accepted"
        assert "not yet implemented" in data["message"].lower()
