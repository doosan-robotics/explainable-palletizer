"""Tests for app server Pydantic request/response models."""

from __future__ import annotations

import pytest
from dr_ai_palletizer.api_models import (
    HealthResponse,
    PalletizeRequest,
    PalletizeResponse,
    PlanRequest,
    PlanResponse,
    ServiceHealth,
    StatusResponse,
)
from pydantic import ValidationError


class TestHealthResponse:
    def test_defaults(self) -> None:
        resp = HealthResponse()
        assert resp.status == "ok"


class TestServiceHealth:
    def test_healthy(self) -> None:
        sh = ServiceHealth(name="sim-server", healthy=True)
        assert sh.detail == ""

    def test_unhealthy_with_detail(self) -> None:
        sh = ServiceHealth(name="sim-server", healthy=False, detail="timeout")
        assert sh.healthy is False
        assert sh.detail == "timeout"


class TestStatusResponse:
    def test_defaults(self) -> None:
        resp = StatusResponse()
        assert resp.status == "ok"
        assert resp.services == []

    def test_with_services(self) -> None:
        resp = StatusResponse(
            status="degraded",
            services=[
                ServiceHealth(name="sim", healthy=True),
                ServiceHealth(name="inference", healthy=False),
            ],
        )
        assert len(resp.services) == 2
        assert resp.status == "degraded"


class TestPlanRequest:
    def test_valid(self) -> None:
        req = PlanRequest(scenario_text="boxes on conveyor")
        assert req.scenario_text == "boxes on conveyor"
        assert req.system_prompt is None

    def test_with_system_prompt(self) -> None:
        req = PlanRequest(scenario_text="test", system_prompt="custom prompt")
        assert req.system_prompt == "custom prompt"

    def test_rejects_empty_scenario(self) -> None:
        with pytest.raises(ValidationError):
            PlanRequest(scenario_text="")

    def test_requires_scenario_text(self) -> None:
        with pytest.raises(ValidationError):
            PlanRequest()


class TestPlanResponse:
    def test_creation(self) -> None:
        resp = PlanResponse(plan="pick box A", model="cosmos-2b")
        assert resp.plan == "pick box A"
        assert resp.model == "cosmos-2b"

    def test_defaults(self) -> None:
        resp = PlanResponse(plan="result")
        assert resp.model == ""


class TestPalletizeRequest:
    def test_defaults(self) -> None:
        req = PalletizeRequest()
        assert req.scenario_text == ""
        assert req.auto_execute is False


class TestPalletizeResponse:
    def test_defaults(self) -> None:
        resp = PalletizeResponse()
        assert resp.status == "accepted"
        assert "not yet implemented" in resp.message.lower()
