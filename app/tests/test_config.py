"""Tests for app server Settings configuration."""

from __future__ import annotations

import pytest
from dr_ai_palletizer.config import Settings


class TestSettings:
    def test_defaults(self) -> None:
        settings = Settings()
        assert settings.sim_server_url == "http://sim-server:8100"
        assert settings.inference_server_url == "http://inference-server:8200/v1"
        assert settings.inference_model == "nvidia/Cosmos-Reason2-2B"
        assert settings.request_timeout == 30.0
        assert settings.app_host == "0.0.0.0"
        assert settings.app_port == 8000

    def test_lora_adapter_path_default_empty(self) -> None:
        settings = Settings()
        assert settings.lora_adapter_path == ""
        assert not settings.lora_adapter_path

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIM_SERVER_URL", "http://localhost:9999")
        monkeypatch.setenv("INFERENCE_MODEL", "my-model")
        monkeypatch.setenv("REQUEST_TIMEOUT", "60.0")

        settings = Settings()
        assert settings.sim_server_url == "http://localhost:9999"
        assert settings.inference_model == "my-model"
        assert settings.request_timeout == 60.0

    def test_lora_adapter_path_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LORA_ADAPTER_PATH", "/adapters/palletizer")
        settings = Settings()
        assert settings.lora_adapter_path == "/adapters/palletizer"

    def test_lora_adapter_path_strips_whitespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LORA_ADAPTER_PATH", "  ")
        settings = Settings()
        assert settings.lora_adapter_path == ""
        assert not settings.lora_adapter_path
