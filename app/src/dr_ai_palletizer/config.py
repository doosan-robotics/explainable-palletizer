"""Application settings loaded from environment variables."""

from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration for the app server.

    All fields are populated from environment variables (case-insensitive).
    A ``.env`` file in the working directory is loaded automatically.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    sim_server_url: str = "http://sim-server:8100"
    inference_server_url: str = "http://inference-server:8200/v1"
    inference_model: str = "nvidia/Cosmos-Reason2-2B"
    lora_model: str = ""
    request_timeout: float = 30.0
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    max_thinking_tokens: int = 512
    max_completion_tokens: int = 2048
    lora_adapter_path: str = ""

    @field_validator("lora_adapter_path", mode="before")
    @classmethod
    def _strip_lora_path(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v

    @property
    def active_model(self) -> str:
        """Return the LoRA adapter model name if configured, else the base model."""
        return self.lora_model or self.inference_model
