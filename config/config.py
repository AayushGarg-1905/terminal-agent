from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()


class ModelConfig(BaseModel):
    name: str = Field(
        default_factory=lambda: os.environ.get("DEFAULT_MODEL", "your-default-model")
    )
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    context_window: int = 256_000


class Config(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    model: ModelConfig = Field(default_factory=ModelConfig)
    cwd: Path = Field(default_factory=Path.cwd)
    max_turns: int = 100
    user_instructions: str | None = None
    debug: bool = False

    @property
    def api_key(self) -> str | None:
        return os.environ.get("OPENROUTER_API_KEY")

    @property
    def base_url(self) -> str | None:
        return os.environ.get("LLM_PROVIDER_BASE_URL")

    @property
    def model_name(self) -> str:
        return self.model.name

    @model_name.setter
    def model_name(self, value: str):
        self.model.name = value

    @property
    def temperature(self) -> float:
        return self.model.temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        self.model.temperature = value

    def validate(self) -> list[str]:
        errors: list[str] = []

        if not self.api_key:
            errors.append("No API key found. Set API_KEY environment variable")

        if not self.cwd.exists():
            errors.append(f"Working directory does not exist: {self.cwd}")

        return errors
