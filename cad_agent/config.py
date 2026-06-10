"""Configuration helpers for the CAD agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL = "gemini-2.5-pro"
ENV_KEY_NAME = "GEMINI_2.5_PRO"
DEFAULT_GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_ENV_KEY_NAME = "GROQ_API_KEY"


def load_env_value(name: str, env_file: Path | str = ".env") -> str | None:
    """Load a value from process env, falling back to a local .env file.

    The requested Gemini key name contains dots, which many shells cannot
    export as a variable. Reading the .env file directly keeps the user-facing
    contract exactly as requested: os.getenv("GEMINI_2.5_PRO") is checked first.
    """

    value = os.getenv(name)
    if value:
        return value

    path = Path(env_file)
    if not path.exists():
        return None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        if key.strip() == name:
            return raw_value.strip().strip('"').strip("'")
    return None


@dataclass(frozen=True)
class AgentConfig:
    model: str = DEFAULT_MODEL
    api_key_name: str = ENV_KEY_NAME
    env_file: Path = Path(".env")
    max_repair_attempts: int = 2
    groq_model: str = DEFAULT_GROQ_MODEL
    groq_api_key_name: str = GROQ_ENV_KEY_NAME

    @property
    def api_key(self) -> str:
        value = load_env_value(self.api_key_name, self.env_file)
        if not value:
            raise RuntimeError(
                f"Missing Gemini API key. Set {self.api_key_name} in the environment or .env."
            )
        return value

    @property
    def groq_api_key(self) -> str | None:
        return load_env_value(self.groq_api_key_name, self.env_file)
