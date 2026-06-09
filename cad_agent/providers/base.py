"""Provider protocol used by all agent stages."""

from __future__ import annotations

from typing import Any, Protocol


class LLMProvider(Protocol):
    def generate_json(
        self,
        *,
        stage: str,
        system_prompt: str,
        payload: dict[str, Any],
        image_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return a JSON object for one agent stage."""
