"""Gemini 2.5 Pro provider using only the Python standard library."""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from cad_agent.config import AgentConfig

logger = logging.getLogger(__name__)


class GeminiProvider:
    """Calls Gemini's generateContent endpoint and extracts a JSON object."""

    def __init__(self, config: AgentConfig | None = None, timeout_seconds: int = 90):
        self.config = config or AgentConfig()
        self.timeout_seconds = timeout_seconds

    def generate_json(
        self,
        *,
        stage: str,
        system_prompt: str,
        payload: dict[str, Any],
        image_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        image_paths = image_paths or []
        started_at = time.monotonic()
        logger.info("Gemini stage '%s' started with %d image(s)", stage, len(image_paths))
        parts: list[dict[str, Any]] = [
            {
                "text": (
                    f"{system_prompt}\n\n"
                    "Return one valid JSON object and no markdown.\n\n"
                    f"Stage: {stage}\nInput:\n{json.dumps(payload, indent=2)}"
                )
            }
        ]
        for image_path in image_paths:
            path = Path(image_path)
            mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
            image_bytes = path.read_bytes()
            logger.info(
                "Gemini stage '%s' attaching image %s (%s, %.1f KB)",
                stage,
                path,
                mime_type,
                len(image_bytes) / 1024,
            )
            parts.append(
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64.b64encode(image_bytes).decode("ascii"),
                    }
                }
            )

        body = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"response_mime_type": "application/json"},
        }
        query = urllib.parse.urlencode({"key": self.config.api_key})
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.config.model}:generateContent?{query}"
        )
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            logger.info("Gemini stage '%s' sending request to model %s", stage, self.config.model)
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini request failed for {stage}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Gemini request failed for {stage}: {exc}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise RuntimeError(
                f"Gemini request timed out during {stage} after {self.timeout_seconds}s. "
                "Try rerunning with a larger --timeout value."
            ) from exc

        text = "".join(
            part.get("text", "")
            for candidate in data.get("candidates", [])
            for part in candidate.get("content", {}).get("parts", [])
        )
        if not text.strip():
            raise RuntimeError(f"Gemini returned no JSON text for {stage}.")
        parsed = _parse_json_object(text)
        logger.info(
            "Gemini stage '%s' completed in %.1fs with keys: %s",
            stage,
            time.monotonic() - started_at,
            ", ".join(sorted(parsed.keys())),
        )
        return parsed


def _parse_json_object(text: str) -> dict[str, Any]:
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?", "", clean).strip()
        clean = re.sub(r"```$", "", clean).strip()
    try:
        value = json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Gemini response must be a JSON object.")
    return value
