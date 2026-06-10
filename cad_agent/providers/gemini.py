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
        try:
            return self._generate_gemini_json(
                stage=stage,
                system_prompt=system_prompt,
                payload=payload,
                image_paths=image_paths,
            )
        except Exception as exc:
            groq_key = self.config.groq_api_key
            if not groq_key:
                logger.warning(
                    "Gemini stage '%s' failed, and no GROQ_API_KEY is configured for fallback: %s",
                    stage,
                    exc,
                )
                raise
            
            logger.warning(
                "Gemini stage '%s' failed: %s. Attempting fallback to Groq model %s.",
                stage,
                exc,
                self.config.groq_model,
            )
            try:
                return self._generate_groq_json(
                    stage=stage,
                    system_prompt=system_prompt,
                    payload=payload,
                    image_paths=image_paths,
                    groq_key=groq_key,
                )
            except Exception as groq_exc:
                logger.error(
                    "Groq fallback also failed for stage '%s': %s",
                    stage,
                    groq_exc,
                )
                raise RuntimeError(
                    f"Both Gemini and Groq fallback failed for stage '{stage}'. "
                    f"Gemini error: {exc}. Groq error: {groq_exc}"
                ) from groq_exc

    def _generate_gemini_json(
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
        while True:
            try:
                logger.info("Gemini stage '%s' sending request to model %s", stage, self.config.model)
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                is_retryable = False
                if exc.code == 503:
                    is_retryable = True
                else:
                    try:
                        err_data = json.loads(detail)
                        err_code = err_data.get("error", {}).get("code")
                        err_status = err_data.get("error", {}).get("status")
                        err_msg = err_data.get("error", {}).get("message")
                        if (
                            err_code == 503
                            or err_status == "UNAVAILABLE"
                            or "experiencing high demand" in (err_msg or "")
                        ):
                            is_retryable = True
                    except Exception:
                        pass
                
                if is_retryable:
                    logger.warning(
                        "Gemini stage '%s' failed with 503 UNAVAILABLE (model experiencing high demand). "
                        "Retrying request after 2.0s... Error detail: %s",
                        stage,
                        detail,
                    )
                    time.sleep(2.0)
                    continue
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

    def _generate_groq_json(
        self,
        *,
        stage: str,
        system_prompt: str,
        payload: dict[str, Any],
        image_paths: list[str] | None = None,
        groq_key: str,
    ) -> dict[str, Any]:
        started_at = time.monotonic()
        image_paths = image_paths or []
        logger.info(
            "Groq stage '%s' fallback started with %d image(s)",
            stage,
            len(image_paths),
        )

        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": f"Stage: {stage}\nInput:\n{json.dumps(payload, indent=2)}",
            }
        ]

        for image_path in image_paths:
            path = Path(image_path)
            mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
            image_bytes = path.read_bytes()
            b64_data = base64.b64encode(image_bytes).decode("ascii")
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{b64_data}"
                    }
                }
            )

        messages = [
            {
                "role": "system",
                "content": f"{system_prompt}\n\nReturn one valid JSON object and no markdown.",
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        body = {
            "model": self.config.groq_model,
            "messages": messages,
            "temperature": 1,
            "response_format": {"type": "json_object"},
        }

        url = "https://api.groq.com/openai/v1/chat/completions"
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {groq_key}",
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            },
            method="POST",
        )

        try:
            logger.info(
                "Groq stage '%s' sending request to model %s",
                stage,
                self.config.groq_model,
            )
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Groq request failed for {stage}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Groq request failed for {stage}: {exc}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise RuntimeError(
                f"Groq request timed out during {stage} after {self.timeout_seconds}s."
            ) from exc

        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                f"Unexpected response format from Groq: {data}"
            ) from exc

        if not text.strip():
            raise RuntimeError(f"Groq returned no text for {stage}.")

        parsed = _parse_json_object(text)
        logger.info(
            "Groq stage '%s' completed in %.1fs with keys: %s",
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
