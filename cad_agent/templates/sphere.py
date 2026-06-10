"""Deterministic sphere template."""

from __future__ import annotations

from typing import Any

from cad_agent.memory import DesignMemory


def build_sphere_dsl(memory: DesignMemory) -> dict[str, Any]:
    """Build a simple analytic sphere from shared design memory."""

    radius = _clamp(_dimension(memory, "radius", 150), 10, 2000)
    return {
        "name": _name_from_prompt(memory.prompt),
        "unit": "mm",
        "template": memory.template,
        "assumptions": memory.assumptions,
        "components": [
            {
                "name": "sphere_body",
                "material": "default",
                "origin": [0, 0, 0],
                "connects_to": [],
                "geometry": {
                    "type": "sphere",
                    "radius": radius,
                },
            }
        ],
    }


def _dimension(memory: DesignMemory, key: str, default: float) -> float:
    for source in (memory.dimensions, memory.planner):
        value = source.get(key)
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, dict):
            nested = value.get(key)
            if isinstance(nested, int | float):
                return float(nested)
    overall = memory.dimensions.get("overall_dimensions")
    if isinstance(overall, dict):
        for candidate in ("diameter", "width", "height", "depth"):
            value = overall.get(candidate)
            if isinstance(value, int | float):
                return float(value) / 2
    return default


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _name_from_prompt(prompt: str) -> str:
    words = [word.strip(".,:;!?()[]{}").lower() for word in prompt.split()]
    useful = [word for word in words if word and word not in {"create", "this", "the", "a", "an"}]
    return "_".join(useful[:6] or ["generated_sphere"])
