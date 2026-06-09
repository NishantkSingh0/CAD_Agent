"""Shared design memory carried across the CAD generation flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DesignMemory:
    prompt: str
    image_paths: list[str] = field(default_factory=list)
    planner: dict[str, Any] = field(default_factory=dict)
    topology: dict[str, Any] = field(default_factory=dict)
    dimensions: dict[str, Any] = field(default_factory=dict)
    template: str = "lounge_tub_chair"
    assumptions: list[str] = field(default_factory=list)

    @classmethod
    def from_agent_outputs(
        cls,
        *,
        prompt: str,
        image_paths: list[str],
        planner: dict[str, Any],
        topology: dict[str, Any],
        dimensions: dict[str, Any],
    ) -> "DesignMemory":
        return cls(
            prompt=prompt,
            image_paths=image_paths,
            planner=planner,
            topology=topology,
            dimensions=dimensions,
            template=_select_template(prompt, planner),
            assumptions=[
                "single image is treated as visual reference, not exact reconstruction",
                "hidden rear geometry is inferred from furniture ergonomics",
                "dimensions are assumed in millimeters from common lounge-chair proportions",
            ],
        )

    def overall(self, key: str, default: float) -> float:
        overall = self.dimensions.get("overall_dimensions", {})
        if isinstance(overall, dict) and isinstance(overall.get(key), int | float):
            return float(overall[key])
        if isinstance(self.dimensions.get(key), int | float):
            return float(self.dimensions[key])
        return default


def _select_template(prompt: str, planner: dict[str, Any]) -> str:
    text = f"{prompt} {planner}".lower()
    if "chair" in text or "lounge" in text or "armrest" in text:
        return "lounge_tub_chair"
    return "lounge_tub_chair"
