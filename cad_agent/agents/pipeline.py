"""Planner, topology, dimension, and surface agent orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cad_agent.providers.base import LLMProvider


PROMPTS = {
    "planner": (
        "You are the Planner Agent for a prompt-to-CAD furniture system. "
        "Convert the user prompt and optional reference images into object intent, "
        "component names, materials, style notes, and image observations. Do not emit geometry."
    ),
    "topology": (
        "You are the Topology Agent. Define component relationships, anchors, and assembly "
        "hierarchy so the CAD compiler avoids floating parts. Do not emit CAD code."
    ),
    "dimension": (
        "You are the Dimension Agent. Generate realistic millimeter dimensions for furniture "
        "using ergonomic reasoning. Keep all units in mm. Do not emit CAD code."
    ),
    "surface": (
        "You are the Surface Agent. Generate the Geometry DSL only. Curved furniture features "
        "must be represented as bezier_sweep or nurbs_surface control points with thickness. "
        "Use parametric geometry descriptions, never Build123D, FreeCAD, or Python code."
    ),
    "repair": (
        "You are the Repair Agent. Fix only validation issues in the Geometry DSL while "
        "preserving the design intent and curved surface requirements. Return the full DSL."
    ),
}


@dataclass
class AgentPipeline:
    provider: LLMProvider

    def run(self, prompt: str, image_paths: list[str] | None = None) -> dict[str, Any]:
        image_paths = image_paths or []
        planner = self.provider.generate_json(
            stage="planner",
            system_prompt=PROMPTS["planner"],
            payload={"prompt": prompt, "image_count": len(image_paths)},
            image_paths=image_paths,
        )
        topology = self.provider.generate_json(
            stage="topology",
            system_prompt=PROMPTS["topology"],
            payload={"prompt": prompt, "planner": planner},
            image_paths=image_paths,
        )
        dimensions = self.provider.generate_json(
            stage="dimension",
            system_prompt=PROMPTS["dimension"],
            payload={"prompt": prompt, "planner": planner, "topology": topology},
            image_paths=image_paths,
        )
        return self.provider.generate_json(
            stage="surface",
            system_prompt=PROMPTS["surface"],
            payload={
                "prompt": prompt,
                "planner": planner,
                "topology": topology,
                "dimensions": dimensions,
            },
            image_paths=image_paths,
        )

    def repair(
        self,
        *,
        prompt: str,
        dsl: dict[str, Any],
        issues: list[str],
        image_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        return self.provider.generate_json(
            stage="repair",
            system_prompt=PROMPTS["repair"],
            payload={"prompt": prompt, "dsl": dsl, "issues": issues},
            image_paths=image_paths or [],
        )
