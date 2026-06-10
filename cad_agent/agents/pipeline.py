"""Planner, topology, dimension, and surface agent orchestration."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from cad_agent.memory import DesignMemory
from cad_agent.providers.base import LLMProvider

logger = logging.getLogger(__name__)


PROMPTS = {
    "planner": (
        "You are the Planner Agent for a prompt-to-CAD system. "
        "Convert the user prompt and optional reference images into a compact visual design spec. "
        "Return short JSON only with object_intent, component_names, materials, style_notes, "
        "image_observations, and curvature_requirements. Keep image_observations to at most 6 bullets. "
        "Do not emit geometry."
    ),
    "topology": (
        "You are the Topology Agent. Define component relationships, anchors, and assembly "
        "hierarchy so the CAD compiler avoids floating parts. Do not emit CAD code."
    ),
    "dimension": (
        "You are the Dimension Agent. Generate realistic millimeter dimensions for the requested object. "
        "Use domain reasoning only when relevant, such as ergonomics for furniture. "
        "Keep all units in mm. Do not emit CAD code."
    ),
    "surface": (
        "You are the Surface Agent. Generate the Geometry DSL only. Curved furniture features "
        "must be represented as bezier_sweep or nurbs_surface control points with thickness. "
        "Use parametric geometry descriptions, never Build123D, FreeCAD, or Python code. "
        "Return the DSL object itself with this exact top-level shape: "
        "{'name': string, 'unit': 'mm', 'components': [{'name': string, 'origin': [x,y,z], "
        "'connects_to': [string], 'geometry': {'type': 'rounded_box|box|tapered_cylinder|"
        "cylinder|sphere|bezier_sweep|nurbs_surface', ...}}]}. Do not wrap it in geometry_dsl or dsl."
    ),
    "repair": (
        "You are the Repair Agent. Fix only validation issues in the Geometry DSL while "
        "preserving the design intent and curved surface requirements. Return the full DSL "
        "object itself with top-level name, unit, and components. Do not wrap it in geometry_dsl or dsl."
    ),
}


@dataclass
class AgentPipeline:
    provider: LLMProvider
    image_policy: str = "planner-only"

    def run(self, prompt: str, image_paths: list[str] | None = None) -> dict[str, Any]:
        memory = self.run_to_memory(prompt, image_paths=image_paths)
        return self.run_surface(prompt, memory, image_paths=image_paths)

    def run_surface(
        self,
        prompt: str,
        memory: DesignMemory,
        image_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        return self._run_stage(
            "surface",
            {
                "prompt": prompt,
                "planner": memory.planner,
                "topology": memory.topology,
                "dimensions": memory.dimensions,
            },
            self._images_for_stage("surface", image_paths or []),
        )

    def run_to_memory(self, prompt: str, image_paths: list[str] | None = None) -> DesignMemory:
        image_paths = image_paths or []
        planner = self._run_stage(
            "planner",
            {"prompt": prompt, "image_count": len(image_paths)},
            self._images_for_stage("planner", image_paths),
        )
        topology = self._run_stage(
            "topology",
            {"prompt": prompt, "planner": planner},
            self._images_for_stage("topology", image_paths),
        )
        dimensions = self._run_stage(
            "dimension",
            {"prompt": prompt, "planner": planner, "topology": topology},
            self._images_for_stage("dimension", image_paths),
        )
        return DesignMemory.from_agent_outputs(
            prompt=prompt,
            image_paths=image_paths,
            planner=planner,
            topology=topology,
            dimensions=dimensions,
        )

    def repair(
        self,
        *,
        prompt: str,
        dsl: dict[str, Any],
        issues: list[str],
        image_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        return self._run_stage(
            "repair",
            {"prompt": prompt, "dsl": dsl, "issues": issues},
            self._images_for_stage("repair", image_paths or []),
        )

    def _run_stage(
        self,
        stage: str,
        payload: dict[str, Any],
        image_paths: list[str],
    ) -> dict[str, Any]:
        started_at = time.monotonic()
        logger.info("Agent stage '%s' starting", stage)
        result = self.provider.generate_json(
            stage=stage,
            system_prompt=PROMPTS[stage],
            payload=payload,
            image_paths=image_paths,
        )
        logger.info("Agent stage '%s' finished in %.1fs", stage, time.monotonic() - started_at)
        return result

    def _images_for_stage(self, stage: str, image_paths: list[str]) -> list[str]:
        if self.image_policy == "all":
            return image_paths
        if self.image_policy == "planner-surface" and stage in {"planner", "surface"}:
            return image_paths
        if self.image_policy == "planner-only" and stage == "planner":
            return image_paths
        return []
