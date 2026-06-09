"""End-to-end prompt-to-CAD flow."""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cad_agent.agents import AgentPipeline
from cad_agent.compiler import CompileResult, MeshCompiler
from cad_agent.config import AgentConfig
from cad_agent.dsl import ValidationReport, normalize_geometry_dsl, validate_geometry_dsl
from cad_agent.providers import GeminiProvider, LLMProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CADGenerationResult:
    dsl: dict[str, Any]
    validation: ValidationReport
    compile_result: CompileResult


def generate_cad(
    prompt: str,
    *,
    image_paths: list[str] | None = None,
    output_dir: Path | str = "outputs",
    provider: LLMProvider | None = None,
    config: AgentConfig | None = None,
    image_policy: str = "planner-only",
) -> CADGenerationResult:
    """Run Planner -> Topology -> Dimension -> Surface -> Validate -> Compile."""

    image_paths = image_paths or []
    config = config or AgentConfig()
    logger.info("CAD generation started")
    logger.info("Prompt: %s", prompt)
    logger.info("Reference images: %d", len(image_paths))
    for image_path in image_paths:
        logger.info("Reference image: %s", image_path)
    logger.info("Output directory: %s", output_dir)
    logger.info("Image policy: %s", image_policy)

    agent_pipeline = AgentPipeline(provider or GeminiProvider(config), image_policy=image_policy)
    dsl = normalize_geometry_dsl(agent_pipeline.run(prompt, image_paths=image_paths))

    logger.info("Validating Geometry DSL")
    validation = validate_geometry_dsl(dsl)
    attempts = 0
    while not validation.ok and attempts < config.max_repair_attempts:
        logger.warning("Validation failed: %s", "; ".join(validation.issues))
        _write_debug_dsl(output_dir, f"failed_dsl_attempt_{attempts + 1}.json", dsl)
        logger.info("Repair attempt %d of %d", attempts + 1, config.max_repair_attempts)
        dsl = normalize_geometry_dsl(
            agent_pipeline.repair(prompt=prompt, dsl=dsl, issues=validation.issues, image_paths=image_paths)
        )
        validation = validate_geometry_dsl(dsl)
        attempts += 1

    if not validation.ok:
        _write_debug_dsl(output_dir, "failed_dsl_final.json", dsl)
    validation.require_ok()
    logger.info("Validation passed")
    logger.info("Compiling Geometry DSL into mesh artifacts")
    compile_result = MeshCompiler().compile(dsl, output_dir)
    logger.info(
        "Compiled %d vertices, %d faces, curved components: %s",
        len(compile_result.mesh.vertices),
        len(compile_result.mesh.faces),
        ", ".join(compile_result.curved_components) or "none",
    )
    for artifact_type, artifact_path in compile_result.artifacts.items():
        logger.info("Wrote %s: %s", artifact_type.upper(), artifact_path)
    logger.info("CAD generation completed")
    return CADGenerationResult(dsl=dsl, validation=validation, compile_result=compile_result)


def _write_debug_dsl(output_dir: Path | str, filename: str, dsl: dict[str, Any]) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / filename
    path.write_text(json.dumps(dsl, indent=2), encoding="utf-8")
    logger.info("Wrote debug DSL snapshot: %s", path)
