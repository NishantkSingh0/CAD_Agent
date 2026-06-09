"""End-to-end prompt-to-CAD flow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cad_agent.agents import AgentPipeline
from cad_agent.compiler import CompileResult, MeshCompiler
from cad_agent.config import AgentConfig
from cad_agent.dsl import ValidationReport, validate_geometry_dsl
from cad_agent.providers import GeminiProvider, LLMProvider


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
) -> CADGenerationResult:
    """Run Planner -> Topology -> Dimension -> Surface -> Validate -> Compile."""

    config = config or AgentConfig()
    agent_pipeline = AgentPipeline(provider or GeminiProvider(config))
    dsl = agent_pipeline.run(prompt, image_paths=image_paths)

    validation = validate_geometry_dsl(dsl)
    attempts = 0
    while not validation.ok and attempts < config.max_repair_attempts:
        dsl = agent_pipeline.repair(prompt=prompt, dsl=dsl, issues=validation.issues, image_paths=image_paths)
        validation = validate_geometry_dsl(dsl)
        attempts += 1

    validation.require_ok()
    compile_result = MeshCompiler().compile(dsl, output_dir)
    return CADGenerationResult(dsl=dsl, validation=validation, compile_result=compile_result)
