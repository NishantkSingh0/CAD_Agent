"""Command line entry point for the CAD flow."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from cad_agent.config import AgentConfig
from cad_agent.pipeline import generate_cad
from cad_agent.providers import GeminiProvider


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CAD artifacts from a prompt and optional reference images.")
    parser.add_argument("prompt", help="Natural language design prompt.")
    parser.add_argument("--image", action="append", default=[], help="Reference image path. Can be repeated.")
    parser.add_argument("--out", default="outputs", help="Output directory.")
    parser.add_argument("--timeout", type=int, default=180, help="Gemini timeout per agent stage in seconds.")
    parser.add_argument(
        "--image-policy",
        choices=["planner-only", "planner-surface", "all"],
        default="planner-only",
        help="Which agent stages receive raw image input. Later stages always receive planner observations as text.",
    )
    parser.add_argument(
        "--geometry-mode",
        choices=["template", "llm-dsl"],
        default="template",
        help="Use deterministic semantic templates or ask Gemini to emit raw Geometry DSL.",
    )
    parser.add_argument(
        "--compiler",
        choices=["auto", "build123d", "mesh"],
        default="auto",
        help="CAD compiler backend. auto prefers Build123D/OpenCascade and falls back to mesh.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs.")
    args = parser.parse_args()

    _configure_logging(args.verbose)
    provider = GeminiProvider(AgentConfig(), timeout_seconds=args.timeout)
    try:
        result = generate_cad(
            args.prompt,
            image_paths=args.image,
            output_dir=args.out,
            provider=provider,
            image_policy=args.image_policy,
            geometry_mode=args.geometry_mode,
            compiler_backend=args.compiler,
        )
    except Exception as exc:
        logging.getLogger(__name__).error("CAD generation failed: %s", exc)
        sys.exit(1)

    print(json.dumps({key: str(path) for key, path in result.compile_result.artifacts.items()}, indent=2))


if __name__ == "__main__":
    main()
