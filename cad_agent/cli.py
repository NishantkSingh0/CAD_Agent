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
    _configure_logging(args.verbose)
    provider = GeminiProvider(AgentConfig(), timeout_seconds=args.timeout)
    try:
        result = generate_cad(
            "Create a premium chair from this reference",
            image_paths=["./referenceImages/sample1.png"],
            output_dir="outputs",
            provider=provider,
            image_policy="all",   # ["planner-only", "planner-surface", "all"]
            geometry_mode="template",   # ["template", "llm-dsl"]
            compiler_backend="auto",   # ["auto", "build123d", "mesh"]
        )
    except Exception as exc:
        logging.getLogger(__name__).error("CAD generation failed: %s", exc)
        sys.exit(1)

    print(json.dumps({key: str(path) for key, path in result.compile_result.artifacts.items()}, indent=2))


if __name__ == "__main__":
    main()
