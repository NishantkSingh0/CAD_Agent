"""Command line entry point for the CAD flow."""

from __future__ import annotations

import argparse
import json

from cad_agent.pipeline import generate_cad


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CAD artifacts from a prompt and optional reference images.")
    parser.add_argument("prompt", help="Natural language design prompt.")
    parser.add_argument("--image", action="append", default=[], help="Reference image path. Can be repeated.")
    parser.add_argument("--out", default="outputs", help="Output directory.")
    args = parser.parse_args()

    result = generate_cad(args.prompt, image_paths=args.image, output_dir=args.out)
    print(json.dumps({key: str(path) for key, path in result.compile_result.artifacts.items()}, indent=2))


if __name__ == "__main__":
    main()
