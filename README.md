# CAD Agent

Prompt-to-CAD flow based on the provided architecture document:

```text
Prompt + optional reference images
  -> Planner Agent
  -> Topology Agent
  -> Dimension Agent
  -> Shared DesignMemory
  -> Semantic template or Surface Agent
  -> Geometry DSL
  -> Validation / Repair
  -> Build123D/OpenCascade compiler + mesh preview fallback
  -> STEP, STL, OBJ, DSL JSON
```

Gemini 2.5 Pro is used for every agentic stage through `GeminiProvider`. The key is read with
`os.getenv("GEMINI_2.5_PRO")` first, then from `.env` because dots in the key name are not shell-exportable in many environments.

## Run Recommended Flow

```bash
python3 -m cad_agent.cli "Create a Scandinavian lounge chair with curved armrests and a rounded backrest" \
  --image path/to/reference.png \
  --out outputs \
  --timeout 240
```

Default behavior:

- `--geometry-mode template`: Gemini extracts visual/design memory, then prompt-aware routing uses a deterministic template when one matches (`sphere`, `lounge_tub_chair`) or falls back to the Surface Agent for Geometry DSL.
- `--image-policy planner-only`: only the Planner receives the raw image; later stages receive the compact visual observations as text.
- `--compiler auto`: use Build123D/OpenCascade when installed, otherwise fall back to the mesh compiler.

Outputs:

- `*.step` when Build123D/OpenCascade is available
- `*.stl`
- `*.cad.stl` for Build123D solid-only export
- `*.obj`
- `*.dsl.json`

For raw LLM-authored Geometry DSL experiments:

```bash
python3 -m cad_agent.cli "Create a premium chair from this reference" \
  --image path/to/reference.png \
  --out outputs \
  --geometry-mode llm-dsl \
  --image-policy planner-surface
```

For a simple analytic primitive:

```bash
python3 -m cad_agent.cli "create a sphere" \
  --out outputs \
  --timeout 40
```

For mesh-only fallback:

```bash
python3 -m cad_agent.cli "Create a premium chair from this reference" \
  --image path/to/reference.png \
  --out outputs \
  --compiler mesh
```

## Test

```bash
python3 -m unittest
```

The regression tests use fake Gemini providers to verify both the template flow and the raw `geometry_dsl` wrapper normalization.
