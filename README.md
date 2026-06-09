# CAD Agent

Prompt-to-CAD flow based on the provided architecture document:

```text
Prompt + optional reference images
  -> Planner Agent
  -> Topology Agent
  -> Dimension Agent
  -> Surface Agent
  -> Geometry DSL
  -> Validation / Repair
  -> Mesh CAD compiler
  -> STL, OBJ, DSL JSON
```

Gemini 2.5 Pro is used for every agentic stage through `GeminiProvider`. The key is read with
`os.getenv("GEMINI_2.5_PRO")` first, then from `.env` because dots in the key name are not shell-exportable in many environments.

## Run

```bash
python3 -m cad_agent.cli "Create a Scandinavian lounge chair with curved armrests and a rounded backrest" --image path/to/reference.png --out outputs
```

The no-dependency compiler writes:

- `*.stl`
- `*.obj`
- `*.dsl.json`

Build123D/OpenCascade can be added behind `cad_agent.compiler` later without changing the agent flow.

## Test

```bash
python3 -m unittest
```

The regression test uses a fake Gemini provider that receives an image path, emits image-aware DSL, and verifies that the final model includes a `nurbs_surface` backrest plus `bezier_sweep` armrests.
