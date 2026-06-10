"""Compiles Geometry DSL into STL/OBJ mesh artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cad_agent.compiler.mesh import (
    Mesh,
    bezier_sweep_mesh,
    box_mesh,
    curved_surface_mesh,
    sphere_mesh,
    tapered_cylinder_mesh,
)
from cad_agent.dsl.validation import validate_geometry_dsl


@dataclass(frozen=True)
class CompileResult:
    mesh: Mesh
    artifacts: dict[str, Path]
    curved_components: list[str]


class MeshCompiler:
    """No-dependency compiler for testable CAD output.

    This is deliberately isolated behind a compiler boundary so a Build123D
    implementation can be added later without changing the agent flow.
    """

    def compile(self, dsl: dict[str, Any], output_dir: Path | str) -> CompileResult:
        validate_geometry_dsl(dsl).require_ok()
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        assembly = Mesh()
        curved_components: list[str] = []
        for component in dsl["components"]:
            geometry = component["geometry"]
            origin = _vec3(component.get("origin", [0, 0, 0]))
            geometry_type = geometry["type"]
            if geometry_type in {"box", "rounded_box"}:
                part_mesh = box_mesh(geometry["width"], geometry["depth"], geometry["height"], origin)
            elif geometry_type == "cylinder":
                axis = _vec3(geometry["axis"]) if "axis" in geometry else None
                part_mesh = tapered_cylinder_mesh(geometry["radius"], geometry["radius"], geometry["height"], origin, axis)
            elif geometry_type == "tapered_cylinder":
                part_mesh = tapered_cylinder_mesh(
                    geometry["radius_top"],
                    geometry["radius_bottom"],
                    geometry["height"],
                    origin,
                    _vec3(geometry["axis"]) if "axis" in geometry else None,
                )
            elif geometry_type == "sphere":
                curved_components.append(component["name"])
                part_mesh = sphere_mesh(geometry["radius"], origin)
            elif geometry_type == "bezier_sweep":
                curved_components.append(component["name"])
                part_mesh = bezier_sweep_mesh(
                    [_offset(_vec3(point), origin) for point in geometry["control_points"]],
                    geometry.get("radius", geometry["thickness"] / 2),
                )
            elif geometry_type == "nurbs_surface":
                curved_components.append(component["name"])
                part_mesh = curved_surface_mesh(
                    [[_offset(_vec3(point), origin) for point in row] for row in geometry["control_points"]],
                    geometry["thickness"],
                )
            else:
                raise ValueError(f"Unsupported geometry type {geometry_type}.")
            assembly.merge(part_mesh)

        name = dsl.get("name", "generated_model").replace(" ", "_")
        stl_path = output / f"{name}.stl"
        obj_path = output / f"{name}.obj"
        dsl_path = output / f"{name}.dsl.json"
        assembly.write_stl(stl_path, name=name)
        assembly.write_obj(obj_path)
        dsl_path.write_text(json.dumps(dsl, indent=2), encoding="utf-8")
        return CompileResult(
            mesh=assembly,
            artifacts={"stl": stl_path, "obj": obj_path, "dsl": dsl_path},
            curved_components=curved_components,
        )


def _vec3(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"Expected [x, y, z], got {value!r}.")
    return float(value[0]), float(value[1]), float(value[2])


def _offset(point: tuple[float, float, float], origin: tuple[float, float, float]) -> tuple[float, float, float]:
    return point[0] + origin[0], point[1] + origin[1], point[2] + origin[2]
