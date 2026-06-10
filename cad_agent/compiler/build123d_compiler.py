"""Build123D/OpenCascade compiler for solid CAD artifacts."""

from __future__ import annotations

import importlib
import math
import os
import warnings
from pathlib import Path
from typing import Any

from cad_agent.compiler.mesh_compiler import CompileResult, MeshCompiler
from cad_agent.dsl.validation import validate_geometry_dsl

Vec3 = tuple[float, float, float]


class Build123DCompiler:
    """Compile supported DSL components to OpenCascade STEP/STL artifacts.

    Curved upholstery surfaces still use the mesh fallback for preview output in
    this iteration. Solid components are exported as real CAD solids in STEP.
    """

    def compile(self, dsl: dict[str, Any], output_dir: Path | str) -> CompileResult:
        validate_geometry_dsl(dsl).require_ok()
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("XDG_CACHE_HOME", str(output / ".cache"))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"build123d\..*")
            b = importlib.import_module("build123d")

        solids: list[Any] = []
        for component in dsl["components"]:
            geometry = component["geometry"]
            geometry_type = geometry["type"]
            origin = _vec3(component.get("origin", [0, 0, 0]))
            if geometry_type in {"box", "rounded_box"}:
                solid = _build_box(b, geometry, origin)
            elif geometry_type in {"cylinder", "tapered_cylinder"}:
                solid = _build_cylinder(b, geometry, origin)
            elif geometry_type == "sphere":
                solid = _build_sphere(b, geometry, origin)
            else:
                continue
            solids.append(solid)

        mesh_result = MeshCompiler().compile(dsl, output)
        if not solids:
            return mesh_result

        name = dsl.get("name", "generated_model").replace(" ", "_")
        compound = b.Compound(children=solids)
        step_path = output / f"{name}.step"
        cad_stl_path = output / f"{name}.cad.stl"
        b.export_step(compound, step_path)
        b.export_stl(compound, cad_stl_path)

        artifacts = dict(mesh_result.artifacts)
        artifacts["step"] = step_path
        artifacts["cad_stl"] = cad_stl_path
        return CompileResult(
            mesh=mesh_result.mesh,
            artifacts=artifacts,
            curved_components=mesh_result.curved_components,
        )


def is_build123d_available() -> bool:
    return importlib.util.find_spec("build123d") is not None


def _build_box(b: Any, geometry: dict[str, Any], origin: Vec3) -> Any:
    width = float(geometry["width"])
    depth = float(geometry["depth"])
    height = float(geometry["height"])
    solid = b.Box(width, depth, height)
    radius = float(geometry.get("corner_radius", 0) or 0)
    if radius > 0:
        safe_radius = min(radius, width, depth, height) * 0.45
        solid = b.fillet(solid.edges(), safe_radius)
    return solid.translate((origin[0], origin[1], origin[2] + height / 2))


def _build_cylinder(b: Any, geometry: dict[str, Any], origin: Vec3) -> Any:
    radius_bottom = float(geometry.get("radius_bottom", geometry.get("radius", 10)))
    radius_top = float(geometry.get("radius_top", geometry.get("radius", radius_bottom)))
    axis = _vec3(geometry["axis"]) if "axis" in geometry else (0, 0, float(geometry["height"]))
    height = _length(axis)
    radius = (radius_bottom + radius_top) / 2

    # Build123D cone signatures have changed across versions; a cylinder is
    # still a real CAD solid and avoids version-specific cone code here.
    solid = b.Cylinder(radius, height)
    solid = _align_z_to_axis(b, solid, axis)
    midpoint = (origin[0] + axis[0] / 2, origin[1] + axis[1] / 2, origin[2] + axis[2] / 2)
    return solid.translate(midpoint)


def _build_sphere(b: Any, geometry: dict[str, Any], origin: Vec3) -> Any:
    return b.Sphere(float(geometry["radius"])).translate(origin)


def _align_z_to_axis(b: Any, solid: Any, axis: Vec3) -> Any:
    target = _normalize(axis)
    z_axis = (0.0, 0.0, 1.0)
    dot = max(-1.0, min(1.0, _dot(z_axis, target)))
    angle = math.degrees(math.acos(dot))
    if angle < 1e-6:
        return solid
    rotation_axis = _cross(z_axis, target)
    if _length(rotation_axis) < 1e-6:
        rotation_axis = (1.0, 0.0, 0.0)
    return solid.rotate(b.Axis((0, 0, 0), rotation_axis), angle)


def _vec3(value: Any) -> Vec3:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError(f"Expected [x, y, z], got {value!r}.")
    return float(value[0]), float(value[1]), float(value[2])


def _length(value: Vec3) -> float:
    return math.sqrt(sum(item * item for item in value))


def _normalize(value: Vec3) -> Vec3:
    length = _length(value)
    if length < 1e-9:
        return (0.0, 0.0, 1.0)
    return value[0] / length, value[1] / length, value[2] / length


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )
