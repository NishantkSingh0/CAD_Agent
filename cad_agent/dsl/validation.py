"""Validation for the intermediate geometry DSL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CURVED_TYPES = {"bezier_sweep", "nurbs_surface"}
SOLID_TYPES = {"rounded_box", "box", "tapered_cylinder", "cylinder"}


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    issues: list[str]

    def require_ok(self) -> None:
        if not self.ok:
            raise ValueError("; ".join(self.issues))


def validate_geometry_dsl(dsl: dict[str, Any]) -> ValidationReport:
    issues: list[str] = []
    if dsl.get("unit") != "mm":
        issues.append("DSL unit must be mm.")
    components = dsl.get("components")
    if not isinstance(components, list) or not components:
        issues.append("DSL must contain a non-empty components list.")
        return ValidationReport(False, issues)

    names: set[str] = set()
    curved_count = 0
    for index, component in enumerate(components):
        if not isinstance(component, dict):
            issues.append(f"Component {index} must be an object.")
            continue
        name = component.get("name")
        if not isinstance(name, str) or not name:
            issues.append(f"Component {index} is missing a name.")
        elif name in names:
            issues.append(f"Duplicate component name: {name}.")
        else:
            names.add(name)

        geometry = component.get("geometry")
        if not isinstance(geometry, dict):
            issues.append(f"Component {name or index} is missing geometry.")
            continue
        geometry_type = geometry.get("type")
        if geometry_type not in CURVED_TYPES | SOLID_TYPES:
            issues.append(f"Component {name or index} has unsupported geometry type {geometry_type!r}.")
        if geometry_type in CURVED_TYPES:
            curved_count += 1
            _validate_curved_geometry(name or str(index), geometry, issues)
        else:
            _validate_solid_geometry(name or str(index), geometry, issues)

        connects_to = component.get("connects_to", [])
        if not isinstance(connects_to, list):
            issues.append(f"Component {name or index} connects_to must be a list.")
        for target in connects_to:
            if not isinstance(target, str):
                issues.append(f"Component {name or index} has a non-string connection target.")

    for component in components:
        if isinstance(component, dict):
            for target in component.get("connects_to", []) or []:
                if target not in names:
                    issues.append(f"Component {component.get('name')} connects to unknown component {target}.")

    if len(components) > 1 and not any(component.get("connects_to") for component in components if isinstance(component, dict)):
        issues.append("At least one component must connect to another component.")
    if curved_count == 0:
        issues.append("DSL must include at least one curved geometry component.")

    return ValidationReport(not issues, issues)


def _validate_curved_geometry(name: str, geometry: dict[str, Any], issues: list[str]) -> None:
    points = geometry.get("control_points")
    if not isinstance(points, list) or len(points) < 3:
        issues.append(f"Curved component {name} needs at least 3 control points.")
    elif not _has_actual_curvature(points):
        issues.append(f"Curved component {name} control points are collinear.")
    thickness = geometry.get("thickness")
    if not _positive_number(thickness):
        issues.append(f"Curved component {name} needs positive thickness.")


def _validate_solid_geometry(name: str, geometry: dict[str, Any], issues: list[str]) -> None:
    required_by_type = {
        "box": ("width", "depth", "height"),
        "rounded_box": ("width", "depth", "height", "corner_radius"),
        "cylinder": ("radius", "height"),
        "tapered_cylinder": ("radius_top", "radius_bottom", "height"),
    }
    for field in required_by_type.get(geometry.get("type"), ()):
        if not _positive_number(geometry.get(field)):
            issues.append(f"Solid component {name} requires positive {field}.")


def _positive_number(value: Any) -> bool:
    return isinstance(value, int | float) and value > 0


def _has_actual_curvature(points: list[Any]) -> bool:
    flat_points = [_flatten_point(point) for point in points]
    flat_points = [point for point in flat_points if point is not None]
    if len(flat_points) < 3:
        return False
    a, b, c = flat_points[0], flat_points[len(flat_points) // 2], flat_points[-1]
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    return sum(value * value for value in cross) > 1e-6


def _flatten_point(point: Any) -> tuple[float, float, float] | None:
    if isinstance(point, list) and point and isinstance(point[0], list):
        point = point[0]
    if not isinstance(point, list) or len(point) != 3:
        return None
    if not all(isinstance(value, int | float) for value in point):
        return None
    return float(point[0]), float(point[1]), float(point[2])
