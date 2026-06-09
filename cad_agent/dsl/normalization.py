"""Normalization helpers for LLM-emitted Geometry DSL."""

from __future__ import annotations

from typing import Any


WRAPPER_KEYS = ("geometry_dsl", "dsl", "cad_dsl", "model", "result")
GEOMETRY_KEYS = ("geometry", "geometry_definition", "shape", "surface", "solid")
NAME_KEYS = ("name", "part", "component", "component_name", "id")


def normalize_geometry_dsl(value: dict[str, Any]) -> dict[str, Any]:
    """Unwrap and lightly normalize common LLM response shapes.

    The agents are instructed to return the DSL directly, but real model output
    often adds a wrapper key. This keeps validation focused on geometry quality
    instead of failing on harmless response packaging.
    """

    dsl = _unwrap(value)
    if not isinstance(dsl, dict):
        return value

    normalized = dict(dsl)
    normalized.setdefault("unit", "mm")
    components = normalized.get("components")
    if isinstance(components, list):
        normalized["components"] = [_normalize_component(component) for component in components]
    return normalized


def _unwrap(value: dict[str, Any]) -> dict[str, Any]:
    current = value
    seen: set[int] = set()
    while isinstance(current, dict) and id(current) not in seen:
        seen.add(id(current))
        for key in WRAPPER_KEYS:
            nested = current.get(key)
            if isinstance(nested, dict):
                current = nested
                break
        else:
            return current
    return current


def _normalize_component(component: Any) -> Any:
    if not isinstance(component, dict):
        return component
    normalized = dict(component)
    if "name" not in normalized:
        for key in NAME_KEYS:
            if isinstance(normalized.get(key), str):
                normalized["name"] = normalized[key]
                break

    if "geometry" not in normalized:
        for key in GEOMETRY_KEYS:
            geometry = normalized.get(key)
            if isinstance(geometry, dict):
                normalized["geometry"] = geometry
                break
    return normalized
