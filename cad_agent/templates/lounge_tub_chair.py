"""Deterministic lounge/tub chair template."""

from __future__ import annotations

from typing import Any

from cad_agent.memory import DesignMemory


def build_lounge_tub_chair_dsl(memory: DesignMemory) -> dict[str, Any]:
    """Build a stable chair DSL from shared design memory.

    The template captures the visible reference class: upholstered wraparound
    shell, separate cushion, angled round wooden legs, and rectangular rails.
    """

    width = _clamp(memory.overall("width", memory.overall("overall_width", 820)), 680, 920)
    depth = _clamp(memory.overall("depth", memory.overall("overall_depth", 760)), 650, 880)
    height = _clamp(memory.overall("height", memory.overall("overall_height", 880)), 760, 980)

    seat_height = _clamp(_dimension(memory, "seat_height", 430), 380, 470)
    cushion_height = _clamp(_dimension(memory, "cushion_thickness", 95), 70, 120)
    shell_thickness = _clamp(_dimension(memory, "shell_thickness", 42), 32, 58)
    leg_radius_top = _clamp(_dimension(memory, "leg_radius_top", 32), 24, 42)
    leg_radius_bottom = _clamp(_dimension(memory, "leg_radius_bottom", 22), 16, 34)

    half_w = width / 2
    half_d = depth / 2
    leg_z = seat_height - 45
    back_height = height - seat_height

    shell_grid = _shell_control_grid(half_w, half_d, back_height)

    return {
        "name": _name_from_prompt(memory.prompt),
        "unit": "mm",
        "template": memory.template,
        "assumptions": memory.assumptions,
        "reference_image_features": _image_features(memory),
        "components": [
            _leg("front_left_leg", [-half_w + 75, -half_d + 75, 0], [-42, -70, leg_z], leg_radius_top, leg_radius_bottom),
            _leg("front_right_leg", [half_w - 75, -half_d + 75, 0], [42, -70, leg_z], leg_radius_top, leg_radius_bottom),
            _leg("rear_left_leg", [-half_w + 95, half_d - 90, 0], [-58, 86, leg_z + 185], leg_radius_top, leg_radius_bottom),
            _leg("rear_right_leg", [half_w - 95, half_d - 90, 0], [58, 86, leg_z + 185], leg_radius_top, leg_radius_bottom),
            {
                "name": "left_side_rail",
                "material": "wood",
                "origin": [-half_w + 70, 0, seat_height - 120],
                "connects_to": ["front_left_leg", "rear_left_leg"],
                "geometry": {"type": "box", "width": 48, "depth": depth - 170, "height": 34},
            },
            {
                "name": "right_side_rail",
                "material": "wood",
                "origin": [half_w - 70, 0, seat_height - 120],
                "connects_to": ["front_right_leg", "rear_right_leg"],
                "geometry": {"type": "box", "width": 48, "depth": depth - 170, "height": 34},
            },
            {
                "name": "front_rail",
                "material": "wood",
                "origin": [0, -half_d + 75, seat_height - 120],
                "connects_to": ["front_left_leg", "front_right_leg"],
                "geometry": {"type": "rounded_box", "width": width - 160, "depth": 45, "height": 34, "corner_radius": 10},
            },
            {
                "name": "rear_rail",
                "material": "wood",
                "origin": [0, half_d - 85, seat_height - 105],
                "connects_to": ["rear_left_leg", "rear_right_leg"],
                "geometry": {"type": "rounded_box", "width": width - 190, "depth": 45, "height": 34, "corner_radius": 10},
            },
            {
                "name": "upholstered_wraparound_shell",
                "material": "upholstery",
                "origin": [0, 0, seat_height - 45],
                "connects_to": ["front_left_leg", "front_right_leg", "rear_left_leg", "rear_right_leg", "seat_cushion"],
                "geometry": {
                    "type": "nurbs_surface",
                    "thickness": shell_thickness,
                    "control_points": shell_grid,
                },
            },
            {
                "name": "seat_cushion",
                "material": "upholstery",
                "origin": [0, -half_d * 0.18, seat_height - 5],
                "connects_to": ["upholstered_wraparound_shell"],
                "geometry": {
                    "type": "rounded_box",
                    "width": width - 210,
                    "depth": depth * 0.58,
                    "height": cushion_height,
                    "corner_radius": 38,
                },
            },
        ],
    }


def _leg(
    name: str,
    origin: list[float],
    axis: list[float],
    radius_top: float,
    radius_bottom: float,
) -> dict[str, Any]:
    return {
        "name": name,
        "material": "wood",
        "origin": origin,
        "connects_to": ["left_side_rail" if "left" in name else "right_side_rail"],
        "geometry": {
            "type": "tapered_cylinder",
            "axis": axis,
            "radius_top": radius_top,
            "radius_bottom": radius_bottom,
            "height": _length(axis),
        },
    }


def _shell_control_grid(half_w: float, half_d: float, back_height: float) -> list[list[list[float]]]:
    seat_y = -half_d * 0.42
    back_y = half_d * 0.36
    return [
        [
            [-half_w + 42, seat_y, 95],
            [-half_w * 0.34, seat_y - 22, 35],
            [half_w * 0.34, seat_y - 22, 35],
            [half_w - 42, seat_y, 95],
        ],
        [
            [-half_w + 22, -half_d * 0.16, 170],
            [-half_w * 0.46, -half_d * 0.12, 5],
            [half_w * 0.46, -half_d * 0.12, 5],
            [half_w - 22, -half_d * 0.16, 170],
        ],
        [
            [-half_w + 5, half_d * 0.08, 300],
            [-half_w * 0.62, half_d * 0.18, 130],
            [half_w * 0.62, half_d * 0.18, 130],
            [half_w - 5, half_d * 0.08, 300],
        ],
        [
            [-half_w + 45, back_y, back_height * 0.78],
            [-half_w * 0.42, back_y + 34, back_height * 0.72],
            [half_w * 0.42, back_y + 34, back_height * 0.72],
            [half_w - 45, back_y, back_height * 0.78],
        ],
        [
            [-half_w + 105, back_y + 28, back_height],
            [-half_w * 0.26, back_y + 48, back_height * 0.98],
            [half_w * 0.26, back_y + 48, back_height * 0.98],
            [half_w - 105, back_y + 28, back_height],
        ],
    ]


def _image_features(memory: DesignMemory) -> list[str]:
    observations = memory.planner.get("image_observations", [])
    if isinstance(observations, list) and observations:
        return [str(item) for item in observations[:8]]
    return [
        "wraparound upholstered back and arms",
        "separate rounded seat cushion",
        "angled round wood legs",
        "rectangular lower wooden rail frame",
    ]


def _dimension(memory: DesignMemory, key: str, default: float) -> float:
    for source in (memory.dimensions, memory.planner):
        value = source.get(key)
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, dict):
            nested = value.get(key)
            if isinstance(nested, int | float):
                return float(nested)
    return default


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _length(axis: list[float]) -> float:
    return sum(value * value for value in axis) ** 0.5


def _name_from_prompt(prompt: str) -> str:
    words = [word.strip(".,:;!?()[]{}").lower() for word in prompt.split()]
    useful = [word for word in words if word and word not in {"create", "this", "the", "a", "an"}]
    return "_".join(useful[:6] or ["generated_chair"])
