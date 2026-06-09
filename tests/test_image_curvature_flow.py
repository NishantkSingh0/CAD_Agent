from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from cad_agent.pipeline import generate_cad


class FakeGeminiProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate_json(
        self,
        *,
        stage: str,
        system_prompt: str,
        payload: dict[str, Any],
        image_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        self.calls.append({"stage": stage, "image_paths": image_paths or [], "payload": payload})
        if stage == "planner":
            return {
                "object": "chair",
                "style": "scandinavian",
                "components": ["seat", "backrest", "armrest_left", "armrest_right", "legs"],
                "image_notes": [
                    "reference image indicates a rounded back and continuous curved armrests"
                ],
            }
        if stage == "topology":
            return {
                "seat": {"connects_to": ["backrest", "armrest_left", "armrest_right", "front_left_leg"]},
                "backrest": {"connects_to": ["seat"]},
                "armrest_left": {"connects_to": ["seat", "backrest"]},
                "armrest_right": {"connects_to": ["seat", "backrest"]},
            }
        if stage == "dimension":
            return {
                "overall_width": 760,
                "overall_depth": 780,
                "seat_height": 430,
                "arm_height": 610,
                "backrest_height": 900,
            }
        if stage == "surface":
            return _curved_chair_dsl()
        raise AssertionError(f"Unexpected stage: {stage}")


class WrappedDslGeminiProvider(FakeGeminiProvider):
    def generate_json(
        self,
        *,
        stage: str,
        system_prompt: str,
        payload: dict[str, Any],
        image_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        if stage == "surface":
            return {"geometry_dsl": _curved_chair_dsl()}
        return super().generate_json(
            stage=stage,
            system_prompt=system_prompt,
            payload=payload,
            image_paths=image_paths,
        )


class ImageCurvatureFlowTest(unittest.TestCase):
    def test_image_aware_pipeline_outputs_curved_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "rounded_chair_reference.ppm"
            image_path.write_text("P3\n2 2\n255\n255 255 255 0 0 0 0 0 0 255 255 255\n", encoding="ascii")

            provider = FakeGeminiProvider()
            result = generate_cad(
                "Build a Scandinavian lounge chair with curved armrests and a rounded backrest.",
                image_paths=[str(image_path)],
                output_dir=temp_path / "out",
                provider=provider,
            )

            self.assertEqual(provider.calls[0]["image_paths"], [str(image_path)])
            self.assertTrue(all(call["image_paths"] == [] for call in provider.calls[1:]))
            self.assertTrue(result.validation.ok)
            self.assertIn("backrest", result.compile_result.curved_components)
            self.assertIn("armrest_left", result.compile_result.curved_components)
            self.assertGreater(len(result.compile_result.mesh.vertices), 200)
            self.assertGreater(len(result.compile_result.mesh.faces), 300)
            self.assertTrue(result.compile_result.artifacts["stl"].exists())
            self.assertTrue(result.compile_result.artifacts["obj"].exists())
            self.assertTrue(result.compile_result.artifacts["dsl"].exists())

            curved_types = {
                component["name"]: component["geometry"]["type"]
                for component in result.dsl["components"]
                if component["geometry"]["type"] in {"bezier_sweep", "nurbs_surface"}
            }
            self.assertEqual(curved_types["backrest"], "nurbs_surface")
            self.assertEqual(curved_types["armrest_left"], "bezier_sweep")
            self.assertIn("reference_image_features", result.dsl)

    def test_pipeline_accepts_geometry_dsl_wrapper_from_gemini(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "rounded_chair_reference.ppm"
            image_path.write_text("P3\n1 1\n255\n255 255 255\n", encoding="ascii")

            result = generate_cad(
                "Create this standard premium chair.",
                image_paths=[str(image_path)],
                output_dir=temp_path / "out",
                provider=WrappedDslGeminiProvider(),
                image_policy="all",
            )

            self.assertEqual(result.dsl["unit"], "mm")
            self.assertIn("components", result.dsl)
            self.assertTrue(result.compile_result.artifacts["stl"].exists())


def _curved_chair_dsl() -> dict[str, Any]:
    return {
        "name": "scandinavian_curved_lounge_chair",
        "unit": "mm",
        "reference_image_features": ["rounded back silhouette", "sweeping curved armrests"],
        "components": [
            {
                "name": "seat",
                "material": "fabric",
                "origin": [0, 0, 420],
                "connects_to": ["backrest", "armrest_left", "armrest_right"],
                "geometry": {
                    "type": "rounded_box",
                    "width": 620,
                    "depth": 560,
                    "height": 90,
                    "corner_radius": 45,
                },
            },
            {
                "name": "backrest",
                "material": "bent plywood",
                "origin": [0, 230, 480],
                "connects_to": ["seat"],
                "geometry": {
                    "type": "nurbs_surface",
                    "thickness": 28,
                    "control_points": [
                        [[-300, 0, 0], [0, -34, 20], [300, 0, 0]],
                        [[-280, 35, 190], [0, -70, 230], [280, 35, 190]],
                        [[-230, 60, 390], [0, -25, 440], [230, 60, 390]],
                    ],
                },
            },
            {
                "name": "armrest_left",
                "material": "wood",
                "origin": [-360, 0, 455],
                "connects_to": ["seat", "backrest"],
                "geometry": {
                    "type": "bezier_sweep",
                    "thickness": 42,
                    "radius": 21,
                    "control_points": [[0, -230, 0], [-25, 20, 120], [0, 250, 190]],
                },
            },
            {
                "name": "armrest_right",
                "material": "wood",
                "origin": [360, 0, 455],
                "connects_to": ["seat", "backrest"],
                "geometry": {
                    "type": "bezier_sweep",
                    "thickness": 42,
                    "radius": 21,
                    "control_points": [[0, -230, 0], [25, 20, 120], [0, 250, 190]],
                },
            },
            {
                "name": "front_left_leg",
                "material": "wood",
                "origin": [-245, -205, 0],
                "connects_to": ["seat"],
                "geometry": {"type": "tapered_cylinder", "radius_top": 28, "radius_bottom": 18, "height": 420},
            },
            {
                "name": "front_right_leg",
                "material": "wood",
                "origin": [245, -205, 0],
                "connects_to": ["seat"],
                "geometry": {"type": "tapered_cylinder", "radius_top": 28, "radius_bottom": 18, "height": 420},
            },
            {
                "name": "rear_left_leg",
                "material": "wood",
                "origin": [-245, 205, 0],
                "connects_to": ["seat", "backrest"],
                "geometry": {"type": "tapered_cylinder", "radius_top": 28, "radius_bottom": 18, "height": 430},
            },
            {
                "name": "rear_right_leg",
                "material": "wood",
                "origin": [245, 205, 0],
                "connects_to": ["seat", "backrest"],
                "geometry": {"type": "tapered_cylinder", "radius_top": 28, "radius_bottom": 18, "height": 430},
            },
        ],
    }


if __name__ == "__main__":
    unittest.main()
