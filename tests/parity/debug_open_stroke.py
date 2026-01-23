#!/usr/bin/env python3
"""Debug winding number behavior for OPEN stroked paths."""

import sys
sys.path.insert(0, '/workspace')

import torch
from diffvg_torch.scene import flatten_scene
from diffvg_torch.render import _compute_winding_number_py, _compute_closest_distance_py

def log(msg):
    print(f"[INFO] {msg}")


class MockPath:
    def __init__(self, points, num_control_points, is_closed, stroke_width):
        self.points = points
        self.num_control_points = num_control_points
        self.is_closed = is_closed
        self.stroke_width = stroke_width if isinstance(stroke_width, torch.Tensor) else torch.tensor([stroke_width])
        self.thickness = None


class MockShapeGroup:
    def __init__(self, shape_ids, fill_color, stroke_color):
        self.shape_ids = shape_ids if isinstance(shape_ids, torch.Tensor) else torch.tensor(shape_ids, dtype=torch.int32)
        self.fill_color = fill_color
        self.stroke_color = stroke_color
        self.use_even_odd_rule = True
        self.shape_to_canvas = None


def main():
    log("="*60)
    log("DEBUG: WINDING NUMBER FOR OPEN STROKED PATHS")
    log("="*60)

    # Same cubic bezier as diagnostic - OPEN path
    points = torch.tensor([
        [5.0, 14.0],   # Start
        [9.0, 5.0],    # Control 1
        [19.0, 5.0],   # Control 2
        [23.0, 14.0],  # End
    ], dtype=torch.float32)

    num_ctrl = torch.tensor([2], dtype=torch.int32)

    path = MockPath(
        points=points,
        num_control_points=num_ctrl,
        is_closed=False,  # OPEN path
        stroke_width=2.0
    )

    group = MockShapeGroup([0], None, torch.tensor([0.0, 0.0, 0.0, 1.0]))

    scene = flatten_scene(28, 28, [path], [group], device=torch.device('cpu'))

    log(f"\nPath is_closed in scene: {scene.paths.is_closed.tolist()}")
    log(f"Path stroke_width: {scene.paths.stroke_width.tolist()}")

    # Test points that showed discrepancy:
    # pydiffvg: pixel[9,14] = RGBA[0.0, 0.0, 0.0, 0.0]
    # triton:   pixel[9,14] = RGBA[0.0, 0.0, 0.0, 1.0]  <- should be 0!

    # pydiffvg: pixel[10,10] = RGBA[0.0, 0.0, 0.0, 0.25]
    # triton:   pixel[10,10] = RGBA[0.0, 0.0, 0.0, 1.0]  <- should be 0.25!

    test_points = [
        (7.5, 14.5),   # pixel[7,14] - on curve
        (8.5, 14.5),   # pixel[8,14] - on curve
        (9.5, 14.5),   # pixel[9,14] - should NOT be inside for open path
        (10.5, 10.5),  # pixel[10,10] - should be partial
        (14.5, 6.5),   # pixel[14,6] - near top of curve
    ]

    log("\n--- Testing winding numbers ---")
    stroke_width = scene.paths.stroke_width[0].item()
    half_width = stroke_width / 2.0

    for pt in test_points:
        winding = _compute_winding_number_py(pt, scene.paths, 0)
        dist, closest, t = _compute_closest_distance_py(pt, scene.paths, 0)

        is_inside = winding != 0
        within_stroke = dist <= half_width

        log(f"  Point {pt}:")
        log(f"    winding={winding}, is_inside={is_inside}")
        log(f"    dist={dist:.4f}, half_width={half_width:.4f}, within_stroke={within_stroke}")
        log(f"    closest_point={closest}, t={t:.4f}")
        log(f"    Should cover? = {is_inside or within_stroke}")


if __name__ == "__main__":
    main()
