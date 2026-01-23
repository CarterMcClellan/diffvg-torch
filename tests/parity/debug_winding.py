#!/usr/bin/env python3
"""Debug winding number computation."""

import sys
sys.path.insert(0, '/workspace')

import torch
import json

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
    log("DEBUG WINDING NUMBER COMPUTATION")
    log("="*60)

    # Simple test: closed triangle
    log("\n--- TEST 1: Closed triangle ---")
    points_tri = torch.tensor([
        [5.0, 5.0],
        [23.0, 5.0],
        [14.0, 23.0],
    ], dtype=torch.float32)

    path_tri = MockPath(
        points=points_tri,
        num_control_points=torch.tensor([0, 0], dtype=torch.int32),  # 2 line segments
        is_closed=True,
        stroke_width=2.0
    )

    group = MockShapeGroup([0], None, torch.tensor([1.0, 1.0, 1.0, 1.0]))
    scene_tri = flatten_scene(28, 28, [path_tri], [group], device=torch.device('cpu'))

    test_points = [
        (14.0, 10.0),  # Inside triangle
        (14.0, 14.0),  # Inside triangle
        (5.0, 20.0),   # Outside triangle
        (25.0, 25.0),  # Outside triangle
    ]

    for pt in test_points:
        winding = _compute_winding_number_py(pt, scene_tri.paths, 0)
        dist, _, _ = _compute_closest_distance_py(pt, scene_tri.paths, 0)
        log(f"  Point {pt}: winding={winding}, dist={dist:.4f}")

    # Test 2: Open path (like MNIST VAE uses)
    log("\n--- TEST 2: Open cubic path ---")
    with open("/workspace/tests/results/mnist_test_params.json") as f:
        params = json.load(f)

    points = torch.tensor(params["points"], dtype=torch.float32)

    path_open = MockPath(
        points=points,
        num_control_points=torch.tensor([2, 2, 2], dtype=torch.int32),
        is_closed=False,  # OPEN path
        stroke_width=2.0
    )

    scene_open = flatten_scene(28, 28, [path_open], [group], device=torch.device('cpu'))

    log(f"  Path is_closed: {scene_open.paths.is_closed.tolist()}")

    test_points2 = [
        (14.0, 14.0),
        (10.0, 10.0),
        (20.0, 20.0),
        (5.0, 5.0),
    ]

    for pt in test_points2:
        winding = _compute_winding_number_py(pt, scene_open.paths, 0)
        dist, _, _ = _compute_closest_distance_py(pt, scene_open.paths, 0)
        log(f"  Point {pt}: winding={winding}, dist={dist:.4f}")

    # Test 3: Closed version of same path
    log("\n--- TEST 3: CLOSED version of same path ---")

    path_closed = MockPath(
        points=points,
        num_control_points=torch.tensor([2, 2, 2], dtype=torch.int32),
        is_closed=True,  # CLOSED path
        stroke_width=2.0
    )

    scene_closed = flatten_scene(28, 28, [path_closed], [group], device=torch.device('cpu'))

    log(f"  Path is_closed: {scene_closed.paths.is_closed.tolist()}")

    for pt in test_points2:
        winding = _compute_winding_number_py(pt, scene_closed.paths, 0)
        dist, _, _ = _compute_closest_distance_py(pt, scene_closed.paths, 0)
        log(f"  Point {pt}: winding={winding}, dist={dist:.4f}")


if __name__ == "__main__":
    main()
