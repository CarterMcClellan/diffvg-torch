#!/usr/bin/env python3
"""Debug why stroke is not rendering in diffvg-torch."""

import sys
sys.path.insert(0, '/workspace')

import torch
import numpy as np
import json

from diffvg_torch.scene import flatten_scene, ShapeType
from diffvg_torch.render import render_scene_py, RenderConfig, _compute_closest_distance_py


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
    log("DEBUG STROKE RENDERING")
    log("="*60)

    # Load test params
    with open("/workspace/tests/results/mnist_test_params.json") as f:
        params = json.load(f)

    points = torch.tensor(params["points"], dtype=torch.float32)
    stroke_width = torch.tensor(params["stroke_width"])
    alpha = torch.tensor(params["alpha"])
    num_segments = params["num_segments"]
    imsize = 28

    log(f"\nPoints ({len(points)} points for {num_segments} cubic segments):")
    for i, pt in enumerate(points):
        log(f"  [{i}] ({pt[0]:.2f}, {pt[1]:.2f})")

    log(f"\nStroke width: {stroke_width.item()}")
    log(f"Alpha: {alpha.item()}")

    # Create scene
    num_ctrl_pts = torch.zeros(num_segments, dtype=torch.int32) + 2
    color = torch.cat([torch.ones(3), alpha.view(1,)])

    path = MockPath(
        points=points,
        num_control_points=num_ctrl_pts,
        is_closed=False,
        stroke_width=stroke_width
    )

    group = MockShapeGroup(
        shape_ids=[0],
        fill_color=None,
        stroke_color=color
    )

    # Flatten scene
    device = torch.device('cpu')
    scene = flatten_scene(imsize, imsize, [path], [group], device=device)

    log(f"\nFlattened scene:")
    log(f"  paths.num_paths: {scene.paths.num_paths}")
    log(f"  paths.num_segments: {scene.paths.num_segments.tolist()}")
    log(f"  paths.segment_types: {scene.paths.segment_types[0, :scene.paths.num_segments[0].item()].tolist()}")
    log(f"  paths.stroke_width: {scene.paths.stroke_width.tolist()}")
    log(f"  paths.points shape: {scene.paths.points.shape}")
    log(f"  paths.points: {scene.paths.points.tolist()}")

    log(f"\nGroups:")
    log(f"  num_groups: {scene.groups.num_groups}")
    log(f"  has_stroke: {scene.groups.has_stroke.tolist()}")
    log(f"  stroke_color: {scene.groups.stroke_color.tolist() if scene.groups.stroke_color is not None else None}")

    # Test distance computation at a few points
    log(f"\n" + "="*60)
    log("DISTANCE COMPUTATION TEST")
    log("="*60)

    test_points = [
        (14.0, 14.0),  # Center of image
        (15.0, 13.5),  # Near first point
        (16.0, 19.0),  # Near second point
        (10.0, 10.0),  # Somewhere else
    ]

    for pt in test_points:
        dist, closest, t = _compute_closest_distance_py(pt, scene.paths, 0)
        log(f"  Point {pt}: dist={dist:.4f}, closest={closest}, t={t:.4f}")
        log(f"    Inside stroke (dist <= {stroke_width.item()/2})? {dist <= stroke_width.item()/2}")

    # Check what segments we actually have
    log(f"\n" + "="*60)
    log("SEGMENT ANALYSIS")
    log("="*60)

    seg_types = scene.paths.segment_types[0].cpu().numpy()
    num_segs = scene.paths.num_segments[0].item()
    pts = scene.paths.points.cpu().numpy()

    log(f"Number of segments: {num_segs}")

    current_point = 0
    for seg_idx in range(num_segs):
        seg_type = seg_types[seg_idx]
        type_name = ["LINE", "QUADRATIC", "CUBIC"][seg_type]

        if seg_type == 2:  # Cubic
            p0 = pts[current_point]
            p1 = pts[current_point + 1]
            p2 = pts[current_point + 2]
            p3 = pts[current_point + 3]
            log(f"  Segment {seg_idx} ({type_name}):")
            log(f"    P0: ({p0[0]:.2f}, {p0[1]:.2f})")
            log(f"    P1: ({p1[0]:.2f}, {p1[1]:.2f})")
            log(f"    P2: ({p2[0]:.2f}, {p2[1]:.2f})")
            log(f"    P3: ({p3[0]:.2f}, {p3[1]:.2f})")
            current_point += 3
        else:
            log(f"  Segment {seg_idx}: unexpected type {seg_type}")

    # Now render and check
    log(f"\n" + "="*60)
    log("RENDERING")
    log("="*60)

    config = RenderConfig(
        num_samples_x=2,
        num_samples_y=2,
        background_color=(1.0, 1.0, 1.0, 1.0),
    )

    img = render_scene_py(scene, config)

    log(f"Output shape: {img.shape}")
    log(f"Output range: [{img.min():.4f}, {img.max():.4f}]")

    # Find non-white pixels
    non_white = (img[:, :, 0] < 0.99) | (img[:, :, 3] < 0.99)
    num_non_white = non_white.sum().item()
    log(f"Non-white pixels: {num_non_white}")

    if num_non_white > 0:
        ys, xs = torch.where(non_white)
        log(f"Sample non-white locations: {list(zip(ys[:5].tolist(), xs[:5].tolist()))}")


if __name__ == "__main__":
    main()
