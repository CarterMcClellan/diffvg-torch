#!/usr/bin/env python3
"""
Diagnostic script for diffvg-torch rendering - outputs detailed info for parity comparison.

Run in diffvg-torch container:
    docker exec diffvg-torch python /workspace/tests/render_diagnostic_torch.py
"""

import sys
import os
import json
import numpy as np
import torch

# Ensure diffvg_torch is importable
sys.path.insert(0, '/workspace')

def log(msg, level="INFO"):
    print(f"[{level}] {msg}")


class MockPath:
    """Mock Path for diffvg-torch."""
    def __init__(self, points, num_control_points, is_closed, stroke_width):
        self.points = points
        self.num_control_points = num_control_points
        self.is_closed = is_closed
        self.stroke_width = stroke_width if isinstance(stroke_width, torch.Tensor) else torch.tensor([stroke_width])
        self.thickness = None


class MockShapeGroup:
    """Mock ShapeGroup for diffvg-torch."""
    def __init__(self, shape_ids, fill_color, stroke_color):
        self.shape_ids = shape_ids if isinstance(shape_ids, torch.Tensor) else torch.tensor(shape_ids, dtype=torch.int32)
        self.fill_color = fill_color
        self.stroke_color = stroke_color
        self.use_even_odd_rule = True
        self.shape_to_canvas = None


def create_test_scene():
    """Create a simple stroked cubic bezier - same as pydiffvg test."""
    points = torch.tensor([
        [5.0, 14.0],   # Start
        [9.0, 5.0],    # Control 1
        [19.0, 5.0],   # Control 2
        [23.0, 14.0],  # End
    ], dtype=torch.float32)

    num_control_points = torch.tensor([2], dtype=torch.int32)  # 2 = cubic

    path = MockPath(
        points=points,
        num_control_points=num_control_points,
        is_closed=False,
        stroke_width=2.0
    )

    group = MockShapeGroup(
        shape_ids=[0],
        fill_color=None,
        stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0])  # Black stroke
    )

    return [path], [group], points


def render_scene(shapes, groups, width, height, num_samples=2):
    """Render using diffvg-torch."""
    from diffvg_torch.scene import flatten_scene
    from diffvg_torch.render import render_scene_py, RenderConfig

    device = torch.device('cpu')
    scene = flatten_scene(width, height, shapes, groups, device=device)

    config = RenderConfig(
        num_samples_x=num_samples,
        num_samples_y=num_samples,
        background_color=(1.0, 1.0, 1.0, 1.0),
    )

    img = render_scene_py(scene, config)
    return img


def main():
    from diffvg_torch.scene import flatten_scene
    from diffvg_torch.render import RenderConfig

    log("="*60)
    log("DIFFVG-TORCH RENDER DIAGNOSTIC")
    log("="*60)

    width, height = 28, 28
    num_samples = 2

    log(f"Canvas: {width}x{height}")
    log(f"Samples: {num_samples}x{num_samples}")

    # Create scene
    shapes, groups, points = create_test_scene()

    log(f"\nScene:")
    log(f"  Points: {points.tolist()}")
    log(f"  Stroke width: {shapes[0].stroke_width}")
    log(f"  Stroke color: {groups[0].stroke_color.tolist()}")
    log(f"  num_control_points: {shapes[0].num_control_points.tolist()}")

    # Flatten scene
    device = torch.device('cpu')
    scene = flatten_scene(width, height, shapes, groups, device=device)

    log(f"\nFlattened scene:")
    log(f"  paths.num_paths: {scene.paths.num_paths}")
    log(f"  paths.points: {scene.paths.points.tolist()}")
    log(f"  paths.segment_types: {scene.paths.segment_types.tolist()}")
    log(f"  paths.num_segments: {scene.paths.num_segments.tolist()}")
    log(f"  paths.stroke_width: {scene.paths.stroke_width.tolist()}")
    log(f"  groups.stroke_color: {scene.groups.stroke_color.tolist()}")
    log(f"  groups.has_stroke: {scene.groups.has_stroke.tolist()}")

    # Render
    log("\nRendering...")
    img = render_scene(shapes, groups, width, height, num_samples)
    log(f"Output shape: {img.shape}")
    log(f"Output range: [{img.min():.4f}, {img.max():.4f}]")

    # Output pixel values for comparison
    log("\n" + "="*60)
    log("PIXEL VALUES (for parity comparison)")
    log("="*60)

    result = {
        "width": width,
        "height": height,
        "num_samples": num_samples,
        "pixels": {}
    }

    # Output pixels in the region of interest
    log("\nPixels near curve (y=5-15, x=5-25):")
    for y in range(5, 16):
        row_str = f"y={y:2d}: "
        for x in range(5, 26):
            val = img[y, x, 0].item()  # Red channel
            result["pixels"][f"{y},{x}"] = round(val, 4)
            if val < 0.99:
                row_str += f"{val:.2f} "
            else:
                row_str += ".... "
        log(row_str)

    # Save full image as numpy
    output_dir = "/workspace/tests/results"
    os.makedirs(output_dir, exist_ok=True)

    img_np = img.numpy()
    np.save(f"{output_dir}/torch_output.npy", img_np)
    log(f"\nSaved full output to {output_dir}/torch_output.npy")

    # Save as JSON
    with open(f"{output_dir}/torch_pixels.json", "w") as f:
        json.dump(result, f, indent=2)
    log(f"Saved pixel data to {output_dir}/torch_pixels.json")

    # Print specific test pixels
    log("\n" + "="*60)
    log("SPECIFIC TEST PIXELS")
    log("="*60)

    test_pixels = [
        (7, 14),   # Should be on the curve
        (8, 14),   # Should be on the curve
        (9, 14),   # Should be on the curve
        (10, 10),  # Near curve
        (14, 6),   # Top of curve
        (14, 7),   # Just below top
    ]

    for y, x in test_pixels:
        rgba = img[y, x].tolist()
        log(f"  pixel[{y},{x}] = RGBA{[round(v, 4) for v in rgba]}")


if __name__ == "__main__":
    main()
