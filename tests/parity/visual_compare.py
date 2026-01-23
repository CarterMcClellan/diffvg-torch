#!/usr/bin/env python3
"""Visual comparison of triton rendering with pydiffvg."""

import sys
sys.path.insert(0, '/workspace')

import torch
from diffvg_torch.scene import flatten_scene
from diffvg_torch.render import render_scene_py, RenderConfig


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
    log("VISUAL COMPARISON - TORCH OUTPUT")
    log("="*60)

    # Same cubic bezier - OPEN path
    points = torch.tensor([
        [5.0, 14.0],
        [9.0, 5.0],
        [19.0, 5.0],
        [23.0, 14.0],
    ], dtype=torch.float32)

    num_ctrl = torch.tensor([2], dtype=torch.int32)

    path = MockPath(
        points=points,
        num_control_points=num_ctrl,
        is_closed=False,
        stroke_width=2.0
    )

    group = MockShapeGroup([0], None, torch.tensor([0.0, 0.0, 0.0, 1.0]))

    width, height = 28, 28
    device = torch.device('cpu')
    scene = flatten_scene(width, height, [path], [group], device=device)

    config = RenderConfig(
        num_samples_x=2,
        num_samples_y=2,
    )

    img = render_scene_py(scene, config, pydiffvg_compatible=True)

    log(f"Output shape: {img.shape}")

    # Show alpha channel visualization (same format as pydiffvg test)
    log("\n--- Alpha channel (y=5-15, x=5-25) ---")
    log("(## = >0.9, ** = >0.5, .. = >0.1, space = <=0.1)")
    for y in range(5, 16):
        row = f"y={y:2d}: "
        for x in range(5, 26):
            a = img[y, x, 3].item()
            if a > 0.9:
                row += "##"
            elif a > 0.5:
                row += "**"
            elif a > 0.1:
                row += ".."
            else:
                row += "  "
        log(row)

    # Show test pixels
    log("\n--- Test pixels ---")
    test_pixels = [
        (7, 14),
        (8, 14),
        (9, 14),
        (10, 10),
        (14, 6),
    ]
    for y, x in test_pixels:
        rgba = img[y, x].tolist()
        log(f"  pixel[{y},{x}]: RGBA{[round(v, 4) for v in rgba]}")


if __name__ == "__main__":
    main()
