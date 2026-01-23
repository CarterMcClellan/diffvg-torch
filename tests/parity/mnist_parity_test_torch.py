#!/usr/bin/env python3
"""
Direct comparison of MNIST VAE rendering for diffvg-torch.

Loads the same test parameters saved by pydiffvg test and renders with torch.
"""

import sys
sys.path.insert(0, '/workspace')

import torch
import numpy as np
import json
import os

from diffvg_torch.scene import flatten_scene
from diffvg_torch.render import render_scene_py, RenderConfig


def log(msg):
    print(f"[INFO] {msg}")


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


def create_mnist_vae_scene_torch(points, stroke_width, alpha, num_segments):
    """Create scene exactly as mnist_vae.py does for triton."""
    num_ctrl_pts = torch.zeros(num_segments, dtype=torch.int32) + 2  # All cubic

    color = torch.cat([torch.ones(3), alpha.view(1,)])  # RGB=white, A=variable

    path = MockPath(
        points=points,
        num_control_points=num_ctrl_pts,
        is_closed=False,
        stroke_width=stroke_width
    )

    path_group = MockShapeGroup(
        shape_ids=[0],
        fill_color=None,
        stroke_color=color
    )

    return [path], [path_group]


def render_torch(shapes, groups, width, height, num_samples=4):
    """Render using diffvg-torch."""
    device = torch.device('cpu')
    scene = flatten_scene(width, height, shapes, groups, device=device)

    config = RenderConfig(
        num_samples_x=num_samples,
        num_samples_y=num_samples,
        background_color=(1.0, 1.0, 1.0, 1.0),  # White background
    )

    img = render_scene_py(scene, config)
    return img


def process_torch_output(img, imsize):
    """Process output exactly as pydiffvg version does."""
    # Torch format, discard alpha, make gray
    out = img.permute(2, 0, 1).view(4, imsize, imsize)[:3].mean(0, keepdim=True)
    # Map to [-1, 1]
    out = out * 2.0 - 1.0
    return out


def main():
    log("="*60)
    log("MNIST VAE PARITY TEST - DIFFVG-TORCH")
    log("="*60)

    # Load test parameters from pydiffvg test
    input_dir = "/workspace/tests/results"

    params_file = f"{input_dir}/mnist_test_params.json"
    if not os.path.exists(params_file):
        log(f"ERROR: {params_file} not found. Run mnist_parity_test.py in diffvg container first.")
        return

    with open(params_file) as f:
        params = json.load(f)

    points = torch.tensor(params["points"], dtype=torch.float32)
    stroke_width = torch.tensor(params["stroke_width"])
    alpha = torch.tensor(params["alpha"])
    imsize = params["imsize"]
    num_segments = params["num_segments"]
    num_samples = params["num_samples"]

    log(f"\nTest scene (loaded from pydiffvg test):")
    log(f"  imsize: {imsize}")
    log(f"  num_segments: {num_segments}")
    log(f"  num_samples: {num_samples}")
    log(f"  stroke_width: {stroke_width.item()}")
    log(f"  alpha: {alpha.item()}")
    log(f"  points shape: {points.shape}")

    # Create and render
    shapes, groups = create_mnist_vae_scene_torch(points, stroke_width, alpha, num_segments)

    log("\nRendering with diffvg-torch...")
    raw_img = render_torch(shapes, groups, imsize, imsize, num_samples)
    processed = process_torch_output(raw_img.detach(), imsize)

    log(f"  Raw output shape: {raw_img.shape}")
    log(f"  Processed output shape: {processed.shape}")
    log(f"  Processed range: [{processed.min():.4f}, {processed.max():.4f}]")

    # Convert processed to [0, 1] for display
    display = (processed + 1) / 2  # [0, 1]
    display = display.squeeze()  # [H, W]

    log("\nProcessed output (grayscale, mapped to 0-1):")
    for y in range(0, 28, 4):
        row = ""
        for x in range(28):
            v = display[y, x].item()
            if v < 0.3:
                row += "##"
            elif v < 0.7:
                row += ".."
            else:
                row += "  "
        log(f"  y={y:2d}: {row}")

    # Save results
    np.save(f"{input_dir}/mnist_torch_raw.npy", raw_img.numpy())
    np.save(f"{input_dir}/mnist_torch_processed.npy", processed.numpy())

    log(f"\nSaved results to {input_dir}/")

    # Raw channel statistics
    log("\nRaw output channel statistics:")
    for c, name in enumerate(['R', 'G', 'B', 'A']):
        ch = raw_img[:, :, c]
        log(f"  {name}: min={ch.min():.4f}, max={ch.max():.4f}, mean={ch.mean():.4f}")

    # Compare with pydiffvg if available
    pydiffvg_file = f"{input_dir}/mnist_pydiffvg_processed.npy"
    if os.path.exists(pydiffvg_file):
        log("\n" + "="*60)
        log("COMPARISON WITH PYDIFFVG")
        log("="*60)

        pydiffvg_proc = np.load(pydiffvg_file)
        torch_proc = processed.numpy()

        diff = np.abs(pydiffvg_proc - torch_proc)
        log(f"\nProcessed output difference:")
        log(f"  Max diff: {diff.max():.4f}")
        log(f"  Mean diff: {diff.mean():.4f}")
        log(f"  Pixels with diff > 0.1: {(diff > 0.1).sum()}")

        # Visual diff
        log("\nDifference map (## = large diff, .. = small diff):")
        diff_2d = diff.squeeze()
        for y in range(0, 28, 4):
            row = ""
            for x in range(28):
                d = diff_2d[y, x]
                if d > 0.5:
                    row += "##"
                elif d > 0.1:
                    row += ".."
                else:
                    row += "  "
            log(f"  y={y:2d}: {row}")
    else:
        log(f"\nNote: {pydiffvg_file} not found, skipping comparison")


if __name__ == "__main__":
    main()
