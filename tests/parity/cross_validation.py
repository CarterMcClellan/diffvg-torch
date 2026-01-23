#!/usr/bin/env python3
"""
Cross-validation script for comparing pydiffvg and diffvg-torch outputs.

This script should be run inside the diffvg container with diffvg-torch mounted.

Usage:
    docker exec -v /path/to/diffvg-torch:/diffvg-torch diffvg \
        python /diffvg-torch/tests/cross_validation.py
"""

import sys
import os
import torch
import numpy as np

# Add diffvg-torch to path if needed
TORCH_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TORCH_PATH not in sys.path:
    sys.path.insert(0, TORCH_PATH)


def render_with_pydiffvg(shapes, shape_groups, width, height, num_samples=2):
    """Render scene using original pydiffvg."""
    import pydiffvg

    pydiffvg.set_use_gpu(torch.cuda.is_available())

    scene_args = pydiffvg.RenderFunction.serialize_scene(
        width, height, shapes, shape_groups
    )

    render = pydiffvg.RenderFunction.apply
    img = render(
        width, height,
        num_samples, num_samples,
        0,  # seed
        None,  # background
        *scene_args
    )

    return img.detach().cpu()


def render_with_torch(shapes, shape_groups, width, height, num_samples=2):
    """Render scene using diffvg-torch."""
    from diffvg_torch.render import render

    img = render(
        canvas_width=width,
        canvas_height=height,
        shapes=shapes,
        shape_groups=shape_groups,
        num_samples_x=num_samples,
        num_samples_y=num_samples,
        seed=0,
        background_color=torch.tensor([1.0, 1.0, 1.0, 1.0])
    )

    return img.detach().cpu()


def create_test_path_scene():
    """Create a test scene with paths for both implementations."""
    import pydiffvg

    # Square path
    points = torch.tensor([
        [64.0, 64.0],
        [192.0, 64.0],
        [192.0, 192.0],
        [64.0, 192.0]
    ], dtype=torch.float32)

    num_control_points = torch.tensor([0, 0, 0, 0], dtype=torch.int32)

    path = pydiffvg.Path(
        num_control_points=num_control_points,
        points=points,
        is_closed=True,
        stroke_width=torch.tensor(1.0)
    )

    group = pydiffvg.ShapeGroup(
        shape_ids=torch.tensor([0]),
        fill_color=torch.tensor([0.0, 0.5, 1.0, 1.0])  # Blue
    )

    return [path], [group]


def create_test_bezier_scene():
    """Create a test scene with bezier curves."""
    import pydiffvg

    # Cubic bezier curve
    points = torch.tensor([
        [50.0, 200.0],   # Start
        [100.0, 50.0],   # Control 1
        [200.0, 50.0],   # Control 2
        [250.0, 200.0],  # End
    ], dtype=torch.float32)

    num_control_points = torch.tensor([2], dtype=torch.int32)  # Cubic

    path = pydiffvg.Path(
        num_control_points=num_control_points,
        points=points,
        is_closed=False,
        stroke_width=torch.tensor(5.0)
    )

    group = pydiffvg.ShapeGroup(
        shape_ids=torch.tensor([0]),
        fill_color=None,
        stroke_color=torch.tensor([1.0, 0.0, 0.0, 1.0])  # Red stroke
    )

    return [path], [group]


def compare_images(img1, img2, name=""):
    """Compare two images and report differences."""
    diff = torch.abs(img1 - img2)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Count pixels with significant difference
    significant_pixels = (diff > 0.01).sum().item()
    total_pixels = diff.numel()

    print(f"\n{name} Comparison:")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    print(f"  Significant pixels: {significant_pixels}/{total_pixels} ({100*significant_pixels/total_pixels:.2f}%)")

    return max_diff, mean_diff


def save_comparison(img1, img2, prefix):
    """Save images for visual comparison."""
    try:
        import pydiffvg

        pydiffvg.imwrite(img1, f'{prefix}_pydiffvg.png', gamma=2.2)
        pydiffvg.imwrite(img2, f'{prefix}_torch.png', gamma=2.2)

        # Difference image (amplified)
        diff = torch.abs(img1 - img2) * 10
        diff = diff.clamp(0, 1)
        pydiffvg.imwrite(diff, f'{prefix}_diff.png', gamma=1.0)

        print(f"  Saved: {prefix}_pydiffvg.png, {prefix}_torch.png, {prefix}_diff.png")
    except Exception as e:
        print(f"  Could not save images: {e}")


def run_cross_validation():
    """Run cross-validation tests."""
    print("=" * 60)
    print("Cross-validation: pydiffvg vs diffvg-torch")
    print("=" * 60)

    width, height = 256, 256
    num_samples = 2

    # Test 1: Filled square path
    print("\n[Test 1] Filled Square Path")
    shapes, groups = create_test_path_scene()

    print("  Rendering with pydiffvg...")
    img_diffvg = render_with_pydiffvg(shapes, groups, width, height, num_samples)

    print("  Rendering with diffvg-torch...")
    img_torch = render_with_torch(shapes, groups, width, height, num_samples)

    max_diff, mean_diff = compare_images(img_diffvg, img_torch, "Filled Square")
    save_comparison(img_diffvg, img_torch, "results/cross_val_square")

    # Test 2: Stroked bezier curve
    print("\n[Test 2] Stroked Bezier Curve")
    shapes, groups = create_test_bezier_scene()

    print("  Rendering with pydiffvg...")
    img_diffvg = render_with_pydiffvg(shapes, groups, width, height, num_samples)

    print("  Rendering with diffvg-torch...")
    img_torch = render_with_torch(shapes, groups, width, height, num_samples)

    max_diff, mean_diff = compare_images(img_diffvg, img_torch, "Bezier Curve")
    save_comparison(img_diffvg, img_torch, "results/cross_val_bezier")

    print("\n" + "=" * 60)
    print("Cross-validation complete!")
    print("=" * 60)


if __name__ == '__main__':
    # Create results directory
    os.makedirs('results', exist_ok=True)

    run_cross_validation()
