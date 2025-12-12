#!/usr/bin/env python3
"""
VAE Parity Test for diffvg-triton vs pydiffvg.

This script runs in either container and produces comparable outputs
for diagnosing rendering and gradient flow differences.

Usage:
    # In diffvg-triton container:
    docker exec diffvg-triton python /workspace/tests/vae_parity_test.py

    # In diffvg container:
    docker exec diffvg python /workspace/tests/vae_parity_test.py
"""

import os
import sys
import json
import argparse
import numpy as np
import torch as th

# Detect which backend we're using
try:
    import pydiffvg
    BACKEND = "pydiffvg"
except ImportError:
    BACKEND = "triton"

print(f"[INFO] Detected backend: {BACKEND}")

# Output directory - use container-specific subdirectory
OUTPUT_DIR = f"/workspace/tests/results/{BACKEND}_parity"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def log(msg):
    print(f"[{BACKEND}] {msg}")


def save_image(tensor, path, gamma=2.2):
    """Save tensor as PNG image."""
    try:
        from PIL import Image
    except ImportError:
        log(f"PIL not available, skipping image save: {path}")
        return

    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 4:  # [B, C, H, W]
        arr = arr[0, 0]  # Take first batch, first channel
    elif arr.ndim == 3 and arr.shape[0] in [1, 3, 4]:  # [C, H, W]
        arr = arr[0]  # Take first channel

    # Apply gamma
    arr = np.clip(arr, 0, 1)
    arr = np.power(arr, 1.0 / gamma)
    arr = (arr * 255).astype(np.uint8)

    Image.fromarray(arr, mode='L').save(path)
    log(f"Saved: {path}")


def render_pydiffvg(canvas_size, all_points, all_widths, all_alphas, num_segments, samples=2):
    """Render using pydiffvg backend."""
    import pydiffvg

    bs = all_points.shape[0]
    num_paths = all_points.shape[1]
    outputs = []

    for b in range(bs):
        shapes = []
        shape_groups = []

        for p in range(num_paths):
            points = all_points[b, p].contiguous().cpu()
            width = all_widths[b, p].cpu()
            alpha = all_alphas[b, p].cpu()

            color = th.cat([th.ones(3), alpha.view(1,)])
            num_ctrl_pts = th.zeros(num_segments, dtype=th.int32) + 2

            path = pydiffvg.Path(
                num_control_points=num_ctrl_pts,
                points=points,
                stroke_width=width,
                is_closed=False
            )
            shapes.append(path)

            path_group = pydiffvg.ShapeGroup(
                shape_ids=th.tensor([len(shapes) - 1]),
                fill_color=None,
                stroke_color=color
            )
            shape_groups.append(path_group)

        # Render
        _render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_size, canvas_size, shapes, shape_groups)
        img = _render(canvas_size, canvas_size, samples, samples, 0, None, *scene_args)

        # Convert to grayscale [H, W, 4] -> [1, H, W]
        img = img.permute(2, 0, 1)[:3].mean(0, keepdim=True)
        outputs.append(img)

    output = th.stack(outputs)  # [B, 1, H, W]
    return output


def render_triton(canvas_size, all_points, all_widths, all_alphas, num_segments, samples=2):
    """Render using diffvg-triton backend."""
    sys.path.insert(0, '/workspace')
    from diffvg_triton.render_batch import render_batch_fast

    bs = all_points.shape[0]
    num_paths = all_points.shape[1]
    device = all_points.device

    # Convert points to control_points format [B, P, S, 4, 2]
    control_points = []
    for seg_idx in range(num_segments):
        start_idx = seg_idx * 3
        seg_points = all_points[:, :, start_idx:start_idx+4, :]  # [B, P, 4, 2]
        control_points.append(seg_points)
    control_points = th.stack(control_points, dim=2)  # [B, P, S, 4, 2]

    # Render
    output = render_batch_fast(
        canvas_size, canvas_size,
        control_points,
        all_widths,
        all_alphas,
        num_samples=samples,
        use_fill=True,
        background=1.0,
    )  # [B, 1, H, W] in [0, 1] range (white bg, black strokes)

    # Invert to match pydiffvg convention
    output = 1.0 - output

    return output


def generate_test_inputs(device='cpu', seed=42):
    """Generate deterministic test inputs."""
    th.manual_seed(seed)
    np.random.seed(seed)

    # Test parameters matching MNIST VAE
    imsize = 28
    num_segments = 3
    num_paths = 1
    num_points_per_path = num_segments * 3 + 1  # 10 points

    # Generate random points in valid range
    # Tanh output scaled: points = tanh(x) * (imsize//2 - 2) + imsize//2
    # So points are in [2, 26] for imsize=28
    raw_points = th.randn(1, num_paths, num_points_per_path, 2)
    all_points = th.tanh(raw_points) * (imsize // 2 - 2) + imsize // 2
    all_points = all_points.to(device)
    all_points.requires_grad_(True)

    # Widths in [1, 3]
    raw_widths = th.rand(1, num_paths)
    all_widths = 1.0 + 2.0 * raw_widths
    all_widths = all_widths.to(device)
    all_widths.requires_grad_(True)

    # Alphas in [0, 1]
    all_alphas = th.rand(1, num_paths).to(device)
    all_alphas.requires_grad_(True)

    return {
        'imsize': imsize,
        'num_segments': num_segments,
        'num_paths': num_paths,
        'all_points': all_points,
        'all_widths': all_widths,
        'all_alphas': all_alphas,
    }


def run_forward_test(inputs, samples=2):
    """Run forward pass and capture outputs."""
    log("="*60)
    log("FORWARD PASS TEST")
    log("="*60)

    imsize = inputs['imsize']
    num_segments = inputs['num_segments']
    all_points = inputs['all_points']
    all_widths = inputs['all_widths']
    all_alphas = inputs['all_alphas']

    log(f"Input shapes:")
    log(f"  all_points: {all_points.shape}")
    log(f"  all_widths: {all_widths.shape}")
    log(f"  all_alphas: {all_alphas.shape}")
    log(f"  points range: [{all_points.min():.3f}, {all_points.max():.3f}]")
    log(f"  widths: {all_widths.detach().cpu().numpy()}")
    log(f"  alphas: {all_alphas.detach().cpu().numpy()}")

    # Render
    if BACKEND == "pydiffvg":
        output = render_pydiffvg(imsize, all_points, all_widths, all_alphas, num_segments, samples)
    else:
        output = render_triton(imsize, all_points, all_widths, all_alphas, num_segments, samples)

    log(f"\nRaw output shape: {output.shape}")
    log(f"Raw output range: [{output.min():.4f}, {output.max():.4f}]")
    log(f"Raw output mean: {output.mean():.4f}")

    # Map to [-1, 1] (matching VAE training)
    output_normalized = output * 2.0 - 1.0

    log(f"Normalized output range: [{output_normalized.min():.4f}, {output_normalized.max():.4f}]")

    # Compute statistics
    stroke_pixels = (output_normalized > 0).float().sum().item()
    bg_pixels = (output_normalized < 0).float().sum().item()
    total_pixels = imsize * imsize

    log(f"\nStatistics:")
    log(f"  Stroke pixels (normalized > 0): {stroke_pixels} ({100*stroke_pixels/total_pixels:.1f}%)")
    log(f"  Background pixels (normalized < 0): {bg_pixels} ({100*bg_pixels/total_pixels:.1f}%)")

    # Save outputs
    np.save(f"{OUTPUT_DIR}/forward_raw.npy", output.detach().cpu().numpy())
    np.save(f"{OUTPUT_DIR}/forward_normalized.npy", output_normalized.detach().cpu().numpy())

    # Save image
    output_for_img = (output_normalized + 1) / 2  # Back to [0, 1] for display
    save_image(output_for_img, f"{OUTPUT_DIR}/forward.png")

    # Visualize in console
    log("\n--- Rendered output (## = stroke, .. = edge, space = bg) ---")
    img = output_normalized[0, 0].detach().cpu().numpy()
    for y in range(0, 28, 2):
        row = f"y={y:2d}: "
        for x in range(28):
            v = img[y, x]
            if v > 0.5:
                row += "##"
            elif v > -0.5:
                row += ".."
            else:
                row += "  "
        log(row)

    return output, output_normalized


def run_gradient_test(inputs, samples=2):
    """Run backward pass and capture gradients."""
    log("\n" + "="*60)
    log("GRADIENT TEST")
    log("="*60)

    imsize = inputs['imsize']
    num_segments = inputs['num_segments']
    all_points = inputs['all_points'].clone().detach().requires_grad_(True)
    all_widths = inputs['all_widths'].clone().detach().requires_grad_(True)
    all_alphas = inputs['all_alphas'].clone().detach().requires_grad_(True)

    # Forward pass
    if BACKEND == "pydiffvg":
        output = render_pydiffvg(imsize, all_points, all_widths, all_alphas, num_segments, samples)
    else:
        output = render_triton(imsize, all_points, all_widths, all_alphas, num_segments, samples)

    # Normalize
    output_normalized = output * 2.0 - 1.0

    # Create a fake target (random MNIST-like image)
    th.manual_seed(123)
    target = th.randn_like(output_normalized) * 0.5

    # MSE loss
    loss = th.nn.functional.mse_loss(output_normalized, target)

    log(f"Loss: {loss.item():.6f}")

    # Backward
    loss.backward()

    # Capture gradients
    points_grad = all_points.grad
    widths_grad = all_widths.grad
    alphas_grad = all_alphas.grad

    log(f"\nGradient shapes:")
    log(f"  points_grad: {points_grad.shape if points_grad is not None else 'None'}")
    log(f"  widths_grad: {widths_grad.shape if widths_grad is not None else 'None'}")
    log(f"  alphas_grad: {alphas_grad.shape if alphas_grad is not None else 'None'}")

    if points_grad is not None:
        log(f"\nPoints gradient statistics:")
        log(f"  range: [{points_grad.min():.6f}, {points_grad.max():.6f}]")
        log(f"  mean: {points_grad.mean():.6f}")
        log(f"  std: {points_grad.std():.6f}")
        log(f"  norm: {points_grad.norm():.6f}")
        np.save(f"{OUTPUT_DIR}/grad_points.npy", points_grad.detach().cpu().numpy())

    if widths_grad is not None:
        log(f"\nWidths gradient: {widths_grad.detach().cpu().numpy()}")
        np.save(f"{OUTPUT_DIR}/grad_widths.npy", widths_grad.detach().cpu().numpy())

    if alphas_grad is not None:
        log(f"Alphas gradient: {alphas_grad.detach().cpu().numpy()}")
        np.save(f"{OUTPUT_DIR}/grad_alphas.npy", alphas_grad.detach().cpu().numpy())

    return {
        'loss': loss.item(),
        'points_grad': points_grad,
        'widths_grad': widths_grad,
        'alphas_grad': alphas_grad,
    }


def save_inputs(inputs):
    """Save input tensors for exact reproduction."""
    log("\nSaving inputs...")
    np.save(f"{OUTPUT_DIR}/input_points.npy", inputs['all_points'].detach().cpu().numpy())
    np.save(f"{OUTPUT_DIR}/input_widths.npy", inputs['all_widths'].detach().cpu().numpy())
    np.save(f"{OUTPUT_DIR}/input_alphas.npy", inputs['all_alphas'].detach().cpu().numpy())

    # Also save config
    config = {
        'imsize': inputs['imsize'],
        'num_segments': inputs['num_segments'],
        'num_paths': inputs['num_paths'],
        'backend': BACKEND,
    }
    with open(f"{OUTPUT_DIR}/config.json", 'w') as f:
        json.dump(config, f, indent=2)

    log(f"Inputs saved to {OUTPUT_DIR}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--samples', type=int, default=2, help='AA samples per axis')
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()

    device = 'cuda' if args.cuda and th.cuda.is_available() else 'cpu'

    log(f"Device: {device}")
    log(f"Seed: {args.seed}")
    log(f"Samples: {args.samples}")
    log(f"Output dir: {OUTPUT_DIR}")

    # Generate test inputs
    inputs = generate_test_inputs(device=device, seed=args.seed)
    save_inputs(inputs)

    # Run forward test
    output, output_normalized = run_forward_test(inputs, samples=args.samples)

    # Run gradient test
    grad_results = run_gradient_test(inputs, samples=args.samples)

    log("\n" + "="*60)
    log("TEST COMPLETE")
    log(f"Results saved to: {OUTPUT_DIR}/")
    log("="*60)


if __name__ == "__main__":
    main()
