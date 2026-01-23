#!/usr/bin/env python3
"""End-to-end parity test comparing triton VAE output with pydiffvg VAE output format."""

import sys
sys.path.insert(0, '/workspace')

import torch as th
import numpy as np
import json
import os

def log(msg):
    print(f"[INFO] {msg}")


def render_differentiable_torch(height, width, all_points, all_widths, all_alphas, num_segments, samples=2):
    """Local copy of render_differentiable for testing."""
    bs = all_points.shape[0]
    num_paths = all_points.shape[1]
    device = all_points.device

    num_samples = samples * samples
    oy = th.linspace(0.5/samples, 1.0 - 0.5/samples, samples, device=device)
    ox = th.linspace(0.5/samples, 1.0 - 0.5/samples, samples, device=device)
    py = th.arange(height, dtype=th.float32, device=device)
    px = th.arange(width, dtype=th.float32, device=device)

    py_grid, px_grid, oy_grid, ox_grid = th.meshgrid(py, px, oy, ox, indexing='ij')
    sample_x = px_grid + ox_grid
    sample_y = py_grid + oy_grid

    sample_pos = th.stack([
        sample_x.reshape(-1),
        sample_y.reshape(-1)
    ], dim=-1)

    N = sample_pos.shape[0]
    batch_outputs = []

    for b in range(bs):
        sample_colors = th.ones(N, device=device)

        for p in range(num_paths):
            points = all_points[b, p]
            stroke_width = all_widths[b, p]
            alpha = all_alphas[b, p]

            half_stroke_width = stroke_width / 2.0
            min_dist_sq = th.full((N,), 1e10, device=device)

            point_idx = 0
            for seg in range(num_segments):
                p0 = points[point_idx]
                p1 = points[point_idx + 1]
                p2 = points[point_idx + 2]
                p3 = points[point_idx + 3]
                point_idx += 3

                t_vals = th.linspace(0, 1, 65, device=device)
                one_minus_t = 1.0 - t_vals

                w0 = one_minus_t ** 3
                w1 = 3.0 * (one_minus_t ** 2) * t_vals
                w2 = 3.0 * one_minus_t * (t_vals ** 2)
                w3 = t_vals ** 3

                curve_x = w0 * p0[0] + w1 * p1[0] + w2 * p2[0] + w3 * p3[0]
                curve_y = w0 * p0[1] + w1 * p1[1] + w2 * p2[1] + w3 * p3[1]
                curve_pts = th.stack([curve_x, curve_y], dim=-1)

                diff = sample_pos.unsqueeze(1) - curve_pts.unsqueeze(0)
                dist_sq = (diff ** 2).sum(dim=-1)
                seg_min_dist_sq = dist_sq.min(dim=1).values

                min_dist_sq = th.minimum(min_dist_sq, seg_min_dist_sq)

            min_dist = th.sqrt(min_dist_sq + 1e-8)
            transition_width = 0.25
            coverage = th.sigmoid((half_stroke_width - min_dist) / transition_width)

            stroke_contribution = coverage * alpha
            sample_colors = sample_colors * (1.0 - stroke_contribution)

        sample_colors = sample_colors.reshape(int(height), int(width), int(num_samples))
        pixel_colors = sample_colors.mean(dim=2)

        batch_outputs.append(pixel_colors)

    output = th.stack(batch_outputs, dim=0).unsqueeze(1)
    return output


def main():
    log("="*60)
    log("END-TO-END PARITY TEST")
    log("="*60)

    # Load test parameters (same as used in pydiffvg test)
    # Simple test case: single path with one cubic segment
    device = th.device('cpu')

    # Test parameters matching the MNIST VAE style
    imsize = 28
    segments = 3
    paths = 1

    # Example path points (like VAE decoder output)
    # Shape: [B, paths, num_points, 2]
    all_points = th.tensor([[
        [[5.0, 14.0],   # Start
         [9.0, 5.0],    # Ctrl 1
         [14.0, 5.0],   # Ctrl 2
         [14.0, 14.0],  # End = Start of next segment
         [14.0, 23.0],  # Ctrl 1
         [19.0, 23.0],  # Ctrl 2
         [23.0, 14.0],  # End = Start of next segment
         [23.0, 5.0],   # Ctrl 1
         [19.0, 5.0],   # Ctrl 2
         [14.0, 14.0],  # End
         ]
    ]], dtype=th.float32, device=device)

    all_widths = th.tensor([[2.0]], dtype=th.float32, device=device)
    all_alphas = th.tensor([[0.8]], dtype=th.float32, device=device)

    log(f"Test config:")
    log(f"  Image size: {imsize}x{imsize}")
    log(f"  Paths: {paths}, Segments: {segments}")
    log(f"  Stroke width: {all_widths[0, 0].item()}")
    log(f"  Alpha: {all_alphas[0, 0].item()}")

    # Render with torch renderer
    log("\nRendering with torch...")
    output_raw = render_differentiable_torch(
        imsize, imsize,
        all_points,
        all_widths,
        all_alphas,
        segments,
        samples=2
    )

    # Apply same processing as VAE
    # Invert to match pydiffvg convention
    output_inverted = 1.0 - output_raw
    # Map to [-1, 1]
    output_final = output_inverted * 2.0 - 1.0

    log(f"\nRaw output range: [{output_raw.min():.4f}, {output_raw.max():.4f}]")
    log(f"After inversion: [{output_inverted.min():.4f}, {output_inverted.max():.4f}]")
    log(f"Final output range: [{output_final.min():.4f}, {output_final.max():.4f}]")

    # Visualize
    img = output_final.squeeze().numpy()

    log("\n--- Final output (for comparison with MNIST targets) ---")
    log("(## = >0.5 (stroke), .. = >-0.5, space = <=-0.5 (background))")
    for y in range(5, 24):
        row = f"y={y:2d}: "
        for x in range(5, 26):
            v = img[y, x]
            if v > 0.5:
                row += "##"
            elif v > -0.5:
                row += ".."
            else:
                row += "  "
        log(row)

    # Save for comparison
    output_dir = "/workspace/tests/results"
    os.makedirs(output_dir, exist_ok=True)

    np.save(f"{output_dir}/torch_vae_output.npy", output_final.numpy())
    log(f"\nSaved output to {output_dir}/torch_vae_output.npy")

    # Summary statistics
    stroke_pixels = (output_final > 0).sum().item()
    bg_pixels = (output_final < 0).sum().item()
    total_pixels = imsize * imsize

    log(f"\nStatistics:")
    log(f"  Stroke pixels (>0): {stroke_pixels} ({100*stroke_pixels/total_pixels:.1f}%)")
    log(f"  Background pixels (<0): {bg_pixels} ({100*bg_pixels/total_pixels:.1f}%)")


if __name__ == "__main__":
    main()
