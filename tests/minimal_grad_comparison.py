#!/usr/bin/env python3
"""
Minimal gradient comparison between triton and pydiffvg renderers.

This script creates identical inputs and compares the gradients
from loss.backward() for both backends.
"""

import os
import sys
import json
import numpy as np
import torch as th

# Detect backend
try:
    import pydiffvg
    BACKEND = "pydiffvg"
    sys.path.insert(0, '/workspace/apps/generative_models')
except ImportError:
    BACKEND = "triton"
    sys.path.insert(0, '/workspace')

print(f"[INFO] Backend: {BACKEND}")

OUTPUT_DIR = f"/workspace/tests/results/{BACKEND}_minimal"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    # Fixed seeds
    th.manual_seed(42)
    np.random.seed(42)

    device = th.device('cpu')

    # Create simple test case: single batch, single path, 3 segments
    B, P, S = 1, 1, 3
    H, W = 28, 28

    # Fixed control points (a curved path in center of canvas)
    control_points = th.tensor([[[[
        [10.0, 14.0], [12.0, 10.0], [16.0, 10.0], [18.0, 14.0]  # seg 1
    ], [
        [18.0, 14.0], [20.0, 18.0], [16.0, 22.0], [14.0, 18.0]  # seg 2
    ], [
        [14.0, 18.0], [12.0, 14.0], [10.0, 18.0], [10.0, 14.0]  # seg 3
    ]]]], dtype=th.float32, device=device, requires_grad=True)

    stroke_width = 2.0
    alpha = 0.8
    num_samples = 2

    stats = {'backend': BACKEND}

    if BACKEND == "triton":
        from diffvg_triton.render_batch import render_batch_fast

        stroke_widths = th.tensor([[stroke_width]], device=device, requires_grad=True)
        alphas = th.tensor([[alpha]], device=device, requires_grad=True)

        # Render
        output = render_batch_fast(
            H, W, control_points, stroke_widths, alphas,
            num_samples=num_samples, use_fill=True, background=1.0
        )

        # Invert and map to [-1, 1] like VAE does
        output = 1.0 - output
        output = output * 2.0 - 1.0

    else:  # pydiffvg
        # Need to convert control_points to pydiffvg format
        # pydiffvg expects flat points and renders one sample at a time

        stroke_widths = th.tensor([[stroke_width]], device=device, requires_grad=True)
        alphas = th.tensor([[alpha]], device=device, requires_grad=True)

        # Convert control points to flat format: [num_points_per_path, 2]
        # For 3 segments with 4 control points each (shared endpoints): 3*3+1 = 10 points
        # Actually for cubic bezier: each segment has 4 pts, sharing endpoints: 3+3+3+1 = 10
        cp = control_points[0, 0]  # [S, 4, 2]
        flat_points = []
        for seg_idx in range(S):
            if seg_idx == 0:
                flat_points.append(cp[seg_idx, 0])  # start point
            flat_points.append(cp[seg_idx, 1])  # ctrl1
            flat_points.append(cp[seg_idx, 2])  # ctrl2
            flat_points.append(cp[seg_idx, 3])  # end point

        flat_points = th.stack(flat_points).clone().detach().requires_grad_(True)  # [10, 2]

        # Create path
        num_control_points = th.tensor([2, 2, 2])  # 3 cubic segments
        path = pydiffvg.Path(
            num_control_points=num_control_points,
            points=flat_points,
            stroke_width=th.tensor(stroke_width),
            is_closed=False
        )

        # Shape group
        shape_group = pydiffvg.ShapeGroup(
            shape_ids=th.tensor([0]),
            fill_color=None,
            stroke_color=th.tensor([1.0, 1.0, 1.0, alpha])
        )

        # Render
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            W, H, [path], [shape_group]
        )
        render = pydiffvg.RenderFunction.apply
        img = render(W, H, num_samples, num_samples, 0, None, *scene_args)

        # img is [H, W, 4] RGBA
        # Convert to grayscale and [B, 1, H, W]
        output = img[:, :, 3:4].permute(2, 0, 1).unsqueeze(0)  # alpha channel

        # Map to [-1, 1]
        output = output * 2.0 - 1.0

        # Store points grad hook
        captured_grads = {}
        def save_grad(name):
            def hook(grad):
                captured_grads[name] = grad.clone()
            return hook
        flat_points.register_hook(save_grad('points'))

    # Save rendered output
    stats['output_mean'] = float(output.mean())
    stats['output_min'] = float(output.min())
    stats['output_max'] = float(output.max())
    np.save(f"{OUTPUT_DIR}/output.npy", output.detach().cpu().numpy())

    print(f"[{BACKEND}] Output: mean={output.mean():.6f}, range=[{output.min():.6f}, {output.max():.6f}]")

    # Create simple target (on same device as output)
    th.manual_seed(123)
    target = th.randn(B, 1, H, W, device=output.device) * 0.3

    # MSE loss
    loss = th.nn.functional.mse_loss(output, target)
    stats['loss'] = float(loss)
    print(f"[{BACKEND}] Loss: {loss.item():.6f}")

    # Backward
    loss.backward()

    if BACKEND == "triton":
        grad = control_points.grad
        stats['control_points_grad_norm'] = float(grad.norm())
        stats['control_points_grad_mean'] = float(grad.mean())
        stats['control_points_grad_max'] = float(grad.abs().max())
        np.save(f"{OUTPUT_DIR}/control_points_grad.npy", grad.detach().cpu().numpy())
        print(f"[{BACKEND}] control_points grad: norm={grad.norm():.6f}, mean={grad.mean():.6f}, max={grad.abs().max():.6f}")
    else:
        grad = captured_grads.get('points', flat_points.grad)
        if grad is not None:
            stats['points_grad_norm'] = float(grad.norm())
            stats['points_grad_mean'] = float(grad.mean())
            stats['points_grad_max'] = float(grad.abs().max())
            np.save(f"{OUTPUT_DIR}/points_grad.npy", grad.detach().cpu().numpy())
            print(f"[{BACKEND}] points grad: norm={grad.norm():.6f}, mean={grad.mean():.6f}, max={grad.abs().max():.6f}")
        else:
            print(f"[{BACKEND}] No gradient captured for points")

    # Save stats
    with open(f"{OUTPUT_DIR}/stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n[{BACKEND}] Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
