#!/usr/bin/env python3
"""
Refine SVG to match a target image using gradient descent.

Usage:
    python refine_svg.py input.svg target.png
    python refine_svg.py input.svg target.png --use_lpips_loss
    python refine_svg.py input.svg target.png --num_iter 500
"""

import argparse
import os
import sys

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image

import diffvg_torch

gamma = 1.0


def main(args):
    device = diffvg_torch.get_device()
    print(f'Using device: {device}')

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load LPIPS if requested
    perception_loss = None
    if args.use_lpips_loss:
        try:
            import lpips
            perception_loss = lpips.LPIPS(net='vgg').to(device)
        except ImportError:
            print('LPIPS not available, using L2 loss')

    # Load target image
    target = torch.from_numpy(np.array(Image.open(args.target).convert('RGB'))).float() / 255.0
    target = target.pow(gamma).to(device).unsqueeze(0).permute(0, 3, 1, 2)

    # Load SVG
    canvas_width, canvas_height, shapes, shape_groups = diffvg_torch.svg_to_scene(args.svg)

    # Move to device
    for shape in shapes:
        shape.points = shape.points.to(device)
        shape.num_control_points = shape.num_control_points.to(device)
    for group in shape_groups:
        if group.fill_color is not None:
            group.fill_color = group.fill_color.to(device)

    # Initial render
    img = diffvg_torch.render(canvas_width, canvas_height, shapes, shape_groups)
    diffvg_torch.imwrite(img.cpu(), f'{output_dir}/init.png', gamma=gamma)

    # Setup optimization
    points_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)

    color_vars = {}
    for group in shape_groups:
        if group.fill_color is not None:
            group.fill_color.requires_grad = True
            color_vars[group.fill_color.data_ptr()] = group.fill_color
    color_vars = list(color_vars.values())

    points_optim = torch.optim.Adam(points_vars, lr=1.0)
    color_optim = torch.optim.Adam(color_vars, lr=0.01) if color_vars else None

    # Optimization loop
    for t in range(args.num_iter):
        print(f'iteration: {t}')
        points_optim.zero_grad()
        if color_optim:
            color_optim.zero_grad()

        # Render
        img = diffvg_torch.render(canvas_width, canvas_height, shapes, shape_groups)

        # Composite with white background
        img = img[:, :, 3:4] * img[:, :, :3] + (1 - img[:, :, 3:4])

        diffvg_torch.imwrite(img.cpu(), f'{output_dir}/iter_{t}.png', gamma=gamma)

        # Compute loss
        img = img.unsqueeze(0).permute(0, 3, 1, 2)
        if perception_loss:
            loss = perception_loss(img, target)
        else:
            loss = (img - target).pow(2).mean()

        print(f'render loss: {loss.item()}')
        loss.backward()

        points_optim.step()
        if color_optim:
            color_optim.step()
            for group in shape_groups:
                if group.fill_color is not None:
                    group.fill_color.data.clamp_(0.0, 1.0)

        if t % 10 == 0 or t == args.num_iter - 1:
            diffvg_torch.save_svg(f'{output_dir}/iter_{t}.svg',
                                   canvas_width, canvas_height, shapes, shape_groups)

    # Final render
    img = diffvg_torch.render(canvas_width, canvas_height, shapes, shape_groups)
    diffvg_torch.imwrite(img.cpu(), f'{output_dir}/final.png', gamma=gamma)

    # Create video
    from subprocess import call
    call(["ffmpeg", "-y", "-framerate", "24", "-i", f"{output_dir}/iter_%d.png",
          "-vb", "20M", f"{output_dir}/out.mp4"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("svg", help="source SVG path")
    parser.add_argument("target", help="target image path")
    parser.add_argument("--use_lpips_loss", dest='use_lpips_loss', action='store_true')
    parser.add_argument("--num_iter", type=int, default=250)
    parser.add_argument("--output_dir", type=str, default='results/refine_svg', help="output directory")
    args = parser.parse_args()
    main(args)
