#!/usr/bin/env python3
"""
Benchmark diffvg-torch vs pydiffvg rendering performance.

Measures forward and backward pass times for batched bezier rendering.
"""

import time
import torch
import numpy as np


def benchmark_torch(batch_sizes=[1, 8, 32, 64], num_warmup=5, num_runs=20):
    """Benchmark diffvg-torch batched renderer."""
    from diffvg_torch.render_batch import render_batch_fast

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"\n{'='*60}")
    print("diffvg-torch Benchmark")
    print(f"{'='*60}\n")

    results = []

    for bs in batch_sizes:
        # Create test data: [B, num_paths, num_segments, 4, 2]
        num_paths = 1
        num_segments = 3
        canvas_size = 28

        control_points = torch.randn(bs, num_paths, num_segments, 4, 2, device=device) * 5 + 14
        control_points.requires_grad_(True)
        widths = torch.ones(bs, num_paths, device=device) * 2.0
        widths.requires_grad_(True)
        alphas = torch.ones(bs, num_paths, device=device)
        alphas.requires_grad_(True)

        # Warmup
        for _ in range(num_warmup):
            out = render_batch_fast(canvas_size, canvas_size, control_points, widths, alphas, num_samples=4)
            loss = out.sum()
            loss.backward()
            control_points.grad = None
            widths.grad = None
            alphas.grad = None

        torch.cuda.synchronize()

        # Forward pass timing
        forward_times = []
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = render_batch_fast(canvas_size, canvas_size, control_points, widths, alphas, num_samples=4)
            torch.cuda.synchronize()
            forward_times.append(time.perf_counter() - start)

        # Backward pass timing
        backward_times = []
        for _ in range(num_runs):
            out = render_batch_fast(canvas_size, canvas_size, control_points, widths, alphas, num_samples=4)
            loss = out.sum()
            torch.cuda.synchronize()
            start = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            backward_times.append(time.perf_counter() - start)
            control_points.grad = None
            widths.grad = None
            alphas.grad = None

        fwd_mean = np.mean(forward_times) * 1000
        fwd_std = np.std(forward_times) * 1000
        bwd_mean = np.mean(backward_times) * 1000
        bwd_std = np.std(backward_times) * 1000

        print(f"Batch size {bs:3d}: Forward {fwd_mean:7.2f}ms ± {fwd_std:5.2f}ms | Backward {bwd_mean:7.2f}ms ± {bwd_std:5.2f}ms")

        results.append({
            'batch_size': bs,
            'forward_ms': fwd_mean,
            'forward_std': fwd_std,
            'backward_ms': bwd_mean,
            'backward_std': bwd_std,
        })

    return results


def benchmark_pydiffvg(batch_sizes=[1, 8, 32, 64], num_warmup=3, num_runs=10):
    """Benchmark pydiffvg (sequential, one scene at a time)."""
    try:
        import pydiffvg
    except ImportError:
        print("pydiffvg not available, skipping comparison")
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print("pydiffvg Benchmark (sequential rendering)")
    print(f"{'='*60}\n")

    results = []

    for bs in batch_sizes:
        canvas_size = 28
        num_segments = 3

        # Create paths for pydiffvg
        def create_scene():
            points = torch.randn(num_segments * 3 + 1, 2, device=device) * 5 + 14
            points.requires_grad_(True)

            path = pydiffvg.Path(
                num_control_points=torch.tensor([2] * num_segments, dtype=torch.int32),
                points=points,
                stroke_width=torch.tensor(2.0),
                is_closed=False
            )

            group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([0]),
                stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0])
            )

            return [path], [group], points

        # Warmup
        for _ in range(num_warmup):
            for _ in range(bs):
                shapes, groups, pts = create_scene()
                scene = pydiffvg.RenderFunction.serialize_scene(canvas_size, canvas_size, shapes, groups)
                img = pydiffvg.RenderFunction.apply(canvas_size, canvas_size, 4, 4, 0, None, *scene)
                loss = img.sum()
                loss.backward()

        torch.cuda.synchronize()

        # Forward timing (render bs scenes sequentially)
        forward_times = []
        for _ in range(num_runs):
            scenes = [create_scene() for _ in range(bs)]
            torch.cuda.synchronize()
            start = time.perf_counter()
            for shapes, groups, _ in scenes:
                scene = pydiffvg.RenderFunction.serialize_scene(canvas_size, canvas_size, shapes, groups)
                img = pydiffvg.RenderFunction.apply(canvas_size, canvas_size, 4, 4, 0, None, *scene)
            torch.cuda.synchronize()
            forward_times.append(time.perf_counter() - start)

        # Backward timing
        backward_times = []
        for _ in range(num_runs):
            scenes = [create_scene() for _ in range(bs)]
            imgs = []
            for shapes, groups, _ in scenes:
                scene = pydiffvg.RenderFunction.serialize_scene(canvas_size, canvas_size, shapes, groups)
                img = pydiffvg.RenderFunction.apply(canvas_size, canvas_size, 4, 4, 0, None, *scene)
                imgs.append(img)
            loss = sum(img.sum() for img in imgs)
            torch.cuda.synchronize()
            start = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            backward_times.append(time.perf_counter() - start)

        fwd_mean = np.mean(forward_times) * 1000
        fwd_std = np.std(forward_times) * 1000
        bwd_mean = np.mean(backward_times) * 1000
        bwd_std = np.std(backward_times) * 1000

        print(f"Batch size {bs:3d}: Forward {fwd_mean:7.2f}ms ± {fwd_std:5.2f}ms | Backward {bwd_mean:7.2f}ms ± {bwd_std:5.2f}ms")

        results.append({
            'batch_size': bs,
            'forward_ms': fwd_mean,
            'forward_std': fwd_std,
            'backward_ms': bwd_mean,
            'backward_std': bwd_std,
        })

    return results


def print_comparison(torch_results, pydiffvg_results):
    """Print speedup comparison."""
    if pydiffvg_results is None:
        return

    print(f"\n{'='*60}")
    print("Speedup (diffvg-torch vs pydiffvg)")
    print(f"{'='*60}\n")

    for t, p in zip(torch_results, pydiffvg_results):
        bs = t['batch_size']
        fwd_speedup = p['forward_ms'] / t['forward_ms']
        bwd_speedup = p['backward_ms'] / t['backward_ms']
        print(f"Batch size {bs:3d}: Forward {fwd_speedup:5.1f}x faster | Backward {bwd_speedup:5.1f}x faster")


if __name__ == '__main__':
    batch_sizes = [1, 8, 32, 64]

    torch_results = benchmark_torch(batch_sizes)
    pydiffvg_results = benchmark_pydiffvg(batch_sizes)
    print_comparison(torch_results, pydiffvg_results)
