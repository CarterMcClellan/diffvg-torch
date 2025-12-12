#!/usr/bin/env python3
"""Compare parity test results between triton and pydiffvg."""

import os
import json
import numpy as np

TRITON_DIR = "/home/carter/code/diffvg-parity-tests/results/triton_parity"
PYDIFFVG_DIR = "/home/carter/code/diffvg-parity-tests/results/pydiffvg_parity"


def load_results(dir_path):
    """Load all results from a directory."""
    results = {}

    for name in ['forward_raw', 'forward_normalized', 'input_points',
                 'input_widths', 'input_alphas', 'grad_points',
                 'grad_widths', 'grad_alphas']:
        path = os.path.join(dir_path, f"{name}.npy")
        if os.path.exists(path):
            results[name] = np.load(path)

    config_path = os.path.join(dir_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            results['config'] = json.load(f)

    return results


def print_section(title):
    print(f"\n{'='*60}")
    print(title)
    print('='*60)


def main():
    print_section("LOADING RESULTS")

    triton = load_results(TRITON_DIR)
    pydiffvg = load_results(PYDIFFVG_DIR)

    print(f"Triton results: {list(triton.keys())}")
    print(f"pydiffvg results: {list(pydiffvg.keys())}")

    # Verify inputs match
    print_section("INPUT VERIFICATION")

    for name in ['input_points', 'input_widths', 'input_alphas']:
        if name in triton and name in pydiffvg:
            diff = np.abs(triton[name] - pydiffvg[name]).max()
            print(f"{name}: max diff = {diff:.10f}")
            if diff > 1e-6:
                print(f"  WARNING: Inputs don't match!")

    # Compare forward outputs
    print_section("FORWARD PASS COMPARISON")

    triton_raw = triton.get('forward_raw')
    pydiffvg_raw = pydiffvg.get('forward_raw')

    if triton_raw is not None and pydiffvg_raw is not None:
        print(f"Triton raw:   shape={triton_raw.shape}, range=[{triton_raw.min():.4f}, {triton_raw.max():.4f}], mean={triton_raw.mean():.4f}")
        print(f"pydiffvg raw: shape={pydiffvg_raw.shape}, range=[{pydiffvg_raw.min():.4f}, {pydiffvg_raw.max():.4f}], mean={pydiffvg_raw.mean():.4f}")

        diff = np.abs(triton_raw - pydiffvg_raw)
        mse = np.mean((triton_raw - pydiffvg_raw)**2)
        print(f"\nPixel-wise difference:")
        print(f"  Max: {diff.max():.4f}")
        print(f"  Mean: {diff.mean():.4f}")
        print(f"  MSE: {mse:.6f}")

        # Where are the biggest differences?
        diff_2d = diff[0, 0]
        max_idx = np.unravel_index(np.argmax(diff_2d), diff_2d.shape)
        print(f"\n  Max diff location: y={max_idx[0]}, x={max_idx[1]}")
        print(f"  Triton value: {triton_raw[0, 0, max_idx[0], max_idx[1]]:.4f}")
        print(f"  pydiffvg value: {pydiffvg_raw[0, 0, max_idx[0], max_idx[1]]:.4f}")

        # Show difference heatmap
        print("\n  Difference heatmap (# = large diff, . = small, space = none):")
        for y in range(0, 28, 2):
            row = f"  y={y:2d}: "
            for x in range(28):
                d = diff_2d[y, x]
                if d > 0.3:
                    row += "##"
                elif d > 0.1:
                    row += ".."
                else:
                    row += "  "
            print(row)

    # Compare gradients
    print_section("GRADIENT COMPARISON")

    for name in ['grad_points', 'grad_widths', 'grad_alphas']:
        t = triton.get(name)
        p = pydiffvg.get(name)

        if t is not None and p is not None:
            print(f"\n{name}:")
            print(f"  Triton:   range=[{t.min():.6f}, {t.max():.6f}], mean={t.mean():.6f}, norm={np.linalg.norm(t):.6f}")
            print(f"  pydiffvg: range=[{p.min():.6f}, {p.max():.6f}], mean={p.mean():.6f}, norm={np.linalg.norm(p):.6f}")

            diff = np.abs(t - p)
            print(f"  Max diff: {diff.max():.6f}")
            print(f"  Mean diff: {diff.mean():.6f}")

            # Correlation
            t_flat = t.flatten()
            p_flat = p.flatten()
            if len(t_flat) > 1:
                corr = np.corrcoef(t_flat, p_flat)[0, 1]
                print(f"  Correlation: {corr:.4f}")

            # Ratio analysis
            nonzero_mask = (np.abs(p) > 1e-8) & (np.abs(t) > 1e-8)
            if nonzero_mask.any():
                ratios = t[nonzero_mask] / p[nonzero_mask]
                print(f"  Ratio (triton/pydiffvg where both nonzero): mean={ratios.mean():.4f}, std={ratios.std():.4f}")

    print_section("SUMMARY")
    print("""
Key observations:
1. Forward pass has some differences - need to analyze rendering pipeline
2. Gradient differences indicate potential issues with:
   - Alpha gradient flow (pydiffvg has near-zero alpha gradients)
   - Width gradient sign (opposite signs!)
   - Points gradient magnitude (different by ~2x)

Likely causes:
- Different stroke width interpretation (diameter vs radius)
- Different alpha compositing
- Different transition/coverage functions
""")


if __name__ == "__main__":
    main()
