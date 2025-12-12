#!/usr/bin/env python3
"""
Single batch VAE training step comparison.

This runs one forward/backward pass through the full VAE to diagnose
training behavior differences.
"""

import os
import sys
import json
import numpy as np
import torch as th

# Detect which backend we're using
try:
    import pydiffvg
    BACKEND = "pydiffvg"
    sys.path.insert(0, '/workspace/apps/generative_models')
except ImportError:
    BACKEND = "triton"
    sys.path.insert(0, '/workspace/examples')

print(f"[INFO] Detected backend: {BACKEND}")

OUTPUT_DIR = f"/workspace/tests/results/{BACKEND}_vae_step"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def log(msg):
    print(f"[{BACKEND}] {msg}")


def main():
    # Fixed seeds
    th.manual_seed(42)
    np.random.seed(42)

    device = th.device('cpu')  # Use CPU for reproducibility

    # Import VAE model
    from mnist_vae import VectorMNISTVAE

    # Create model with identical parameters
    model = VectorMNISTVAE(
        imsize=28,
        paths=1,
        segments=3,
        samples=2,  # 2x2 = 4 samples
        zdim=20,
        conditional=True,
        variational=True,
        fc=True,
    )
    model.to(device)
    model.train()

    log(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create fake MNIST batch
    th.manual_seed(123)  # Different seed for data
    batch_size = 4
    im = th.randn(batch_size, 1, 28, 28, device=device) * 0.3  # Roughly MNIST range
    im = (im - im.min()) / (im.max() - im.min() + 1e-8)  # Normalize to [0, 1]
    im = (im - 0.5) / 0.5  # To [-1, 1]

    label = th.tensor([0, 1, 2, 3], device=device)

    log(f"Input batch: shape={im.shape}, range=[{im.min():.4f}, {im.max():.4f}]")

    # Forward pass
    log("\n--- Forward Pass ---")
    rendering, auxdata = model(im, label)

    log(f"Rendering: shape={rendering.shape}, range=[{rendering.min():.4f}, {rendering.max():.4f}]")
    log(f"Rendering mean: {rendering.mean():.4f}")

    mu = auxdata["mu"]
    logvar = auxdata["logvar"]
    log(f"mu: shape={mu.shape}, range=[{mu.min():.4f}, {mu.max():.4f}]")
    log(f"logvar: shape={logvar.shape}, range=[{logvar.min():.4f}, {logvar.max():.4f}]")

    # Compute loss
    data_loss = th.nn.functional.mse_loss(rendering, im)
    kld = -0.5 * th.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
    kld = kld.mean()
    loss = data_loss + 1.0 * kld

    log(f"\nLoss: {loss.item():.6f}")
    log(f"  Data loss: {data_loss.item():.6f}")
    log(f"  KLD: {kld.item():.6f}")

    # Backward pass
    log("\n--- Backward Pass ---")
    loss.backward()

    # Capture gradients
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            g = param.grad
            grad_stats[name] = {
                'norm': g.norm().item(),
                'mean': g.mean().item(),
                'std': g.std().item(),
                'max': g.abs().max().item(),
            }

    log("\nGradient statistics:")
    for name, stats in sorted(grad_stats.items()):
        log(f"  {name}: norm={stats['norm']:.6f}, max={stats['max']:.6f}")

    # Save specific gradients we care about
    log("\n--- Key Parameter Gradients ---")

    if hasattr(model, 'point_predictor'):
        for i, layer in enumerate(model.point_predictor):
            if hasattr(layer, 'weight'):
                g = layer.weight.grad
                log(f"point_predictor[{i}].weight: norm={g.norm():.6f}, mean={g.mean():.6f}")

    if hasattr(model, 'width_predictor'):
        for i, layer in enumerate(model.width_predictor):
            if hasattr(layer, 'weight'):
                g = layer.weight.grad
                log(f"width_predictor[{i}].weight: norm={g.norm():.6f}, mean={g.mean():.6f}")

    if hasattr(model, 'alpha_predictor'):
        for i, layer in enumerate(model.alpha_predictor):
            if hasattr(layer, 'weight'):
                g = layer.weight.grad
                log(f"alpha_predictor[{i}].weight: norm={g.norm():.6f}, mean={g.mean():.6f}")

    # Save results
    results = {
        'backend': BACKEND,
        'loss': loss.item(),
        'data_loss': data_loss.item(),
        'kld': kld.item(),
        'rendering_mean': rendering.mean().item(),
        'rendering_min': rendering.min().item(),
        'rendering_max': rendering.max().item(),
        'grad_stats': grad_stats,
    }

    with open(f"{OUTPUT_DIR}/results.json", 'w') as f:
        json.dump(results, f, indent=2)

    np.save(f"{OUTPUT_DIR}/rendering.npy", rendering.detach().cpu().numpy())

    log(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
