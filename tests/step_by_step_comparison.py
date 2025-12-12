#!/usr/bin/env python3
"""
Step-by-step comparison of forward/backward passes between triton and pydiffvg.

This script traces through each stage with fixed seeds and identical inputs,
saving intermediate values for comparison.
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

print(f"[INFO] Backend: {BACKEND}")

OUTPUT_DIR = f"/workspace/tests/results/{BACKEND}_step_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def log(msg):
    print(f"[{BACKEND}] {msg}")


def save_tensor(name, tensor, stats_dict):
    """Save tensor stats to dict and numpy file."""
    if tensor is None:
        stats_dict[name] = None
        return

    t = tensor.detach().cpu()
    stats_dict[name] = {
        'shape': list(t.shape),
        'min': float(t.min()),
        'max': float(t.max()),
        'mean': float(t.mean()),
        'std': float(t.std()) if t.numel() > 1 else 0.0,
        'norm': float(t.norm()),
        'sum': float(t.sum()),
    }
    np.save(f"{OUTPUT_DIR}/{name}.npy", t.numpy())


def main():
    # Fixed seeds - MUST be identical for both backends
    th.manual_seed(42)
    np.random.seed(42)

    device = th.device('cpu')  # CPU for reproducibility

    stats = {'backend': BACKEND}

    # Import model
    from mnist_vae import VectorMNISTVAE

    # Create model with identical parameters
    log("Creating model...")
    model = VectorMNISTVAE(
        imsize=28,
        paths=1,
        segments=3,
        samples=2,
        zdim=20,
        conditional=True,
        variational=True,
        fc=True,
    )
    model.to(device)
    model.train()

    # Log model parameter count
    param_count = sum(p.numel() for p in model.parameters())
    log(f"Model parameters: {param_count}")
    stats['param_count'] = param_count

    # Create identical input batch
    th.manual_seed(123)  # Different seed for data
    batch_size = 4
    im = th.randn(batch_size, 1, 28, 28, device=device) * 0.3
    im = (im - im.min()) / (im.max() - im.min() + 1e-8)
    im = (im - 0.5) / 0.5
    label = th.tensor([0, 1, 2, 3], device=device)

    save_tensor('input_image', im, stats)
    log(f"Input: shape={im.shape}, range=[{im.min():.6f}, {im.max():.6f}]")

    # ========== ENCODER ==========
    log("\n===== ENCODER =====")

    # Get encoder output
    if model.variational:
        mu, logvar = model.encode(im, label)
        z = model.reparameterize(mu, logvar)
    else:
        mu = model.encode(im, label)
        z = mu
        logvar = th.zeros_like(mu)

    save_tensor('encoder_mu', mu, stats)
    save_tensor('encoder_logvar', logvar, stats)
    save_tensor('encoder_z', z, stats)

    log(f"mu: range=[{mu.min():.6f}, {mu.max():.6f}], mean={mu.mean():.6f}")
    log(f"logvar: range=[{logvar.min():.6f}, {logvar.max():.6f}], mean={logvar.mean():.6f}")
    log(f"z: range=[{z.min():.6f}, {z.max():.6f}], mean={z.mean():.6f}")

    # ========== DECODER (before rendering) ==========
    log("\n===== DECODER (pre-render) =====")

    # We'll capture decoder intermediates from the full forward pass
    # since the internal feature dimensions depend on fc vs conv encoder

    # ========== RENDERING ==========
    log("\n===== RENDERING =====")

    # Full forward pass to get rendering
    th.manual_seed(42)  # Reset for any randomness in forward
    rendering, auxdata = model(im, label)

    # Register hooks to capture intermediate gradients
    captured_grads = {}
    def make_hook(name):
        def hook(grad):
            captured_grads[name] = grad.clone().detach()
            return grad
        return hook

    # Hook the control_points or points tensor if available
    if 'control_points' in auxdata and auxdata['control_points'].requires_grad:
        auxdata['control_points'].register_hook(make_hook('control_points'))
    if 'points' in auxdata and auxdata['points'].requires_grad:
        auxdata['points'].register_hook(make_hook('points'))

    save_tensor('rendering', rendering, stats)
    log(f"rendering: range=[{rendering.min():.6f}, {rendering.max():.6f}], mean={rendering.mean():.6f}")

    # Save decoder outputs from auxdata
    if 'points' in auxdata:
        save_tensor('decoder_points', auxdata['points'], stats)
        log(f"decoder_points: range=[{auxdata['points'].min():.6f}, {auxdata['points'].max():.6f}]")

    # Check for widths/alphas in auxdata (triton has these, pydiffvg may not)
    if 'widths' in auxdata:
        save_tensor('decoder_widths', auxdata['widths'], stats)
        log(f"decoder_widths: range=[{auxdata['widths'].min():.6f}, {auxdata['widths'].max():.6f}]")

    if 'alphas' in auxdata:
        save_tensor('decoder_alphas', auxdata['alphas'], stats)
        log(f"decoder_alphas: range=[{auxdata['alphas'].min():.6f}, {auxdata['alphas'].max():.6f}]")

    # Count stroke pixels (where rendering > 0, i.e., visible strokes)
    # In [-1, 1] range, background is -1, strokes are toward +1
    stroke_pixels = (rendering > 0).float().sum().item()
    stats['stroke_pixels'] = stroke_pixels
    log(f"stroke_pixels (rendering > 0): {stroke_pixels}")

    # Histogram of rendering values
    rendering_flat = rendering.detach().cpu().flatten()
    hist_bins = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for i in range(len(hist_bins) - 1):
        count = ((rendering_flat >= hist_bins[i]) & (rendering_flat < hist_bins[i+1])).sum().item()
        log(f"  pixels in [{hist_bins[i]:.1f}, {hist_bins[i+1]:.1f}): {count}")

    # ========== LOSS ==========
    log("\n===== LOSS =====")

    # Reconstruction loss
    data_loss = th.nn.functional.mse_loss(rendering, im)
    save_tensor('data_loss', data_loss.unsqueeze(0), stats)
    log(f"data_loss (MSE): {data_loss.item():.6f}")

    # Per-sample MSE for debugging
    per_sample_mse = ((rendering - im) ** 2).mean(dim=(1, 2, 3))
    save_tensor('per_sample_mse', per_sample_mse, stats)
    log(f"per_sample_mse: {per_sample_mse.tolist()}")

    # KLD
    kld_per_sample = -0.5 * th.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kld = kld_per_sample.mean()
    save_tensor('kld_per_sample', kld_per_sample, stats)
    save_tensor('kld', kld.unsqueeze(0), stats)
    log(f"kld: {kld.item():.6f}")
    log(f"kld_per_sample: {kld_per_sample.tolist()}")

    # Total loss
    loss = data_loss + 1.0 * kld
    save_tensor('total_loss', loss.unsqueeze(0), stats)
    log(f"total_loss: {loss.item():.6f}")

    # ========== BACKWARD ==========
    log("\n===== BACKWARD =====")

    loss.backward()

    # Report captured intermediate gradients
    log("\nIntermediate gradients (from hooks):")
    for name, grad in captured_grads.items():
        save_tensor(f'grad_{name}', grad, stats)
        log(f"  {name}: shape={list(grad.shape)}, norm={grad.norm():.6f}, max={grad.abs().max():.6f}")

    # Gradient statistics for each predictor
    grad_stats = {}

    log("\nPoint predictor gradients:")
    for i, layer in enumerate(model.point_predictor):
        if hasattr(layer, 'weight') and layer.weight.grad is not None:
            g = layer.weight.grad
            key = f'grad_point_predictor_{i}'
            save_tensor(key, g, stats)
            log(f"  layer[{i}]: norm={g.norm():.6f}, mean={g.mean():.6f}, max={g.abs().max():.6f}")

    log("\nWidth predictor gradients:")
    for i, layer in enumerate(model.width_predictor):
        if hasattr(layer, 'weight') and layer.weight.grad is not None:
            g = layer.weight.grad
            key = f'grad_width_predictor_{i}'
            save_tensor(key, g, stats)
            log(f"  layer[{i}]: norm={g.norm():.6f}, mean={g.mean():.6f}, max={g.abs().max():.6f}")

    log("\nAlpha predictor gradients:")
    for i, layer in enumerate(model.alpha_predictor):
        if hasattr(layer, 'weight') and layer.weight.grad is not None:
            g = layer.weight.grad
            key = f'grad_alpha_predictor_{i}'
            save_tensor(key, g, stats)
            log(f"  layer[{i}]: norm={g.norm():.6f}, mean={g.mean():.6f}, max={g.abs().max():.6f}")

    log("\nEncoder gradients:")
    if hasattr(model, 'mu_predictor') and model.mu_predictor.weight.grad is not None:
        g = model.mu_predictor.weight.grad
        save_tensor('grad_mu_predictor', g, stats)
        log(f"  mu_predictor: norm={g.norm():.6f}")

    if hasattr(model, 'logvar_predictor') and model.logvar_predictor.weight.grad is not None:
        g = model.logvar_predictor.weight.grad
        save_tensor('grad_logvar_predictor', g, stats)
        log(f"  logvar_predictor: norm={g.norm():.6f}")

    # ========== SAVE SUMMARY ==========
    with open(f"{OUTPUT_DIR}/stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    log(f"\n===== COMPLETE =====")
    log(f"Results saved to {OUTPUT_DIR}/")

    return stats


if __name__ == "__main__":
    main()
