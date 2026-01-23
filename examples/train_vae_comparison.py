#!/usr/bin/env python3
"""
Train and compare three VAE architectures on MNIST using diffvg_torch:
1. Vanilla VAE - standard fixed-point VAE
2. Adaptive Complexity VAE - learned importance weights per point
3. Hierarchical VAE - coarse-to-fine refinement

All models use the diffvg_torch renderer for SVG-based output.

Usage:
    python examples/train_vae_comparison.py --num_epochs 10
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms

# Import diffvg_torch renderer
from diffvg_torch.render_batch import render_batch_fast

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None


# ============================================================================
# Shared Components
# ============================================================================

class SharedEncoder(nn.Module):
    """Shared CNN encoder for all VAE variants."""

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        # Match mnist_vae.py FC encoder style
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        h = self.fc(x)
        return self.fc_mu(h), self.fc_logvar(h)


def render_points_to_image(control_points, widths, alphas, canvas_size=28):
    """
    Render control points using diffvg_torch.

    Args:
        control_points: [B, P, S, 4, 2] bezier control points
        widths: [B, P] stroke widths
        alphas: [B, P] alpha values
        canvas_size: output image size

    Returns:
        [B, 1, H, W] rendered images (white digits on black background)
    """
    rendered = render_batch_fast(
        canvas_size, canvas_size,
        control_points,
        widths,
        alphas,
        num_samples=4,
        use_fill=True,
        background=1.0,  # White background
    )  # [B, 1, H, W] - black strokes on white bg

    # Invert to get white digits on black background (MNIST style)
    rendered = 1.0 - rendered

    return rendered


# ============================================================================
# Model 1: Vanilla VAE (Fixed Points)
# ============================================================================

class VanillaDecoder(nn.Module):
    """Standard decoder with fixed number of bezier paths/segments."""

    def __init__(self, latent_dim: int = 64, num_paths: int = 2, num_segments: int = 3, canvas_size: int = 28):
        super().__init__()
        self.num_paths = num_paths
        self.num_segments = num_segments
        self.canvas_size = canvas_size
        self.points_per_path = num_segments * 3 + 1  # Cubic bezier

        hidden_dim = 512

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
        )

        self.point_head = nn.Sequential(
            nn.Linear(hidden_dim, num_paths * self.points_per_path * 2),
            nn.Tanh(),
        )

        self.width_head = nn.Sequential(
            nn.Linear(hidden_dim, num_paths),
            nn.Sigmoid(),
        )

        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dim, num_paths),
            nn.Sigmoid(),
        )

    def forward(self, z):
        batch_size = z.shape[0]

        h = self.net(z)

        # Points
        points = self.point_head(h)
        points = points.view(batch_size, self.num_paths, self.points_per_path, 2)
        margin = 2
        points = points * (self.canvas_size / 2 - margin) + self.canvas_size / 2

        # Convert to segment format [B, P, S, 4, 2]
        control_points = []
        for seg_idx in range(self.num_segments):
            start_idx = seg_idx * 3
            seg_points = points[:, :, start_idx:start_idx+4, :]
            control_points.append(seg_points)
        control_points = torch.stack(control_points, dim=2)

        # Widths and alphas
        widths = self.width_head(h) * 2.0 + 1.0  # [1, 3]
        alphas = self.alpha_head(h)

        return control_points, widths, alphas


class VanillaVAE(nn.Module):
    """Standard VAE with fixed bezier paths."""

    def __init__(self, latent_dim: int = 64, num_paths: int = 2, num_segments: int = 3, canvas_size: int = 28):
        super().__init__()
        self.encoder = SharedEncoder(latent_dim)
        self.decoder = VanillaDecoder(latent_dim, num_paths, num_segments, canvas_size)
        self.canvas_size = canvas_size

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        control_points, widths, alphas = self.decoder(z)
        rendered = render_points_to_image(control_points, widths, alphas, self.canvas_size)
        return rendered, mu, logvar, {'control_points': control_points, 'widths': widths, 'alphas': alphas}


# ============================================================================
# Model 2: Adaptive Complexity VAE
# ============================================================================

class AdaptiveDecoder(nn.Module):
    """Decoder with learned importance weights per path."""

    def __init__(self, latent_dim: int = 64, max_paths: int = 4, num_segments: int = 3, canvas_size: int = 28):
        super().__init__()
        self.max_paths = max_paths
        self.num_segments = num_segments
        self.canvas_size = canvas_size
        self.points_per_path = num_segments * 3 + 1

        hidden_dim = 512

        self.backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
        )

        self.point_head = nn.Sequential(
            nn.Linear(hidden_dim, max_paths * self.points_per_path * 2),
            nn.Tanh(),
        )

        self.width_head = nn.Sequential(
            nn.Linear(hidden_dim, max_paths),
            nn.Sigmoid(),
        )

        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dim, max_paths),
            nn.Sigmoid(),
        )

        # Importance weights per path
        self.importance_head = nn.Sequential(
            nn.Linear(hidden_dim, max_paths),
            nn.Sigmoid(),
        )

    def forward(self, z):
        batch_size = z.shape[0]

        h = self.backbone(z)

        # Points
        points = self.point_head(h)
        points = points.view(batch_size, self.max_paths, self.points_per_path, 2)
        margin = 2
        points = points * (self.canvas_size / 2 - margin) + self.canvas_size / 2

        # Convert to segment format
        control_points = []
        for seg_idx in range(self.num_segments):
            start_idx = seg_idx * 3
            seg_points = points[:, :, start_idx:start_idx+4, :]
            control_points.append(seg_points)
        control_points = torch.stack(control_points, dim=2)

        # Widths and alphas
        widths = self.width_head(h) * 2.0 + 1.0
        alphas = self.alpha_head(h)

        # Importance weights - modulate alphas
        importance = self.importance_head(h)
        alphas = alphas * importance  # Soft masking

        return control_points, widths, alphas, importance


class AdaptiveVAE(nn.Module):
    """VAE with adaptive complexity (learned path importance)."""

    def __init__(self, latent_dim: int = 64, max_paths: int = 4, num_segments: int = 3, canvas_size: int = 28):
        super().__init__()
        self.encoder = SharedEncoder(latent_dim)
        self.decoder = AdaptiveDecoder(latent_dim, max_paths, num_segments, canvas_size)
        self.canvas_size = canvas_size

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        control_points, widths, alphas, importance = self.decoder(z)
        rendered = render_points_to_image(control_points, widths, alphas, self.canvas_size)
        return rendered, mu, logvar, {
            'control_points': control_points,
            'widths': widths,
            'alphas': alphas,
            'importance': importance
        }


# ============================================================================
# Model 3: Hierarchical VAE
# ============================================================================

class HierarchicalDecoder(nn.Module):
    """Coarse-to-fine decoder that progressively refines paths."""

    def __init__(self, latent_dim: int = 64, num_levels: int = 2, num_paths: int = 2,
                 base_segments: int = 2, canvas_size: int = 28):
        super().__init__()
        self.num_levels = num_levels
        self.num_paths = num_paths
        self.base_segments = base_segments
        self.canvas_size = canvas_size

        hidden_dim = 512

        # Coarse level
        self.coarse_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
        )

        coarse_points = base_segments * 3 + 1
        self.coarse_point_head = nn.Sequential(
            nn.Linear(hidden_dim, num_paths * coarse_points * 2),
            nn.Tanh(),
        )

        # Refinement levels - each adds detail
        self.refinement_nets = nn.ModuleList()
        current_segments = base_segments
        for level in range(1, num_levels):
            # Refinement takes latent + coarse points, outputs offsets
            current_points = current_segments * 3 + 1
            input_dim = latent_dim + num_paths * current_points * 2

            # Each refinement doubles segments
            next_segments = current_segments * 2
            next_points = next_segments * 3 + 1

            self.refinement_nets.append(nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SELU(),
                nn.Linear(hidden_dim, num_paths * next_points * 2),
                nn.Tanh(),
            ))
            current_segments = next_segments

        self.final_segments = current_segments
        self.final_points = current_segments * 3 + 1

        # Shared width/alpha heads
        self.width_head = nn.Sequential(
            nn.Linear(hidden_dim, num_paths),
            nn.Sigmoid(),
        )
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dim, num_paths),
            nn.Sigmoid(),
        )

    def forward(self, z):
        batch_size = z.shape[0]
        device = z.device

        h = self.coarse_net(z)

        # Coarse points
        coarse_points = self.coarse_point_head(h)
        coarse_points = coarse_points.view(batch_size, self.num_paths, self.base_segments * 3 + 1, 2)
        margin = 2
        coarse_points = coarse_points * (self.canvas_size / 2 - margin) + self.canvas_size / 2

        points = coarse_points

        # Refinement levels
        for refine_net in self.refinement_nets:
            # Normalize current points for input
            points_norm = (points - self.canvas_size / 2) / (self.canvas_size / 2 - margin)
            points_flat = points_norm.view(batch_size, -1)

            # Get refined points
            h_input = torch.cat([z, points_flat], dim=-1)
            refined = refine_net(h_input)

            # Reshape to new resolution
            current_num_points = points.shape[2]
            new_num_points = current_num_points * 2 - 1  # Approximately doubles

            # For simplicity, just use the refined output directly
            new_num_points = refined.shape[1] // (self.num_paths * 2)
            refined = refined.view(batch_size, self.num_paths, new_num_points, 2)
            points = refined * (self.canvas_size / 2 - margin) + self.canvas_size / 2

        # Convert to segment format
        actual_segments = (points.shape[2] - 1) // 3
        control_points = []
        for seg_idx in range(actual_segments):
            start_idx = seg_idx * 3
            seg_points = points[:, :, start_idx:start_idx+4, :]
            control_points.append(seg_points)
        control_points = torch.stack(control_points, dim=2)

        # Widths and alphas
        widths = self.width_head(h) * 2.0 + 1.0
        alphas = self.alpha_head(h)

        return control_points, widths, alphas


class HierarchicalVAE(nn.Module):
    """VAE with hierarchical coarse-to-fine decoder."""

    def __init__(self, latent_dim: int = 64, num_levels: int = 2, num_paths: int = 2,
                 base_segments: int = 2, canvas_size: int = 28):
        super().__init__()
        self.num_levels = num_levels
        self.encoder = SharedEncoder(latent_dim)
        self.decoder = HierarchicalDecoder(latent_dim, num_levels, num_paths, base_segments, canvas_size)
        self.canvas_size = canvas_size

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        control_points, widths, alphas = self.decoder(z)
        rendered = render_points_to_image(control_points, widths, alphas, self.canvas_size)
        return rendered, mu, logvar, {'control_points': control_points, 'widths': widths, 'alphas': alphas}


# ============================================================================
# Training
# ============================================================================

def compute_loss(rendered, target, mu, logvar, params, model_type, kl_weight=0.001):
    """Compute VAE loss with model-specific terms."""

    # Reconstruction loss
    recon_loss = F.mse_loss(rendered, target)

    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss + kl_weight * kl_loss

    # Model-specific losses
    if model_type == 'adaptive' and 'importance' in params:
        importance = params['importance']
        # Sparsity loss
        mean_importance = importance.mean()
        sparsity_loss = (mean_importance - 0.6).abs() * 0.1
        total_loss = total_loss + sparsity_loss

    return total_loss, recon_loss.item(), kl_loss.item()


def train_model(model, model_type, dataloader, num_epochs, device, output_dir):
    """Train a single model and return loss history."""

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss_history = []

    for epoch in range(num_epochs):
        model.train()
        epoch_recon = 0
        epoch_kl = 0
        num_batches = 0

        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)

            # Forward pass
            rendered, mu, logvar, params = model(images)

            # Compute loss
            loss, recon, kl = compute_loss(rendered, images, mu, logvar, params, model_type)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_recon += recon
            epoch_kl += kl
            num_batches += 1

            if batch_idx % 200 == 0:
                print(f"  [{model_type}] Epoch {epoch+1}, Batch {batch_idx}, "
                      f"Recon: {recon:.4f}, KL: {kl:.4f}")

        avg_recon = epoch_recon / num_batches
        avg_kl = epoch_kl / num_batches
        loss_history.append({
            'epoch': epoch + 1,
            'recon_loss': avg_recon,
            'kl_loss': avg_kl,
            'total_loss': avg_recon + 0.001 * avg_kl
        })

        print(f"[{model_type}] Epoch {epoch+1}/{num_epochs} - Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}")

    return loss_history


def save_samples(model, model_type, dataloader, device, output_path, num_samples=16):
    """Save sample reconstructions."""
    model.eval()

    # Get a batch
    images, _ = next(iter(dataloader))
    images = images[:num_samples].to(device)

    with torch.no_grad():
        rendered, _, _, _ = model(images)

    # Create comparison image
    nrow = 8
    H, W = 28, 28
    grid = np.ones((H * 2 + 40, nrow * W), dtype=np.uint8) * 255

    # First row: originals, Second row: reconstructions
    for idx in range(min(num_samples, nrow)):
        col = idx

        # Original (row 0)
        img = images[idx, 0].cpu().numpy()
        grid[40:40+H, col*W:(col+1)*W] = (img * 255).astype(np.uint8)

        # Reconstruction (row 1)
        img = rendered[idx, 0].cpu().numpy()
        grid[40+H:40+2*H, col*W:(col+1)*W] = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    if Image:
        img = Image.fromarray(grid)
        try:
            draw = ImageDraw.Draw(img)
            draw.text((5, 5), f"{model_type}", fill=0)
            draw.text((5, 20), "Top: Input, Bottom: Recon", fill=128)
        except:
            pass
        img.save(output_path)
        print(f"Saved samples to {output_path}")


def create_comparison_plot(all_histories, output_path):
    """Create loss curves comparison plot."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {'vanilla': 'blue', 'adaptive': 'green', 'hierarchical': 'red'}

    # Plot reconstruction loss
    ax = axes[0]
    for model_type, history in all_histories.items():
        epochs = [h['epoch'] for h in history]
        recon = [h['recon_loss'] for h in history]
        ax.plot(epochs, recon, label=model_type, color=colors.get(model_type, 'black'), linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reconstruction Loss (MSE)')
    ax.set_title('Reconstruction Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot total loss
    ax = axes[1]
    for model_type, history in all_histories.items():
        epochs = [h['epoch'] for h in history]
        total = [h['total_loss'] for h in history]
        ax.plot(epochs, total, label=model_type, color=colors.get(model_type, 'black'), linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss (Recon + KL)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved loss curves to {output_path}")


def create_combined_samples(sample_paths, output_path):
    """Combine sample images into one comparison."""
    if not Image:
        return

    images = []
    for path in sample_paths:
        if os.path.exists(path):
            images.append(Image.open(path))

    if not images:
        return

    # Stack vertically
    total_height = sum(img.height for img in images)
    max_width = max(img.width for img in images)

    combined = Image.new('L', (max_width, total_height), 255)
    y_offset = 0
    for img in images:
        combined.paste(img, (0, y_offset))
        y_offset += img.height

    combined.save(output_path)
    print(f"Saved combined samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train and compare VAE architectures")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs per model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--latent_dim", type=int, default=32, help="Latent dimension")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./vae_comparison", help="Output directory")
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = dset.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print(f"Dataset loaded: {len(dataset)} images")

    # Model configs
    canvas_size = 28
    latent_dim = args.latent_dim

    models = {
        'vanilla': VanillaVAE(
            latent_dim=latent_dim,
            num_paths=2,
            num_segments=3,
            canvas_size=canvas_size
        ),
        'adaptive': AdaptiveVAE(
            latent_dim=latent_dim,
            max_paths=4,
            num_segments=3,
            canvas_size=canvas_size
        ),
        'hierarchical': HierarchicalVAE(
            latent_dim=latent_dim,
            num_levels=2,
            num_paths=2,
            base_segments=2,
            canvas_size=canvas_size
        ),
    }

    # Print model info
    for name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name}: {n_params:,} parameters")

    # Train each model sequentially
    all_histories = {}
    sample_paths = []

    for model_type, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} VAE")
        print(f"{'='*60}")

        model = model.to(device)

        # Train
        history = train_model(
            model, model_type, dataloader,
            args.num_epochs, device, output_dir
        )
        all_histories[model_type] = history

        # Save samples
        sample_path = os.path.join(output_dir, f"samples_{model_type}.png")
        save_samples(model, model_type, dataloader, device, sample_path)
        sample_paths.append(sample_path)

        # Save model
        torch.save(model.state_dict(), os.path.join(output_dir, f"model_{model_type}.pt"))

        # Free GPU memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save loss histories
    with open(os.path.join(output_dir, "loss_histories.json"), 'w') as f:
        json.dump(all_histories, f, indent=2)

    # Create comparison plots
    create_comparison_plot(all_histories, os.path.join(output_dir, "loss_curves.png"))

    # Create combined samples image
    create_combined_samples(sample_paths, os.path.join(output_dir, "samples_combined.png"))

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")

    # Print final losses
    print("\nFinal Reconstruction Losses:")
    for model_type, history in all_histories.items():
        final_recon = history[-1]['recon_loss']
        print(f"  {model_type}: {final_recon:.4f}")


if __name__ == "__main__":
    main()
