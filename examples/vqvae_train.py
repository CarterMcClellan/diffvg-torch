#!/usr/bin/env python3
"""
Train a VQ-VAE for MNIST using diffvg_torch.

VQ-VAE (Vector Quantized Variational Autoencoder) uses discrete latent codes
instead of continuous latent vectors. This allows for better reconstruction
quality and enables generation via autoregressive models.

Usage:
    python examples/vqvae_train.py --num_epochs 10 --bs 32
"""

import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms

from diffvg_torch.render_batch import render_batch_fast


class VectorQuantizerEMA(nn.Module):
    """
    Vector Quantization layer with Exponential Moving Average (EMA) codebook updates.

    EMA updates are more stable than gradient-based updates and help prevent
    codebook collapse. Based on the original VQ-VAE paper.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self._initialized = False

        # Codebook - will be initialized from first batch
        embedding = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer('embedding', embedding)
        self.register_buffer('ema_cluster_size', torch.ones(num_embeddings))
        self.register_buffer('ema_embedding_sum', embedding.clone())

    def _initialize_from_data(self, flat_z: torch.Tensor):
        """Initialize codebook from data using k-means++ style initialization."""
        n_samples = flat_z.shape[0]
        device = flat_z.device

        if n_samples < self.num_embeddings:
            # Not enough samples, use random init with some noise
            indices = torch.randint(0, n_samples, (self.num_embeddings,), device=device)
            self.embedding.data.copy_(flat_z[indices] + torch.randn_like(self.embedding) * 0.1)
        else:
            # K-means++ initialization
            centroids = [flat_z[torch.randint(0, n_samples, (1,)).item()]]

            for _ in range(1, self.num_embeddings):
                # Compute distances to nearest centroid
                centroid_stack = torch.stack(centroids, dim=0)  # [k, D]
                dists = torch.cdist(flat_z, centroid_stack)  # [N, k]
                min_dists = dists.min(dim=1).values  # [N]

                # Sample proportional to squared distance
                probs = min_dists ** 2
                probs = probs / probs.sum()
                idx = torch.multinomial(probs, 1).item()
                centroids.append(flat_z[idx])

            self.embedding.data.copy_(torch.stack(centroids, dim=0))

        self.ema_embedding_sum.data.copy_(self.embedding.data.clone())
        self._initialized = True
        print(f"[INFO] Codebook initialized from {n_samples} samples")

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: [B, D] continuous latent vectors

        Returns:
            z_q: [B, D] quantized vectors (with straight-through gradient)
            loss: commitment loss only (codebook updated via EMA)
            indices: [B] codebook indices
            perplexity: codebook usage metric
        """
        # Flatten if needed and compute distances
        flat_z = z.reshape(-1, self.embedding_dim)

        # Initialize codebook from first batch
        if self.training and not self._initialized:
            self._initialize_from_data(flat_z.detach())

        # Compute distances to codebook entries
        distances = (
            torch.sum(flat_z ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding ** 2, dim=1)
            - 2 * torch.matmul(flat_z, self.embedding.t())
        )  # [N, K]

        # Find nearest codebook entries
        indices = torch.argmin(distances, dim=1)  # [N]

        # One-hot encodings for EMA update
        encodings = F.one_hot(indices, self.num_embeddings).float()  # [N, K]

        # Quantize
        z_q = F.embedding(indices, self.embedding)  # [N, D]
        z_q = z_q.view_as(z)

        # EMA codebook update (only during training)
        if self.training:
            # Update cluster sizes
            cluster_size = encodings.sum(0)  # [K]
            self.ema_cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)

            # Update embedding sums
            embedding_sum = torch.matmul(encodings.t(), flat_z.detach())  # [K, D]
            self.ema_embedding_sum.data.mul_(self.decay).add_(embedding_sum, alpha=1 - self.decay)

            # Laplace smoothing to avoid division by zero
            n = self.ema_cluster_size.sum()
            cluster_size_smoothed = (
                (self.ema_cluster_size + self.epsilon) /
                (n + self.num_embeddings * self.epsilon) * n
            )

            # Update embeddings
            self.embedding.data.copy_(self.ema_embedding_sum / cluster_size_smoothed.unsqueeze(1))

            # Reset dead codes (codes that haven't been used)
            usage = self.ema_cluster_size / self.ema_cluster_size.sum()
            dead_codes = usage < (1.0 / self.num_embeddings / 10)  # Very rarely used
            if dead_codes.any():
                n_dead = dead_codes.sum().item()
                # Replace dead codes with random encoder outputs + noise
                random_indices = torch.randint(0, flat_z.shape[0], (n_dead,), device=flat_z.device)
                self.embedding.data[dead_codes] = flat_z.detach()[random_indices] + torch.randn(n_dead, self.embedding_dim, device=flat_z.device) * 0.1
                self.ema_cluster_size.data[dead_codes] = 1.0
                self.ema_embedding_sum.data[dead_codes] = self.embedding.data[dead_codes].clone()

        # Commitment loss (encoder should commit to codebook entries)
        commitment_loss = F.mse_loss(z_q.detach(), z)

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        # Compute perplexity (measure of codebook usage)
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q, self.commitment_cost * commitment_loss, indices.view(z.shape[0]), perplexity


class Encoder(nn.Module):
    """Simple FC encoder for MNIST images - matches working VAE architecture."""

    def __init__(self, latent_dim: int):
        super().__init__()
        # Simple FC encoder like the working VAE
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class BezierDecoder(nn.Module):
    """
    Decoder that outputs Bezier curve parameters for vector graphics rendering.
    Matches the working VAE architecture.
    """

    def __init__(
        self,
        latent_dim: int,
        num_paths: int = 2,
        num_segments: int = 3,
        canvas_size: int = 28,
    ):
        super().__init__()
        self.num_paths = num_paths
        self.num_segments = num_segments
        self.canvas_size = canvas_size

        # Points per path: start + 3 control points per segment
        self.points_per_path = num_segments * 3 + 1

        # Match working VAE: simple MLP with SELU
        hidden_dim = 1024
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SELU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(inplace=True),
        )

        # Output heads - same as working VAE
        self.point_head = nn.Sequential(
            nn.Linear(hidden_dim, num_paths * self.points_per_path * 2),
            nn.Tanh(),  # Output in [-1, 1], scaled to canvas
        )

        self.width_head = nn.Sequential(
            nn.Linear(hidden_dim, num_paths),
            nn.Sigmoid(),  # Output in [0, 1], scaled to width range
        )

        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dim, num_paths),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: [B, D] latent vectors

        Returns:
            rendered: [B, 1, H, W] rendered images
            params: dict of bezier parameters
        """
        batch_size = z.shape[0]
        device = z.device

        h = self.decoder(z)

        # Predict bezier parameters
        points = self.point_head(h)  # [B, P * N * 2]
        points = points.view(batch_size, self.num_paths, self.points_per_path, 2)
        # Scale from [-1, 1] to canvas coordinates with margin
        margin = 2
        points = points * (self.canvas_size / 2 - margin) + self.canvas_size / 2

        widths = self.width_head(h)  # [B, P]
        widths = widths * 2.0 + 1.0  # Scale to [1, 3]

        alphas = self.alpha_head(h)  # [B, P]

        # Convert flat points to segment format [B, P, S, 4, 2]
        control_points = []
        for seg_idx in range(self.num_segments):
            start_idx = seg_idx * 3
            seg_points = points[:, :, start_idx:start_idx+4, :]
            control_points.append(seg_points)
        control_points = torch.stack(control_points, dim=2)

        # Render using diffvg_torch
        rendered = render_batch_fast(
            self.canvas_size, self.canvas_size,
            control_points,
            widths,
            alphas,
            num_samples=4,
            use_fill=True,
            background=1.0,
        )  # [B, 1, H, W] white bg, black strokes

        # Invert: make strokes white on black background (match MNIST)
        rendered = 1.0 - rendered

        params = {
            'points': points,
            'control_points': control_points,
            'widths': widths,
            'alphas': alphas,
        }

        return rendered, params


class VectorQuantizerGumbel(nn.Module):
    """
    Vector Quantization with Gumbel-Softmax for differentiable discrete sampling.

    This avoids the codebook collapse problem by using soft assignment during training.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        temperature: float = 1.0,
        straight_through: bool = True,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.straight_through = straight_through

        # Learnable codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

        # Project encoder output to logits over codebook
        self.to_logits = nn.Linear(embedding_dim, num_embeddings)

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: [B, D] continuous latent vectors

        Returns:
            z_q: [B, D] quantized vectors
            loss: regularization loss
            indices: [B] hard codebook indices
            perplexity: codebook usage metric
        """
        # Get logits for each codebook entry
        logits = self.to_logits(z)  # [B, K]

        if self.training:
            # Gumbel-softmax for differentiable sampling
            soft_onehot = F.gumbel_softmax(logits, tau=self.temperature, hard=self.straight_through)
        else:
            # Hard assignment at eval
            indices = logits.argmax(dim=-1)
            soft_onehot = F.one_hot(indices, self.num_embeddings).float()

        # Quantize: weighted sum of embeddings
        z_q = torch.matmul(soft_onehot, self.embedding.weight)  # [B, D]

        # Hard indices for logging
        indices = logits.argmax(dim=-1)

        # Entropy regularization to encourage codebook usage
        probs = F.softmax(logits, dim=-1)
        avg_probs = probs.mean(0)

        # Maximize entropy (minimize negative entropy)
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        max_entropy = torch.log(torch.tensor(self.num_embeddings, dtype=torch.float32, device=z.device))

        # Loss encourages uniform usage
        entropy_loss = (max_entropy - entropy) * 0.1

        # Perplexity
        perplexity = torch.exp(entropy)

        return z_q, entropy_loss, indices, perplexity


class VQVAE(nn.Module):
    """
    Vector Quantized VAE for MNIST with vector graphics decoder.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        num_embeddings: int = 512,
        num_paths: int = 2,
        num_segments: int = 3,
        commitment_cost: float = 0.25,
        use_gumbel: bool = True,
        use_vq: bool = True,  # Can disable VQ for debugging
    ):
        super().__init__()
        self.use_vq = use_vq
        self.encoder = Encoder(latent_dim)
        if use_vq:
            if use_gumbel:
                self.vq = VectorQuantizerGumbel(num_embeddings, latent_dim, temperature=1.0)
            else:
                self.vq = VectorQuantizerEMA(num_embeddings, latent_dim, commitment_cost)
        self.decoder = BezierDecoder(latent_dim, num_paths, num_segments)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 1, 28, 28] MNIST images in [0, 1]

        Returns:
            recon: [B, 1, 28, 28] reconstructed images
            vq_loss: VQ loss
            params: bezier parameters (includes perplexity)
        """
        z_e = self.encoder(x)

        if self.use_vq:
            z_q, vq_loss, indices, perplexity = self.vq(z_e)
        else:
            # No VQ - just pass through (deterministic autoencoder)
            z_q = z_e
            vq_loss = torch.tensor(0.0, device=x.device)
            indices = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            perplexity = torch.tensor(0.0, device=x.device)

        recon, params = self.decoder(z_q)

        params['z_e'] = z_e
        params['z_q'] = z_q
        params['indices'] = indices
        params['perplexity'] = perplexity

        return recon, vq_loss, params

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to discrete codes."""
        z_e = self.encoder(x)
        if self.use_vq:
            _, _, indices, _ = self.vq(z_e)
            return indices
        return torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode from discrete codes."""
        z_q = F.embedding(indices, self.vq.embedding)
        recon, _ = self.decoder(z_q)
        return recon


def save_image_grid(images: torch.Tensor, path: str, nrow: int = 8):
    """Save a grid of images."""
    try:
        from PIL import Image
    except ImportError:
        print(f"[WARN] PIL not available, skipping save to {path}")
        return

    B, C, H, W = images.shape
    ncol = (B + nrow - 1) // nrow

    grid = torch.zeros(ncol * H, nrow * W)
    for idx in range(B):
        i, j = idx // nrow, idx % nrow
        grid[i*H:(i+1)*H, j*W:(j+1)*W] = images[idx, 0]

    grid = (grid.clamp(0, 1) * 255).cpu().numpy().astype(np.uint8)
    Image.fromarray(grid, mode='L').save(path)
    print(f"[INFO] Saved {path}")


def train(args):
    """Main training loop."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"vqvae_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")

    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = dset.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=0)

    # Model
    model = VQVAE(
        latent_dim=args.latent_dim,
        num_embeddings=args.num_embeddings,
        num_paths=args.paths,
        num_segments=args.segments,
        commitment_cost=args.commitment_cost,
        use_vq=not args.no_vq,
        use_gumbel=not args.use_ema,
    ).to(device)

    print(f"[INFO] Model: latent_dim={args.latent_dim}, num_embeddings={args.num_embeddings}")
    print(f"[INFO] Decoder: paths={args.paths}, segments={args.segments}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training
    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        total_recon_loss = 0
        total_vq_loss = 0
        total_perplexity = 0
        num_batches = 0

        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)

            # Forward
            recon, vq_loss, params = model(images)
            perplexity = params['perplexity']

            # Reconstruction loss (MSE)
            recon_loss = F.mse_loss(recon, images)

            # Total loss
            loss = recon_loss + vq_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_perplexity += perplexity.item()
            num_batches += 1
            global_step += 1

            if batch_idx % 100 == 0:
                # Count unique codes in this batch
                unique_codes = len(torch.unique(params['indices']))
                print(f"[INFO] Epoch {epoch+1}/{args.num_epochs}, Batch {batch_idx}, "
                      f"Recon: {recon_loss.item():.4f}, VQ: {vq_loss.item():.4f}, "
                      f"Perplexity: {perplexity.item():.1f}, Unique codes: {unique_codes}")

            if args.max_batches and batch_idx >= args.max_batches:
                break

        avg_recon = total_recon_loss / num_batches
        avg_vq = total_vq_loss / num_batches
        avg_perplexity = total_perplexity / num_batches
        print(f"[INFO] Epoch {epoch+1} complete. Recon: {avg_recon:.4f}, VQ: {avg_vq:.4f}, Perplexity: {avg_perplexity:.1f}")

        # Save sample reconstructions
        model.eval()
        with torch.no_grad():
            sample_images = images[:16]
            sample_recon, _, _ = model(sample_images)

            # Save comparison: original | reconstruction
            comparison = torch.cat([sample_images, sample_recon], dim=0)
            save_image_grid(comparison, os.path.join(output_dir, f"recon_epoch_{epoch+1}.png"), nrow=16)

        # Save checkpoint
        if (epoch + 1) % 10 == 0 or epoch == args.num_epochs - 1:
            ckpt_path = os.path.join(output_dir, f"model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            print(f"[INFO] Saved checkpoint: {ckpt_path}")

    print("[INFO] Training complete!")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="VQ-VAE for MNIST with vector graphics")

    # Training args
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--bs", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max_batches", type=int, default=None, help="Max batches per epoch (for testing)")

    # Model args
    parser.add_argument("--latent_dim", type=int, default=64, help="Latent dimension")
    parser.add_argument("--num_embeddings", type=int, default=512, help="Codebook size")
    parser.add_argument("--commitment_cost", type=float, default=0.25, help="Commitment loss weight")

    # Decoder args
    parser.add_argument("--paths", type=int, default=2, help="Number of bezier paths")
    parser.add_argument("--segments", type=int, default=3, help="Segments per path")

    # Ablation flags
    parser.add_argument("--no_vq", action="store_true", help="Disable VQ (pure autoencoder)")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA VQ instead of Gumbel")

    # Paths
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
