#!/usr/bin/env python3
"""
Train a VAE MNIST generator using diffvg_triton.

This is a standalone example demonstrating differentiable vector graphics
rendering with the Triton backend.

Usage:
    python mnist_vae.py train
    python mnist_vae.py sample
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch as th
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms

# Import from diffvg_triton package
from diffvg_triton.render_batch import render_batch_fast


# Simple logging
def log(msg, *args):
    print(f"[INFO] {msg}" % args if args else f"[INFO] {msg}")


# Output directory - write to shared results folder for comparison
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_BASE = "/workspace/tests/results/triton"
OUTPUT_DIR = None  # Will be set in train() with timestamp


class Flatten(th.nn.Module):
    """Flatten layer for the encoder."""
    def forward(self, x):
        return x.view(x.shape[0], -1)


class Path:
    """
    Minimal Path class compatible with the Triton backend.

    This replaces pydiffvg.Path for standalone usage.
    """
    def __init__(self, num_control_points, points, stroke_width=1.0, is_closed=False):
        self.num_control_points = num_control_points
        self.points = points
        self.stroke_width = stroke_width if isinstance(stroke_width, th.Tensor) else th.tensor([stroke_width])
        self.is_closed = is_closed
        self.thickness = None


class ShapeGroup:
    """
    Minimal ShapeGroup class compatible with the Triton backend.

    This replaces pydiffvg.ShapeGroup for standalone usage.
    """
    def __init__(self, shape_ids, fill_color=None, stroke_color=None, use_even_odd_rule=True):
        self.shape_ids = shape_ids if isinstance(shape_ids, th.Tensor) else th.tensor(shape_ids, dtype=th.int32)
        self.fill_color = fill_color
        self.stroke_color = stroke_color
        self.use_even_odd_rule = use_even_odd_rule
        self.shape_to_canvas = None


def imwrite(image, path, gamma=1.0):
    """Save image to file."""
    try:
        from PIL import Image
    except ImportError:
        log("PIL not available, skipping image save")
        return

    if isinstance(image, th.Tensor):
        image = image.detach().cpu().numpy()

    # Apply gamma
    if gamma != 1.0:
        image = np.power(np.clip(image, 0, 1), 1.0 / gamma)

    # Convert to uint8
    image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    # Handle different shapes
    if image.ndim == 2:
        mode = 'L'
    elif image.ndim == 3 and image.shape[2] == 4:
        mode = 'RGBA'
    elif image.ndim == 3 and image.shape[2] == 3:
        mode = 'RGB'
    else:
        mode = 'L'
        image = image.squeeze()

    Image.fromarray(image, mode=mode).save(path)
    log(f"Saved image to {path}")


def _onehot(label):
    """Convert label to one-hot encoding."""
    bs = label.shape[0]
    label_onehot = label.new_zeros(bs, 10)
    label_onehot.scatter_(1, label.unsqueeze(1), 1)
    return label_onehot.float()


class VectorMNISTVAE(th.nn.Module):
    """
    VAE that generates vector graphics for MNIST digits.

    Encoder: CNN or FC that maps image -> latent space
    Decoder: MLP that maps latent -> Bezier control points
    Renderer: Triton backend renders paths to image
    """

    def __init__(self, imsize=28, paths=4, segments=5, samples=2, zdim=128,
                 conditional=False, variational=True, stroke_width=None, fc=False):
        super(VectorMNISTVAE, self).__init__()

        self.samples = samples
        self.imsize = imsize
        self.paths = paths
        self.segments = segments
        self.zdim = zdim
        self.conditional = conditional
        self.variational = variational
        self.fc = fc

        if stroke_width is None:
            # Match pydiffvg default
            self.stroke_width = (1.0, 3.0)
        else:
            self.stroke_width = stroke_width

        ncond = 10 if self.conditional else 0
        mult = 1

        if not self.fc:
            # Encoder (convolutional) - matches pydiffvg architecture
            # padding=0 gives: 28->12->4->1 with kernel=4, stride=2
            self.encoder = th.nn.Sequential(
                th.nn.Conv2d(1 + ncond, mult*64, 4, padding=0, stride=2),
                th.nn.LeakyReLU(0.2, inplace=True),
                th.nn.Conv2d(mult*64, mult*128, 4, padding=0, stride=2),
                th.nn.LeakyReLU(0.2, inplace=True),
                th.nn.Conv2d(mult*128, mult*256, 4, padding=0, stride=2),
                th.nn.LeakyReLU(0.2, inplace=True),
                Flatten(),
            )
            # Encoder output size: 28->12->4->1 with padding=0
            encoder_out_size = 256 * 1 * 1
        else:
            # FC encoder - matches pydiffvg fc=True mode
            # Note: input is already flattened in encode() for FC mode
            fc_input_size = 28*28 + ncond if self.conditional else 28*28
            self.encoder = th.nn.Sequential(
                th.nn.Linear(fc_input_size, mult*256),
                th.nn.LeakyReLU(0.2, inplace=True),
                th.nn.Linear(mult*256, mult*256),
                th.nn.LeakyReLU(0.2, inplace=True),
            )
            encoder_out_size = mult * 256

        self.mu_predictor = th.nn.Linear(encoder_out_size, zdim)
        if self.variational:
            self.logvar_predictor = th.nn.Linear(encoder_out_size, zdim)

        # Decoder - match pydiffvg's nc=1024
        nc = 1024
        self.decoder = th.nn.Sequential(
            th.nn.Linear(zdim + ncond, nc),
            th.nn.SELU(inplace=True),
            th.nn.Linear(nc, nc),
            th.nn.SELU(inplace=True),
        )

        # Output heads
        # 4 points bezier with n_segments -> 3*n_segments + 1 points
        num_points = self.segments * 3 + 1
        self.point_predictor = th.nn.Sequential(
            th.nn.Linear(nc, 2 * self.paths * num_points),
            th.nn.Tanh()
        )

        self.width_predictor = th.nn.Sequential(
            th.nn.Linear(nc, self.paths),
            th.nn.Sigmoid()
        )

        self.alpha_predictor = th.nn.Sequential(
            th.nn.Linear(nc, self.paths),
            th.nn.Sigmoid()
        )

    def encode(self, im, label):
        bs, _, h, w = im.shape

        if self.conditional:
            label_onehot = _onehot(label)
            if not self.fc:
                label_onehot = label_onehot.view(bs, 10, 1, 1).repeat(1, 1, h, w)
                x = th.cat([im, label_onehot], 1)
            else:
                # FC mode: flatten image first, then concat label
                x = th.cat([im.view(bs, -1), label_onehot], 1)
        else:
            if self.fc:
                x = im.view(bs, -1)
            else:
                x = im

        out = self.encoder(x)
        mu = self.mu_predictor(out)

        if self.variational:
            logvar = self.logvar_predictor(out)
            return mu, logvar
        else:
            return mu

    def reparameterize(self, mu, logvar):
        std = th.exp(0.5 * logvar)
        eps = th.randn_like(logvar)
        return mu + std * eps

    def decode(self, z, label=None):
        bs = z.shape[0]
        device = z.device

        if self.conditional and label is not None:
            label_onehot = _onehot(label)
            z = th.cat([z, label_onehot], 1)

        feats = self.decoder(z)

        # Predict path parameters - keep everything on GPU for gradients
        all_points = self.point_predictor(feats)

        # Reshape to [B, P, S, 4, 2] for batched renderer
        # The point predictor outputs 3*S + 1 points per path (cubic bezier format)
        # We need to convert to per-segment format with 4 control points each
        num_points_per_path = self.segments * 3 + 1
        all_points = all_points.view(bs, self.paths, num_points_per_path, 2)
        all_points = all_points * (self.imsize // 2 - 2) + self.imsize // 2

        # Convert flat point format to per-segment [B, P, S, 4, 2]
        # Each segment needs: start, ctrl1, ctrl2, end
        # Points are: start, ctrl1, ctrl2, end1, ctrl3, ctrl4, end2, ...
        # For segment i: points[3*i], points[3*i+1], points[3*i+2], points[3*i+3]
        control_points = []
        for seg_idx in range(self.segments):
            start_idx = seg_idx * 3
            seg_points = all_points[:, :, start_idx:start_idx+4, :]  # [B, P, 4, 2]
            control_points.append(seg_points)
        control_points = th.stack(control_points, dim=2)  # [B, P, S, 4, 2]

        all_widths = self.width_predictor(feats)
        min_width, max_width = self.stroke_width
        all_widths = (max_width - min_width) * all_widths + min_width

        all_alphas = self.alpha_predictor(feats)

        # Use optimized batched renderer
        output = render_batch_fast(
            self.imsize, self.imsize,
            control_points,
            all_widths,
            all_alphas,
            num_samples=self.samples,
            use_fill=True,
            background=1.0,
        )  # [B, 1, H, W] in [0, 1] range (white bg, black strokes)

        # Invert to match pydiffvg convention: strokes = 1, background = 0
        # pydiffvg uses white strokes on transparent (0) background
        # Our renderer uses black strokes on white (1) background
        output = 1.0 - output

        auxdata = {
            "points": all_points,
            "control_points": control_points,
            "widths": all_widths,
            "alphas": all_alphas,
        }

        # Map to [-1, 1]: now strokes = +1, background = -1 (matches MNIST target)
        output = output * 2.0 - 1.0

        return output, auxdata

    def forward(self, im, label):
        if self.variational:
            mu, logvar = self.encode(im, label)
            z = self.reparameterize(mu, logvar)
        else:
            mu = self.encode(im, label)
            z = mu
            logvar = th.zeros_like(mu)

        if self.conditional:
            output, aux = self.decode(z, label=label)
        else:
            output, aux = self.decode(z)

        aux["logvar"] = logvar
        aux["mu"] = mu

        return output, aux


class MNISTDataset(th.utils.data.Dataset):
    """MNIST dataset wrapper."""

    def __init__(self, data_dir):
        super().__init__()
        self.mnist = dset.MNIST(
            root=data_dir,
            download=True,
            transform=transforms.ToTensor()
        )

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        im, label = self.mnist[idx]
        # Normalize to [-1, 1] - per-image normalization (matches pydiffvg)
        im = im - im.min()
        im = im / (im.max() + 1e-8)
        im = (im - 0.5) / 0.5
        return im, label


def train(args):
    """Train the VAE."""
    global OUTPUT_DIR

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = os.path.join(RESULTS_BASE, f"run_{timestamp}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log(f"Output directory: {OUTPUT_DIR}")

    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = th.device('cuda' if args.cuda and th.cuda.is_available() else 'cpu')
    log(f"Using device: {device}")

    # Dataset
    dataset = MNISTDataset(args.data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=0  # Use 0 for debugging
    )

    # Model
    model = VectorMNISTVAE(
        imsize=28,
        paths=args.paths,
        segments=args.segments,
        samples=args.samples,
        zdim=args.zdim,
        conditional=args.conditional,
        variational=True,
        fc=args.fc
    )
    model.to(device)

    log(f"Model config: paths={args.paths}, segments={args.segments}, zdim={args.zdim}")
    log(f"Encoder: {'FC' if args.fc else 'Conv'}, conditional={args.conditional}")

    # Optimizer - match pydiffvg betas
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.5), eps=1e-12)

    # Constant KLD weight (matches pydiffvg)
    kld_weight = args.kld_weight

    # Training loop
    log(f"Starting training for {args.num_epochs} epochs")
    log(f"Config: paths={args.paths}, segments={args.segments}, zdim={args.zdim}")
    log(f"KLD weight: {kld_weight}")

    global_step = 0

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        total_data_loss = 0
        total_kld = 0
        num_batches = 0

        for batch_idx, (im, label) in enumerate(dataloader):
            im = im.to(device)
            label = label.to(device)

            # Forward
            rendering, auxdata = model(im, label)

            # Reconstruction loss - simple MSE (matches pydiffvg)
            data_loss = th.nn.functional.mse_loss(rendering, im)

            # KLD loss (matches pydiffvg)
            mu = auxdata["mu"]
            logvar = auxdata["logvar"]

            # KLD: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
            kld = -0.5 * th.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            kld = kld.mean()

            # Total loss with annealed KLD weight
            loss = data_loss + kld_weight * kld

            # Backward
            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            total_loss += loss.item()
            total_data_loss += data_loss.item()
            total_kld += kld.item()
            num_batches += 1
            global_step += 1

            if batch_idx % 100 == 0:
                log(f"Epoch {epoch+1}/{args.num_epochs}, Batch {batch_idx}, "
                    f"Loss: {loss.item():.4f}, Data: {data_loss.item():.4f}, "
                    f"KLD: {kld.item():.4f}, β={kld_weight:.4f}")

            # Limit batches for testing
            if args.max_batches and batch_idx >= args.max_batches:
                break

        avg_loss = total_loss / num_batches
        avg_data = total_data_loss / num_batches
        avg_kld = total_kld / num_batches
        log(f"Epoch {epoch+1} complete. Loss: {avg_loss:.4f}, Data: {avg_data:.4f}, KLD: {avg_kld:.4f}")

        # Save training stats
        stats_path = os.path.join(OUTPUT_DIR, "training_stats.txt")
        with open(stats_path, "a") as f:
            f.write(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Data={avg_data:.4f}, KLD={avg_kld:.4f}\n")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(OUTPUT_DIR, f"model_epoch_{epoch+1}.pt")
            th.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            log(f"Saved checkpoint to {ckpt_path}")

        # Generate sample images every epoch for comparison
        generate_comparison(model, dataset, epoch + 1, device)

    log("Training complete!")


def generate_comparison(model, dataset, epoch, device):
    """Generate comparison images: reconstructions vs references."""
    model.eval()

    # Get a fixed batch for comparison
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    ref_batch, labels = next(iter(dataloader))
    ref_batch = ref_batch.to(device)
    labels = labels.to(device)

    with th.no_grad():
        # Run through autoencoder
        rendering, _ = model(ref_batch, labels)
        rendering = (rendering + 1) / 2  # Map to [0, 1]
        ref_display = (ref_batch + 1) / 2  # Map to [0, 1]

    # Create side-by-side comparison: 4x4 grid of (ref | rendered)
    n = 4
    comparison = th.zeros(n * 28, n * 28 * 2)

    for i in range(n):
        for j in range(n):
            idx = i * n + j
            # Reference on left
            comparison[i*28:(i+1)*28, j*56:j*56+28] = ref_display[idx, 0].cpu()
            # Rendered on right
            comparison[i*28:(i+1)*28, j*56+28:(j+1)*56] = rendering[idx, 0].cpu()

    comparison = th.clamp(comparison, 0, 1).numpy()
    path = os.path.join(OUTPUT_DIR, f"comparison_epoch_{epoch}.png")
    imwrite(comparison, path, gamma=2.2)

    # Also save just the renderings in a grid
    grid = rendering.view(n, n, 28, 28).permute(0, 2, 1, 3)
    grid = grid.contiguous().view(n * 28, n * 28)
    grid = th.clamp(grid, 0, 1).cpu().numpy()
    path = os.path.join(OUTPUT_DIR, f"rendered_epoch_{epoch}.png")
    imwrite(grid, path, gamma=2.2)

    model.train()


def generate_samples(model, epoch, device):
    """Generate sample images from the model."""
    model.eval()

    with th.no_grad():
        # Random samples
        z = th.randn(16, model.zdim).to(device)

        if model.conditional:
            label = th.arange(10).repeat(2)[:16].to(device)
        else:
            label = None

        images, _ = model.decode(z, label=label)
        images = (images + 1) / 2  # Map to [0, 1]

    # Create grid
    n = 4
    grid = images.view(n, n, 28, 28).permute(0, 2, 1, 3)
    grid = grid.contiguous().view(n * 28, n * 28)
    grid = th.clamp(grid, 0, 1).cpu().numpy()

    path = os.path.join(OUTPUT_DIR, f"samples_epoch_{epoch}.png")
    imwrite(grid, path, gamma=2.2)


def main():
    parser = argparse.ArgumentParser(description="MNIST VAE with Triton backend")
    parser.add_argument("command", choices=["train", "sample"], help="Command to run")
    parser.add_argument("--cuda", action="store_true", default=th.cuda.is_available(),
                       help="Use CUDA if available")
    parser.add_argument("--data_dir", default="./data", help="Data directory")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")  # Match pydiffvg

    # Training args - defaults match pydiffvg
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")  # Match pydiffvg
    parser.add_argument("--bs", type=int, default=8, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--kld_weight", type=float, default=1.0, help="KLD loss weight")
    parser.add_argument("--max_batches", type=int, default=None, help="Max batches per epoch (for testing)")

    # Model args - defaults match pydiffvg
    parser.add_argument("--paths", type=int, default=1, help="Number of paths")  # Match pydiffvg
    parser.add_argument("--segments", type=int, default=3, help="Segments per path")
    parser.add_argument("--samples", type=int, default=4, help="AA samples")  # Match pydiffvg
    parser.add_argument("--zdim", type=int, default=20, help="Latent dimension")  # Match pydiffvg
    parser.add_argument("--no-conditional", dest="conditional", action="store_false",
                       default=True, help="Disable conditional VAE")  # Match pydiffvg default
    parser.add_argument("--no-fc", dest="fc", action="store_false",
                       default=True, help="Use conv encoder instead of FC")  # Match pydiffvg default

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "sample":
        log("Sample generation not yet implemented")


if __name__ == "__main__":
    main()
