"""
Batched differentiable rendering for training.

GPU-optimized, fully-vectorized rendering that:
1. Processes entire batches in parallel
2. Uses PyTorch operations for automatic gradient computation
3. Eliminates Python loops for GPU efficiency
"""

import torch
from typing import Tuple, Optional


def _sample_cubic_bezier_batch(
    control_points: torch.Tensor,  # [B, P, S, 4, 2]
    num_curve_samples: int = 17,
) -> torch.Tensor:
    """
    Sample points along cubic bezier curves.

    Args:
        control_points: [B, P, S, 4, 2] control points for cubic beziers
        num_curve_samples: number of samples per segment

    Returns:
        [B, P, S, num_curve_samples, 2] sampled curve points
    """
    device = control_points.device
    dtype = control_points.dtype

    t = torch.linspace(0, 1, num_curve_samples, device=device, dtype=dtype)

    one_minus_t = 1.0 - t
    w0 = one_minus_t ** 3
    w1 = 3.0 * (one_minus_t ** 2) * t
    w2 = 3.0 * one_minus_t * (t ** 2)
    w3 = t ** 3

    weights = torch.stack([w0, w1, w2, w3], dim=-1)
    weights = weights.view(1, 1, 1, num_curve_samples, 4)

    curve_points = torch.einsum('...tc,bpscd->bpstd', weights.squeeze(0).squeeze(0).squeeze(0), control_points)

    return curve_points


def _eval_cubic_bezier(t: torch.Tensor, control_points: torch.Tensor) -> torch.Tensor:
    """Evaluate cubic Bezier at parameter t."""
    t = t.unsqueeze(-1)
    one_minus_t = 1.0 - t

    w0 = one_minus_t ** 3
    w1 = 3.0 * (one_minus_t ** 2) * t
    w2 = 3.0 * one_minus_t * (t ** 2)
    w3 = t ** 3

    p0 = control_points[..., 0, :]
    p1 = control_points[..., 1, :]
    p2 = control_points[..., 2, :]
    p3 = control_points[..., 3, :]

    return w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3


def _eval_cubic_bezier_deriv(t: torch.Tensor, control_points: torch.Tensor) -> torch.Tensor:
    """Evaluate derivative of cubic Bezier at parameter t."""
    t = t.unsqueeze(-1)
    one_minus_t = 1.0 - t

    p0 = control_points[..., 0, :]
    p1 = control_points[..., 1, :]
    p2 = control_points[..., 2, :]
    p3 = control_points[..., 3, :]

    # dB/dt = 3*(1-t)^2*(p1-p0) + 6*(1-t)*t*(p2-p1) + 3*t^2*(p3-p2)
    return (3.0 * one_minus_t ** 2 * (p1 - p0) +
            6.0 * one_minus_t * t * (p2 - p1) +
            3.0 * t ** 2 * (p3 - p2))


class _ClosestPointCubicBezier(torch.autograd.Function):
    """
    Custom autograd function for closest point on cubic Bezier.
    Uses implicit function theorem for gradients.
    """

    @staticmethod
    def forward(ctx, pt, control_points, num_init_samples=8, num_newton_iters=4):
        device = pt.device
        dtype = pt.dtype

        t_samples = torch.linspace(0, 1, num_init_samples + 1, device=device, dtype=dtype)
        shape = control_points.shape[:-2]
        T = len(t_samples)

        t_exp = t_samples.view(*([1] * len(shape)), T).expand(*shape, T)
        cp_exp = control_points.unsqueeze(-3).expand(*shape, T, 4, 2)
        curve_pts = _eval_cubic_bezier(t_exp, cp_exp)

        pt_exp = pt.unsqueeze(-2)
        diff = curve_pts - pt_exp
        dist_sq = (diff ** 2).sum(dim=-1)

        best_idx = dist_sq.argmin(dim=-1)
        t = t_samples[best_idx].clone()

        p0 = control_points[..., 0, :]
        p1 = control_points[..., 1, :]
        p2 = control_points[..., 2, :]
        p3 = control_points[..., 3, :]

        # Newton refinement
        for _ in range(num_newton_iters):
            B = _eval_cubic_bezier(t, control_points)
            dB = _eval_cubic_bezier_deriv(t, control_points)

            residual = B - pt
            f_prime = 2.0 * (residual * dB).sum(dim=-1)

            t_unsq = t.unsqueeze(-1)
            one_minus_t = 1.0 - t_unsq
            d2B = 6.0 * one_minus_t * (p2 - 2*p1 + p0) + 6.0 * t_unsq * (p3 - 2*p2 + p1)

            f_double_prime = 2.0 * ((dB * dB).sum(dim=-1) + (residual * d2B).sum(dim=-1))

            step = f_prime / (f_double_prime.abs() + 1e-8)
            t = t - step
            t = torch.clamp(t, 0.0, 1.0)

        B_final = _eval_cubic_bezier(t, control_points)
        dist = torch.sqrt(((B_final - pt) ** 2).sum(dim=-1) + 1e-8)

        ctx.save_for_backward(pt, control_points, t, B_final)
        return dist

    @staticmethod
    def backward(ctx, grad_dist):
        pt, control_points, t, B_final = ctx.saved_tensors

        p0 = control_points[..., 0, :]
        p1 = control_points[..., 1, :]
        p2 = control_points[..., 2, :]
        p3 = control_points[..., 3, :]

        dB = _eval_cubic_bezier_deriv(t, control_points)

        t_unsq = t.unsqueeze(-1)
        one_minus_t = 1.0 - t_unsq

        d2B = 6.0 * one_minus_t * (p2 - 2*p1 + p0) + 6.0 * t_unsq * (p3 - 2*p2 + p1)

        residual = B_final - pt
        dist_sq = (residual ** 2).sum(dim=-1)
        dist = torch.sqrt(dist_sq + 1e-8)

        d_dist_d_B = residual / dist.unsqueeze(-1)

        tt = 1.0 - t_unsq
        b0 = tt * tt * tt
        b1 = 3.0 * tt * tt * t_unsq
        b2 = 3.0 * tt * t_unsq * t_unsq
        b3 = t_unsq * t_unsq * t_unsq

        grad_dist_exp = grad_dist.unsqueeze(-1)
        d_p0 = d_dist_d_B * b0 * grad_dist_exp
        d_p1 = d_dist_d_B * b1 * grad_dist_exp
        d_p2 = d_dist_d_B * b2 * grad_dist_exp
        d_p3 = d_dist_d_B * b3 * grad_dist_exp

        d_pt = -d_dist_d_B * grad_dist.unsqueeze(-1)
        d_control_points = torch.stack([d_p0, d_p1, d_p2, d_p3], dim=-2)

        return d_pt, d_control_points, None, None


def _closest_point_cubic_bezier_newton(
    pt: torch.Tensor,
    control_points: torch.Tensor,
    num_init_samples: int = 8,
    num_newton_iters: int = 4,
) -> torch.Tensor:
    """Find closest point on cubic Bezier using sampling + Newton refinement."""
    return _ClosestPointCubicBezier.apply(pt, control_points, num_init_samples, num_newton_iters)


def _compute_min_distance_bezier_batch(
    sample_pos: torch.Tensor,
    control_points: torch.Tensor,
    stroke_widths: torch.Tensor,
) -> torch.Tensor:
    """
    Compute minimum distance from samples to cubic Bezier curves.

    Uses bounding box culling for efficiency.

    Args:
        sample_pos: [N, 2] sample positions
        control_points: [B, P, S, 4, 2] control points for cubic beziers
        stroke_widths: [B, P] stroke widths per path

    Returns:
        [B, P, N] minimum distances
    """
    B, P, S, _, _ = control_points.shape
    N = sample_pos.shape[0]
    device = sample_pos.device

    # Compute bounding box expanded by stroke_width + transition
    expand_dist = stroke_widths.unsqueeze(-1).unsqueeze(-1) + 1.5

    seg_min = control_points.min(dim=-2).values
    seg_max = control_points.max(dim=-2).values

    seg_min_exp = seg_min - expand_dist
    seg_max_exp = seg_max + expand_dist

    path_min = seg_min_exp.min(dim=2).values
    path_max = seg_max_exp.max(dim=2).values

    sample_2d = sample_pos.view(1, 1, N, 2)
    path_min_exp = path_min.unsqueeze(2)
    path_max_exp = path_max.unsqueeze(2)

    in_bbox = ((sample_2d >= path_min_exp) & (sample_2d <= path_max_exp)).all(dim=-1)

    sample_exp = sample_pos.view(1, 1, N, 1, 2).expand(B, P, N, S, 2)
    cp_exp = control_points.unsqueeze(2).expand(B, P, N, S, 4, 2)

    sample_flat = sample_exp.reshape(-1, 2)
    cp_flat = cp_exp.reshape(-1, 4, 2)

    dist_flat = _closest_point_cubic_bezier_newton(sample_flat, cp_flat)

    dist = dist_flat.view(B, P, N, S)
    min_dist = dist.min(dim=-1).values

    # Detach gradients for samples outside bounding box
    min_dist_culled = torch.where(
        in_bbox,
        min_dist,
        min_dist.detach()
    )

    return min_dist_culled


def render_batch_fast(
    canvas_width: int,
    canvas_height: int,
    control_points: torch.Tensor,  # [B, P, S, 4, 2]
    stroke_widths: torch.Tensor,   # [B, P]
    alphas: torch.Tensor,          # [B, P]
    num_samples: int = 2,
    use_fill: bool = True,
    background: float = 1.0,
) -> torch.Tensor:
    """
    Fast batched rendering - fully vectorized PyTorch operations.

    Args:
        canvas_width: Output width
        canvas_height: Output height
        control_points: [B, P, S, 4, 2] cubic bezier control points
        stroke_widths: [B, P] stroke widths per path
        alphas: [B, P] alpha values per path (unused for grayscale)
        num_samples: Anti-aliasing samples per axis
        use_fill: Whether to fill paths
        background: Background color (grayscale)

    Returns:
        [B, 1, H, W] rendered images
    """
    device = control_points.device
    dtype = control_points.dtype
    B, P, S, _, _ = control_points.shape
    H, W = canvas_height, canvas_width

    num_curve_samples = 17

    # Sample all bezier curves
    curve_samples = _sample_cubic_bezier_batch(control_points, num_curve_samples)

    C = S * num_curve_samples
    curve_points = curve_samples.reshape(B, P, C, 2).contiguous()

    # Generate sample positions
    samples_per_axis = num_samples
    total_samples = samples_per_axis * samples_per_axis

    py = torch.arange(H, device=device, dtype=dtype)
    px = torch.arange(W, device=device, dtype=dtype)
    oy = (torch.arange(samples_per_axis, device=device, dtype=dtype) + 0.5) / samples_per_axis
    ox = (torch.arange(samples_per_axis, device=device, dtype=dtype) + 0.5) / samples_per_axis

    py_grid, px_grid, oy_grid, ox_grid = torch.meshgrid(py, px, oy, ox, indexing='ij')
    sample_x = px_grid + ox_grid
    sample_y = py_grid + oy_grid

    N = H * W * total_samples
    sample_x_flat = sample_x.reshape(N)
    sample_y_flat = sample_y.reshape(N)

    sample_pos = torch.stack([sample_x_flat, sample_y_flat], dim=-1)

    # Compute distances
    min_dist = _compute_min_distance_bezier_batch(sample_pos, control_points, stroke_widths)

    # Winding number computation
    if use_fill:
        curve_closed = torch.cat([curve_points, curve_points[:, :, :1, :]], dim=2)

        p0 = curve_closed[:, :, :-1, :]
        p1 = curve_closed[:, :, 1:, :]

        p0_exp = p0.view(B, P, 1, C, 2)
        p1_exp = p1.view(B, P, 1, C, 2)

        dy = p1_exp[..., 1] - p0_exp[..., 1]

        pt_y = sample_pos[:, 1].view(1, 1, N, 1)
        pt_x = sample_pos[:, 0].view(1, 1, N, 1)

        dy_safe = torch.where(torch.abs(dy) > 1e-8, dy, torch.ones_like(dy) * 1e-8)
        t = (pt_y - p0_exp[..., 1]) / dy_safe

        x_int = p0_exp[..., 0] + t * (p1_exp[..., 0] - p0_exp[..., 0])

        softness = 0.1
        t_valid = torch.sigmoid((t + 0.01) / softness) * torch.sigmoid((1.01 - t) / softness)
        x_valid = torch.sigmoid((x_int - pt_x + 0.01) / softness)

        direction = torch.where(dy > 0, torch.ones_like(dy), -torch.ones_like(dy))
        contrib = torch.where(torch.abs(dy) > 1e-8, direction * t_valid * x_valid, torch.zeros_like(t_valid))

        winding = contrib.sum(dim=-1)
        inside = (torch.abs(winding) >= 0.5).float()
        inside = inside.detach()
    else:
        inside = torch.zeros(B, P, N, device=device, dtype=dtype)

    # Stroke coverage using smoothstep
    half_widths = stroke_widths.view(B, P, 1)
    alphas_exp = alphas.view(B, P, 1)

    def smoothstep(x):
        t = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    dist_threshold = half_widths + 1.5
    in_gradient_zone = min_dist < dist_threshold

    min_dist_masked = torch.where(
        in_gradient_zone,
        min_dist,
        min_dist.detach()
    )

    abs_d_plus_w = min_dist_masked + half_widths
    abs_d_minus_w = min_dist_masked - half_widths
    stroke_edge = smoothstep(abs_d_plus_w) - smoothstep(abs_d_minus_w)

    coverage = torch.maximum(inside, stroke_edge)
    stroke_contrib = coverage

    _ = alphas_exp  # Unused for grayscale output

    # Composite over paths
    one_minus_contrib = 1.0 - stroke_contrib
    combined = one_minus_contrib.prod(dim=1)
    sample_colors = background * combined

    # Reshape and average
    sample_colors = sample_colors.view(B, H, W, total_samples)
    pixel_colors = sample_colors.mean(dim=-1)

    return pixel_colors.unsqueeze(1)
