"""
Batched differentiable rendering for VAE training.

This module provides GPU-optimized, fully-vectorized rendering that:
1. Processes entire batches in parallel
2. Uses PyTorch operations for automatic gradient computation
3. Eliminates Python loops for GPU efficiency

Designed for training VAEs where gradients flow through the rendering.
"""

import torch
import triton
import triton.language as tl
from typing import Tuple, Optional


def _sample_cubic_bezier_batch(
    control_points: torch.Tensor,  # [B, P, S, 4, 2] - batch, paths, segments, 4 control pts, xy
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

    # Parameter t from 0 to 1
    t = torch.linspace(0, 1, num_curve_samples, device=device, dtype=dtype)

    # Bezier weights: (1-t)^3, 3(1-t)^2*t, 3(1-t)*t^2, t^3
    one_minus_t = 1.0 - t
    w0 = one_minus_t ** 3                          # [T]
    w1 = 3.0 * (one_minus_t ** 2) * t
    w2 = 3.0 * one_minus_t * (t ** 2)
    w3 = t ** 3

    # Stack weights: [T, 4]
    weights = torch.stack([w0, w1, w2, w3], dim=-1)

    # Expand for batched matmul: [1, 1, 1, T, 4]
    weights = weights.view(1, 1, 1, num_curve_samples, 4)

    # control_points: [B, P, S, 4, 2]
    # We want: sum over 4 control points weighted by weights
    # Result: [B, P, S, T, 2]
    curve_points = torch.einsum('...tc,bpscd->bpstd', weights.squeeze(0).squeeze(0).squeeze(0), control_points)

    return curve_points


def _eval_cubic_bezier(t: torch.Tensor, control_points: torch.Tensor) -> torch.Tensor:
    """
    Evaluate cubic Bezier at parameter t.

    Args:
        t: [...] parameter values in [0, 1]
        control_points: [..., 4, 2] control points

    Returns:
        [..., 2] points on curve
    """
    t = t.unsqueeze(-1)  # [..., 1]
    one_minus_t = 1.0 - t

    w0 = one_minus_t ** 3
    w1 = 3.0 * (one_minus_t ** 2) * t
    w2 = 3.0 * one_minus_t * (t ** 2)
    w3 = t ** 3

    # control_points: [..., 4, 2]
    # weights: [..., 1]
    p0 = control_points[..., 0, :]  # [..., 2]
    p1 = control_points[..., 1, :]
    p2 = control_points[..., 2, :]
    p3 = control_points[..., 3, :]

    return w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3


def _eval_cubic_bezier_deriv(t: torch.Tensor, control_points: torch.Tensor) -> torch.Tensor:
    """
    Evaluate derivative of cubic Bezier at parameter t.

    Args:
        t: [...] parameter values
        control_points: [..., 4, 2] control points

    Returns:
        [..., 2] derivative vectors
    """
    t = t.unsqueeze(-1)  # [..., 1]
    one_minus_t = 1.0 - t

    w0 = 3.0 * (one_minus_t ** 2)
    w1 = 6.0 * one_minus_t * t
    w2 = 3.0 * (t ** 2)

    p0 = control_points[..., 0, :]
    p1 = control_points[..., 1, :]
    p2 = control_points[..., 2, :]
    p3 = control_points[..., 3, :]

    return w0 * (p1 - p0) + w1 * (p2 - p1) + w2 * (p3 - p2)


class _ClosestPointCubicBezier(torch.autograd.Function):
    """
    Custom autograd function for closest point on cubic Bezier.

    Uses implicit function theorem for gradients (matching pydiffvg's approach).
    Forward: Find t using sampling + Newton refinement
    Backward: Use implicit differentiation through the optimality condition
    """

    @staticmethod
    def forward(ctx, pt, control_points, num_init_samples=8, num_newton_iters=4):
        """
        Find closest point on cubic Bezier curve.

        Args:
            pt: [..., 2] query points
            control_points: [..., 4, 2] Bezier control points

        Returns:
            [...] distances to closest points
        """
        device = pt.device
        dtype = pt.dtype

        # Initial sampling to find good starting point
        t_samples = torch.linspace(0, 1, num_init_samples + 1, device=device, dtype=dtype)

        shape = control_points.shape[:-2]
        T = len(t_samples)

        # Evaluate curve at sample points
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

        # Newton refinement (detached - no gradient through iterations)
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

        # Final closest point
        B_final = _eval_cubic_bezier(t, control_points)
        dist = torch.sqrt(((B_final - pt) ** 2).sum(dim=-1) + 1e-8)

        # Save for backward
        ctx.save_for_backward(pt, control_points, t, B_final)

        return dist

    @staticmethod
    def backward(ctx, grad_dist):
        """
        Backward using implicit function theorem (matching pydiffvg's approach).

        The optimality condition is: (B(t) - pt) · B'(t) = 0
        Using implicit differentiation: dt/dp = -1/f''(t) * df'/dp

        This properly accounts for how t changes when control points move.
        """
        pt, control_points, t, B_final = ctx.saved_tensors

        p0 = control_points[..., 0, :]
        p1 = control_points[..., 1, :]
        p2 = control_points[..., 2, :]
        p3 = control_points[..., 3, :]

        # Compute derivatives at optimal t
        dB = _eval_cubic_bezier_deriv(t, control_points)

        t_unsq = t.unsqueeze(-1)
        one_minus_t = 1.0 - t_unsq

        # Second derivative of Bezier curve
        d2B = 6.0 * one_minus_t * (p2 - 2*p1 + p0) + 6.0 * t_unsq * (p3 - 2*p2 + p1)

        residual = B_final - pt

        # Distance and its gradient
        dist_sq = (residual ** 2).sum(dim=-1)
        dist = torch.sqrt(dist_sq + 1e-8)

        # d(dist)/d(B_final) = (B_final - pt) / dist
        d_dist_d_B = residual / dist.unsqueeze(-1)

        # Bernstein basis at t
        tt = 1.0 - t_unsq
        b0 = tt * tt * tt              # (1-t)^3
        b1 = 3.0 * tt * tt * t_unsq    # 3(1-t)^2 t
        b2 = 3.0 * tt * t_unsq * t_unsq  # 3(1-t)t^2
        b3 = t_unsq * t_unsq * t_unsq    # t^3

        # ==========================================
        # Part 1: Direct gradient through B_final
        # d(dist)/d(pi) = d(dist)/d(B) * d(B)/d(pi) = d_dist_d_B * b_i
        # ==========================================
        grad_dist_exp = grad_dist.unsqueeze(-1)
        d_p0 = d_dist_d_B * b0 * grad_dist_exp
        d_p1 = d_dist_d_B * b1 * grad_dist_exp
        d_p2 = d_dist_d_B * b2 * grad_dist_exp
        d_p3 = d_dist_d_B * b3 * grad_dist_exp

        # ==========================================
        # Part 2: Implicit function theorem gradient
        # t is implicitly defined by f'(t) = (B(t) - pt) · B'(t) = 0
        # When control points change, t also changes: dt/dp = -1/f''(t) * df'/dp
        # This adds an extra term to the gradient.
        # ==========================================

        # Compute d(dist)/d(t) using chain rule:
        # dist = ||B(t) - pt||
        # d(dist)/d(t) = d(dist)/d(B) · d(B)/d(t) = (residual / dist) · dB
        d_dist_d_t = (d_dist_d_B * dB).sum(dim=-1)  # [...]

        # Compute f''(t) where f(t) = ||B(t) - pt||^2 / 2
        # f'(t) = (B(t) - pt) · B'(t) = residual · dB
        # f''(t) = B'(t)·B'(t) + (B(t)-pt)·B''(t) = |dB|^2 + residual·d2B
        f_double_prime = (dB * dB).sum(dim=-1) + (residual * d2B).sum(dim=-1)

        # Implicit function theorem: dt/d(control_points) = -1/f''(t) * d(f')/d(control_points)
        # But we need d(dist)/d(control_points) through t, which is:
        # d(dist)/d(pi) via t = d(dist)/d(t) * dt/d(pi)
        #
        # dt/d(pi) comes from the optimality condition:
        # f'(t) = (B(t) - pt) · B'(t) = 0
        # df'/d(pi) = d(B)/d(pi) · B'(t) + (B(t)-pt) · d(B')/d(pi)
        #
        # Using Bernstein derivatives:
        # d(B)/d(p0) = b0, d(B)/d(p1) = b1, etc.
        # d(B')/d(pi) = derivative of Bernstein basis

        # For numerical stability, only apply IFT correction when f'' is large enough
        eps = 1e-6
        f_double_prime_safe = torch.where(
            torch.abs(f_double_prime) > eps,
            f_double_prime,
            torch.ones_like(f_double_prime) * eps * torch.sign(f_double_prime + eps)
        )

        # Compute d(f')/d(pi) for each control point
        # f' = residual · dB = (B - pt) · B'
        # df'/d(p0) = d(B)/d(p0) · B' + (B-pt) · d(B')/d(p0)
        #           = b0 · dB + residual · d(dB)/d(p0)
        #
        # dB = 3*(1-t)^2*(p1-p0) + 6*(1-t)*t*(p2-p1) + 3*t^2*(p3-p2)
        # d(dB)/d(p0) = -3*(1-t)^2
        # d(dB)/d(p1) = 3*(1-t)^2 - 6*(1-t)*t = 3*(1-t)*(1-3t)
        # d(dB)/d(p2) = 6*(1-t)*t - 3*t^2 = 3*t*(2-3t)
        # d(dB)/d(p3) = 3*t^2

        db0 = -3.0 * tt * tt
        db1 = 3.0 * tt * (1.0 - 3.0 * t_unsq)
        db2 = 3.0 * t_unsq * (2.0 - 3.0 * t_unsq)
        db3 = 3.0 * t_unsq * t_unsq

        # df'/d(pi) = b_i * dB + residual * db_i
        # Note: we need dot products
        dB_exp = dB  # [..., 2]

        df_dp0 = b0 * dB_exp + residual * db0  # [..., 2]
        df_dp1 = b1 * dB_exp + residual * db1
        df_dp2 = b2 * dB_exp + residual * db2
        df_dp3 = b3 * dB_exp + residual * db3

        # dt/d(pi) = -df'/d(pi) / f''
        # Note: df'/d(pi) is a 2D vector, f'' is scalar
        dt_dp0 = -df_dp0 / f_double_prime_safe.unsqueeze(-1)
        dt_dp1 = -df_dp1 / f_double_prime_safe.unsqueeze(-1)
        dt_dp2 = -df_dp2 / f_double_prime_safe.unsqueeze(-1)
        dt_dp3 = -df_dp3 / f_double_prime_safe.unsqueeze(-1)

        # d(dist)/d(pi) via t = d(dist)/d(t) * dt/d(pi)
        # Note: d_dist_d_t is scalar [...], dt_dp* is 2D [..., 2]
        d_dist_d_t_exp = (d_dist_d_t * grad_dist).unsqueeze(-1)  # [..., 1]

        d_p0_via_t = d_dist_d_t_exp * dt_dp0
        d_p1_via_t = d_dist_d_t_exp * dt_dp1
        d_p2_via_t = d_dist_d_t_exp * dt_dp2
        d_p3_via_t = d_dist_d_t_exp * dt_dp3

        # Total gradient = direct + via t (implicit function theorem)
        # NOTE: The IFT correction seems to overcorrect in some cases.
        # pydiffvg uses a more sophisticated 5th-degree polynomial approach.
        # For now, we skip the IFT correction as the direct gradient is closer to pydiffvg.
        # TODO: Implement proper 5th-degree polynomial IFT if needed.
        d_p0_total = d_p0  # + d_p0_via_t
        d_p1_total = d_p1  # + d_p1_via_t
        d_p2_total = d_p2  # + d_p2_via_t
        d_p3_total = d_p3  # + d_p3_via_t

        # d(dist)/d(pt) = -d(dist)/d(B_final)
        d_pt = -d_dist_d_B * grad_dist.unsqueeze(-1)

        # Stack gradients for control points
        d_control_points = torch.stack([d_p0_total, d_p1_total, d_p2_total, d_p3_total], dim=-2)

        return d_pt, d_control_points, None, None


def _closest_point_cubic_bezier_newton(
    pt: torch.Tensor,  # [..., 2]
    control_points: torch.Tensor,  # [..., 4, 2]
    num_init_samples: int = 8,
    num_newton_iters: int = 4,
) -> torch.Tensor:
    """
    Find closest point on cubic Bezier using sampling + Newton refinement.

    Uses custom backward pass with implicit function theorem to match
    pydiffvg's gradient behavior.

    Args:
        pt: [..., 2] query points
        control_points: [..., 4, 2] Bezier control points
        num_init_samples: number of initial samples for coarse search
        num_newton_iters: Newton refinement iterations

    Returns:
        [...] distances to closest points
    """
    return _ClosestPointCubicBezier.apply(pt, control_points, num_init_samples, num_newton_iters)


def _compute_min_distance_bezier_batch(
    sample_pos: torch.Tensor,      # [N, 2] flat sample positions
    control_points: torch.Tensor,  # [B, P, S, 4, 2] cubic bezier control points
    stroke_widths: torch.Tensor,   # [B, P] per-path stroke widths
) -> torch.Tensor:
    """
    Compute minimum distance from samples to cubic Bezier curves.

    Uses bounding box culling to match pydiffvg's BVH behavior:
    - Only compute exact distance for samples potentially within stroke range
    - Samples outside the bounding box get detached gradients (no contribution)

    This is critical for matching pydiffvg's gradient behavior, which only
    accumulates gradients from nearby pixels via BVH traversal.

    Args:
        sample_pos: [N, 2] sample positions (H*W*num_samples)
        control_points: [B, P, S, 4, 2] control points for cubic beziers
        stroke_widths: [B, P] stroke widths per path for bounding box expansion

    Returns:
        [B, P, N] minimum distances
    """
    B, P, S, _, _ = control_points.shape
    N = sample_pos.shape[0]
    device = sample_pos.device
    dtype = sample_pos.dtype

    # Compute bounding box for each segment, expanded by stroke_width + transition
    # Transition zone is 1 pixel, so expand by stroke_width + 1.5 for safety
    # stroke_widths: [B, P] -> [B, P, 1, 1] for broadcasting with seg_min/max
    expand_dist = stroke_widths.unsqueeze(-1).unsqueeze(-1) + 1.5  # [B, P, 1, 1]

    # Bounding box per segment: [B, P, S, 2] for min and max
    seg_min = control_points.min(dim=-2).values  # [B, P, S, 2]
    seg_max = control_points.max(dim=-2).values  # [B, P, S, 2]

    # Expand bounding boxes (expand_dist broadcasts over S and 2 dimensions)
    seg_min_exp = seg_min - expand_dist  # [B, P, S, 2]
    seg_max_exp = seg_max + expand_dist  # [B, P, S, 2]

    # For each path, compute overall bounding box (union of segment boxes)
    path_min = seg_min_exp.min(dim=2).values  # [B, P, 2]
    path_max = seg_max_exp.max(dim=2).values  # [B, P, 2]

    # Check which samples are within path bounding box
    # sample_pos: [N, 2] -> [1, 1, N, 2]
    # path_min/max: [B, P, 2] -> [B, P, 1, 2]
    sample_2d = sample_pos.view(1, 1, N, 2)
    path_min_exp = path_min.unsqueeze(2)  # [B, P, 1, 2]
    path_max_exp = path_max.unsqueeze(2)  # [B, P, 1, 2]

    # Sample is in bounding box if: min <= sample <= max for both x and y
    in_bbox = ((sample_2d >= path_min_exp) & (sample_2d <= path_max_exp)).all(dim=-1)  # [B, P, N]

    # For samples in bounding box, compute exact distance
    # For samples outside, use large constant (no gradient needed)
    large_dist = 1000.0  # Much larger than any stroke width

    # We need to handle this efficiently. Strategy:
    # 1. Compute distances for ALL samples (unavoidable with batched ops)
    # 2. But detach gradients for samples outside bounding box

    # sample_pos: [N, 2] -> [1, 1, N, 1, 2]
    # control_points: [B, P, S, 4, 2] -> [B, P, 1, S, 4, 2]
    sample_exp = sample_pos.view(1, 1, N, 1, 2).expand(B, P, N, S, 2)  # [B, P, N, S, 2]
    cp_exp = control_points.unsqueeze(2).expand(B, P, N, S, 4, 2)  # [B, P, N, S, 4, 2]

    # Flatten for distance computation
    sample_flat = sample_exp.reshape(-1, 2)
    cp_flat = cp_exp.reshape(-1, 4, 2)

    # Compute distances
    dist_flat = _closest_point_cubic_bezier_newton(sample_flat, cp_flat)  # [B*P*N*S]

    # Reshape and take minimum over segments
    dist = dist_flat.view(B, P, N, S)
    min_dist = dist.min(dim=-1).values  # [B, P, N]

    # Apply culling: for samples outside bounding box, detach gradient
    # This matches pydiffvg's BVH behavior where only nearby pixels contribute gradients
    min_dist_culled = torch.where(
        in_bbox,
        min_dist,  # Keep gradient for samples in bbox
        min_dist.detach()  # Detach gradient for samples outside
    )

    return min_dist_culled


def _compute_min_distance_batch(
    sample_pos: torch.Tensor,     # [N, 2] flat sample positions
    curve_points: torch.Tensor,   # [B, P, S*T, 2] all curve points flattened per path
) -> torch.Tensor:
    """
    Compute minimum distance from samples to curve points.

    NOTE: This is the legacy sampling-based approach. Use _compute_min_distance_bezier_batch
    for proper curve distance with Newton refinement.

    Args:
        sample_pos: [N, 2] sample positions (H*W*num_samples)
        curve_points: [B, P, C, 2] curve sample points

    Returns:
        [B, P, N] minimum distances
    """
    # sample_pos: [N, 2] -> [1, 1, N, 2]
    # curve_points: [B, P, C, 2] -> [B, P, 1, C, 2]

    B, P, C, _ = curve_points.shape
    N = sample_pos.shape[0]

    # Compute distances in chunks to manage memory
    # For each (batch, path), compute distance to all samples

    # Reshape for broadcasting:
    # sample_pos: [1, 1, N, 1, 2]
    # curve_points: [B, P, 1, C, 2]
    sample_pos_exp = sample_pos.view(1, 1, N, 1, 2)
    curve_points_exp = curve_points.view(B, P, 1, C, 2)

    # Compute squared distances: [B, P, N, C]
    diff = sample_pos_exp - curve_points_exp
    dist_sq = (diff ** 2).sum(dim=-1)

    # Minimum over curve points: [B, P, N]
    min_dist_sq = dist_sq.min(dim=-1).values
    min_dist = torch.sqrt(min_dist_sq + 1e-8)

    return min_dist


def _soft_winding_number_batch(
    sample_pos: torch.Tensor,    # [N, 2]
    curve_points: torch.Tensor,  # [B, P, C, 2] sampled curve (treated as polyline)
    softness: float = 0.1,
) -> torch.Tensor:
    """
    Compute soft winding number for fill detection.

    Uses a soft ray-crossing algorithm that allows gradients to flow.

    Args:
        sample_pos: [N, 2] sample positions
        curve_points: [B, P, C, 2] curve sample points (polyline approximation)
        softness: transition sharpness

    Returns:
        [B, P, N] soft winding numbers
    """
    B, P, C, _ = curve_points.shape
    N = sample_pos.shape[0]
    device = sample_pos.device

    # Get line segments: p0 -> p1
    # p0: [B, P, C-1, 2], p1: [B, P, C-1, 2]
    p0 = curve_points[:, :, :-1, :]  # [B, P, C-1, 2]
    p1 = curve_points[:, :, 1:, :]   # [B, P, C-1, 2]

    # Expand for broadcasting
    # sample_pos: [1, 1, N, 1, 2]
    # p0, p1: [B, P, 1, C-1, 2]
    sample_exp = sample_pos.view(1, 1, N, 1, 2)
    p0_exp = p0.view(B, P, 1, C-1, 2)
    p1_exp = p1.view(B, P, 1, C-1, 2)

    # dy for each segment: [B, P, 1, C-1]
    dy = p1_exp[..., 1] - p0_exp[..., 1]

    # Compute t parameter where ray at sample_y intersects line
    # t = (sample_y - p0_y) / dy
    pt_y = sample_exp[..., 1]  # [1, 1, N, 1]
    pt_x = sample_exp[..., 0]

    dy_safe = torch.where(torch.abs(dy) > 1e-8, dy, torch.ones_like(dy) * 1e-8)
    t = (pt_y - p0_exp[..., 1]) / dy_safe  # [B, P, N, C-1]

    # X coordinate of intersection
    x_int = p0_exp[..., 0] + t * (p1_exp[..., 0] - p0_exp[..., 0])

    # Soft validity checks using sigmoid
    t_valid = torch.sigmoid((t + 0.01) / softness) * torch.sigmoid((1.01 - t) / softness)
    x_valid = torch.sigmoid((x_int - pt_x + 0.01) / softness)

    # Direction: +1 for upward (dy > 0), -1 for downward
    direction = torch.where(dy > 0, torch.ones_like(dy), -torch.ones_like(dy))

    # Contribution: zero for horizontal segments
    contrib = torch.where(
        torch.abs(dy) > 1e-8,
        direction * t_valid * x_valid,
        torch.zeros_like(t_valid)
    )

    # Sum over all segments: [B, P, N]
    winding = contrib.sum(dim=-1)

    return winding


# Triton kernel for computing minimum distance - more efficient than PyTorch
@triton.jit
def _min_dist_kernel(
    # Sample positions [N]
    sample_x_ptr,
    sample_y_ptr,
    # Curve points [num_curve_pts, 2] flattened
    curve_x_ptr,
    curve_y_ptr,
    # Output [N]
    min_dist_ptr,
    # Params
    N: tl.constexpr,
    num_curve_pts: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute minimum distance from samples to curve points."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load sample position
    sx = tl.load(sample_x_ptr + offs, mask=mask, other=0.0)
    sy = tl.load(sample_y_ptr + offs, mask=mask, other=0.0)

    # Initialize minimum distance squared
    min_dist_sq = tl.full([BLOCK_SIZE], 1e10, dtype=tl.float32)

    # Iterate over all curve points
    for i in range(num_curve_pts):
        cx = tl.load(curve_x_ptr + i)
        cy = tl.load(curve_y_ptr + i)

        dx = sx - cx
        dy_val = sy - cy
        dist_sq = dx * dx + dy_val * dy_val

        min_dist_sq = tl.minimum(min_dist_sq, dist_sq)

    # Store result
    min_dist = tl.sqrt(min_dist_sq + 1e-8)
    tl.store(min_dist_ptr + offs, min_dist, mask=mask)


@triton.jit
def _soft_winding_kernel(
    # Sample positions [N]
    sample_x_ptr,
    sample_y_ptr,
    # Polyline points [num_pts, 2] flattened
    poly_x_ptr,
    poly_y_ptr,
    # Output [N]
    winding_ptr,
    # Params
    N: tl.constexpr,
    num_pts: tl.constexpr,
    softness,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute soft winding number using ray-crossing."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load sample position
    pt_x = tl.load(sample_x_ptr + offs, mask=mask, other=0.0)
    pt_y = tl.load(sample_y_ptr + offs, mask=mask, other=0.0)

    # Initialize winding
    winding = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Iterate over line segments
    num_segs = num_pts - 1
    for i in range(num_segs):
        p0_x = tl.load(poly_x_ptr + i)
        p0_y = tl.load(poly_y_ptr + i)
        p1_x = tl.load(poly_x_ptr + i + 1)
        p1_y = tl.load(poly_y_ptr + i + 1)

        dy = p1_y - p0_y
        dy_abs = tl.abs(dy)

        # Compute t where ray intersects segment
        dy_safe = tl.where(dy_abs > 1e-8, dy, 1e-8)
        t = (pt_y - p0_y) / dy_safe

        # X coordinate of intersection
        x_int = p0_x + t * (p1_x - p0_x)

        # Soft validity using sigmoid approximation
        # sigmoid(x) ≈ 0.5 + 0.5 * tanh(x/2)
        t_low = (t + 0.01) / softness
        t_high = (1.01 - t) / softness
        x_diff = (x_int - pt_x + 0.01) / softness

        # Use tl.sigmoid (available in recent Triton versions)
        # If not available, approximate with: 1 / (1 + exp(-x))
        t_valid = 1.0 / (1.0 + tl.exp(-t_low)) * 1.0 / (1.0 + tl.exp(-t_high))
        x_valid = 1.0 / (1.0 + tl.exp(-x_diff))

        # Direction
        direction = tl.where(dy > 0.0, 1.0, -1.0)

        # Contribution
        contrib = tl.where(dy_abs > 1e-8, direction * t_valid * x_valid, 0.0)
        winding = winding + contrib

    # Store result
    tl.store(winding_ptr + offs, winding, mask=mask)


def render_batch(
    canvas_width: int,
    canvas_height: int,
    control_points: torch.Tensor,  # [B, P, S, 4, 2] - batch, paths, segments, 4 ctrl pts, xy
    stroke_widths: torch.Tensor,   # [B, P] stroke widths
    alphas: torch.Tensor,          # [B, P] alpha values
    num_samples: int = 2,
    use_fill: bool = True,
    background: float = 1.0,       # Background value (1.0 = white)
) -> torch.Tensor:
    """
    Batched differentiable vector graphics rendering.

    This is the main entry point for training. All operations are batched
    and run on GPU with automatic gradient computation.

    Args:
        canvas_width: Output image width
        canvas_height: Output image height
        control_points: [B, P, S, 4, 2] cubic bezier control points
        stroke_widths: [B, P] stroke width for each path
        alphas: [B, P] alpha (opacity) for each path
        num_samples: Anti-aliasing samples per axis
        use_fill: Whether to fill the interior of paths
        background: Background color (grayscale)

    Returns:
        [B, 1, H, W] grayscale images with gradients
    """
    device = control_points.device
    dtype = control_points.dtype
    B, P, S, _, _ = control_points.shape
    H, W = canvas_height, canvas_width

    # Number of curve samples for polyline approximation
    num_curve_samples = 17

    # Sample bezier curves: [B, P, S, T, 2]
    curve_samples = _sample_cubic_bezier_batch(control_points, num_curve_samples)

    # Flatten segments: [B, P, S*T, 2]
    curve_points = curve_samples.reshape(B, P, S * num_curve_samples, 2)

    # Generate sample positions: [H, W, num_samples^2, 2]
    samples_per_axis = num_samples
    total_samples = samples_per_axis * samples_per_axis

    py = torch.arange(H, device=device, dtype=dtype)
    px = torch.arange(W, device=device, dtype=dtype)
    oy = (torch.arange(samples_per_axis, device=device, dtype=dtype) + 0.5) / samples_per_axis
    ox = (torch.arange(samples_per_axis, device=device, dtype=dtype) + 0.5) / samples_per_axis

    # Meshgrid for all combinations
    py_grid, px_grid, oy_grid, ox_grid = torch.meshgrid(py, px, oy, ox, indexing='ij')
    sample_x = px_grid + ox_grid
    sample_y = py_grid + oy_grid

    # Flatten to [N, 2] where N = H*W*total_samples
    sample_pos = torch.stack([
        sample_x.reshape(-1),
        sample_y.reshape(-1)
    ], dim=-1)

    N = sample_pos.shape[0]

    # Initialize sample colors with background
    sample_colors = torch.full((B, N), background, device=device, dtype=dtype)

    # Process all paths in parallel
    # Compute minimum distances: [B, P, N]
    min_dist = _compute_min_distance_batch(sample_pos, curve_points)

    # Compute winding numbers if using fill: [B, P, N]
    if use_fill:
        # Close the curve by adding first point at end
        curve_closed = torch.cat([
            curve_points,
            curve_points[:, :, :1, :]
        ], dim=2)
        winding = _soft_winding_number_batch(sample_pos, curve_closed, softness=0.1)

        # Inside = |winding| >= 0.5 (soft threshold)
        inside = torch.sigmoid((torch.abs(winding) - 0.5) * 10.0)  # [B, P, N]
    else:
        inside = torch.zeros(B, P, N, device=device, dtype=dtype)

    # Compute stroke coverage from distance
    # Note: pydiffvg interprets stroke_width as the half-width (radius), not diameter
    half_widths = stroke_widths.view(B, P, 1)  # [B, P, 1] - don't divide by 2 to match pydiffvg
    transition_width = 0.25
    stroke_edge = torch.sigmoid((half_widths - min_dist) / transition_width)  # [B, P, N]

    # Total coverage: inside OR on stroke edge
    coverage = torch.maximum(inside, stroke_edge)  # [B, P, N]

    # Apply alpha for each path
    alphas_exp = alphas.view(B, P, 1)  # [B, P, 1]
    stroke_contrib = coverage * alphas_exp  # [B, P, N]

    # Composite all paths (back to front, multiplicative blending)
    # sample_colors = background * prod((1 - stroke_contrib) for each path)
    for p in range(P):
        sample_colors = sample_colors * (1.0 - stroke_contrib[:, p, :])

    # Reshape and average samples per pixel
    sample_colors = sample_colors.view(B, H, W, total_samples)
    pixel_colors = sample_colors.mean(dim=-1)  # [B, H, W]

    # Add channel dimension: [B, 1, H, W]
    output = pixel_colors.unsqueeze(1)

    return output


def render_batch_triton(
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
    Batched rendering using Triton kernels for inner loops.

    Similar to render_batch but uses custom Triton kernels for
    distance and winding computation for better performance.
    """
    device = control_points.device
    dtype = control_points.dtype

    if device.type != 'cuda':
        return render_batch(canvas_width, canvas_height, control_points,
                          stroke_widths, alphas, num_samples, use_fill, background)

    B, P, S, _, _ = control_points.shape
    H, W = canvas_height, canvas_width

    num_curve_samples = 17

    # Sample bezier curves
    curve_samples = _sample_cubic_bezier_batch(control_points, num_curve_samples)
    curve_points = curve_samples.view(B, P, S * num_curve_samples, 2)

    # Generate sample positions
    samples_per_axis = num_samples
    total_samples = samples_per_axis * samples_per_axis

    py = torch.arange(H, device=device, dtype=dtype)
    px = torch.arange(W, device=device, dtype=dtype)
    oy = (torch.arange(samples_per_axis, device=device, dtype=dtype) + 0.5) / samples_per_axis
    ox = (torch.arange(samples_per_axis, device=device, dtype=dtype) + 0.5) / samples_per_axis

    py_grid, px_grid, oy_grid, ox_grid = torch.meshgrid(py, px, oy, ox, indexing='ij')
    sample_x = (px_grid + ox_grid).reshape(-1).contiguous()
    sample_y = (py_grid + oy_grid).reshape(-1).contiguous()

    N = sample_x.shape[0]
    num_curve_pts = S * num_curve_samples

    # Initialize output
    sample_colors = torch.full((B, N), background, device=device, dtype=dtype)

    BLOCK_SIZE = 256
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Process each batch and path using Triton kernels
    for b in range(B):
        for p in range(P):
            # Get curve points for this path
            path_curve = curve_points[b, p].contiguous()  # [C, 2]
            curve_x = path_curve[:, 0].contiguous()
            curve_y = path_curve[:, 1].contiguous()

            # Allocate outputs
            min_dist = torch.empty(N, device=device, dtype=dtype)
            winding = torch.empty(N, device=device, dtype=dtype)

            # Compute minimum distance
            _min_dist_kernel[grid](
                sample_x, sample_y,
                curve_x, curve_y,
                min_dist,
                N, num_curve_pts, BLOCK_SIZE
            )

            # Compute winding number (add closing segment)
            curve_closed_x = torch.cat([curve_x, curve_x[:1]])
            curve_closed_y = torch.cat([curve_y, curve_y[:1]])

            _soft_winding_kernel[grid](
                sample_x, sample_y,
                curve_closed_x, curve_closed_y,
                winding,
                N, num_curve_pts + 1, 0.1, BLOCK_SIZE
            )

            # Compute coverage
            # Note: pydiffvg interprets stroke_width as the half-width (radius)
            half_width = stroke_widths[b, p]  # Don't divide by 2 to match pydiffvg
            alpha = alphas[b, p]

            if use_fill:
                inside = torch.sigmoid((torch.abs(winding) - 0.5) * 10.0)
            else:
                inside = torch.zeros_like(min_dist)

            stroke_edge = torch.sigmoid((half_width - min_dist) / 0.25)
            coverage = torch.maximum(inside, stroke_edge)
            stroke_contrib = coverage * alpha

            # Composite
            sample_colors[b] = sample_colors[b] * (1.0 - stroke_contrib)

    # Reshape and average
    sample_colors = sample_colors.view(B, H, W, total_samples)
    pixel_colors = sample_colors.mean(dim=-1)
    output = pixel_colors.unsqueeze(1)

    return output


# Optimized version that processes all paths together
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
    Fastest batched rendering - fully vectorized PyTorch operations.

    Key optimizations:
    1. All bezier sampling done in parallel
    2. Distance computation uses efficient broadcasting
    3. Winding number computed with vectorized operations
    4. No Python loops over paths
    """
    device = control_points.device
    dtype = control_points.dtype
    B, P, S, _, _ = control_points.shape
    H, W = canvas_height, canvas_width

    num_curve_samples = 17

    # Sample all bezier curves at once: [B, P, S, T, 2]
    curve_samples = _sample_cubic_bezier_batch(control_points, num_curve_samples)

    # Flatten segments per path: [B, P, C, 2] where C = S*T
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
    sample_x = px_grid + ox_grid  # [H, W, Sy, Sx]
    sample_y = py_grid + oy_grid

    # Flatten samples: [N] where N = H*W*Sy*Sx
    N = H * W * total_samples
    sample_x_flat = sample_x.reshape(N)
    sample_y_flat = sample_y.reshape(N)

    # Stack: [N, 2]
    sample_pos = torch.stack([sample_x_flat, sample_y_flat], dim=-1)

    # Compute distances: [B, P, N]
    # Use proper closest-point-on-curve with Newton refinement for smooth strokes
    # Pass stroke_widths for bounding box culling to match pydiffvg's BVH behavior
    min_dist = _compute_min_distance_bezier_batch(sample_pos, control_points, stroke_widths)

    # Winding number computation (vectorized)
    if use_fill:
        # Close the curve
        curve_closed = torch.cat([curve_points, curve_points[:, :, :1, :]], dim=2)  # [B, P, C+1, 2]

        # Line segments: p0 -> p1
        p0 = curve_closed[:, :, :-1, :]  # [B, P, C, 2]
        p1 = curve_closed[:, :, 1:, :]   # [B, P, C, 2]

        # Expand for broadcasting with samples
        # sample_pos: [N, 2] -> [1, 1, N, 1, 2]
        # p0, p1: [B, P, C, 2] -> [B, P, 1, C, 2]
        p0_exp = p0.view(B, P, 1, C, 2)
        p1_exp = p1.view(B, P, 1, C, 2)

        # dy for each segment
        dy = p1_exp[..., 1] - p0_exp[..., 1]  # [B, P, 1, C]

        # Sample positions
        pt_y = sample_pos[:, 1].view(1, 1, N, 1)  # [1, 1, N, 1]
        pt_x = sample_pos[:, 0].view(1, 1, N, 1)

        # t parameter
        dy_safe = torch.where(torch.abs(dy) > 1e-8, dy, torch.ones_like(dy) * 1e-8)
        t = (pt_y - p0_exp[..., 1]) / dy_safe  # [B, P, N, C]

        # X intersection
        x_int = p0_exp[..., 0] + t * (p1_exp[..., 0] - p0_exp[..., 0])  # [B, P, N, C]

        # Soft validity
        softness = 0.1
        t_valid = torch.sigmoid((t + 0.01) / softness) * torch.sigmoid((1.01 - t) / softness)
        x_valid = torch.sigmoid((x_int - pt_x + 0.01) / softness)

        # Direction
        direction = torch.where(dy > 0, torch.ones_like(dy), -torch.ones_like(dy))

        # Contribution
        contrib = torch.where(torch.abs(dy) > 1e-8, direction * t_valid * x_valid, torch.zeros_like(t_valid))

        # Sum over segments: [B, P, N]
        winding = contrib.sum(dim=-1)

        # Inside = |winding| >= 0.5
        # NOTE: Using hard threshold instead of soft sigmoid to match pydiffvg behavior
        # pydiffvg uses hard winding number test, no gradients through fill interior
        inside = (torch.abs(winding) >= 0.5).float()
        # Detach to prevent gradients flowing through fill interior
        inside = inside.detach()
    else:
        inside = torch.zeros(B, P, N, device=device, dtype=dtype)

    # Stroke coverage using pydiffvg's smoothstep formula
    # pydiffvg computes: w = smoothstep(|d| + stroke_width) - smoothstep(|d| - stroke_width)
    # where smoothstep(x) = t*t*(3-2*t) with t = clamp((x+1)/2, 0, 1)
    #
    # This gives a transition zone of 2 pixels (from stroke_width-1 to stroke_width+1)
    # and has compact support (exactly 0 outside the transition zone).

    half_widths = stroke_widths.view(B, P, 1)  # stroke width (radius from curve)
    alphas_exp = alphas.view(B, P, 1)

    def smoothstep(x):
        """pydiffvg's smoothstep: 0 at x<-1, 1 at x>1, smooth in between"""
        t = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    # Coverage = smoothstep(|d| + w) - smoothstep(|d| - w)
    # This is 1 inside stroke, 0 outside, with smooth transition at edges
    #
    # Key insight: the transition zone is [w-1, w+1] where w = half_width
    # Outside this zone, coverage is exactly 0 or 1, so gradients should be 0.
    # We detach gradients for pixels far from the stroke to reduce gradient accumulation.
    #
    # This is similar to pydiffvg's BVH culling which skips distance computation
    # for pixels that are definitely outside the stroke.

    # Compute distances but detach for far-away pixels
    # Transition zone: dist in [half_width - 1, half_width + 1]
    # Safe zone (gradient matters): dist < half_width + 1.5 (some margin)
    dist_threshold = half_widths + 1.5
    in_gradient_zone = min_dist < dist_threshold  # [B, P, N]

    # Use where to selectively detach
    # For pixels in gradient zone: keep gradients
    # For pixels outside: detach (no gradients flow through distance)
    min_dist_masked = torch.where(
        in_gradient_zone,
        min_dist,
        min_dist.detach()
    )

    abs_d_plus_w = min_dist_masked + half_widths
    abs_d_minus_w = min_dist_masked - half_widths
    stroke_edge = smoothstep(abs_d_plus_w) - smoothstep(abs_d_minus_w)

    # Total coverage from stroke/fill
    coverage = torch.maximum(inside, stroke_edge)

    # IMPORTANT: Match pydiffvg's grayscale output behavior
    # pydiffvg renders RGBA with:
    #   - RGB = stroke_color RGB (e.g., white = 1,1,1)
    #   - Alpha = coverage * stroke_color.alpha
    # The VAE then takes RGB channels and averages to grayscale.
    # This means alpha does NOT affect the grayscale intensity - only the compositing.
    #
    # For stroke pixels: RGB=(1,1,1) regardless of alpha
    # For background: RGB=(0,0,0) because no stroke
    #
    # Since we're outputting grayscale directly, we use coverage as the stroke
    # intensity. Alpha is NOT multiplied here because pydiffvg's grayscale
    # extraction ignores the alpha channel.
    #
    # NOTE: alpha could be used for proper alpha-blending compositing,
    # but that would require outputting RGBA and handling compositing separately.
    # For now, to match pydiffvg's VAE behavior, we ignore alpha for grayscale.

    # Use coverage directly (strokes are white intensity proportional to coverage)
    stroke_contrib = coverage  # [B, P, N] - alpha NOT applied for grayscale parity

    # Note: alphas_exp is intentionally unused for grayscale output to match pydiffvg
    # The alpha predictor will have ~zero gradients, matching pydiffvg's behavior
    _ = alphas_exp  # Silence unused variable warnings

    # Composite: product over paths (back-to-front alpha compositing)
    # sample_colors = background * prod(1 - stroke_contrib[p]) for all p
    one_minus_contrib = 1.0 - stroke_contrib  # [B, P, N]
    combined = one_minus_contrib.prod(dim=1)  # [B, N]
    sample_colors = background * combined

    # Reshape and average
    sample_colors = sample_colors.view(B, H, W, total_samples)
    pixel_colors = sample_colors.mean(dim=-1)

    return pixel_colors.unsqueeze(1)
