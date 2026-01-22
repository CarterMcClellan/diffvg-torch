"""
Distance field computation for curves and shapes.

Computes closest point on curves and signed distance for:
- Line segments
- Quadratic Bezier curves
- Cubic Bezier curves
- Circles
- Rectangles

Ported from diffvg/compute_distance.h
"""

import torch
import triton
import triton.language as tl
import math

from .solve import solve_quadratic_kernel, solve_cubic_kernel


@triton.jit
def closest_point_line(
    pt_x, pt_y,      # Query point
    p0_x, p0_y,      # Line start
    p1_x, p1_y,      # Line end
):
    """
    Find closest point on line segment to query point.

    Returns: (closest_x, closest_y, t, distance_squared)
    where t is the parameter [0, 1] along the segment.
    """
    # Direction vector
    dx = p1_x - p0_x
    dy = p1_y - p0_y

    # Vector from p0 to query point
    px = pt_x - p0_x
    py = pt_y - p0_y

    # Project onto line: t = dot(p, d) / dot(d, d)
    len_sq = dx * dx + dy * dy

    # Handle degenerate line (point)
    is_point = len_sq < 1e-10

    t = tl.where(is_point, 0.0, (px * dx + py * dy) / len_sq)

    # Clamp to segment
    t = tl.minimum(tl.maximum(t, 0.0), 1.0)

    # Closest point
    closest_x = p0_x + t * dx
    closest_y = p0_y + t * dy

    # Distance squared
    diff_x = pt_x - closest_x
    diff_y = pt_y - closest_y
    dist_sq = diff_x * diff_x + diff_y * diff_y

    return closest_x, closest_y, t, dist_sq


@triton.jit
def eval_quadratic_bezier(t, p0_x, p0_y, p1_x, p1_y, p2_x, p2_y):
    """Evaluate quadratic Bezier at parameter t."""
    # B(t) = (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2
    one_minus_t = 1.0 - t
    w0 = one_minus_t * one_minus_t
    w1 = 2.0 * one_minus_t * t
    w2 = t * t

    x = w0 * p0_x + w1 * p1_x + w2 * p2_x
    y = w0 * p0_y + w1 * p1_y + w2 * p2_y

    return x, y


@triton.jit
def eval_quadratic_bezier_deriv(t, p0_x, p0_y, p1_x, p1_y, p2_x, p2_y):
    """Evaluate derivative of quadratic Bezier at parameter t."""
    # B'(t) = 2(1-t)(P1-P0) + 2t(P2-P1) = 2[(1-t)(P1-P0) + t(P2-P1)]
    one_minus_t = 1.0 - t

    dx = 2.0 * (one_minus_t * (p1_x - p0_x) + t * (p2_x - p1_x))
    dy = 2.0 * (one_minus_t * (p1_y - p0_y) + t * (p2_y - p1_y))

    return dx, dy


@triton.jit
def closest_point_quadratic_bezier(
    pt_x, pt_y,           # Query point
    p0_x, p0_y,           # Start point
    p1_x, p1_y,           # Control point
    p2_x, p2_y,           # End point
):
    """
    Find closest point on quadratic Bezier to query point.

    Solves: minimize |B(t) - pt|^2
    Which requires finding roots of: (B(t) - pt) · B'(t) = 0
    This is a cubic equation in t.

    Returns: (closest_x, closest_y, t, distance_squared)
    """
    # Quadratic Bezier: B(t) = (1-t)^2*P0 + 2(1-t)t*P1 + t^2*P2
    # Expanding: B(t) = (P0 - 2P1 + P2)t^2 + (-2P0 + 2P1)t + P0
    #                 = A*t^2 + B*t + C
    # where A = P0 - 2P1 + P2, B = -2P0 + 2P1, C = P0

    # Derivative: B'(t) = 2A*t + B = 2(P0 - 2P1 + P2)t + (-2P0 + 2P1)

    # We want (B(t) - pt) · B'(t) = 0
    # Let D = B(t) - pt = A*t^2 + B*t + (C - pt)
    # D · B' = (A*t^2 + B*t + C') · (2A*t + B) where C' = C - pt
    #        = 2A·A*t^3 + A·B*t^2 + 2B·A*t^2 + B·B*t + 2C'·A*t + C'·B
    #        = 2|A|^2*t^3 + 3(A·B)*t^2 + (|B|^2 + 2A·C')*t + B·C'

    Ax = p0_x - 2.0 * p1_x + p2_x
    Ay = p0_y - 2.0 * p1_y + p2_y
    Bx = -2.0 * p0_x + 2.0 * p1_x
    By = -2.0 * p0_y + 2.0 * p1_y
    Cx = p0_x - pt_x
    Cy = p0_y - pt_y

    # Dot products
    A_dot_A = Ax * Ax + Ay * Ay
    A_dot_B = Ax * Bx + Ay * By
    B_dot_B = Bx * Bx + By * By
    A_dot_C = Ax * Cx + Ay * Cy
    B_dot_C = Bx * Cx + By * Cy

    # Cubic coefficients: a*t^3 + b*t^2 + c*t + d = 0
    a = 2.0 * A_dot_A
    b = 3.0 * A_dot_B
    c = B_dot_B + 2.0 * A_dot_C
    d = B_dot_C

    # Solve cubic
    num_roots, t0, t1, t2 = solve_cubic_kernel(a, b, c, d)

    # Check all candidate t values (roots + endpoints)
    # Start with endpoints
    best_t = 0.0
    bx, by = eval_quadratic_bezier(0.0, p0_x, p0_y, p1_x, p1_y, p2_x, p2_y)
    best_dist_sq = (bx - pt_x) * (bx - pt_x) + (by - pt_y) * (by - pt_y)

    # t = 1
    ex, ey = eval_quadratic_bezier(1.0, p0_x, p0_y, p1_x, p1_y, p2_x, p2_y)
    end_dist_sq = (ex - pt_x) * (ex - pt_x) + (ey - pt_y) * (ey - pt_y)
    better = end_dist_sq < best_dist_sq
    best_t = tl.where(better, 1.0, best_t)
    best_dist_sq = tl.where(better, end_dist_sq, best_dist_sq)

    # Check root 0
    t0_valid = (num_roots >= 1) & (t0 >= 0.0) & (t0 <= 1.0)
    r0x, r0y = eval_quadratic_bezier(t0, p0_x, p0_y, p1_x, p1_y, p2_x, p2_y)
    r0_dist_sq = (r0x - pt_x) * (r0x - pt_x) + (r0y - pt_y) * (r0y - pt_y)
    better0 = t0_valid & (r0_dist_sq < best_dist_sq)
    best_t = tl.where(better0, t0, best_t)
    best_dist_sq = tl.where(better0, r0_dist_sq, best_dist_sq)

    # Check root 1
    t1_valid = (num_roots >= 2) & (t1 >= 0.0) & (t1 <= 1.0)
    r1x, r1y = eval_quadratic_bezier(t1, p0_x, p0_y, p1_x, p1_y, p2_x, p2_y)
    r1_dist_sq = (r1x - pt_x) * (r1x - pt_x) + (r1y - pt_y) * (r1y - pt_y)
    better1 = t1_valid & (r1_dist_sq < best_dist_sq)
    best_t = tl.where(better1, t1, best_t)
    best_dist_sq = tl.where(better1, r1_dist_sq, best_dist_sq)

    # Check root 2
    t2_valid = (num_roots >= 3) & (t2 >= 0.0) & (t2 <= 1.0)
    r2x, r2y = eval_quadratic_bezier(t2, p0_x, p0_y, p1_x, p1_y, p2_x, p2_y)
    r2_dist_sq = (r2x - pt_x) * (r2x - pt_x) + (r2y - pt_y) * (r2y - pt_y)
    better2 = t2_valid & (r2_dist_sq < best_dist_sq)
    best_t = tl.where(better2, t2, best_t)
    best_dist_sq = tl.where(better2, r2_dist_sq, best_dist_sq)

    # Compute final closest point
    closest_x, closest_y = eval_quadratic_bezier(
        best_t, p0_x, p0_y, p1_x, p1_y, p2_x, p2_y
    )

    return closest_x, closest_y, best_t, best_dist_sq


@triton.jit
def eval_cubic_bezier(t, p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y):
    """Evaluate cubic Bezier at parameter t."""
    # B(t) = (1-t)^3*P0 + 3(1-t)^2*t*P1 + 3(1-t)*t^2*P2 + t^3*P3
    one_minus_t = 1.0 - t
    one_minus_t_sq = one_minus_t * one_minus_t
    one_minus_t_cb = one_minus_t_sq * one_minus_t
    t_sq = t * t
    t_cb = t_sq * t

    w0 = one_minus_t_cb
    w1 = 3.0 * one_minus_t_sq * t
    w2 = 3.0 * one_minus_t * t_sq
    w3 = t_cb

    x = w0 * p0_x + w1 * p1_x + w2 * p2_x + w3 * p3_x
    y = w0 * p0_y + w1 * p1_y + w2 * p2_y + w3 * p3_y

    return x, y


@triton.jit
def eval_cubic_bezier_deriv(t, p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y):
    """Evaluate derivative of cubic Bezier at parameter t."""
    # B'(t) = 3(1-t)^2*(P1-P0) + 6(1-t)*t*(P2-P1) + 3*t^2*(P3-P2)
    one_minus_t = 1.0 - t
    one_minus_t_sq = one_minus_t * one_minus_t
    t_sq = t * t

    w0 = 3.0 * one_minus_t_sq
    w1 = 6.0 * one_minus_t * t
    w2 = 3.0 * t_sq

    dx = w0 * (p1_x - p0_x) + w1 * (p2_x - p1_x) + w2 * (p3_x - p2_x)
    dy = w0 * (p1_y - p0_y) + w1 * (p2_y - p1_y) + w2 * (p3_y - p2_y)

    return dx, dy


@triton.jit
def newton_bisection_refine(
    t_init, t_min, t_max,
    # Polynomial coefficients for derivative equation (degree 5)
    c5, c4, c3, c2, c1, c0,
    # Bezier control points for distance evaluation
    pt_x, pt_y,
    p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y,
):
    """
    Refine root using Newton-bisection hybrid.
    20 iterations as in original diffvg.
    """
    t = t_init
    lo = t_min
    hi = t_max

    # 20 iterations of refinement
    for _ in range(20):
        # Evaluate polynomial and derivative at t
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t

        # f(t) = c5*t^5 + c4*t^4 + c3*t^3 + c2*t^2 + c1*t + c0
        f = c5 * t5 + c4 * t4 + c3 * t3 + c2 * t2 + c1 * t + c0

        # f'(t) = 5*c5*t^4 + 4*c4*t^3 + 3*c3*t^2 + 2*c2*t + c1
        df = 5.0 * c5 * t4 + 4.0 * c4 * t3 + 3.0 * c3 * t2 + 2.0 * c2 * t + c1

        # Newton step
        newton_ok = tl.abs(df) > 1e-6
        t_newton = t - f / tl.where(newton_ok, df, 1.0)

        # Check if Newton step is in bounds
        newton_in_bounds = newton_ok & (t_newton >= lo) & (t_newton <= hi)

        # Bisection fallback
        t_bisect = 0.5 * (lo + hi)

        # Choose method
        t_new = tl.where(newton_in_bounds, t_newton, t_bisect)

        # Update bounds for bisection
        f_lo = c5 * (lo**5) + c4 * (lo**4) + c3 * (lo**3) + c2 * (lo**2) + c1 * lo + c0
        same_sign = (f * f_lo) > 0
        lo = tl.where(same_sign, t, lo)
        hi = tl.where(same_sign, hi, t)

        t = t_new

    return t


@triton.jit
def closest_point_cubic_bezier(
    pt_x, pt_y,           # Query point
    p0_x, p0_y,           # Start point
    p1_x, p1_y,           # Control point 1
    p2_x, p2_y,           # Control point 2
    p3_x, p3_y,           # End point
):
    """
    Find closest point on cubic Bezier to query point.

    Solves: minimize |B(t) - pt|^2
    Which requires: (B(t) - pt) · B'(t) = 0
    This is a 5th degree polynomial in t.

    Uses sampling + Newton-bisection refinement approach.

    Returns: (closest_x, closest_y, t, distance_squared)
    """
    # Cubic Bezier expanded:
    # B(t) = A*t^3 + B*t^2 + C*t + D
    # where:
    #   A = -P0 + 3P1 - 3P2 + P3
    #   B = 3P0 - 6P1 + 3P2
    #   C = -3P0 + 3P1
    #   D = P0

    # Derivative:
    # B'(t) = 3A*t^2 + 2B*t + C

    # Distance derivative equation (B(t) - pt) · B'(t) = 0 is degree 5
    # We use a simpler approach: sample + refine

    # Coefficients
    Ax = -p0_x + 3.0 * p1_x - 3.0 * p2_x + p3_x
    Ay = -p0_y + 3.0 * p1_y - 3.0 * p2_y + p3_y
    Bx = 3.0 * p0_x - 6.0 * p1_x + 3.0 * p2_x
    By = 3.0 * p0_y - 6.0 * p1_y + 3.0 * p2_y
    Cx = -3.0 * p0_x + 3.0 * p1_x
    Cy = -3.0 * p0_y + 3.0 * p1_y
    Dx = p0_x - pt_x
    Dy = p0_y - pt_y

    # Build degree-5 polynomial coefficients for (B(t)-pt)·B'(t) = 0
    # Let M(t) = A*t^3 + B*t^2 + C*t + D (where D = P0 - pt)
    # Let N(t) = 3A*t^2 + 2B*t + C
    # M·N is degree 5

    # Dot products for building polynomial
    AA = Ax * Ax + Ay * Ay
    AB = Ax * Bx + Ay * By
    AC = Ax * Cx + Ay * Cy
    AD = Ax * Dx + Ay * Dy
    BB = Bx * Bx + By * By
    BC = Bx * Cx + By * Cy
    BD = Bx * Dx + By * Dy
    CC = Cx * Cx + Cy * Cy
    CD = Cx * Dx + Cy * Dy

    # Polynomial: (A*t^3 + B*t^2 + C*t + D) · (3A*t^2 + 2B*t + C)
    # = 3AA*t^5 + 2AB*t^4 + AC*t^3 + 3AB*t^4 + 2BB*t^3 + BC*t^2
    #   + 3AC*t^3 + 2BC*t^2 + CC*t + 3AD*t^2 + 2BD*t + CD
    # = 3AA*t^5 + (2AB + 3AB)*t^4 + (AC + 2BB + 3AC)*t^3
    #   + (BC + 2BC + 3AD)*t^2 + (CC + 2BD)*t + CD
    # = 3AA*t^5 + 5AB*t^4 + (4AC + 2BB)*t^3 + (3BC + 3AD)*t^2 + (CC + 2BD)*t + CD

    c5 = 3.0 * AA
    c4 = 5.0 * AB
    c3 = 4.0 * AC + 2.0 * BB
    c2 = 3.0 * (BC + AD)
    c1 = CC + 2.0 * BD
    c0 = CD

    # Sample at multiple points to find good starting interval
    # Use 8 samples
    best_t = 0.0
    ex, ey = eval_cubic_bezier(0.0, p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y)
    best_dist_sq = (ex - pt_x) * (ex - pt_x) + (ey - pt_y) * (ey - pt_y)

    # Sample at t = 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0
    for i in range(1, 9):
        t_sample = i * 0.125
        sx, sy = eval_cubic_bezier(t_sample, p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y)
        dist_sq = (sx - pt_x) * (sx - pt_x) + (sy - pt_y) * (sy - pt_y)
        better = dist_sq < best_dist_sq
        best_t = tl.where(better, t_sample, best_t)
        best_dist_sq = tl.where(better, dist_sq, best_dist_sq)

    # Refine with Newton-bisection
    t_lo = tl.maximum(best_t - 0.125, 0.0)
    t_hi = tl.minimum(best_t + 0.125, 1.0)

    refined_t = newton_bisection_refine(
        best_t, t_lo, t_hi,
        c5, c4, c3, c2, c1, c0,
        pt_x, pt_y,
        p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y
    )

    # Clamp to valid range
    refined_t = tl.minimum(tl.maximum(refined_t, 0.0), 1.0)

    # Evaluate at refined t
    rx, ry = eval_cubic_bezier(refined_t, p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y)
    refined_dist_sq = (rx - pt_x) * (rx - pt_x) + (ry - pt_y) * (ry - pt_y)

    # Choose best between sampled and refined
    use_refined = refined_dist_sq < best_dist_sq
    final_t = tl.where(use_refined, refined_t, best_t)
    final_dist_sq = tl.where(use_refined, refined_dist_sq, best_dist_sq)

    # Final closest point
    closest_x, closest_y = eval_cubic_bezier(
        final_t, p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y
    )

    return closest_x, closest_y, final_t, final_dist_sq


@triton.jit
def closest_point_circle(
    pt_x, pt_y,      # Query point
    center_x, center_y,  # Circle center
    radius,          # Circle radius
):
    """
    Find closest point on circle boundary to query point.

    Returns: (closest_x, closest_y, distance_squared)
    """
    # Direction from center to point
    dx = pt_x - center_x
    dy = pt_y - center_y

    # Distance to center
    dist_to_center = tl.sqrt(dx * dx + dy * dy)

    # Handle point at center (degenerate)
    at_center = dist_to_center < 1e-10

    # Normalize direction
    inv_dist = 1.0 / tl.where(at_center, 1.0, dist_to_center)
    nx = tl.where(at_center, 1.0, dx * inv_dist)
    ny = tl.where(at_center, 0.0, dy * inv_dist)

    # Closest point on circle
    closest_x = center_x + radius * nx
    closest_y = center_y + radius * ny

    # Distance is |dist_to_center - radius|
    signed_dist = dist_to_center - radius
    dist_sq = signed_dist * signed_dist

    return closest_x, closest_y, dist_sq


@triton.jit
def closest_point_rect(
    pt_x, pt_y,      # Query point
    p_min_x, p_min_y,  # Rectangle min corner
    p_max_x, p_max_y,  # Rectangle max corner
):
    """
    Find closest point on rectangle boundary to query point.

    Returns: (closest_x, closest_y, distance_squared)
    """
    # Clamp point to rectangle interior for closest point computation
    clamped_x = tl.minimum(tl.maximum(pt_x, p_min_x), p_max_x)
    clamped_y = tl.minimum(tl.maximum(pt_y, p_min_y), p_max_y)

    # Check if point is inside rectangle
    inside = (pt_x >= p_min_x) & (pt_x <= p_max_x) & (pt_y >= p_min_y) & (pt_y <= p_max_y)

    # If inside, find closest edge
    # Distances to each edge
    dist_left = pt_x - p_min_x
    dist_right = p_max_x - pt_x
    dist_bottom = pt_y - p_min_y
    dist_top = p_max_y - pt_y

    # Find minimum distance to edge
    min_dist_x = tl.minimum(dist_left, dist_right)
    min_dist_y = tl.minimum(dist_bottom, dist_top)

    # Project to closest edge if inside
    use_left = dist_left <= dist_right
    use_bottom = dist_bottom <= dist_top
    project_to_x = min_dist_x < min_dist_y

    closest_x_inside = tl.where(
        project_to_x,
        tl.where(use_left, p_min_x, p_max_x),
        pt_x
    )
    closest_y_inside = tl.where(
        project_to_x,
        pt_y,
        tl.where(use_bottom, p_min_y, p_max_y)
    )

    # If outside, closest point is clamped point
    closest_x = tl.where(inside, closest_x_inside, clamped_x)
    closest_y = tl.where(inside, closest_y_inside, clamped_y)

    # Distance squared
    diff_x = pt_x - closest_x
    diff_y = pt_y - closest_y
    dist_sq = diff_x * diff_x + diff_y * diff_y

    return closest_x, closest_y, dist_sq


@triton.jit
def compute_signed_distance_path_segment(
    pt_x, pt_y,
    segment_type,  # 0=line, 1=quadratic, 2=cubic
    # All control points (some may be unused)
    p0_x, p0_y,
    p1_x, p1_y,
    p2_x, p2_y,
    p3_x, p3_y,
):
    """
    Compute unsigned distance to a path segment.

    Returns: (closest_x, closest_y, t, distance_squared)
    """
    # Line segment
    line_cx, line_cy, line_t, line_dist_sq = closest_point_line(
        pt_x, pt_y, p0_x, p0_y, p1_x, p1_y
    )

    # Quadratic Bezier
    quad_cx, quad_cy, quad_t, quad_dist_sq = closest_point_quadratic_bezier(
        pt_x, pt_y, p0_x, p0_y, p1_x, p1_y, p2_x, p2_y
    )

    # Cubic Bezier
    cubic_cx, cubic_cy, cubic_t, cubic_dist_sq = closest_point_cubic_bezier(
        pt_x, pt_y, p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y
    )

    # Select based on segment type
    is_line = segment_type == 0
    is_quad = segment_type == 1

    closest_x = tl.where(is_line, line_cx, tl.where(is_quad, quad_cx, cubic_cx))
    closest_y = tl.where(is_line, line_cy, tl.where(is_quad, quad_cy, cubic_cy))
    t = tl.where(is_line, line_t, tl.where(is_quad, quad_t, cubic_t))
    dist_sq = tl.where(is_line, line_dist_sq, tl.where(is_quad, quad_dist_sq, cubic_dist_sq))

    return closest_x, closest_y, t, dist_sq


# Python reference implementations for testing
def closest_point_line_py(pt, p0, p1):
    """Python reference for closest point on line segment."""
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]

    len_sq = dx * dx + dy * dy
    if len_sq < 1e-10:
        return p0, 0.0, (pt[0] - p0[0])**2 + (pt[1] - p0[1])**2

    t = ((pt[0] - p0[0]) * dx + (pt[1] - p0[1]) * dy) / len_sq
    t = max(0.0, min(1.0, t))

    closest = (p0[0] + t * dx, p0[1] + t * dy)
    dist_sq = (pt[0] - closest[0])**2 + (pt[1] - closest[1])**2

    return closest, t, dist_sq


def eval_quadratic_bezier_py(t, p0, p1, p2):
    """Evaluate quadratic Bezier."""
    one_minus_t = 1.0 - t
    w0 = one_minus_t * one_minus_t
    w1 = 2.0 * one_minus_t * t
    w2 = t * t
    return (
        w0 * p0[0] + w1 * p1[0] + w2 * p2[0],
        w0 * p0[1] + w1 * p1[1] + w2 * p2[1]
    )


def eval_cubic_bezier_py(t, p0, p1, p2, p3):
    """Evaluate cubic Bezier."""
    one_minus_t = 1.0 - t
    one_minus_t_sq = one_minus_t * one_minus_t
    one_minus_t_cb = one_minus_t_sq * one_minus_t
    t_sq = t * t
    t_cb = t_sq * t

    w0 = one_minus_t_cb
    w1 = 3.0 * one_minus_t_sq * t
    w2 = 3.0 * one_minus_t * t_sq
    w3 = t_cb

    return (
        w0 * p0[0] + w1 * p1[0] + w2 * p2[0] + w3 * p3[0],
        w0 * p0[1] + w1 * p1[1] + w2 * p2[1] + w3 * p3[1]
    )


def _newton_refine_quadratic(pt, p0, p1, p2, t_init, num_iters=10):
    """Newton refinement for closest point on quadratic bezier."""
    t = t_init
    for _ in range(num_iters):
        # B(t) and B'(t)
        bpt = eval_quadratic_bezier_py(t, p0, p1, p2)
        # B'(t) = 2(1-t)(P1-P0) + 2t(P2-P1)
        one_minus_t = 1.0 - t
        dpx = 2.0 * (one_minus_t * (p1[0] - p0[0]) + t * (p2[0] - p1[0]))
        dpy = 2.0 * (one_minus_t * (p1[1] - p0[1]) + t * (p2[1] - p1[1]))
        # B''(t) = 2(P2 - 2P1 + P0)
        ddpx = 2.0 * (p2[0] - 2*p1[0] + p0[0])
        ddpy = 2.0 * (p2[1] - 2*p1[1] + p0[1])

        # f(t) = (B(t) - pt) · B'(t)
        dx = bpt[0] - pt[0]
        dy = bpt[1] - pt[1]
        f = dx * dpx + dy * dpy
        # f'(t) = B'(t) · B'(t) + (B(t) - pt) · B''(t)
        df = dpx * dpx + dpy * dpy + dx * ddpx + dy * ddpy

        if abs(df) < 1e-10:
            break
        t_new = t - f / df
        t = max(0.0, min(1.0, t_new))
    return t


def closest_point_quadratic_bezier_py(pt, p0, p1, p2, num_samples=16):
    """Python reference for closest point on quadratic Bezier with Newton refinement."""
    best_t = 0.0
    best_dist_sq = float('inf')

    # Coarse sampling to find initial guess
    for i in range(num_samples + 1):
        t = i / num_samples
        bpt = eval_quadratic_bezier_py(t, p0, p1, p2)
        dist_sq = (bpt[0] - pt[0])**2 + (bpt[1] - pt[1])**2
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_t = t

    # Newton refinement
    best_t = _newton_refine_quadratic(pt, p0, p1, p2, best_t)

    closest = eval_quadratic_bezier_py(best_t, p0, p1, p2)
    best_dist_sq = (closest[0] - pt[0])**2 + (closest[1] - pt[1])**2
    return closest, best_t, best_dist_sq


def _newton_refine_cubic(pt, p0, p1, p2, p3, t_init, num_iters=10):
    """Newton refinement for closest point on cubic bezier."""
    t = t_init
    for _ in range(num_iters):
        # B(t)
        bpt = eval_cubic_bezier_py(t, p0, p1, p2, p3)
        # B'(t) = 3(1-t)^2(P1-P0) + 6(1-t)t(P2-P1) + 3t^2(P3-P2)
        one_minus_t = 1.0 - t
        one_minus_t_sq = one_minus_t * one_minus_t
        t_sq = t * t

        d10 = (p1[0] - p0[0], p1[1] - p0[1])
        d21 = (p2[0] - p1[0], p2[1] - p1[1])
        d32 = (p3[0] - p2[0], p3[1] - p2[1])

        dpx = 3.0 * one_minus_t_sq * d10[0] + 6.0 * one_minus_t * t * d21[0] + 3.0 * t_sq * d32[0]
        dpy = 3.0 * one_minus_t_sq * d10[1] + 6.0 * one_minus_t * t * d21[1] + 3.0 * t_sq * d32[1]

        # B''(t) = 6(1-t)(P2-2P1+P0) + 6t(P3-2P2+P1)
        dd0 = (p2[0] - 2*p1[0] + p0[0], p2[1] - 2*p1[1] + p0[1])
        dd1 = (p3[0] - 2*p2[0] + p1[0], p3[1] - 2*p2[1] + p1[1])
        ddpx = 6.0 * one_minus_t * dd0[0] + 6.0 * t * dd1[0]
        ddpy = 6.0 * one_minus_t * dd0[1] + 6.0 * t * dd1[1]

        # f(t) = (B(t) - pt) · B'(t)
        dx = bpt[0] - pt[0]
        dy = bpt[1] - pt[1]
        f = dx * dpx + dy * dpy
        # f'(t) = B'(t) · B'(t) + (B(t) - pt) · B''(t)
        df = dpx * dpx + dpy * dpy + dx * ddpx + dy * ddpy

        if abs(df) < 1e-10:
            break
        t_new = t - f / df
        t = max(0.0, min(1.0, t_new))
    return t


def closest_point_cubic_bezier_py(pt, p0, p1, p2, p3, num_samples=16):
    """Python reference for closest point on cubic Bezier with Newton refinement."""
    best_t = 0.0
    best_dist_sq = float('inf')

    # Coarse sampling to find initial guess
    for i in range(num_samples + 1):
        t = i / num_samples
        bpt = eval_cubic_bezier_py(t, p0, p1, p2, p3)
        dist_sq = (bpt[0] - pt[0])**2 + (bpt[1] - pt[1])**2
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_t = t

    # Newton refinement
    best_t = _newton_refine_cubic(pt, p0, p1, p2, p3, best_t)

    closest = eval_cubic_bezier_py(best_t, p0, p1, p2, p3)
    best_dist_sq = (closest[0] - pt[0])**2 + (closest[1] - pt[1])**2
    return closest, best_t, best_dist_sq
