"""
Main rendering functions for diffvg-triton.

Implements vectorized GPU rendering pipeline:
1. Sample generation (multi-sample anti-aliasing)
2. Winding number computation (fills)
3. Distance computation (strokes)
4. Alpha compositing
"""

import torch
from typing import Tuple
from dataclasses import dataclass
from enum import IntEnum

from .scene import FlattenedScene, FlattenedPaths, ShapeType


class RenderMode(IntEnum):
    """Rendering mode selection."""
    HARD = 0       # Hard edges, multi-sample AA
    PREFILTER = 1  # Smooth edges using SDF


@dataclass
class RenderConfig:
    """Configuration for rendering."""
    num_samples_x: int = 2
    num_samples_y: int = 2
    seed: int = 42
    background_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)


def render(
    canvas_width: int,
    canvas_height: int,
    shapes: list,
    shape_groups: list,
    num_samples_x: int = 2,
    num_samples_y: int = 2,
    seed: int = 42,
    background_color: torch.Tensor = None,
    original_width: int = None,
    original_height: int = None,
) -> torch.Tensor:
    """
    Render shapes to an image.

    Args:
        canvas_width: Output image width
        canvas_height: Output image height
        shapes: List of shape objects (Path, Circle, etc.)
        shape_groups: List of ShapeGroup objects
        num_samples_x: Anti-aliasing samples in x
        num_samples_y: Anti-aliasing samples in y
        seed: Random seed
        background_color: [4] background RGBA, defaults to white
        original_width: Original viewBox width (for scaling)
        original_height: Original viewBox height (for scaling)

    Returns:
        [H, W, 4] RGBA image tensor
    """
    from .scene import flatten_scene
    import copy

    if background_color is None:
        background_color = torch.tensor([1.0, 1.0, 1.0, 1.0])

    # Handle scaling if output size differs from original viewBox size
    if original_width is None:
        original_width = canvas_width
    if original_height is None:
        original_height = canvas_height

    scale_x = canvas_width / original_width
    scale_y = canvas_height / original_height

    # Scale shapes if needed
    if scale_x != 1.0 or scale_y != 1.0:
        scaled_shapes = []
        for shape in shapes:
            new_shape = copy.copy(shape)
            scaled_points = shape.points.clone()
            scaled_points[:, 0] *= scale_x
            scaled_points[:, 1] *= scale_y
            new_shape.points = scaled_points
            if hasattr(shape, 'stroke_width') and shape.stroke_width is not None:
                avg_scale = (scale_x + scale_y) / 2
                new_shape.stroke_width = shape.stroke_width * avg_scale
            scaled_shapes.append(new_shape)
        shapes = scaled_shapes

    # Flatten scene
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scene = flatten_scene(canvas_width, canvas_height, shapes, shape_groups, device=device)

    # Configure renderer
    config = RenderConfig(
        num_samples_x=num_samples_x,
        num_samples_y=num_samples_y,
        seed=seed,
        background_color=tuple(background_color.cpu().tolist()),
    )

    return render_scene_vectorized(scene, config)


def _generate_sample_positions(
    width: int, height: int,
    num_samples_x: int, num_samples_y: int,
    device: torch.device
) -> torch.Tensor:
    """Generate stratified sample positions for all pixels."""
    py = torch.arange(height, device=device, dtype=torch.float32)
    px = torch.arange(width, device=device, dtype=torch.float32)
    sy = torch.arange(num_samples_y, device=device, dtype=torch.float32)
    sx = torch.arange(num_samples_x, device=device, dtype=torch.float32)

    ox = (sx + 0.5) / num_samples_x
    oy = (sy + 0.5) / num_samples_y

    py_grid, px_grid, oy_grid, ox_grid = torch.meshgrid(py, px, oy, ox, indexing='ij')

    sample_x = px_grid + ox_grid
    sample_y = py_grid + oy_grid

    num_samples = num_samples_x * num_samples_y
    sample_x = sample_x.reshape(height, width, num_samples)
    sample_y = sample_y.reshape(height, width, num_samples)

    return torch.stack([sample_x, sample_y], dim=-1)


def _compute_distance_quadratic(
    samples: torch.Tensor,
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    num_curve_samples: int = 65,
) -> torch.Tensor:
    """Compute distance from samples to quadratic bezier."""
    t = torch.linspace(0, 1, num_curve_samples, device=samples.device, dtype=samples.dtype)
    w0 = (1 - t) ** 2
    w1 = 2 * (1 - t) * t
    w2 = t ** 2
    curve_points = w0.unsqueeze(-1) * p0 + w1.unsqueeze(-1) * p1 + w2.unsqueeze(-1) * p2
    diff = samples.unsqueeze(1) - curve_points.unsqueeze(0)
    dist_sq = (diff ** 2).sum(dim=-1)
    return torch.sqrt(dist_sq.min(dim=1).values)


def _compute_distance_cubic(
    samples: torch.Tensor,
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    num_curve_samples: int = 65,
) -> torch.Tensor:
    """Compute distance from samples to cubic bezier."""
    t = torch.linspace(0, 1, num_curve_samples, device=samples.device, dtype=samples.dtype)
    one_minus_t = 1.0 - t
    w0 = one_minus_t ** 3
    w1 = 3.0 * (one_minus_t ** 2) * t
    w2 = 3.0 * one_minus_t * (t ** 2)
    w3 = t ** 3

    curve_points = (
        w0.unsqueeze(-1) * p0 +
        w1.unsqueeze(-1) * p1 +
        w2.unsqueeze(-1) * p2 +
        w3.unsqueeze(-1) * p3
    )

    diff = samples.unsqueeze(-2) - curve_points
    dist_sq = (diff ** 2).sum(dim=-1)
    return torch.sqrt(dist_sq.min(dim=-1).values)


def _compute_winding_line(
    samples: torch.Tensor,
    p0: torch.Tensor,
    p1: torch.Tensor,
) -> torch.Tensor:
    """Compute winding number contribution from line segment."""
    dy = p1[1] - p0[1]

    if abs(dy.item()) < 1e-10:
        return torch.zeros(samples.shape[0], device=samples.device, dtype=torch.int32)

    t = (samples[:, 1] - p0[1]) / dy
    x_int = p0[0] + t * (p1[0] - p0[0])
    valid = (t >= 0) & (t <= 1) & (x_int >= samples[:, 0])
    sign = torch.where(dy > 0, 1, -1)

    return torch.where(valid, sign, 0).to(torch.int32)


def _compute_winding_quadratic(
    samples: torch.Tensor,
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    num_subdivisions: int = 8,
) -> torch.Tensor:
    """Compute winding number for quadratic bezier using subdivision."""
    device = samples.device
    winding = torch.zeros(samples.shape[0], device=device, dtype=torch.int32)
    t_vals = torch.linspace(0, 1, num_subdivisions + 1, device=device)

    for i in range(num_subdivisions):
        t0, t1 = t_vals[i], t_vals[i + 1]
        w0_0, w1_0, w2_0 = (1 - t0) ** 2, 2 * (1 - t0) * t0, t0 ** 2
        w0_1, w1_1, w2_1 = (1 - t1) ** 2, 2 * (1 - t1) * t1, t1 ** 2

        seg_p0 = w0_0 * p0 + w1_0 * p1 + w2_0 * p2
        seg_p1 = w0_1 * p0 + w1_1 * p1 + w2_1 * p2
        winding = winding + _compute_winding_line(samples, seg_p0, seg_p1)

    return winding


def _compute_winding_cubic(
    samples: torch.Tensor,
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    num_subdivisions: int = 8,
) -> torch.Tensor:
    """Compute winding number for cubic bezier using subdivision."""
    device = samples.device
    winding = torch.zeros(samples.shape[0], device=device, dtype=torch.int32)
    t_vals = torch.linspace(0, 1, num_subdivisions + 1, device=device)

    for i in range(num_subdivisions):
        t0, t1 = t_vals[i], t_vals[i + 1]
        w0_0 = (1 - t0) ** 3
        w1_0 = 3 * (1 - t0) ** 2 * t0
        w2_0 = 3 * (1 - t0) * t0 ** 2
        w3_0 = t0 ** 3
        w0_1 = (1 - t1) ** 3
        w1_1 = 3 * (1 - t1) ** 2 * t1
        w2_1 = 3 * (1 - t1) * t1 ** 2
        w3_1 = t1 ** 3

        seg_p0 = w0_0 * p0 + w1_0 * p1 + w2_0 * p2 + w3_0 * p3
        seg_p1 = w0_1 * p0 + w1_1 * p1 + w2_1 * p2 + w3_1 * p3
        winding = winding + _compute_winding_line(samples, seg_p0, seg_p1)

    return winding


def render_scene_vectorized(
    scene: FlattenedScene,
    config: RenderConfig = None,
) -> torch.Tensor:
    """
    Render a scene using vectorized PyTorch operations.

    Supports fills (via winding number) and strokes (via distance).

    Args:
        scene: Flattened scene data
        config: Render configuration

    Returns:
        [H, W, 4] RGBA image tensor
    """
    if config is None:
        config = RenderConfig()

    width = scene.canvas_width
    height = scene.canvas_height
    device = scene.device

    num_samples_x = config.num_samples_x
    num_samples_y = config.num_samples_y
    num_samples = num_samples_x * num_samples_y

    # Generate sample positions: [H, W, num_samples, 2]
    sample_pos = _generate_sample_positions(
        width, height, num_samples_x, num_samples_y, device
    )

    # Initialize with background color
    bg = torch.tensor(config.background_color, device=device, dtype=torch.float32)

    if scene.paths is None:
        return bg.view(1, 1, 4).expand(height, width, 4).clone()

    paths = scene.paths
    groups = scene.groups

    # Flatten sample positions: [H*W*num_samples, 2]
    flat_samples = sample_pos.reshape(-1, 2)
    N = flat_samples.shape[0]

    # Initialize sample colors with background
    sample_colors = bg.view(1, 4).expand(N, 4).clone()

    for group_idx in range(groups.num_groups):
        if not groups.shape_mask[group_idx, 0].item():
            continue

        has_fill = groups.has_fill[group_idx].item()
        has_stroke = groups.has_stroke[group_idx].item()
        use_even_odd = groups.use_even_odd_rule[group_idx].item()

        fill_color = None
        if has_fill and groups.fill_color is not None:
            fill_color = groups.fill_color[group_idx]

        stroke_color = None
        if has_stroke and groups.stroke_color is not None:
            stroke_color = groups.stroke_color[group_idx]

        num_shapes = groups.num_shapes[group_idx].item()
        shape_ids = groups.shape_ids[group_idx, :num_shapes]

        for i in range(num_shapes):
            shape_id = shape_ids[i].item()
            shape_type = scene.shape_types[shape_id].item()
            shape_idx = scene.shape_indices[shape_id].item()

            if shape_type != ShapeType.PATH:
                continue

            point_offset = paths.point_offsets[shape_idx].item()
            num_points = paths.num_points[shape_idx].item()
            num_segments = paths.num_segments[shape_idx].item()
            seg_types = paths.segment_types[shape_idx, :num_segments]
            is_closed = paths.is_closed[shape_idx].item()

            # Helper to get point with wrapping for closed paths
            def get_point(idx):
                local_idx = idx - point_offset
                if is_closed and local_idx >= num_points:
                    local_idx = local_idx % num_points
                return paths.points[point_offset + local_idx]

            # === FILL ===
            if has_fill and fill_color is not None:
                winding = torch.zeros(N, device=device, dtype=torch.int32)
                current_point = point_offset

                for seg_idx in range(num_segments):
                    seg_type = seg_types[seg_idx].item()

                    if seg_type == 2:  # Cubic
                        p0 = get_point(current_point)
                        p1 = get_point(current_point + 1)
                        p2 = get_point(current_point + 2)
                        p3 = get_point(current_point + 3)
                        winding = winding + _compute_winding_cubic(flat_samples, p0, p1, p2, p3)
                        current_point += 3
                    elif seg_type == 1:  # Quadratic
                        p0 = get_point(current_point)
                        p1 = get_point(current_point + 1)
                        p2 = get_point(current_point + 2)
                        winding = winding + _compute_winding_quadratic(flat_samples, p0, p1, p2)
                        current_point += 2
                    else:  # Line
                        p0 = get_point(current_point)
                        p1 = get_point(current_point + 1)
                        winding = winding + _compute_winding_line(flat_samples, p0, p1)
                        current_point += 1

                if use_even_odd:
                    inside = (winding % 2) != 0
                else:
                    inside = winding != 0

                alpha = fill_color[3]
                sample_colors = torch.where(
                    inside.unsqueeze(-1),
                    fill_color * alpha + sample_colors * (1 - alpha),
                    sample_colors
                )

            # === STROKE ===
            if has_stroke and stroke_color is not None:
                stroke_width = paths.stroke_width[shape_idx].item()
                if stroke_width > 0:
                    half_width = stroke_width / 2.0
                    min_dist = torch.full((N,), float('inf'), device=device)
                    current_point = point_offset

                    for seg_idx in range(num_segments):
                        seg_type = seg_types[seg_idx].item()

                        if seg_type == 2:  # Cubic
                            p0 = get_point(current_point)
                            p1 = get_point(current_point + 1)
                            p2 = get_point(current_point + 2)
                            p3 = get_point(current_point + 3)
                            dist = _compute_distance_cubic(flat_samples, p0, p1, p2, p3)
                            min_dist = torch.minimum(min_dist, dist)
                            current_point += 3
                        elif seg_type == 1:  # Quadratic
                            p0 = get_point(current_point)
                            p1 = get_point(current_point + 1)
                            p2 = get_point(current_point + 2)
                            dist = _compute_distance_quadratic(flat_samples, p0, p1, p2)
                            min_dist = torch.minimum(min_dist, dist)
                            current_point += 2
                        else:  # Line
                            p0 = get_point(current_point)
                            p1 = get_point(current_point + 1)
                            d = p1 - p0
                            len_sq = (d ** 2).sum()
                            if len_sq > 1e-10:
                                v = flat_samples - p0
                                t = torch.clamp((v * d).sum(dim=-1) / len_sq, 0, 1)
                                closest = p0 + t.unsqueeze(-1) * d
                                dist = torch.sqrt(((flat_samples - closest) ** 2).sum(dim=-1))
                            else:
                                dist = torch.sqrt(((flat_samples - p0) ** 2).sum(dim=-1))
                            min_dist = torch.minimum(min_dist, dist)
                            current_point += 1

                    inside_stroke = min_dist <= half_width
                    alpha = stroke_color[3]
                    sample_colors = torch.where(
                        inside_stroke.unsqueeze(-1),
                        stroke_color * alpha + sample_colors * (1 - alpha),
                        sample_colors
                    )

    # Average samples per pixel
    sample_colors = sample_colors.reshape(height, width, num_samples, 4)
    output = sample_colors.mean(dim=2)

    return output
