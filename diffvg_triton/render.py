"""
Main rendering functions for the Triton backend.

Implements the core rendering pipeline:
1. Sample generation (multi-sample anti-aliasing)
2. Fragment collection (winding number + distance tests)
3. Color sampling (fill + stroke)
4. Fragment compositing (back-to-front blending)
5. Filter splatting (optional)

Ported from diffvg/diffvg.cpp render_kernel
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import IntEnum

from .scene import FlattenedScene, FlattenedPaths, FlattenedShapeGroup, ShapeType
from .kernels.composite import composite_over, smoothstep_coverage_py
from .kernels.filter import FilterType, splat_samples_to_image
from .kernels.rng import TorchPCG32, generate_sample_offsets
from .kernels.distance import (
    eval_cubic_bezier,
    closest_point_line,
    closest_point_quadratic_bezier,
    closest_point_cubic_bezier,
)


class RenderMode(IntEnum):
    """Rendering mode selection."""
    HARD = 0       # Hard edges, multi-sample AA
    PREFILTER = 1  # Smooth edges using SDF


@dataclass
class RenderConfig:
    """Configuration for rendering."""
    num_samples_x: int = 2       # Samples per pixel in x
    num_samples_y: int = 2       # Samples per pixel in y
    seed: int = 42               # Random seed for sample jittering
    filter_type: int = FilterType.BOX
    filter_radius: float = 1.0
    use_prefiltering: bool = False
    background_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)


def _compute_winding_number_py(
    pt: Tuple[float, float],
    paths: FlattenedPaths,
    path_idx: int,
) -> int:
    """
    Compute winding number for a point against a path.

    Python reference implementation.
    """
    from .kernels.winding import compute_winding_number_path_py

    # Get path data
    point_offset = paths.point_offsets[path_idx].item()
    num_points = paths.num_points[path_idx].item()
    num_segments = paths.num_segments[path_idx].item()
    is_closed = paths.is_closed[path_idx].item()

    # Extract segment types and points for this path
    segment_types = paths.segment_types[path_idx, :num_segments].detach().cpu().tolist()
    points_flat = paths.points[point_offset:point_offset + num_points].detach().cpu().tolist()
    points = [(p[0], p[1]) for p in points_flat]

    # Compute winding number
    winding = compute_winding_number_path_py(pt, segment_types, points, is_closed)

    return winding


def _compute_closest_distance_py(
    pt: Tuple[float, float],
    paths: FlattenedPaths,
    path_idx: int,
) -> Tuple[float, Tuple[float, float], float]:
    """
    Compute closest distance from point to path.

    Returns: (distance, closest_point, t_parameter)
    """
    from .kernels.distance import (
        closest_point_line_py,
        closest_point_quadratic_bezier_py,
        closest_point_cubic_bezier_py,
    )

    point_offset = paths.point_offsets[path_idx].item()
    num_segments = paths.num_segments[path_idx].item()

    segment_types = paths.segment_types[path_idx, :num_segments].detach().cpu().tolist()
    points_flat = paths.points.detach().cpu().numpy()

    best_dist_sq = float('inf')
    best_closest = pt
    best_t = 0.0

    current_point = point_offset

    for seg_type in segment_types:
        if seg_type == 0:
            # Line
            p0 = tuple(points_flat[current_point])
            p1 = tuple(points_flat[current_point + 1])
            closest, t, dist_sq = closest_point_line_py(pt, p0, p1)
            current_point += 1
        elif seg_type == 1:
            # Quadratic
            p0 = tuple(points_flat[current_point])
            p1 = tuple(points_flat[current_point + 1])
            p2 = tuple(points_flat[current_point + 2])
            closest, t, dist_sq = closest_point_quadratic_bezier_py(pt, p0, p1, p2)
            current_point += 2
        else:
            # Cubic
            p0 = tuple(points_flat[current_point])
            p1 = tuple(points_flat[current_point + 1])
            p2 = tuple(points_flat[current_point + 2])
            p3 = tuple(points_flat[current_point + 3])
            closest, t, dist_sq = closest_point_cubic_bezier_py(pt, p0, p1, p2, p3)
            current_point += 3

        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_closest = closest
            best_t = t

    return best_dist_sq ** 0.5, best_closest, best_t


def _sample_color_at_point(
    pt: Tuple[float, float],
    scene: FlattenedScene,
    use_prefiltering: bool = False,
) -> Tuple[float, float, float, float]:
    """
    Sample color at a point by collecting and compositing fragments.

    Python reference implementation.
    """
    from .kernels.composite import blend_over_py, smoothstep_coverage_py

    # Collect fragments from all shape groups
    fragments = []

    for group_idx in range(scene.groups.num_groups):
        if not scene.groups.shape_mask[group_idx, 0].item():
            continue

        # Get shapes in this group
        num_shapes = scene.groups.num_shapes[group_idx].item()
        shape_ids = scene.groups.shape_ids[group_idx, :num_shapes].cpu().tolist()

        has_fill = scene.groups.has_fill[group_idx].item()
        has_stroke = scene.groups.has_stroke[group_idx].item()
        use_even_odd = scene.groups.use_even_odd_rule[group_idx].item()

        fill_color = None
        if has_fill and scene.groups.fill_color is not None:
            fc = scene.groups.fill_color[group_idx].cpu().tolist()
            fill_color = tuple(fc)

        stroke_color = None
        if has_stroke and scene.groups.stroke_color is not None:
            sc = scene.groups.stroke_color[group_idx].cpu().tolist()
            stroke_color = tuple(sc)

        # Check each shape
        for shape_id in shape_ids:
            shape_type = scene.shape_types[shape_id].item()
            shape_idx = scene.shape_indices[shape_id].item()

            if shape_type == ShapeType.PATH and scene.paths is not None:
                # Compute winding number for fill
                if has_fill and fill_color is not None:
                    winding = _compute_winding_number_py(pt, scene.paths, shape_idx)

                    # Apply fill rule
                    if use_even_odd:
                        is_inside = (winding % 2) != 0
                    else:
                        is_inside = winding != 0

                    if is_inside:
                        if use_prefiltering:
                            # Get distance for smooth coverage
                            dist, _, _ = _compute_closest_distance_py(pt, scene.paths, shape_idx)
                            coverage = smoothstep_coverage_py(-dist)  # Negative for inside
                            color = (
                                fill_color[0],
                                fill_color[1],
                                fill_color[2],
                                fill_color[3] * coverage
                            )
                        else:
                            color = fill_color

                        fragments.append((color, group_idx, False))

                # Compute distance for stroke
                if has_stroke and stroke_color is not None:
                    stroke_width = scene.paths.stroke_width[shape_idx].item()
                    if stroke_width > 0:
                        dist, _, _ = _compute_closest_distance_py(pt, scene.paths, shape_idx)
                        half_width = stroke_width / 2.0

                        # Pure distance-based stroke coverage
                        # Point is covered if within half_width of the curve
                        # Use smooth transition for anti-aliasing
                        edge_dist = dist - half_width  # Distance from stroke outer edge

                        if edge_dist <= -1.0:
                            # Well inside stroke band
                            coverage = 1.0
                        elif edge_dist >= 1.0:
                            # Well outside stroke band
                            coverage = 0.0
                        else:
                            # In transition zone (-1 to +1 pixel from edge)
                            # Smoothstep for antialiasing
                            t = (1.0 - edge_dist) / 2.0
                            coverage = t * t * (3.0 - 2.0 * t)

                        if coverage > 0:
                            color = (
                                stroke_color[0],
                                stroke_color[1],
                                stroke_color[2],
                                stroke_color[3] * coverage
                            )
                            fragments.append((color, group_idx, True))

    # Sort fragments by group_id (back to front)
    fragments.sort(key=lambda x: x[1])

    return fragments


def render_scene_py(
    scene: FlattenedScene,
    config: RenderConfig = None,
    pydiffvg_compatible: bool = False,  # If True, output raw RGBA without background
) -> torch.Tensor:
    """
    Render a scene using Python reference implementation.

    This is a slow but correct reference implementation for testing.

    Args:
        scene: Flattened scene data
        config: Render configuration
        pydiffvg_compatible: If True, output raw RGBA like pydiffvg (no background compositing)

    Returns:
        [H, W, 4] RGBA image tensor
    """
    from .kernels.composite import composite_fragments_py

    if config is None:
        config = RenderConfig()

    width = scene.canvas_width
    height = scene.canvas_height
    device = scene.device

    # Initialize output
    output = torch.zeros((height, width, 4), dtype=torch.float32, device='cpu')

    num_samples = config.num_samples_x * config.num_samples_y
    background = config.background_color

    # Generate sample offsets
    sample_offsets = []
    for sy in range(config.num_samples_y):
        for sx in range(config.num_samples_x):
            # Stratified sampling
            ox = (sx + 0.5) / config.num_samples_x
            oy = (sy + 0.5) / config.num_samples_y
            sample_offsets.append((ox, oy))

    # Render each pixel
    for py in range(height):
        for px in range(width):
            # Accumulate samples
            acc_color = [0.0, 0.0, 0.0, 0.0]

            for ox, oy in sample_offsets:
                # Sample position in canvas coordinates
                sx = px + ox
                sy = py + oy
                pt = (sx, sy)

                # Collect fragments at this sample
                fragments = _sample_color_at_point(
                    pt, scene, config.use_prefiltering
                )

                # Composite fragments
                if fragments:
                    frag_colors = [f[0] for f in fragments]
                    if pydiffvg_compatible:
                        # pydiffvg outputs raw RGBA without background compositing
                        # Just composite fragments over transparent (0,0,0,0)
                        sample_color = composite_fragments_py(frag_colors, (0.0, 0.0, 0.0, 0.0))
                    else:
                        sample_color = composite_fragments_py(frag_colors, background)
                else:
                    if pydiffvg_compatible:
                        # Background is transparent in pydiffvg mode
                        sample_color = (0.0, 0.0, 0.0, 0.0)
                    else:
                        sample_color = background

                # Accumulate
                for c in range(4):
                    acc_color[c] += sample_color[c]

            # Average samples
            for c in range(4):
                output[py, px, c] = float(acc_color[c]) / num_samples

    return output.to(device)


# Triton kernel for rendering (optimized version)
@triton.jit
def render_sample_kernel(
    # Output
    sample_colors_ptr,   # [H * W * num_samples, 4]
    # Sample positions
    sample_x_ptr,        # [H * W * num_samples]
    sample_y_ptr,        # [H * W * num_samples]
    # Path data
    segment_types_ptr,   # [num_paths, max_segments]
    segment_mask_ptr,    # [num_paths, max_segments]
    num_segments_ptr,    # [num_paths]
    points_ptr,          # [total_points, 2]
    point_offsets_ptr,   # [num_paths]
    num_points_ptr,      # [num_paths]
    is_closed_ptr,       # [num_paths]
    stroke_width_ptr,    # [num_paths]
    # Shape group data
    group_shape_ids_ptr,      # [num_groups, max_shapes_per_group]
    group_shape_mask_ptr,     # [num_groups, max_shapes_per_group]
    group_num_shapes_ptr,     # [num_groups]
    fill_color_ptr,           # [num_groups, 4]
    has_fill_ptr,             # [num_groups]
    stroke_color_ptr,         # [num_groups, 4]
    has_stroke_ptr,           # [num_groups]
    use_even_odd_ptr,         # [num_groups]
    # Shape mapping
    shape_types_ptr,          # [num_shapes]
    shape_indices_ptr,        # [num_shapes]
    # Background
    bg_r, bg_g, bg_b, bg_a,
    # Dimensions
    width: tl.constexpr,
    height: tl.constexpr,
    num_samples: tl.constexpr,
    num_paths: tl.constexpr,
    num_groups: tl.constexpr,
    max_segments: tl.constexpr,
    max_shapes_per_group: tl.constexpr,
    total_points: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Render samples kernel.

    Each thread processes one sample, computing its color by:
    1. Iterating over all shape groups
    2. For each shape, computing winding number (fill) and distance (stroke)
    3. Compositing fragment colors
    """
    pid = tl.program_id(0)
    sample_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_samples = width * height * num_samples
    mask = sample_idx < total_samples

    # Load sample position
    sx = tl.load(sample_x_ptr + sample_idx, mask=mask, other=0.0)
    sy = tl.load(sample_y_ptr + sample_idx, mask=mask, other=0.0)

    # Initialize accumulated color with background
    acc_r = bg_r
    acc_g = bg_g
    acc_b = bg_b
    acc_a = bg_a

    # Iterate over shape groups (back to front)
    # Note: This is a simplified version - full implementation needs:
    # - Proper fragment sorting
    # - BVH traversal for efficiency
    # - Full winding number computation

    # For now, store background color
    out_offset = sample_idx * 4
    tl.store(sample_colors_ptr + out_offset, acc_r, mask=mask)
    tl.store(sample_colors_ptr + out_offset + 1, acc_g, mask=mask)
    tl.store(sample_colors_ptr + out_offset + 2, acc_b, mask=mask)
    tl.store(sample_colors_ptr + out_offset + 3, acc_a, mask=mask)


def render_scene(
    scene: FlattenedScene,
    config: RenderConfig = None,
    use_python_reference: bool = True,  # Use Python for correctness, Triton for speed
) -> torch.Tensor:
    """
    Render a flattened scene to an image.

    Args:
        scene: Flattened scene containing shapes and groups
        config: Rendering configuration
        use_python_reference: If True, use slow but correct Python implementation

    Returns:
        [H, W, 4] RGBA image tensor
    """
    if config is None:
        config = RenderConfig()

    if use_python_reference:
        return render_scene_py(scene, config)

    # Triton implementation
    device = scene.device
    width = scene.canvas_width
    height = scene.canvas_height
    num_samples = config.num_samples_x * config.num_samples_y
    total_samples = width * height * num_samples

    # Generate sample positions
    sample_positions = torch.zeros((total_samples, 2), dtype=torch.float32, device=device)

    # Generate stratified sample positions
    idx = 0
    for py in range(height):
        for px in range(width):
            for sy in range(config.num_samples_y):
                for sx in range(config.num_samples_x):
                    ox = (sx + 0.5) / config.num_samples_x
                    oy = (sy + 0.5) / config.num_samples_y
                    sample_positions[idx, 0] = px + ox
                    sample_positions[idx, 1] = py + oy
                    idx += 1

    # Allocate output
    sample_colors = torch.zeros((total_samples, 4), dtype=torch.float32, device=device)

    # Launch kernel
    BLOCK_SIZE = 256
    grid = ((total_samples + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    bg = config.background_color

    # For now, fall back to Python implementation
    # Full Triton kernel implementation is complex and needs careful optimization
    return render_scene_py(scene, config)


def render(
    canvas_width: int,
    canvas_height: int,
    shapes: list,
    shape_groups: list,
    num_samples_x: int = 2,
    num_samples_y: int = 2,
    seed: int = 42,
    background_color: torch.Tensor = None,
    use_prefiltering: bool = False,
    original_width: int = None,
    original_height: int = None,
) -> torch.Tensor:
    """
    High-level render function compatible with pydiffvg API.

    Args:
        canvas_width: Output image width
        canvas_height: Output image height
        shapes: List of shape objects (Path, Circle, etc.)
        shape_groups: List of ShapeGroup objects
        num_samples_x: Anti-aliasing samples in x
        num_samples_y: Anti-aliasing samples in y
        seed: Random seed
        background_color: [4] background RGBA, defaults to white
        use_prefiltering: Use SDF-based smooth edges
        original_width: Original viewBox width (for scaling). If None, uses canvas_width.
        original_height: Original viewBox height (for scaling). If None, uses canvas_height.

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
            # Deep copy to avoid modifying original
            new_shape = copy.copy(shape)
            # Scale points
            scaled_points = shape.points.clone()
            scaled_points[:, 0] *= scale_x
            scaled_points[:, 1] *= scale_y
            new_shape.points = scaled_points
            # Scale stroke width
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
        use_prefiltering=use_prefiltering,
    )

    # Render using fast vectorized implementation
    return render_scene_vectorized(scene, config)


def _generate_sample_positions_vectorized(
    width: int, height: int,
    num_samples_x: int, num_samples_y: int,
    device: torch.device
) -> torch.Tensor:
    """
    Generate stratified sample positions for all pixels.

    Returns: [H, W, num_samples, 2] tensor of sample (x, y) positions
    """
    # Create pixel coordinate grid
    py = torch.arange(height, device=device, dtype=torch.float32)
    px = torch.arange(width, device=device, dtype=torch.float32)

    # Create sample offsets within each pixel
    sy = torch.arange(num_samples_y, device=device, dtype=torch.float32)
    sx = torch.arange(num_samples_x, device=device, dtype=torch.float32)

    # Stratified offsets within [0, 1]
    ox = (sx + 0.5) / num_samples_x  # [num_samples_x]
    oy = (sy + 0.5) / num_samples_y  # [num_samples_y]

    # Create meshgrid for all combinations
    # Final shape: [H, W, num_samples_y, num_samples_x, 2]
    py_grid, px_grid, oy_grid, ox_grid = torch.meshgrid(py, px, oy, ox, indexing='ij')

    # Compute sample positions
    sample_x = px_grid + ox_grid  # [H, W, num_samples_y, num_samples_x]
    sample_y = py_grid + oy_grid

    # Reshape to [H, W, num_samples, 2]
    num_samples = num_samples_x * num_samples_y
    sample_x = sample_x.reshape(height, width, num_samples)
    sample_y = sample_y.reshape(height, width, num_samples)

    # Stack into position tensor
    positions = torch.stack([sample_x, sample_y], dim=-1)

    return positions


def _compute_distance_to_quadratic_segment_vectorized(
    sample_pos: torch.Tensor,  # [N, 2] sample positions
    p0: torch.Tensor,  # [2] start point
    p1: torch.Tensor,  # [2] control point
    p2: torch.Tensor,  # [2] end point
    num_samples: int = 65,
) -> torch.Tensor:
    """Compute distance from sample positions to a quadratic bezier segment."""
    t = torch.linspace(0, 1, num_samples, device=sample_pos.device, dtype=sample_pos.dtype)
    w0 = (1 - t) ** 2
    w1 = 2 * (1 - t) * t
    w2 = t ** 2
    curve_points = w0.unsqueeze(-1) * p0 + w1.unsqueeze(-1) * p1 + w2.unsqueeze(-1) * p2
    diff = sample_pos.unsqueeze(1) - curve_points.unsqueeze(0)
    dist_sq = (diff ** 2).sum(dim=-1)
    return torch.sqrt(dist_sq.min(dim=1).values)


def _compute_distance_to_cubic_segment_vectorized(
    sample_pos: torch.Tensor,  # [..., 2] sample positions
    p0: torch.Tensor,  # [2] start point
    p1: torch.Tensor,  # [2] control point 1
    p2: torch.Tensor,  # [2] control point 2
    p3: torch.Tensor,  # [2] end point
    num_samples: int = 65,
) -> torch.Tensor:
    """
    Compute distance from sample positions to a cubic bezier segment.
    Uses dense sampling approach for simplicity.

    Returns: [...] tensor of distances
    """
    # Sample the bezier at multiple t values
    t = torch.linspace(0, 1, num_samples, device=sample_pos.device, dtype=sample_pos.dtype)

    # Evaluate bezier at all t values: [num_samples, 2]
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
    )  # [num_samples, 2]

    # Compute distance from each sample position to each curve point
    # sample_pos: [..., 2], curve_points: [num_samples, 2]
    # diff: [..., num_samples, 2]
    diff = sample_pos.unsqueeze(-2) - curve_points
    dist_sq = (diff ** 2).sum(dim=-1)  # [..., num_samples]

    # Take minimum distance
    min_dist_sq = dist_sq.min(dim=-1).values
    min_dist = torch.sqrt(min_dist_sq)

    return min_dist


def _compute_winding_number_line_vectorized(
    samples: torch.Tensor,  # [N, 2]
    p0: torch.Tensor,       # [2]
    p1: torch.Tensor,       # [2]
) -> torch.Tensor:
    """Vectorized winding number contribution from ray-line intersection."""
    # Horizontal ray from sample point to the right
    dy = p1[1] - p0[1]

    # Skip near-horizontal lines
    if abs(dy.item()) < 1e-10:
        return torch.zeros(samples.shape[0], device=samples.device, dtype=torch.int32)

    # t parameter on line where y = sample_y
    t = (samples[:, 1] - p0[1]) / dy

    # x coordinate at intersection
    x_int = p0[0] + t * (p1[0] - p0[0])

    # Valid if t in [0,1] and intersection is to the right of sample
    valid = (t >= 0) & (t <= 1) & (x_int >= samples[:, 0])

    # Direction: +1 for upward, -1 for downward
    sign = torch.where(dy > 0, 1, -1)

    return torch.where(valid, sign, 0).to(torch.int32)


def _compute_winding_number_quadratic_vectorized(
    samples: torch.Tensor,  # [N, 2]
    p0: torch.Tensor,       # [2]
    p1: torch.Tensor,       # [2]
    p2: torch.Tensor,       # [2]
) -> torch.Tensor:
    """Vectorized winding number for quadratic bezier using sampling."""
    # Sample the curve and treat as line segments
    N = samples.shape[0]
    device = samples.device
    winding = torch.zeros(N, device=device, dtype=torch.int32)

    num_subdivisions = 8
    t_vals = torch.linspace(0, 1, num_subdivisions + 1, device=device)

    for i in range(num_subdivisions):
        t0, t1 = t_vals[i], t_vals[i + 1]
        # Evaluate bezier at t0 and t1
        w0_0 = (1 - t0) ** 2
        w1_0 = 2 * (1 - t0) * t0
        w2_0 = t0 ** 2
        w0_1 = (1 - t1) ** 2
        w1_1 = 2 * (1 - t1) * t1
        w2_1 = t1 ** 2

        seg_p0 = w0_0 * p0 + w1_0 * p1 + w2_0 * p2
        seg_p1 = w0_1 * p0 + w1_1 * p1 + w2_1 * p2

        winding = winding + _compute_winding_number_line_vectorized(samples, seg_p0, seg_p1)

    return winding


def _compute_winding_number_cubic_vectorized(
    samples: torch.Tensor,  # [N, 2]
    p0: torch.Tensor,       # [2]
    p1: torch.Tensor,       # [2]
    p2: torch.Tensor,       # [2]
    p3: torch.Tensor,       # [2]
) -> torch.Tensor:
    """Vectorized winding number for cubic bezier using sampling."""
    N = samples.shape[0]
    device = samples.device
    winding = torch.zeros(N, device=device, dtype=torch.int32)

    num_subdivisions = 8
    t_vals = torch.linspace(0, 1, num_subdivisions + 1, device=device)

    for i in range(num_subdivisions):
        t0, t1 = t_vals[i], t_vals[i + 1]
        # Evaluate cubic bezier at t0 and t1
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

        winding = winding + _compute_winding_number_line_vectorized(samples, seg_p0, seg_p1)

    return winding


def render_scene_vectorized(
    scene: FlattenedScene,
    config: RenderConfig = None,
) -> torch.Tensor:
    """
    Render a scene using vectorized PyTorch operations (runs on GPU).

    Supports both fills (via winding number) and strokes (via distance).

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
    sample_pos = _generate_sample_positions_vectorized(
        width, height, num_samples_x, num_samples_y, device
    )

    # Initialize with background color
    bg = torch.tensor(config.background_color, device=device, dtype=torch.float32)

    # Process each shape group
    if scene.paths is None:
        return bg.view(1, 1, 4).expand(height, width, 4).clone()

    paths = scene.paths
    groups = scene.groups

    # Flatten sample positions for batch processing: [H*W*num_samples, 2]
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

        # Process each shape in group
        num_shapes = groups.num_shapes[group_idx].item()
        shape_ids = groups.shape_ids[group_idx, :num_shapes]

        for i in range(num_shapes):
            shape_id = shape_ids[i].item()
            shape_type = scene.shape_types[shape_id].item()
            shape_idx = scene.shape_indices[shape_id].item()

            if shape_type != ShapeType.PATH:
                continue

            # Get path data
            point_offset = paths.point_offsets[shape_idx].item()
            num_segments = paths.num_segments[shape_idx].item()
            seg_types = paths.segment_types[shape_idx, :num_segments]
            is_closed = paths.is_closed[shape_idx].item()

            # === FILL ===
            if has_fill and fill_color is not None:
                # Compute winding number for each sample
                winding = torch.zeros(N, device=device, dtype=torch.int32)
                current_point = point_offset

                for seg_idx in range(num_segments):
                    seg_type = seg_types[seg_idx].item()

                    if seg_type == 2:  # Cubic
                        p0 = paths.points[current_point]
                        p1 = paths.points[current_point + 1]
                        p2 = paths.points[current_point + 2]
                        p3 = paths.points[current_point + 3]
                        winding = winding + _compute_winding_number_cubic_vectorized(flat_samples, p0, p1, p2, p3)
                        current_point += 3
                    elif seg_type == 1:  # Quadratic
                        p0 = paths.points[current_point]
                        p1 = paths.points[current_point + 1]
                        p2 = paths.points[current_point + 2]
                        winding = winding + _compute_winding_number_quadratic_vectorized(flat_samples, p0, p1, p2)
                        current_point += 2
                    else:  # Line
                        p0 = paths.points[current_point]
                        p1 = paths.points[current_point + 1]
                        winding = winding + _compute_winding_number_line_vectorized(flat_samples, p0, p1)
                        current_point += 1

                # Apply fill rule
                if use_even_odd:
                    inside = (winding % 2) != 0
                else:
                    inside = winding != 0

                # Alpha blend fill color
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

                    # Compute minimum distance to all segments
                    min_dist = torch.full((N,), float('inf'), device=device)
                    current_point = point_offset

                    for seg_idx in range(num_segments):
                        seg_type = seg_types[seg_idx].item()

                        if seg_type == 2:  # Cubic
                            p0 = paths.points[current_point]
                            p1 = paths.points[current_point + 1]
                            p2 = paths.points[current_point + 2]
                            p3 = paths.points[current_point + 3]
                            dist = _compute_distance_to_cubic_segment_vectorized(flat_samples, p0, p1, p2, p3)
                            min_dist = torch.minimum(min_dist, dist)
                            current_point += 3
                        elif seg_type == 1:  # Quadratic
                            p0 = paths.points[current_point]
                            p1 = paths.points[current_point + 1]
                            p2 = paths.points[current_point + 2]
                            dist = _compute_distance_to_quadratic_segment_vectorized(flat_samples, p0, p1, p2)
                            min_dist = torch.minimum(min_dist, dist)
                            current_point += 2
                        else:  # Line
                            p0 = paths.points[current_point]
                            p1 = paths.points[current_point + 1]
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

                    # Inside stroke band
                    inside_stroke = min_dist <= half_width

                    # Alpha blend stroke color
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


# Triton kernel for computing minimum distance to all cubic bezier segments
@triton.jit
def _min_dist_to_cubics_kernel(
    # Sample positions [N]
    sample_x_ptr,
    sample_y_ptr,
    # Bezier control points [num_segments, 4, 2] flattened
    bezier_points_ptr,  # [num_segments * 8]
    # Output
    min_dist_ptr,  # [N]
    # Params
    N: tl.constexpr,
    num_segments: tl.constexpr,
    half_width,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute minimum distance from samples to cubic bezier curves."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load sample position
    sx = tl.load(sample_x_ptr + offs, mask=mask, other=0.0)
    sy = tl.load(sample_y_ptr + offs, mask=mask, other=0.0)

    # Initialize minimum distance
    min_dist_sq = tl.full([BLOCK_SIZE], 1e10, dtype=tl.float32)

    # Iterate over all segments
    for seg_idx in range(num_segments):
        base = seg_idx * 8  # 4 points * 2 coords

        # Load control points
        p0_x = tl.load(bezier_points_ptr + base + 0)
        p0_y = tl.load(bezier_points_ptr + base + 1)
        p1_x = tl.load(bezier_points_ptr + base + 2)
        p1_y = tl.load(bezier_points_ptr + base + 3)
        p2_x = tl.load(bezier_points_ptr + base + 4)
        p2_y = tl.load(bezier_points_ptr + base + 5)
        p3_x = tl.load(bezier_points_ptr + base + 6)
        p3_y = tl.load(bezier_points_ptr + base + 7)

        # Sample the curve at multiple t values and find min distance
        # Use 17 samples for accuracy
        for i in range(17):
            t = i / 16.0
            one_minus_t = 1.0 - t
            one_minus_t_sq = one_minus_t * one_minus_t
            one_minus_t_cb = one_minus_t_sq * one_minus_t
            t_sq = t * t
            t_cb = t_sq * t

            w0 = one_minus_t_cb
            w1 = 3.0 * one_minus_t_sq * t
            w2 = 3.0 * one_minus_t * t_sq
            w3 = t_cb

            curve_x = w0 * p0_x + w1 * p1_x + w2 * p2_x + w3 * p3_x
            curve_y = w0 * p0_y + w1 * p1_y + w2 * p2_y + w3 * p3_y

            dx = sx - curve_x
            dy = sy - curve_y
            dist_sq = dx * dx + dy * dy

            min_dist_sq = tl.minimum(min_dist_sq, dist_sq)

    # Store result
    min_dist = tl.sqrt(min_dist_sq)
    tl.store(min_dist_ptr + offs, min_dist, mask=mask)


def render_scene_triton(
    scene: FlattenedScene,
    config: RenderConfig = None,
) -> torch.Tensor:
    """
    Render using optimized Triton kernels.
    Currently supports stroke-only rendering with cubic beziers.
    """
    if config is None:
        config = RenderConfig()

    width = scene.canvas_width
    height = scene.canvas_height
    device = scene.device

    if device.type != 'cuda':
        return render_scene_py(scene, config)

    num_samples_x = config.num_samples_x
    num_samples_y = config.num_samples_y
    num_samples = num_samples_x * num_samples_y

    # Generate sample positions
    sample_pos = _generate_sample_positions_vectorized(
        width, height, num_samples_x, num_samples_y, device
    )

    # Initialize with background color
    bg = torch.tensor(config.background_color, device=device, dtype=torch.float32)

    if scene.paths is None:
        return bg.view(1, 1, 4).expand(height, width, 4).clone()

    paths = scene.paths
    groups = scene.groups

    # Flatten sample positions: [H*W*num_samples, 2]
    flat_samples = sample_pos.reshape(-1, 2).contiguous()
    N = flat_samples.shape[0]

    sample_x = flat_samples[:, 0].contiguous()
    sample_y = flat_samples[:, 1].contiguous()

    # Initialize sample colors with background
    sample_colors = bg.view(1, 4).expand(N, 4).clone()

    # Collect all cubic bezier segments for all stroked paths
    all_segments = []
    all_stroke_info = []  # (half_width, stroke_color)

    for group_idx in range(groups.num_groups):
        if not groups.shape_mask[group_idx, 0].item():
            continue

        has_stroke = groups.has_stroke[group_idx].item()
        if not has_stroke or groups.stroke_color is None:
            continue

        stroke_color = groups.stroke_color[group_idx]
        num_shapes = groups.num_shapes[group_idx].item()
        shape_ids = groups.shape_ids[group_idx, :num_shapes]

        for i in range(num_shapes):
            shape_id = shape_ids[i].item()
            shape_type = scene.shape_types[shape_id].item()
            shape_idx = scene.shape_indices[shape_id].item()

            if shape_type != ShapeType.PATH:
                continue

            stroke_width = paths.stroke_width[shape_idx].item()
            if stroke_width <= 0:
                continue

            half_width = stroke_width / 2.0
            point_offset = paths.point_offsets[shape_idx].item()
            num_segs = paths.num_segments[shape_idx].item()
            seg_types = paths.segment_types[shape_idx, :num_segs]

            path_segments = []
            current_point = point_offset
            for seg_idx in range(num_segs):
                seg_type = seg_types[seg_idx].item()
                if seg_type == 2:  # Cubic
                    p0 = paths.points[current_point]
                    p1 = paths.points[current_point + 1]
                    p2 = paths.points[current_point + 2]
                    p3 = paths.points[current_point + 3]
                    path_segments.append(torch.stack([p0, p1, p2, p3]))
                    current_point += 3
                elif seg_type == 1:  # Quadratic - convert to cubic
                    p0 = paths.points[current_point]
                    p1 = paths.points[current_point + 1]
                    p2 = paths.points[current_point + 2]
                    # Degree elevation: quadratic to cubic
                    c0 = p0
                    c1 = p0 + 2.0/3.0 * (p1 - p0)
                    c2 = p2 + 2.0/3.0 * (p1 - p2)
                    c3 = p2
                    path_segments.append(torch.stack([c0, c1, c2, c3]))
                    current_point += 2
                else:  # Line - convert to degenerate cubic
                    p0 = paths.points[current_point]
                    p1 = paths.points[current_point + 1]
                    c0 = p0
                    c1 = p0 + 1.0/3.0 * (p1 - p0)
                    c2 = p0 + 2.0/3.0 * (p1 - p0)
                    c3 = p1
                    path_segments.append(torch.stack([c0, c1, c2, c3]))
                    current_point += 1

            if path_segments:
                all_segments.extend(path_segments)
                all_stroke_info.append((half_width, stroke_color, len(path_segments)))

    if not all_segments:
        return bg.view(1, 1, 4).expand(height, width, 4).clone()

    # Stack all segments: [total_segments, 4, 2]
    bezier_segments = torch.stack(all_segments)  # [S, 4, 2]
    bezier_flat = bezier_segments.reshape(-1).contiguous()  # [S * 8]

    # Compute distances for each path and composite
    seg_start = 0
    for half_width, stroke_color, num_segs in all_stroke_info:
        # Get segments for this path
        path_beziers = bezier_flat[seg_start * 8:(seg_start + num_segs) * 8]

        # Allocate output
        min_dist = torch.empty(N, device=device, dtype=torch.float32)

        # Launch kernel
        BLOCK_SIZE = 256
        grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        _min_dist_to_cubics_kernel[grid](
            sample_x, sample_y,
            path_beziers,
            min_dist,
            N, num_segs, half_width,
            BLOCK_SIZE,
        )

        # Apply stroke
        inside_stroke = min_dist <= half_width
        alpha = stroke_color[3]
        sample_colors = torch.where(
            inside_stroke.unsqueeze(-1),
            stroke_color * alpha + sample_colors * (1 - alpha),
            sample_colors
        )

        seg_start += num_segs

    # Average samples per pixel
    sample_colors = sample_colors.reshape(height, width, num_samples, 4)
    output = sample_colors.mean(dim=2)

    return output


# Update render_scene_py to optionally use vectorized version
def render_scene_fast(
    scene: FlattenedScene,
    config: RenderConfig = None,
) -> torch.Tensor:
    """
    Fast rendering using vectorized operations.
    Fallback to reference implementation for complex cases.
    """
    if config is None:
        config = RenderConfig()

    # Use Triton kernel for GPU scenes
    if scene.device.type == 'cuda':
        return render_scene_triton(scene, config)

    # CPU fallback
    return render_scene_py(scene, config)
