"""
Scene representation for the PyTorch backend.

Converts pydiffvg shapes into flattened tensor representation suitable for
Triton kernel processing.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import numpy as np


@dataclass
class FlattenedPaths:
    """
    Flattened representation of all Path shapes in a scene.

    Paths have variable numbers of segments and points, so we flatten everything
    into contiguous tensors with index arrays for lookup.

    Memory layout:
    - segment_types[path_idx, seg_idx] = type (0=line, 1=quadratic, 2=cubic)
    - segment_mask[path_idx, seg_idx] = True if segment is valid
    - points[point_idx, :] = (x, y) coordinates
    - point_offsets[path_idx] = starting index in points array for this path
    """
    # Per-segment data [num_paths, max_segments]
    segment_types: torch.Tensor      # int32: 0=line, 1=quadratic, 2=cubic
    segment_mask: torch.Tensor       # bool: valid segment mask
    num_segments: torch.Tensor       # int32: [num_paths] actual segment count per path

    # All control points concatenated [total_points, 2]
    points: torch.Tensor             # float32: (x, y) coordinates

    # Index arrays [num_paths]
    point_offsets: torch.Tensor      # int32: where each path's points start
    num_points: torch.Tensor         # int32: number of points per path

    # Per-path metadata [num_paths]
    is_closed: torch.Tensor          # bool
    stroke_width: torch.Tensor       # float32: uniform stroke width per path

    # Optional per-point stroke thickness [total_points] or None
    thickness: Optional[torch.Tensor] = None

    @property
    def num_paths(self) -> int:
        return self.point_offsets.shape[0]

    @property
    def max_segments(self) -> int:
        return self.segment_types.shape[1]

    @property
    def total_points(self) -> int:
        return self.points.shape[0]

    @property
    def device(self) -> torch.device:
        return self.points.device

    def to(self, device: torch.device) -> 'FlattenedPaths':
        """Move all tensors to the specified device."""
        return FlattenedPaths(
            segment_types=self.segment_types.to(device),
            segment_mask=self.segment_mask.to(device),
            num_segments=self.num_segments.to(device),
            points=self.points.to(device),
            point_offsets=self.point_offsets.to(device),
            num_points=self.num_points.to(device),
            is_closed=self.is_closed.to(device),
            stroke_width=self.stroke_width.to(device),
            thickness=self.thickness.to(device) if self.thickness is not None else None,
        )


@dataclass
class FlattenedShapeGroup:
    """
    Flattened representation of ShapeGroups.

    Each group can contain multiple shapes and has fill/stroke colors.
    """
    # Shape membership [num_groups, max_shapes_per_group]
    shape_ids: torch.Tensor          # int32: indices into shapes list
    shape_mask: torch.Tensor         # bool: valid shape mask
    num_shapes: torch.Tensor         # int32: [num_groups] actual shape count

    # Fill color [num_groups, 4] for constant colors
    # For gradients, we'll add separate arrays
    fill_color: Optional[torch.Tensor]        # float32: RGBA
    has_fill: torch.Tensor           # bool: [num_groups]

    # Stroke color [num_groups, 4]
    stroke_color: Optional[torch.Tensor]      # float32: RGBA
    has_stroke: torch.Tensor         # bool: [num_groups]

    # Fill rule [num_groups]
    use_even_odd_rule: torch.Tensor  # bool

    # Transformation [num_groups, 3, 3]
    shape_to_canvas: torch.Tensor    # float32: transformation matrix
    canvas_to_shape: torch.Tensor    # float32: inverse transformation

    @property
    def num_groups(self) -> int:
        return self.shape_ids.shape[0]

    @property
    def device(self) -> torch.device:
        return self.shape_ids.device

    def to(self, device: torch.device) -> 'FlattenedShapeGroup':
        return FlattenedShapeGroup(
            shape_ids=self.shape_ids.to(device),
            shape_mask=self.shape_mask.to(device),
            num_shapes=self.num_shapes.to(device),
            fill_color=self.fill_color.to(device) if self.fill_color is not None else None,
            has_fill=self.has_fill.to(device),
            stroke_color=self.stroke_color.to(device) if self.stroke_color is not None else None,
            has_stroke=self.has_stroke.to(device),
            use_even_odd_rule=self.use_even_odd_rule.to(device),
            shape_to_canvas=self.shape_to_canvas.to(device),
            canvas_to_shape=self.canvas_to_shape.to(device),
        )


@dataclass
class FlattenedScene:
    """
    Complete flattened scene representation for PyTorch rendering.

    Contains all shapes flattened into GPU-friendly tensors.
    """
    canvas_width: int
    canvas_height: int

    # Shapes (currently only paths for Phase 1)
    paths: Optional[FlattenedPaths]

    # Shape groups
    groups: FlattenedShapeGroup

    # Mapping from group shape_ids to actual shape type and index
    # shape_types[shape_id] = ShapeType enum value
    # shape_indices[shape_id] = index within that shape type's array
    shape_types: torch.Tensor        # int32: [num_shapes]
    shape_indices: torch.Tensor      # int32: [num_shapes]

    @property
    def device(self) -> torch.device:
        if self.paths is not None:
            return self.paths.device
        return self.groups.device

    def to(self, device: torch.device) -> 'FlattenedScene':
        return FlattenedScene(
            canvas_width=self.canvas_width,
            canvas_height=self.canvas_height,
            paths=self.paths.to(device) if self.paths is not None else None,
            groups=self.groups.to(device),
            shape_types=self.shape_types.to(device),
            shape_indices=self.shape_indices.to(device),
        )


# Shape type constants (matching diffvg)
class ShapeType:
    CIRCLE = 0
    ELLIPSE = 1
    PATH = 2
    RECT = 3


def flatten_paths(paths: list, max_segments: int = 512, device: torch.device = None) -> FlattenedPaths:
    """
    Flatten a list of Path objects into FlattenedPaths tensor representation.

    Args:
        paths: List of pydiffvg.Path objects
        max_segments: Maximum segments per path (paths with more segments will error)
        device: Target device for tensors

    Returns:
        FlattenedPaths with all path data in contiguous tensors
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_paths = len(paths)
    if num_paths == 0:
        # Return empty structure
        return FlattenedPaths(
            segment_types=torch.zeros((0, max_segments), dtype=torch.int32, device=device),
            segment_mask=torch.zeros((0, max_segments), dtype=torch.bool, device=device),
            num_segments=torch.zeros((0,), dtype=torch.int32, device=device),
            points=torch.zeros((0, 2), dtype=torch.float32, device=device),
            point_offsets=torch.zeros((0,), dtype=torch.int32, device=device),
            num_points=torch.zeros((0,), dtype=torch.int32, device=device),
            is_closed=torch.zeros((0,), dtype=torch.bool, device=device),
            stroke_width=torch.zeros((0,), dtype=torch.float32, device=device),
            thickness=None,
        )

    # Collect data
    all_segment_types = []
    all_segment_masks = []
    all_points = []
    point_offsets = []
    num_points_list = []
    num_segments_list = []
    is_closed_list = []
    stroke_width_list = []
    has_thickness = False

    current_point_offset = 0

    for path in paths:
        # Get segment info from num_control_points
        # num_control_points[i] = 0 means line, 1 means quadratic, 2 means cubic
        num_ctrl = path.num_control_points.cpu().numpy()
        n_segments = len(num_ctrl)

        if n_segments > max_segments:
            raise ValueError(f"Path has {n_segments} segments, exceeding max of {max_segments}")

        # Pad to max_segments
        seg_types = np.zeros(max_segments, dtype=np.int32)
        seg_mask = np.zeros(max_segments, dtype=bool)

        seg_types[:n_segments] = num_ctrl
        seg_mask[:n_segments] = True

        all_segment_types.append(seg_types)
        all_segment_masks.append(seg_mask)
        num_segments_list.append(n_segments)

        # Points
        pts = path.points.cpu()  # [N, 2] or [N*2] flattened
        if pts.dim() == 1:
            pts = pts.view(-1, 2)
        n_points = pts.shape[0]

        all_points.append(pts)
        point_offsets.append(current_point_offset)
        num_points_list.append(n_points)
        current_point_offset += n_points

        # Metadata
        is_closed_list.append(path.is_closed)

        # Stroke width
        if hasattr(path, 'stroke_width') and path.stroke_width is not None:
            sw = path.stroke_width
            if isinstance(sw, torch.Tensor):
                sw = sw.item() if sw.numel() == 1 else sw.mean().item()
            stroke_width_list.append(float(sw))
        else:
            stroke_width_list.append(1.0)

        # Check for per-point thickness
        if hasattr(path, 'thickness') and path.thickness is not None:
            has_thickness = True

    # Stack into tensors
    segment_types = torch.tensor(np.stack(all_segment_types), dtype=torch.int32, device=device)
    segment_mask = torch.tensor(np.stack(all_segment_masks), dtype=torch.bool, device=device)
    num_segments = torch.tensor(num_segments_list, dtype=torch.int32, device=device)
    points = torch.cat(all_points, dim=0).to(dtype=torch.float32, device=device)
    point_offsets_t = torch.tensor(point_offsets, dtype=torch.int32, device=device)
    num_points_t = torch.tensor(num_points_list, dtype=torch.int32, device=device)
    is_closed = torch.tensor(is_closed_list, dtype=torch.bool, device=device)
    stroke_width = torch.tensor(stroke_width_list, dtype=torch.float32, device=device)

    # Per-point thickness (optional)
    thickness = None
    if has_thickness:
        all_thickness = []
        for path in paths:
            if hasattr(path, 'thickness') and path.thickness is not None:
                th = path.thickness.cpu()
            else:
                n = path.points.shape[0] if path.points.dim() == 2 else path.points.shape[0] // 2
                th = torch.ones(n) * stroke_width_list[paths.index(path)]
            all_thickness.append(th)
        thickness = torch.cat(all_thickness).to(dtype=torch.float32, device=device)

    return FlattenedPaths(
        segment_types=segment_types,
        segment_mask=segment_mask,
        num_segments=num_segments,
        points=points,
        point_offsets=point_offsets_t,
        num_points=num_points_t,
        is_closed=is_closed,
        stroke_width=stroke_width,
        thickness=thickness,
    )


def flatten_shape_groups(
    shape_groups: list,
    num_shapes: int,
    max_shapes_per_group: int = 64,
    device: torch.device = None
) -> FlattenedShapeGroup:
    """
    Flatten ShapeGroup objects into tensor representation.

    Args:
        shape_groups: List of pydiffvg.ShapeGroup objects
        num_shapes: Total number of shapes in scene
        max_shapes_per_group: Maximum shapes per group
        device: Target device

    Returns:
        FlattenedShapeGroup
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_groups = len(shape_groups)
    if num_groups == 0:
        return FlattenedShapeGroup(
            shape_ids=torch.zeros((0, max_shapes_per_group), dtype=torch.int32, device=device),
            shape_mask=torch.zeros((0, max_shapes_per_group), dtype=torch.bool, device=device),
            num_shapes=torch.zeros((0,), dtype=torch.int32, device=device),
            fill_color=None,
            has_fill=torch.zeros((0,), dtype=torch.bool, device=device),
            stroke_color=None,
            has_stroke=torch.zeros((0,), dtype=torch.bool, device=device),
            use_even_odd_rule=torch.zeros((0,), dtype=torch.bool, device=device),
            shape_to_canvas=torch.zeros((0, 3, 3), dtype=torch.float32, device=device),
            canvas_to_shape=torch.zeros((0, 3, 3), dtype=torch.float32, device=device),
        )

    all_shape_ids = []
    all_shape_masks = []
    all_num_shapes = []
    all_fill_colors = []
    all_has_fill = []
    all_stroke_colors = []
    all_has_stroke = []
    all_even_odd = []
    all_shape_to_canvas = []
    all_canvas_to_shape = []

    for group in shape_groups:
        # Shape IDs
        ids = group.shape_ids.cpu().numpy()
        n_shapes = len(ids)

        if n_shapes > max_shapes_per_group:
            raise ValueError(f"Group has {n_shapes} shapes, exceeding max of {max_shapes_per_group}")

        padded_ids = np.zeros(max_shapes_per_group, dtype=np.int32)
        mask = np.zeros(max_shapes_per_group, dtype=bool)
        padded_ids[:n_shapes] = ids
        mask[:n_shapes] = True

        all_shape_ids.append(padded_ids)
        all_shape_masks.append(mask)
        all_num_shapes.append(n_shapes)

        # Fill color (constant only for Phase 1)
        if group.fill_color is not None:
            fc = group.fill_color
            if isinstance(fc, torch.Tensor):
                fc = fc.detach().cpu().numpy()
                if fc.shape == (4,):
                    all_fill_colors.append(fc)
                else:
                    # Gradient - use first stop color for now
                    all_fill_colors.append(np.array([1.0, 1.0, 1.0, 1.0]))
            else:
                # Gradient object
                all_fill_colors.append(np.array([1.0, 1.0, 1.0, 1.0]))
            all_has_fill.append(True)
        else:
            all_fill_colors.append(np.array([0.0, 0.0, 0.0, 0.0]))
            all_has_fill.append(False)

        # Stroke color
        if group.stroke_color is not None:
            sc = group.stroke_color
            if isinstance(sc, torch.Tensor):
                sc = sc.detach().cpu().numpy()
                if sc.shape == (4,):
                    all_stroke_colors.append(sc)
                else:
                    all_stroke_colors.append(np.array([0.0, 0.0, 0.0, 1.0]))
            else:
                all_stroke_colors.append(np.array([0.0, 0.0, 0.0, 1.0]))
            all_has_stroke.append(True)
        else:
            all_stroke_colors.append(np.array([0.0, 0.0, 0.0, 0.0]))
            all_has_stroke.append(False)

        # Fill rule
        even_odd = getattr(group, 'use_even_odd_rule', True)
        all_even_odd.append(even_odd)

        # Transformation
        if hasattr(group, 'shape_to_canvas') and group.shape_to_canvas is not None:
            xform = group.shape_to_canvas.cpu().numpy()
            if xform.shape == (3, 3):
                all_shape_to_canvas.append(xform)
                # Compute inverse
                try:
                    inv_xform = np.linalg.inv(xform)
                except np.linalg.LinAlgError:
                    inv_xform = np.eye(3)
                all_canvas_to_shape.append(inv_xform)
            else:
                all_shape_to_canvas.append(np.eye(3))
                all_canvas_to_shape.append(np.eye(3))
        else:
            all_shape_to_canvas.append(np.eye(3))
            all_canvas_to_shape.append(np.eye(3))

    return FlattenedShapeGroup(
        shape_ids=torch.tensor(np.stack(all_shape_ids), dtype=torch.int32, device=device),
        shape_mask=torch.tensor(np.stack(all_shape_masks), dtype=torch.bool, device=device),
        num_shapes=torch.tensor(all_num_shapes, dtype=torch.int32, device=device),
        fill_color=torch.tensor(np.stack(all_fill_colors), dtype=torch.float32, device=device),
        has_fill=torch.tensor(all_has_fill, dtype=torch.bool, device=device),
        stroke_color=torch.tensor(np.stack(all_stroke_colors), dtype=torch.float32, device=device),
        has_stroke=torch.tensor(all_has_stroke, dtype=torch.bool, device=device),
        use_even_odd_rule=torch.tensor(all_even_odd, dtype=torch.bool, device=device),
        shape_to_canvas=torch.tensor(np.stack(all_shape_to_canvas), dtype=torch.float32, device=device),
        canvas_to_shape=torch.tensor(np.stack(all_canvas_to_shape), dtype=torch.float32, device=device),
    )


def flatten_scene(
    canvas_width: int,
    canvas_height: int,
    shapes: list,
    shape_groups: list,
    device: torch.device = None,
) -> FlattenedScene:
    """
    Convert pydiffvg scene into FlattenedScene for PyTorch rendering.

    Args:
        canvas_width: Width of output image
        canvas_height: Height of output image
        shapes: List of shape objects (Path, Circle, etc.)
        shape_groups: List of ShapeGroup objects
        device: Target device

    Returns:
        FlattenedScene ready for PyTorch operations
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Separate shapes by type
    paths = []
    circles = []
    ellipses = []
    rects = []

    shape_types = []
    shape_indices = []

    for shape in shapes:
        type_name = type(shape).__name__

        # Check for Path-like objects (has points, num_control_points, is_closed)
        is_path = (
            type_name == 'Path' or
            type_name == 'MockPath' or
            (hasattr(shape, 'points') and hasattr(shape, 'num_control_points'))
        )

        if is_path:
            shape_types.append(ShapeType.PATH)
            shape_indices.append(len(paths))
            paths.append(shape)
        elif type_name == 'Circle' or (hasattr(shape, 'center') and hasattr(shape, 'radius') and not hasattr(shape, 'radius_x')):
            shape_types.append(ShapeType.CIRCLE)
            shape_indices.append(len(circles))
            circles.append(shape)
        elif type_name == 'Ellipse' or (hasattr(shape, 'center') and hasattr(shape, 'radius_x')):
            shape_types.append(ShapeType.ELLIPSE)
            shape_indices.append(len(ellipses))
            ellipses.append(shape)
        elif type_name == 'Rect' or (hasattr(shape, 'p_min') and hasattr(shape, 'p_max')):
            shape_types.append(ShapeType.RECT)
            shape_indices.append(len(rects))
            rects.append(shape)
        else:
            raise ValueError(f"Unknown shape type: {type_name}")

    # Flatten paths
    flattened_paths = flatten_paths(paths, device=device) if paths else None

    # Flatten shape groups
    flattened_groups = flatten_shape_groups(
        shape_groups, len(shapes), device=device
    )

    return FlattenedScene(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        paths=flattened_paths,
        groups=flattened_groups,
        shape_types=torch.tensor(shape_types, dtype=torch.int32, device=device),
        shape_indices=torch.tensor(shape_indices, dtype=torch.int32, device=device),
    )
