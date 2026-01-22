"""
Tests for the rendering pipeline.

Tests scene flattening, rendering, and gradient computation.
"""

import pytest
import torch
import math


# Mock pydiffvg shape classes for testing
class MockPath:
    """Mock Path object for testing."""

    def __init__(self, points, num_control_points, is_closed=True, stroke_width=1.0):
        self.points = torch.tensor(points, dtype=torch.float32)
        self.num_control_points = torch.tensor(num_control_points, dtype=torch.int32)
        self.is_closed = is_closed
        self.stroke_width = torch.tensor([stroke_width], dtype=torch.float32)
        self.thickness = None


class MockShapeGroup:
    """Mock ShapeGroup object for testing."""

    def __init__(
        self,
        shape_ids,
        fill_color=None,
        stroke_color=None,
        use_even_odd_rule=True,
        shape_to_canvas=None,
    ):
        self.shape_ids = torch.tensor(shape_ids, dtype=torch.int32)
        self.fill_color = torch.tensor(fill_color, dtype=torch.float32) if fill_color else None
        self.stroke_color = torch.tensor(stroke_color, dtype=torch.float32) if stroke_color else None
        self.use_even_odd_rule = use_even_odd_rule
        self.shape_to_canvas = shape_to_canvas


class TestSceneFlattening:
    """Tests for scene flattening functions."""

    def test_flatten_single_path(self):
        """Test flattening a single path."""
        from diffvg_triton.scene import flatten_paths

        # Triangle path (3 line segments)
        path = MockPath(
            points=[[0, 0], [100, 0], [50, 100]],
            num_control_points=[0, 0, 0],  # All lines
            is_closed=True,
        )

        flattened = flatten_paths([path], device=torch.device('cpu'))

        assert flattened.num_paths == 1
        assert flattened.total_points == 3
        assert flattened.num_segments[0].item() == 3

    def test_flatten_multiple_paths(self):
        """Test flattening multiple paths."""
        from diffvg_triton.scene import flatten_paths

        path1 = MockPath(
            points=[[0, 0], [10, 0], [10, 10], [0, 10]],
            num_control_points=[0, 0, 0, 0],
            is_closed=True,
        )
        path2 = MockPath(
            points=[[20, 20], [30, 20], [25, 30]],
            num_control_points=[0, 0, 0],
            is_closed=True,
        )

        flattened = flatten_paths([path1, path2], device=torch.device('cpu'))

        assert flattened.num_paths == 2
        assert flattened.total_points == 7  # 4 + 3

    def test_flatten_shape_groups(self):
        """Test flattening shape groups."""
        from diffvg_triton.scene import flatten_shape_groups

        group = MockShapeGroup(
            shape_ids=[0],
            fill_color=[1.0, 0.0, 0.0, 1.0],  # Red
            stroke_color=None,
        )

        flattened = flatten_shape_groups([group], num_shapes=1, device=torch.device('cpu'))

        assert flattened.num_groups == 1
        assert flattened.has_fill[0].item() == True
        assert flattened.has_stroke[0].item() == False

    def test_flatten_scene(self):
        """Test full scene flattening."""
        from diffvg_triton.scene import flatten_scene

        path = MockPath(
            points=[[10, 10], [90, 10], [90, 90], [10, 90]],
            num_control_points=[0, 0, 0, 0],
            is_closed=True,
        )
        group = MockShapeGroup(
            shape_ids=[0],
            fill_color=[0.0, 0.0, 1.0, 1.0],  # Blue
        )

        scene = flatten_scene(
            canvas_width=100,
            canvas_height=100,
            shapes=[path],
            shape_groups=[group],
            device=torch.device('cpu'),
        )

        assert scene.canvas_width == 100
        assert scene.canvas_height == 100
        assert scene.paths is not None
        assert scene.paths.num_paths == 1


class TestRendering:
    """Tests for the rendering pipeline."""

    def test_render_empty_scene(self):
        """Test rendering with no shapes returns background."""
        from diffvg_triton.render import render

        image = render(
            canvas_width=10,
            canvas_height=10,
            shapes=[],
            shape_groups=[],
            num_samples_x=1,
            num_samples_y=1,
            background_color=torch.tensor([1.0, 1.0, 1.0, 1.0]),
        )

        assert image.shape == (10, 10, 4)
        # Should be white background
        assert torch.allclose(image, torch.ones_like(image), atol=0.01)

    def test_render_filled_square(self):
        """Test rendering a filled square."""
        from diffvg_triton.render import render

        # Create a square that covers pixels (2,2) to (7,7)
        path = MockPath(
            points=[[2, 2], [8, 2], [8, 8], [2, 8]],
            num_control_points=[0, 0, 0, 0],
            is_closed=True,
        )
        group = MockShapeGroup(
            shape_ids=[0],
            fill_color=[1.0, 0.0, 0.0, 1.0],  # Red
        )

        image = render(
            canvas_width=10,
            canvas_height=10,
            shapes=[path],
            shape_groups=[group],
            num_samples_x=1,
            num_samples_y=1,
            background_color=torch.tensor([1.0, 1.0, 1.0, 1.0]),
        )

        assert image.shape == (10, 10, 4)

        # Check that center pixel is red
        center_pixel = image[5, 5]
        assert center_pixel[0] > 0.9  # Red
        assert center_pixel[1] < 0.1  # Not green
        assert center_pixel[2] < 0.1  # Not blue

        # Check that corner pixel is white (background)
        corner_pixel = image[0, 0]
        assert corner_pixel[0] > 0.9
        assert corner_pixel[1] > 0.9
        assert corner_pixel[2] > 0.9


class TestHighLevelAPI:
    """Tests for the high-level rendering API."""

    def test_render_function(self):
        """Test the main render function."""
        from diffvg_triton.render import render

        path = MockPath(
            points=[[5, 5], [15, 5], [15, 15], [5, 15]],
            num_control_points=[0, 0, 0, 0],
            is_closed=True,
        )
        group = MockShapeGroup(
            shape_ids=[0],
            fill_color=[1.0, 1.0, 0.0, 1.0],  # Yellow
        )

        image = render(
            canvas_width=20,
            canvas_height=20,
            shapes=[path],
            shape_groups=[group],
            num_samples_x=1,
            num_samples_y=1,
        )

        assert image.shape == (20, 20, 4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
