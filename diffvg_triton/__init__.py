"""
diffvg_triton - Differentiable Vector Graphics with Triton

A pure Python/Triton implementation of differentiable vector graphics rendering.
This is a standalone module that can be used independently of the original diffvg.

Key components:
- scene: Scene flattening (convert shapes to GPU-friendly tensors)
- render: Main rendering pipeline (SVG rendering)
- render_batch: Batched rendering for training (fully differentiable)

Usage:
    from diffvg_triton import render, render_batch_fast

    # SVG rendering
    image = render(width, height, shapes, shape_groups)

    # Batched differentiable rendering (for training)
    images = render_batch_fast(width, height, control_points, stroke_widths, alphas)
"""

from .scene import (
    FlattenedPaths,
    FlattenedShapeGroup,
    FlattenedScene,
    ShapeType,
    flatten_paths,
    flatten_shape_groups,
    flatten_scene,
)

from .render import (
    RenderMode,
    RenderConfig,
    render,
)

from .render_batch import (
    render_batch_fast,
)

from .svg import (
    Path,
    ShapeGroup,
    svg_to_scene,
    save_svg,
)

from .io import (
    get_device,
    imwrite,
)



__all__ = [
    # Scene
    'FlattenedPaths',
    'FlattenedShapeGroup',
    'FlattenedScene',
    'ShapeType',
    'flatten_paths',
    'flatten_shape_groups',
    'flatten_scene',
    # Render
    'RenderMode',
    'RenderConfig',
    'render',
    # Batched rendering
    'render_batch_fast',
    # SVG utilities
    'Path',
    'ShapeGroup',
    'svg_to_scene',
    'save_svg',
    # I/O utilities
    'get_device',
    'imwrite',
]


# Version info
__version__ = '0.1.0'
__backend__ = 'triton'
