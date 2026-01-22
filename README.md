# diffvg-torch

A fast, pure-PyTorch reimplementation of [diffvg](https://github.com/BachiLi/diffvg). Differentiable and optimized for batched GPU workloads.

## Key Features

- **Pure PyTorch** - No C++ compilation, no custom CUDA kernels
- **Batched rendering** - Efficient parallel rendering of multiple scenes
- **Full gradient support** - Backpropagation through the rendering pipeline

## SVG Rendering

| Cairo | diffvg-torch |
|:-----:|:------------:|
| ![Cairo](assets/smiley_cairo.png) | ![diffvg-torch](assets/smiley_diffvg.png) |

## Performance

### Batched Rendering (render_batch_fast)

| Batch Size | Total Time | Per Image |
|------------|------------|-----------|
| 1 | 2.0ms | 2.0ms |
| 4 | 7.6ms | 1.9ms |
| 16 | 41.3ms | 2.6ms |
| 64 | 176.9ms | 2.8ms |

### Backward Pass Speedup vs pydiffvg

28x28 canvas, 1 path, 3 segments:

| Batch Size | diffvg-torch | pydiffvg | Speedup |
|------------|--------------|----------|---------|
| 8 | 5.2ms | 16.2ms | **3.1x** |
| 32 | 11.6ms | 71.7ms | **6.2x** |
| 64 | 23.1ms | 137.9ms | **6.0x** |

## Demo: MNIST VAE

Vector graphics reconstruction using 1 bezier path with 3 segments per digit:

![MNIST VAE Demo](assets/mnist_vae_demo.png)

## Installation

```bash
pip install torch numpy
pip install -e .
```

## Quick Start

```python
from diffvg_triton import render_batch_fast

# Render batched bezier paths
# control_points: [B, num_paths, num_segments, 4, 2]
# stroke_widths: [B, num_paths]
# alphas: [B, num_paths]
output = render_batch_fast(
    canvas_width=28,
    canvas_height=28,
    control_points=control_points,
    stroke_widths=stroke_widths,
    alphas=alphas,
    num_samples=4,
    use_fill=True,
)
# output: [B, 1, H, W] with gradients
```

### SVG File Rendering

```python
from diffvg_triton import render, svg_to_scene
import torch

canvas_w, canvas_h, shapes, shape_groups = svg_to_scene("input.svg")

output = render(
    canvas_width=256,
    canvas_height=256,
    shapes=shapes,
    shape_groups=shape_groups,
    num_samples_x=2,
    num_samples_y=2,
    background_color=torch.tensor([1.0, 1.0, 1.0, 1.0]),
    original_width=canvas_w,
    original_height=canvas_h,
)
# output: [H, W, 4] RGBA tensor
```

## Supported SVG Features

- **Shapes**: `<path>`, `<circle>`, `<ellipse>`, `<rect>`
- **Path commands**: M, L, Q, C, Z (absolute and relative)
- **Styling**: fill, stroke, stroke-width, opacity
- **Colors**: Named colors, hex (#RGB, #RRGGBB), rgb(), rgba()

## Citation

```bibtex
@software{diffvg_torch,
    title={diffvg-torch: Fast Differentiable Vector Graphics in Pure PyTorch},
    author={Carter McClellan},
    year={2025},
    url={https://github.com/CarterMcClellan/diffvg-torch}
}
```

## License

MIT
