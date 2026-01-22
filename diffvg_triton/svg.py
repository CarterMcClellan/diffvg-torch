"""
SVG parsing and saving utilities for diffvg_triton.

Provides functions to load SVG files into shapes/shape_groups and
save them back to SVG format.
"""

import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional
import re
import torch


class Path:
    """Bezier path shape."""
    def __init__(self, num_control_points, points, stroke_width=1.0, is_closed=False):
        self.num_control_points = num_control_points
        self.points = points
        self.stroke_width = stroke_width if isinstance(stroke_width, torch.Tensor) else torch.tensor([stroke_width])
        self.is_closed = is_closed
        self.thickness = None


class ShapeGroup:
    """Group of shapes with fill/stroke colors."""
    def __init__(self, shape_ids, fill_color=None, stroke_color=None, use_even_odd_rule=True):
        self.shape_ids = shape_ids if isinstance(shape_ids, torch.Tensor) else torch.tensor(shape_ids, dtype=torch.int32)
        self.fill_color = fill_color
        self.stroke_color = stroke_color
        self.use_even_odd_rule = use_even_odd_rule
        self.shape_to_canvas = None


def _parse_path_d(d: str) -> Tuple[List[int], List[Tuple[float, float]]]:
    """Parse SVG path 'd' attribute into control points and segment types."""
    points = []
    num_control_points = []
    tokens = re.findall(r'[MmLlHhVvCcSsQqTtAaZz]|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', d)

    i = 0
    current_x, current_y = 0.0, 0.0
    start_x, start_y = 0.0, 0.0
    last_control = None
    last_command = None

    while i < len(tokens):
        cmd = tokens[i]
        if cmd in 'Mm':
            relative = cmd == 'm'
            i += 1
            x, y = float(tokens[i]), float(tokens[i+1])
            i += 2
            if relative:
                current_x += x; current_y += y
            else:
                current_x, current_y = x, y
            start_x, start_y = current_x, current_y
            points.append((current_x, current_y))
            last_command, last_control = cmd, None
            while i < len(tokens) and tokens[i] not in 'MmLlHhVvCcSsQqTtAaZz':
                x, y = float(tokens[i]), float(tokens[i+1])
                i += 2
                if relative:
                    current_x += x; current_y += y
                else:
                    current_x, current_y = x, y
                points.append((current_x, current_y))
                num_control_points.append(0)
        elif cmd in 'Ll':
            relative = cmd == 'l'
            i += 1
            while i < len(tokens) and tokens[i] not in 'MmLlHhVvCcSsQqTtAaZz':
                x, y = float(tokens[i]), float(tokens[i+1])
                i += 2
                if relative:
                    current_x += x; current_y += y
                else:
                    current_x, current_y = x, y
                points.append((current_x, current_y))
                num_control_points.append(0)
            last_command, last_control = cmd, None
        elif cmd in 'Hh':
            relative = cmd == 'h'
            i += 1
            while i < len(tokens) and tokens[i] not in 'MmLlHhVvCcSsQqTtAaZz':
                x = float(tokens[i])
                i += 1
                current_x = current_x + x if relative else x
                points.append((current_x, current_y))
                num_control_points.append(0)
            last_command, last_control = cmd, None
        elif cmd in 'Vv':
            relative = cmd == 'v'
            i += 1
            while i < len(tokens) and tokens[i] not in 'MmLlHhVvCcSsQqTtAaZz':
                y = float(tokens[i])
                i += 1
                current_y = current_y + y if relative else y
                points.append((current_x, current_y))
                num_control_points.append(0)
            last_command, last_control = cmd, None
        elif cmd in 'Cc':
            relative = cmd == 'c'
            i += 1
            while i < len(tokens) and tokens[i] not in 'MmLlHhVvCcSsQqTtAaZz':
                x1, y1 = float(tokens[i]), float(tokens[i+1])
                x2, y2 = float(tokens[i+2]), float(tokens[i+3])
                x, y = float(tokens[i+4]), float(tokens[i+5])
                i += 6
                if relative:
                    x1 += current_x; y1 += current_y
                    x2 += current_x; y2 += current_y
                    x += current_x; y += current_y
                points.extend([(x1, y1), (x2, y2), (x, y)])
                num_control_points.append(2)
                last_control = (x2, y2)
                current_x, current_y = x, y
            last_command = cmd
        elif cmd in 'Ss':
            relative = cmd == 's'
            i += 1
            while i < len(tokens) and tokens[i] not in 'MmLlHhVvCcSsQqTtAaZz':
                if last_command in 'CcSs' and last_control:
                    x1 = 2 * current_x - last_control[0]
                    y1 = 2 * current_y - last_control[1]
                else:
                    x1, y1 = current_x, current_y
                x2, y2 = float(tokens[i]), float(tokens[i+1])
                x, y = float(tokens[i+2]), float(tokens[i+3])
                i += 4
                if relative:
                    x2 += current_x; y2 += current_y
                    x += current_x; y += current_y
                points.extend([(x1, y1), (x2, y2), (x, y)])
                num_control_points.append(2)
                last_control = (x2, y2)
                current_x, current_y = x, y
            last_command = cmd
        elif cmd in 'Qq':
            relative = cmd == 'q'
            i += 1
            while i < len(tokens) and tokens[i] not in 'MmLlHhVvCcSsQqTtAaZz':
                x1, y1 = float(tokens[i]), float(tokens[i+1])
                x, y = float(tokens[i+2]), float(tokens[i+3])
                i += 4
                if relative:
                    x1 += current_x; y1 += current_y
                    x += current_x; y += current_y
                points.extend([(x1, y1), (x, y)])
                num_control_points.append(1)
                last_control = (x1, y1)
                current_x, current_y = x, y
            last_command = cmd
        elif cmd in 'Zz':
            i += 1
            if (current_x, current_y) != (start_x, start_y):
                points.append((start_x, start_y))
                num_control_points.append(0)
            current_x, current_y = start_x, start_y
            last_command, last_control = cmd, None
        elif cmd in 'Aa':
            # Simplified arc handling - just draw line
            relative = cmd == 'a'
            i += 1
            while i + 6 < len(tokens) and tokens[i] not in 'MmLlHhVvCcSsQqTtAaZz':
                i += 5  # Skip rx, ry, rotation, large_arc, sweep
                x, y = float(tokens[i]), float(tokens[i+1])
                i += 2
                if relative:
                    x += current_x; y += current_y
                points.append((x, y))
                num_control_points.append(0)
                current_x, current_y = x, y
            last_command, last_control = cmd, None
        else:
            i += 1
    return num_control_points, points


def _parse_color(color_str: str) -> Optional[Tuple[float, float, float, float]]:
    """Parse SVG color string to RGBA tuple (0-1 range)."""
    if not color_str or color_str.lower() == 'none':
        return None
    color_str = color_str.strip()
    if color_str.startswith('#'):
        h = color_str[1:]
        if len(h) == 3:
            h = ''.join(c * 2 for c in h)
        if len(h) == 6:
            return (int(h[0:2], 16)/255, int(h[2:4], 16)/255, int(h[4:6], 16)/255, 1.0)
    if color_str.startswith('rgb'):
        m = re.match(r'rgba?\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([\d.]+))?\s*\)', color_str)
        if m:
            return (int(m.group(1))/255, int(m.group(2))/255, int(m.group(3))/255, float(m.group(4)) if m.group(4) else 1.0)
    # Extended CSS named colors
    colors = {
        # Basic colors
        'black': (0, 0, 0, 1), 'white': (1, 1, 1, 1), 'red': (1, 0, 0, 1),
        'green': (0, 0.5, 0, 1), 'blue': (0, 0, 1, 1), 'yellow': (1, 1, 0, 1),
        'cyan': (0, 1, 1, 1), 'magenta': (1, 0, 1, 1), 'aqua': (0, 1, 1, 1),
        # Extended colors
        'orange': (1, 0.647, 0, 1), 'purple': (0.5, 0, 0.5, 1), 'pink': (1, 0.753, 0.796, 1),
        'brown': (0.647, 0.165, 0.165, 1), 'gray': (0.5, 0.5, 0.5, 1), 'grey': (0.5, 0.5, 0.5, 1),
        'lime': (0, 1, 0, 1), 'navy': (0, 0, 0.5, 1), 'teal': (0, 0.5, 0.5, 1),
        'olive': (0.5, 0.5, 0, 1), 'maroon': (0.5, 0, 0, 1), 'silver': (0.753, 0.753, 0.753, 1),
        'gold': (1, 0.843, 0, 1), 'coral': (1, 0.498, 0.314, 1), 'salmon': (0.98, 0.502, 0.447, 1),
        'khaki': (0.941, 0.902, 0.549, 1), 'violet': (0.933, 0.51, 0.933, 1),
        'indigo': (0.294, 0, 0.51, 1), 'beige': (0.961, 0.961, 0.863, 1),
        'crimson': (0.863, 0.078, 0.235, 1), 'darkblue': (0, 0, 0.545, 1),
        'darkgreen': (0, 0.392, 0, 1), 'darkred': (0.545, 0, 0, 1),
        'lightblue': (0.678, 0.847, 0.902, 1), 'lightgreen': (0.565, 0.933, 0.565, 1),
        'lightgray': (0.827, 0.827, 0.827, 1), 'lightgrey': (0.827, 0.827, 0.827, 1),
        'darkgray': (0.663, 0.663, 0.663, 1), 'darkgrey': (0.663, 0.663, 0.663, 1),
    }
    return colors.get(color_str.lower(), (0, 0, 0, 1))


def svg_to_scene(svg_path: str) -> Tuple[int, int, List[Path], List[ShapeGroup]]:
    """Load SVG file and convert to shapes/shape_groups."""
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Get canvas size
    width = root.get('width', '100')
    height = root.get('height', '100')
    canvas_width = int(float(re.sub(r'[a-z%]+', '', str(width))))
    canvas_height = int(float(re.sub(r'[a-z%]+', '', str(height))))

    viewbox = root.get('viewBox')
    if viewbox:
        parts = viewbox.split()
        if len(parts) >= 4:
            canvas_width, canvas_height = int(float(parts[2])), int(float(parts[3]))

    shapes, shape_groups = [], []

    def get_attr(elem, attr, default=None):
        val = elem.get(attr)
        if val: return val
        style = elem.get('style', '')
        for part in style.split(';'):
            if ':' in part:
                k, v = part.split(':', 1)
                if k.strip() == attr: return v.strip()
        return default

    def process(elem, inherited_fill=None):
        nonlocal shapes, shape_groups
        fill_str = get_attr(elem, 'fill', inherited_fill)
        stroke_str = get_attr(elem, 'stroke')
        tag = elem.tag.split('}')[-1]

        if tag == 'path':
            d = elem.get('d', '')
            if d:
                num_ctrl, pts = _parse_path_d(d)
                if len(pts) >= 2:
                    stroke_width = float(get_attr(elem, 'stroke-width', '1.0') or '1.0')
                    path = Path(torch.tensor(num_ctrl, dtype=torch.int32),
                               torch.tensor(pts, dtype=torch.float32),
                               stroke_width=stroke_width,
                               is_closed=d.strip().upper().endswith('Z'))
                    shapes.append(path)
                    fc = _parse_color(fill_str)
                    sc = _parse_color(stroke_str)
                    shape_groups.append(ShapeGroup(
                        [len(shapes)-1],
                        fill_color=torch.tensor(fc, dtype=torch.float32) if fc else None,
                        stroke_color=torch.tensor(sc, dtype=torch.float32) if sc else None))
        elif tag == 'rect':
            x, y = float(elem.get('x', 0)), float(elem.get('y', 0))
            w, h = float(elem.get('width', 0)), float(elem.get('height', 0))
            if w > 0 and h > 0:
                stroke_width = float(get_attr(elem, 'stroke-width', '1.0') or '1.0')
                pts = [(x,y), (x+w,y), (x+w,y+h), (x,y+h), (x,y)]
                path = Path(torch.tensor([0,0,0,0], dtype=torch.int32),
                           torch.tensor(pts, dtype=torch.float32),
                           stroke_width=stroke_width, is_closed=True)
                shapes.append(path)
                fc = _parse_color(fill_str)
                sc = _parse_color(stroke_str)
                shape_groups.append(ShapeGroup(
                    [len(shapes)-1],
                    fill_color=torch.tensor(fc, dtype=torch.float32) if fc else None,
                    stroke_color=torch.tensor(sc, dtype=torch.float32) if sc else None))
        elif tag == 'circle':
            cx, cy, r = float(elem.get('cx', 0)), float(elem.get('cy', 0)), float(elem.get('r', 0))
            if r > 0:
                stroke_width = float(get_attr(elem, 'stroke-width', '1.0') or '1.0')
                k = 0.5522847498
                pts = [(cx, cy-r), (cx+k*r, cy-r), (cx+r, cy-k*r), (cx+r, cy),
                       (cx+r, cy+k*r), (cx+k*r, cy+r), (cx, cy+r),
                       (cx-k*r, cy+r), (cx-r, cy+k*r), (cx-r, cy),
                       (cx-r, cy-k*r), (cx-k*r, cy-r), (cx, cy-r)]
                path = Path(torch.tensor([2,2,2,2], dtype=torch.int32),
                           torch.tensor(pts, dtype=torch.float32),
                           stroke_width=stroke_width, is_closed=True)
                shapes.append(path)
                fc = _parse_color(fill_str)
                sc = _parse_color(stroke_str)
                shape_groups.append(ShapeGroup(
                    [len(shapes)-1],
                    fill_color=torch.tensor(fc, dtype=torch.float32) if fc else None,
                    stroke_color=torch.tensor(sc, dtype=torch.float32) if sc else None))
        elif tag == 'ellipse':
            cx = float(elem.get('cx', 0))
            cy = float(elem.get('cy', 0))
            rx = float(elem.get('rx', 0))
            ry = float(elem.get('ry', 0))
            if rx > 0 and ry > 0:
                stroke_width = float(get_attr(elem, 'stroke-width', '1.0') or '1.0')
                k = 0.5522847498  # Magic number for cubic bezier circle approximation
                pts = [
                    (cx, cy-ry), (cx+k*rx, cy-ry), (cx+rx, cy-k*ry), (cx+rx, cy),
                    (cx+rx, cy+k*ry), (cx+k*rx, cy+ry), (cx, cy+ry),
                    (cx-k*rx, cy+ry), (cx-rx, cy+k*ry), (cx-rx, cy),
                    (cx-rx, cy-k*ry), (cx-k*rx, cy-ry), (cx, cy-ry)
                ]
                path = Path(torch.tensor([2,2,2,2], dtype=torch.int32),
                           torch.tensor(pts, dtype=torch.float32),
                           stroke_width=stroke_width, is_closed=True)
                shapes.append(path)
                fc = _parse_color(fill_str)
                sc = _parse_color(stroke_str)
                shape_groups.append(ShapeGroup(
                    [len(shapes)-1],
                    fill_color=torch.tensor(fc, dtype=torch.float32) if fc else None,
                    stroke_color=torch.tensor(sc, dtype=torch.float32) if sc else None))
        elif tag in ('g', 'svg'):
            for child in elem:
                process(child, fill_str)

    process(root)
    return canvas_width, canvas_height, shapes, shape_groups


def save_svg(path: str, canvas_width: int, canvas_height: int,
             shapes: List[Path], shape_groups: List[ShapeGroup]):
    """Save shapes to SVG file."""
    svg = ET.Element('svg')
    svg.set('xmlns', 'http://www.w3.org/2000/svg')
    svg.set('width', str(canvas_width))
    svg.set('height', str(canvas_height))
    svg.set('viewBox', f'0 0 {canvas_width} {canvas_height}')

    for group in shape_groups:
        for shape_idx in group.shape_ids.tolist():
            if shape_idx >= len(shapes): continue
            shape = shapes[shape_idx]
            points = shape.points.detach().cpu().numpy()
            num_ctrl = shape.num_control_points.detach().cpu().numpy()
            if len(points) == 0: continue

            d_parts = [f'M {points[0][0]:.4f} {points[0][1]:.4f}']
            pt_idx = 1
            for seg_type in num_ctrl:
                if seg_type == 0 and pt_idx < len(points):
                    d_parts.append(f'L {points[pt_idx][0]:.4f} {points[pt_idx][1]:.4f}')
                    pt_idx += 1
                elif seg_type == 1 and pt_idx + 1 < len(points):
                    d_parts.append(f'Q {points[pt_idx][0]:.4f} {points[pt_idx][1]:.4f} '
                                  f'{points[pt_idx+1][0]:.4f} {points[pt_idx+1][1]:.4f}')
                    pt_idx += 2
                elif seg_type == 2 and pt_idx + 2 < len(points):
                    d_parts.append(f'C {points[pt_idx][0]:.4f} {points[pt_idx][1]:.4f} '
                                  f'{points[pt_idx+1][0]:.4f} {points[pt_idx+1][1]:.4f} '
                                  f'{points[pt_idx+2][0]:.4f} {points[pt_idx+2][1]:.4f}')
                    pt_idx += 3
            if shape.is_closed:
                d_parts.append('Z')

            path_elem = ET.SubElement(svg, 'path')
            path_elem.set('d', ' '.join(d_parts))
            if group.fill_color is not None:
                c = group.fill_color.detach().cpu().numpy()
                path_elem.set('fill', f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})')
                if c[3] < 1.0:
                    path_elem.set('fill-opacity', f'{c[3]:.4f}')
            else:
                path_elem.set('fill', 'none')

    tree = ET.ElementTree(svg)
    ET.indent(tree, space='  ')
    tree.write(path, encoding='unicode', xml_declaration=True)
