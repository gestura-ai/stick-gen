# Cinematic Rendering (2.5D)

This document describes the cinematic rendering system in Stick-Gen, which adds depth perception and 3D-like effects to 2D stick figure animations.

## Overview

The `CinematicRenderer` class (`src/data_gen/renderer.py`) extends the base `StickFigure` renderer with "2.5D" features that simulate 3D depth without requiring full 3D meshes or complex geometry.

## Features

### 1. Perspective Projection
- Simulates depth through foreshortening
- Objects closer to camera appear larger
- Creates natural depth cues

### 2. Dynamic Line Width
- Line thickness varies based on Z-depth
- Closer limbs are rendered thicker
- Adds visual weight and depth perception

### 3. Z-Sorting (Painter's Algorithm)
- Limbs are drawn in depth order
- Farthest elements drawn first
- Solves occlusion (limbs crossing)

## Technical Implementation

### Perspective Projection Math

The projection uses a weak perspective model:

```python
def project_3d_to_2d(self, x, y, z, camera_zoom=1.0):
    """
    Project 3D point to 2D with perspective.
    Formula: x' = x * (f / (f + z))
    """
    focal_length = 10.0  # Virtual focal length
    depth = max(focal_length + z, 0.1)  # Avoid division by zero
    scale = (focal_length / depth) * camera_zoom
    
    x_proj = x * scale
    y_proj = y * scale
    
    return x_proj, y_proj, scale
```

### Z-Depth Assignments

Each limb is assigned a default Z-depth:

| Limb Index | Body Part | Z-Depth | Notes |
|------------|-----------|---------|-------|
| 0 | Torso | 0.0 | Center reference |
| 1 | Left Leg | -0.2 | Slightly behind |
| 2 | Right Leg | +0.2 | Slightly in front |
| 3 | Left Arm | -0.3 | Further behind |
| 4 | Right Arm | +0.3 | Further in front |

**Positive Z = Closer to Camera**

```python
# Z-depth definitions
z_depths = [0.0, -0.2, 0.2, -0.3, 0.3]  # Torso, L-Leg, R-Leg, L-Arm, R-Arm
```

### Dynamic Line Width

Line width scales with distance:

```python
# Camera at Z=-10, looking at origin
camera_z = -10.0
dist = z - camera_z  # Distance from camera to point

# Scale factor: closer = larger
scale = 10.0 / dist

# Line width proportional to scale (base width = 2.0)
width = 2.0 * scale
```

**Examples:**
- Point at Z=0: `dist=10`, `scale=1.0`, `width=2.0`
- Point at Z=2: `dist=8`, `scale=1.25`, `width=2.5` (closer, thicker)
- Point at Z=-2: `dist=12`, `scale=0.83`, `width=1.67` (farther, thinner)

### Z-Sorting (Painter's Algorithm)

Elements are sorted by Z before drawing:

```python
# Sort by Z (ascending = farthest first)
cinematic_lines.sort(key=lambda x: x[3])

# Result: draw order ensures proper occlusion
# Farthest limbs drawn first, closest limbs drawn last (on top)
```

## Enabling Cinematic Mode

### Via Python API

```python
from src.data_gen.renderer import Renderer

renderer = Renderer()

# Enable cinematic rendering
renderer.render_scene(
    scene=my_scene,
    output_path="cinematic_output.mp4",
    cinematic=True  # Enable 2.5D effects
)
```

### Via Generator Class

```python
from src.inference.generator import AnimationGenerator

generator = AnimationGenerator()

# Standard rendering (2D)
generator.generate(prompt, "standard.mp4")

# Cinematic rendering (2.5D)  
# Note: Add cinematic parameter to generate() method
```

### Internal Switching

When `cinematic=True`, the renderer uses `CinematicRenderer` instead of `StickFigure`:

```python
def render_scene(self, scene, output_path, camera_mode="static", cinematic=False):
    if cinematic:
        actors = [CinematicRenderer(a) for a in scene.actors]
    else:
        actors = [StickFigure(a) for a in scene.actors]
```

## CinematicRenderer Class

```python
class CinematicRenderer(StickFigure):
    """
    Advanced renderer with '2.5D' features:
    - Perspective projection (foreshortening)
    - Dynamic line width (depth cue)
    - Z-sorting (occlusion)
    """
    
    def get_pose(self, t: float, dt: float = 0.04):
        # Get base 2D pose from parent
        base_lines, head_center = super().get_pose(t, dt)
        
        # Apply perspective projection to each limb
        cinematic_lines = []
        for i, (start, end) in enumerate(base_lines):
            z = z_depths[i]
            
            # Project points
            start_proj = start * scale
            end_proj = end * scale
            
            # Calculate depth-based line width
            width = 2.0 * scale
            
            cinematic_lines.append((start_proj, end_proj, width, z))
        
        # Sort by Z (painter's algorithm)
        cinematic_lines.sort(key=lambda x: x[3])
        
        return cinematic_lines, head_proj
```

## Visual Comparison

### Standard Rendering (2D)
- All limbs have same line width
- No depth ordering
- Flat appearance

### Cinematic Rendering (2.5D)
- Variable line widths based on depth
- Proper occlusion handling
- Natural depth perception
- Subtle foreshortening effects

## Performance Considerations

- **Overhead**: ~5-10% additional render time
- **Memory**: Minimal (stores Z values per limb)
- **GPU**: Not required (CPU rendering)

## Future Enhancements

1. **Action-based Z-offsets**: Vary Z-depths during different actions
2. **Camera integration**: Link camera Z-position to projection
3. **Shadow rendering**: Add simple drop shadows
4. **Motion blur**: Blur based on Z-velocity
5. **Depth of field**: Blur distant/close elements

## Troubleshooting

### Lines appear too thick/thin
Adjust the base width or focal length:
```python
width = 1.5 * scale  # Thinner base
focal_length = 15.0  # Less dramatic perspective
```

### Occlusion looks wrong
Check Z-depth assignments match expected limb order.

### Projection distorts figure
Increase focal length for less dramatic perspective:
```python
focal_length = 20.0  # More orthographic-like
```

