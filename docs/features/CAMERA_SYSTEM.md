# Camera System

This document describes the camera system in Stick-Gen, which provides dynamic camera movements and cinematic effects for animations.

## Overview

The camera system (`src/data_gen/camera.py`) enables sophisticated camera behaviors including panning, zooming, and actor tracking. Camera data is stored as keyframes and interpolated during rendering.

## Core Components

### CameraState

The fundamental camera state representation:

```python
@dataclass
class CameraState:
    x: float        # Camera center X position
    y: float        # Camera center Y position  
    zoom: float     # Zoom level (1.0 = normal, 2.0 = 2x magnified)
    rotation: float = 0.0  # Rotation in radians (future use)
```

### CameraKeyframe Schema

Keyframes define camera state at specific frames:

```python
class CameraKeyframe(BaseModel):
    frame: int          # Frame number (0-indexed)
    x: float            # Camera X position
    y: float            # Camera Y position
    zoom: float = 1.0   # Zoom level
    interpolation: str = "linear"  # Interpolation type: "linear" or "smooth"
```

## Camera Movement Types

### 1. Static Camera

Fixed camera position throughout the scene.

```python
from src.data_gen.camera import StaticCamera

# Camera at origin with default zoom
camera = StaticCamera(x=0.0, y=0.0, zoom=1.0)

# Camera offset to the right
camera = StaticCamera(x=2.0, y=0.0, zoom=1.5)
```

### 2. Pan Movement

Smooth horizontal/vertical camera movement.

```python
from src.data_gen.camera import Pan

# Pan from left to right over 3 seconds
pan = Pan(
    start_pos=(-3.0, 0.0),  # Start position (x, y)
    end_pos=(3.0, 0.0),     # End position (x, y)
    start_time=0.0,         # When to start (seconds)
    duration=3.0,           # Duration (seconds)
    zoom=1.0                # Zoom level during pan
)
```

### 3. Zoom Movement

Smooth zoom in/out while maintaining center.

```python
from src.data_gen.camera import Zoom

# Zoom in on center point
zoom = Zoom(
    center=(0.0, 0.0),      # Zoom center point
    start_zoom=1.0,         # Initial zoom level
    end_zoom=2.0,           # Final zoom level
    start_time=2.0,         # Start at 2 seconds
    duration=1.5            # Complete over 1.5 seconds
)
```

### 4. Tracking Camera

Follow an actor's movement.

```python
from src.data_gen.camera import TrackingCamera

# Track actor with ID "actor_0"
tracker = TrackingCamera(
    actor_id="actor_0",     # ID of actor to follow
    scene_actors=actors,    # List of all actors
    zoom=1.5,               # Zoom level (closer for tracking)
    smooth_factor=0.1       # Smoothing (0.0-1.0, lower = smoother)
)
```

## Defining Camera Keyframes in Scenes

### In Python

```python
from src.data_gen.schema import Scene, CameraKeyframe

scene = Scene(
    duration=10.0,
    actors=my_actors,
    description="A dramatic scene",
    camera_keyframes=[
        CameraKeyframe(frame=0, x=0.0, y=0.0, zoom=1.0),
        CameraKeyframe(frame=50, x=2.0, y=0.0, zoom=1.2, interpolation="smooth"),
        CameraKeyframe(frame=150, x=0.0, y=1.0, zoom=1.5),
        CameraKeyframe(frame=250, x=0.0, y=0.0, zoom=1.0)
    ]
)
```

### In JSON (for LLM-generated scripts)

```json
{
  "camera_keyframes": [
    {"frame": 0, "x": 0.0, "y": 0.0, "zoom": 1.0, "interpolation": "linear"},
    {"frame": 100, "x": 2.0, "y": 1.0, "zoom": 1.5, "interpolation": "smooth"}
  ]
}
```

## Interpolation Types

### Linear Interpolation
Direct interpolation between keyframes. Good for mechanical movements.

```python
CameraKeyframe(frame=0, x=0.0, y=0.0, interpolation="linear")
```

### Smooth Interpolation  
Uses smoothstep function for natural ease-in/ease-out:

```python
# Smoothstep formula: t² × (3 - 2t)
progress = progress * progress * (3 - 2 * progress)
```

```python
CameraKeyframe(frame=100, x=5.0, y=0.0, interpolation="smooth")
```

## Integration with Rendering

### Basic Usage

```python
from src.data_gen.renderer import Renderer

renderer = Renderer()

# Render with static camera
renderer.render_scene(scene, "output.mp4", camera_mode="static")

# Render with dynamic camera (follows first actor)
renderer.render_scene(scene, "output.mp4", camera_mode="dynamic")
```

### Camera Update Loop

The `Camera.update()` method is called each frame:

```python
def update(self, t: float, actors_dict: dict = None):
    # 1. Apply active movements (Pan, Zoom)
    for move in self.movements:
        state = move.get_state(t)
        if state:
            self.state = state
    
    # 2. If tracking an actor, smoothly follow
    if self.target_actor_id and actors_dict:
        actor = actors_dict[self.target_actor_id]
        # Smooth lerp to actor position
        self.state.x += (actor.pos[0] - self.state.x) * 0.1
        self.state.y += (actor.pos[1] - self.state.y) * 0.1
```

### View Limits Calculation

```python
camera = Camera(width=10.0, height=10.0)
camera.state = CameraState(x=2.0, y=1.0, zoom=2.0)

# Get visible area (zoom 2.0 = half the base area visible)
xmin, xmax, ymin, ymax = camera.get_view_limits()
# Returns: (-0.5, 4.5, -1.5, 3.5)
```

## Camera Data in Training

Camera keyframes are converted to tensors for model training:

```python
# Shape: [num_frames, 3] where columns are (x, y, zoom)
camera_tensor = torch.zeros(250, 3)
camera_tensor[:, 2] = 1.0  # Default zoom

# Keyframes are interpolated to fill all frames
for i in range(len(keyframes) - 1):
    # Linear interpolation between keyframes
    ...
```

## Best Practices

1. **Start with static**: Ensure animation works before adding camera movement
2. **Subtle movements**: Large, fast camera movements can be disorienting
3. **Match action**: Camera should follow the narrative (zoom on important moments)
4. **Smooth tracking**: Use `smooth_factor=0.1` for natural following behavior
5. **Frame alignment**: Use frame numbers that align with action keyframes

