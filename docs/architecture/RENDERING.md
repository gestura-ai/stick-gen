# High-Performance Rendering Pipeline

## Overview

Stick-Gen uses a **Decoupled Rendering Architecture**. Instead of baking video files on the server (which is slow and non-interactive), the model exports a lightweight Motion Schema (`.motion` JSON) that is rendered client-side.

This approach enables:
- **60+ FPS Rendering**: Using client GPU via WebGL/Three.js
- **Interactivity**: User can rotate, zoom, and scrub the timeline
- **Visual Fidelity**: Neon glows, bloom effects, shadows, and 3D environments
- **Low Bandwidth**: JSON files are ~100x smaller than MP4s

## Architecture

```mermaid
graph LR
    A[Model Inference<br/>Transformer + Diffusion] -->|Pose + Actions + Physics| B[Raw Tensors]
    B -->|MotionExporter| C[".motion JSON<br/>(~10-50KB)"]
    C -->|Network/File| D[Client Application]
    D -->|Three.js| E[Web Renderer<br/>60fps WebGL]
    D -->|PyGame| F[Local Debugger<br/>1000+ fps]

    subgraph SERVER["Server (RunPod/Local)"]
        A
        B
        C
    end

    subgraph CLIENT["Client (Browser/Desktop)"]
        D
        E
        F
    end
```

## Motion Schema (`.motion`)

The schema is a minified JSON object designed for efficient parsing.

### Structure
```json
{
  "meta": {
    "version": "1.0",
    "fps": 25,
    "total_frames": 250,
    "description": "A stick figure jumping"
  },
  "skeleton": {
    "type": "stick_figure_5_segment",
    "joints": ["Torso", "L_Leg", "R_Leg", "L_Arm", "R_Arm"]
  },
  "motion": [ ...flat float array... ],
  "actions": ["idle", "jump", "jump", ...]
}
```

### Data Layout
The `motion` array is a flattened float32 array.
- **Stride**: `input_dim` (default 20)
- **Indexing**: `frame_i = motion[i * stride : (i+1) * stride]`
- **Coordinates**: xy_lines format (x1, y1, x2, y2) per segment.

## Renderers

### Web (Three.js / React-Three-Fiber)
For the web frontend, use the `StickRefinery` component (external library) which consumes this schema.
- **Lighting**: UnrealBloomPass for neon aesthetics.
- **Geometry**: `LineSegments` or `MeshLine` for thick, glowing strokes.

### Local (PyGame)
For fast training debugging locally.
- **Path**: `src/vis/pygame_viewer.py` (Planned)
- **Performance**: >1000 FPS draw rate.
