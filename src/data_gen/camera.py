import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class CameraState:
    x: float
    y: float
    zoom: float
    rotation: float = 0.0

class CameraMovement:
    """Base class for camera movements"""
    def get_state(self, t: float) -> Optional[CameraState]:
        raise NotImplementedError

class StaticCamera(CameraMovement):
    def __init__(self, x: float, y: float, zoom: float = 1.0):
        self.state = CameraState(x, y, zoom)

    def get_state(self, t: float) -> Optional[CameraState]:
        return self.state

class Pan(CameraMovement):
    def __init__(self, start_pos: Tuple[float, float], end_pos: Tuple[float, float], 
                 start_time: float, duration: float, zoom: float = 1.0):
        self.start_pos = np.array(start_pos)
        self.end_pos = np.array(end_pos)
        self.start_time = start_time
        self.duration = duration
        self.zoom = zoom

    def get_state(self, t: float) -> Optional[CameraState]:
        if t < self.start_time:
            return None
        
        progress = (t - self.start_time) / self.duration
        if progress > 1.0:
            progress = 1.0
        
        # Smooth step interpolation
        progress = progress * progress * (3 - 2 * progress)
        
        current_pos = self.start_pos + (self.end_pos - self.start_pos) * progress
        return CameraState(current_pos[0], current_pos[1], self.zoom)

class Zoom(CameraMovement):
    def __init__(self, center: Tuple[float, float], start_zoom: float, end_zoom: float,
                 start_time: float, duration: float):
        self.center = center
        self.start_zoom = start_zoom
        self.end_zoom = end_zoom
        self.start_time = start_time
        self.duration = duration

    def get_state(self, t: float) -> Optional[CameraState]:
        if t < self.start_time:
            return None
            
        progress = (t - self.start_time) / self.duration
        if progress > 1.0:
            progress = 1.0
            
        # Smooth step
        progress = progress * progress * (3 - 2 * progress)
        
        current_zoom = self.start_zoom + (self.end_zoom - self.start_zoom) * progress
        return CameraState(self.center[0], self.center[1], current_zoom)

class TrackingCamera(CameraMovement):
    def __init__(self, actor_id: str, scene_actors: List, zoom: float = 1.5, smooth_factor: float = 0.1):
        self.actor_id = actor_id
        self.actors = {a.id: a for a in scene_actors}
        self.zoom = zoom
        self.smooth_factor = smooth_factor
        self.last_pos = None

    def get_state(self, t: float) -> Optional[CameraState]:
        actor = self.actors.get(self.actor_id)
        if not actor:
            return None

        # Get actor position (need to access the StickFigure's current position ideally,
        # but here we might need to calculate it from the actor data if we don't have the live object)
        # For now, we'll assume we can get it from the actor's movement path or initial pos
        # In the renderer loop, we'll pass the live StickFigure object ideally.
        # But to keep this decoupled, let's rely on the renderer to update the camera with the target's position.
        # So this class might be a configuration that the renderer uses logic to apply.

        # Actually, let's make the Camera class manage the state and the Renderer update it.
        return None # Logic handled in Camera controller


class Dolly(CameraMovement):
    """
    Dolly movement: Camera moves toward or away from subject (zoom via position).
    Unlike Zoom which scales the view, Dolly physically moves the camera,
    creating parallax effects in 3D scenes.

    In 2.5D, we simulate this by combining position change with zoom.
    """
    def __init__(self, center: Tuple[float, float], start_distance: float, end_distance: float,
                 start_time: float, duration: float, base_zoom: float = 1.0):
        self.center = np.array(center)
        self.start_distance = start_distance
        self.end_distance = end_distance
        self.start_time = start_time
        self.duration = duration
        self.base_zoom = base_zoom

    def get_state(self, t: float) -> Optional[CameraState]:
        if t < self.start_time:
            return None

        progress = min((t - self.start_time) / self.duration, 1.0)
        # Smooth step interpolation
        progress = progress * progress * (3 - 2 * progress)

        # Interpolate distance
        current_distance = self.start_distance + (self.end_distance - self.start_distance) * progress

        # Zoom is inversely proportional to distance (closer = more zoom)
        # Normalize so start_distance gives base_zoom
        zoom = self.base_zoom * (self.start_distance / max(current_distance, 0.1))

        return CameraState(self.center[0], self.center[1], zoom)


class Crane(CameraMovement):
    """
    Crane movement: Camera moves vertically (up/down) while maintaining horizontal position.
    Often used for dramatic reveals or establishing shots.
    """
    def __init__(self, x: float, start_y: float, end_y: float,
                 start_time: float, duration: float, zoom: float = 1.0):
        self.x = x
        self.start_y = start_y
        self.end_y = end_y
        self.start_time = start_time
        self.duration = duration
        self.zoom = zoom

    def get_state(self, t: float) -> Optional[CameraState]:
        if t < self.start_time:
            return None

        progress = min((t - self.start_time) / self.duration, 1.0)
        # Smooth step interpolation
        progress = progress * progress * (3 - 2 * progress)

        current_y = self.start_y + (self.end_y - self.start_y) * progress

        return CameraState(self.x, current_y, self.zoom)


class Orbit(CameraMovement):
    """
    Orbit movement: Camera rotates around a center point at a fixed radius.
    Creates a circular path around the subject.

    In 2D, this translates to circular motion of the camera position.
    """
    def __init__(self, center: Tuple[float, float], radius: float,
                 start_angle: float, end_angle: float,
                 start_time: float, duration: float, zoom: float = 1.0):
        """
        Args:
            center: Point to orbit around (x, y)
            radius: Distance from center
            start_angle: Starting angle in degrees (0 = right, 90 = up)
            end_angle: Ending angle in degrees
            start_time: When orbit begins
            duration: How long the orbit takes
            zoom: Camera zoom level
        """
        self.center = np.array(center)
        self.radius = radius
        self.start_angle = np.radians(start_angle)
        self.end_angle = np.radians(end_angle)
        self.start_time = start_time
        self.duration = duration
        self.zoom = zoom

    def get_state(self, t: float) -> Optional[CameraState]:
        if t < self.start_time:
            return None

        progress = min((t - self.start_time) / self.duration, 1.0)
        # Smooth step interpolation
        progress = progress * progress * (3 - 2 * progress)

        # Interpolate angle
        current_angle = self.start_angle + (self.end_angle - self.start_angle) * progress

        # Calculate position on circle
        x = self.center[0] + self.radius * np.cos(current_angle)
        y = self.center[1] + self.radius * np.sin(current_angle)

        # Rotation to keep camera facing center
        rotation = np.degrees(current_angle) + 180  # Face toward center

        return CameraState(x, y, self.zoom, rotation)

class Camera:
    def __init__(self, width: float = 10.0, height: float = 10.0):
        self.base_width = width
        self.base_height = height
        self.state = CameraState(0, 0, 1.0)
        self.movements: List[CameraMovement] = []
        self.target_actor_id: Optional[str] = None
        
    def add_movement(self, movement: CameraMovement):
        self.movements.append(movement)
        
    def track_actor(self, actor_id: str):
        self.target_actor_id = actor_id
        
    def update(self, t: float, actors_dict: dict = None):
        # 1. Check for active movements
        active_movement = None
        for move in self.movements:
            state = move.get_state(t)
            if state:
                self.state = state
                active_movement = move
                # We keep going to find the latest applicable one or blend them? 
                # For simplicity, later movements override earlier ones if they overlap
        
        # 2. If tracking an actor and no override movement
        if self.target_actor_id and actors_dict and self.target_actor_id in actors_dict:
            actor = actors_dict[self.target_actor_id]
            # Smooth follow
            target_x, target_y = actor.pos
            
            # Simple lerp for smoothness
            smooth = 0.1
            self.state.x += (target_x - self.state.x) * smooth
            self.state.y += (target_y - self.state.y) * smooth
            
            # Ensure zoom is set for tracking if not already
            if self.state.zoom == 1.0:
                self.state.zoom = 1.5

    def get_view_limits(self) -> Tuple[float, float, float, float]:
        """Return (xmin, xmax, ymin, ymax)"""
        # Calculate visible width/height based on zoom
        # Zoom 2.0 means we see half the area (magnified)
        visible_w = self.base_width / self.state.zoom
        visible_h = self.base_height / self.state.zoom
        
        xmin = self.state.x - visible_w / 2
        xmax = self.state.x + visible_w / 2
        ymin = self.state.y - visible_h / 2
        ymax = self.state.y + visible_h / 2
        
        return (xmin, xmax, ymin, ymax)
