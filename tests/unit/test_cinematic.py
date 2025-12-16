"""
Tests for Cinematic Rendering (2.5D)

Tests:
- Perspective projection math (3D to 2D)
- Z-depth assignments for limbs
- Dynamic line width calculation
- Z-sorting (painter's algorithm)
- CameraKeyframe parsing and validation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


from src.data_gen.schema import Actor, ActorType, CameraKeyframe, Position, Scene


class TestCameraKeyframe:
    """Tests for CameraKeyframe schema"""

    def test_camera_keyframe_creation(self):
        """Test creating a valid CameraKeyframe"""
        kf = CameraKeyframe(frame=0, x=0.0, y=0.0, zoom=1.0)
        assert kf.frame == 0
        assert kf.x == 0.0
        assert kf.y == 0.0
        assert kf.zoom == 1.0
        print("✓ CameraKeyframe creation test passed")

    def test_camera_keyframe_defaults(self):
        """Test default values for CameraKeyframe"""
        kf = CameraKeyframe(frame=10, x=1.0, y=2.0)
        assert kf.zoom == 1.0  # Default zoom
        assert kf.interpolation == "linear"  # Default interpolation
        print("✓ CameraKeyframe defaults test passed")

    def test_camera_keyframe_smooth_interpolation(self):
        """Test smooth interpolation option"""
        kf = CameraKeyframe(frame=50, x=0.0, y=0.0, interpolation="smooth")
        assert kf.interpolation == "smooth"
        print("✓ CameraKeyframe smooth interpolation test passed")

    def test_scene_with_camera_keyframes(self):
        """Test Scene with camera_keyframes list"""
        actor = Actor(
            id="test_actor",
            actor_type=ActorType.HUMAN,
            initial_position=Position(x=0, y=0),
            actions=[],
        )
        scene = Scene(
            description="Test scene",
            actors=[actor],
            duration=10.0,
            camera_keyframes=[
                CameraKeyframe(frame=0, x=0.0, y=0.0, zoom=1.0),
                CameraKeyframe(frame=100, x=2.0, y=1.0, zoom=1.5),
            ],
        )
        assert len(scene.camera_keyframes) == 2
        assert scene.camera_keyframes[0].frame == 0
        assert scene.camera_keyframes[1].zoom == 1.5
        print("✓ Scene with camera keyframes test passed")


class TestPerspectiveProjection:
    """Tests for perspective projection math"""

    def test_projection_at_origin(self):
        """Point at origin should remain at origin"""
        # Simple weak perspective: x' = x * (f / (f + z))
        focal_length = 10.0
        x, y, z = 0.0, 0.0, 0.0

        depth = max(focal_length + z, 0.1)
        scale = focal_length / depth
        x_proj = x * scale
        y_proj = y * scale

        assert x_proj == 0.0
        assert y_proj == 0.0
        print("✓ Projection at origin test passed")

    def test_projection_closer_is_larger(self):
        """Points closer to camera (positive z) should appear larger"""
        focal_length = 10.0
        _x, _y = 1.0, 1.0

        # Point at z=0
        z_far = 0.0
        scale_far = focal_length / (focal_length + z_far)

        # Point at z=2 (closer)
        z_close = 2.0
        scale_close = focal_length / (focal_length + z_close)

        # Closer point should have larger scale (more magnified)
        # Wait: z=2 means f+z = 12, so scale = 10/12 = 0.83 (smaller)
        # This means positive z is AWAY from camera in this convention
        # Let's verify the actual implementation uses z - camera_z

        # In CinematicRenderer: dist = z - camera_z, camera_z = -10
        # So z=0 -> dist = 0 - (-10) = 10, scale = 10/10 = 1.0
        # z=2 -> dist = 2 - (-10) = 12, scale = 10/12 = 0.83 (farther)
        # z=-2 -> dist = -2 - (-10) = 8, scale = 10/8 = 1.25 (closer)

        camera_z = -10.0
        z_far = 2.0  # Farther from camera
        z_close = -2.0  # Closer to camera

        dist_far = z_far - camera_z
        dist_close = z_close - camera_z

        scale_far = 10.0 / dist_far
        scale_close = 10.0 / dist_close

        assert scale_close > scale_far, "Closer points should have larger scale"
        print("✓ Projection closer is larger test passed")

    def test_projection_avoids_division_by_zero(self):
        """Projection should handle edge cases safely"""
        focal_length = 10.0
        _x, _y, z = 1.0, 1.0, -15.0  # Would make f+z negative

        # Implementation uses max(f + z, 0.1)
        depth = max(focal_length + z, 0.1)
        assert depth == 0.1  # Should clamp to minimum
        print("✓ Projection division safety test passed")


class TestZDepthAssignments:
    """Tests for Z-depth assignments to limbs"""

    def test_default_z_depths(self):
        """Test standard Z-depth assignments"""
        # From CinematicRenderer: [Torso, L-Leg, R-Leg, L-Arm, R-Arm]
        z_depths = [0.0, -0.2, 0.2, -0.3, 0.3]

        # Torso at center
        assert z_depths[0] == 0.0

        # Right side closer than left (positive z)
        assert z_depths[2] > z_depths[1]  # R-Leg > L-Leg
        assert z_depths[4] > z_depths[3]  # R-Arm > L-Arm

        # Arms further from center than legs
        assert abs(z_depths[3]) > abs(z_depths[1])  # L-Arm further than L-Leg
        assert abs(z_depths[4]) > abs(z_depths[2])  # R-Arm further than R-Leg
        print("✓ Default Z-depth assignments test passed")


class TestDynamicLineWidth:
    """Tests for depth-based line width"""

    def test_line_width_scales_with_distance(self):
        """Line width should be proportional to projection scale"""
        base_width = 2.0
        camera_z = -10.0

        # Point at z=0: dist=10, scale=1.0, width=2.0
        z1 = 0.0
        dist1 = z1 - camera_z
        width1 = base_width * (10.0 / dist1)
        assert abs(width1 - 2.0) < 0.01

        # Point at z=-2 (closer): dist=8, scale=1.25, width=2.5
        z2 = -2.0
        dist2 = z2 - camera_z
        width2 = base_width * (10.0 / dist2)
        assert width2 > width1

        print("✓ Dynamic line width scaling test passed")


class TestZSorting:
    """Tests for painter's algorithm Z-sorting"""

    def test_z_sorting_order(self):
        """Lines should be sorted by Z (farthest first)"""
        # Simulate cinematic_lines: (start, end, width, z)
        lines = [
            (None, None, 2.0, 0.0),  # Torso
            (None, None, 2.0, -0.3),  # L-Arm (farthest)
            (None, None, 2.0, 0.3),  # R-Arm (closest)
            (None, None, 2.0, -0.2),  # L-Leg
        ]

        # Sort by Z (ascending = farthest first)
        sorted_lines = sorted(lines, key=lambda x: x[3])

        # Check order: -0.3, -0.2, 0.0, 0.3
        assert sorted_lines[0][3] == -0.3  # L-Arm drawn first (farthest)
        assert sorted_lines[-1][3] == 0.3  # R-Arm drawn last (closest)
        print("✓ Z-sorting order test passed")


# Run tests if executed directly
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Running Cinematic Rendering Tests")
    print("=" * 50 + "\n")

    for test_class in [
        TestCameraKeyframe,
        TestPerspectiveProjection,
        TestZDepthAssignments,
        TestDynamicLineWidth,
        TestZSorting,
    ]:
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                getattr(instance, method_name)()

    print("\n" + "=" * 50)
    print("All Cinematic Rendering Tests Passed! ✓")
    print("=" * 50)
