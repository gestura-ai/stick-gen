"""
Tests for Camera System

Tests:
- CameraState dataclass
- Camera movement types (Static, Pan, Zoom, Track)
- Smooth interpolation functions
- View limits calculation
- Camera update loop
"""

import sys

sys.path.insert(0, "/Users/bc/gestura/stick-gen")


from src.data_gen.camera import (
    Camera,
    CameraState,
    Pan,
    StaticCamera,
    Zoom,
)


class TestCameraState:
    """Tests for CameraState dataclass"""

    def test_camera_state_defaults(self):
        """Test default values"""
        state = CameraState(x=0.0, y=0.0, zoom=1.0)
        assert state.x == 0.0
        assert state.y == 0.0
        assert state.zoom == 1.0
        assert state.rotation == 0.0  # Default
        print("✓ CameraState defaults test passed")

    def test_camera_state_custom_values(self):
        """Test custom values including rotation"""
        state = CameraState(x=2.5, y=-1.0, zoom=1.5, rotation=0.5)
        assert state.x == 2.5
        assert state.y == -1.0
        assert state.zoom == 1.5
        assert state.rotation == 0.5
        print("✓ CameraState custom values test passed")


class TestStaticCamera:
    """Tests for StaticCamera"""

    def test_static_camera_returns_constant_state(self):
        """Static camera should return same state for all time values"""
        cam = StaticCamera(x=1.0, y=2.0, zoom=1.5)

        state0 = cam.get_state(0.0)
        state1 = cam.get_state(1.0)
        state10 = cam.get_state(10.0)

        assert state0.x == state1.x == state10.x == 1.0
        assert state0.y == state1.y == state10.y == 2.0
        assert state0.zoom == state1.zoom == state10.zoom == 1.5
        print("✓ StaticCamera constant state test passed")


class TestPan:
    """Tests for Pan movement"""

    def test_pan_before_start_time(self):
        """Pan should return None before start time"""
        pan = Pan(
            start_pos=(0.0, 0.0), end_pos=(10.0, 0.0), start_time=2.0, duration=3.0
        )
        assert pan.get_state(0.0) is None
        assert pan.get_state(1.9) is None
        print("✓ Pan before start time test passed")

    def test_pan_at_start(self):
        """Pan should return start position at start time"""
        pan = Pan(
            start_pos=(0.0, 0.0), end_pos=(10.0, 0.0), start_time=0.0, duration=2.0
        )
        state = pan.get_state(0.0)
        assert state is not None
        assert abs(state.x - 0.0) < 0.01
        print("✓ Pan at start test passed")

    def test_pan_at_end(self):
        """Pan should return end position after duration"""
        pan = Pan(
            start_pos=(0.0, 0.0), end_pos=(10.0, 5.0), start_time=0.0, duration=2.0
        )
        state = pan.get_state(2.0)
        assert state is not None
        assert abs(state.x - 10.0) < 0.01
        assert abs(state.y - 5.0) < 0.01
        print("✓ Pan at end test passed")

    def test_pan_smooth_interpolation(self):
        """Pan should use smooth step interpolation"""
        pan = Pan(
            start_pos=(0.0, 0.0), end_pos=(10.0, 0.0), start_time=0.0, duration=2.0
        )
        # At halfway point with smoothstep, progress should be 0.5 * 0.5 * (3 - 2*0.5) = 0.5
        state = pan.get_state(1.0)
        assert state is not None
        # Smoothstep at t=0.5 gives 0.5, so x should be 5.0
        assert abs(state.x - 5.0) < 0.01
        print("✓ Pan smooth interpolation test passed")


class TestZoom:
    """Tests for Zoom movement"""

    def test_zoom_before_start(self):
        """Zoom should return None before start time"""
        zoom = Zoom(
            center=(0.0, 0.0),
            start_zoom=1.0,
            end_zoom=2.0,
            start_time=1.0,
            duration=2.0,
        )
        assert zoom.get_state(0.5) is None
        print("✓ Zoom before start test passed")

    def test_zoom_at_start(self):
        """Zoom should return start zoom at start time"""
        zoom = Zoom(
            center=(0.0, 0.0),
            start_zoom=1.0,
            end_zoom=2.0,
            start_time=0.0,
            duration=2.0,
        )
        state = zoom.get_state(0.0)
        assert state is not None
        assert abs(state.zoom - 1.0) < 0.01
        print("✓ Zoom at start test passed")

    def test_zoom_at_end(self):
        """Zoom should return end zoom after duration"""
        zoom = Zoom(
            center=(0.0, 0.0),
            start_zoom=1.0,
            end_zoom=3.0,
            start_time=0.0,
            duration=1.0,
        )
        state = zoom.get_state(1.0)
        assert state is not None
        assert abs(state.zoom - 3.0) < 0.01
        print("✓ Zoom at end test passed")

    def test_zoom_maintains_center(self):
        """Zoom should maintain center position"""
        zoom = Zoom(
            center=(2.5, -1.0),
            start_zoom=1.0,
            end_zoom=2.0,
            start_time=0.0,
            duration=1.0,
        )
        state = zoom.get_state(0.5)
        assert state is not None
        assert state.x == 2.5
        assert state.y == -1.0
        print("✓ Zoom maintains center test passed")


class TestCamera:
    """Tests for Camera controller class"""

    def test_camera_initialization(self):
        """Test camera default initialization"""
        camera = Camera()
        assert camera.base_width == 10.0
        assert camera.base_height == 10.0
        assert camera.state.x == 0.0
        assert camera.state.y == 0.0
        assert camera.state.zoom == 1.0
        print("✓ Camera initialization test passed")

    def test_camera_custom_dimensions(self):
        """Test camera with custom dimensions"""
        camera = Camera(width=20.0, height=15.0)
        assert camera.base_width == 20.0
        assert camera.base_height == 15.0
        print("✓ Camera custom dimensions test passed")

    def test_add_movement(self):
        """Test adding movements to camera"""
        camera = Camera()
        pan = Pan((0, 0), (10, 0), 0, 2)
        camera.add_movement(pan)
        assert len(camera.movements) == 1
        print("✓ Camera add movement test passed")

    def test_track_actor(self):
        """Test setting actor tracking"""
        camera = Camera()
        camera.track_actor("actor_0")
        assert camera.target_actor_id == "actor_0"
        print("✓ Camera track actor test passed")

    def test_view_limits_default(self):
        """Test view limits with default zoom"""
        camera = Camera(width=10.0, height=10.0)
        camera.state = CameraState(x=0.0, y=0.0, zoom=1.0)

        xmin, xmax, ymin, ymax = camera.get_view_limits()
        assert xmin == -5.0
        assert xmax == 5.0
        assert ymin == -5.0
        assert ymax == 5.0
        print("✓ Camera view limits default test passed")

    def test_view_limits_zoomed_in(self):
        """Test view limits when zoomed in (smaller visible area)"""
        camera = Camera(width=10.0, height=10.0)
        camera.state = CameraState(x=0.0, y=0.0, zoom=2.0)

        xmin, xmax, ymin, ymax = camera.get_view_limits()
        # Zoom 2.0 = half the area visible
        assert xmin == -2.5
        assert xmax == 2.5
        assert ymin == -2.5
        assert ymax == 2.5
        print("✓ Camera view limits zoomed in test passed")

    def test_view_limits_with_offset(self):
        """Test view limits with camera offset"""
        camera = Camera(width=10.0, height=10.0)
        camera.state = CameraState(x=5.0, y=3.0, zoom=1.0)

        xmin, xmax, ymin, ymax = camera.get_view_limits()
        assert xmin == 0.0
        assert xmax == 10.0
        assert ymin == -2.0
        assert ymax == 8.0
        print("✓ Camera view limits with offset test passed")

    def test_camera_update_with_movement(self):
        """Test camera update applies active movements"""
        camera = Camera()
        pan = Pan((0, 0), (10, 0), 0, 2.0, zoom=1.0)
        camera.add_movement(pan)

        # Update at midpoint
        camera.update(1.0)

        # Should have moved to approximately middle
        assert camera.state.x > 4.0
        assert camera.state.x < 6.0
        print("✓ Camera update with movement test passed")


# Run tests if executed directly
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Running Camera System Tests")
    print("=" * 50 + "\n")

    # CameraState tests
    test_state = TestCameraState()
    test_state.test_camera_state_defaults()
    test_state.test_camera_state_custom_values()

    # StaticCamera tests
    test_static = TestStaticCamera()
    test_static.test_static_camera_returns_constant_state()

    # Pan tests
    test_pan = TestPan()
    test_pan.test_pan_before_start_time()
    test_pan.test_pan_at_start()
    test_pan.test_pan_at_end()
    test_pan.test_pan_smooth_interpolation()

    # Zoom tests
    test_zoom = TestZoom()
    test_zoom.test_zoom_before_start()
    test_zoom.test_zoom_at_start()
    test_zoom.test_zoom_at_end()
    test_zoom.test_zoom_maintains_center()

    # Camera tests
    test_camera = TestCamera()
    test_camera.test_camera_initialization()
    test_camera.test_camera_custom_dimensions()
    test_camera.test_add_movement()
    test_camera.test_track_actor()
    test_camera.test_view_limits_default()
    test_camera.test_view_limits_zoomed_in()
    test_camera.test_view_limits_with_offset()
    test_camera.test_camera_update_with_movement()

    print("\n" + "=" * 50)
    print("All Camera System Tests Passed! ✓")
    print("=" * 50)
