"""Manual facial expression rendering helper.

This file is intentionally *not* a pytest test. It provides a small
CLI-style helper to render short MP4 clips for each facial expression so
you can visually inspect them. Pytest will still import this module when
running the features suite, but there are no test functions here.
"""

from src.data_gen.schema import FacialExpression, ActionType, Position, Scene
from src.data_gen.renderer import Renderer
from src.data_gen.story_engine import create_actor_with_expression


def render_expression(
    expression: FacialExpression,
    action: ActionType,
    output_file: str,
) -> None:
    """Render a single facial expression to an MP4 file.

    Note:
        This is a manual visual check helper, not a pytest test function.
    """

    print(f"Testing {expression.value} expression with {action.value} action...")

    # Create actor with specific expression
    actor = create_actor_with_expression(
        actor_id="test_actor",
        position=Position(x=0, y=0),
        actions=[(0.0, action)],
        color="black",
    )

    # Create simple scene
    scene = Scene(
        duration=3.0,
        actors=[actor],
        objects=[],
        background_color="white",
        description=f"Testing {expression.value} expression",
    )

    # Render
    renderer = Renderer(width=640, height=480)
    renderer.render_scene(scene, output_file)
    print(f"✓ Saved to {output_file}")


def main() -> None:
    """Render a short clip for each of the six facial expressions."""

    print("=" * 60)
    print("FACIAL EXPRESSIONS MANUAL CHECK")
    print("=" * 60)
    print()

    tests = [
        (FacialExpression.NEUTRAL, ActionType.IDLE, "test_expression_neutral.mp4"),
        (FacialExpression.HAPPY, ActionType.WAVE, "test_expression_happy.mp4"),
        (FacialExpression.SAD, ActionType.CRY, "test_expression_sad.mp4"),
        (
            FacialExpression.SURPRISED,
            ActionType.LOOKING_AROUND,
            "test_expression_surprised.mp4",
        ),
        (FacialExpression.ANGRY, ActionType.PUNCH, "test_expression_angry.mp4"),
        (FacialExpression.EXCITED, ActionType.CELEBRATE, "test_expression_excited.mp4"),
    ]

    for expression, action, output_file in tests:
        try:
            render_expression(expression, action, output_file)
        except Exception as exc:  # pragma: no cover - manual helper
            print(f"✗ Error testing {expression.value}: {exc}")

    print()
    print("=" * 60)
    print("Done. Check the generated MP4 files.")
    print("=" * 60)


if __name__ == "__main__":  # pragma: no cover
    main()
