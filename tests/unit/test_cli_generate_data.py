import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.cli.main import main_generate_data  # noqa: E402


def test_generate_data_parallax_args_parsed_and_forwarded(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("training: {}\ndata: {}\n", encoding="utf-8")

    with mock.patch("src.cli.main.generate_dataset") as mock_gen, mock.patch(
        "src.cli.main.generate_parallax_for_dataset"
    ) as mock_parallax:
        mock_gen.return_value = str(tmp_path / "train.pt")
        argv = [
            "--config",
            str(cfg_path),
            "--augment-parallax",
            "--views-per-motion",
            "10",
            "--frames-per-view",
            "3",
            "--output",
            str(tmp_path / "out"),
        ]
        main_generate_data(argv)

    mock_gen.assert_called_once()
    mock_parallax.assert_called_once()
    kwargs = mock_parallax.call_args.kwargs

    assert kwargs["dataset_path"] == str(tmp_path / "train.pt")
    assert kwargs["output_dir"] == str(tmp_path / "out")
    assert kwargs["views_per_motion"] == 10
    assert kwargs["frames_per_view"] == 3
