import tempfile
from pathlib import Path

from automlcli.commands import create_parser
from automlcli.commands.train import TrainCommand  # noqa: F401

FIXTURE_PATH = Path("tests/fixtures")


def test_train_command():
    config_path = FIXTURE_PATH / "configs" / "config.yml"

    with tempfile.TemporaryDirectory() as tempdir:
        output_dir = Path(tempdir) / "output"

        parser = create_parser()
        args = parser.parse_args(
            [
                "train",
                str(config_path),
                "--overrides",
                "model.target_column=target",
                "--serialization-dir",
                str(output_dir),
            ]
        )

        args.func(args)

        assert output_dir.is_dir()
        assert (output_dir / "metrics.json").is_file()
        assert (output_dir / "model.pkl").is_file()


def test_train_command_with_command_args():
    config_path = FIXTURE_PATH / "configs" / "config.yml"
    train_file = FIXTURE_PATH / "data" / "train.csv"
    validation_file = FIXTURE_PATH / "data" / "dev.csv"

    with tempfile.TemporaryDirectory() as tempdir:
        output_dir = Path(tempdir) / "output"

        parser = create_parser()
        args = parser.parse_args(
            [
                "train",
                str(config_path),
                str(train_file),
                "--validation",
                str(validation_file),
                "--overrides",
                "model.target_column=target",
                "--serialization-dir",
                str(output_dir),
            ]
        )

        args.func(args)

        assert output_dir.is_dir()
        assert (output_dir / "metrics.json").is_file()
        assert (output_dir / "model.pkl").is_file()
