import tempfile
from pathlib import Path

from automlcli.commands import create_parser
from automlcli.commands.train import TrainCommand

FIXTURE_PATH = Path("tests/fixtures")


def test_train_command():
    config_path = FIXTURE_PATH / "configs" / "config.yml"
    train_path = FIXTURE_PATH / "data" / "train.csv"
    validation_path = FIXTURE_PATH / "data" / "dev.csv"

    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)
        model_path = tempdir / "model.pkl"

        parser = create_parser()
        args = parser.parse_args([
            "train",
            str(config_path),
            str(train_path),
            str(model_path),
            "model.target_column=target",
            "--validation",
            str(validation_path),
        ])

        args.func(args)

        assert model_path.is_file()
