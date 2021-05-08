import tempfile
from pathlib import Path

from automlcli.commands import create_parser
from automlcli.commands.retrain import RetrainCommand  # noqa: F401

FIXTURE_PATH = Path("tests/fixtures")


def test_train_command():
    model_path = FIXTURE_PATH / "data" / "model.pkl"
    train_path = FIXTURE_PATH / "data" / "train.csv"

    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)
        retrained_model_path = tempdir / "retrained_model.pkl"

        parser = create_parser()
        args = parser.parse_args(
            [
                "retrain",
                str(model_path),
                str(train_path),
                str(retrained_model_path),
            ]
        )

        args.func(args)

        assert retrained_model_path.is_file()
