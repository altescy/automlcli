import tempfile
from pathlib import Path

import pandas

from automlcli.commands import create_parser
from automlcli.commands.predict import PredictCommand  # noqa: F401

FIXTURE_PATH = Path("tests/fixtures")


def test_train_command() -> None:
    model_path = FIXTURE_PATH / "data" / "model.pkl"
    test_path = FIXTURE_PATH / "data" / "test.csv"

    with tempfile.TemporaryDirectory() as _tempdir:
        tempdir = Path(_tempdir)
        prediction_path = tempdir / "predictions.csv"

        parser = create_parser()
        args = parser.parse_args(
            [
                "predict",
                str(model_path),
                str(test_path),
                "--output-file",
                str(prediction_path),
            ]
        )

        args.func(args)

        assert prediction_path.is_file()

        test_df = pandas.read_csv(test_path)
        predictions = pandas.read_csv(prediction_path)

        assert len(test_df) == len(predictions)
