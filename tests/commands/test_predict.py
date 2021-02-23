import tempfile
from pathlib import Path

import numpy
import pandas
import pickle

from automlcli.commands import create_parser
from automlcli.commands.predict import PredictCommand

FIXTURE_PATH = Path("tests/fixtures")


def test_train_command():
    config_path = FIXTURE_PATH / "configs" / "config.yml"
    model_path = FIXTURE_PATH / "data" / "model.pkl"
    test_path = FIXTURE_PATH / "data" / "test.csv"

    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)
        prediction_path = tempdir / "preds.pkl"

        parser = create_parser()
        args = parser.parse_args([
            "predict",
            str(model_path),
            str(test_path),
            "--output-file",
            str(prediction_path),
        ])

        args.func(args)

        assert prediction_path.is_file()

        test_df = pandas.read_csv(test_path)
        with open(prediction_path, "rb") as fp:
            predictions = pickle.load(fp)

        assert len(test_df) == len(predictions)
