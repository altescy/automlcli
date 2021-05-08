import tempfile
from pathlib import Path

import mlflow

from automlcli.models.flaml import FLAML

FIXTURE_PATH = Path("tests/fixtures")


def test_flaml_train():
    data_path = FIXTURE_PATH / "data" / "train.csv"

    model = FLAML(target_column="target", time_budget=1)
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)
        with mlflow.start_run():
            model.train(
                data_path,
                data_path,
                tempdir,
            )

        assert (tempdir / "flaml.log").is_file()
