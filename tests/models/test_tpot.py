import tempfile
from pathlib import Path

from automlcli.models.tpot import Tpot

FIXTURE_PATH = Path("tests/fixtures")


def test_tpot_train():
    data_path = FIXTURE_PATH / "data" / "train.csv"

    model = Tpot(
        target_column="target",
        task="classification",
        generations=1,
        population_size=3,
        cv=3,
    )
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)
        model.train(
            data_path,
            data_path,
            tempdir,
        )

        assert (tempdir / "tpot.log").is_file()
        assert (tempdir / "pipeline.py").is_file()
