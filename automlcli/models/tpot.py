import pickle
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

from sklearn.base import BaseEstimator

try:
    import tpot
except ImportError:
    tpot = None

from automlcli.exceptions import ConfigurationError
from automlcli.io import TeeingIO
from automlcli.models.model import Model


@Model.register("tpot")
class Tpot(Model):
    TPOT_TASKS = ("classification", "regression")

    def __init__(
        self,
        task: str,
        target_column: str,
        cv_after_training: bool = False,
        **kwargs: Any,
    ) -> None:
        if tpot is None:
            raise ImportError(
                "Failed to import flaml. Make sure " "flaml is successfully installed"
            )
        if task not in self.TPOT_TASKS:
            raise ConfigurationError("task must be 'classification' " "or 'regression'")

        super().__init__(target_column)
        self._task = task
        self._kwargs = kwargs
        self._cv_after_trainnig = cv_after_training
        self._estimator: Optional[BaseEstimator] = None

        if "n_jobs" not in self._kwargs:
            self._kwargs["n_jobs"] = -1
        if "scoring" not in self._kwargs:
            self._kwargs["scoring"] = "accuracy"
        if "verbosity" not in self._kwargs:
            self._kwargs["verbosity"] = 2

    @property
    def estimator(self) -> BaseEstimator:
        if self._estimator is None:
            raise RuntimeError("Tpot model is not trained.")
        return self._estimator

    def train(
        self,
        train_file: Union[str, Path],
        validation_file: Optional[Union[str, Path]] = None,
        workdir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, float]:
        X_train, y_train = self.load_data(train_file)
        assert y_train is not None

        with tempfile.TemporaryDirectory() as tempdir:
            workdir = Path(workdir or tempdir)
            log_file_name = workdir / "tpot.log"
            pipeline_file_name = workdir / "fitted_pipeline.pkl"
            pipeline_code_file_name = workdir / "pipeline.py"

            with open(log_file_name, "w") as log_file:
                teeing_log_file = TeeingIO(log_file, sys.stdout)
                if self._task == "classification":
                    model = tpot.TPOTClassifier(
                        log_file=teeing_log_file,
                        **self._kwargs,
                    )
                else:
                    model = tpot.TPOTRegressor(log_file=teeing_log_file, **self._kwargs)

                model.fit(X_train, y_train)

            with open(log_file_name) as log_file:
                tpot_log = log_file.read()

            model.export(str(pipeline_code_file_name))
            self._estimator = model.fitted_pipeline_
            with open(pipeline_file_name, "wb") as pipeline_file:
                pickle.dump(self._estimator, pipeline_file)

        metrics = self._get_metrics_from_log(tpot_log)

        if validation_file is not None:
            X_val, y_val = self.load_data(validation_file)
            assert y_val is not None
            metrics["validation_score"] = model.score(X_val, y_val)

        return metrics

    def retrain(self, train_file: Union[str, Path]) -> None:
        X_train, y_train = self.load_data(train_file)
        assert y_train is not None
        self.estimator.fit(X_train, y_train)

    @staticmethod
    def _get_metrics_from_log(log: str) -> Dict[str, float]:
        best_cv_score = -float("inf")
        for line in log.splitlines():
            match = re.search(r"Current best internal CV score: ([0-9.]+)", line)
            if match is None:
                continue
            cv_score = float(match.group(1))
            best_cv_score = max(best_cv_score, cv_score)

        metrics = {"best_cv_score": best_cv_score}
        return metrics
