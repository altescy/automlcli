from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy
from sklearn.base import BaseEstimator

try:
    import flaml
except ImportError:
    flaml = None

from automlcli.models.model import Model
from automlcli.util import cached_path


@Model.register("flaml")
class FLAML(Model):
    def __init__(self, target_column: str, **kwargs: Any) -> None:
        if flaml is None:
            raise ImportError(
                "Failed to import flaml. Make sure " "flaml is successfully installed"
            )
        super().__init__(target_column)
        self._target_column = target_column
        self._kwargs = kwargs
        self._flaml_log: Optional[str] = None
        self._flaml_best_model: Optional[BaseEstimator] = None
        self._flaml_best_estimator: Optional[str] = None
        self._flaml_best_config: Optional[Dict[str, Any]] = None
        self._flaml_best_iteration: Optional[int] = None

    @property
    def estimator(self) -> BaseEstimator:
        if self._flaml_best_model is None:
            raise RuntimeError("FLAML model is not trained.")
        return self._flaml_best_model

    def train(
        self,
        train_file: Union[str, Path],
        validation_file: Optional[Union[str, Path]] = None,
        workdir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, float]:
        train_file = cached_path(train_file)
        X_train, y_train = self.load_data(train_file)
        assert y_train is not None

        X_val: Optional[numpy.ndarray] = None
        y_val: Optional[numpy.ndarray] = None
        if validation_file is not None:
            X_val, y_val = self.load_data(validation_file)
            assert y_val is not None

        automl = flaml.AutoML()

        with tempfile.NamedTemporaryFile() as temp_file:
            automl.fit(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                log_file_name=temp_file.name,
                **self._kwargs,
            )

            temp_file.seek(0)
            self._flaml_log = temp_file.read().decode()
            self._flaml_best_model = automl.model
            self._flaml_best_estimator = automl.best_estimator
            self._flaml_best_config = automl.best_config
            self._flaml_best_iteration = automl.best_iteration

        metrics = {
            "best_loss": automl.best_loss,
        }

        if workdir is not None:
            workdir = Path(workdir)
            log_file = workdir / "flaml.log"
            best_file = workdir / "best.json"

            with open(log_file, "w") as fp:
                fp.write(self._flaml_log)

            best = {
                "best_estimator": self._flaml_best_estimator,
                "best_config": self._flaml_best_config,
                "best_iteration": self._flaml_best_iteration,
            }
            with open(best_file, "w") as fp:
                json.dump(best, fp)

        return metrics

    def retrain(self, train_file: Union[str, Path]) -> None:
        X_train, y_train = self.load_data(train_file)
        if y_train is None:
            raise ValueError(
                f"Target column ({self._target_column}) does "
                f"not exists in {train_file}"
            )
        self._flaml_best_model = self.estimator.fit(X_train, y_train)
