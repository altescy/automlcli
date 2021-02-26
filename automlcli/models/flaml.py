from __future__ import annotations
from typing import Optional, Union
from pathlib import Path
import tempfile

import numpy

from automlcli.models.model import Model
from automlcli.util import cached_path


def _flaml_automl():
    try:
        from flaml import AutoML  # pylint: disable=import-outside-toplevel
    except ImportError as err:
        msg = ("Failed to import flaml. Make sure "
               "flaml is successfully installed")
        raise ImportError(msg) from err
    return AutoML()


@Model.register("flaml")
class FLAML(Model):
    def __init__(self, target_column: str, **kwargs) -> None:
        super().__init__(target_column)
        self._target_column = target_column
        self._kwargs = kwargs
        self._flaml_log: Optional[str] = None
        self._best_model = None
        self._best_estimator = None
        self._best_config = None

    def train(
        self,
        train_file: Union[str, Path],
        validation_file: Optional[Union[str, Path]] = None,
    ) -> None:
        train_file = cached_path(train_file)
        X_train, y_train = self.load_data(train_file)
        assert y_train is not None

        X_val: Optional[numpy.ndarray] = None
        y_val: Optional[numpy.ndarray] = None
        if validation_file is not None:
            X_val, y_val = self.load_data(validation_file)
            assert y_val is not None

        automl = _flaml_automl()

        with tempfile.NamedTemporaryFile() as logfile:
            try:
                automl.fit(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    log_file_name=logfile.name,
                    **self._kwargs,
                )
            finally:
                logfile.seek(0)
                self._flaml_log = logfile.read().decode()
                self._best_model = automl.model
                self._best_estimator = automl.best_estimator
                self._best_config = automl.best_config

    def retrain(self, train_file: Union[str, Path]) -> None:
        if self._best_model is None:
            raise RuntimeError("Model is not trained.")
        file_path = cached_path(train_file)
        X, y = self.load_data(file_path)
        self._best_model.fit(X, y)

    def predict(
        self,
        file_path: Union[str, Path],
        probability: bool = False,
    ) -> numpy.ndarray:
        if self._best_model is None:
            raise RuntimeError("Model is not trained.")
        file_path = cached_path(file_path)
        X, _ = self.load_data(file_path)
        if probability:
            return self._best_model.predict_proba(X)  # type: ignore
        return self._best_model.predict(X)  # type: ignore
