from __future__ import annotations
from typing import cast, Optional, Tuple, Union
from pathlib import Path
import tempfile

import numpy
import pandas

from automlcli.exceptions import ConfigurationError
from automlcli.models.model import Model
from automlcli.util import cached_path, get_file_ext


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
        self._target_column = target_column
        self._kwargs = kwargs
        self._flaml_log: Optional[str] = None
        self._best_model = None
        self._best_estimator = None
        self._best_config = None

    @staticmethod
    def _read_dataframe(file_path: Union[str, Path]) -> pandas.DataFrame:
        ext = get_file_ext(file_path)
        if ext in (".pkl", "pickle"):
            return cast(numpy.ndarray,
                        pandas.read_pickle(file_path).to_numpy())
        if ext in (".csv", ):
            return pandas.read_csv(file_path)
        raise ConfigurationError(f"Could not read the given file: {file_path}")

    def _dataframe_to_numpy(
        self,
        df: pandas.DataFrame,
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray]]:
        y: Optional[numpy.ndarray] = None
        if self._target_column in df.columns:
            y = cast(numpy.ndarray, df.pop(self._target_column).to_numpy())
        X = cast(numpy.ndarray, df.to_numpy())
        return X, y

    def _read_datafile(
        self,
        file_path: Union[str, Path],
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray]]:
        df = self._read_dataframe(file_path)
        X, y = self._dataframe_to_numpy(df)
        return X, y

    def train(
        self,
        train_file: Union[str, Path],
        validation_file: Optional[Union[str, Path]] = None,
    ) -> None:
        train_file = cached_path(train_file)
        X_train, y_train = self._read_datafile(train_file)
        assert y_train is not None

        X_val: Optional[numpy.ndarray] = None
        y_val: Optional[numpy.ndarray] = None
        if validation_file is not None:
            X_val, y_val = self._read_datafile(validation_file)
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
        X, y = self._read_datafile(file_path)
        self._best_model.fit(X, y)

    def predict(
        self,
        file_path: Union[str, Path],
        probability: bool = False,
    ) -> numpy.ndarray:
        if self._best_model is None:
            raise RuntimeError("Model is not trained.")
        file_path = cached_path(file_path)
        X, _ = self._read_datafile(file_path)
        if probability:
            return self._best_model.predict_proba(X)  # type: ignore
        return self._best_model.predict(X)  # type: ignore
