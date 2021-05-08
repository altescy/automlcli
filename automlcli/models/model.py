from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union, cast

import colt
import numpy
import pandas
from sklearn.base import BaseEstimator

from automlcli.util import cached_path, get_file_ext


class Model(colt.Registrable):  # type: ignore
    def __init__(
        self,
        target_column: str,
    ) -> None:
        self._target_column = target_column

    @property
    def estimator(self) -> BaseEstimator:
        raise NotImplementedError

    def load_data(
        self, file_path: Union[str, Path]
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray]]:
        file_path = cached_path(file_path)
        ext = get_file_ext(file_path)
        if ext in (".pkl", ".pickle"):
            df = pandas.read_pickle(file_path)
        elif ext in (".csv",):
            df = pandas.read_csv(file_path)
        elif ext in (".tsv",):
            df = pandas.read_csv(file_path, sep="\t")
        elif ext in (".jsonl",):
            df = pandas.read_json(file_path, orient="records", lines=True)
        else:
            raise ValueError(f"Not supported file format: {file_path}")

        df = cast(pandas.DataFrame, pandas.read_csv(file_path))
        y: Optional[numpy.ndarray] = None
        if self._target_column in df.columns:
            y = df.pop(self._target_column).to_numpy()
        X = df.to_numpy(dtype=float)
        return X, y

    def train(
        self,
        train_file: Union[str, Path],
        validation_file: Optional[Union[str, Path]] = None,
        workdir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, float]:
        raise NotImplementedError

    def retrain(self, train_file: Union[str, Path]) -> None:
        raise NotImplementedError

    def predict(
        self,
        file_path: Union[str, Path],
    ) -> numpy.ndarray:
        X, _ = self.load_data(file_path)
        return cast(numpy.ndarray, self.estimator.predict(X))
