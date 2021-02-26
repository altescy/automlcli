from __future__ import annotations
from typing import Dict, Optional, Tuple, Union
from pathlib import Path

import colt
import numpy
import pandas

from automlcli.util import get_file_ext


class Model(colt.Registrable):
    def __init__(self, target_column: str) -> None:
        self._target_column = target_column

    def load_data(
        self, file_path: Union[str, Path]
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray]]:
        ext = get_file_ext(file_path)
        if ext in (".pkl", ".pickle"):
            df = pandas.read_pickle(file_path)
        elif ext in (".csv", ):
            df = pandas.read_csv(file_path)
        elif ext in (".tsv", ):
            df = pandas.read_csv(file_path, sep="\t")
        elif ext in (".jsonl", ):
            df = pandas.read_json(file_path, orient="records", lines=True)
        else:
            raise ValueError(f"Not supported file format: {file_path}")

        df = pandas.read_csv(file_path)
        y: Optional[numpy.ndarray] = None
        if self._target_column in df.columns:
            y = df.pop(self._target_column).to_numpy()
        X = df.to_numpy()
        return X, y

    def train(
        self,
        train_file: Union[str, Path],
        validation_file: Optional[Union[str, Path]] = None,
        workdir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, float]:
        raise NotImplementedError

    def retrain(self, train_file: Union[str, Path]) -> None:
        X, y = self.load_data(train_file)
        assert y is not None
        self.fit(X, y)

    def predict_from_file(
        self,
        file_path: Union[str, Path],
    ) -> numpy.ndarray:
        X, _ = self.load_data(file_path)
        return self.predict(X)

    def fit(self, X: numpy.ndarray, y: numpy.ndarray) -> Model:
        raise NotImplementedError

    def predict(self, X: numpy.ndarray) -> numpy.ndarray:
        raise NotImplementedError
