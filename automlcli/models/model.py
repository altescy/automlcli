from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

import colt
import numpy
import pandas
from sklearn.base import BaseEstimator

from automlcli.util import cached_path, ext_match


class Model(colt.Registrable):  # type: ignore
    def __init__(
        self,
        target_column: str,
        index_column: Optional[str] = None,
        ignored_columns: Optional[List[str]] = None,
    ) -> None:
        self._target_column = target_column
        self._index_column = index_column
        self._ignored_columns = ignored_columns or []

    @property
    def estimator(self) -> BaseEstimator:
        raise NotImplementedError

    def load_dataframe(
        self,
        file_path: Union[str, Path],
    ) -> pandas.DataFrame:
        file_path = cached_path(file_path)
        if ext_match(file_path, ["pkl", "pickle"]):
            df = pandas.read_pickle(file_path)
        elif ext_match(file_path, ["csv"]):
            df = pandas.read_csv(file_path)
        elif ext_match(file_path, ["tsv"]):
            df = pandas.read_csv(file_path, sep="\t")
        elif ext_match(file_path, ["jsonl"]):
            df = pandas.read_json(file_path, orient="records", lines=True)
        else:
            raise ValueError(f"Not supported file format: {file_path}")
        return df

    def _dataframe_to_array(
        self, df: pandas.DataFrame
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray]]:
        y: Optional[numpy.ndarray] = None

        if self._index_column is not None:
            df = df.set_index(self._index_column)

        if self._target_column in df.columns:
            y = df.pop(self._target_column).to_numpy()

        for column in self._ignored_columns:
            df.pop(column)

        X = df.to_numpy(dtype=float)
        return X, y

    def load_data(
        self, file_path: Union[str, Path]
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray]]:
        df = self.load_dataframe(file_path)
        X, y = self._dataframe_to_array(df)
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
    ) -> pandas.DataFrame:
        df = self.load_dataframe(file_path)
        X, _ = self._dataframe_to_array(df)

        y_pred = cast(numpy.ndarray, self.estimator.predict(X))

        df["prediction"] = y_pred

        return df[["prediction"]]
