from __future__ import annotations
from typing import Optional, Union
from pathlib import Path

import colt
import numpy


class Model(colt.Registrable):
    def train(
        self,
        train_file: Union[str, Path],
        validation_file: Optional[Union[str, Path]] = None,
    ) -> None:
        raise NotImplementedError

    def retrain(self, train_file: Union[str, Path]) -> None:
        raise NotImplementedError

    def predict(
        self,
        file_path: Union[str, Path],
        probability: bool = False,
    ) -> numpy.ndarray:
        raise NotImplementedError
