import logging
import os
import random
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple, Union
from urllib.parse import urlparse

import numpy
from fs import open_fs
from fs.copy import copy_fs

logger = logging.getLogger(__name__)


def set_random_seed(
    random_seed: Optional[int] = None,
    numpy_seed: Optional[int] = None,
) -> None:
    if random_seed:
        random.seed(random_seed)
    if numpy_seed:
        numpy.random.seed(numpy_seed)


def ext_match(file_path: Union[str, Path], exts: Iterable[str]) -> bool:
    filename = urlparse(str(file_path)).path
    pattern = re.compile(rf".+\.{ '|'.join(exts) }(\..*)?$")
    match = re.match(pattern, filename)
    return match is not None


def get_file_ext(file_path: Union[str, Path]) -> str:
    file_path = Path(file_path)
    return file_path.suffix


def get_parent_path_and_filename(file_path: str) -> Tuple[str, str]:
    splitted = str(file_path).rsplit("/", 1)
    if len(splitted) == 2:
        parent, name = splitted
    else:
        parent = "./"
        name = str(file_path)
    return parent, name


@contextmanager
def create_workdir(
    path: Optional[Union[str, Path]] = None, exist_ok: bool = False
) -> Iterator[Path]:
    if path is None:
        with tempfile.TemporaryDirectory() as tempdir:
            yield Path(tempdir)
            return

    parsed = urlparse(str(path))
    if parsed.scheme in ("", "file", "osfs"):
        os.makedirs(parsed.path, exist_ok=exist_ok)
        yield Path(parsed.path)
        return

    path = str(path)
    with open_fs(path) as fs:
        if not exist_ok and fs.exists:
            raise FileExistsError(f"File exists: {path}")

    with tempfile.TemporaryDirectory() as tempdir:
        yield Path(tempdir)
        copy_fs(tempdir, path)
