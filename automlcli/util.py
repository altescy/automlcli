from typing import Any, IO, Iterator, Union
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse
import hashlib
import logging
import os
import tempfile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from fs import open_fs
from tqdm import tqdm

from automlcli.settings import CACHE_DIRRECTORY

logger = logging.getLogger(__name__)


def get_file_ext(file_path: Union[str, Path]) -> str:
    file_path = Path(file_path)
    return file_path.suffix


def cached_path(url_or_filename: Union[str, Path],
                cache_dir: Union[str, Path] = None) -> Path:
    cache_dir = Path(cache_dir or CACHE_DIRRECTORY)

    os.makedirs(cache_dir, exist_ok=True)

    parsed = urlparse(str(url_or_filename))

    if parsed.scheme in ("", "file", "osfs"):
        return Path(url_or_filename)

    cache_path = cache_dir / _get_cached_filename(url_or_filename)
    if cache_path.exists():
        logger.info("use cache for %s: %s", str(url_or_filename),
                    str(cache_path))
        return cache_path

    cache_fp = open(cache_path, "w+b")
    try:
        with open_file(url_or_filename, "r+b") as fp:
            cache_fp.write(fp.read())
    finally:
        cache_fp.close()

    return cache_path


@contextmanager
def open_file(file_path: Union[str, Path],
              mode: str = "r",
              **kwargs) -> Iterator[IO[Any]]:
    parsed = urlparse(str(file_path))

    if parsed.scheme in ("http", "https"):
        if not mode.startswith("r"):
            raise ValueError(f"invalid mode for http(s): {mode}")

        url = str(file_path)
        temp_file = tempfile.NamedTemporaryFile(delete=False)

        try:
            _http_get(url, temp_file)
            temp_file.close()
            with open(temp_file.name, mode) as fp:
                yield fp
        finally:
            os.remove(temp_file.name)
    else:
        with open_file_with_fs(file_path, mode=mode, **kwargs) as fp:
            yield fp


@contextmanager
def open_file_with_fs(file_path: Union[str, Path], *args,
                      **kwargs) -> Iterator[IO[Any]]:
    file_path = Path(file_path)
    parent = str(file_path.parent)
    name = str(file_path.name)

    with open_fs(parent) as fs:
        with fs.open(name, *args, **kwargs) as fp:
            yield fp


def _session_with_backoff() -> requests.Session:
    """
    https://stackoverflow.com/questions/23267409/how-to-implement-retry-mechanism-into-python-requests-library
    """
    session = requests.Session()
    retries = Retry(total=5,
                    backoff_factor=1,
                    status_forcelist=[502, 503, 504])
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))

    return session


def _http_get(url: str, temp_file: IO) -> None:
    with _session_with_backoff() as session:
        req = session.get(url, stream=True)
        req.raise_for_status()
        content_length = req.headers.get("Content-Length")
        total = int(content_length) if content_length is not None else None
        progress = tqdm(unit="B", total=total, desc="downloading")
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                progress.update(len(chunk))
                temp_file.write(chunk)
        progress.close()


def _get_cached_filename(path: Union[str, Path]) -> str:
    encoded_path = str(path).encode()
    name = hashlib.md5(encoded_path).hexdigest()
    ext = get_file_ext(path)
    return name + ext
