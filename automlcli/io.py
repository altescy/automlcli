from typing import IO, Any


class TeeingIO:
    def __init__(self, stream: IO[Any], out: IO[Any]) -> None:
        self._stream = stream
        self._out = out

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def write(self, data: Any) -> None:
        self._stream.write(data)
        self._out.write(data)

    def flush(self) -> None:
        self._stream.flush()
        self._out.flush()
