from typing import IO


class TeeingIO:
    def __init__(self, stream: IO, out: IO):
        self._stream = stream
        self._out = out

    def __getattr__(self, name: str):
        return getattr(self._stream, name)

    def write(self, data):
        self._stream.write(data)
        self._out.write(data)

    def flush(self):
        self._stream.flush()
        self._out.flush()
