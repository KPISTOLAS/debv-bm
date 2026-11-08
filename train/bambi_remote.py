"""Seekable file view of a URL over HTTP range requests, with a block cache.

Used by build_bambi_detect.py to open BAMBI's Zenodo zip archives without downloading them:
``zipfile`` reads the central directory at the end and any member at its offset, so small
annotation files cost a few requests and a video member can be decompressed as a stream.
"""

from __future__ import annotations

import io
import time
from collections import OrderedDict

import requests

BLOCK = 4 * 1024 * 1024  # bytes per range request
MAX_BLOCKS = 8           # small cache: reads are mostly sequential


class RangeFile(io.RawIOBase):
    """Seekable window [start, start+size) of a URL, read in cached blocks via HTTP ranges."""

    def __init__(self, session: requests.Session, url: str, start: int = 0, size: int | None = None):
        self.s, self.url, self.start = session, url, start
        if size is None:
            for attempt in range(8):  # Zenodo answers 429 when many flights open at once
                try:
                    r = session.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
                    r.raise_for_status()
                    break
                except (requests.RequestException, OSError):
                    if attempt == 7:
                        raise
                    time.sleep(2 ** attempt)
            if r.status_code != 206:
                raise RuntimeError("server does not honour HTTP range requests")
            size = int(r.headers["Content-Range"].rsplit("/", 1)[1]) - start
        self.size, self.pos = size, 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.fetched = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.pos = offset if whence == io.SEEK_SET else self.pos + offset if whence == io.SEEK_CUR else self.size + offset
        return self.pos

    def _block(self, i: int) -> bytes:
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        lo = self.start + i * BLOCK
        hi = min(self.start + self.size, lo + BLOCK) - 1
        for attempt in range(6):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=120)
                r.raise_for_status()
                data = r.content
                break
            except (requests.RequestException, OSError):
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        self.fetched += len(data)
        self.cache[i] = data
        if len(self.cache) > MAX_BLOCKS:
            self.cache.popitem(last=False)
        return data

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        out = bytearray()
        while n > 0:
            i, off = divmod(self.pos, BLOCK)
            chunk = self._block(i)[off:off + n]
            if not chunk:
                break
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)
