"""Inner-product nearest-neighbour index with FAISS backend and numpy fallback.

For ~5k MSD graphs numpy is fast enough; the FAISS path matters once corpora
grow into the 100k+ range. Vectors are persisted as `.npy` regardless of
backend so saved indices are portable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class FlatIPIndex:
    def __init__(self, dim: int):
        self.dim = int(dim)
        self._data: np.ndarray | None = None
        self._faiss = None
        try:
            import faiss  # type: ignore

            self._faiss = faiss.IndexFlatIP(self.dim)
        except ImportError:
            self._faiss = None

    def add(self, x: np.ndarray) -> None:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"Expected (N, {self.dim}) matrix, got shape {x.shape}")
        if self._faiss is not None:
            self._faiss.add(x)
        self._data = x if self._data is None else np.vstack([self._data, x])

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(q, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        if self._faiss is not None:
            return self._faiss.search(q, k)
        sims = q @ self._data.T
        idx = np.argsort(-sims, axis=1)[:, :k]
        scores = np.take_along_axis(sims, idx, axis=1)
        return scores, idx

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._data is None:
            raise RuntimeError("Cannot save an empty index")
        np.save(path, self._data)

    @classmethod
    def load(cls, path: str | Path, dim: int | None = None) -> "FlatIPIndex":
        data = np.load(Path(path))
        idx = cls(dim if dim is not None else data.shape[1])
        idx.add(data)
        return idx
