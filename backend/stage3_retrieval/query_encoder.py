from __future__ import annotations

from typing import Optional, Protocol, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

class QueryEmbeddingModel(Protocol):
    model_name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        ...


@dataclass(slots=True)
class SentenceTransformerQueryEncoder:
    model_name: str = "all-MiniLM-L6-v2"
    device: Optional[str] = None
    normalize_embeddings: bool = True
    batch_size: int = 1
    _model: Any = field(init=False, repr=False)


    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency issue
            raise RuntimeError("sentence_transformers is required for Stage 3 query encoding") from exc

        self._model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, texts: Sequence[str] | str) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        vector = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )[0]
        return np.asarray(vector, dtype=np.float32)


@dataclass(slots=True)
class QueryEncoder:
    model: QueryEmbeddingModel

    def encode(self, job_description: str) -> np.ndarray:
        embeddings = self.model.encode([job_description])
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim == 1:
            return array
        if array.ndim == 2 and array.shape[0] > 0:
            return array[0]
        raise ValueError("query encoder must return a 1D vector")
