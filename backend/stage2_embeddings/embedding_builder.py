from __future__ import annotations

from dataclasses import dataclass, field

from typing import Iterable, List, Optional, Protocol, Sequence, Union
from typing import Any

import numpy as np

from backend.schema.candidate_profile import CandidateProfile
from stage1_features.candidate_builder import build_candidate_document
from stage1_features.models import CandidateDocument

from .models import EmbeddedCandidate


class EmbeddingModel(Protocol):
    model_name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        ...


@dataclass(slots=True)
class SentenceTransformerEmbeddingBuilder:
    model_name: str = "all-MiniLM-L6-v2"
    device: Optional[str] = None
    normalize_embeddings: bool = True
    batch_size: int = 32

    _model: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import importlib

            sentence_transformers = importlib.import_module("sentence_transformers")
            SentenceTransformer = sentence_transformers.SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency issue
            raise RuntimeError(
                "sentence_transformers is required for Stage 2 embedding generation"
            ) from exc

        self._model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        embeddings = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)


@dataclass(slots=True)
class EmbeddingBuilder:
    model: EmbeddingModel

    def build_embedding_vector(self, embedding_text: str) -> List[float]:
        vector = self.model.encode([embedding_text])[0]
        return np.asarray(vector, dtype=np.float32).tolist()

    def build_embedded_candidate(self, candidate_document: CandidateDocument) -> EmbeddedCandidate:
        return EmbeddedCandidate(
            candidate_id=candidate_document.candidate_id,
            embedding=self.build_embedding_vector(candidate_document.embedding_text),
            metadata=candidate_document.metadata,
            ranking_features=candidate_document.ranking_features,
            embedding_model_name=getattr(self.model, "model_name", ""),
        )


def build_embedded_candidate(
    candidate: Union[CandidateProfile, CandidateDocument],
    builder: Optional[EmbeddingBuilder] = None,
) -> EmbeddedCandidate:
    candidate_document = candidate if isinstance(candidate, CandidateDocument) else build_candidate_document(candidate)
    if builder is None:
        builder = EmbeddingBuilder(model=SentenceTransformerEmbeddingBuilder())
    return builder.build_embedded_candidate(candidate_document)

