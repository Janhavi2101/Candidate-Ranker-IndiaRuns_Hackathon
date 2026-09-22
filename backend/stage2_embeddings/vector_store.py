from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import faiss
import numpy as np

from .models import CandidateSearchResult, EmbeddedCandidate


def _ensure_float32_matrix(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("vectors must be a 2D matrix")
    return matrix


def _ensure_float32_vector(vector: Sequence[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError("query vector must be 1D")
    return array


@dataclass(slots=True)
class CandidateVectorStore:
    dimension: int
    use_cosine_similarity: bool = True
    index: faiss.Index = field(init=False)
    candidate_ids: List[str] = field(default_factory=list)
    candidates: Dict[str, EmbeddedCandidate] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.index = faiss.IndexFlatIP(self.dimension) if self.use_cosine_similarity else faiss.IndexFlatL2(self.dimension)

    def _prepare_vectors(self, vectors: np.ndarray) -> np.ndarray:
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"expected vectors with dimension {self.dimension}, got {vectors.shape[1]}")
        if self.use_cosine_similarity:
            faiss.normalize_L2(vectors)
        return vectors

    def add(self, embedded_candidates: Iterable[EmbeddedCandidate]) -> None:
        batch = list(embedded_candidates)
        if not batch:
            return
        vectors = _ensure_float32_matrix([candidate.embedding for candidate in batch])
        vectors = self._prepare_vectors(vectors)
        self.index.add(vectors)
        for candidate in batch:
            self.candidate_ids.append(candidate.candidate_id)
            self.candidates[candidate.candidate_id] = candidate

    def search(self, query_vector: Sequence[float], top_k: int = 10) -> List[CandidateSearchResult]:
        if self.index.ntotal == 0:
            return []
        vector = _ensure_float32_vector(query_vector).reshape(1, -1)
        if vector.shape[1] != self.dimension:
            raise ValueError(f"expected query vector with dimension {self.dimension}, got {vector.shape[1]}")
        if self.use_cosine_similarity:
            faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, top_k)
        results: List[CandidateSearchResult] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0 or index >= len(self.candidate_ids):
                continue
            candidate_id = self.candidate_ids[index]
            results.append(
                CandidateSearchResult(
                    candidate_id=candidate_id,
                    score=float(score),
                    embedded_candidate=self.candidates[candidate_id],
                )
            )
        return results

    def save(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(target / "faiss.index"))
        payload = {
            "dimension": self.dimension,
            "use_cosine_similarity": self.use_cosine_similarity,
            "candidate_ids": self.candidate_ids,
            "candidates": {candidate_id: candidate.to_dict() for candidate_id, candidate in self.candidates.items()},
        }
        (target / "lookup.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "CandidateVectorStore":
        source = Path(directory)
        payload = json.loads((source / "lookup.json").read_text(encoding="utf-8"))
        store = cls(
            dimension=int(payload["dimension"]),
            use_cosine_similarity=bool(payload["use_cosine_similarity"]),
        )
        store.index = faiss.read_index(str(source / "faiss.index"))
        store.candidate_ids = list(payload["candidate_ids"])
        store.candidates = {
            candidate_id: EmbeddedCandidate(
                candidate_id=item["candidate_id"],
                embedding=list(item["embedding"]),
                metadata=_metadata_from_dict(item["metadata"]),
                ranking_features=_ranking_features_from_dict(item["ranking_features"]),
                embedding_model_name=item.get("embedding_model_name", ""),
            )
            for candidate_id, item in payload["candidates"].items()
        }
        return store


def _metadata_from_dict(payload: Dict[str, object]):
    from stage1_features.models import CandidateMetadata

    return CandidateMetadata(**payload)


def _ranking_features_from_dict(payload: Dict[str, object]):
    from stage1_features.models import RankingFeatures

    return RankingFeatures(**payload)

