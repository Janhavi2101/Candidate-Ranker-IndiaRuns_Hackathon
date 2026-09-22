from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from backend.preprocessing.sample import parse_job_profile, parse_job_profile_from_text
from backend.schema.job_profile import JobProfile

from ..stage2_embeddings.bm25_builder import BM25Index, BM25SearchHit
from .models import RetrievedCandidate
from .query_builder import build_bm25_query, build_semantic_query
from .query_encoder import QueryEncoder


@dataclass(slots=True)
class HybridCandidateRetriever:
    bm25_index: BM25Index
    query_encoder: QueryEncoder
    vector_store: Any | None = None
    vector_store_path: str | Path | None = None
    rrf_k: int = 60

    @classmethod
    def load(
        cls,
        vector_store_dir: str | Path,
        bm25_index_path: str | Path,
        query_encoder: QueryEncoder,
        rrf_k: int = 60,
    ) -> "HybridCandidateRetriever":
        return cls(
            vector_store=None,
            vector_store_path=Path(vector_store_dir),
            bm25_index=BM25Index.load(bm25_index_path),
            query_encoder=query_encoder,
            rrf_k=rrf_k,
        )

    def _rrf_increment(self, rank: int) -> float:
        return 1.0 / (self.rrf_k + rank)

    def _coerce_job_profile(self, job_source: str | Path | JobProfile) -> JobProfile:
        if isinstance(job_source, JobProfile):
            return job_source
        try:
            return parse_job_profile(job_source)
        except (OSError, ValueError):
            return parse_job_profile_from_text(str(job_source))

    def _load_vector_store(self):
        if self.vector_store is not None:
            return self.vector_store
        if self.vector_store_path is None:
            raise ValueError("vector_store is not configured")

        from backend.stage2_embeddings.vector_store import CandidateVectorStore

        self.vector_store = CandidateVectorStore.load(self.vector_store_path)
        return self.vector_store

    def search_semantic(self, job_source: str | Path | JobProfile, top_k: int = 200):
        job_profile = self._coerce_job_profile(job_source)
        semantic_query = build_semantic_query(job_profile)
        query_vector = self.query_encoder.encode(semantic_query)
        vector_store = self._load_vector_store()
        return vector_store.search(query_vector, top_k=top_k)

    def search_bm25(self, job_source: str | Path | JobProfile, top_k: int = 200) -> List[BM25SearchHit]:
        job_profile = self._coerce_job_profile(job_source)
        bm25_query = build_bm25_query(job_profile)
        bm25_hits = self.bm25_index.search(bm25_query, top_k=top_k)
        return bm25_hits

    def retrieve(
        self,
        job_source: str | Path | JobProfile | None = None,
        top_k: int = 200,
        *,
        job_description: str | Path | JobProfile | None = None,
    ) -> List[RetrievedCandidate]:
        if job_source is None:
            job_source = job_description
        if job_source is None:
            raise ValueError("job_source or job_description must be provided")

        semantic_hits = self.search_semantic(job_source, top_k=top_k)
        bm25_hits = self.search_bm25(job_source, top_k=top_k)

        fused: Dict[str, RetrievedCandidate] = {}

        for rank, hit in enumerate(semantic_hits, start=1):
            fused[hit.candidate_id] = RetrievedCandidate(
                candidate_id=hit.candidate_id,
                semantic_score=float(hit.score),
                bm25_score=None,
                rrf_score=self._rrf_increment(rank),
            )

        for rank, hit in enumerate(bm25_hits, start=1):
            existing = fused.get(hit.candidate_id)
            if existing is None:
                fused[hit.candidate_id] = RetrievedCandidate(
                    candidate_id=hit.candidate_id,
                    semantic_score=None,
                    bm25_score=float(hit.score),
                    rrf_score=self._rrf_increment(rank),
                )
                continue

            existing.bm25_score = float(hit.score)
            existing.rrf_score += self._rrf_increment(rank)

        ranked = sorted(
            fused.values(),
            key=lambda item: (-item.rrf_score, item.candidate_id),
        )
        return ranked[:top_k]
