from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from stage1_features.models import CandidateMetadata, RankingFeatures


@dataclass(slots=True)
class EmbeddedCandidate:
    candidate_id: str
    embedding: List[float]
    metadata: CandidateMetadata
    ranking_features: RankingFeatures
    embedding_model_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "embedding": list(self.embedding),
            "metadata": asdict(self.metadata),
            "ranking_features": asdict(self.ranking_features),
            "embedding_model_name": self.embedding_model_name,
        }


@dataclass(slots=True)
class CandidateSearchResult:
    candidate_id: str
    score: float
    embedded_candidate: EmbeddedCandidate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "score": self.score,
            "embedded_candidate": self.embedded_candidate.to_dict(),
        }

