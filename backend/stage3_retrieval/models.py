from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(slots=True)
class RetrievedCandidate:
    candidate_id: str
    semantic_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

