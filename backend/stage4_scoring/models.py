from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class IndividualScores:
    retrieval: float = 0.0
    technical: float = 0.0
    experience: float = 0.0
    education: float = 0.0
    company: float = 0.0
    behavioral: float = 0.0
    consistency: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class ScoreBreakdown:
    raw_component_scores: Dict[str, float] = field(default_factory=dict)
    weighted_component_scores: Dict[str, float] = field(default_factory=dict)
    bonus_details: Dict[str, float] = field(default_factory=dict)
    penalty_details: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ScoredCandidate:
    candidate_id: str
    individual_scores: IndividualScores
    bonuses: float
    penalties: float
    final_score: float
    score_breakdown: ScoreBreakdown

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "individual_scores": self.individual_scores.to_dict(),
            "bonuses": self.bonuses,
            "penalties": self.penalties,
            "final_score": self.final_score,
            "score_breakdown": self.score_breakdown.to_dict(),
        }

