from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


def _serialize(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    return value


@dataclass(slots=True)
class CandidateMetadata:
    candidate_id: str
    years_of_experience: float
    location: str
    country: str
    current_industry: str
    current_company_size: str
    open_to_work: bool
    preferred_work_mode: str
    willing_to_relocate: bool
    skills: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RankingFeatures:
    profile_completeness_score: float
    github_activity_score: float
    recruiter_response_rate: float
    interview_completion_rate: float
    offer_acceptance_rate: float
    profile_views_received_30d: int
    search_appearance_30d: int
    saved_by_recruiters_30d: int
    endorsements_received: int
    connection_count: int
    notice_period_days: int
    applications_submitted_30d: int
    avg_response_time_hours: float
    open_to_work_flag: bool
    willing_to_relocate: bool
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool
    consistency_score: float | None = None
    career_stability: float | None = None
    average_job_tenure: float | None = None
    company_progression: float | None = None
    profile_quality: float | None = None
    skill_assessment_scores: Dict[str, float] = field(default_factory=dict)
    expected_salary_range_inr_lpa: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CandidateDocument:
    candidate_id: str
    embedding_text: str
    metadata: CandidateMetadata
    ranking_features: RankingFeatures

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "embedding_text": self.embedding_text,
            "metadata": _serialize(self.metadata),
            "ranking_features": _serialize(self.ranking_features),
        }
