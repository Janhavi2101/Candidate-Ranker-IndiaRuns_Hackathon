from __future__ import annotations

from backend.stage4_scoring.config import DEFAULT_STAGE4_CONFIG
from backend.schema.candidate_profile import CandidateProfile
from stage1_features.models import RankingFeatures

from .common import MatcherResult, clamp01, log_scale, to_percent, weighted_average


def score_behavioral_match(
    candidate_profile: CandidateProfile,
    ranking_features: RankingFeatures,
    weights: dict[str, float] | None = None,
) -> MatcherResult:
    behavioral_weights = weights or DEFAULT_STAGE4_CONFIG.behavioral_weights
    completeness = clamp01(ranking_features.profile_completeness_score / 100.0)
    recruiter_response = clamp01(ranking_features.recruiter_response_rate)
    interview_completion = clamp01(ranking_features.interview_completion_rate)
    github_activity = clamp01(ranking_features.github_activity_score / 100.0)
    endorsements = log_scale(ranking_features.endorsements_received, base=100.0)
    search_visibility = log_scale(ranking_features.search_appearance_30d, base=1000.0)

    scores = {
        "profile_completeness": completeness,
        "recruiter_response_rate": recruiter_response,
        "github_activity": github_activity,
        "interview_completion": interview_completion,
        "endorsements": endorsements,
        "search_visibility": search_visibility,
    }
    score = weighted_average(scores, behavioral_weights)
    return MatcherResult(
        score=to_percent(score),
        details={
            "profile_completeness": completeness,
            "recruiter_response_rate": recruiter_response,
            "interview_completion": interview_completion,
            "github_activity": github_activity,
            "endorsements": endorsements,
            "search_visibility": search_visibility,
            "weights": dict(behavioral_weights),
        },
    )
