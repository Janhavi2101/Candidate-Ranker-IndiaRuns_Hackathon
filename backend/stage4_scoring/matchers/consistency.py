from __future__ import annotations

from stage1_features.models import RankingFeatures

from .common import MatcherResult, clamp01, to_percent


def score_consistency_match(ranking_features: RankingFeatures) -> MatcherResult:
    candidates = [
        ranking_features.consistency_score,
        ranking_features.career_stability,
        ranking_features.average_job_tenure,
        ranking_features.company_progression,
        ranking_features.profile_quality,
    ]
    values = [float(value) for value in candidates if value is not None]
    if not values:
        return MatcherResult(score=50.0, details={"consistency_source": "neutral_default"})
    return MatcherResult(
        score=to_percent(sum(values) / len(values)),
        details={"consistency_source": "offline_features", "signals_used": len(values)},
    )
