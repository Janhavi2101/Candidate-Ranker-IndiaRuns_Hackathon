from __future__ import annotations

from backend.stage4_scoring.config import DEFAULT_STAGE4_CONFIG
from backend.schema.candidate_profile import CandidateProfile
from backend.schema.job_profile import JobProfile

from .common import MatcherResult, clamp01, normalize_alnum_text, title_similarity, to_percent, weighted_average


def _experience_match(candidate_years: float, minimum_years: float | None) -> float:
    if minimum_years is None or minimum_years <= 0:
        return 0.5
    if candidate_years >= minimum_years:
        return 1.0
    return clamp01(candidate_years / minimum_years)


def _domain_experience(candidate_profile: CandidateProfile, job_profile: JobProfile) -> float:
    if not job_profile.industry:
        return 0.5
    job_industry = normalize_alnum_text(job_profile.industry)
    industries = [normalize_alnum_text(role.industry) for role in candidate_profile.career_history if role.industry]
    if not industries:
        return 0.0
    matches = sum(1 for industry in industries if job_industry in industry or industry in job_industry)
    return clamp01(matches / len(industries))


def score_experience_match(
    job_profile: JobProfile,
    candidate_profile: CandidateProfile,
    weights: dict[str, float] | None = None,
) -> MatcherResult:
    experience_weights = weights or DEFAULT_STAGE4_CONFIG.experience_weights
    current_title = candidate_profile.profile.current_title
    candidate_years = candidate_profile.profile.years_of_experience
    years_score = _experience_match(candidate_years, job_profile.minimum_experience)
    title_score = title_similarity(current_title, job_profile.title)
    domain_score = _domain_experience(candidate_profile, job_profile)

    scores = {
        "years": years_score,
        "title_similarity": title_score,
        "industry_match": domain_score,
    }
    score = weighted_average(scores, experience_weights)

    return MatcherResult(
        score=to_percent(score),
        details={
            "years_score": years_score,
            "title_score": title_score,
            "industry_match": domain_score,
            "weights": dict(experience_weights),
        },
    )
