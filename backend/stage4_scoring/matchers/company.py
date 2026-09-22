from __future__ import annotations

from backend.stage4_scoring.config import DEFAULT_STAGE4_CONFIG
from backend.schema.candidate_profile import CandidateProfile
from backend.schema.job_profile import JobProfile

from .common import MatcherResult, clamp01, normalize_alnum_text, normalize_text, to_percent, weighted_average


SIZE_BUCKETS = {
    "1-10": 1,
    "11-50": 2,
    "51-200": 3,
    "201-500": 4,
    "501-1000": 5,
    "1001-5000": 6,
    "5001-10000": 7,
    "10001+": 8,
}


def _size_bucket(value: str) -> int:
    return SIZE_BUCKETS.get(normalize_text(value).upper(), 4)


def score_company_match(
    job_profile: JobProfile,
    candidate_profile: CandidateProfile,
    weights: dict[str, float] | None = None,
) -> MatcherResult:
    company_weights = weights or DEFAULT_STAGE4_CONFIG.company_weights
    candidate_size = _size_bucket(candidate_profile.profile.current_company_size)
    job_company_size = getattr(job_profile, "company_size", None)
    if isinstance(job_company_size, str) and job_company_size.strip():
        target_bucket = _size_bucket(job_company_size)
        company_size_similarity = 1.0 - min(abs(candidate_size - target_bucket) / 7.0, 1.0)
    else:
        company_size_similarity = 0.5

    job_industry = normalize_alnum_text(job_profile.industry)
    career_industries = [
        normalize_alnum_text(candidate_profile.profile.current_industry),
        *[normalize_alnum_text(role.industry) for role in candidate_profile.career_history if role.industry],
    ]
    industry_similarity = max(
        (
            1.0
            if job_industry and (job_industry in industry or industry in job_industry)
            else 0.0
        )
        for industry in career_industries
    ) if career_industries else 0.0

    score = weighted_average(
        {
            "industry_match": industry_similarity,
            "company_size_match": company_size_similarity,
        },
        company_weights,
    )
    return MatcherResult(
        score=to_percent(score),
        details={
            "company_size_similarity": company_size_similarity,
            "industry_similarity": industry_similarity,
            "candidate_company_size_bucket": candidate_size,
            "weights": dict(company_weights),
        },
    )
