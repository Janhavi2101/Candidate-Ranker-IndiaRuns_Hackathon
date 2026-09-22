from __future__ import annotations

from backend.stage4_scoring.config import DEFAULT_STAGE4_CONFIG
from backend.schema.candidate_profile import CandidateProfile
from backend.schema.job_profile import JobProfile

from .common import MatcherResult, clamp01, normalize_text, to_percent, weighted_average


DEGREE_LEVELS = {
    "phd": 4,
    "doctor": 4,
    "master": 3,
    "ms": 3,
    "m.s": 3,
    "m.sc": 3,
    "mba": 3,
    "mtech": 3,
    "msc": 3,
    "bachelor": 2,
    "be": 2,
    "btech": 2,
    "b.sc": 2,
    "bsc": 2,
    "associate": 1,
    "diploma": 1,
}


def _degree_level(text: str) -> int:
    lowered = normalize_text(text)
    for keyword, level in DEGREE_LEVELS.items():
        if keyword in lowered:
            return level
    return 0


def _highest_candidate_degree(candidate_profile: CandidateProfile) -> int:
    if not candidate_profile.education:
        return 0
    return max(_degree_level(education.degree) for education in candidate_profile.education)


def score_education_match(
    job_profile: JobProfile,
    candidate_profile: CandidateProfile,
    weights: dict[str, float] | None = None,
) -> MatcherResult:
    education_weights = weights or DEFAULT_STAGE4_CONFIG.education_weights
    if not job_profile.education_requirements:
        return MatcherResult(score=0.5, details={"education_requirement_present": False})

    requirement_levels = [_degree_level(item) for item in job_profile.education_requirements]
    requirement_level = max(requirement_levels) if requirement_levels else 0
    candidate_level = _highest_candidate_degree(candidate_profile)
    degree_score = 1.0 if candidate_level >= requirement_level and requirement_level > 0 else (0.5 if requirement_level == 0 else clamp01(candidate_level / requirement_level))

    requirement_text = normalize_text(" ".join(job_profile.education_requirements))
    candidate_fields = normalize_text(" ".join(education.field_of_study for education in candidate_profile.education))
    candidate_institutions = normalize_text(" ".join(education.institution for education in candidate_profile.education))
    requirement_tokens = requirement_text.split()
    field_score = 0.5 if not requirement_tokens else (1.0 if any(token in candidate_fields for token in requirement_tokens) else 0.0)
    institution_score = 0.5 if not requirement_tokens else (1.0 if any(token in candidate_institutions for token in requirement_tokens) else 0.0)

    score = weighted_average(
        {
            "degree": degree_score,
            "field": field_score,
            "institution": institution_score,
        },
        education_weights,
    )
    return MatcherResult(
        score=to_percent(score),
        details={
            "degree_score": degree_score,
            "field_score": field_score,
            "institution_score": institution_score,
            "requirement_level": requirement_level,
            "candidate_level": candidate_level,
            "weights": dict(education_weights),
        },
    )
