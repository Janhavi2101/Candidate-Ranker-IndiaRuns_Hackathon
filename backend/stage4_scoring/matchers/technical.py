from __future__ import annotations

from backend.stage4_scoring.config import DEFAULT_STAGE4_CONFIG
from backend.schema.job_profile import JobProfile
from backend.schema.candidate_profile import CandidateProfile

from .common import MatcherResult, PROFICIENCY_SCORES, average, clamp01, jaccard_similarity, normalize_text, to_percent, weighted_average


def score_technical_match(
    job_profile: JobProfile,
    candidate_profile: CandidateProfile,
    weights: dict[str, float] | None = None,
) -> MatcherResult:
    technical_weights = weights or DEFAULT_STAGE4_CONFIG.technical_weights
    candidate_skills = {normalize_text(skill.name): skill for skill in candidate_profile.skills}
    required = [normalize_text(skill) for skill in job_profile.required_skills]
    preferred = [normalize_text(skill) for skill in job_profile.preferred_skills]
    job_skills = required + preferred

    required_matches = [skill for skill in required if skill in candidate_skills]
    preferred_matches = [skill for skill in preferred if skill in candidate_skills]

    required_coverage = len(required_matches) / len(required) if required else 0.5
    preferred_coverage = len(preferred_matches) / len(preferred) if preferred else 0.5
    skill_similarity = jaccard_similarity(candidate_skills.keys(), job_skills) if job_skills else 0.5

    proficiency_scores = []
    for skill_name in required_matches:
        proficiency = candidate_skills[skill_name].proficiency.lower()
        proficiency_scores.append(PROFICIENCY_SCORES.get(proficiency, 0.5))
    proficiency_overlap = average(proficiency_scores) if proficiency_scores else (0.5 if not required else 0.0)

    scores = {
        "required_skill_coverage": required_coverage,
        "preferred_skill_coverage": preferred_coverage,
        "skill_similarity": skill_similarity,
        "skill_proficiency": proficiency_overlap,
    }
    score = weighted_average(scores, technical_weights)

    return MatcherResult(
        score=to_percent(score),
        details={
            "required_coverage": required_coverage,
            "preferred_coverage": preferred_coverage,
            "skill_similarity": skill_similarity,
            "proficiency_overlap": proficiency_overlap,
            "weights": dict(technical_weights),
            "required_matches": required_matches,
            "preferred_matches": preferred_matches,
        },
    )
