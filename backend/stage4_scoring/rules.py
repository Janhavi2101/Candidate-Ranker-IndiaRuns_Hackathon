from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Tuple

from backend.schema.job_profile import JobProfile
from backend.schema.candidate_profile import CandidateProfile
from stage1_features.models import CandidateMetadata, RankingFeatures

from .config import Stage4Config
from .matchers.common import normalize_text


@dataclass(slots=True)
class RuleResult:
    score: float
    details: Dict[str, float]


def apply_bonus_rules(
    job_profile: JobProfile,
    candidate_profile: CandidateProfile,
    metadata: CandidateMetadata,
    ranking_features: RankingFeatures,
    config: Stage4Config,
) -> RuleResult:
    details: Dict[str, float] = {}
    total = 0.0

    if metadata.open_to_work:
        total += config.bonus_rules.get("open_to_work", 0.0)
        details["open_to_work"] = config.bonus_rules.get("open_to_work", 0.0)
    if ranking_features.verified_email:
        total += config.bonus_rules.get("verified_email", 0.0)
        details["verified_email"] = config.bonus_rules.get("verified_email", 0.0)
    if ranking_features.verified_phone:
        total += config.bonus_rules.get("verified_phone", 0.0)
        details["verified_phone"] = config.bonus_rules.get("verified_phone", 0.0)
    if ranking_features.linkedin_connected:
        total += config.bonus_rules.get("linkedin_connected", 0.0)
        details["linkedin_connected"] = config.bonus_rules.get("linkedin_connected", 0.0)
    if ranking_features.github_activity_score >= config.thresholds.get("exceptional_github_activity", 80.0):
        bonus = config.bonus_rules.get("exceptional_github_activity", 0.0)
        total += bonus
        details["exceptional_github_activity"] = bonus
    if ranking_features.skill_assessment_scores:
        average_score = sum(ranking_features.skill_assessment_scores.values()) / len(ranking_features.skill_assessment_scores)
        if average_score >= config.thresholds.get("excellent_assessment_score", 70.0):
            bonus = config.bonus_rules.get("excellent_assessment_scores", 0.0)
            total += bonus
            details["excellent_assessment_scores"] = bonus
    if ranking_features.recruiter_response_rate >= config.thresholds.get("high_recruiter_response_rate", 0.65):
        bonus = config.bonus_rules.get("high_recruiter_engagement", 0.0)
        total += bonus
        details["high_recruiter_engagement"] = bonus

    return RuleResult(score=total, details=details)


def apply_penalty_rules(
    job_profile: JobProfile,
    candidate_profile: CandidateProfile,
    metadata: CandidateMetadata,
    ranking_features: RankingFeatures,
    config: Stage4Config,
) -> RuleResult:
    details: Dict[str, float] = {}
    total = 0.0

    candidate_skill_names = {normalize_text(skill.name) for skill in candidate_profile.skills}
    missing_required = [skill for skill in job_profile.required_skills if normalize_text(skill) not in candidate_skill_names]
    if missing_required:
        penalty = config.penalty_rules.get("missing_required_skill", 0.0) * len(missing_required)
        penalty = min(penalty, config.penalty_caps.get("missing_required_skill", penalty))
        total += penalty
        details["missing_required_skill"] = penalty

    if job_profile.minimum_experience is not None and candidate_profile.profile.years_of_experience < job_profile.minimum_experience:
        gap = job_profile.minimum_experience - candidate_profile.profile.years_of_experience
        penalty = config.penalty_rules.get("experience_gap_per_year", 0.0) * gap
        penalty = min(penalty, config.penalty_caps.get("experience_gap", penalty))
        total += penalty
        details["experience_gap"] = penalty

    if ranking_features.notice_period_days > config.thresholds.get("long_notice_period_days", 90.0):
        penalty = config.penalty_rules.get("notice_period_long", 0.0)
        total += penalty
        details["notice_period_long"] = penalty

    if job_profile.salary is not None and candidate_profile.redrob_signals.expected_salary_range_inr_lpa:
        candidate_salary_max = candidate_profile.redrob_signals.expected_salary_range_inr_lpa.maximum
        if candidate_salary_max > job_profile.salary.maximum + config.thresholds.get("salary_buffer", 0.0):
            penalty = config.penalty_rules.get("salary_above_budget", 0.0)
            total += penalty
            details["salary_above_budget"] = penalty

    if metadata.preferred_work_mode and job_profile.work_mode:
        candidate_mode = normalize_text(metadata.preferred_work_mode)
        job_mode = normalize_text(job_profile.work_mode)
        compatible_modes = {
            "flexible": {"hybrid", "remote", "onsite", "unspecified", "flexible"},
            "hybrid": {"hybrid", "flexible"},
            "remote": {"remote", "flexible"},
            "onsite": {"onsite", "flexible"},
            "unspecified": {"unspecified", "flexible"},
        }
        if job_mode not in compatible_modes.get(candidate_mode, {candidate_mode}):
            penalty = config.penalty_rules.get("work_mode_mismatch", 0.0)
            total += penalty
            details["work_mode_mismatch"] = penalty

    if job_profile.location and metadata.location and metadata.location.lower() not in job_profile.location.lower() and not metadata.willing_to_relocate:
        penalty = config.penalty_rules.get("relocation_mismatch", 0.0) * 0.5
        total += penalty
        details["relocation_mismatch"] = penalty

    if ranking_features.profile_completeness_score < config.thresholds.get("low_profile_quality", 70.0):
        penalty = config.penalty_rules.get("low_profile_quality", 0.0)
        total += penalty
        details["low_profile_quality"] = penalty

    if ranking_features.recruiter_response_rate < config.thresholds.get("low_recruiter_response_rate", 0.3):
        penalty = config.penalty_rules.get("low_recruiter_response", 0.0)
        total += penalty
        details["low_recruiter_response"] = penalty

    last_active_days = _days_since(candidate_profile.redrob_signals.last_active_date)
    if last_active_days is not None and last_active_days > config.thresholds.get("inactive_profile_days", 45.0):
        penalty = config.penalty_rules.get("inactive_profile", 0.0)
        total += penalty
        details["inactive_profile"] = penalty

    if ranking_features.career_stability is not None and ranking_features.career_stability < 0.5:
        penalty = config.penalty_rules.get("poor_career_stability", 0.0)
        total += penalty
        details["poor_career_stability"] = penalty

    return RuleResult(score=total, details=details)


def _days_since(iso_date_text: str) -> float | None:
    try:
        parsed = date.fromisoformat(iso_date_text)
    except ValueError:
        return None
    return float((date.today() - parsed).days)
