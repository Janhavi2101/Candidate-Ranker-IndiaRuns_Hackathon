from __future__ import annotations

from typing import Iterable, List

from backend.schema.candidate_profile import CandidateProfile

from .models import CandidateDocument, CandidateMetadata, RankingFeatures


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().split())


def _normalize_lower(value: str) -> str:
    return _normalize_text(value).lower()


def _normalized_unique_items(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    normalized: List[str] = []
    for value in values:
        item = _normalize_lower(value)
        if not item or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return normalized


def _format_section(title: str, lines: List[str]) -> str:
    if not lines:
        return f"{title}: None"
    return "\n".join([f"{title}:", *[f"- {line}" for line in lines]])


def _build_profile_section(candidate: CandidateProfile) -> List[str]:
    profile = candidate.profile
    return [
        f"Headline: {profile.headline}",
        f"Summary: {profile.summary}",
        f"Years of experience: {profile.years_of_experience:.1f}",
        f"Current title: {profile.current_title}",
        f"Current company: {profile.current_company}",
        f"Current industry: {profile.current_industry}",
        f"Location: {profile.location}, {profile.country}",
    ]


def _build_career_history_section(candidate: CandidateProfile) -> List[str]:
    if not candidate.career_history:
        return []
    lines: List[str] = []
    for item in candidate.career_history:
        lines.append(f"{item.title} at {item.company} ({item.industry})")
        lines.append(item.description)
    return lines


def _build_education_section(candidate: CandidateProfile) -> List[str]:
    if not candidate.education:
        return []
    lines: List[str] = []
    for item in candidate.education:
        lines.append(f"{item.degree} in {item.field_of_study} at {item.institution}")
    return lines


def _build_skills_section(candidate: CandidateProfile) -> List[str]:
    return [skill.name for skill in candidate.skills]


def _build_certifications_section(candidate: CandidateProfile) -> List[str]:
    if not candidate.certifications:
        return []
    return [f"{item.name} — {item.issuer} ({item.year})" for item in candidate.certifications]


def _build_languages_section(candidate: CandidateProfile) -> List[str]:
    if not candidate.languages:
        return []
    return [f"{item.language} ({item.proficiency})" for item in candidate.languages]


def build_embedding_text(candidate: CandidateProfile) -> str:
    sections = [
        "\n".join(_build_profile_section(candidate)),
        _format_section("Career history", _build_career_history_section(candidate)),
        _format_section("Education", _build_education_section(candidate)),
        _format_section("Skills", _build_skills_section(candidate)),
        _format_section("Certifications", _build_certifications_section(candidate)),
        _format_section("Languages", _build_languages_section(candidate)),
    ]
    return "\n\n".join(section for section in sections if section)


def build_metadata(candidate: CandidateProfile) -> CandidateMetadata:
    return CandidateMetadata(
        candidate_id=candidate.candidate_id,
        years_of_experience=candidate.profile.years_of_experience,
        location=candidate.profile.location,
        country=candidate.profile.country,
        current_industry=candidate.profile.current_industry,
        current_company_size=candidate.profile.current_company_size,
        open_to_work=candidate.redrob_signals.open_to_work_flag,
        preferred_work_mode=_normalize_lower(candidate.redrob_signals.preferred_work_mode),
        willing_to_relocate=candidate.redrob_signals.willing_to_relocate,
        skills=_normalized_unique_items(skill.name for skill in candidate.skills),
        languages=_normalized_unique_items(language.language for language in candidate.languages),
    )


def build_ranking_features(candidate: CandidateProfile) -> RankingFeatures:
    signals = candidate.redrob_signals
    return RankingFeatures(
        profile_completeness_score=signals.profile_completeness_score,
        github_activity_score=signals.github_activity_score,
        recruiter_response_rate=signals.recruiter_response_rate,
        interview_completion_rate=signals.interview_completion_rate,
        offer_acceptance_rate=signals.offer_acceptance_rate,
        profile_views_received_30d=signals.profile_views_received_30d,
        search_appearance_30d=signals.search_appearance_30d,
        saved_by_recruiters_30d=signals.saved_by_recruiters_30d,
        endorsements_received=signals.endorsements_received,
        connection_count=signals.connection_count,
        notice_period_days=signals.notice_period_days,
        applications_submitted_30d=signals.applications_submitted_30d,
        avg_response_time_hours=signals.avg_response_time_hours,
        open_to_work_flag=signals.open_to_work_flag,
        willing_to_relocate=signals.willing_to_relocate,
        verified_email=signals.verified_email,
        verified_phone=signals.verified_phone,
        linkedin_connected=signals.linkedin_connected,
        skill_assessment_scores=dict(signals.skill_assessment_scores),
        expected_salary_range_inr_lpa=signals.expected_salary_range_inr_lpa.to_dict(),
    )


def build_candidate_document(candidate: CandidateProfile) -> CandidateDocument:
    return CandidateDocument(
        candidate_id=candidate.candidate_id,
        embedding_text=build_embedding_text(candidate),
        metadata=build_metadata(candidate),
        ranking_features=build_ranking_features(candidate),
    )

