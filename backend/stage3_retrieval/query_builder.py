from __future__ import annotations

from backend.schema.job_profile import JobProfile


def _join_non_empty(parts: list[str]) -> str:
    return " ".join(part.strip() for part in parts if isinstance(part, str) and part.strip())


def _meaningful(value: str | None) -> str:
    if not isinstance(value, str):
        return ""
    normalized = value.strip()
    if not normalized:
        return ""
    lowered = normalized.lower()
    if lowered in {"unknown", "unspecified", "n/a", "na", "none"}:
        return ""
    return normalized


def build_semantic_query(job_profile: JobProfile) -> str:
    structured_parts = [
        f"Job title: {_meaningful(job_profile.title)}" if _meaningful(job_profile.title) else "",
        f"Company: {_meaningful(job_profile.company)}" if _meaningful(job_profile.company) else "",
        f"Industry: {_meaningful(job_profile.industry)}" if _meaningful(job_profile.industry) else "",
        f"Location: {_meaningful(job_profile.location)}" if _meaningful(job_profile.location) else "",
        f"Work mode: {_meaningful(job_profile.work_mode)}" if _meaningful(job_profile.work_mode) else "",
        f"Minimum experience: {job_profile.minimum_experience:.1f} years" if job_profile.minimum_experience is not None else "",
        f"Required skills: {' '.join(job_profile.required_skills)}" if job_profile.required_skills else "",
        f"Preferred skills: {' '.join(job_profile.preferred_skills)}" if job_profile.preferred_skills else "",
        f"Education: {' '.join(job_profile.education_requirements)}" if job_profile.education_requirements else "",
        f"Domain keywords: {' '.join(job_profile.domain_keywords)}" if job_profile.domain_keywords else "",
        f"Soft skills: {' '.join(job_profile.soft_skills)}" if job_profile.soft_skills else "",
    ]
    query = _join_non_empty(structured_parts)
    if query:
        return query
    return job_profile.raw_text.strip()


def build_bm25_query(job_profile: JobProfile) -> str:
    structured_parts = [
        _meaningful(job_profile.title),
        _meaningful(job_profile.company),
        _meaningful(job_profile.industry),
        _meaningful(job_profile.location),
        _meaningful(job_profile.work_mode),
        f"{job_profile.minimum_experience:.1f} years experience" if job_profile.minimum_experience is not None else "",
        " ".join(job_profile.required_skills),
        " ".join(job_profile.preferred_skills),
        " ".join(job_profile.education_requirements),
        " ".join(job_profile.domain_keywords),
        " ".join(job_profile.soft_skills),
    ]
    query = _join_non_empty(structured_parts)
    if query:
        return query
    return build_semantic_query(job_profile)
