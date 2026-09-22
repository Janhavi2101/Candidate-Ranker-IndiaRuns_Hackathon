from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Dict, Iterable, List, Mapping, Optional


class CandidateProfileError(ValueError):
    pass


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateProfileError(f"{path} must be an object")
    return value


def _require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise CandidateProfileError(f"{path} must be a list")
    return value


def _require_str(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CandidateProfileError(f"{path} must be a non-empty string")
    return value.strip()


def _optional_str(value: Any, path: str) -> Optional[str]:
    if value is None:
        return None
    return _require_str(value, path)


def _require_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CandidateProfileError(f"{path} must be an integer")
    return value


def _require_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateProfileError(f"{path} must be a number")
    return float(value)


def _require_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise CandidateProfileError(f"{path} must be a boolean")
    return value


def _parse_iso_date(value: Any, path: str) -> str:
    text = _require_str(value, path)
    try:
        date.fromisoformat(text)
    except ValueError as exc:
        raise CandidateProfileError(f"{path} must be an ISO date in YYYY-MM-DD format") from exc
    return text


def _mapping_to_float_dict(value: Any, path: str) -> Dict[str, float]:
    mapping = _require_mapping(value, path)
    parsed: Dict[str, float] = {}
    for key, item in mapping.items():
        parsed[_require_str(key, f"{path} key")] = _require_float(item, f"{path}.{key}")
    return parsed


def _serialize_dataclass(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, list):
        return [_serialize_dataclass(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_dataclass(item) for key, item in value.items()}
    return value


@dataclass(slots=True)
class Profile:
    anonymized_name: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float
    current_title: str
    current_company: str
    current_company_size: str
    current_industry: str

    @classmethod
    def from_dict(cls, value: Any) -> "Profile":
        payload = _require_mapping(value, "profile")
        return cls(
            anonymized_name=_require_str(payload.get("anonymized_name"), "profile.anonymized_name"),
            headline=_require_str(payload.get("headline"), "profile.headline"),
            summary=_require_str(payload.get("summary"), "profile.summary"),
            location=_require_str(payload.get("location"), "profile.location"),
            country=_require_str(payload.get("country"), "profile.country"),
            years_of_experience=_require_float(payload.get("years_of_experience"), "profile.years_of_experience"),
            current_title=_require_str(payload.get("current_title"), "profile.current_title"),
            current_company=_require_str(payload.get("current_company"), "profile.current_company"),
            current_company_size=_require_str(payload.get("current_company_size"), "profile.current_company_size"),
            current_industry=_require_str(payload.get("current_industry"), "profile.current_industry"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CareerHistory:
    company: str
    title: str
    start_date: str
    end_date: Optional[str]
    duration_months: int
    is_current: bool
    industry: str
    company_size: str
    description: str

    @classmethod
    def from_dict(cls, value: Any) -> "CareerHistory":
        payload = _require_mapping(value, "career_history item")
        is_current = _require_bool(payload.get("is_current"), "career_history.is_current")
        end_date = payload.get("end_date")
        if end_date is not None:
            end_date = _parse_iso_date(end_date, "career_history.end_date")
        return cls(
            company=_require_str(payload.get("company"), "career_history.company"),
            title=_require_str(payload.get("title"), "career_history.title"),
            start_date=_parse_iso_date(payload.get("start_date"), "career_history.start_date"),
            end_date=end_date,
            duration_months=_require_int(payload.get("duration_months"), "career_history.duration_months"),
            is_current=is_current,
            industry=_require_str(payload.get("industry"), "career_history.industry"),
            company_size=_require_str(payload.get("company_size"), "career_history.company_size"),
            description=_require_str(payload.get("description"), "career_history.description"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Education:
    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    grade: Optional[str] = None
    tier: str = ""

    @classmethod
    def from_dict(cls, value: Any) -> "Education":
        payload = _require_mapping(value, "education item")
        return cls(
            institution=_require_str(payload.get("institution"), "education.institution"),
            degree=_require_str(payload.get("degree"), "education.degree"),
            field_of_study=_require_str(payload.get("field_of_study"), "education.field_of_study"),
            start_year=_require_int(payload.get("start_year"), "education.start_year"),
            end_year=_require_int(payload.get("end_year"), "education.end_year"),
            grade=_optional_str(payload.get("grade"), "education.grade"),
            tier=_require_str(payload.get("tier"), "education.tier"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Skill:
    name: str
    proficiency: str
    endorsements: int
    duration_months: int = 0

    @classmethod
    def from_dict(cls, value: Any) -> "Skill":
        payload = _require_mapping(value, "skills item")
        return cls(
            name=_require_str(payload.get("name"), "skills.name"),
            proficiency=_require_str(payload.get("proficiency"), "skills.proficiency"),
            endorsements=_require_int(payload.get("endorsements"), "skills.endorsements"),
            duration_months=_require_int(payload.get("duration_months", 0), "skills.duration_months"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Certification:
    name: str
    issuer: str
    year: int

    @classmethod
    def from_dict(cls, value: Any) -> "Certification":
        payload = _require_mapping(value, "certifications item")
        return cls(
            name=_require_str(payload.get("name"), "certifications.name"),
            issuer=_require_str(payload.get("issuer"), "certifications.issuer"),
            year=_require_int(payload.get("year"), "certifications.year"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Language:
    language: str
    proficiency: str

    @classmethod
    def from_dict(cls, value: Any) -> "Language":
        payload = _require_mapping(value, "languages item")
        return cls(
            language=_require_str(payload.get("language"), "languages.language"),
            proficiency=_require_str(payload.get("proficiency"), "languages.proficiency"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SalaryRange:
    minimum: float
    maximum: float

    @classmethod
    def from_dict(cls, value: Any) -> "SalaryRange":
        payload = _require_mapping(value, "redrob_signals.expected_salary_range_inr_lpa")
        minimum = _require_float(payload.get("min"), "redrob_signals.expected_salary_range_inr_lpa.min")
        maximum = _require_float(payload.get("max"), "redrob_signals.expected_salary_range_inr_lpa.max")
        if minimum > maximum:
            minimum, maximum = maximum, minimum
        return cls(minimum=minimum, maximum=maximum)

    def to_dict(self) -> Dict[str, Any]:
        return {"min": self.minimum, "max": self.maximum}


@dataclass(slots=True)
class RedrobSignals:
    profile_completeness_score: float
    signup_date: str
    last_active_date: str
    open_to_work_flag: bool
    profile_views_received_30d: int
    applications_submitted_30d: int
    recruiter_response_rate: float
    avg_response_time_hours: float
    skill_assessment_scores: Dict[str, float]
    connection_count: int
    endorsements_received: int
    notice_period_days: int
    expected_salary_range_inr_lpa: SalaryRange
    preferred_work_mode: str
    willing_to_relocate: bool
    github_activity_score: float
    search_appearance_30d: int
    saved_by_recruiters_30d: int
    interview_completion_rate: float
    offer_acceptance_rate: float
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool

    @classmethod
    def from_dict(cls, value: Any) -> "RedrobSignals":
        payload = _require_mapping(value, "redrob_signals")
        return cls(
            profile_completeness_score=_require_float(payload.get("profile_completeness_score"), "redrob_signals.profile_completeness_score"),
            signup_date=_parse_iso_date(payload.get("signup_date"), "redrob_signals.signup_date"),
            last_active_date=_parse_iso_date(payload.get("last_active_date"), "redrob_signals.last_active_date"),
            open_to_work_flag=_require_bool(payload.get("open_to_work_flag"), "redrob_signals.open_to_work_flag"),
            profile_views_received_30d=_require_int(payload.get("profile_views_received_30d"), "redrob_signals.profile_views_received_30d"),
            applications_submitted_30d=_require_int(payload.get("applications_submitted_30d"), "redrob_signals.applications_submitted_30d"),
            recruiter_response_rate=_require_float(payload.get("recruiter_response_rate"), "redrob_signals.recruiter_response_rate"),
            avg_response_time_hours=_require_float(payload.get("avg_response_time_hours"), "redrob_signals.avg_response_time_hours"),
            skill_assessment_scores=_mapping_to_float_dict(payload.get("skill_assessment_scores", {}), "redrob_signals.skill_assessment_scores"),
            connection_count=_require_int(payload.get("connection_count"), "redrob_signals.connection_count"),
            endorsements_received=_require_int(payload.get("endorsements_received"), "redrob_signals.endorsements_received"),
            notice_period_days=_require_int(payload.get("notice_period_days"), "redrob_signals.notice_period_days"),
            expected_salary_range_inr_lpa=SalaryRange.from_dict(payload.get("expected_salary_range_inr_lpa")),
            preferred_work_mode=_require_str(payload.get("preferred_work_mode"), "redrob_signals.preferred_work_mode"),
            willing_to_relocate=_require_bool(payload.get("willing_to_relocate"), "redrob_signals.willing_to_relocate"),
            github_activity_score=_require_float(payload.get("github_activity_score"), "redrob_signals.github_activity_score"),
            search_appearance_30d=_require_int(payload.get("search_appearance_30d"), "redrob_signals.search_appearance_30d"),
            saved_by_recruiters_30d=_require_int(payload.get("saved_by_recruiters_30d"), "redrob_signals.saved_by_recruiters_30d"),
            interview_completion_rate=_require_float(payload.get("interview_completion_rate"), "redrob_signals.interview_completion_rate"),
            offer_acceptance_rate=_require_float(payload.get("offer_acceptance_rate"), "redrob_signals.offer_acceptance_rate"),
            verified_email=_require_bool(payload.get("verified_email"), "redrob_signals.verified_email"),
            verified_phone=_require_bool(payload.get("verified_phone"), "redrob_signals.verified_phone"),
            linkedin_connected=_require_bool(payload.get("linkedin_connected"), "redrob_signals.linkedin_connected"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_completeness_score": self.profile_completeness_score,
            "signup_date": self.signup_date,
            "last_active_date": self.last_active_date,
            "open_to_work_flag": self.open_to_work_flag,
            "profile_views_received_30d": self.profile_views_received_30d,
            "applications_submitted_30d": self.applications_submitted_30d,
            "recruiter_response_rate": self.recruiter_response_rate,
            "avg_response_time_hours": self.avg_response_time_hours,
            "skill_assessment_scores": dict(self.skill_assessment_scores),
            "connection_count": self.connection_count,
            "endorsements_received": self.endorsements_received,
            "notice_period_days": self.notice_period_days,
            "expected_salary_range_inr_lpa": self.expected_salary_range_inr_lpa.to_dict(),
            "preferred_work_mode": self.preferred_work_mode,
            "willing_to_relocate": self.willing_to_relocate,
            "github_activity_score": self.github_activity_score,
            "search_appearance_30d": self.search_appearance_30d,
            "saved_by_recruiters_30d": self.saved_by_recruiters_30d,
            "interview_completion_rate": self.interview_completion_rate,
            "offer_acceptance_rate": self.offer_acceptance_rate,
            "verified_email": self.verified_email,
            "verified_phone": self.verified_phone,
            "linkedin_connected": self.linkedin_connected,
        }


@dataclass(slots=True)
class CandidateProfile:
    candidate_id: str
    profile: Profile
    career_history: List[CareerHistory]
    education: List[Education]
    skills: List[Skill]
    redrob_signals: RedrobSignals
    certifications: List[Certification] = field(default_factory=list)
    languages: List[Language] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: Any) -> "CandidateProfile":
        payload = _require_mapping(value, "candidate")
        candidate_id = _require_str(payload.get("candidate_id"), "candidate_id")
        profile = Profile.from_dict(payload.get("profile"))
        career_history = [CareerHistory.from_dict(item) for item in _require_list(payload.get("career_history"), "career_history")]
        education = [Education.from_dict(item) for item in _require_list(payload.get("education"), "education")]
        skills = [Skill.from_dict(item) for item in _require_list(payload.get("skills"), "skills")]
        certifications = [Certification.from_dict(item) for item in payload.get("certifications", []) or []]
        languages = [Language.from_dict(item) for item in payload.get("languages", []) or []]
        redrob_signals = RedrobSignals.from_dict(payload.get("redrob_signals"))

        current_roles = [item for item in career_history if item.is_current]
        if current_roles:
            current_role = current_roles[0]
            if profile.current_company != current_role.company or profile.current_title != current_role.title:
                raise CandidateProfileError(
                    "profile current title/company must match the current career_history record"
                )

        return cls(
            candidate_id=candidate_id,
            profile=profile,
            career_history=sorted(career_history, key=lambda item: item.start_date, reverse=True),
            education=sorted(education, key=lambda item: (item.end_year, item.start_year), reverse=True),
            skills=sorted(skills, key=lambda item: (item.endorsements, item.duration_months), reverse=True),
            certifications=certifications,
            languages=languages,
            redrob_signals=redrob_signals,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "profile": self.profile.to_dict(),
            "career_history": [_serialize_dataclass(item) for item in self.career_history],
            "education": [_serialize_dataclass(item) for item in self.education],
            "skills": [_serialize_dataclass(item) for item in self.skills],
            "certifications": [_serialize_dataclass(item) for item in self.certifications],
            "languages": [_serialize_dataclass(item) for item in self.languages],
            "redrob_signals": self.redrob_signals.to_dict(),
        }

    def build_profile_text(self) -> str:
        top_skills = ", ".join(skill.name for skill in self.skills[:8])
        recent_roles = " | ".join(
            f"{role.title} at {role.company}"
            for role in self.career_history[:3]
        )
        education = " | ".join(
            f"{item.degree} {item.field_of_study} @ {item.institution}"
            for item in self.education[:2]
        )
        return " || ".join(
            part
            for part in [
                self.profile.anonymized_name,
                self.profile.headline,
                self.profile.summary,
                f"Current role: {self.profile.current_title} at {self.profile.current_company}",
                f"Experience: {self.profile.years_of_experience:.1f} years",
                f"Career: {recent_roles}" if recent_roles else "",
                f"Education: {education}" if education else "",
                f"Skills: {top_skills}" if top_skills else "",
            ]
            if part
        )
