from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class SalaryRange:
    minimum: float
    maximum: float
    currency: str = "INR"
    unit: str = "LPA"
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class JobProfile:
# ---------- Basic ----------
    title: str
    company: Optional[str] = None
    industry: str = "Unknown"

    # ---------- Location ----------
    location: str = "Unknown"
    work_mode: str = "unspecified"

    # ---------- Experience ----------
    minimum_experience: Optional[float] = None

    # ---------- Compensation ----------
    salary: Optional[SalaryRange] = None

    # ---------- Skills ----------
    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    soft_skills: list[str] = field(default_factory=list)

    # ---------- Responsibilities ----------
    responsibilities: list[str] = field(default_factory=list)

    # ---------- Domain ----------
    domain_keywords: list[str] = field(default_factory=list)

    # ---------- Education ----------
    education_requirements: list[str] = field(default_factory=list)

    # ---------- Original JD ----------
    raw_text: str = ""

    source_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.salary is not None:
            payload["salary"] = self.salary.to_dict()
        return payload
