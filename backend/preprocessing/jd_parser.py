from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
from docx import Document

import pdfplumber

from backend.schema.job_profile import JobProfile, SalaryRange


SECTION_HEADERS = {
    "required": [
        "required skills",
        "requirements",
        "must have",
        "must-haves",
        "must haves",
        "qualifications",
        "what you need",
        "what we need",
    ],
    "preferred": [
        "preferred skills",
        "preferred qualifications",
        "nice to have",
        "good to have",
        "bonus",
        "bonus points",
        "nice-to-have",
    ],
    "education": [
        "education",
        "education requirements",
        "educational requirements",
        "qualifications",
        "academic qualifications",
    ],
}

TITLE_LABELS = ["job title", "title", "role", "position", "opening for"]
INDUSTRY_LABELS = ["industry", "domain", "sector"]
LOCATION_LABELS = ["location", "based in", "work location", "office location"]
WORK_MODE_LABELS = ["work mode", "mode", "work arrangement", "employment type"]

SKILL_SPLIT_PATTERN = re.compile(r"[•·\-\u2022\t]|,|/|;")
EXPERIENCE_PATTERN = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?|yr)\s+(?:of\s+)?experience",
    re.IGNORECASE,
)
SALARY_PATTERN = re.compile(
    r"(?P<min>\d+(?:\.\d+)?)\s*(?:-|to|–|—)\s*(?P<max>\d+(?:\.\d+)?)\s*(?P<unit>lpa|lac|lakh|lakhs|k|kpa|pa)?",
    re.IGNORECASE,
)


def _normalize_whitespace(text: str) -> str:
    return "\n".join(line.strip() for line in text.replace("\r", "\n").split("\n"))


def _clean_text(text: str) -> str:
    return _normalize_whitespace(text).strip()


def _normalize_item(value: str) -> str:
    return " ".join(value.strip().split())


def _normalize_skill(value: str) -> str:
    return _normalize_item(value).lower()


def _dedupe_preserve_order(values: Iterable[str], normalize=lambda item: item) -> List[str]:
    seen = set()
    output: List[str] = []
    for value in values:
        item = normalize(value)
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _extract_pdf_text(pdf_path: str | Path) -> str:
    path = Path(pdf_path)
    texts: List[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                texts.append(page_text)
    text = _clean_text("\n".join(texts))
    if text:
        return text

    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - optional fallback
        raise ValueError(f"Could not extract text from {path}") from exc

    doc = fitz.open(str(path))
    try:
        fallback_texts = []
        for page in doc:
            page_text = page.get_text().strip()
            if page_text:
                fallback_texts.append(page_text)
        text = _clean_text("\n".join(fallback_texts))
        if text:
            return text
    finally:
        doc.close()

    raise ValueError(f"Could not extract text from {path}")


def _split_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _first_match_label(lines: Sequence[str], labels: Sequence[str]) -> Optional[str]:
    for line in lines:
        lowered = line.lower()
        for label in labels:
            if lowered.startswith(label):
                value = line.split(":", 1)[1].strip() if ":" in line else line[len(label):].strip(" :-")
                if value:
                    return value
    return None


def _first_meaningful_line(lines: Sequence[str]) -> str:
    for line in lines:
        lowered = line.lower()
        if lowered.startswith(("job description", "job summary", "about the role", "about us")):
            continue
        if len(line) >= 4:
            return line
    return ""


def _find_section_start(lines: Sequence[str], header_candidates: Sequence[str]) -> Optional[int]:
    for index, line in enumerate(lines):
        lowered = line.lower().rstrip(":")
        if lowered in header_candidates:
            return index
    return None


def _collect_section_bullets(lines: Sequence[str], start_index: int) -> List[str]:
    collected: List[str] = []
    for line in lines[start_index + 1 :]:
        lowered = line.lower().rstrip(":")
        if lowered and any(lowered in header_group for header_group in SECTION_HEADERS.values()):
            break
        if lowered.startswith(("responsibilities", "about the role", "about us", "what you will do", "salary", "compensation", "package", "ctc")):
            break
        if collected and ":" in line and not line.startswith(("-", "•", "*", "–", "—")):
            break
        if line.startswith(("-", "•", "*", "–", "—")):
            item = line.lstrip("-•*–— ").strip()
            if item:
                collected.append(item)
        elif line and len(line.split()) <= 8 and ":" not in line and not collected:
            continue
        elif collected and len(line.split()) <= 18:
            collected.append(line)
    return collected


def _extract_skills_from_text_block(text_block: str) -> List[str]:
    raw_items = []
    for chunk in SKILL_SPLIT_PATTERN.split(text_block):
        item = _normalize_item(chunk)
        if not item:
            continue
        if len(item) > 60:
            continue
        if item.lower() in {"and", "or", "the", "of", "a", "an"}:
            continue
        raw_items.append(item)
    return _dedupe_preserve_order(raw_items, normalize=_normalize_skill)


def _extract_experience(lines: Sequence[str], text: str) -> Optional[float]:
    label_value = _first_match_label(lines, ["experience", "minimum experience", "years of experience"])
    if label_value:
        match = EXPERIENCE_PATTERN.search(label_value)
        if match:
            return float(match.group("value"))
        number = re.search(r"\d+(?:\.\d+)?", label_value)
        if number:
            return float(number.group(0))

    for match in EXPERIENCE_PATTERN.finditer(text):
        return float(match.group("value"))
    return None


def _extract_salary(text: str) -> Optional[SalaryRange]:
    match = SALARY_PATTERN.search(text)
    if not match:
        return None
    minimum = float(match.group("min"))
    maximum = float(match.group("max"))
    if minimum > maximum:
        minimum, maximum = maximum, minimum
    unit = match.group("unit") or "LPA"
    unit = unit.upper()
    if unit in {"LAC", "LAKH", "LAKHS"}:
        unit = "LPA"
    return SalaryRange(
        minimum=minimum,
        maximum=maximum,
        currency="INR",
        unit=unit,
        raw_text=match.group(0),
    )


def _extract_primary_fields(lines: Sequence[str], text: str) -> dict:
    title = _first_match_label(lines, TITLE_LABELS)
    industry = _first_match_label(lines, INDUSTRY_LABELS)
    location = _first_match_label(lines, LOCATION_LABELS)
    work_mode = _first_match_label(lines, WORK_MODE_LABELS)

    if not title:
        title = _first_meaningful_line(lines)

    lowered_text = text.lower()
    if not work_mode:
        if "work from home" in lowered_text or "wfh" in lowered_text or "remote" in lowered_text:
            work_mode = "remote"
        elif "hybrid" in lowered_text:
            work_mode = "hybrid"
        elif "onsite" in lowered_text or "on-site" in lowered_text or "office-based" in lowered_text:
            work_mode = "onsite"

    return {
        "title": _normalize_item(title or "Unknown Role"),
        "industry": _normalize_item(industry or "Unknown"),
        "location": _normalize_item(location or "Unknown"),
        "work_mode": _normalize_item(work_mode or "unspecified").lower(),
    }


def _extract_skills(lines: Sequence[str], text: str) -> dict:
    required: List[str] = []
    preferred: List[str] = []

    for index, line in enumerate(lines):
        lowered = line.lower().rstrip(":")
        if lowered in SECTION_HEADERS["required"]:
            for item in _collect_section_bullets(lines, index):
                required.extend(_extract_skills_from_text_block(item))
        elif lowered in SECTION_HEADERS["preferred"]:
            for item in _collect_section_bullets(lines, index):
                preferred.extend(_extract_skills_from_text_block(item))

    if not required:
        for pattern in [r"must have[:\s-]*(.+)", r"required skills[:\s-]*(.+)", r"requirements[:\s-]*(.+)", r"qualifications[:\s-]*(.+)"]:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if match:
                required.extend(_extract_skills_from_text_block(match.group(1).split("\n", 1)[0]))
                break

    if not preferred:
        for pattern in [r"preferred skills[:\s-]*(.+)", r"nice to have[:\s-]*(.+)", r"good to have[:\s-]*(.+)", r"bonus[:\s-]*(.+)"]:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if match:
                preferred.extend(_extract_skills_from_text_block(match.group(1).split("\n", 1)[0]))
                break

    if not required and not preferred:
        skill_match = re.search(r"(skills?|tech stack|technologies)[:\s-]*(.+)", text, flags=re.IGNORECASE)
        if skill_match:
            required.extend(_extract_skills_from_text_block(skill_match.group(2).split("\n", 1)[0]))

    return {
        "required_skills": _dedupe_preserve_order(required, normalize=_normalize_skill),
        "preferred_skills": _dedupe_preserve_order(preferred, normalize=_normalize_skill),
    }


def _extract_education_requirements(lines: Sequence[str], text: str) -> List[str]:
    education: List[str] = []
    for index, line in enumerate(lines):
        lowered = line.lower().rstrip(":")
        if lowered in SECTION_HEADERS["education"]:
            for item in _collect_section_bullets(lines, index):
                education.append(_normalize_item(item))

    if not education:
        patterns = [
            r"(bachelor(?:'s)? degree[^.\n]*)",
            r"(master(?:'s)? degree[^.\n]*)",
            r"(ph\.?d[^.\n]*)",
            r"(mba[^.\n]*)",
            r"(computer science[^.\n]*)",
            r"(engineering degree[^.\n]*)",
        ]
        lowered = text.lower()
        for pattern in patterns:
            for match in re.finditer(pattern, lowered, flags=re.IGNORECASE):
                education.append(_normalize_item(match.group(1)))

    return _dedupe_preserve_order(education, normalize=lambda item: item.lower())


def parse_job_profile_from_text(text: str, source_name: str = "") -> JobProfile:
    cleaned = _clean_text(text)
    lines = _split_lines(cleaned)
    fields = _extract_primary_fields(lines, cleaned)
    skills = _extract_skills(lines, cleaned)
    education_requirements = _extract_education_requirements(lines, cleaned)
    minimum_experience = _extract_experience(lines, cleaned)
    salary = _extract_salary(cleaned)
    return JobProfile(
        title=fields["title"],
        industry=fields["industry"],
        location=fields["location"],
        work_mode=fields["work_mode"],
        minimum_experience=minimum_experience,
        salary=salary,
        required_skills=skills["required_skills"],
        preferred_skills=skills["preferred_skills"],
        education_requirements=education_requirements,
        raw_text=cleaned,
        source_name=source_name,
    )

def _extract_docx_text(path: Path) -> str:
    """Extract text from a .docx file."""
    doc = Document(path)
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)

###def parse_job_profile(source: str | Path) -> JobProfile:
    path = Path(source)
    if path.exists() and path.is_file():
        return parse_job_profile_from_text(_extract_pdf_text(path), source_name=path.name)
    return parse_job_profile_from_text(str(source), source_name="")
###
def parse_job_profile(source: str | Path) -> JobProfile:
    path = Path(source)

    if path.exists() and path.is_file():
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            text = _extract_pdf_text(path)

        elif suffix == ".txt":
            text = path.read_text(encoding="utf-8")

        elif suffix == ".docx":
            text = _extract_docx_text(path)

        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        return parse_job_profile_from_text(
            text,
            source_name=path.name,
        )

    # If the input isn't a file, treat it as raw job description text.
    return parse_job_profile_from_text(
        str(source),
        source_name="",
    )