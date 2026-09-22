from __future__ import annotations

import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.preprocessing.jd_parser import parse_job_profile, parse_job_profile_from_text


SAMPLE_JD = """
Job Title: Senior Backend Engineer
Industry: Fintech
Location: Bengaluru, India
Work Mode: Hybrid
Minimum Experience: 5+ years of experience

Required Skills
- Python
- FastAPI
- SQL
- Kafka

Preferred Skills
- AWS
- Docker
- Kubernetes

Salary: 25 to 40 LPA
""".strip()


def _write_pdf(path: Path, text: str) -> None:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=11)
    doc.save(path)
    doc.close()


def test_parse_job_profile_from_text() -> None:
    profile = parse_job_profile_from_text(SAMPLE_JD, source_name="sample-jd.txt")

    assert profile.title == "Senior Backend Engineer"
    assert profile.industry == "Fintech"
    assert profile.location == "Bengaluru, India"
    assert profile.work_mode == "hybrid"
    assert profile.minimum_experience == 5.0
    assert profile.required_skills == ["python", "fastapi", "sql", "kafka"]
    assert profile.preferred_skills == ["aws", "docker", "kubernetes"]
    assert profile.salary is not None
    assert profile.salary.minimum == 25.0
    assert profile.salary.maximum == 40.0


def test_parse_job_profile_from_pdf() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / "sample_jd.pdf"
        _write_pdf(pdf_path, SAMPLE_JD)
        profile = parse_job_profile(pdf_path)

    assert profile.title == "Senior Backend Engineer"
    assert profile.industry == "Fintech"
    assert profile.location == "Bengaluru, India"
    assert profile.work_mode == "hybrid"
    assert profile.minimum_experience == 5.0
    assert profile.required_skills == ["python", "fastapi", "sql", "kafka"]
    assert profile.preferred_skills == ["aws", "docker", "kubernetes"]
    assert profile.salary is not None
    assert profile.salary.minimum == 25.0
    assert profile.salary.maximum == 40.0


if __name__ == "__main__":
    test_parse_job_profile_from_text()
    test_parse_job_profile_from_pdf()
    print("JD parser smoke tests passed")

