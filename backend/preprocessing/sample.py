from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pdfplumber
from docx import Document

from backend.schema.job_profile import JobProfile, SalaryRange

TITLE_HEADERS = [
    "job title",
    "title",
    "role",
    "position",
]

LOCATION_HEADERS = [
    "location",
    "work location",
    "office",
    "based in",
]
COMPANY_HEADERS = [
    "company",
    "organization",
    "employer",
]
RESPONSIBILITY_HEADERS = [

    "responsibilities",

    "what you'll do",

    "what you will do",

    "your responsibilities",

    "key responsibilities",

    "day-to-day",

]

WORK_MODE_KEYWORDS = {
    "remote": ["remote", "work from home", "wfh"],
    "hybrid": ["hybrid"],
    "onsite": ["onsite", "on-site", "office"],
}
INDUSTRY_KEYWORDS = {
    "AI / ML": [
        "llm",
        "machine learning",
        "deep learning",
        "embedding",
        "retrieval",
        "rag",
        "vector database",
        "faiss",
        "pinecone",
        "qdrant",
        "weaviate",
        "langchain",
    ],

    "Backend": [
        "django",
        "flask",
        "fastapi",
        "spring",
        "node",
        "express",
    ],

    "Frontend": [
        "react",
        "angular",
        "vue",
        "javascript",
        "typescript",
    ],

    "Data Engineering": [
        "spark",
        "kafka",
        "hadoop",
        "etl",
        "airflow",
    ],
}

DEGREES = {
    "b.tech",
    "btech",
    "be",
    "b.e",
    "bachelor",
    "master",
    "m.tech",
    "mtech",
    "mba",
    "phd",
    "computer science",
}
SKILL_ALIASES = {

    "python": "Python",

    "java": "Java",

    "c++": "C++",

    "golang": "Go",

    "go": "Go",

    "fastapi": "FastAPI",

    "flask": "Flask",

    "django": "Django",

    "react": "React",

    "node": "Node.js",

    "express": "Express",

    "tensorflow": "TensorFlow",

    "pytorch": "PyTorch",

    "scikit-learn": "Scikit-learn",

    "numpy": "NumPy",

    "pandas": "Pandas",

    "sql": "SQL",

    "postgres": "PostgreSQL",

    "postgresql": "PostgreSQL",

    "mongodb": "MongoDB",

    "redis": "Redis",

    "docker": "Docker",

    "kubernetes": "Kubernetes",

    "aws": "AWS",

    "azure": "Azure",

    "gcp": "GCP",

    "git": "Git",

    "langchain": "LangChain",

    "langgraph": "LangGraph",

    "llm": "LLM",

    "rag": "RAG",

    "embedding": "Embeddings",

    "embeddings": "Embeddings",

    "retrieval": "Retrieval",

    "ranking": "Ranking",

    "faiss": "FAISS",

    "pinecone": "Pinecone",

    "qdrant": "Qdrant",

    "weaviate": "Weaviate",

    "milvus": "Milvus",

    "opensearch": "OpenSearch",

    "elasticsearch": "Elasticsearch",

    "bm25": "BM25",

    "lora": "LoRA",

    "qlora": "QLoRA",

    "peft": "PEFT",

    "ndcg": "NDCG",

    "mrr": "MRR",

    "map": "MAP",

}
SOFT_SKILLS = {

    "communication",

    "leadership",

    "ownership",

    "mentoring",

    "collaboration",

    "teamwork",

    "problem solving",

    "critical thinking",

    "adaptability",

    "decision making",

    "stakeholder management",

    "presentation",

    "writing",

}
EDUCATION_KEYWORDS = {
    "Bachelor",
    "Master",
    "B.Tech",
    "BE",
    "M.Tech",
    "MBA",
    "MS",
    "PhD",
    "Computer Science",
    "Information Technology", "Artificial Intelligence", "Data Science", "Software Engineering", }

EXPERIENCE_REGEXES = [

    # 5+ years
    r"(\d+(?:\.\d+)?)\s*\+\s*(?:years?|yrs?)",

    # 5-7 years / 5–7 years
    r"(\d+(?:\.\d+)?)\s*[-–—]\s*\d+(?:\.\d+)?\s*(?:years?|yrs?)",

    # Minimum 5 years
    r"minimum\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)",

    # Experience Required: 5 years
    r"experience\s*required\s*:?\s*(\d+(?:\.\d+)?)",

    # 5 yrs
    r"(\d+(?:\.\d+)?)\s*yrs?",

    # 5 YOE
    r"(\d+(?:\.\d+)?)\s*yoe",

]
SALARY_KEYWORDS = [

    "salary",

    "ctc",

    "package",

    "compensation",

    "pay",

    "salary range",

    "annual compensation",

]

def extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF file."""
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text())
    return _clean_text("\n".join(filter(None, text)))

def extract_docx_text(path: Path) -> str:
    """Extract text from a .docx file."""
    doc = Document(path)
    return _clean_text("\n".join(paragraph.text for paragraph in doc.paragraphs))

def extract_txt_text(path: Path) -> str:
    """Extract text from a .txt file."""
    return _clean_text(path.read_text(encoding="utf-8"))

def _normalize_whitespace(text: str) -> str:
    return "\n".join(line.strip() for line in text.replace("\r", "\n").split("\n"))


def _clean_text(text: str) -> str:
    return _normalize_whitespace(text).strip()


def _split_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]

def find_label_value(lines: Sequence[str], labels: Sequence[str]) -> Optional[str]:
    """
    Finds values like:

    Title: Senior AI Engineer
    Location: Pune
    Role: Backend Developer
    """

    for line in lines:
        stripped = line.strip()

        for label in labels:
            pattern = rf"^{re.escape(label)}\s*:\s*(.+)$"

            match = re.match(pattern, stripped, flags=re.IGNORECASE)

            if match:
                return match.group(1).strip()

    return None

def extract_title(lines: Sequence[str]) -> str:

    title = find_label_value(lines, TITLE_HEADERS)

    if title:
        return title

    for line in lines:

        if line.lower().startswith("job description"):

            parts = line.split(":", 1)

            if len(parts) == 2:
                return parts[1].strip()

    for line in lines[:15]:

        if (
            3 <= len(line.split()) <= 8
            and not line.endswith(":")
            and "company" not in line.lower()
            and "location" not in line.lower()
        ):
            return line

    return "Unknown Role"

def extract_location(lines: Sequence[str]) -> str:

    location = find_label_value(lines, LOCATION_HEADERS)

    if location:
        return location

    for line in lines:

        if line.lower().startswith("location:"):

            return line.split(":",1)[1].strip()

    return "Unknown"

def extract_work_mode(text: str) -> str:

    lower = text.lower()

    for mode, keywords in WORK_MODE_KEYWORDS.items():

        for keyword in keywords:

            if keyword in lower:
                return mode

    return "unspecified"

###def extract_experience(text: str) -> Optional[float]:

    lower = text.lower()

    for pattern in EXPERIENCE_REGEXES:

        match = re.search(pattern, lower)

        if match:

            return float(match.group(1))

    return None###
def extract_experience(lines: list[str]) -> Optional[float]:

    for line in lines:

        lower = line.lower()

        if "experience" not in lower:
            continue

        # Experience Required: 5–9 years
        match = re.search(
            r"(\d+(?:\.\d+)?)\s*[-–—]\s*\d+(?:\.\d+)?",
            lower,
        )

        if match:
            return float(match.group(1))

        # Experience Required: 5+ years
        match = re.search(
            r"(\d+(?:\.\d+)?)\+?",
            lower,
        )

        if match:
            return float(match.group(1))

    return None

def extract_salary(text: str) -> Optional[SalaryRange]:

    lower = text.lower()

    keyword_position = -1

    for word in SALARY_KEYWORDS:

        keyword_position = lower.find(word)

        if keyword_position != -1:
            break

    if keyword_position == -1:
        return None

    nearby = text[keyword_position : keyword_position + 250]    #===================

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)",
        nearby,
    )

    if not match:
        return None

    return SalaryRange(
        minimum=float(match.group(1)),
        maximum=float(match.group(2)),
        currency="INR",
        unit="LPA",
        raw_text=match.group(0),
    )

def infer_industry(text: str) -> str:

    lower = text.lower()

    scores = {}

    for industry, keywords in INDUSTRY_KEYWORDS.items():

        score = 0

        for keyword in keywords:

            if keyword.lower() in lower:
                score += 1

        scores[industry] = score

    if not scores:
        return "Unknown"

    best = max(scores, key=scores.get)

    if scores[best] == 0:
        return "Unknown"

    return best

def find_skills(text: str) -> list[str]:

    lower = text.lower()

    found = []

    for alias, canonical in SKILL_ALIASES.items():

        pattern = rf"\b{re.escape(alias)}\b"

        if re.search(pattern, lower):

            found.append(canonical)

    return sorted(set(found))

def get_section(text: str, start_patterns, end_patterns):

    lower = text.lower()

    start = -1

    for pattern in start_patterns:

        start = lower.find(pattern.lower())

        if start != -1:
            break

    if start == -1:
        return ""

    end = len(text)

    for pattern in end_patterns:

        idx = lower.find(pattern.lower(), start + 1)

        if idx != -1:

            end = min(end, idx)

    return text[start:end]

def extract_required_skills(text: str):

    section = get_section(

        text,

        start_patterns=[

            "things you absolutely need",

            "must have",

            "required skills",

            "requirements",

            "qualifications",

        ],

        end_patterns=[

            "things we'd like",

            "preferred",

            "nice to have",

            "bonus",

        ],

    )

    if not section:

        section = text

    return find_skills(section)

def extract_preferred_skills(text: str):

    section = get_section(

        text,

        start_patterns=[

            "things we'd like",

            "preferred",

            "nice to have",

            "bonus",

        ],

        end_patterns=[

            "things we explicitly",

            "education",

            "location",

            "salary",

        ],

    )

    return find_skills(section)

def extract_company(lines: Sequence[str]) -> Optional[str]:
    """
    Extract company name.

    Examples:
        Company: Redrob AI
        Organization: Google
    """

    company = find_label_value(lines, COMPANY_HEADERS)

    if company:
        return company

    # Fallback:
    # Company: XYZ appears somewhere in the first 20 lines.
    for line in lines[:20]:

        match = re.search(
            r"company\s*:\s*(.+)",
            line,
            flags=re.IGNORECASE,
        )

        if match:
            return match.group(1).strip()

    return None

def extract_responsibilities(text: str) -> list[str]:

    section = get_section(

        text,

        RESPONSIBILITY_HEADERS,

        [

            "requirements",

            "skills",

            "preferred",

            "education",

            "salary",

        ],

    )

    if not section:
        return []

    responsibilities = []

    for line in section.splitlines():

        line = line.strip()

        if not line:
            continue

        line = line.lstrip("-•* ")

        if len(line.split()) >= 3:
            responsibilities.append(line)

    return responsibilities

def extract_soft_skills(text: str) -> list[str]:

    lower = text.lower()

    found = []

    for skill in SOFT_SKILLS:

        if skill in lower:
            found.append(skill.title())

    return sorted(set(found))

def extract_education(text: str) -> list[str]:

    lower = text.lower()

    found = []

    for degree in EDUCATION_KEYWORDS:

        if degree.lower() in lower:
            found.append(degree)

    return sorted(set(found))

def parse_job_profile_from_text(
    text: str,
    source_name: str = "",
) -> JobProfile:
    """
    Parse a raw Job Description into a structured JobProfile.
    """

    # Preprocessing
    cleaned = _clean_text(text)
    lines = _split_lines(cleaned)

    # Primary Fields
    title = extract_title(lines)
    company = extract_company(lines)
    industry = infer_industry(cleaned)
    location = extract_location(lines)
    work_mode = extract_work_mode(cleaned)

    # Experience & Salary
    minimum_experience = extract_experience(lines)
    salary = extract_salary(cleaned)

    # Skills
    required_skills = extract_required_skills(cleaned)
    preferred_skills = extract_preferred_skills(cleaned)
    soft_skills = extract_soft_skills(cleaned)

    # Responsibilities
    responsibilities = extract_responsibilities(cleaned)


    # Education
    education_requirements = extract_education(cleaned)

    # Build JobProfile
    return JobProfile(

        # ---------- Basic ----------
        title=title,
        company=company,
        industry=industry,

        # ---------- Location ----------
        location=location,
        work_mode=work_mode,

        # ---------- Experience ----------
        minimum_experience=minimum_experience,

        # ---------- Salary ----------
        salary=salary,

        # ---------- Skills ----------
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        soft_skills=soft_skills,

        # ---------- Responsibilities ----------
        responsibilities=responsibilities,

        # ---------- Education ----------
        education_requirements=education_requirements,

        # ---------- Raw ----------
        raw_text=cleaned,
        source_name=source_name,
    )

def parse_job_profile(source: str | Path) -> JobProfile:
    path = Path(source)

    if path.exists() and path.is_file():
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            text = extract_pdf_text(path)

        elif suffix == ".txt":
            text = path.read_text(encoding="utf-8")

        elif suffix == ".docx":
            text = extract_docx_text(path)

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
