from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")

PROFICIENCY_SCORES = {
    "beginner": 0.25,
    "intermediate": 0.5,
    "advanced": 0.75,
    "expert": 1.0,
}

SENIORITY_KEYWORDS = {
    "intern": 0,
    "junior": 1,
    "associate": 1,
    "mid": 2,
    "senior": 3,
    "lead": 4,
    "principal": 5,
    "staff": 5,
    "head": 6,
}


@dataclass(slots=True)
class MatcherResult:
    score: float
    details: dict


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def to_percent(value: float) -> float:
    return clamp01(value) * 100.0


def normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def normalize_alnum_text(value: str) -> str:
    return " ".join(ALNUM_PATTERN.sub(" ", value.strip().lower()).split())


def tokenize(value: str) -> List[str]:
    return TOKEN_PATTERN.findall(normalize_text(value))


def jaccard_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = {normalize_text(item) for item in left if normalize_text(item)}
    right_set = {normalize_text(item) for item in right if normalize_text(item)}
    if not left_set and not right_set:
        return 0.0
    intersection = left_set & right_set
    union = left_set | right_set
    return len(intersection) / len(union) if union else 0.0


def title_similarity(left: str, right: str) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens & right_tokens
    return len(overlap) / len(left_tokens | right_tokens)


def seniority_level(title: str) -> int:
    lowered = normalize_text(title)
    for keyword, level in SENIORITY_KEYWORDS.items():
        if keyword in lowered:
            return level
    return 2


def average(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def weighted_average(scores: dict[str, float], weights: dict[str, float]) -> float:
    if not scores:
        return 0.0
    total_weight = sum(max(weights.get(name, 0.0), 0.0) for name in scores)
    if total_weight <= 0:
        return average(scores.values())
    weighted_total = sum(scores[name] * max(weights.get(name, 0.0), 0.0) for name in scores)
    return weighted_total / total_weight


def log_scale(value: float, base: float = 100.0) -> float:
    if value <= 0:
        return 0.0
    return clamp01(math.log1p(value) / math.log1p(base))
