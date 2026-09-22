from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


DEFAULT_STAGE4_CONFIG_PATH = Path("config/stage4_scoring.json")


@dataclass(slots=True)
class Stage4Config:
    component_weights: Dict[str, float] = field(default_factory=dict)
    retrieval_subweights: Dict[str, float] = field(default_factory=dict)
    technical_weights: Dict[str, float] = field(default_factory=dict)
    experience_weights: Dict[str, float] = field(default_factory=dict)
    education_weights: Dict[str, float] = field(default_factory=dict)
    company_weights: Dict[str, float] = field(default_factory=dict)
    behavioral_weights: Dict[str, float] = field(default_factory=dict)
    bonus_rules: Dict[str, float] = field(default_factory=dict)
    penalty_rules: Dict[str, float] = field(default_factory=dict)
    penalty_caps: Dict[str, float] = field(default_factory=dict)
    thresholds: Dict[str, float] = field(default_factory=dict)
    base_score: float = 55.0
    final_score_scale: float = 100.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_weights": dict(self.component_weights),
            "retrieval_subweights": dict(self.retrieval_subweights),
            "technical_weights": dict(self.technical_weights),
            "experience_weights": dict(self.experience_weights),
            "education_weights": dict(self.education_weights),
            "company_weights": dict(self.company_weights),
            "behavioral_weights": dict(self.behavioral_weights),
            "bonus_rules": dict(self.bonus_rules),
            "penalty_rules": dict(self.penalty_rules),
            "penalty_caps": dict(self.penalty_caps),
            "thresholds": dict(self.thresholds),
            "base_score": self.base_score,
            "final_score_scale": self.final_score_scale,
        }


DEFAULT_STAGE4_CONFIG = Stage4Config(
    component_weights={
        "retrieval": 0.25,
        "technical": 0.22,
        "experience": 0.18,
        "education": 0.08,
        "company": 0.08,
        "behavioral": 0.14,
        "consistency": 0.05,
    },
    retrieval_subweights={
        "semantic": 0.4,
        "bm25": 0.35,
        "rrf": 0.25,
    },
    technical_weights={
        "required_skill_coverage": 0.45,
        "preferred_skill_coverage": 0.15,
        "skill_similarity": 0.25,
        "skill_proficiency": 0.15,
    },
    experience_weights={
        "years": 0.45,
        "title_similarity": 0.30,
        "industry_match": 0.25,
    },
    education_weights={
        "degree": 0.40,
        "field": 0.40,
        "institution": 0.20,
    },
    company_weights={
        "industry_match": 0.70,
        "company_size_match": 0.30,
    },
    behavioral_weights={
        "profile_completeness": 0.30,
        "response_rate": 0.25,
        "github": 0.20,
        "interview_completion": 0.15,
        "endorsements": 0.10,
    },
    bonus_rules={
        "open_to_work": 1.5,
        "verified_email": 1.0,
        "verified_phone": 1.0,
        "linkedin_connected": 1.0,
        "exceptional_github_activity": 1.5,
        "excellent_assessment_scores": 1.5,
        "high_recruiter_engagement": 1.0,
    },
    penalty_rules={
        "missing_required_skill": 5.0,
        "experience_gap_per_year": 4.0,
        "notice_period_long": 2.0,
        "salary_above_budget": 4.0,
        "work_mode_mismatch": 3.0,
        "relocation_mismatch": 3.0,
        "low_profile_quality": 3.0,
        "low_recruiter_response": 3.0,
        "inactive_profile": 2.0,
        "poor_career_stability": 3.0,
    },
    penalty_caps={
        "missing_required_skill": 10.0,
        "experience_gap": 5.0,
    },
    thresholds={
        "exceptional_github_activity": 80.0,
        "excellent_assessment_score": 70.0,
        "high_recruiter_response_rate": 0.65,
        "low_profile_quality": 70.0,
        "low_recruiter_response_rate": 0.3,
        "inactive_profile_days": 45.0,
        "long_notice_period_days": 90.0,
        "salary_buffer": 0.0,
    },
    base_score=35.0,
    final_score_scale=100.0,
)


def _merge_dicts(base: Dict[str, float], override: Dict[str, float]) -> Dict[str, float]:
    merged = dict(base)
    merged.update(override)
    return merged


def _map_legacy_weight_keys(payload: Dict[str, float], mapping: Dict[str, str]) -> Dict[str, float]:
    normalized = dict(payload)
    for legacy_key, current_key in mapping.items():
        if legacy_key in normalized and current_key not in normalized:
            normalized[current_key] = normalized.pop(legacy_key)
    return normalized


def load_stage4_config(path: str | Path | None = None) -> Stage4Config:
    config_path = Path(path) if path is not None else DEFAULT_STAGE4_CONFIG_PATH
    if not config_path.exists():
        return DEFAULT_STAGE4_CONFIG

    payload = json.loads(config_path.read_text(encoding="utf-8"))

    experience_weights = _map_legacy_weight_keys(
        payload.get("experience_weights", {}),
        {"years_match": "years"},
    )
    education_weights = _map_legacy_weight_keys(
        payload.get("education_weights", {}),
        {
            "degree_match": "degree",
            "field_match": "field",
            "institution_tier": "institution",
        },
    )
    behavioral_weights = _map_legacy_weight_keys(
        payload.get("behavioral_weights", {}),
        {
            "recruiter_response_rate": "response_rate",
            "github_activity": "github",
        },
    )

    return Stage4Config(
        component_weights=_merge_dicts(DEFAULT_STAGE4_CONFIG.component_weights, payload.get("component_weights", {})),
        retrieval_subweights=_merge_dicts(DEFAULT_STAGE4_CONFIG.retrieval_subweights, payload.get("retrieval_subweights", {})),
        technical_weights=_merge_dicts(DEFAULT_STAGE4_CONFIG.technical_weights, payload.get("technical_weights", {})),
        experience_weights=_merge_dicts(DEFAULT_STAGE4_CONFIG.experience_weights, experience_weights),
        education_weights=_merge_dicts(DEFAULT_STAGE4_CONFIG.education_weights, education_weights),
        company_weights=_merge_dicts(DEFAULT_STAGE4_CONFIG.company_weights, payload.get("company_weights", {})),
        behavioral_weights=_merge_dicts(DEFAULT_STAGE4_CONFIG.behavioral_weights, behavioral_weights),
        bonus_rules=_merge_dicts(DEFAULT_STAGE4_CONFIG.bonus_rules, payload.get("bonus_rules", {})),
        penalty_rules=_merge_dicts(DEFAULT_STAGE4_CONFIG.penalty_rules, payload.get("penalty_rules", {})),
        penalty_caps=_merge_dicts(DEFAULT_STAGE4_CONFIG.penalty_caps, payload.get("penalty_caps", {})),
        thresholds=_merge_dicts(DEFAULT_STAGE4_CONFIG.thresholds, payload.get("thresholds", {})),
        base_score=float(payload.get("base_score", DEFAULT_STAGE4_CONFIG.base_score)),
        final_score_scale=float(payload.get("final_score_scale", DEFAULT_STAGE4_CONFIG.final_score_scale)),
    )
