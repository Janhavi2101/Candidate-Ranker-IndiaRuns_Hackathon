from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from stage1_features.candidate_builder import build_candidate_document
from stage1_features.models import CandidateDocument


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "we",
    "with",
    "you",
    "your",
}


def _normalize_token(token: str) -> str:
    if token.isdigit():
        return token
    normalized = token.lower()
    if normalized in STOPWORDS:
        return ""
    for suffix in ("ingly", "edly", "ing", "edly", "ed", "es", "s"):
        if len(normalized) > 4 and normalized.endswith(suffix):
            stemmed = normalized[: -len(suffix)]
            if len(stemmed) >= 3:
                normalized = stemmed
                break
    return normalized


def tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for token in TOKEN_PATTERN.findall(text.lower()):
        normalized = _normalize_token(token)
        if normalized:
            tokens.append(normalized)
    return tokens


@dataclass(slots=True)
class BM25SearchHit:
    candidate_id: str
    score: float

    def to_dict(self) -> Dict[str, float | str]:
        return {"candidate_id": self.candidate_id, "score": self.score}


@dataclass(slots=True)
class BM25Index:
    candidate_ids: List[str]
    tokenized_documents: List[List[str]]
    k1: float = 1.5
    b: float = 0.75
    document_lengths: List[int] = field(init=False)
    average_document_length: float = field(init=False)
    document_frequencies: Dict[str, int] = field(init=False)
    idf: Dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        self.document_lengths = [len(tokens) for tokens in self.tokenized_documents]
        self.average_document_length = (sum(self.document_lengths) / len(self.document_lengths)) if self.document_lengths else 0.0
        self.document_frequencies = {}
        for tokens in self.tokenized_documents:
            for token in set(tokens):
                self.document_frequencies[token] = self.document_frequencies.get(token, 0) + 1
        total_documents = len(self.tokenized_documents)
        self.idf = {
            token: math.log(1.0 + ((total_documents - frequency + 0.5) / (frequency + 0.5)))
            for token, frequency in self.document_frequencies.items()
        }

    def score(self, query_text: str) -> List[float]:
        query_tokens = list(dict.fromkeys(tokenize(query_text)))
        if not query_tokens or not self.tokenized_documents:
            return [0.0 for _ in self.tokenized_documents]

        scores: List[float] = []
        for document_tokens, document_length in zip(self.tokenized_documents, self.document_lengths):
            term_frequencies: Dict[str, int] = {}
            for token in document_tokens:
                term_frequencies[token] = term_frequencies.get(token, 0) + 1

            score = 0.0
            for token in query_tokens:
                frequency = term_frequencies.get(token, 0)
                if frequency == 0:
                    continue
                idf = self.idf.get(token)
                if idf is None:
                    continue
                denominator = frequency + self.k1 * (1.0 - self.b + self.b * document_length / self.average_document_length) if self.average_document_length else frequency + self.k1
                score += idf * (frequency * (self.k1 + 1.0)) / denominator
            scores.append(score)
        return scores

    def search(self, query_text: str, top_k: int = 200) -> List[BM25SearchHit]:
        scores = self.score(query_text)
        ranked = sorted(
            zip(self.candidate_ids, scores),
            key=lambda item: (-item[1], item[0]),
        )
        return [BM25SearchHit(candidate_id=candidate_id, score=float(score)) for candidate_id, score in ranked[:top_k]]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        payload = {
            "candidate_ids": self.candidate_ids,
            "tokenized_documents": self.tokenized_documents,
            "k1": self.k1,
            "b": self.b,
        }
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            candidate_ids=list(payload["candidate_ids"]),
            tokenized_documents=[
                [token for token in (_normalize_token(item) for item in tokens) if token]
                for tokens in payload["tokenized_documents"]
            ],
            k1=float(payload.get("k1", 1.5)),
            b=float(payload.get("b", 0.75)),
        )


def build_bm25_index_from_documents(documents: Iterable[CandidateDocument]) -> BM25Index:
    candidate_ids: List[str] = []
    tokenized_documents: List[List[str]] = []
    for document in documents:
        candidate_ids.append(document.candidate_id)
        tokenized_documents.append(tokenize(document.embedding_text))
    return BM25Index(candidate_ids=candidate_ids, tokenized_documents=tokenized_documents)


def build_bm25_index_from_candidate_profiles(candidate_profiles: Iterable[object]) -> BM25Index:
    documents = [build_candidate_document(candidate_profile) for candidate_profile in candidate_profiles]
    return build_bm25_index_from_documents(documents)
