from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.preprocessing.loader import load_candidates
from backend.preprocessing.sample import extract_docx_text, parse_job_profile_from_text
from backend.schema.candidate_profile import CandidateProfile
from backend.stage2_embeddings.bm25_builder import BM25Index
from backend.stage3_retrieval.hybrid_retriever import HybridCandidateRetriever
from backend.stage3_retrieval.query_encoder import QueryEncoder, SentenceTransformerQueryEncoder
from backend.stage4_scoring.scorer import score_candidates
from stage1_features.candidate_builder import build_candidate_document


JD_PATH = Path("data/job_description.docx")
VECTOR_STORE_DIR = Path("data/stage2")
BM25_INDEX_PATH = VECTOR_STORE_DIR / "bm25_index.json"
TOP_K = 200


def _load_lookup(candidate_ids: set[str]) -> tuple[dict[str, CandidateProfile], dict[str, object]]:
    candidate_profiles: dict[str, CandidateProfile] = {}
    candidate_documents: dict[str, object] = {}

    for loaded in load_candidates("data/candidates.jsonl"):
        profile = CandidateProfile.from_dict(loaded.raw)
        if profile.candidate_id not in candidate_ids:
            continue

        document = build_candidate_document(profile)
        candidate_profiles[profile.candidate_id] = profile
        candidate_documents[profile.candidate_id] = document

        if len(candidate_profiles) == len(candidate_ids):
            break

    missing = candidate_ids - set(candidate_profiles)
    if missing:
        raise ValueError(f"Missing candidate lookups for: {sorted(missing)[:10]}")

    return candidate_profiles, candidate_documents


def _print_table(rows: list[dict[str, str]], headers: list[str]) -> None:
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row.get(header, ""))))

    def format_row(row: dict[str, str]) -> str:
        return " | ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers)

    separator = "-+-".join("-" * widths[header] for header in headers)
    print(format_row({header: header for header in headers}))
    print(separator)
    for row in rows:
        print(format_row(row))


def main() -> None:
    job_text = extract_docx_text(JD_PATH)
    job_profile = parse_job_profile_from_text(job_text, source_name=JD_PATH.name)

    retriever = HybridCandidateRetriever.load(
        vector_store_dir=VECTOR_STORE_DIR,
        bm25_index_path=BM25_INDEX_PATH,
        query_encoder=QueryEncoder(model=SentenceTransformerQueryEncoder()),
    )

    retrieved_candidates = retriever.retrieve(job_text, top_k=TOP_K)
    retrieved_ids = {candidate.candidate_id for candidate in retrieved_candidates}
    candidate_profiles, candidate_documents = _load_lookup(retrieved_ids)

    scored_candidates = score_candidates(
        retrieved_candidates=retrieved_candidates,
        candidate_profiles=candidate_profiles,
        candidate_documents=candidate_documents,
        job_profile=job_profile,
    )

    print("=" * 80)
    print("Stage 4 Scoring Test")
    print("=" * 80)
    print(f"Job title      : {job_profile.title}")
    print(f"Retrieved pool  : {len(retrieved_candidates)}")
    print(f"Scored pool      : {len(scored_candidates)}")
    print()
    table_rows: list[dict[str, str]] = []
    for rank, item in enumerate(scored_candidates[:100], start=1):
        scores = item.individual_scores.to_dict()
        table_rows.append(
            {
                "rank": str(rank),
                "candidate_id": item.candidate_id,
                "final_score": f"{item.final_score:.2f}",
                "retrieval": f"{scores['retrieval']:.3f}",
                "technical": f"{scores['technical']:.3f}",
                "experience": f"{scores['experience']:.3f}",
                "education": f"{scores['education']:.3f}",
                "company": f"{scores['company']:.3f}",
                "behavioral": f"{scores['behavioral']:.3f}",
            }
        )

    _print_table(
        table_rows,
        [
            "rank",
            "candidate_id",
            "final_score",
            "retrieval",
            "technical",
            "experience",
            "education",
            "company",
            "behavioral",
        ],
    )
#   if scored_candidates:
#       print("-" * 80)
 #       print("Top candidate breakdown:")
  #      print(json.dumps(scored_candidates[0].to_dict(), indent=2, ensure_ascii=False))
#
if __name__ == "__main__":
    main()
