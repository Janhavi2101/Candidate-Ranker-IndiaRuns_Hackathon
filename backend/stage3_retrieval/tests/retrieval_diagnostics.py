from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.preprocessing.sample import parse_job_profile
from backend.stage3_retrieval.hybrid_retriever import HybridCandidateRetriever
from backend.stage3_retrieval.query_builder import build_bm25_query, build_semantic_query
from backend.stage3_retrieval.query_encoder import QueryEncoder, SentenceTransformerQueryEncoder


DEFAULT_STAGE2_DIR = Path("data/stage2")
DEFAULT_JOB_DESCRIPTION = Path("data/job_description.docx")


def _overlap(left: Iterable[str], right: Iterable[str]) -> int:
    return len(set(left) & set(right))


def _print_hits(title: str, hits, score_label: str, top_n: int = 20) -> None:
    print(f"\n{title} Top {top_n}")
    print("-" * 72)
    for rank, hit in enumerate(hits[:top_n], start=1):
        score = getattr(hit, score_label)
        print(f"{rank:>3}. {hit.candidate_id}  {score:.6f}")


def run_diagnostics(
    stage2_dir: str | Path = DEFAULT_STAGE2_DIR,
    job_description_path: str | Path = DEFAULT_JOB_DESCRIPTION,
    top_k: int = 1000,
    model_name: str = "all-MiniLM-L6-v2",
) -> None:
    stage2_path = Path(stage2_dir)
    job_path = Path(job_description_path)

    job_profile = parse_job_profile(job_path)

    query_encoder = QueryEncoder(model=SentenceTransformerQueryEncoder(model_name=model_name))
    retriever = HybridCandidateRetriever.load(
        vector_store_dir=stage2_path,
        bm25_index_path=stage2_path / "bm25_index.json",
        query_encoder=query_encoder,
    )

    semantic_query = build_semantic_query(job_profile)
    bm25_query = build_bm25_query(job_profile)
    semantic_hits = retriever.search_semantic(job_path, top_k=top_k)
    bm25_hits = retriever.search_bm25(job_path, top_k=top_k)
    hybrid_hits = retriever.retrieve(job_path, top_k=top_k)

    semantic_ids = [hit.candidate_id for hit in semantic_hits]
    bm25_ids = [hit.candidate_id for hit in bm25_hits]
    hybrid_ids = [hit.candidate_id for hit in hybrid_hits]
    total_overlap = _overlap(semantic_ids, bm25_ids)

    print("=" * 72)
    print("Retrieval Diagnostics")
    print("=" * 72)
    print(f"Job file         : {job_path}")
    print(f"Semantic query   : {semantic_query[:220].replace(chr(10), ' ')}")
    print(f"BM25 query       : {bm25_query[:220].replace(chr(10), ' ')}")
    print(f"Semantic hits    : {len(semantic_hits)}")
    print(f"BM25 hits        : {len(bm25_hits)}")
    print(f"Hybrid hits      : {len(hybrid_hits)}")
    print(f"Total overlap    : {total_overlap}")

    _print_hits("Semantic Retrieval", semantic_hits, "score")
    _print_hits("BM25 Retrieval", bm25_hits, "score")
    print("\nHybrid Retrieval Top 20")
    print("-" * 72)
    for rank, hit in enumerate(hybrid_hits[:20], start=1):
        semantic_score = "-" if hit.semantic_score is None else f"{hit.semantic_score:.6f}"
        bm25_score = "-" if hit.bm25_score is None else f"{hit.bm25_score:.6f}"
        print(
            f"{rank:>3}. {hit.candidate_id}  semantic={semantic_score}  bm25={bm25_score}  rrf={hit.rrf_score:.6f}"
        )

    print("\nOverlap summary")
    print("-" * 72)
    for cutoff in [20, 50, 100, 200, 500, 1000]:
        capped_semantic = semantic_ids[:cutoff]
        capped_bm25 = bm25_ids[:cutoff]
        capped_hybrid = hybrid_ids[:cutoff]
        print(
            f"Top{cutoff:<4} semantic∩bm25={_overlap(capped_semantic, capped_bm25):>3}  "
            f"semantic∩hybrid={_overlap(capped_semantic, capped_hybrid):>3}  "
            f"bm25∩hybrid={_overlap(capped_bm25, capped_hybrid):>3}"
        )


if __name__ == "__main__":
    run_diagnostics()
