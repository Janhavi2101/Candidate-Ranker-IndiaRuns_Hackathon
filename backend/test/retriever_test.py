"""
Test Stage 3 - Hybrid Retrieval

Pipeline:

JobProfile
      ↓
HybridRetriever
      ↓
Top Retrieved Candidates

Prints:
- Candidate ID
- Semantic Score
- BM25 Score
- RRF Score
"""

from pathlib import Path

from backend.preprocessing.sample import (
    parse_job_profile,
    extract_docx_text,
)


# change these imports according to your project
from backend.stage3_retrieval.hybrid_retriever import HybridCandidateRetriever
from backend.schema.job_profile import JobProfile

from backend.stage3_retrieval.query_encoder import (
    QueryEncoder,
    SentenceTransformerQueryEncoder,
)

JD_PATH = Path("data/job_description.docx")

VECTOR_STORE_DIR = Path("data/stage2")

BM25_INDEX_PATH = VECTOR_STORE_DIR / "bm25_index.json"


def main():

    print("=" * 60)
    print("Parsing Job Description...")
    print("=" * 60)

    job_profile = parse_job_profile(JD_PATH)

    print(job_profile.title)

    print("\n" + "=" * 60)
    print("Loading Hybrid Retriever...")
    print("=" * 60)

    encoder = QueryEncoder(
        SentenceTransformerQueryEncoder()
    )

    retriever = HybridCandidateRetriever.load(
        vector_store_dir=VECTOR_STORE_DIR,
        bm25_index_path=BM25_INDEX_PATH,
        query_encoder=encoder,
    )

    print("✓ Hybrid Retriever Loaded")

    print("\n" + "=" * 60)
    print("Running Hybrid Retrieval...")
    print("=" * 60)

    job_text = extract_docx_text(JD_PATH)

    results = retriever.retrieve(
        job_description=job_text,
        top_k=100,
    )

    print("\n")
    print("-" * 70)
    print("Top 20 Retrieved Candidates")
    print("-" * 70)

    for rank, candidate in enumerate(results, start=1):

        print(f"{rank:>2}. {candidate.candidate_id}")
        semantic = ( f"{candidate.semantic_score:.4f}"
                   if candidate.semantic_score is not None
                  else "-"
                    )

        bm25 = (f"{candidate.bm25_score:.4f}"
              if candidate.bm25_score is not None
                 else "-" )

        print(f"    Semantic Score : {semantic}")
        print(f"    BM25 Score     : {bm25}")
        print(f"    RRF Score      : {candidate.rrf_score:.4f}")
        print("-" * 70)
    


if __name__ == "__main__":
    main()