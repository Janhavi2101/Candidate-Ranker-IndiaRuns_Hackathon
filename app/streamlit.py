from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st


# ===================================================================
# MAKE PROJECT ROOT IMPORTABLE
# ===================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ===================================================================
# PROJECT IMPORTS
# ===================================================================

from backend.preprocessing.jd_parser import parse_job_profile
from backend.preprocessing.loader import load_candidates

from backend.schema.candidate_profile import CandidateProfile

from stage1_features.candidate_builder import (
    build_candidate_document,
)

from backend.stage3_retrieval.hybrid_retriever import (
    HybridCandidateRetriever,
)

from backend.stage3_retrieval.query_encoder import (
    QueryEncoder,
    SentenceTransformerQueryEncoder,
)

from backend.stage4_scoring.scorer import score_candidates


# ===================================================================
# PAGE CONFIGURATION
# ===================================================================

st.set_page_config(
    page_title="Intelligent Candidate Ranker",
    page_icon="🎯",
    layout="wide",
)


# ===================================================================
# PATHS
# ===================================================================

DATA_DIR = PROJECT_ROOT / "data"

CANDIDATES_PATH = DATA_DIR / "candidates.jsonl"

STAGE2_DIR = DATA_DIR / "stage2"

BM25_INDEX_PATH = STAGE2_DIR / "bm25_index.json"


# ===================================================================
# RETRIEVAL / DISPLAY SETTINGS
# ===================================================================

# Stage 3 + Stage 4 can score up to 200 candidates.
TOP_K = 200

# UI will display only the top 100.
MAX_DISPLAY_CANDIDATES = 100

# Exactly 10 candidates on each page.
CANDIDATES_PER_PAGE = 10


# ===================================================================
# SESSION STATE
# ===================================================================

if "ranked" not in st.session_state:
    st.session_state.ranked = False

if "selected_candidate" not in st.session_state:
    st.session_state.selected_candidate = None

if "scored_candidates" not in st.session_state:
    st.session_state.scored_candidates = None

if "candidate_profiles" not in st.session_state:
    st.session_state.candidate_profiles = None

if "candidate_documents" not in st.session_state:
    st.session_state.candidate_documents = None

if "job_profile" not in st.session_state:
    st.session_state.job_profile = None

if "current_page" not in st.session_state:
    st.session_state.current_page = 1


# ===================================================================
# CACHED STAGE 3 RETRIEVER
# ===================================================================

@st.cache_resource
def load_retriever():

    return HybridCandidateRetriever.load(
        vector_store_dir=STAGE2_DIR,
        bm25_index_path=BM25_INDEX_PATH,
        query_encoder=QueryEncoder(
            model=SentenceTransformerQueryEncoder()
        ),
    )


# ===================================================================
# CANDIDATE LOOKUP
# ===================================================================

def load_candidate_lookup(
    candidate_ids: set[str],
):

    candidate_profiles: dict[str, CandidateProfile] = {}
    candidate_documents = {}

    for loaded in load_candidates(CANDIDATES_PATH):

        profile = CandidateProfile.from_dict(
            loaded.raw
        )

        if profile.candidate_id not in candidate_ids:
            continue

        document = build_candidate_document(
            profile
        )

        candidate_profiles[
            profile.candidate_id
        ] = profile

        candidate_documents[
            profile.candidate_id
        ] = document

        if len(candidate_profiles) == len(candidate_ids):
            break

    missing = (
        candidate_ids
        - set(candidate_profiles)
    )

    if missing:

        raise ValueError(
            "Missing candidate lookups for: "
            f"{sorted(missing)[:10]}"
        )

    return (
        candidate_profiles,
        candidate_documents,
    )


# ===================================================================
# SAVE UPLOADED JD TEMPORARILY
# ===================================================================

def save_uploaded_file(
    uploaded_file,
) -> Path:

    suffix = Path(
        uploaded_file.name
    ).suffix.lower()

    if suffix not in {".pdf", ".docx"}:

        raise ValueError(
            "Only .pdf and .docx files are supported."
        )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp_file:

        temp_file.write(
            uploaded_file.getvalue()
        )

        return Path(
            temp_file.name
        )


# ===================================================================
# SCORE BAR
# ===================================================================

def score_bar(
    label: str,
    score: float,
) -> None:

    score = max(
        0.0,
        min(
            100.0,
            float(score),
        ),
    )

    st.markdown(
        f"**{label}**  \n"
        f"{score:.1f} / 100"
    )

    st.progress(
        int(score)
    )


# ===================================================================
# BUILD "WHY THEY STAND OUT"
# ===================================================================

def build_reasons(
    scored_candidate,
) -> list[str]:

    reasons = []

    scores = (
        scored_candidate
        .individual_scores
        .to_dict()
    )

    component_labels = {

        "retrieval":
            "Strong initial retrieval match",

        "technical":
            "Strong technical match",

        "experience":
            "Strong relevant experience",

        "education":
            "Strong education match",

        "company":
            "Strong company / industry match",

        "behavioral":
            "Strong behavioral signals",

        "consistency":
            "Strong profile consistency",
    }

    # ---------------------------------------------------------------
    # Sort components by score
    # ---------------------------------------------------------------

    ranked_components = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    # ---------------------------------------------------------------
    # Strong components
    # ---------------------------------------------------------------

    for component, score in ranked_components:

        if score >= 80:

            reasons.append(
                component_labels.get(
                    component,
                    component.title(),
                )
            )

    # ---------------------------------------------------------------
    # Bonus
    # ---------------------------------------------------------------

    if scored_candidate.bonuses > 0:

        reasons.append(
            "Positive profile signals "
            f"(+{scored_candidate.bonuses:.1f})"
        )

    # ---------------------------------------------------------------
    # Penalty
    # ---------------------------------------------------------------

    if scored_candidate.penalties > 0:

        reasons.append(
            "Some matching penalties "
            f"({scored_candidate.penalties:.1f})"
        )

    # ---------------------------------------------------------------
    # Fallback
    # ---------------------------------------------------------------

    if not reasons and ranked_components:

        best_component = ranked_components[0]

        reasons.append(
            f"{best_component[0].title()} "
            f"score: {best_component[1]:.1f}"
        )

    return reasons[:4]


# ===================================================================
# BASIC CANDIDATE INFORMATION
# ===================================================================

def display_candidate_basic_info(
    profile: CandidateProfile,
):

    st.markdown(
        f"### {profile.profile.current_title}"
    )

    info_col1, info_col2 = st.columns(2)

    with info_col1:

        st.markdown(
            "**Experience**"
        )

        st.write(
            f"{profile.profile.years_of_experience:.1f} years"
        )

    with info_col2:

        st.markdown(
            "**Current Company**"
        )

        st.write(
            profile.profile.current_company
        )

    # ---------------------------------------------------------------
    # Industry
    # ---------------------------------------------------------------

    if profile.profile.current_industry:

        st.markdown(
            "**Industry**"
        )

        st.write(
            profile.profile.current_industry
        )

    # ---------------------------------------------------------------
    # Location
    # ---------------------------------------------------------------

    if profile.profile.location:

        st.markdown(
            "**Location**"
        )

        st.write(
            profile.profile.location
        )


# ===================================================================
# CANDIDATE DETAILED PROFILE
# ===================================================================

def display_candidate_profile(
    selected,
    profile: CandidateProfile,
):

    # ---------------------------------------------------------------
    # Header / Close button
    # ---------------------------------------------------------------

    close_col, id_col = st.columns(
        [1.2, 2.8]
    )

    with close_col:

        if st.button(
            "✕ Close Candidate",
            key="close_candidate",
            use_container_width=True,
        ):

            st.session_state.selected_candidate = None

            st.rerun()

    with id_col:

        st.markdown(
            f"**Candidate ID:** "
            f"`{selected.candidate_id}`"
        )

    st.divider()

    # ---------------------------------------------------------------
    # Basic candidate information
    # ---------------------------------------------------------------

    display_candidate_basic_info(
        profile
    )

    st.divider()

    # ---------------------------------------------------------------
    # Overall score
    # ---------------------------------------------------------------

    st.markdown(
        "### Overall Score"
    )

    st.markdown(
        f"# {selected.final_score:.2f} / 100"
    )

    st.divider()

    # ---------------------------------------------------------------
    # Score breakdown
    # ---------------------------------------------------------------

    st.markdown(
        "### Score Breakdown"
    )

    scores = (
        selected
        .individual_scores
        .to_dict()
    )

    score_bar(
        "Retrieval",
        scores.get(
            "retrieval",
            0,
        ),
    )

    score_bar(
        "Technical",
        scores.get(
            "technical",
            0,
        ),
    )

    score_bar(
        "Experience",
        scores.get(
            "experience",
            0,
        ),
    )

    score_bar(
        "Education",
        scores.get(
            "education",
            0,
        ),
    )

    score_bar(
        "Company Match",
        scores.get(
            "company",
            0,
        ),
    )

    score_bar(
        "Behavioral",
        scores.get(
            "behavioral",
            0,
        ),
    )

    score_bar(
        "Consistency",
        scores.get(
            "consistency",
            0,
        ),
    )

    # ---------------------------------------------------------------
    # Bonus / Penalty
    # ---------------------------------------------------------------

    st.divider()

    bonus_col, penalty_col = st.columns(2)

    with bonus_col:

        st.markdown(
            "### Bonus"
        )

        if selected.bonuses > 0:

            st.success(
                f"+{selected.bonuses:.2f}"
            )

        else:

            st.write(
                "0.00"
            )

    with penalty_col:

        st.markdown(
            "### Penalty"
        )

        if selected.penalties > 0:

            st.error(
                f"-{selected.penalties:.2f}"
            )

        else:

            st.write(
                "0.00"
            )

    # ---------------------------------------------------------------
    # Why candidate stands out
    # ---------------------------------------------------------------

    st.divider()

    st.markdown(
        "### Why This Candidate Stands Out"
    )

    reasons = build_reasons(
        selected
    )

    for reason in reasons:

        st.markdown(
            f"• {reason}"
        )

    # ---------------------------------------------------------------
    # Skills
    # ---------------------------------------------------------------

    if profile.skills:

        st.divider()

        st.markdown(
            "### Skills"
        )

        skills = [

            skill.name

            for skill in profile.skills

            if getattr(
                skill,
                "name",
                None,
            )
        ]

        if skills:

            st.write(
                ", ".join(skills)
            )

    # ---------------------------------------------------------------
    # Career history
    # ---------------------------------------------------------------

    if profile.career_history:

        st.divider()

        st.markdown(
            "### Career History"
        )

        for role in profile.career_history:

            company = getattr(
                role,
                "company",
                "",
            )

            title = getattr(
                role,
                "title",
                "",
            )

            duration = getattr(
                role,
                "duration_months",
                None,
            )

            st.markdown(
                f"**{title} — {company}**"
            )

            if duration is not None:

                st.caption(
                    f"{duration} months"
                )

    # ---------------------------------------------------------------
    # Education
    # ---------------------------------------------------------------

    if profile.education:

        st.divider()

        st.markdown(
            "### Education"
        )

        for education in profile.education:

            degree = getattr(
                education,
                "degree",
                "",
            )

            field = getattr(
                education,
                "field_of_study",
                "",
            )

            institution = getattr(
                education,
                "institution",
                "",
            )

            text_parts = [

                part

                for part in [
                    degree,
                    field,
                    institution,
                ]

                if part
            ]

            st.write(
                " — ".join(
                    text_parts
                )
            )


# ===================================================================
# HEADER
# ===================================================================

st.title(
    "🎯 Intelligent Candidate Ranker"
)

st.caption(
    "Rank candidates against a job description "
    "using hybrid retrieval and Stage 4 scoring."
)

st.divider()


# ===================================================================
# JOB DESCRIPTION UPLOAD
# ===================================================================

st.subheader(
    "Upload Job Description"
)

uploaded_file = st.file_uploader(
    "Choose a .docx / .pdf file",
    type=[
        "docx",
        "pdf",
    ],
)

if uploaded_file is not None:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )


# ===================================================================
# RANK BUTTON
# ===================================================================

rank_button = st.button(
    "Rank Candidates",
    type="primary",
    use_container_width=True,
)


if rank_button:

    if uploaded_file is None:

        st.warning(
            "Please upload a job description first."
        )

    elif not CANDIDATES_PATH.exists():

        st.error(
            "Candidate dataset not found:\n"
            f"{CANDIDATES_PATH}"
        )

    elif not STAGE2_DIR.exists():

        st.error(
            "Stage 2 index directory not found:\n"
            f"{STAGE2_DIR}"
        )

    elif not BM25_INDEX_PATH.exists():

        st.error(
            "BM25 index not found:\n"
            f"{BM25_INDEX_PATH}"
        )

    else:

        temp_path = None

        try:

            # -------------------------------------------------------
            # Parse JD
            # -------------------------------------------------------

            with st.spinner(
                "Parsing job description..."
            ):

                temp_path = (
                    save_uploaded_file(
                        uploaded_file
                    )
                )

                job_profile = (
                    parse_job_profile(
                        temp_path
                    )
                )

            # -------------------------------------------------------
            # Stage 3 retrieval
            # -------------------------------------------------------

            with st.spinner(
                "Running hybrid retrieval..."
            ):

                retriever = (
                    load_retriever()
                )

                retrieved_candidates = (
                    retriever.retrieve(
                        job_profile.raw_text,
                        top_k=TOP_K,
                    )
                )

            # -------------------------------------------------------
            # Candidate lookup
            # -------------------------------------------------------

            with st.spinner(
                "Loading candidate profiles..."
            ):

                retrieved_ids = {

                    candidate.candidate_id

                    for candidate
                    in retrieved_candidates
                }

                (
                    candidate_profiles,
                    candidate_documents,
                ) = load_candidate_lookup(
                    retrieved_ids
                )

            # -------------------------------------------------------
            # Stage 4 scoring
            # -------------------------------------------------------

            with st.spinner(
                "Running Stage 4 scoring..."
            ):

                scored_candidates = (
                    score_candidates(
                        retrieved_candidates=(
                            retrieved_candidates
                        ),
                        candidate_profiles=(
                            candidate_profiles
                        ),
                        candidate_documents=(
                            candidate_documents
                        ),
                        job_profile=(
                            job_profile
                        ),
                    )
                )

            # -------------------------------------------------------
            # Store results
            # -------------------------------------------------------

            st.session_state.job_profile = (
                job_profile
            )

            st.session_state.scored_candidates = (
                scored_candidates
            )

            st.session_state.candidate_profiles = (
                candidate_profiles
            )

            st.session_state.candidate_documents = (
                candidate_documents
            )

            st.session_state.ranked = True

            st.session_state.selected_candidate = None

            # Always start at page 1
            st.session_state.current_page = 1

            st.success(
                f"Ranked {len(scored_candidates)} candidates."
            )

            st.rerun()

        except Exception as exc:

            st.error(
                "Ranking failed."
            )

            st.exception(
                exc
            )

        finally:

            if temp_path is not None:

                try:

                    temp_path.unlink(
                        missing_ok=True
                    )

                except Exception:
                    pass


# ===================================================================
# RESULTS
# ===================================================================

if st.session_state.ranked:

    scored_candidates = (
        st.session_state.scored_candidates
    )

    candidate_profiles = (
        st.session_state.candidate_profiles
    )

    if not scored_candidates:

        st.warning(
            "No candidates were returned."
        )

    else:

        # ===========================================================
        # LIMIT UI TO TOP 100
        # ===========================================================

        display_candidates = (
            scored_candidates[
                :MAX_DISPLAY_CANDIDATES
            ]
        )

        total_candidates = len(
            display_candidates
        )

        total_pages = (
            (
                total_candidates
                + CANDIDATES_PER_PAGE
                - 1
            )
            // CANDIDATES_PER_PAGE
        )

        # Make sure current page is valid
        if (
            st.session_state.current_page
            > total_pages
        ):

            st.session_state.current_page = (
                total_pages
            )

        # ===========================================================
        # PAGE CALCULATION
        # ===========================================================

        current_page = (
            st.session_state.current_page
        )

        start_index = (
            (
                current_page - 1
            )
            * CANDIDATES_PER_PAGE
        )

        end_index = (
            start_index
            + CANDIDATES_PER_PAGE
        )

        page_candidates = (
            display_candidates[
                start_index:end_index
            ]
        )

        # ===========================================================
        # TWO PANEL LAYOUT
        # ===========================================================

        left_col, right_col = st.columns(
            [1.05, 0.95],
            gap="large",
        )

        # ===========================================================
        # LEFT PANEL
        # ===========================================================

        with left_col:

            st.subheader(
                "Top Candidates"
            )

            st.caption(
                f"Showing top {total_candidates} "
                f"of {len(scored_candidates)} "
                f"scored candidates."
            )

            # -------------------------------------------------------
            # PAGE NAVIGATION
            # -------------------------------------------------------

            nav_left, nav_middle, nav_right = (
                st.columns(
                    [
                        1,
                        2,
                        1,
                    ]
                )
            )

            with nav_left:

                previous_clicked = st.button(
                    "← Previous",
                    disabled=(
                        current_page == 1
                    ),
                    use_container_width=True,
                    key="previous_page",
                )

            with nav_middle:

                st.markdown(
                    f"<div style='text-align:center; "
                    f"padding-top:7px;'>"
                    f"<b>Page {current_page} "
                    f"of {total_pages}</b>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with nav_right:

                next_clicked = st.button(
                    "Next →",
                    disabled=(
                        current_page
                        == total_pages
                    ),
                    use_container_width=True,
                    key="next_page",
                )

            # -------------------------------------------------------
            # HANDLE PAGE NAVIGATION
            # -------------------------------------------------------

            if previous_clicked:

                st.session_state.current_page -= 1

                # Close candidate profile
                st.session_state.selected_candidate = None

                st.rerun()

            if next_clicked:

                st.session_state.current_page += 1

                # Close candidate profile
                st.session_state.selected_candidate = None

                st.rerun()

            st.divider()

            # -------------------------------------------------------
            # 10 CANDIDATES FOR CURRENT PAGE
            # -------------------------------------------------------

            for local_index, candidate in enumerate(
                page_candidates
            ):

                rank = (
                    start_index
                    + local_index
                    + 1
                )

                candidate_id = (
                    candidate.candidate_id
                )

                reasons = build_reasons(
                    candidate
                )

                is_selected = (
                    st.session_state.selected_candidate
                    == candidate_id
                )

                # ---------------------------------------------------
                # Candidate card
                # ---------------------------------------------------

                with st.container(
                    border=True
                ):

                    (
                        col_rank,
                        col_candidate,
                        col_score,
                        col_reason,
                    ) = st.columns(
                        [
                            0.55,
                            1.45,
                            1.0,
                            3.8,
                        ]
                    )

                    # -----------------------------------------------
                    # Rank
                    # -----------------------------------------------

                    with col_rank:

                        st.markdown(
                            f"### {rank}"
                        )

                    # -----------------------------------------------
                    # Candidate ID
                    # -----------------------------------------------

                    with col_candidate:

                        st.markdown(
                            "**Candidate ID**"
                        )

                        st.write(
                            candidate_id
                        )

                    # -----------------------------------------------
                    # Score
                    # -----------------------------------------------

                    with col_score:

                        st.markdown(
                            "**Score**"
                        )

                        st.metric(
                            label="",
                            value=(
                                f"{candidate.final_score:.2f}"
                            ),
                        )

                    # -----------------------------------------------
                    # Why they stand out
                    # -----------------------------------------------

                    with col_reason:

                        st.markdown(
                            "**Why They Stand Out**"
                        )

                        for reason in reasons:

                            st.markdown(
                                f"• {reason}"
                            )

                        # -------------------------------------------
                        # View Candidate
                        # -------------------------------------------

                        button_label = (

                            "Viewing Candidate"

                            if is_selected

                            else "View Candidate"
                        )

                        if st.button(
                            button_label,
                            key=(
                                f"view_"
                                f"{candidate_id}"
                            ),
                            use_container_width=True,
                        ):

                            st.session_state.selected_candidate = (
                                candidate_id
                            )

                            st.rerun()

            # -------------------------------------------------------
            # BOTTOM PAGE NAVIGATION
            # -------------------------------------------------------

            st.divider()

            bottom_left, bottom_middle, bottom_right = (
                st.columns(
                    [
                        1,
                        2,
                        1,
                    ]
                )
            )

            with bottom_left:

                if st.button(
                    "← Previous",
                    disabled=(
                        current_page == 1
                    ),
                    use_container_width=True,
                    key="previous_page_bottom",
                ):

                    st.session_state.current_page -= 1

                    st.session_state.selected_candidate = None

                    st.rerun()

            with bottom_middle:

                st.markdown(
                    f"<div style='text-align:center; "
                    f"padding-top:7px;'>"
                    f"Page {current_page} "
                    f"of {total_pages}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with bottom_right:

                if st.button(
                    "Next →",
                    disabled=(
                        current_page
                        == total_pages
                    ),
                    use_container_width=True,
                    key="next_page_bottom",
                ):

                    st.session_state.current_page += 1

                    st.session_state.selected_candidate = None

                    st.rerun()

        # ===========================================================
        # RIGHT PANEL
        # ===========================================================

        with right_col:

            st.subheader(
                "Candidate Profile"
            )

            selected_id = (
                st.session_state.selected_candidate
            )

            # -------------------------------------------------------
            # Nothing selected
            # -------------------------------------------------------

            if selected_id is None:

                st.info(
                    "Click **View Candidate** "
                    "to see candidate details here."
                )

            else:

                # ---------------------------------------------------
                # Find selected scored candidate
                # ---------------------------------------------------

                selected = next(
                    (
                        candidate

                        for candidate
                        in scored_candidates

                        if candidate.candidate_id
                        == selected_id
                    ),
                    None,
                )

                # ---------------------------------------------------
                # Find selected profile
                # ---------------------------------------------------

                profile = (

                    candidate_profiles.get(
                        selected_id
                    )

                    if candidate_profiles

                    else None
                )

                # ---------------------------------------------------
                # Error handling
                # ---------------------------------------------------

                if selected is None:

                    st.error(
                        "Selected candidate was not "
                        "found in the scored results."
                    )

                elif profile is None:

                    st.error(
                        "Candidate profile data "
                        "could not be loaded."
                    )

                else:

                    # -----------------------------------------------
                    # DISPLAY PROFILE
                    # -----------------------------------------------

                    display_candidate_profile(
                        selected,
                        profile,
                    )