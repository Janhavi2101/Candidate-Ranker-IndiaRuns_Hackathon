   # ===============================================================
    # RIGHT: CANDIDATE DETAILS
    # ===============================================================

    if (
        right_column is not None
        and st.session_state.selected_candidate
    ):

        selected = next(
            (
                candidate
                for candidate in scored_candidates
                if candidate.candidate_id
                == st.session_state.selected_candidate
            ),
            None,
        )

        if selected:

            with right_column:

                # ---------------------------------------------------
                # Header
                # ---------------------------------------------------

                st.subheader("Candidate Profile")

                st.markdown(
                    f"**Candidate ID:** "
                    f"`{selected.candidate_id}`"
                )

                # ---------------------------------------------------
                # Candidate basic information
                # ---------------------------------------------------

                profile = get_candidate_profile(
                    selected.candidate_id
                )

                if profile:

                    st.markdown(
                        f"### {profile.profile.current_title}"
                    )

                    basic_col1, basic_col2 = st.columns(2)

                    with basic_col1:

                        st.markdown(
                            "**Experience**"
                        )

                        st.write(
                            f"{profile.profile.years_of_experience:.1f} years"
                        )

                    with basic_col2:

                        st.markdown(
                            "**Current Company**"
                        )

                        st.write(
                            profile.profile.current_company
                            or "Not specified"
                        )

                # ---------------------------------------------------
                # Overall score
                # ---------------------------------------------------

                st.markdown("### Overall Score")

                st.metric(
                    "",
                    f"{selected.final_score:.2f} / 100",
                )

                st.divider()

                # ---------------------------------------------------
                # Component scores
                # ---------------------------------------------------

                scores = selected.individual_scores.to_dict()

                st.markdown(
                    "### Score Breakdown"
                )

                score_bar(
                    "Retrieval",
                    scores["retrieval"],
                )

                score_bar(
                    "Technical",
                    scores["technical"],
                )

                score_bar(
                    "Experience",
                    scores["experience"],
                )

                score_bar(
                    "Education",
                    scores["education"],
                )

                score_bar(
                    "Company Match",
                    scores["company"],
                )

                score_bar(
                    "Behavioral",
                    scores["behavioral"],
                )

                score_bar(
                    "Consistency",
                    scores["consistency"],
                )

                st.divider()

                # ---------------------------------------------------
                # Bonuses / penalties
                # ---------------------------------------------------

                bonus_col, penalty_col = st.columns(2)

                with bonus_col:

                    st.markdown(
                        "### Bonuses"
                    )

                    if selected.bonuses > 0:

                        st.success(
                            f"+{selected.bonuses:.2f}"
                        )

                    else:

                        st.write("None")

                with penalty_col:

                    st.markdown(
                        "### Penalties"
                    )

                    if selected.penalties > 0:

                        st.error(
                            f"-{selected.penalties:.2f}"
                        )

                    else:

                        st.write("None")

                # ---------------------------------------------------
                # Why candidate stands out
                # ---------------------------------------------------

                st.divider()

                st.markdown(
                    "### Why This Candidate Stands Out"
                )

                for reason in generate_reasons(selected):

                    st.markdown(
                        f"• {reason}"
                    )

                # ---------------------------------------------------
                # Skills
                # ---------------------------------------------------

                if profile:

                    st.divider()

                    st.markdown(
                        "### Skills"
                    )

                    skills = [
                        skill.name
                        for skill in profile.skills
                    ]

                    if skills:

                        st.write(
                            ", ".join(skills)
                        )

                    else:

                        st.write(
                            "No skills available."
                        )

                    # ------------------------------------------------
                    # Education
                    # ------------------------------------------------

                    st.markdown(
                        "### Education"
                    )

                    if profile.education:

                        for education in profile.education:

                            education_text = (
                                f"**{education.degree}**"
                            )

                            if education.field_of_study:

                                education_text += (
                                    f" — {education.field_of_study}"
                                )

                            st.markdown(
                                education_text
                            )

                            if education.institution:

                                st.caption(
                                    education.institution
                                )

                    else:

                        st.write(
                            "No education information available."
                        )

                # ---------------------------------------------------
                # Detailed Stage 4 breakdown
                # ---------------------------------------------------

                st.divider()

                with st.expander(
                    "View detailed Stage 4 scoring"
                ):

                    st.json(
                        selected.score_breakdown.to_dict()
                    )