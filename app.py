import pandas as pd
import numpy as np
import streamlit as st
try:
    import ollama as _ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
ROLES = [
    ("Data Analyst",              "Data Analytics",      "excel, sql, python, data visualization, statistics, power bi"),
    ("Data Scientist",            "Data Science & AI",   "python, machine learning, statistics, pandas, sql, data visualization"),
    ("Machine Learning Engineer", "Data Science & AI",   "python, machine learning, deep learning, tensorflow, git, docker"),
    ("Computer Vision Engineer",  "Data Science & AI",   "python, opencv, deep learning, machine learning, numpy"),
    ("Python Backend Developer",  "Software Development","python, fastapi, sql, rest api, git, docker"),
    ("Java Developer",            "Software Development","java, sql, spring boot, rest api, git, oop"),
    ("Mobile App Developer",      "Software Development","kotlin, flutter, rest api, git, ui design"),
    ("Frontend Developer",        "Web Development",     "html, css, javascript, react, git, responsive design"),
    ("Full Stack Developer",      "Web Development",     "html, css, javascript, react, node.js, sql, git"),
    ("Software Tester (QA)",      "Quality Assurance",   "manual testing, selenium, sql, test cases, java"),
    ("DevOps Engineer",           "Cloud & DevOps",      "linux, docker, git, ci/cd, cloud computing, shell scripting"),
    ("Cloud Engineer",            "Cloud & DevOps",      "cloud computing, linux, networking, docker, python"),
    ("Cybersecurity Analyst",     "Security",            "networking, linux, cybersecurity, python, risk assessment"),
    ("Database Administrator",    "Data & Databases",    "sql, database design, backup and recovery, linux, performance tuning"),
    ("Data Engineer",             "Data & Databases",    "python, sql, etl, apache spark, cloud computing, data warehousing"),
    ("Business Analyst",          "Business",            "excel, sql, communication, documentation, power bi, problem solving"),
    ("UI/UX Designer",            "Design",              "figma, ui design, user research, wireframing, prototyping"),
    ("Digital Marketing Executive","Marketing",          "seo, social media, content writing, google analytics, communication"),
    ("Content Writer",            "Marketing",           "content writing, seo, communication, research, editing"),
    ("Technical Support Engineer","IT Support",          "troubleshooting, networking, communication, windows, linux"),
]

ALIASES = {
    "ml": "machine learning", "dl": "deep learning", "js": "javascript",
    "powerbi": "power bi", "power-bi": "power bi", "nodejs": "node.js", "node": "node.js",
    "cv": "opencv", "tf": "tensorflow", "rest": "rest api", "restapi": "rest api",
    "cloud": "cloud computing", "aws": "cloud computing", "azure": "cloud computing",
    "dataviz": "data visualization", "visualization": "data visualization",
    "spring": "spring boot", "ui": "ui design", "testing": "manual testing",
    "ci": "ci/cd", "cicd": "ci/cd", "spark": "apache spark",
}

INTEREST_AREAS = [
    "No preference", "Data Analytics", "Data Science & AI", "Software Development",
    "Web Development", "Quality Assurance", "Cloud & DevOps", "Security",
    "Data & Databases", "Business", "Design", "Marketing", "IT Support",
]

AREA_ICONS = {
    "Data Analytics": "📊", "Data Science & AI": "🤖", "Software Development": "💻",
    "Web Development": "🌐", "Quality Assurance": "🔍", "Cloud & DevOps": "☁️",
    "Security": "🔒", "Data & Databases": "🗄️", "Business": "📈",
    "Design": "🎨", "Marketing": "📣", "IT Support": "🛠️",
}

AREA_COLORS = [
    "#2a6fdb","#7c3aed","#059669","#dc2626","#d97706","#0891b2",
    "#be185d","#65a30d","#4f46e5","#b45309","#0f766e","#9333ea",
]

# ──────────────────────────────────────────────────────────────────────────────
# Model  (cached — built once per session)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def build_model():
    jobs = pd.DataFrame(ROLES, columns=["role", "interest_area", "required_skills"])

    def skill_tokenizer(text):
        return [s.strip().lower() for s in text.split(",") if s.strip()]

    vectorizer = TfidfVectorizer(tokenizer=skill_tokenizer, token_pattern=None, lowercase=False)
    role_matrix = vectorizer.fit_transform(jobs["required_skills"])
    known_skills = sorted(vectorizer.get_feature_names_out())
    tfidf_dense  = role_matrix.toarray()
    return jobs, vectorizer, role_matrix, set(known_skills), known_skills, skill_tokenizer, tfidf_dense


def clean_skills(raw):
    skills = []
    for s in raw.split(","):
        s = s.strip().lower()
        if s:
            skills.append(ALIASES.get(s, s))
    return list(dict.fromkeys(skills))


def recommend(user_skills_raw, interest=None, top_n=5):
    jobs, vectorizer, role_matrix, known_set, known_skills, skill_tokenizer, _ = build_model()
    skills  = clean_skills(user_skills_raw)
    unknown = [s for s in skills if s not in known_set]
    skills  = [s for s in skills if s in known_set]

    if not skills:
        return pd.DataFrame(), unknown

    user_vec   = vectorizer.transform([", ".join(skills)])
    similarity = cosine_similarity(user_vec, role_matrix)[0]

    rows = []
    for i, row in jobs.iterrows():
        required = skill_tokenizer(row["required_skills"])
        have     = [s for s in required if s in skills]
        missing  = [s for s in required if s not in skills]
        coverage = len(have) / len(required)

        score = 0.6 * similarity[i] + 0.4 * coverage
        if interest and interest.lower() == row["interest_area"].lower():
            score += 0.05
        score = min(score, 1.0)

        rows.append({
            "Role": row["role"],
            "Interest Area": row["interest_area"],
            "Match %": round(score * 100, 1),
            "Cosine Sim": round(float(similarity[i]) * 100, 1),
            "Coverage %": round(coverage * 100, 1),
            "Skills You Have": ", ".join(have) if have else "—",
            "Skills To Learn Next": ", ".join(missing) if missing else "✅ None — you are ready!",
            "_have_count":    len(have),
            "_missing_count": len(missing),
        })

    result = (
        pd.DataFrame(rows)
        .sort_values("Match %", ascending=False)
        .query("`Match %` > 0")
        .head(top_n)
        .reset_index(drop=True)
    )
    return result, unknown


# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Career Path Recommender",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — input (shared across all tabs)
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='display:flex;align-items:center;gap:10px'>"
        "<span style='font-size:1.8rem'>🎯</span>"
        "<div><b style='font-size:1.05rem'>Career Path Recommender</b><br>"
        "<span style='font-size:0.75rem;color:#888'>IBM SkillsBuild · AI & ML Internship · UN SDG 8</span></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.subheader("Your Skills")
    skills_input = st.text_area(
        "Enter your skills (comma-separated)",
        placeholder="e.g. python, sql, excel, git",
        height=110,
        help="Shortcuts like 'ml', 'js', 'aws', 'spark' are automatically expanded.",
    )

    st.subheader("Preferences")
    interest_choice = st.selectbox("Domain preference", options=INTEREST_AREAS, index=0)
    top_n = st.slider("Max results", min_value=1, max_value=10, value=5)

    find_btn = st.button("🔍 Find My Career Paths", type="primary", use_container_width=True)

    st.divider()
    with st.expander("ℹ️ Scoring formula"):
        st.markdown(
            "**Match % = 60 % cosine similarity + 40 % skill coverage**\n\n"
            "- **Cosine similarity** — TF-IDF weighted overlap between your skills and the role's skills.\n"
            "- **Skill coverage** — fraction of the role's skills you already hold.\n"
            "- **+5 % bonus** if the role matches your chosen domain."
        )
    with st.expander("🔤 Recognised shortcuts"):
        alias_df = pd.DataFrame(
            [{"Shortcut": k, "Expands to": v} for k, v in sorted(ALIASES.items())]
        )
        st.dataframe(alias_df, hide_index=True, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Main — 5 tabs
# ──────────────────────────────────────────────────────────────────────────────
st.title("AI Career Path Recommender")
st.markdown(
    "Discover which tech careers suit your current skills and exactly what to learn next — "
    "designed for job seekers in **Tier-2/3 cities** 🇮🇳"
)

tab_rec, tab_data, tab_skills, tab_model, tab_sanity, tab_chat = st.tabs([
    "🎯 Recommender",
    "📋 Dataset Explorer",
    "🔬 Skills Analysis",
    "🧠 Model Insights",
    "✅ Sanity Check",
    "🤖 AI Career Coach",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════════
with tab_rec:
    if not find_btn:
        col1, col2, col3 = st.columns(3)
        col1.info("**Step 1** — Enter your skills in the sidebar.", icon="✏️")
        col2.info("**Step 2** — Choose a domain preference (optional).", icon="🗂️")
        col3.info("**Step 3** — Click **Find My Career Paths**.", icon="🔍")

        st.subheader("Covered Domains")
        domain_cols = st.columns(4)
        for idx, area in enumerate(INTEREST_AREAS[1:]):
            domain_cols[idx % 4].markdown(f"{AREA_ICONS.get(area, '🔹')} **{area}**")
    else:
        if not skills_input.strip():
            st.warning("Please enter at least one skill in the sidebar.", icon="⚠️")
            st.stop()

        interest = None if interest_choice == "No preference" else interest_choice
        result, unknown = recommend(skills_input, interest=interest, top_n=top_n)

        if unknown:
            st.warning(
                f"Unrecognised skills were ignored: **{', '.join(unknown)}**  \n"
                "Try the full name or check the shortcuts list in the sidebar.",
                icon="⚠️",
            )

        if result.empty:
            st.error("No matches found. Try adding more recognised skills.", icon="❌")
            st.stop()

        # ── Metrics row ───────────────────────────────────────────────────────
        top = result.iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🥇 Best Match",      top["Role"])
        m2.metric("📊 Match Score",     f"{top['Match %']} %")
        m3.metric("🗂️ Domain",          top["Interest Area"])
        m4.metric("✅ Skills Matched",  top["_have_count"],
                  f"−{top['_missing_count']} to learn")

        st.divider()

        # ── Horizontal bar chart ──────────────────────────────────────────────
        st.subheader("📊 Match Overview")
        chart_df = result[["Role", "Match %"]].set_index("Role").sort_values("Match %")
        st.bar_chart(chart_df, horizontal=True, color="#2a6fdb")

        st.divider()

        # ── Score breakdown table ─────────────────────────────────────────────
        with st.expander("📐 Score Breakdown Table", expanded=False):
            breakdown = result[["Role", "Interest Area", "Match %", "Cosine Sim", "Coverage %"]].copy()
            st.dataframe(breakdown, use_container_width=True, hide_index=True)

        # ── Detailed cards ────────────────────────────────────────────────────
        st.subheader("📋 Detailed Results")
        for rank, (_, row) in enumerate(result.iterrows(), start=1):
            icon      = AREA_ICONS.get(row["Interest Area"], "🔹")
            pct       = row["Match %"]
            bar_color = "#2a6fdb" if pct >= 60 else "#f59e0b" if pct >= 35 else "#ef4444"

            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"### {rank}. {icon} {row['Role']}")
                    st.caption(f"Domain: **{row['Interest Area']}**")
                with c2:
                    st.markdown(
                        f"<div style='text-align:right;font-size:2.2rem;font-weight:700;"
                        f"color:{bar_color};line-height:1'>{pct}%</div>"
                        f"<div style='text-align:right;font-size:0.75rem;color:#888'>"
                        f"sim {row['Cosine Sim']}% · cov {row['Coverage %']}%</div>",
                        unsafe_allow_html=True,
                    )
                    st.progress(int(pct))

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**✅ Skills You Have**")
                    for sk in row["Skills You Have"].split(", "):
                        st.markdown(f"- `{sk}`")
                with col_b:
                    st.markdown("**📚 Skills To Learn Next**")
                    if row["Skills To Learn Next"].startswith("✅"):
                        st.success(row["Skills To Learn Next"])
                    else:
                        for sk in row["Skills To Learn Next"].split(", "):
                            st.markdown(f"- `{sk}`")

        st.divider()
        st.caption("Powered by TF-IDF + Cosine Similarity · IBM SkillsBuild Internship · UN SDG 8")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATASET EXPLORER  (mirrors notebook section 2)
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    jobs, *_ = build_model()

    st.subheader("📋 Full Jobs Dataset")
    st.caption(f"{len(jobs)} roles across {jobs['interest_area'].nunique()} domains")

    # Filter by domain
    domain_filter = st.multiselect(
        "Filter by domain",
        options=sorted(jobs["interest_area"].unique()),
        default=[],
        placeholder="All domains",
    )
    filtered = jobs if not domain_filter else jobs[jobs["interest_area"].isin(domain_filter)]

    display_jobs = filtered.rename(columns={
        "role": "Role", "interest_area": "Domain", "required_skills": "Required Skills"
    })
    st.dataframe(display_jobs, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("📊 Roles per Domain")

    domain_counts = (
        jobs.groupby("interest_area")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    # Colour each bar by domain index
    color_map = {row["interest_area"]: AREA_COLORS[i % len(AREA_COLORS)]
                 for i, row in domain_counts.iterrows()}

    st.bar_chart(
        domain_counts.set_index("interest_area"),
        color="#2a6fdb",
        use_container_width=True,
    )

    st.divider()
    st.subheader("🧩 Skills per Role")
    jobs["skill_count"] = jobs["required_skills"].apply(
        lambda x: len([s for s in x.split(",") if s.strip()])
    )
    skill_count_display = jobs[["role", "interest_area", "skill_count"]].rename(
        columns={"role": "Role", "interest_area": "Domain", "skill_count": "# Skills Required"}
    ).sort_values("# Skills Required", ascending=False)
    st.dataframe(skill_count_display, use_container_width=True, hide_index=True)

    avg_skills = jobs["skill_count"].mean()
    min_role   = jobs.loc[jobs["skill_count"].idxmin(), "role"]
    max_role   = jobs.loc[jobs["skill_count"].idxmax(), "role"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Average skills / role", f"{avg_skills:.1f}")
    c2.metric("Fewest required",  min_role)
    c3.metric("Most required",    max_role)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SKILLS ANALYSIS  (mirrors notebook section 3 — TF-IDF vectors)
# ══════════════════════════════════════════════════════════════════════════════
with tab_skills:
    jobs, vectorizer, role_matrix, known_set, known_skills, skill_tokenizer, tfidf_dense = build_model()

    st.subheader("🔬 Skill Universe")
    st.caption(
        f"The TF-IDF model knows **{len(known_skills)} unique skills** extracted from all role definitions."
    )

    # Skill frequency across roles
    freq = {}
    for _, row in jobs.iterrows():
        for sk in skill_tokenizer(row["required_skills"]):
            freq[sk] = freq.get(sk, 0) + 1

    freq_df = (
        pd.DataFrame(list(freq.items()), columns=["Skill", "Roles Requiring It"])
        .sort_values("Roles Requiring It", ascending=False)
        .reset_index(drop=True)
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("**Top 15 Most Common Skills**")
        st.bar_chart(
            freq_df.head(15).set_index("Skill"),
            color="#2a6fdb",
            use_container_width=True,
        )

    with col_right:
        st.markdown("**All Skills & Frequency**")
        st.dataframe(freq_df, use_container_width=True, hide_index=True, height=340)

    st.divider()
    st.subheader("🔍 Skill Lookup")
    st.caption("Find which roles require a specific skill.")

    search_skill = st.text_input("Search for a skill", placeholder="e.g. python")
    if search_skill.strip():
        query = ALIASES.get(search_skill.strip().lower(), search_skill.strip().lower())
        matching = []
        for _, row in jobs.iterrows():
            if query in skill_tokenizer(row["required_skills"]):
                matching.append({"Role": row["role"], "Domain": row["interest_area"]})
        if matching:
            st.success(f"**{len(matching)} role(s)** require **{query}**")
            st.dataframe(pd.DataFrame(matching), use_container_width=True, hide_index=True)
        else:
            st.warning(f"No roles in the dataset require **{query}**.")

    st.divider()
    st.subheader("🧮 TF-IDF Weight Heatmap (top 20 skills)")
    st.caption("Each cell shows how important a skill is for that role (higher = more distinctive).")

    top20_skills = freq_df.head(20)["Skill"].tolist()
    skill_indices = [list(vectorizer.get_feature_names_out()).index(s) for s in top20_skills
                     if s in vectorizer.get_feature_names_out()]
    heat_matrix   = tfidf_dense[:, skill_indices]
    heat_df = pd.DataFrame(
        heat_matrix,
        index=jobs["role"],
        columns=[list(vectorizer.get_feature_names_out())[i] for i in skill_indices],
    ).round(3)

    st.dataframe(
        heat_df.style.background_gradient(cmap="Blues", axis=None),
        use_container_width=True,
        height=400,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL INSIGHTS  (mirrors notebook section 4 — scoring logic)
# ══════════════════════════════════════════════════════════════════════════════
with tab_model:
    jobs, vectorizer, role_matrix, known_set, known_skills, skill_tokenizer, tfidf_dense = build_model()

    st.subheader("🧠 Model Insights")

    # ── Cosine similarity matrix ──────────────────────────────────────────────
    st.markdown("#### Role-to-Role Cosine Similarity Matrix")
    st.caption(
        "How similar every pair of roles is based on their TF-IDF skill vectors. "
        "High values (→ 1.0) mean the roles share many of the same skills."
    )
    sim_matrix = cosine_similarity(role_matrix)
    sim_df = pd.DataFrame(sim_matrix, index=jobs["role"], columns=jobs["role"]).round(2)
    st.dataframe(
        sim_df.style.background_gradient(cmap="Blues", axis=None),
        use_container_width=True,
        height=420,
    )

    st.divider()

    # ── Sample predictions (mirrors notebook section 5) ───────────────────────
    st.markdown("#### 📌 Sample Predictions — Pre-built Users")
    st.caption("These reproduce the sample runs from the notebook (section 5).")

    SAMPLES = [
        ("Student A — backend skills",  "python, sql, fastapi, git",  None),
        ("Student B — Excel + SQL",      "excel, sql, power bi",        None),
        ("Student C — web basics",       "html, css, javascript",       None),
        ("Student D — AI interest",      "python, ml, opencv",          "Data Science & AI"),
    ]

    for name, skills, interest_area in SAMPLES:
        with st.expander(f"👤 {name}  |  skills: `{skills}`  |  interest: `{interest_area or 'None'}`"):
            sample_result, sample_unknown = recommend(skills, interest=interest_area, top_n=3)
            if sample_unknown:
                st.caption(f"⚠️ Ignored (unknown): {', '.join(sample_unknown)}")
            if sample_result.empty:
                st.info("No matches found for this profile.")
            else:
                display_cols = ["Role", "Interest Area", "Match %", "Skills You Have", "Skills To Learn Next"]
                st.dataframe(sample_result[display_cols], use_container_width=True, hide_index=True)

    st.divider()

    # ── Feature importance (TF-IDF vocabulary) ───────────────────────────────
    st.markdown("#### 📖 Vocabulary & IDF Weights")
    st.caption(
        "IDF (Inverse Document Frequency) weight for each skill. "
        "Rare, specialised skills get higher IDF — they carry more discriminative power."
    )
    idf_vals = vectorizer.idf_
    feat_names = vectorizer.get_feature_names_out()
    idf_df = (
        pd.DataFrame({"Skill": feat_names, "IDF Weight": idf_vals.round(4)})
        .sort_values("IDF Weight", ascending=False)
        .reset_index(drop=True)
    )
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("**Highest IDF (most specialised / rare)**")
        st.dataframe(idf_df.head(15), use_container_width=True, hide_index=True)
    with col_b:
        st.markdown("**Lowest IDF (most common across roles)**")
        st.dataframe(idf_df.tail(15).iloc[::-1].reset_index(drop=True),
                     use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SANITY CHECK  (mirrors notebook section 8)
# ══════════════════════════════════════════════════════════════════════════════
with tab_sanity:
    jobs, *_ = build_model()

    st.subheader("✅ Self-Match Accuracy Check")
    st.markdown(
        "Each role's own skills are fed into the recommender. "
        "The role should rank **#1** in its own results — confirming the model is internally consistent."
    )

    if st.button("▶️ Run Sanity Check", type="primary"):
        rows   = []
        passed = 0
        bar    = st.progress(0, text="Running…")

        for idx, (_, row) in enumerate(jobs.iterrows()):
            top_result, _ = recommend(row["required_skills"], top_n=1)
            predicted = top_result.loc[0, "Role"] if not top_result.empty else "—"
            correct   = predicted == row["role"]
            if correct:
                passed += 1
            rows.append({
                "Role": row["role"],
                "Predicted #1": predicted,
                "Result": "✅ Pass" if correct else "❌ Fail",
            })
            bar.progress((idx + 1) / len(jobs), text=f"Checking {idx + 1}/{len(jobs)}")

        bar.empty()

        accuracy = passed / len(jobs)
        c1, c2, c3 = st.columns(3)
        c1.metric("Roles Tested",  len(jobs))
        c2.metric("Passed",        passed)
        c3.metric("Accuracy",      f"{accuracy:.0%}")

        if accuracy == 1.0:
            st.success("🎉 Perfect self-match accuracy — all 20 roles predict themselves as #1!", icon="✅")
        else:
            st.warning(f"{len(jobs) - passed} role(s) did not self-match. See the table below.", icon="⚠️")

        sanity_df = pd.DataFrame(rows)
        st.dataframe(sanity_df, use_container_width=True, hide_index=True)
    else:
        st.info("Click **Run Sanity Check** to run the accuracy test.", icon="ℹ️")

    st.divider()
    st.subheader("📜 About this Project")
    st.markdown(
        """
        | | |
        |---|---|
        | **Project** | AI Career Path Recommender for Tier-2/3 City Job Seekers |
        | **Programme** | IBM SkillsBuild — AI & Machine Learning Internship |
        | **SDG alignment** | UN SDG 8 · Decent Work and Economic Growth |
        | **Algorithm** | TF-IDF vectorisation + Cosine Similarity |
        | **Dataset** | 20 hand-curated entry-level tech roles |
        | **Score formula** | 60 % cosine similarity + 40 % skill coverage + 5 % domain bonus |

        **Limitations**
        - Small, hand-written dataset (20 roles) — prototype quality.
        - Skills must be typed; spelling variants outside the alias list are ignored.
        - Does not consider location, salary, or experience level.

        **Future scope**
        - Expand with real job-posting data.
        - Add regional-language support.
        - Integrate IBM watsonx to generate personalised career summaries.
        - Link missing skills to free IBM SkillsBuild courses.
        """
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — AI CAREER COACH  (Ollama / phi3 chatbot)
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:

    OLLAMA_MODEL = "phi3:latest"

    if not OLLAMA_AVAILABLE:
        st.warning(
            "**Ollama Python package not found.**\n\n"
            "The AI Career Coach runs locally and requires the `ollama` package and Ollama server.\n\n"
            "Install it with:\n```\npip install ollama\n```\n"
            "Then start Ollama and pull the model:\n```\nollama serve\nollama pull phi3\n```",
            icon="⚠️",
        )
        st.stop()

    ollama = _ollama

    # ── System prompt — injects dataset knowledge + role as career coach ──────
    SYSTEM_PROMPT = """You are an expert AI Career Coach specialising in tech jobs for job seekers
in Tier-2 and Tier-3 cities in India. You help people understand which career paths suit their
skills, what skills they need to learn next, and how to get started.

You have deep knowledge of the following 20 entry-level tech roles and their required skills:

Role | Domain | Required Skills
Data Analyst | Data Analytics | excel, sql, python, data visualization, statistics, power bi
Data Scientist | Data Science & AI | python, machine learning, statistics, pandas, sql, data visualization
Machine Learning Engineer | Data Science & AI | python, machine learning, deep learning, tensorflow, git, docker
Computer Vision Engineer | Data Science & AI | python, opencv, deep learning, machine learning, numpy
Python Backend Developer | Software Development | python, fastapi, sql, rest api, git, docker
Java Developer | Software Development | java, sql, spring boot, rest api, git, oop
Mobile App Developer | Software Development | kotlin, flutter, rest api, git, ui design
Frontend Developer | Web Development | html, css, javascript, react, git, responsive design
Full Stack Developer | Web Development | html, css, javascript, react, node.js, sql, git
Software Tester (QA) | Quality Assurance | manual testing, selenium, sql, test cases, java
DevOps Engineer | Cloud & DevOps | linux, docker, git, ci/cd, cloud computing, shell scripting
Cloud Engineer | Cloud & DevOps | cloud computing, linux, networking, docker, python
Cybersecurity Analyst | Security | networking, linux, cybersecurity, python, risk assessment
Database Administrator | Data & Databases | sql, database design, backup and recovery, linux, performance tuning
Data Engineer | Data & Databases | python, sql, etl, apache spark, cloud computing, data warehousing
Business Analyst | Business | excel, sql, communication, documentation, power bi, problem solving
UI/UX Designer | Design | figma, ui design, user research, wireframing, prototyping
Digital Marketing Executive | Marketing | seo, social media, content writing, google analytics, communication
Content Writer | Marketing | content writing, seo, communication, research, editing
Technical Support Engineer | IT Support | troubleshooting, networking, communication, windows, linux

Scoring formula used by the recommender tool:
  Match % = 60% cosine similarity (TF-IDF) + 40% skill coverage + optional 5% domain bonus

When a user shares their skills, you should:
1. Identify the best-fit roles from the list above.
2. Give a brief personalised summary (2-3 sentences) of why they are a good fit.
3. Point out the most important 2-3 skills they should learn next for their top role.
4. Be encouraging, practical, and concise.
5. If asked about learning resources, suggest free platforms: IBM SkillsBuild, Coursera, YouTube, freeCodeCamp.
6. If asked about salary/job market, give honest general guidance relevant to India.

Always respond in clear, simple English. Avoid jargon unless explaining it."""

    # ── Helper: build context string from recommender results (if available) ──
    def build_career_context():
        if not skills_input.strip():
            return ""
        interest = None if interest_choice == "No preference" else interest_choice
        res, _ = recommend(skills_input, interest=interest, top_n=3)
        if res.empty:
            return ""
        lines = [f"\n[Current recommender results for '{skills_input.strip()}']"]
        for _, r in res.iterrows():
            lines.append(
                f"- {r['Role']} ({r['Interest Area']}): {r['Match %']}% match | "
                f"have: {r['Skills You Have']} | learn: {r['Skills To Learn Next']}"
            )
        return "\n".join(lines)

    # ── Session state for chat history ────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []   # list of {"role": ..., "content": ...}

    # ── Page header ───────────────────────────────────────────────────────────
    st.subheader("🤖 AI Career Coach")
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.caption(
            f"Powered by **Ollama · {OLLAMA_MODEL}** — running locally on your device. "
            "No data leaves your machine."
        )
    with col_h2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # ── Quick-action buttons ──────────────────────────────────────────────────
    st.markdown("**Quick prompts:**")
    qa_cols = st.columns(4)
    quick_prompts = [
        ("🎯 Analyse my skills",   "I have these skills: {skills}. Which roles suit me best and why?"),
        ("📚 What to learn next",  "Based on my skills ({skills}), what are the top 3 skills I should learn next for the best career outcome?"),
        ("📝 Career summary",      "Give me a brief career summary and roadmap for someone with these skills: {skills}."),
        ("💼 Job market tips",     "What is the job market like in India for someone with skills: {skills}? Any tips for Tier-2/3 city job seekers?"),
    ]

    for col, (label, template) in zip(qa_cols, quick_prompts):
        if col.button(label, use_container_width=True):
            filled = template.replace(
                "{skills}", skills_input.strip() if skills_input.strip() else "python, sql, git"
            )
            st.session_state.chat_history.append({"role": "user", "content": filled})
            st.rerun()

    st.divider()

    # ── Render existing chat history ──────────────────────────────────────────
    chat_container = st.container(height=420)
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown(
                "<div style='text-align:center;padding:60px 0;color:#888'>"
                "👋 Ask me anything about your career path, skills to learn, or job market in India."
                "</div>",
                unsafe_allow_html=True,
            )
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # ── Chat input ────────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask your career question…")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.rerun()

    # ── Generate assistant response if last message is from user ─────────────
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":

        career_ctx = build_career_context()
        system_with_ctx = SYSTEM_PROMPT + career_ctx

        # Build messages for Ollama — prepend system, then full history
        messages_for_llm = [{"role": "system", "content": system_with_ctx}]
        messages_for_llm += st.session_state.chat_history

        with chat_container:
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                try:
                    stream = ollama.chat(
                        model=OLLAMA_MODEL,
                        messages=messages_for_llm,
                        stream=True,
                    )
                    for chunk in stream:
                        delta = chunk["message"]["content"]
                        full_response += delta
                        response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)
                except Exception as e:
                    full_response = (
                        f"⚠️ Could not reach Ollama. Make sure it is running "
                        f"(`ollama serve`) and **{OLLAMA_MODEL}** is pulled.\n\n"
                        f"Error: `{e}`"
                    )
                    response_placeholder.warning(full_response)

        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
        st.rerun()
