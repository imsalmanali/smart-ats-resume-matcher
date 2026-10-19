import io
from pathlib import Path
import pandas as pd
import streamlit as st

from src.parser import extract_text
from src.skills_extractor import SkillsExtractor
from src.matcher import compute_ats_score
from src.report import generate_recommendations, generate_summary_text

# Page Config
st.set_page_config(
    page_title="Smart ATS Resume Matcher AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Data Paths
DATA_DIR = Path(__file__).resolve().parent / "data"
SAMPLE_RESUMES_DIR = DATA_DIR / "sample_resumes"
SAMPLE_JOBS_DIR = DATA_DIR / "sample_jobs"

@st.cache_resource
def get_extractor():
    return SkillsExtractor()

extractor = get_extractor()

# --- Sidebar Configuration ---
st.sidebar.title("🎯 ATS Matcher Settings")
st.sidebar.markdown(
    "AI-powered Applicant Tracking System leveraging **NLP, TF-IDF Vector Spaces, and Taxonomy Extraction**."
)

st.sidebar.subheader("⚖️ Scoring Weights")
weight_skill = st.sidebar.slider("Skill Match Weight", min_value=0.1, max_value=0.8, value=0.55, step=0.05)
weight_semantic = st.sidebar.slider("Semantic TF-IDF Weight", min_value=0.1, max_value=0.8, value=0.35, step=0.05)
weight_structure = round(1.0 - (weight_skill + weight_semantic), 2)
st.sidebar.caption(f"Structure / Length Weight: **{weight_structure}** (auto-balanced)")

custom_weights = {
    "skills": weight_skill,
    "semantic": weight_semantic,
    "structure": max(weight_structure, 0.0)
}

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Test pre-loaded resumes from the Demo Preset dropdown inside Tab 1.")

# --- Main Content ---
st.title("🎯 Smart ATS Resume Matcher & Candidate Screening AI")
st.markdown("Evaluate resumes against job descriptions with deep NLP keyword extraction, semantic similarity, and skill gap analysis.")

tab1, tab2, tab3 = st.tabs([
    "📄 Single Match & Evaluation",
    "👥 Batch Candidate Ranking",
    "🔍 Skill Taxonomy Explorer"
])


# ==========================================
# TAB 1: Single Match & Evaluation
# ==========================================
with tab1:
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("1. Resume Input")
        preset_choice = st.selectbox(
            "Load Demo Resume Preset (Optional):",
            ["None (Upload or paste manually)", "Senior ML Engineer", "Frontend Developer", "Data Analyst"]
        )

        loaded_resume_text = ""
        if preset_choice == "Senior ML Engineer":
            path = SAMPLE_RESUMES_DIR / "senior_ml_engineer.txt"
            if path.exists():
                loaded_resume_text = path.read_text(encoding="utf-8")
        elif preset_choice == "Frontend Developer":
            path = SAMPLE_RESUMES_DIR / "frontend_developer.txt"
            if path.exists():
                loaded_resume_text = path.read_text(encoding="utf-8")
        elif preset_choice == "Data Analyst":
            path = SAMPLE_RESUMES_DIR / "data_analyst.txt"
            if path.exists():
                loaded_resume_text = path.read_text(encoding="utf-8")

        uploaded_file = st.file_uploader("Upload Resume (PDF, DOCX, TXT):", type=["pdf", "docx", "txt"])
        if uploaded_file:
            bytes_data = uploaded_file.read()
            loaded_resume_text = extract_text(bytes_data, filename=uploaded_file.name)

        resume_text = st.text_area(
            "Resume Text:",
            value=loaded_resume_text,
            height=280,
            placeholder="Paste candidate resume content here..."
        )

    with col_right:
        st.subheader("2. Target Job Description")
        job_preset_choice = st.selectbox(
            "Load Demo Job Description (Optional):",
            ["None (Paste manually)", "Senior Machine Learning Engineer", "Fullstack Web Developer"]
        )

        loaded_job_text = ""
        if job_preset_choice == "Senior Machine Learning Engineer":
            path = SAMPLE_JOBS_DIR / "machine_learning_engineer_job.txt"
            if path.exists():
                loaded_job_text = path.read_text(encoding="utf-8")
        elif job_preset_choice == "Fullstack Web Developer":
            path = SAMPLE_JOBS_DIR / "fullstack_developer_job.txt"
            if path.exists():
                loaded_job_text = path.read_text(encoding="utf-8")

        job_text = st.text_area(
            "Job Description Text:",
            value=loaded_job_text,
            height=345,
            placeholder="Paste target job description and requirements here..."
        )

    analyze_btn = st.button("🚀 Analyze & Score Match", type="primary", use_container_width=True)

    if analyze_btn:
        if not resume_text.strip() or not job_text.strip():
            st.error("Please provide both a Resume and a Job Description to evaluate.")
        else:
            with st.spinner("Processing NLP embeddings and extracting skills..."):
                r_skills = extractor.extract_flat_skills(resume_text)
                j_skills = extractor.extract_flat_skills(job_text)

                result = compute_ats_score(
                    resume_text=resume_text,
                    job_text=job_text,
                    resume_skills=r_skills,
                    job_skills=j_skills,
                    weights=custom_weights
                )

                score = result["overall_score"]
                tier = result["match_tier"]
                status_color = result["status_color"]
                breakdown = result["breakdown"]
                analysis = result["skill_analysis"]

                st.markdown("---")
                st.subheader("📊 Match Evaluation Results")

                # Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Overall ATS Score", f"{score}%", tier)
                m2.metric("Skill Coverage", f"{breakdown['skill_coverage_score']}%")
                m3.metric("Semantic Similarity", f"{breakdown['semantic_similarity_score']}%")
                m4.metric("Skills Overlap", f"{analysis['matched_count']}/{analysis['total_required']}")

                # Progress Bar
                progress_val = min(max(score / 100.0, 0.0), 1.0)
                st.progress(progress_val)

                # Skills Comparison Columns
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    st.markdown("#### ✅ Matched Skills")
                    if analysis["matched_skills"]:
                        st.write(", ".join([f"`{s}`" for s in analysis["matched_skills"]]))
                    else:
                        st.write("_None detected_")

                with sc2:
                    st.markdown("#### ⚠️ Missing Required Skills")
                    if analysis["missing_skills"]:
                        st.write(", ".join([f"`{s}`" for s in analysis["missing_skills"]]))
                    else:
                        st.write("✨ _No missing skills found!_")

                with sc3:
                    st.markdown("#### ➕ Additional Candidate Skills")
                    if analysis["extra_skills"]:
                        st.write(", ".join([f"`{s}`" for s in analysis["extra_skills"][:10]]))
                    else:
                        st.write("_None detected_")

                # Recommendations
                st.markdown("#### 💡 Actionable Improvement Recommendations")
                recommendations = generate_recommendations(result)
                for rec in recommendations:
                    st.info(f"👉 {rec}")


# ==========================================
# TAB 2: Batch Candidate Ranking
# ==========================================
with tab2:
    st.subheader("👥 Batch Candidate Resume Screening")
    st.markdown("Upload multiple candidate resumes simultaneously to automatically rank and benchmark them against a single job opening.")

    batch_job_text = st.text_area(
        "Target Job Description for Batch Ranking:",
        height=150,
        placeholder="Paste target job description here..."
    )

    batch_files = st.file_uploader(
        "Upload Multiple Resumes (PDF, DOCX, TXT):",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if st.button("🏆 Rank Candidates", type="primary"):
        if not batch_job_text.strip():
            st.error("Please enter a Target Job Description.")
        elif not batch_files:
            st.warning("Please upload at least one resume file.")
        else:
            with st.spinner(f"Evaluating {len(batch_files)} resumes..."):
                j_skills = extractor.extract_flat_skills(batch_job_text)
                rows = []

                for f in batch_files:
                    f_bytes = f.read()
                    f_text = extract_text(f_bytes, filename=f.name)
                    cand_skills = extractor.extract_flat_skills(f_text)
                    ats_res = compute_ats_score(
                        resume_text=f_text,
                        job_text=batch_job_text,
                        resume_skills=cand_skills,
                        job_skills=j_skills,
                        weights=custom_weights
                    )
                    rows.append({
                        "Candidate File": f.name,
                        "ATS Score (%)": ats_res["overall_score"],
                        "Match Tier": ats_res["match_tier"],
                        "Skill Coverage (%)": ats_res["breakdown"]["skill_coverage_score"],
                        "Semantic Score (%)": ats_res["breakdown"]["semantic_similarity_score"],
                        "Matched Skills": ", ".join(ats_res["skill_analysis"]["matched_skills"]) or "None",
                        "Missing Skills": ", ".join(ats_res["skill_analysis"]["missing_skills"]) or "None"
                    })

                df = pd.DataFrame(rows).sort_values(by="ATS Score (%)", ascending=False).reset_index(drop=True)
                df.index = df.index + 1
                df.index.name = "Rank"

                st.dataframe(df, use_container_width=True)

                csv = df.to_csv().encode("utf-8")
                st.download_button(
                    label="📥 Download Ranking Leaderboard as CSV",
                    data=csv,
                    file_name="candidate_ats_ranking.csv",
                    mime="text/csv"
                )


# ==========================================
# TAB 3: Skill Taxonomy Explorer
# ==========================================
with tab3:
    st.subheader("🔍 Skill Taxonomy & NLP Extraction Engine")
    st.markdown("Explore the pre-configured domain taxonomies or test quick text keyword extraction.")

    query = st.text_input("Test Extractor on Text:", placeholder="e.g. Expert in Python, PyTorch, Docker, and AWS.")
    if query:
        extracted = extractor.extract_skills(query)
        st.write("Extracted Entities by Category:")
        st.json(extracted)

    st.markdown("---")
    st.subheader("📚 Loaded Domain Taxonomies")
    for category, skills in extractor.taxonomy.items():
        with st.expander(f"{category} ({len(skills)} skills)"):
            st.write(", ".join([f"`{s}`" for s in skills]))
